"""PPO, in one file, for a single-file task to copy and edit.

A port of rsl_rl's PPO (ETH Zurich / NVIDIA, BSD-3-Clause) with the same
behaviour and no dependency on it: clipped surrogate, clipped value loss,
GAE with timeout bootstrapping, the adaptive learning rate driven by measured
KL, separate actor and critic MLPs with a state-independent Gaussian. It is an
asset, not a library: a task that needs a different algorithm edits its copy.

What rlmcp touches is small and by duck typing. Hyperparameters are the
attributes on :class:`PPO` (``learning_rate``, ``entropy_coef``, ...), and
checkpoints go through :meth:`PPO.save` and :meth:`PPO.load`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain

import torch
from torch import Tensor, nn
from torch.distributions import Normal

# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
  """mjlab's Go1 runner: three hidden layers, ELU, unit initial std, and
  running observation normalisation in front of both networks."""

  hidden_dims: tuple[int, ...] = (512, 256, 128)
  activation: str = "elu"
  init_noise_std: float = 1.0
  obs_normalization: bool = True


@dataclass
class PPOConfig:
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  learning_rate: float = 1e-3
  max_grad_norm: float = 1.0
  clip_param: float = 0.2
  value_loss_coef: float = 1.0
  entropy_coef: float = 0.01
  use_clipped_value_loss: bool = True
  gamma: float = 0.99
  lam: float = 0.95
  schedule: str = "adaptive"
  """``adaptive`` moves the learning rate to hold ``desired_kl``; ``fixed`` does not."""
  desired_kl: float = 0.01


# ---------------------------------------------------------------------------
# Networks.
# ---------------------------------------------------------------------------

_ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU}


def mlp(input_dim: int, output_dim: int, hidden: tuple[int, ...], activation: str) -> nn.Sequential:
  act = _ACTIVATIONS[activation]
  layers: list[nn.Module] = []
  last = input_dim
  for width in hidden:
    layers += [nn.Linear(last, width), act()]
    last = width
  layers.append(nn.Linear(last, output_dim))
  return nn.Sequential(*layers)


class EmpiricalNormalization(nn.Module):
  """Running mean and std of the observations, rsl_rl's: learned while the
  policy trains, frozen in eval, and part of the checkpoint."""

  def __init__(self, shape: int, eps: float = 1e-2):
    super().__init__()
    self.eps = eps
    self.register_buffer("_mean", torch.zeros(1, shape))
    self.register_buffer("_var", torch.ones(1, shape))
    self.register_buffer("_std", torch.ones(1, shape))
    self.register_buffer("count", torch.tensor(0, dtype=torch.long))

  def forward(self, x: Tensor) -> Tensor:
    return (x - self._mean) / (self._std + self.eps)

  @torch.no_grad()
  def update(self, x: Tensor) -> None:
    if not self.training:
      return
    count_x = x.shape[0]
    self.count += count_x
    rate = count_x / self.count
    var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
    mean_x = torch.mean(x, dim=0, keepdim=True)
    delta = mean_x - self._mean
    self._mean += rate * delta
    self._var += rate * (var_x - self._var + delta * (mean_x - self._mean))
    self._std = torch.sqrt(self._var)


class Actor(nn.Module):
  """Gaussian policy: an MLP mean and a learned, state-independent std."""

  def __init__(self, obs_dim: int, action_dim: int, cfg: ModelConfig | None = None):
    super().__init__()
    cfg = cfg or ModelConfig()
    self.normalizer = EmpiricalNormalization(obs_dim) if cfg.obs_normalization else nn.Identity()
    self.net = mlp(obs_dim, action_dim, cfg.hidden_dims, cfg.activation)
    self.std = nn.Parameter(cfg.init_noise_std * torch.ones(action_dim))
    self.distribution: Normal | None = None
    Normal.set_default_validate_args(False)

  def update_normalization(self, obs: Tensor) -> None:
    if isinstance(self.normalizer, EmpiricalNormalization):
      self.normalizer.update(obs)

  def forward(self, obs: Tensor, stochastic: bool = False) -> Tensor:
    mean = self.net(self.normalizer(obs))
    self.distribution = Normal(mean, self.std.expand_as(mean))
    return self.distribution.sample() if stochastic else mean

  def log_prob(self, actions: Tensor) -> Tensor:
    assert self.distribution is not None
    return self.distribution.log_prob(actions).sum(dim=-1)

  @property
  def output_mean(self) -> Tensor:
    assert self.distribution is not None
    return self.distribution.mean

  @property
  def output_std(self) -> Tensor:
    assert self.distribution is not None
    return self.distribution.stddev

  @property
  def entropy(self) -> Tensor:
    assert self.distribution is not None
    return self.distribution.entropy().sum(dim=-1)


class Critic(nn.Module):
  """Value function over the critic's own observations, which may be a
  privileged superset of the actor's."""

  def __init__(self, obs_dim: int, cfg: ModelConfig | None = None):
    super().__init__()
    cfg = cfg or ModelConfig()
    self.normalizer = EmpiricalNormalization(obs_dim) if cfg.obs_normalization else nn.Identity()
    self.net = mlp(obs_dim, 1, cfg.hidden_dims, cfg.activation)

  def update_normalization(self, obs: Tensor) -> None:
    if isinstance(self.normalizer, EmpiricalNormalization):
      self.normalizer.update(obs)

  def forward(self, obs: Tensor) -> Tensor:
    return self.net(self.normalizer(obs))


# ---------------------------------------------------------------------------
# Rollout storage.
# ---------------------------------------------------------------------------


@dataclass
class Batch:
  observations: Tensor
  critic_observations: Tensor
  actions: Tensor
  values: Tensor
  advantages: Tensor
  returns: Tensor
  old_log_prob: Tensor
  old_mu: Tensor
  old_sigma: Tensor


class RolloutStorage:
  """``[T, N, ...]`` buffers for one rollout, and the mini-batch generator."""

  def __init__(self, num_envs: int, num_steps: int, obs_dim: int, action_dim: int,
               device: str | torch.device = "cpu", critic_obs_dim: int | None = None):
    self.device = torch.device(device)
    self.num_envs, self.num_steps = num_envs, num_steps
    t, n = num_steps, num_envs
    z = lambda *shape, **kw: torch.zeros(*shape, device=self.device, **kw)  # noqa: E731
    self.observations = z(t, n, obs_dim)
    self.critic_observations = z(t, n, critic_obs_dim or obs_dim)
    self.actions = z(t, n, action_dim)
    self.rewards = z(t, n, 1)
    self.dones = z(t, n, 1, dtype=torch.uint8)
    self.values = z(t, n, 1)
    self.log_prob = z(t, n, 1)
    self.mu = z(t, n, action_dim)
    self.sigma = z(t, n, action_dim)
    self.returns = z(t, n, 1)
    self.advantages = z(t, n, 1)
    self.step = 0

  def add(self, obs: Tensor, critic_obs: Tensor, actions: Tensor, rewards: Tensor,
          dones: Tensor, values: Tensor, log_prob: Tensor, mu: Tensor, sigma: Tensor) -> None:
    if self.step >= self.num_steps:
      raise OverflowError("Rollout buffer is full; call clear() first.")
    i = self.step
    self.observations[i].copy_(obs)
    self.critic_observations[i].copy_(critic_obs)
    self.actions[i].copy_(actions)
    self.rewards[i].copy_(rewards.view(-1, 1))
    self.dones[i].copy_(dones.view(-1, 1))
    self.values[i].copy_(values)
    self.log_prob[i].copy_(log_prob.view(-1, 1))
    self.mu[i].copy_(mu)
    self.sigma[i].copy_(sigma)
    self.step += 1

  def clear(self) -> None:
    self.step = 0

  def mini_batches(self, num_mini_batches: int, num_epochs: int) -> Iterator[Batch]:
    batch_size = self.num_envs * self.num_steps
    mini = batch_size // num_mini_batches
    flat = {
        name: getattr(self, name).flatten(0, 1)
        for name in ("observations", "critic_observations", "actions", "values", "returns",
                     "log_prob", "advantages", "mu", "sigma")
    }
    for _ in range(num_epochs):
      order = torch.randperm(num_mini_batches * mini, device=self.device)
      for k in range(num_mini_batches):
        idx = order[k * mini:(k + 1) * mini]
        yield Batch(
            observations=flat["observations"][idx],
            critic_observations=flat["critic_observations"][idx],
            actions=flat["actions"][idx],
            values=flat["values"][idx],
            advantages=flat["advantages"][idx],
            returns=flat["returns"][idx],
            old_log_prob=flat["log_prob"][idx],
            old_mu=flat["mu"][idx],
            old_sigma=flat["sigma"][idx],
        )


# ---------------------------------------------------------------------------
# The algorithm.
# ---------------------------------------------------------------------------


class PPO:
  """rsl_rl's PPO update, over an :class:`Actor`, a :class:`Critic` and storage."""

  def __init__(self, actor: Actor, critic: Critic, storage: RolloutStorage,
               cfg: PPOConfig | None = None):
    cfg = cfg or PPOConfig()
    self.actor, self.critic, self.storage = actor, critic, storage
    self.clip_param = cfg.clip_param
    self.num_learning_epochs = cfg.num_learning_epochs
    self.num_mini_batches = cfg.num_mini_batches
    self.value_loss_coef = cfg.value_loss_coef
    self.entropy_coef = cfg.entropy_coef
    self.gamma = cfg.gamma
    self.lam = cfg.lam
    self.max_grad_norm = cfg.max_grad_norm
    self.use_clipped_value_loss = cfg.use_clipped_value_loss
    self.desired_kl = cfg.desired_kl
    self.schedule = cfg.schedule
    self.learning_rate = cfg.learning_rate
    self.optimizer = torch.optim.Adam(
        chain(actor.parameters(), critic.parameters()), lr=cfg.learning_rate)
    self._pending: dict[str, Tensor] = {}

  # Rollout.

  def act(self, obs: Tensor, critic_obs: Tensor | None = None) -> Tensor:
    """Sample actions for ``obs`` and remember what the update needs.

    The normalisers learn here, from the rollout, the way rsl_rl's do.
    """
    critic_obs = obs if critic_obs is None else critic_obs
    self.actor.update_normalization(obs)
    self.critic.update_normalization(critic_obs)
    actions = self.actor(obs, stochastic=True).detach()
    self._pending = {
        "obs": obs,
        "critic_obs": critic_obs,
        "actions": actions,
        "values": self.critic(critic_obs).detach(),
        "log_prob": self.actor.log_prob(actions).detach(),
        "mu": self.actor.output_mean.detach(),
        "sigma": self.actor.output_std.detach(),
    }
    return actions

  def process_env_step(self, rewards: Tensor, dones: Tensor,
                       time_outs: Tensor | None = None) -> None:
    """Record the step. A timeout bootstraps the value, a failure does not."""
    p = self._pending
    rewards = rewards.clone()
    if time_outs is not None:
      rewards += self.gamma * torch.squeeze(p["values"] * time_outs.unsqueeze(1), 1)
    self.storage.add(p["obs"], p["critic_obs"], p["actions"], rewards, dones, p["values"],
                     p["log_prob"], p["mu"], p["sigma"])
    self._pending = {}

  def compute_returns(self, last_critic_obs: Tensor) -> None:
    st = self.storage
    last_values = self.critic(last_critic_obs).detach()
    advantage = torch.zeros_like(last_values)
    for step in reversed(range(st.num_steps)):
      next_values = last_values if step == st.num_steps - 1 else st.values[step + 1]
      not_terminal = 1.0 - st.dones[step].float()
      delta = st.rewards[step] + not_terminal * self.gamma * next_values - st.values[step]
      advantage = delta + not_terminal * self.gamma * self.lam * advantage
      st.returns[step] = advantage + st.values[step]
    st.advantages = st.returns - st.values
    st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

  # Update.

  def update(self) -> dict[str, float]:
    """One PPO update over the stored rollout. Returns mean losses."""
    sums = {"value_loss": 0.0, "surrogate_loss": 0.0, "entropy": 0.0}
    for batch in self.storage.mini_batches(self.num_mini_batches, self.num_learning_epochs):
      self.actor(batch.observations, stochastic=True)
      log_prob = self.actor.log_prob(batch.actions)
      values = self.critic(batch.critic_observations)
      mu, sigma, entropy = self.actor.output_mean, self.actor.output_std, self.actor.entropy

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = torch.sum(
              torch.log(sigma / batch.old_sigma + 1e-5)
              + (torch.square(batch.old_sigma) + torch.square(batch.old_mu - mu))
              / (2.0 * torch.square(sigma)) - 0.5, dim=-1)
          kl_mean = torch.mean(kl)
          if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
          elif 0.0 < kl_mean < self.desired_kl / 2.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

      ratio = torch.exp(log_prob - torch.squeeze(batch.old_log_prob))
      advantages = torch.squeeze(batch.advantages)
      surrogate = -advantages * ratio
      surrogate_clipped = -advantages * torch.clamp(
          ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
            -self.clip_param, self.clip_param)
        value_loss = torch.max(
            (values - batch.returns).pow(2), (value_clipped - batch.returns).pow(2)).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = (surrogate_loss + self.value_loss_coef * value_loss
              - self.entropy_coef * entropy.mean())
      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      sums["value_loss"] += value_loss.item()
      sums["surrogate_loss"] += surrogate_loss.item()
      sums["entropy"] += entropy.mean().item()

    updates = self.num_learning_epochs * self.num_mini_batches
    self.storage.clear()
    out = {name: total / updates for name, total in sums.items()}
    out["learning_rate"] = self.learning_rate
    return out

  # Modes and checkpoints.

  def train_mode(self) -> None:
    self.actor.train()
    self.critic.train()

  def eval_mode(self) -> None:
    self.actor.eval()
    self.critic.eval()

  def save(self) -> dict:
    return {
        "actor_state_dict": self.actor.state_dict(),
        "critic_state_dict": self.critic.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "learning_rate": self.learning_rate,
    }

  def load(self, state: dict, strict: bool = True) -> None:
    self.actor.load_state_dict(state["actor_state_dict"], strict=strict)
    self.critic.load_state_dict(state["critic_state_dict"], strict=strict)
    if "optimizer_state_dict" in state:
      self.optimizer.load_state_dict(state["optimizer_state_dict"])
    if "learning_rate" in state:
      self.learning_rate = float(state["learning_rate"])
