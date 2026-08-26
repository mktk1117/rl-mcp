"""RunnerAdapter for rsl_rl's ``OnPolicyRunner`` (and mjlab's subclasses).

Note on the learning rate: PPO's ``adaptive`` schedule rewrites
``alg.learning_rate`` every update from the measured KL divergence, so a manual
edit is overwritten within one iteration. Setting ``rl.learning_rate`` therefore
also switches the schedule to ``fixed`` and says so in the result, rather than
silently doing nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from rlmcp.adapters.base import NotSupported, RunnerAdapter
from rlmcp.core.parameters.spec import ParameterCategory, ParameterSpec

_ALG_PARAMS: Dict[str, Dict[str, Any]] = {
    "learning_rate": {
        "min": 1e-6,
        "max": 1e-2,
        "desc": "PPO optimiser learning rate (switches schedule to 'fixed' when set)",
    },
    "entropy_coef": {
        "min": 0.0,
        "max": 0.2,
        "desc": "Entropy bonus weight; raise it when the policy collapses too early",
    },
    "clip_param": {"min": 0.01, "max": 0.5, "desc": "PPO surrogate clipping epsilon"},
    "desired_kl": {
        "min": 1e-4,
        "max": 0.1,
        "desc": "Target KL for the adaptive learning-rate schedule",
    },
    "gamma": {"min": 0.9, "max": 0.9999, "desc": "Discount factor"},
    "lam": {"min": 0.8, "max": 1.0, "desc": "GAE lambda"},
    "max_grad_norm": {"min": 0.1, "max": 10.0, "desc": "Gradient-norm clip"},
    "value_loss_coef": {"min": 0.0, "max": 10.0, "desc": "Value loss weight"},
}


class MjlabRunnerAdapter(RunnerAdapter):
  """Live control over an rsl_rl runner's algorithm and checkpoints."""

  def __init__(self, runner: Any):
    self.runner = runner

  @property
  def alg(self) -> Any:
    alg = getattr(self.runner, "alg", None)
    if alg is None:
      raise NotSupported("Runner has no 'alg'; cannot touch hyperparameters.")
    return alg

  def discover_hyperparameters(self) -> List[ParameterSpec]:
    alg = getattr(self.runner, "alg", None)
    if alg is None:
      return []
    specs: List[ParameterSpec] = []
    for name, meta in _ALG_PARAMS.items():
      if not hasattr(alg, name):
        continue
      value = getattr(alg, name)
      if value is None:
        continue
      specs.append(
          ParameterSpec(
              key=f"rl.{name}",
              data_type="float",
              current_value=float(value),
              default_value=float(value),
              min_value=meta["min"],
              max_value=meta["max"],
              description=meta["desc"],
              category=ParameterCategory.RL_HYPERPARAMETER,
          )
      )
    return specs

  def get_hyperparameter(self, key: str) -> Any:
    name = key.split(".", 1)[-1]
    if not hasattr(self.alg, name):
      raise KeyError(f"Runner has no hyperparameter '{name}'.")
    return float(getattr(self.alg, name))

  def set_hyperparameter(self, key: str, value: Any) -> bool:
    name = key.split(".", 1)[-1]
    alg = self.alg
    if name not in _ALG_PARAMS or not hasattr(alg, name):
      # Raise rather than return False: an unknown key is a failure with an
      # explanation, and a falsy return would read as "not applied" with none.
      known = sorted(k for k in _ALG_PARAMS if hasattr(alg, k))
      raise KeyError(
          f"No tunable hyperparameter '{name}' on this runner. "
          f"Available: {known}"
      )

    if name == "learning_rate":
      lr = float(value)
      optimizer = getattr(alg, "optimizer", None)
      if optimizer is not None:
        for group in optimizer.param_groups:
          group["lr"] = lr
      alg.learning_rate = lr
      if getattr(alg, "schedule", None) == "adaptive":
        alg.schedule = "fixed"
      return True

    setattr(alg, name, float(value))
    return True

  def current_iteration(self) -> int:
    return int(getattr(self.runner, "current_learning_iteration", 0))

  def runner_metrics(self) -> Dict[str, float]:
    """Scalars the runner already tracks: reward, episode length, LR, std."""
    out: Dict[str, float] = {}
    alg = getattr(self.runner, "alg", None)
    if alg is not None:
      if getattr(alg, "learning_rate", None) is not None:
        out["Loss/learning_rate"] = float(alg.learning_rate)
      try:
        out["Policy/mean_std"] = float(alg.get_policy().output_std.mean().item())
      except Exception:
        pass
    logger = getattr(self.runner, "logger", None)
    if logger is not None:
      rewards = list(getattr(logger, "rewbuffer", []) or [])
      lengths = list(getattr(logger, "lenbuffer", []) or [])
      if rewards:
        out["Train/mean_reward"] = float(sum(rewards) / len(rewards))
      if lengths:
        out["Train/mean_episode_length"] = float(sum(lengths) / len(lengths))
    return out

  def save_checkpoint(
      self, path: str, infos: Optional[Dict[str, Any]] = None
  ) -> Optional[str]:
    save = getattr(self.runner, "save", None)
    if not callable(save):
      raise NotSupported("Runner does not implement save().")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save(str(path), infos)
    return str(path)

  def load_checkpoint(self, path: str) -> Dict[str, Any]:
    load = getattr(self.runner, "load", None)
    if not callable(load):
      raise NotSupported("Runner does not implement load().")
    if not Path(path).exists():
      raise FileNotFoundError(f"No checkpoint at '{path}'.")
    infos = load(str(path))
    return infos if isinstance(infos, dict) else {}

  def log_dir(self) -> Optional[str]:
    logger = getattr(self.runner, "logger", None)
    return getattr(logger, "log_dir", None) if logger is not None else None

  def request_stop(self) -> bool:
    """No-op: rsl_rl has no stop hook this call could arm.

    Stopping actually happens through the servicing contract -- the wrapper
    polls ``RlMcp.should_stop()`` at each iteration boundary and raises
    ``TrainingStopped`` out of ``learn()``. Returning False says truthfully
    that nothing was armed here, rather than raising into the controller or
    setting a flag nothing reads.
    """
    return False
