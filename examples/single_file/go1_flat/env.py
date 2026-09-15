"""Go1 flat-ground velocity tracking, in one file.

This file *is* the task. Every number an agent might tune is in the config
dataclasses at the top; the physics loop, observations, rewards and
terminations are inline below them, in the order they happen. Nothing is
inherited and nothing is registered: read it top to bottom and you know the
whole environment.

Two things make it steerable from another shell with ``rlmcp``:

* the config is declared with :mod:`rlmcp.declare` -- ``Static[...]`` marks a
  value read once at construction, ``term(...)`` declares a reward term -- so
  rlmcp knows what it may change live and what would need a restart;
* the simulator is behind :mod:`rlmcp.backends`, so ``cfg.backend`` swaps
  MuJoCo Warp, mjbatch or Genesis without touching anything else here.

Run it with ``train.py`` next to this file. Rewards and gains follow mjlab's
``Mjlab-Velocity-Flat-Unitree-Go1`` task, so a number here means what it means
there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from rlmcp.backends import RobotSpec, SimOptions, make_backend
from rlmcp.backends.base import gain_for
from rlmcp.backends.frames import projected_gravity, quat_from_euler_xyz, quat_rotate_inverse
from rlmcp.declare import Static, Term, term, terms


def find_go1_xml() -> str:
  """mjlab's Go1 description, however this machine has mjlab.

  ``$MJLAB_GO1_XML`` wins; otherwise the installed mjlab package. The file is
  plain MJCF and every backend reads it, so there is one robot, not three.
  """
  override = os.environ.get("MJLAB_GO1_XML")
  if override:
    return str(Path(override).expanduser())
  try:
    import mjlab
  except ImportError:
    return "go1.xml"  # Resolved at construction, with a real error message.
  return str(Path(mjlab.__file__).parent / "asset_zoo/robots/unitree_go1/xmls/go1.xml")


# ---------------------------------------------------------------------------
# Configuration. One dataclass tree; rlmcp reads it directly.
# ---------------------------------------------------------------------------


@dataclass
class Robot:
  """The robot, read once when the simulator is built."""

  xml: str = field(default_factory=find_go1_xml)
  stiffness: dict[str, float] = field(
      default_factory=lambda: {"*_calf_joint": 35.76, "*": 15.89})
  damping: dict[str, float] = field(
      default_factory=lambda: {"*_calf_joint": 2.28, "*": 1.01})
  effort_limit: dict[str, float] = field(
      default_factory=lambda: {"*_calf_joint": 35.55, "*": 23.7})
  default_joint_pos: dict[str, float] = field(default_factory=lambda: {
      "FR_hip_joint": 0.1, "FL_hip_joint": -0.1, "RR_hip_joint": 0.1, "RL_hip_joint": -0.1,
      "*_thigh_joint": 0.9, "*_calf_joint": -1.8,
  })
  feet: tuple[str, ...] = ("FR", "FL", "RR", "RL")
  """Sites whose contact force is a foot on the ground."""
  base: str = "trunk"
  """The body whose contact with the ground ends the episode."""
  standing_height: float = 0.278


@dataclass
class Sim:
  """Timing and solver settings, read once when the simulator is built."""

  dt: float = 0.005
  decimation: int = 4
  options: SimOptions = field(default_factory=SimOptions)


@dataclass
class Rewards:
  """One term per line: weight, then the parameters its function reads."""

  tracking_lin_vel: Term = term(2.0, sigma=0.5)
  tracking_ang_vel: Term = term(1.0, sigma=0.7071)
  upright: Term = term(1.0, sigma=0.4472)
  action_rate: Term = term(-0.1)
  dof_pos_limits: Term = term(-1.0)
  feet_air_time: Term = term(0.0, threshold=0.5)
  dof_torques: Term = term(-1e-5)


@dataclass
class Commands:
  """Velocity command ranges; the first three are the ``commands`` columns."""

  lin_vel_x: tuple[float, float] = (0.3, 1.0)
  lin_vel_y: tuple[float, float] = (-0.3, 0.3)
  ang_vel_yaw: tuple[float, float] = (-0.5, 0.5)
  resample_time_s: float = 7.0
  dead_zone: float = 0.2
  """A planar command shorter than this is zeroed: stand still, do not creep."""


@dataclass
class Observation:
  """Scales applied before the policy sees a signal, legged_gym style."""

  lin_vel: float = 2.0
  ang_vel: float = 0.25
  dof_pos: float = 1.0
  dof_vel: float = 0.05


@dataclass
class Noise:
  """Standard deviation of the noise added to each observed signal, in its
  own units, before scaling. ``level`` multiplies all of them; 0 is clean."""

  level: float = 1.0
  base_lin_vel: float = 0.1
  base_ang_vel: float = 0.2
  projected_gravity: float = 0.05
  dof_pos: float = 0.01
  dof_vel: float = 1.5


@dataclass
class Termination:
  max_tilt_rad: float = 0.8
  base_contact_force: float = 1.0
  """Newtons on the base body: any real contact is a fall."""


@dataclass
class Randomization:
  friction: tuple[float, float] = (0.5, 1.25)
  """Sliding friction drawn per environment at every reset."""
  reset_pose_noise: float = 0.1
  """Radians of uniform noise around the default joint pose at reset."""


@dataclass
class EnvConfig:
  backend: Static[str] = "mjwarp"
  """``mjwarp``, ``mjbatch`` or ``genesis``. Same task either way."""
  num_envs: Static[int] = 4096
  device: Static[str] = "cuda"
  episode_length_s: Static[float] = 20.0
  action_scale: float = 0.25
  """Joint target = default pose + action * this, radians."""
  action_clip: float = 100.0
  """Actions are clamped to +-this before scaling. legged_gym's 100 is "not
  at all": a clip at 1.0 would cap every joint at a quarter radian from the
  default pose, which is too little for a gait, and the policy settles for
  standing."""
  robot: Static[Robot] = field(default_factory=Robot)
  sim: Static[Sim] = field(default_factory=Sim)
  reward: Rewards = field(default_factory=Rewards)
  command: Commands = field(default_factory=Commands)
  observation: Observation = field(default_factory=Observation)
  noise: Noise = field(default_factory=Noise)
  termination: Termination = field(default_factory=Termination)
  randomization: Randomization = field(default_factory=Randomization)


# ---------------------------------------------------------------------------
# The environment.
# ---------------------------------------------------------------------------


class Go1FlatEnv:
  """Velocity tracking on flat ground. ``step()`` is the whole story."""

  def __init__(self, cfg: EnvConfig | None = None):
    self.cfg = cfg or EnvConfig()
    cfg = self.cfg
    self.device = torch.device(cfg.device)
    self.num_envs = cfg.num_envs
    self.control_dt = cfg.sim.dt * cfg.sim.decimation
    self.max_episode_steps = int(cfg.episode_length_s / self.control_dt)

    robot = RobotSpec(
        xml=cfg.robot.xml,
        stiffness=cfg.robot.stiffness,
        damping=cfg.robot.damping,
        effort_limit=cfg.robot.effort_limit,
        contact_sites=(*cfg.robot.feet, cfg.robot.base),
    )
    self.sim = make_backend(
        cfg.backend, robot, cfg.num_envs, dt=cfg.sim.dt,
        decimation=cfg.sim.decimation, device=cfg.device, options=cfg.sim.options,
    )
    self.joint_names = self.sim.joint_names
    self.num_actions = self.num_dof = len(self.joint_names)
    self.num_feet = len(cfg.robot.feet)
    self.num_obs = 3 + 3 + 3 + 3 + 2 * self.num_dof + self.num_actions

    self.default_dof_pos = torch.tensor(
        [gain_for(cfg.robot.default_joint_pos, name) for name in self.joint_names],
        dtype=torch.float32, device=self.device,
    )
    self.dof_limits = self.sim.dof_limits.to(self.device)

    n, j = self.num_envs, self.num_dof
    zeros = lambda *shape, **kw: torch.zeros(*shape, device=self.device, **kw)  # noqa: E731
    self.base_pos, self.base_quat = zeros(n, 3), zeros(n, 4)
    self.base_lin_vel, self.base_ang_vel = zeros(n, 3), zeros(n, 3)
    self.projected_gravity = zeros(n, 3)
    self.dof_pos, self.dof_vel, self.torques = zeros(n, j), zeros(n, j), zeros(n, j)
    self.actions, self.last_actions = zeros(n, j), zeros(n, j)
    self.commands = zeros(n, 3)
    self.foot_contact = zeros(n, self.num_feet, dtype=torch.bool)
    self.last_foot_contact = zeros(n, self.num_feet, dtype=torch.bool)
    self.feet_air_time = zeros(n, self.num_feet)
    self.air_time_at_contact = zeros(n, self.num_feet)
    self.base_contact = zeros(n)
    self.episode_length_buf = zeros(n, dtype=torch.long)
    self.episode_reward = zeros(n)
    self.rew_buf = zeros(n)
    self.obs_buf = zeros(n, self.num_obs)

  # -- Reset ----------------------------------------------------------------

  def reset(self, env_ids: Tensor | None = None) -> Tensor:
    """Start fresh episodes for ``env_ids`` (all of them when None)."""
    cfg = self.cfg
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    if len(env_ids) == 0:
      return self._compute_observations()
    n = len(env_ids)

    root_pos = torch.zeros(n, 3, device=self.device)
    root_pos[:, 2] = cfg.robot.standing_height + 0.02
    yaw = torch.empty(n, device=self.device).uniform_(-torch.pi, torch.pi)
    root_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    dof_pos = self.default_dof_pos.expand(n, -1) + torch.empty(
        n, self.num_dof, device=self.device).uniform_(
            -cfg.randomization.reset_pose_noise, cfg.randomization.reset_pose_noise)
    dof_pos = torch.clamp(dof_pos, self.dof_limits[:, 0], self.dof_limits[:, 1])
    self.sim.reset(env_ids, root_pos, root_quat, dof_pos)
    friction = torch.empty(n, device=self.device).uniform_(*cfg.randomization.friction)
    self.sim.set_friction(env_ids, friction)

    self.actions[env_ids] = 0.0
    self.last_actions[env_ids] = 0.0
    self.feet_air_time[env_ids] = 0.0
    self.foot_contact[env_ids] = False
    self.last_foot_contact[env_ids] = False
    self.episode_length_buf[env_ids] = 0
    self.episode_reward[env_ids] = 0.0
    self._resample_commands(env_ids)
    self._read_state()
    return self._compute_observations()

  def _resample_commands(self, env_ids: Tensor) -> None:
    c = self.cfg.command
    n = len(env_ids)
    sample = lambda lo_hi: torch.empty(n, device=self.device).uniform_(*lo_hi)  # noqa: E731
    self.commands[env_ids, 0] = sample(c.lin_vel_x)
    self.commands[env_ids, 1] = sample(c.lin_vel_y)
    self.commands[env_ids, 2] = sample(c.ang_vel_yaw)
    small = torch.norm(self.commands[env_ids, :2], dim=-1) < c.dead_zone
    self.commands[env_ids[small]] = 0.0

  # -- Step -----------------------------------------------------------------

  def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, dict]:
    """One control step: act, simulate, observe, score, terminate, reset."""
    cfg = self.cfg

    # 1. Actions become joint position targets around the default pose.
    self.last_actions[:] = self.actions
    self.actions[:] = torch.clamp(actions.to(self.device), -cfg.action_clip, cfg.action_clip)
    targets = self.default_dof_pos + self.actions * cfg.action_scale

    # 2. Physics: `decimation` substeps, then the state comes back.
    self.sim.set_dof_targets(targets)
    self.sim.step()
    self._read_state()

    # 3. Feet: contact edges and air time.
    first_contact = self.foot_contact & ~self.last_foot_contact
    self.feet_air_time += self.control_dt
    self.air_time_at_contact = self.feet_air_time * first_contact.float()
    self.feet_air_time[self.foot_contact] = 0.0

    # 4. Observe, score, terminate.
    obs = self._compute_observations()
    self.rew_buf, reward_terms = self._compute_rewards()
    time_out = self.episode_length_buf >= self.max_episode_steps - 1
    fell = (self.projected_gravity[:, 2] > -torch.cos(torch.tensor(cfg.termination.max_tilt_rad))) \
        | (self.base_contact > cfg.termination.base_contact_force)
    dones = time_out | fell

    # 5. Bookkeeping, resampling, and the resets for whoever finished.
    self.episode_length_buf += 1
    self.episode_reward += self.rew_buf
    every = max(1, int(cfg.command.resample_time_s / self.control_dt))
    due = (self.episode_length_buf % every == 0).nonzero(as_tuple=False).squeeze(-1)
    if len(due) > 0:
      self._resample_commands(due)
    done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
    info = {
        "reward_terms": {name: float(value.mean()) for name, value in reward_terms.items()},
        "episode_rewards": self.episode_reward[done_ids].clone(),
        "episode_lengths": self.episode_length_buf[done_ids].float().clone(),
        "time_outs": time_out,
    }
    if len(done_ids) > 0:
      obs = self.reset(done_ids)
    return obs, self.rew_buf, dones, info

  # -- State ----------------------------------------------------------------

  def _read_state(self) -> None:
    """Pull the backend's state into the buffers everything else reads."""
    sim, feet = self.sim, self.num_feet
    self.base_pos[:] = sim.root_pos.to(self.device)
    self.base_quat[:] = sim.root_quat.to(self.device)
    self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, sim.root_lin_vel.to(self.device))
    self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, sim.root_ang_vel.to(self.device))
    self.projected_gravity[:] = projected_gravity(self.base_quat)
    self.dof_pos[:] = sim.dof_pos.to(self.device)
    self.dof_vel[:] = sim.dof_vel.to(self.device)
    self.torques[:] = sim.dof_torque.to(self.device)
    contact = sim.contact_forces.to(self.device)
    self.last_foot_contact[:] = self.foot_contact
    self.foot_contact[:] = contact[:, :feet] > 1.0
    self.base_contact[:] = contact[:, feet]

  # -- Observations ---------------------------------------------------------

  def _compute_observations(self) -> Tensor:
    """48 numbers: velocities, gravity, command, joints, last action."""
    noise, scale = self.cfg.noise, self.cfg.observation

    def noisy(value: Tensor, std: float, gain: float = 1.0) -> Tensor:
      if noise.level > 0.0 and std > 0.0:
        value = value + torch.randn_like(value) * (std * noise.level)
      return value * gain

    self.obs_buf = torch.cat([
        noisy(self.base_lin_vel, noise.base_lin_vel, scale.lin_vel),
        noisy(self.base_ang_vel, noise.base_ang_vel, scale.ang_vel),
        noisy(self.projected_gravity, noise.projected_gravity),
        self.commands * torch.tensor(
            [scale.lin_vel, scale.lin_vel, scale.ang_vel], device=self.device),
        noisy(self.dof_pos - self.default_dof_pos, noise.dof_pos, scale.dof_pos),
        noisy(self.dof_vel, noise.dof_vel, scale.dof_vel),
        self.actions,
    ], dim=-1)
    return self.obs_buf

  # -- Rewards --------------------------------------------------------------

  def _compute_rewards(self) -> tuple[Tensor, dict[str, Tensor]]:
    """Every term, scored inline; the table at the top weights them.

    A term the table has that this method does not compute is one rlmcp
    appended at runtime, and it carries its own function.
    """
    r = self.cfg.reward
    lin_err = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=-1)
    ang_err = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    out_of_limits = (
        (self.dof_pos - self.dof_limits[:, 0]).clamp(max=0.0).abs()
        + (self.dof_pos - self.dof_limits[:, 1]).clamp(min=0.0)
    )
    computed = {
        "tracking_lin_vel": torch.exp(
            -(lin_err + torch.square(self.base_lin_vel[:, 2])) / r.tracking_lin_vel.sigma ** 2),
        "tracking_ang_vel": torch.exp(
            -(ang_err + torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1))
            / r.tracking_ang_vel.sigma ** 2),
        "upright": torch.exp(
            -torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1) / r.upright.sigma ** 2),
        "action_rate": torch.sum(torch.square(self.actions - self.last_actions), dim=-1),
        "dof_pos_limits": torch.sum(out_of_limits, dim=-1),
        "feet_air_time": torch.sum(
            (self.air_time_at_contact - r.feet_air_time.threshold).clamp(min=0.0), dim=-1),
        "dof_torques": torch.sum(torch.square(self.torques), dim=-1),
    }

    total = torch.zeros(self.num_envs, device=self.device)
    scored: dict[str, Tensor] = {}
    for name, t in terms(r).items():
      value = computed[name] if name in computed else t.func(self, **t.params)
      scored[name] = value
      total += t.weight * value * self.control_dt
    return total, scored

  # -- Sizes ----------------------------------------------------------------

  @property
  def obs_dim(self) -> int:
    return self.num_obs

  @property
  def action_dim(self) -> int:
    return self.num_actions

  def close(self) -> None:
    self.sim.close()
