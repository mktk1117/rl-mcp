"""Go1 flat-ground velocity tracking, in one file.

This file *is* the task. Every number an agent might tune is in the config
dataclasses at the top; the physics loop, observations, rewards and
terminations are inline below them, in the order they happen. Nothing is
registered and the base class holds only the contract: read it top to bottom
and you know the whole environment.

The task is mjlab's ``Mjlab-Velocity-Flat-Unitree-Go1``, term for term: the
same observations (and the critic's extra ones), the same command generator
with heading control, the same rewards with the same weights and widths, the
same disturbances and startup randomisation, the same termination. Where
this file differs it says so in a comment. A number here means what it means
there, so an agent that learned the mjlab task's levers can drive this one.

Three things make it steerable from another shell with ``rlmcp``:

* the config is declared with :mod:`rlmcp.declare` -- ``Static[...]`` marks a
  value read once at construction, ``term(...)`` declares a reward term -- so
  rlmcp knows what it may change live and what would need a restart;
* the environment is a :class:`~rlmcp.adapters.single_file.SingleFileEnv`,
  which names the buffers rlmcp reads and owns the reward loop, so a term an
  agent adds at runtime is scored without this file knowing about it;
* the simulator is behind :mod:`rlmcp.backends`, one robot-level contract
  with three simulators behind it, so ``cfg.backend`` swaps MuJoCo Warp,
  mjbatch or Genesis without touching anything else here.

Run it with ``train.py`` next to this file.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from rlmcp.adapters.single_file import SingleFileEnv
from rlmcp.backends import RobotSpec, SimOptions, make_backend
from rlmcp.backends.frames import (
  projected_gravity,
  quat_from_euler_xyz,
  quat_rotate_inverse,
  wrap_to_pi,
)
from rlmcp.declare import Static, Term, term


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
  """The robot, read once when the simulator is built. Gains are mjlab's:
  a 10 Hz natural frequency and damping ratio 2 on the reflected rotor
  inertia, effort limits from the Go1 datasheet."""

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
  """Sites at the foot centres: contact force, height and slip are read here.
  (The backend's own answer would be the four calf bodies, the leaves of the
  tree; the sites sit at the ball of each foot, which is what the clearance
  and slip terms want.)"""
  foot_geoms: tuple[str, ...] = (
      "FR_foot_collision", "FL_foot_collision", "RR_foot_collision", "RL_foot_collision")
  """The geoms that get mjlab's foot contact tuning and the friction draw."""
  foot_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
  """mjlab's foot friction (sliding, torsional, rolling), with contact
  priority so it wins against the floor's."""
  geom_solref: tuple[float, float] = (0.01, 1.0)
  """mjlab hardens every collision geom to this."""
  standing_height: float = 0.278
  soft_limit_factor: float = 0.9
  """Joint limits are shrunk to this fraction of their range for the limit
  penalty, as mjlab's ``soft_joint_pos_limit_factor`` does."""


@dataclass
class Sim:
  """Timing and solver settings, read once when the simulator is built.
  mjlab's flat Go1: implicitfast, Newton, elliptic cone, impratio 10."""

  dt: float = 0.005
  decimation: int = 4
  options: SimOptions = field(default_factory=lambda: SimOptions(
      iterations=10, ls_iterations=20, cone="elliptic", impratio=10.0,
      ccd_iterations=50, njmax=300))


@dataclass
class Action:
  """Joint target = default pose + action * scale. mjlab scales each joint by
  0.25 * effort_limit / stiffness, which is these two numbers."""

  scale_hip_thigh: float = 0.373
  scale_calf: float = 0.249


@dataclass
class Rewards:
  """mjlab's flat Go1 reward table: one term per line, weight then params.
  Widths are ``std`` in exp(-x/std^2); every term is multiplied by the
  control timestep, as mjlab's reward manager does."""

  track_linear_velocity: Term = term(2.0, std=0.5)
  track_angular_velocity: Term = term(2.0, std=0.7071)
  upright: Term = term(1.0, std=0.4472)
  pose: Term = term(
      1.0, std_standing_hip_thigh=0.05, std_standing_calf=0.1,
      std_walking_hip_thigh=0.3, std_walking_calf=0.6,
      std_running_hip_thigh=0.3, std_running_calf=0.6,
      walking_threshold=0.05, running_threshold=1.5)
  dof_pos_limits: Term = term(-1.0)
  action_rate_l2: Term = term(-0.1)
  air_time: Term = term(0.0, threshold_min=0.05, threshold_max=0.5, command_threshold=0.5)
  foot_clearance: Term = term(-2.0, target_height=0.1, command_threshold=0.05)
  foot_swing_height: Term = term(-0.25, target_height=0.1, command_threshold=0.05)
  foot_slip: Term = term(-0.1, command_threshold=0.05)
  soft_landing: Term = term(-1e-5, command_threshold=0.05)
  body_ang_vel: Term = term(0.0)


@dataclass
class Commands:
  """mjlab's ``twist`` command. The first three ranges are the ``commands``
  columns; ``heading`` is the target yaw for the heading-controlled envs,
  whose yaw-rate command is ``heading_stiffness * heading error`` instead."""

  lin_vel_x: tuple[float, float] = (-1.0, 1.0)
  lin_vel_y: tuple[float, float] = (-1.0, 1.0)
  ang_vel_z: tuple[float, float] = (-0.5, 0.5)
  heading: tuple[float, float] = (-math.pi, math.pi)
  resampling_time_s: tuple[float, float] = (3.0, 8.0)
  rel_standing_envs: float = 0.1
  """Fraction of envs whose command is zero: learn to stand still too."""
  rel_heading_envs: float = 0.3
  rel_forward_envs: float = 0.2
  """Fraction of envs commanded straight ahead only, at least ``forward_min_speed``."""
  forward_min_speed: float = 0.3
  heading_stiffness: float = 0.5


@dataclass
class Noise:
  """Half-width of the uniform noise added to each actor observation, in its
  own units. ``level`` multiplies all of them; 0 is clean. The critic never
  sees noise."""

  level: float = 1.0
  base_lin_vel: float = 0.5
  base_ang_vel: float = 0.2
  projected_gravity: float = 0.05
  joint_pos: float = 0.01
  joint_vel: float = 1.5


@dataclass
class Termination:
  fell_over_deg: float = 70.0
  """Angle between the base's up axis and the world's that ends the episode."""


@dataclass
class Startup:
  """Drawn once per environment when it is built, never again."""

  foot_friction: tuple[float, float] = (0.3, 1.5)
  encoder_bias: tuple[float, float] = (-0.015, 0.015)
  """Radians added to the joint positions the *actor* sees, per joint."""


@dataclass
class Randomization:
  startup: Static[Startup] = field(default_factory=Startup)
  push_interval_s: tuple[float, float] = (1.0, 3.0)
  push_lin_vel: tuple[float, float] = (-0.5, 0.5)
  """Added to the base's x and y velocity (world frame) at each push."""
  push_lin_vel_z: tuple[float, float] = (-0.4, 0.4)
  push_ang_vel_xy: tuple[float, float] = (-0.52, 0.52)
  push_ang_vel_z: tuple[float, float] = (-0.78, 0.78)
  reset_xy: float = 0.5
  reset_z: tuple[float, float] = (0.01, 0.05)
  """Added to the standing height at reset."""


@dataclass
class EnvConfig:
  backend: Static[str] = "mjwarp"
  """``mjwarp``, ``mjbatch`` or ``genesis``. Same task either way."""
  num_envs: Static[int] = 4096
  device: Static[str] = "cuda"
  episode_length_s: Static[float] = 20.0
  robot: Static[Robot] = field(default_factory=Robot)
  sim: Static[Sim] = field(default_factory=Sim)
  action: Action = field(default_factory=Action)
  reward: Rewards = field(default_factory=Rewards)
  command: Commands = field(default_factory=Commands)
  noise: Noise = field(default_factory=Noise)
  termination: Termination = field(default_factory=Termination)
  randomization: Randomization = field(default_factory=Randomization)


# ---------------------------------------------------------------------------
# The environment.
# ---------------------------------------------------------------------------


def _uniform(n: int, lo_hi: tuple[float, float], device: torch.device) -> Tensor:
  return torch.empty(n, device=device).uniform_(*lo_hi)


class Go1FlatEnv(SingleFileEnv):
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
        default_joint_pos=cfg.robot.default_joint_pos,
        contacts=tuple(cfg.robot.feet),
        contact_geoms=cfg.robot.foot_geoms,
        contact_friction=cfg.robot.foot_friction,
        contact_condim=3,
        contact_priority=1,
        geom_solref=cfg.robot.geom_solref,
    )
    self.sim = make_backend(
        cfg.backend, robot, cfg.num_envs, dt=cfg.sim.dt,
        decimation=cfg.sim.decimation, device=cfg.device, options=cfg.sim.options,
    )
    self.joint_names = self.sim.joint_names
    self.num_actions = self.num_dof = len(self.joint_names)
    self.num_feet = len(cfg.robot.feet)
    self.num_obs = 3 + 3 + 3 + 2 * self.num_dof + self.num_actions + 3
    self.num_critic_obs = self.num_obs + 3 * self.num_feet

    dev = self.device
    self.default_dof_pos = self.sim.default_dof_pos.to(dev)
    limits = self.sim.dof_limits.to(dev)
    mid, half = (limits[:, 0] + limits[:, 1]) / 2, (limits[:, 1] - limits[:, 0]) / 2
    self.soft_limits = torch.stack(
        [mid - cfg.robot.soft_limit_factor * half, mid + cfg.robot.soft_limit_factor * half],
        dim=1)
    self.is_calf = torch.tensor(
        [n.endswith("_calf_joint") for n in self.joint_names], device=dev)

    n, j, f = self.num_envs, self.num_dof, self.num_feet
    zeros = lambda *shape, **kw: torch.zeros(*shape, device=dev, **kw)  # noqa: E731
    # Base and joints, in the conventional names the rlmcp trace reads.
    self.base_pos, self.base_quat = zeros(n, 3), zeros(n, 4)
    self.base_lin_vel, self.base_ang_vel = zeros(n, 3), zeros(n, 3)
    self.projected_gravity = zeros(n, 3)
    self.dof_pos, self.dof_vel, self.torques = zeros(n, j), zeros(n, j), zeros(n, j)
    self.actions, self.last_actions = zeros(n, j), zeros(n, j)
    self.rew_buf = zeros(n)
    self.episode_length_buf = zeros(n, dtype=torch.long)
    self.episode_reward = zeros(n)
    # Feet.
    self.foot_pos, self.foot_vel = zeros(n, f, 3), zeros(n, f, 3)
    self.foot_force = zeros(n, f)
    self.foot_contact = zeros(n, f, dtype=torch.bool)
    self.last_foot_contact = zeros(n, f, dtype=torch.bool)
    self.first_contact = zeros(n, f, dtype=torch.bool)
    self.foot_air_time = zeros(n, f)
    self.foot_peak_height = zeros(n, f)
    # Commands: the buffer the policy sees plus the generator's state.
    self.commands = zeros(n, 3)
    self.command_timer = zeros(n)
    self.heading_target = zeros(n)
    self.is_heading_env = zeros(n, dtype=torch.bool)
    self.is_standing_env = zeros(n, dtype=torch.bool)
    self.is_forward_env = zeros(n, dtype=torch.bool)
    # Disturbances.
    self.push_timer = zeros(n)
    # Startup randomisation: once, and it stays.
    startup = cfg.randomization.startup
    all_ids = torch.arange(n, device=dev)
    self.sim.set_friction(all_ids, _uniform(n, startup.foot_friction, dev))
    self.encoder_bias = torch.empty(n, j, device=dev).uniform_(*startup.encoder_bias)
    self.obs_buf = zeros(n, self.num_obs)
    self.critic_obs_buf = zeros(n, self.num_critic_obs)

  @property
  def action_scale(self) -> Tensor:
    """Per-joint scale, read from the config every step so it is live."""
    a = self.cfg.action
    return torch.where(self.is_calf, torch.full_like(self.default_dof_pos, a.scale_calf),
                       torch.full_like(self.default_dof_pos, a.scale_hip_thigh))

  # -- Reset ----------------------------------------------------------------

  def reset(self, env_ids: Tensor | None = None) -> Tensor:
    """Start fresh episodes for ``env_ids`` (all of them when None)."""
    cfg = self.cfg
    dev = self.device
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=dev)
    if len(env_ids) == 0:
      return self._compute_observations()
    n = len(env_ids)
    r = cfg.randomization

    # Pose: a little offset in the plane, a little above standing, any yaw;
    # joints exactly at the default pose, everything at rest.
    root_pos = torch.zeros(n, 3, device=dev)
    root_pos[:, 0] = _uniform(n, (-r.reset_xy, r.reset_xy), dev)
    root_pos[:, 1] = _uniform(n, (-r.reset_xy, r.reset_xy), dev)
    root_pos[:, 2] = cfg.robot.standing_height + _uniform(n, r.reset_z, dev)
    yaw = _uniform(n, (-math.pi, math.pi), dev)
    root_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    self.sim.reset(env_ids, self.default_dof_pos.expand(n, -1),
                   root_pos=root_pos, root_quat=root_quat)

    self.actions[env_ids] = 0.0
    self.last_actions[env_ids] = 0.0
    self.foot_air_time[env_ids] = 0.0
    self.foot_peak_height[env_ids] = 0.0
    self.foot_contact[env_ids] = False
    self.last_foot_contact[env_ids] = False
    self.first_contact[env_ids] = False
    self.episode_length_buf[env_ids] = 0
    self.episode_reward[env_ids] = 0.0
    self.push_timer[env_ids] = _uniform(n, r.push_interval_s, dev)
    self._read_state()
    self.foot_vel[env_ids] = 0.0
    self._resample_commands(env_ids)
    return self._compute_observations()

  # -- Commands -------------------------------------------------------------

  def _resample_commands(self, env_ids: Tensor) -> None:
    """mjlab's UniformVelocityCommand: a twist per env, a target heading for
    some, zero for some, straight ahead for some."""
    c = self.cfg.command
    n, dev = len(env_ids), self.device
    self.commands[env_ids, 0] = _uniform(n, c.lin_vel_x, dev)
    self.commands[env_ids, 1] = _uniform(n, c.lin_vel_y, dev)
    self.commands[env_ids, 2] = _uniform(n, c.ang_vel_z, dev)
    self.heading_target[env_ids] = _uniform(n, c.heading, dev)
    self.is_heading_env[env_ids] = _uniform(n, (0.0, 1.0), dev) <= c.rel_heading_envs
    self.is_standing_env[env_ids] = _uniform(n, (0.0, 1.0), dev) <= c.rel_standing_envs
    self.is_forward_env[env_ids] = _uniform(n, (0.0, 1.0), dev) <= c.rel_forward_envs
    fwd = env_ids[self.is_forward_env[env_ids]]
    self.commands[fwd, 0] = self.commands[fwd, 0].abs().clamp(min=c.forward_min_speed)
    self.commands[fwd, 1:] = 0.0
    self.command_timer[env_ids] = _uniform(n, c.resampling_time_s, dev)
    self._update_commands()

  def _update_commands(self) -> None:
    """Every step: heading envs steer toward their target yaw; standing envs
    are held at zero."""
    c = self.cfg.command
    q = self.base_quat
    heading = torch.atan2(
        2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
        1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    error = wrap_to_pi(self.heading_target - heading)
    steer = torch.clip(c.heading_stiffness * error, c.ang_vel_z[0], c.ang_vel_z[1])
    self.commands[:, 2] = torch.where(self.is_heading_env, steer, self.commands[:, 2])
    self.commands[self.is_standing_env] = 0.0

  @property
  def command_speed(self) -> Tensor:
    """|planar command| + |yaw command|: what "moving" means to the terms."""
    return torch.norm(self.commands[:, :2], dim=1) + self.commands[:, 2].abs()

  # -- Step -----------------------------------------------------------------

  def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, dict]:
    """One control step: act, simulate, observe, score, terminate, reset."""
    cfg = self.cfg
    dev = self.device

    # 1. Actions become joint position targets around the default pose.
    self.last_actions[:] = self.actions
    self.actions[:] = actions.to(dev)
    self.sim.set_dof_targets(self.default_dof_pos + self.actions * self.action_scale)

    # 2. Disturbances are due for some envs: a kick to the base velocity.
    self.push_timer -= self.control_dt
    push_ids = (self.push_timer <= 0.0).nonzero(as_tuple=False).squeeze(-1)
    if len(push_ids) > 0:
      self._push(push_ids)

    # 3. Physics: `decimation` substeps, then the state comes back.
    self.sim.step()
    self._read_state()

    # 4. The command generator ticks.
    self.command_timer -= self.control_dt
    due = (self.command_timer <= 0.0).nonzero(as_tuple=False).squeeze(-1)
    if len(due) > 0:
      self._resample_commands(due)
    self._update_commands()

    # 5. Observe, score, terminate.
    obs = self._compute_observations()
    # Every term is multiplied by the control timestep, as mjlab's manager does.
    self.rew_buf, reward_terms = self.compute_reward(scale=self.control_dt)
    time_out = self.episode_length_buf >= self.max_episode_steps - 1
    tilt = torch.acos(torch.clamp(-self.projected_gravity[:, 2], -1.0, 1.0))
    fell = tilt > math.radians(cfg.termination.fell_over_deg)
    dones = time_out | fell

    # 6. Bookkeeping, and the resets for whoever finished.
    self.episode_length_buf += 1
    self.episode_reward += self.rew_buf
    done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
    info = self.step_info(
        reward_terms,
        episode_rewards=self.episode_reward[done_ids].clone(),
        episode_lengths=self.episode_length_buf[done_ids].float().clone(),
        time_outs=time_out,
    )
    if len(done_ids) > 0:
      obs = self.reset(done_ids)
    return obs, self.rew_buf, dones, info

  def _push(self, env_ids: Tensor) -> None:
    r = self.cfg.randomization
    n, dev = len(env_ids), self.device
    lin = torch.stack([_uniform(n, r.push_lin_vel, dev), _uniform(n, r.push_lin_vel, dev),
                       _uniform(n, r.push_lin_vel_z, dev)], dim=1)
    ang = torch.stack([_uniform(n, r.push_ang_vel_xy, dev), _uniform(n, r.push_ang_vel_xy, dev),
                       _uniform(n, r.push_ang_vel_z, dev)], dim=1)
    self.sim.push(env_ids, lin, ang)
    self.push_timer[env_ids] = _uniform(n, r.push_interval_s, dev)

  # -- State ----------------------------------------------------------------

  def _read_state(self) -> None:
    """Pull the backend's state into the buffers everything else reads."""
    sim, dev = self.sim, self.device
    self.base_pos[:] = sim.root_pos.to(dev)
    self.base_quat[:] = sim.root_quat.to(dev)
    self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, sim.root_lin_vel.to(dev))
    self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, sim.root_ang_vel.to(dev))
    self.projected_gravity[:] = projected_gravity(self.base_quat)
    self.dof_pos[:] = sim.dof_pos.to(dev)
    self.dof_vel[:] = sim.dof_vel.to(dev)
    self.torques[:] = sim.dof_torque.to(dev)
    # Feet: force, contact edges, air time, height and a finite-difference
    # velocity -- the flat-ground stand-ins for mjlab's contact and height
    # sensors.
    pos = sim.contact_site_pos.to(dev)
    self.foot_vel[:] = (pos - self.foot_pos) / self.control_dt
    self.foot_pos[:] = pos
    self.foot_force[:] = sim.contact_forces.to(dev)
    self.last_foot_contact[:] = self.foot_contact
    self.foot_contact[:] = self.foot_force > 1.0
    self.first_contact[:] = self.foot_contact & ~self.last_foot_contact
    self.foot_air_time += self.control_dt
    self.foot_air_time[self.foot_contact] = 0.0
    self.foot_peak_height[:] = torch.where(
        ~self.foot_contact, torch.maximum(self.foot_peak_height, pos[:, :, 2]),
        self.foot_peak_height)

  # -- Observations ---------------------------------------------------------

  def _compute_observations(self) -> Tensor:
    """The actor's 48 numbers, noisy; the critic's 60, clean and privileged."""
    noise = self.cfg.noise

    def noisy(value: Tensor, half_width: float) -> Tensor:
      if noise.level <= 0.0 or half_width <= 0.0:
        return value
      return value + (torch.rand_like(value) * 2.0 - 1.0) * (half_width * noise.level)

    joint_pos_rel = self.dof_pos - self.default_dof_pos
    self.obs_buf = torch.cat([
        noisy(self.base_lin_vel, noise.base_lin_vel),
        noisy(self.base_ang_vel, noise.base_ang_vel),
        noisy(self.projected_gravity, noise.projected_gravity),
        noisy(joint_pos_rel + self.encoder_bias, noise.joint_pos),
        noisy(self.dof_vel, noise.joint_vel),
        self.actions,
        self.commands,
    ], dim=-1)
    self.critic_obs_buf = torch.cat([
        self.base_lin_vel,
        self.base_ang_vel,
        self.projected_gravity,
        joint_pos_rel,
        self.dof_vel,
        self.actions,
        self.commands,
        self.foot_air_time,
        self.foot_contact.float(),
        torch.log1p(self.foot_force),
    ], dim=-1)
    return self.obs_buf

  # -- Rewards --------------------------------------------------------------

  def compute_reward_terms(self) -> dict[str, Tensor]:
    """Every term this file computes, unweighted; the table at the top weights
    them in :meth:`SingleFileEnv.compute_reward`, which also scores any term
    rlmcp appended at runtime."""
    r = self.cfg.reward
    cmd, speed = self.commands, self.command_speed
    lin_err = torch.sum(torch.square(cmd[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    lin_err = lin_err + torch.square(self.base_lin_vel[:, 2])
    ang_err = torch.square(cmd[:, 2] - self.base_ang_vel[:, 2])
    ang_err = ang_err + torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    tilt_sq = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    # Posture: how far from the default pose, with a tolerance that widens
    # with the commanded speed.
    p = r.pose

    def stds(hip_thigh: float, calf: float) -> Tensor:
      return torch.where(self.is_calf, torch.full_like(self.default_dof_pos, calf),
                         torch.full_like(self.default_dof_pos, hip_thigh))

    standing = (speed < p.walking_threshold).unsqueeze(1)
    running = (speed >= p.running_threshold).unsqueeze(1)
    std = torch.where(standing, stds(p.std_standing_hip_thigh, p.std_standing_calf),
                      torch.where(running, stds(p.std_running_hip_thigh, p.std_running_calf),
                                  stds(p.std_walking_hip_thigh, p.std_walking_calf)))
    pose_err = torch.square(self.dof_pos - self.default_dof_pos) / std ** 2

    out_of_limits = (
        -(self.dof_pos - self.soft_limits[:, 0]).clip(max=0.0)
        + (self.dof_pos - self.soft_limits[:, 1]).clip(min=0.0))

    # Feet, while a command is active.
    def moving(threshold: float) -> Tensor:
      return (speed > threshold).float()

    foot_h = self.foot_pos[:, :, 2]
    foot_v = torch.norm(self.foot_vel[:, :, :2], dim=-1)
    in_range = (self.foot_air_time > r.air_time.threshold_min) & \
        (self.foot_air_time < r.air_time.threshold_max)
    swing_err = torch.square(self.foot_peak_height / r.foot_swing_height.target_height - 1.0)
    swing_cost = torch.sum(swing_err * self.first_contact.float(), dim=1)
    self.foot_peak_height[self.first_contact] = 0.0

    return {
        "track_linear_velocity": torch.exp(-lin_err / r.track_linear_velocity.std ** 2),
        "track_angular_velocity": torch.exp(-ang_err / r.track_angular_velocity.std ** 2),
        "upright": torch.exp(-tilt_sq / r.upright.std ** 2),
        "pose": torch.exp(-torch.mean(pose_err, dim=1)),
        "dof_pos_limits": torch.sum(out_of_limits, dim=1),
        "action_rate_l2": torch.sum(torch.square(self.actions - self.last_actions), dim=1),
        "air_time": torch.sum(in_range.float(), dim=1) * moving(r.air_time.command_threshold),
        "foot_clearance": torch.sum(
            torch.abs(foot_h - r.foot_clearance.target_height) * foot_v, dim=1)
        * moving(r.foot_clearance.command_threshold),
        "foot_swing_height": swing_cost * moving(r.foot_swing_height.command_threshold),
        "foot_slip": torch.sum(torch.square(foot_v) * self.foot_contact.float(), dim=1)
        * moving(r.foot_slip.command_threshold),
        "soft_landing": torch.sum(self.foot_force * self.first_contact.float(), dim=1)
        * moving(r.soft_landing.command_threshold),
        "body_ang_vel": torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1),
    }

  # -- Sizes ----------------------------------------------------------------

  @property
  def obs_dim(self) -> int:
    return self.num_obs

  @property
  def critic_obs_dim(self) -> int:
    return self.num_critic_obs

  @property
  def action_dim(self) -> int:
    return self.num_actions

  def close(self) -> None:
    self.sim.close()
