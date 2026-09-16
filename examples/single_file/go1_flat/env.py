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

It is built from four blocks (:mod:`rlmcp.blocks`), which is what makes it
steerable from another shell with ``rlmcp`` without listing anything twice:

* the **config** is declared with :mod:`rlmcp.declare` -- ``Static[...]``
  marks a value read once at construction, ``term(...)`` declares a reward
  term -- so rlmcp knows what it may change live and what needs a restart;
* the **variables** ``step()`` writes are one :class:`State`, a
  :class:`~rlmcp.blocks.Vars`: every tensor with a name, a shape and labels,
  so ``rlmcp trace`` records all of them and the diagnostics find the
  conventional ones;
* the **observations** are two :class:`~rlmcp.blocks.Obs` groups, each term
  a variable (or a function of the state) followed by its pipe -- the noise
  the actor sees is a stage in that pipe, served as
  ``actor_obs.joint_pos.noise.half_width``, and a ``Delay(k)`` would go in the
  same place;
* each **reward term** is a method with the name the table gives it and
  the table's params as its arguments; the base class sums
  ``weight * method(**params)``, and a term an agent adds at runtime is
  scored by the same loop.

The simulator is behind :mod:`rlmcp.backends`, one robot-level contract with
three simulators behind it, so ``cfg.backend`` swaps MuJoCo Warp, mjbatch or
Genesis without touching anything else here. Run it with ``train.py`` next
to this file.
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
from rlmcp.blocks import Noise, Obs, Offset, Static, Term, Vars, term, var


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
  Each name is a method of :class:`Go1FlatEnv` below, and the params are its
  arguments. Widths are ``std`` in exp(-x/std^2); every term is multiplied
  by the control timestep, as mjlab's reward manager does."""

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
  termination: Termination = field(default_factory=Termination)
  randomization: Randomization = field(default_factory=Randomization)


# ---------------------------------------------------------------------------
# Variables. Everything step() writes, one tensor per name, one row per env.
# ---------------------------------------------------------------------------


class State(Vars):
  """The names the rest of the file reads. rlmcp traces every one of them;
  the conventional ones (``joint_pos``, ``base_lin_vel``, ``commands``,
  ``foot_contact``, ``reward``, ...) also feed its diagnostics."""

  # Base and joints, from the simulator each step.
  base_pos: Tensor = var(3)
  base_quat: Tensor = var(4)
  base_lin_vel: Tensor = var(3, doc="base frame")
  base_ang_vel: Tensor = var(3, doc="base frame")
  projected_gravity: Tensor = var(3)
  joint_pos: Tensor = var("joint")
  joint_vel: Tensor = var("joint")
  joint_torque: Tensor = var("joint")
  actions: Tensor = var("joint")
  last_actions: Tensor = var("joint")
  # Feet: the flat-ground stand-ins for mjlab's contact and height sensors.
  foot_pos: Tensor = var("foot", 3)
  foot_vel: Tensor = var("foot", 3, doc="finite difference of foot_pos")
  foot_force: Tensor = var("foot")
  foot_contact: Tensor = var("foot", dtype=torch.bool)
  last_foot_contact: Tensor = var("foot", dtype=torch.bool)
  first_contact: Tensor = var("foot", dtype=torch.bool)
  foot_air_time: Tensor = var("foot")
  foot_peak_height: Tensor = var("foot", doc="highest point of the current swing")
  # Commands: what the policy sees, plus the generator's state.
  commands: Tensor = var(("lin_vel_x", "lin_vel_y", "ang_vel_z"))
  command_speed: Tensor = var(doc="|planar command| + |yaw command|")
  command_timer: Tensor = var()
  heading_target: Tensor = var()
  is_heading_env: Tensor = var(dtype=torch.bool)
  is_standing_env: Tensor = var(dtype=torch.bool)
  is_forward_env: Tensor = var(dtype=torch.bool)
  # Disturbances and bookkeeping.
  push_timer: Tensor = var()
  reward: Tensor = var()
  episode_length: Tensor = var(dtype=torch.long)
  episode_reward: Tensor = var()


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

    dev = self.device
    self.default_dof_pos = self.sim.default_dof_pos.to(dev)
    limits = self.sim.dof_limits.to(dev)
    mid, half = (limits[:, 0] + limits[:, 1]) / 2, (limits[:, 1] - limits[:, 0]) / 2
    self.soft_limits = torch.stack(
        [mid - cfg.robot.soft_limit_factor * half, mid + cfg.robot.soft_limit_factor * half],
        dim=1)
    self.is_calf = torch.tensor(
        [n.endswith("_calf_joint") for n in self.joint_names], device=dev)

    # The variables, and the startup randomisation: drawn once, and it stays.
    n, j = self.num_envs, self.num_dof
    self.state = State(n, dev, joint=self.joint_names, foot=cfg.robot.feet)
    startup = cfg.randomization.startup
    self.sim.set_friction(torch.arange(n, device=dev), _uniform(n, startup.foot_friction, dev))
    self.encoder_bias = torch.empty(n, j, device=dev).uniform_(*startup.encoder_bias)

    # The observations. The actor's are noisy (mjlab's uniform half-widths,
    # in each signal's own units); the critic's are clean and privileged.
    self.actor_obs = Obs(
        base_lin_vel=("base_lin_vel", Noise(0.5)),
        base_ang_vel=("base_ang_vel", Noise(0.2)),
        projected_gravity=("projected_gravity", Noise(0.05)),
        joint_pos=(self.joint_pos_rel, Offset(self.encoder_bias), Noise(0.01)),
        joint_vel=("joint_vel", Noise(1.5)),
        actions="actions",
        commands="commands",
    )
    self.critic_obs = Obs(
        base_lin_vel="base_lin_vel",
        base_ang_vel="base_ang_vel",
        projected_gravity="projected_gravity",
        joint_pos=self.joint_pos_rel,
        joint_vel="joint_vel",
        actions="actions",
        commands="commands",
        foot_air_time="foot_air_time",
        foot_contact="foot_contact",
        foot_force=lambda s: torch.log1p(s.foot_force),
    )
    self.num_obs = self.actor_obs(self.state).shape[-1]
    self.num_critic_obs = self.critic_obs(self.state).shape[-1]
    self.critic_obs_buf = torch.zeros(n, self.num_critic_obs, device=dev)

  def joint_pos_rel(self, s: State) -> Tensor:
    """Joint positions relative to the default pose."""
    return s.joint_pos - self.default_dof_pos

  @property
  def action_scale(self) -> Tensor:
    """Per-joint scale, read from the config every step so it is live."""
    a = self.cfg.action
    return torch.where(self.is_calf, torch.full_like(self.default_dof_pos, a.scale_calf),
                       torch.full_like(self.default_dof_pos, a.scale_hip_thigh))

  # -- Reset ----------------------------------------------------------------

  def reset(self, env_ids: Tensor | None = None) -> Tensor:
    """Start fresh episodes for ``env_ids`` (all of them when None)."""
    cfg, s = self.cfg, self.state
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

    # Every variable starts at zero, every pipe forgets the old episode.
    s.reset(env_ids)
    self.reset_blocks(env_ids)
    s.push_timer[env_ids] = _uniform(n, r.push_interval_s, dev)
    self._read_state()
    s.foot_vel[env_ids] = 0.0
    self._resample_commands(env_ids)
    return self._compute_observations()

  # -- Commands -------------------------------------------------------------

  def _resample_commands(self, env_ids: Tensor) -> None:
    """mjlab's UniformVelocityCommand: a twist per env, a target heading for
    some, zero for some, straight ahead for some."""
    c, s = self.cfg.command, self.state
    n, dev = len(env_ids), self.device
    s.commands[env_ids, 0] = _uniform(n, c.lin_vel_x, dev)
    s.commands[env_ids, 1] = _uniform(n, c.lin_vel_y, dev)
    s.commands[env_ids, 2] = _uniform(n, c.ang_vel_z, dev)
    s.heading_target[env_ids] = _uniform(n, c.heading, dev)
    s.is_heading_env[env_ids] = _uniform(n, (0.0, 1.0), dev) <= c.rel_heading_envs
    s.is_standing_env[env_ids] = _uniform(n, (0.0, 1.0), dev) <= c.rel_standing_envs
    s.is_forward_env[env_ids] = _uniform(n, (0.0, 1.0), dev) <= c.rel_forward_envs
    fwd = env_ids[s.is_forward_env[env_ids]]
    s.commands[fwd, 0] = s.commands[fwd, 0].abs().clamp(min=c.forward_min_speed)
    s.commands[fwd, 1:] = 0.0
    s.command_timer[env_ids] = _uniform(n, c.resampling_time_s, dev)
    self._update_commands()

  def _update_commands(self) -> None:
    """Every step: heading envs steer toward their target yaw; standing envs
    are held at zero."""
    c, s = self.cfg.command, self.state
    q = s.base_quat
    heading = torch.atan2(
        2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
        1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    error = wrap_to_pi(s.heading_target - heading)
    steer = torch.clip(c.heading_stiffness * error, c.ang_vel_z[0], c.ang_vel_z[1])
    s.commands[:, 2] = torch.where(s.is_heading_env, steer, s.commands[:, 2])
    s.commands[s.is_standing_env] = 0.0
    s.command_speed[:] = torch.norm(s.commands[:, :2], dim=1) + s.commands[:, 2].abs()

  # -- Step -----------------------------------------------------------------

  def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, dict]:
    """One control step: act, simulate, observe, score, terminate, reset."""
    cfg, s = self.cfg, self.state
    dev = self.device

    # 1. Actions become joint position targets around the default pose.
    s.last_actions[:] = s.actions
    s.actions[:] = actions.to(dev)
    self.sim.set_dof_targets(self.default_dof_pos + s.actions * self.action_scale)

    # 2. Disturbances are due for some envs: a kick to the base velocity.
    s.push_timer -= self.control_dt
    push_ids = (s.push_timer <= 0.0).nonzero(as_tuple=False).squeeze(-1)
    if len(push_ids) > 0:
      self._push(push_ids)

    # 3. Physics: `decimation` substeps, then the state comes back.
    self.sim.step()
    self._read_state()

    # 4. The command generator ticks.
    s.command_timer -= self.control_dt
    due = (s.command_timer <= 0.0).nonzero(as_tuple=False).squeeze(-1)
    if len(due) > 0:
      self._resample_commands(due)
    self._update_commands()

    # 5. Observe, score, terminate.
    obs = self._compute_observations()
    # Every term is multiplied by the control timestep, as mjlab's manager does.
    s.reward[:], reward_terms = self.compute_reward(scale=self.control_dt)
    time_out = s.episode_length >= self.max_episode_steps - 1
    tilt = torch.acos(torch.clamp(-s.projected_gravity[:, 2], -1.0, 1.0))
    fell = tilt > math.radians(cfg.termination.fell_over_deg)
    dones = time_out | fell

    # 6. Bookkeeping, and the resets for whoever finished.
    s.episode_length += 1
    s.episode_reward += s.reward
    done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
    info = self.step_info(
        reward_terms,
        episode_rewards=s.episode_reward[done_ids].clone(),
        episode_lengths=s.episode_length[done_ids].float().clone(),
        time_outs=time_out,
    )
    if len(done_ids) > 0:
      obs = self.reset(done_ids)
    return obs, s.reward, dones, info

  def _push(self, env_ids: Tensor) -> None:
    r = self.cfg.randomization
    n, dev = len(env_ids), self.device
    lin = torch.stack([_uniform(n, r.push_lin_vel, dev), _uniform(n, r.push_lin_vel, dev),
                       _uniform(n, r.push_lin_vel_z, dev)], dim=1)
    ang = torch.stack([_uniform(n, r.push_ang_vel_xy, dev), _uniform(n, r.push_ang_vel_xy, dev),
                       _uniform(n, r.push_ang_vel_z, dev)], dim=1)
    self.sim.push(env_ids, lin, ang)
    self.state.push_timer[env_ids] = _uniform(n, r.push_interval_s, dev)

  # -- State ----------------------------------------------------------------

  def _read_state(self) -> None:
    """Pull the backend's state into the variables everything else reads."""
    sim, s, dev = self.sim, self.state, self.device
    s.base_pos[:] = sim.root_pos.to(dev)
    s.base_quat[:] = sim.root_quat.to(dev)
    s.base_lin_vel[:] = quat_rotate_inverse(s.base_quat, sim.root_lin_vel.to(dev))
    s.base_ang_vel[:] = quat_rotate_inverse(s.base_quat, sim.root_ang_vel.to(dev))
    s.projected_gravity[:] = projected_gravity(s.base_quat)
    s.joint_pos[:] = sim.dof_pos.to(dev)
    s.joint_vel[:] = sim.dof_vel.to(dev)
    s.joint_torque[:] = sim.dof_torque.to(dev)
    # Feet: force, contact edges, air time, a finite-difference velocity, and
    # the peak height of the swing -- kept until the step after touchdown
    # so the swing-height term can read it, then cleared.
    pos = sim.contact_site_pos.to(dev)
    s.foot_vel[:] = (pos - s.foot_pos) / self.control_dt
    s.foot_pos[:] = pos
    s.foot_force[:] = sim.contact_forces.to(dev)
    s.last_foot_contact[:] = s.foot_contact
    s.foot_contact[:] = s.foot_force > 1.0
    s.first_contact[:] = s.foot_contact & ~s.last_foot_contact
    s.foot_air_time += self.control_dt
    s.foot_air_time[s.foot_contact] = 0.0
    s.foot_peak_height[s.last_foot_contact] = 0.0
    s.foot_peak_height[:] = torch.where(
        ~s.foot_contact, torch.maximum(s.foot_peak_height, pos[:, :, 2]), s.foot_peak_height)

  # -- Observations ---------------------------------------------------------

  def _compute_observations(self) -> Tensor:
    """The actor's 48 numbers through their pipes; the critic's 60, clean."""
    self.critic_obs_buf = self.critic_obs(self.state)
    return self.actor_obs(self.state)

  # -- Rewards. One method per term in the table; params are the arguments. --

  def _moving(self, threshold: float) -> Tensor:
    """1 where a command above ``threshold`` is active, else 0."""
    return (self.state.command_speed > threshold).float()

  def _joint_stds(self, hip_thigh: float, calf: float) -> Tensor:
    return torch.where(self.is_calf, torch.full_like(self.default_dof_pos, calf),
                       torch.full_like(self.default_dof_pos, hip_thigh))

  def track_linear_velocity(self, std: float) -> Tensor:
    s = self.state
    err = torch.sum(torch.square(s.commands[:, :2] - s.base_lin_vel[:, :2]), dim=1)
    err = err + torch.square(s.base_lin_vel[:, 2])
    return torch.exp(-err / std ** 2)

  def track_angular_velocity(self, std: float) -> Tensor:
    s = self.state
    err = torch.square(s.commands[:, 2] - s.base_ang_vel[:, 2])
    err = err + torch.sum(torch.square(s.base_ang_vel[:, :2]), dim=1)
    return torch.exp(-err / std ** 2)

  def upright(self, std: float) -> Tensor:
    tilt_sq = torch.sum(torch.square(self.state.projected_gravity[:, :2]), dim=1)
    return torch.exp(-tilt_sq / std ** 2)

  def pose(self, std_standing_hip_thigh: float, std_standing_calf: float,
           std_walking_hip_thigh: float, std_walking_calf: float,
           std_running_hip_thigh: float, std_running_calf: float,
           walking_threshold: float, running_threshold: float) -> Tensor:
    """How far from the default pose, with a tolerance that widens with the
    commanded speed."""
    s = self.state
    standing = (s.command_speed < walking_threshold).unsqueeze(1)
    running = (s.command_speed >= running_threshold).unsqueeze(1)
    std = torch.where(
        standing, self._joint_stds(std_standing_hip_thigh, std_standing_calf),
        torch.where(running, self._joint_stds(std_running_hip_thigh, std_running_calf),
                    self._joint_stds(std_walking_hip_thigh, std_walking_calf)))
    err = torch.square(s.joint_pos - self.default_dof_pos) / std ** 2
    return torch.exp(-torch.mean(err, dim=1))

  def dof_pos_limits(self) -> Tensor:
    s = self.state
    out_of_limits = (
        -(s.joint_pos - self.soft_limits[:, 0]).clip(max=0.0)
        + (s.joint_pos - self.soft_limits[:, 1]).clip(min=0.0))
    return torch.sum(out_of_limits, dim=1)

  def action_rate_l2(self) -> Tensor:
    s = self.state
    return torch.sum(torch.square(s.actions - s.last_actions), dim=1)

  def air_time(self, threshold_min: float, threshold_max: float,
               command_threshold: float) -> Tensor:
    s = self.state
    in_range = (s.foot_air_time > threshold_min) & (s.foot_air_time < threshold_max)
    return torch.sum(in_range.float(), dim=1) * self._moving(command_threshold)

  def foot_clearance(self, target_height: float, command_threshold: float) -> Tensor:
    s = self.state
    foot_h = s.foot_pos[:, :, 2]
    foot_v = torch.norm(s.foot_vel[:, :, :2], dim=-1)
    cost = torch.sum(torch.abs(foot_h - target_height) * foot_v, dim=1)
    return cost * self._moving(command_threshold)

  def foot_swing_height(self, target_height: float, command_threshold: float) -> Tensor:
    s = self.state
    err = torch.square(s.foot_peak_height / target_height - 1.0)
    return torch.sum(err * s.first_contact.float(), dim=1) * self._moving(command_threshold)

  def foot_slip(self, command_threshold: float) -> Tensor:
    s = self.state
    foot_v = torch.norm(s.foot_vel[:, :, :2], dim=-1)
    cost = torch.sum(torch.square(foot_v) * s.foot_contact.float(), dim=1)
    return cost * self._moving(command_threshold)

  def soft_landing(self, command_threshold: float) -> Tensor:
    s = self.state
    cost = torch.sum(s.foot_force * s.first_contact.float(), dim=1)
    return cost * self._moving(command_threshold)

  def body_ang_vel(self) -> Tensor:
    return torch.sum(torch.square(self.state.base_ang_vel[:, :2]), dim=1)

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
