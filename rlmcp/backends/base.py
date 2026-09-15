"""What a physics backend is, and the robot description every backend reads.

A single-file environment (``docs/single-file.md``) talks to its simulator
through one object, ``env.sim``, and never through the simulator's own API.
That is what lets the same ``env.py`` run on MuJoCo Warp, on mjbatch and on
Genesis: the environment reads robot state in one vocabulary -- root pose
and velocity, joint position and velocity, contact force per named site --
and writes joint position targets, whatever is integrating underneath.

The contract sits at the level of a robot, not of an ``MjData``. That is the
level the three simulators share. Below it they disagree about everything:
mjbatch and Warp expose ``qpos``; Genesis exposes links and dofs and has no
sensors at all. Above it, an environment does not care.

Two pieces here:

* :class:`RobotSpec` -- the robot as the environment declares it: the MJCF,
  which joints are actuated and in what order, the PD gains and effort limit
  per joint, and which sites report contact. MuJoCo backends compile it into
  a model with position actuators and touch sensors (:func:`compile_model`);
  Genesis sets the same gains on its own dofs and reads link contact forces.
* :class:`SimBackend` -- the interface. Everything a backend returns is a
  torch tensor on the backend's device with a leading ``num_envs`` axis;
  quaternions are ``(w, x, y, z)``; velocities are in the world frame.

Frames, so nobody guesses: ``root_lin_vel`` and ``root_ang_vel`` are world
frame. MuJoCo stores the free joint's angular velocity in the body frame, so
the MuJoCo backends rotate it; Genesis reports world frame natively. An
environment that wants body-frame velocities rotates them with
:func:`rlmcp.backends.frames.quat_rotate_inverse`, once, itself.
"""

from __future__ import annotations

import fnmatch
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

Gains = float | dict[str, float]
"""One number for every joint, or ``{pattern: number}`` matched against joint
names with :func:`fnmatch.fnmatch` -- ``{"*_calf_joint": 35.0, "*": 20.0}``.
The first matching pattern wins, so put the specific ones first."""


def gain_for(table: Gains | None, joint: str, default: float | None = None) -> float | None:
  """The entry of ``table`` that applies to ``joint``."""
  if table is None:
    return default
  if isinstance(table, (int, float)):
    return float(table)
  for pattern, value in table.items():
    if fnmatch.fnmatch(joint, pattern):
      return float(value)
  if default is not None:
    return default
  raise KeyError(
      f"No gain for joint '{joint}': none of {list(table)} matches it. Add a "
      "pattern for it, or a catch-all '*' last."
  )


@dataclass(frozen=True)
class RobotSpec:
  """The robot, as an environment declares it. Backend-neutral."""

  xml: str
  """Path to the MJCF. Genesis reads MJCF too, so one file feeds every backend."""

  joints: tuple[str, ...] = ()
  """The actuated joints, in action order. Empty means every single-dof joint
  in the model, in model order."""

  stiffness: Gains = 20.0
  """PD position gain per joint (N*m/rad)."""

  damping: Gains = 0.5
  """PD velocity gain per joint (N*m*s/rad)."""

  effort_limit: Gains | None = None
  """Torque clamp per joint (N*m); None leaves it unclamped."""

  contact_sites: tuple[str, ...] = ()
  """Names whose contact force the backend reports, in this order. Each is a
  site in the MJCF, or a body: a body gets a box site around its collision
  geoms so a touch sensor can watch the whole body (``"trunk"`` for an
  illegal-contact check)."""

  base_body: str = ""
  """The floating base. Empty means the body carrying the model's free joint."""


@dataclass
class SimOptions:
  """Physics settings. Each backend takes the ones that apply to it."""

  integrator: str = "implicitfast"
  """MuJoCo: ``euler`` or ``implicitfast``."""
  solver: str = "newton"
  """MuJoCo: ``newton``, ``cg`` or ``pgs``."""
  iterations: int = 4
  ls_iterations: int = 6
  cone: str = "pyramidal"
  """MuJoCo: ``pyramidal`` or ``elliptic``."""
  nconmax: int | None = None
  """Warp: contacts per world; None lets mujoco_warp guess."""
  njmax: int | None = None
  """Warp: constraints per world."""
  num_threads: int = 0
  """mjbatch: worker threads; 0 means every logical core."""
  ground_plane: bool = True
  """Add a flat floor when the MJCF has no plane or height field of its own.
  Robot descriptions usually ship without one (mjlab's do), and a robot with
  nothing under it falls forever."""
  extra: dict[str, Any] = field(default_factory=dict)
  """Anything backend-specific that has no field above."""


# The compiled MuJoCo model, and where everything the interface needs lives in it.


@dataclass
class ModelLayout:
  """Indices into a compiled MuJoCo model for one :class:`RobotSpec`."""

  joint_names: list[str]
  qpos_adr: np.ndarray
  """``qpos`` column per actuated joint."""
  qvel_adr: np.ndarray
  """``qvel`` column per actuated joint."""
  actuator_ids: np.ndarray
  """Actuator per actuated joint, same order."""
  limits: np.ndarray
  """``(n_dof, 2)`` joint range, radians."""
  base_body: str
  base_qpos: int
  """First ``qpos`` column of the free joint: 7 entries, pos then quat (wxyz)."""
  base_qvel: int
  """First ``qvel`` column of the free joint: 6 entries, linear then angular."""
  contact_names: list[str]
  contact_sensor_adr: np.ndarray
  """``sensordata`` column per contact site."""
  contact_bodies: list[str]
  """The body each contact site sits on -- what Genesis reads contact from."""
  ground_added: bool = False
  """Whether :func:`compile_model` added a floor the MJCF did not have."""


def _body_aabb(model: Any, body_id: int) -> tuple[np.ndarray, np.ndarray] | None:
  """Centre and half-size, in the body frame, of a body's collision geoms."""
  import mujoco

  lo = np.full(3, np.inf)
  hi = np.full(3, -np.inf)
  found = False
  for g in range(model.ngeom):
    if model.geom_bodyid[g] != body_id:
      continue
    if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
      continue
    found = True
    centre, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[g])
    rot = rot.reshape(3, 3)
    c = model.geom_pos[g] + rot @ centre
    h = np.abs(rot) @ half
    lo = np.minimum(lo, c - h)
    hi = np.maximum(hi, c + h)
  if not found:
    return None
  return (lo + hi) / 2, (hi - lo) / 2


def _has_ground(model: Any) -> bool:
  """Whether anything in the model is a floor: a plane or a height field."""
  import mujoco

  kinds = (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD))
  return any(int(model.geom_type[g]) in kinds for g in range(model.ngeom))


def compile_model(spec: RobotSpec, options: SimOptions | None = None) -> tuple[Any, ModelLayout]:
  """The MJCF plus what the spec asks for, compiled, and where things are.

  Position actuators are added for the actuated joints (``kp`` as the gain,
  ``-kp`` and ``-kd`` as the biases, which is how MuJoCo spells a PD
  controller) and a touch sensor for every contact site. A contact name that
  is a body rather than a site gets a box site the size of the body's
  collision geoms first. Nothing already in the file is removed.
  """
  import mujoco

  options = options or SimOptions()
  plain = mujoco.MjModel.from_xml_path(spec.xml)
  mjspec = mujoco.MjSpec.from_file(spec.xml)

  # Which joints.
  single_dof = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
  joints = list(spec.joints)
  if not joints:
    joints = [
        plain.joint(j).name for j in range(plain.njnt)
        if int(plain.jnt_type[j]) in single_dof
    ]
  for name in joints:
    if mujoco.mj_name2id(plain, mujoco.mjtObj.mjOBJ_JOINT, name) < 0:
      raise KeyError(
          f"RobotSpec names joint '{name}', which {spec.xml} does not have. Joints: "
          f"{[plain.joint(j).name for j in range(plain.njnt)]}"
      )

  # Actuators, unless the file already drives that joint.
  existing = {plain.actuator(a).name for a in range(plain.nu)}
  driven = {
      plain.joint(plain.actuator_trnid[a, 0]).name for a in range(plain.nu)
      if int(plain.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
  }
  actuator_of: dict[str, str] = {}
  for name in joints:
    if name in driven:
      for a in range(plain.nu):
        if (int(plain.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
            and plain.joint(plain.actuator_trnid[a, 0]).name == name):
          actuator_of[name] = plain.actuator(a).name
      continue
    kp = gain_for(spec.stiffness, name)
    kd = gain_for(spec.damping, name)
    limit = gain_for(spec.effort_limit, name, default=0.0) if spec.effort_limit is not None else 0.0
    act_name = f"{name}_act" if name in existing else name
    kwargs: dict[str, Any] = {
        "name": act_name,
        "target": name,
        "trntype": mujoco.mjtTrn.mjTRN_JOINT,
        "gaintype": mujoco.mjtGain.mjGAIN_FIXED,
        "biastype": mujoco.mjtBias.mjBIAS_AFFINE,
        "gainprm": [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "biasprm": [0, -kp, -kd, 0, 0, 0, 0, 0, 0, 0],
    }
    if limit:
      kwargs.update(forcelimited=True, forcerange=[-limit, limit])
    mjspec.add_actuator(**kwargs)
    actuator_of[name] = act_name

  # Contact sites and their touch sensors.
  site_names = {s.name for s in mjspec.sites}
  sensor_of: dict[str, str] = {}
  touch_on = {
      plain.site(plain.sensor_objid[s]).name: plain.sensor(s).name
      for s in range(plain.nsensor)
      if int(plain.sensor_type[s]) == int(mujoco.mjtSensor.mjSENS_TOUCH)
  }
  contact_bodies: list[str] = []
  for name in spec.contact_sites:
    site = name
    if name not in site_names:
      body_id = mujoco.mj_name2id(plain, mujoco.mjtObj.mjOBJ_BODY, name)
      if body_id < 0:
        raise KeyError(
            f"RobotSpec contact site '{name}' is neither a site nor a body in "
            f"{spec.xml}. Sites: {sorted(site_names)}"
        )
      box = _body_aabb(plain, body_id)
      if box is None:
        raise ValueError(f"Body '{name}' has no collision geoms to watch for contact.")
      centre, half = box
      site = f"{name}_contact"
      mjspec.body(name).add_site(
          name=site, type=mujoco.mjtGeom.mjGEOM_BOX,
          pos=[float(v) for v in centre], size=[float(v) for v in half], group=5,
      )
      site_names.add(site)
    if site in touch_on:
      sensor_of[name] = touch_on[site]
    else:
      sensor = f"{site}_touch"
      mjspec.add_sensor(
          name=sensor, type=mujoco.mjtSensor.mjSENS_TOUCH,
          objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site,
      )
      sensor_of[name] = sensor
    site_id = mujoco.mj_name2id(plain, mujoco.mjtObj.mjOBJ_SITE, site)
    if site_id >= 0:
      body_id = int(plain.site_bodyid[site_id])
    else:
      body_id = mujoco.mj_name2id(plain, mujoco.mjtObj.mjOBJ_BODY, name)
    contact_bodies.append(plain.body(body_id).name)

  ground_added = bool(options.ground_plane and not _has_ground(plain))
  if ground_added:
    mjspec.worldbody.add_geom(
        name="rlmcp_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.05],
        contype=1, conaffinity=1, rgba=[0.5, 0.5, 0.55, 1.0],
    )

  model = mjspec.compile()
  apply_options(model, options)

  # The free joint.
  free = [j for j in range(model.njnt)
          if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
  if not free:
    raise ValueError(f"{spec.xml} has no free joint; a floating base is required.")
  free_joint = free[0]
  base_body = spec.base_body or model.body(model.jnt_bodyid[free_joint]).name

  qpos_adr = np.array([model.jnt_qposadr[model.joint(n).id] for n in joints], dtype=np.int64)
  qvel_adr = np.array([model.jnt_dofadr[model.joint(n).id] for n in joints], dtype=np.int64)
  layout = ModelLayout(
      joint_names=joints,
      qpos_adr=qpos_adr,
      qvel_adr=qvel_adr,
      actuator_ids=np.array([model.actuator(actuator_of[n]).id for n in joints], dtype=np.int64),
      limits=np.array([model.jnt_range[model.joint(n).id] for n in joints], dtype=np.float64),
      base_body=base_body,
      base_qpos=int(model.jnt_qposadr[free_joint]),
      base_qvel=int(model.jnt_dofadr[free_joint]),
      contact_names=list(spec.contact_sites),
      contact_sensor_adr=np.array(
          [model.sensor_adr[model.sensor(sensor_of[n]).id] for n in spec.contact_sites],
          dtype=np.int64,
      ),
      contact_bodies=contact_bodies,
      ground_added=ground_added,
  )
  return model, layout


def apply_options(model: Any, options: SimOptions) -> None:
  """Write the MuJoCo-side settings of ``options`` onto ``model.opt``.

  The timestep is not here: it is the backend's ``dt`` argument, since the
  environment owns it and every backend must agree on it.
  """
  import mujoco

  integrators = {
      "euler": mujoco.mjtIntegrator.mjINT_EULER,
      "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
      "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
      "rk4": mujoco.mjtIntegrator.mjINT_RK4,
  }
  solvers = {
      "newton": mujoco.mjtSolver.mjSOL_NEWTON,
      "cg": mujoco.mjtSolver.mjSOL_CG,
      "pgs": mujoco.mjtSolver.mjSOL_PGS,
  }
  cones = {
      "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
      "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
  }
  model.opt.integrator = integrators[options.integrator]
  model.opt.solver = solvers[options.solver]
  model.opt.cone = cones[options.cone]
  model.opt.iterations = int(options.iterations)
  model.opt.ls_iterations = int(options.ls_iterations)


class SimBackend(ABC):
  """One batch of robots in one simulator, behind one vocabulary.

  Construct with the robot, the batch size and the timing; then per control
  step: :meth:`set_dof_targets`, :meth:`step`, read the state properties.
  State properties are current after :meth:`step` and :meth:`reset` return.
  """

  name: str = ""

  def __init__(
      self,
      robot: RobotSpec,
      num_envs: int,
      dt: float,
      decimation: int,
      device: str | torch.device = "cuda",
      options: SimOptions | None = None,
  ):
    self.robot = robot
    self.num_envs = int(num_envs)
    self.dt = float(dt)
    """Physics timestep, seconds."""
    self.decimation = int(decimation)
    """Physics steps per control step."""
    self.device = torch.device(device)
    self.options = options or SimOptions()

  @property
  def control_dt(self) -> float:
    return self.dt * self.decimation

  # What the robot is.

  @property
  @abstractmethod
  def joint_names(self) -> list[str]:
    """Actuated joints, in the order every dof tensor uses."""

  @property
  def n_dof(self) -> int:
    return len(self.joint_names)

  @property
  @abstractmethod
  def dof_limits(self) -> Tensor:
    """``(n_dof, 2)`` joint range, radians, on :attr:`device`."""

  @property
  def contact_names(self) -> list[str]:
    return list(self.robot.contact_sites)

  @property
  def mj_model(self) -> Any | None:
    """The MuJoCo model, when this backend has one. Rendering and the live
    view read it; an environment should not."""
    return None

  # State, all ``(num_envs, ...)`` on :attr:`device`.

  @property
  @abstractmethod
  def root_pos(self) -> Tensor:
    """Base position in the world, ``(N, 3)``."""

  @property
  @abstractmethod
  def root_quat(self) -> Tensor:
    """Base orientation ``(w, x, y, z)``, ``(N, 4)``."""

  @property
  @abstractmethod
  def root_lin_vel(self) -> Tensor:
    """Base linear velocity, world frame, ``(N, 3)``."""

  @property
  @abstractmethod
  def root_ang_vel(self) -> Tensor:
    """Base angular velocity, world frame, ``(N, 3)``."""

  @property
  @abstractmethod
  def dof_pos(self) -> Tensor:
    """Joint positions ``(N, n_dof)``."""

  @property
  @abstractmethod
  def dof_vel(self) -> Tensor:
    """Joint velocities ``(N, n_dof)``."""

  @property
  @abstractmethod
  def dof_torque(self) -> Tensor:
    """Actuator torque applied at each joint ``(N, n_dof)``."""

  @property
  @abstractmethod
  def contact_forces(self) -> Tensor:
    """Normal contact force at each contact site ``(N, n_sites)``, newtons."""

  # Control.

  @abstractmethod
  def set_dof_targets(self, targets: Tensor) -> None:
    """Joint position targets ``(N, n_dof)`` for the PD controller."""

  @abstractmethod
  def step(self) -> None:
    """Advance one control step: ``decimation`` physics steps."""

  @abstractmethod
  def reset(
      self,
      env_ids: Tensor,
      root_pos: Tensor,
      root_quat: Tensor,
      dof_pos: Tensor,
      dof_vel: Tensor | None = None,
      root_lin_vel: Tensor | None = None,
      root_ang_vel: Tensor | None = None,
  ) -> None:
    """Restart ``env_ids`` at the given state. Velocities default to zero.

    Every tensor is indexed like ``env_ids``: ``root_pos`` is
    ``(len(env_ids), 3)`` and so on.
    """

  # Optional.

  def set_friction(self, env_ids: Tensor, coefficient: Tensor) -> None:
    """Sliding friction of every geom in ``env_ids``, one value per env."""
    raise NotImplementedError(f"{self.name} does not randomise friction.")

  def render(self, env_id: int = 0, width: int = 640, height: int = 480) -> np.ndarray:
    """An RGB frame of one environment, ``(height, width, 3)`` uint8."""
    raise NotImplementedError(f"{self.name} cannot render.")

  def close(self) -> None:  # noqa: B027 - releasing nothing is a valid answer.
    """Release simulator resources. Safe to call twice."""

  # Helpers for subclasses.

  def _ids(self, env_ids: Tensor | Sequence[int] | None) -> np.ndarray:
    if env_ids is None:
      return np.arange(self.num_envs)
    if torch.is_tensor(env_ids):
      return env_ids.detach().cpu().numpy().astype(np.int64).reshape(-1)
    return np.asarray(list(env_ids), dtype=np.int64).reshape(-1)


__all__ = [
    "Gains",
    "ModelLayout",
    "RobotSpec",
    "SimBackend",
    "SimOptions",
    "apply_options",
    "compile_model",
    "gain_for",
]
