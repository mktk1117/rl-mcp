"""What a physics backend is, and the robot description every backend reads.

A single-file environment (``docs/single-file.md``) talks to its simulator
through one object, ``env.sim``, and never through the simulator's own API.
That is what lets the same ``env.py`` run on MuJoCo Warp, on mjbatch and on
Genesis: the environment reads robot state in one vocabulary -- root pose
and velocity when there is a floating base, joint position and velocity,
contact force and position per named contact -- and writes joint position
targets, whatever is integrating underneath.

The contract sits at the level of an articulated robot, not of an
``MjData``, because that is the level the simulators share, and it knows
nothing about legs: a quadruped, an arm bolted to a table and a hand are
the same thing here -- joints, optional floating base, contacts.

Two pieces:

* :class:`RobotSpec` -- the robot as the environment declares it. Only the
  MJCF is required. Everything else is found in the file when it is there
  (actuated joints, PD gains from position actuators, the default pose from
  a keyframe, the floating base from the free joint, contacts as the leaf
  bodies that can collide) and declared only when it is not or when the
  task wants otherwise. What was found and what was declared is reported
  once, at construction, by :meth:`ModelLayout.describe`.
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
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

Gains = float | dict[str, float]
"""One number for every joint, or ``{pattern: number}`` matched against joint
names with :func:`fnmatch.fnmatch` -- ``{"*_knee": 35.0, "*": 20.0}``.
The first matching pattern wins, so put the specific ones first."""

POSE_KEYFRAMES = ("home", "init", "default", "standing", "stand", "rest")
"""Keyframe names that mean "the default pose", tried in this order when the
spec names none. Failing those, the first keyframe; failing that, ``qpos0``."""


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
      f"No entry for joint '{joint}': none of {list(table)} matches it. Add a "
      "pattern for it, or a catch-all '*' last."
  )


@dataclass(frozen=True)
class RobotSpec:
  """The robot, as an environment declares it. Backend-neutral.

  Only ``xml`` is required. Every other field has an automatic answer read
  off the file; give a field to override that answer, and read
  :meth:`ModelLayout.describe` (printed when a backend is built) to see
  which answers were found and which were declared.
  """

  xml: str
  """Path to the MJCF. Genesis reads MJCF too, so one file feeds every backend."""

  joints: tuple[str, ...] = ()
  """The actuated joints, in action order. Empty means every single-dof joint
  in the model, in model order."""

  stiffness: Gains | None = None
  """PD position gain per joint. None reads it from the file's own position
  actuators; a joint with neither is an error that says so."""

  damping: Gains | None = None
  """PD velocity gain per joint. None reads it from the file's actuators, or
  0 for a joint the spec gave a stiffness and no damping."""

  effort_limit: Gains | None = None
  """Torque clamp per joint. None reads the actuator's force range when the
  file has one, else leaves the joint unclamped."""

  default_joint_pos: Gains | None = None
  """The pose ``default_dof_pos`` reports and resets return to. None reads
  the keyframe named by ``keyframe`` (or the first keyframe that looks like
  one, see :data:`POSE_KEYFRAMES`), else the model's ``qpos0``."""

  keyframe: str | None = None
  """Which keyframe holds the default pose, when the file has several."""

  contacts: tuple[str, ...] = ()
  """Names whose contact force and position the backend reports, in this
  order. Each is a site or a body in the MJCF; a body gets a box site
  around its collision geoms so a touch sensor watches the whole body.
  Empty means every leaf body of the kinematic tree that can collide -- the
  feet of a legged robot, the fingertips of an arm -- which is what an
  end-effector usually is."""

  contact_geoms: tuple[str, ...] | None = None
  """Collision geoms that :meth:`SimBackend.set_friction` writes and that
  ``contact_friction``/``contact_condim``/``contact_priority`` apply to.
  None means every collision geom on the contact bodies; ``()`` means every
  collision geom of the robot."""

  contact_friction: tuple[float, float, float] | None = None
  """Sliding, torsional and rolling friction of the contact geoms; None
  keeps what the file says."""

  contact_condim: int | None = None
  """Contact dimensionality of the contact geoms; None keeps the file's."""

  contact_priority: int | None = None
  """MuJoCo contact priority of the contact geoms: 1 makes their friction
  win against the floor's. None keeps the file's."""

  geom_solref: tuple[float, float] | None = None
  """Contact ``solref`` for every collision geom of the robot; None keeps
  the file's."""

  base_body: str = ""
  """The floating base. Empty means the body carrying the model's free
  joint, and a model without one is a fixed-base robot: no root state, no
  pushes, resets take joints only."""


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
  impratio: float = 1.0
  """MuJoCo: frictional-to-normal constraint impedance ratio."""
  ccd_iterations: int | None = None
  """MuJoCo: convex collision iterations; None keeps the model's own."""
  nconmax: int | None = None
  """Warp: contacts per world; None lets mujoco_warp guess."""
  njmax: int | None = None
  """Warp: constraints per world."""
  num_threads: int = 0
  """mjbatch: worker threads; 0 means every logical core."""
  ground_plane: bool = True
  """Add a flat floor when the MJCF has no plane or height field of its own.
  Robot descriptions usually ship without one, and a floating robot with
  nothing under it falls forever. A directional light comes with it when the
  file has none, so frames are not a silhouette on a grey floor."""
  extra: dict[str, Any] = field(default_factory=dict)
  """Anything backend-specific that has no field above."""


# The compiled MuJoCo model, and where everything the interface needs lives in it.


@dataclass
class ModelLayout:
  """Indices into a compiled MuJoCo model for one :class:`RobotSpec`, and a
  record of where each answer came from."""

  joint_names: list[str]
  qpos_adr: np.ndarray
  """``qpos`` column per actuated joint."""
  qvel_adr: np.ndarray
  """``qvel`` column per actuated joint."""
  actuator_ids: np.ndarray
  """Actuator per actuated joint, same order."""
  limits: np.ndarray
  """``(n_dof, 2)`` joint range."""
  default_dof_pos: np.ndarray
  """``(n_dof,)`` the default pose."""
  floating: bool
  """Whether the model has a free joint. When it does not, the ``base_*``
  fields below are unset (-1, empty)."""
  base_body: str
  base_qpos: int
  """First ``qpos`` column of the free joint: 7 entries, pos then quat (wxyz)."""
  base_qvel: int
  """First ``qvel`` column of the free joint: 6 entries, linear then angular."""
  contact_names: list[str]
  contact_sensor_adr: np.ndarray
  """``sensordata`` column per contact."""
  contact_bodies: list[str]
  """The body each contact sits on -- what Genesis reads contact from."""
  contact_site_ids: np.ndarray
  """Site id per contact in the compiled model."""
  contact_geom_ids: np.ndarray
  """Geom ids :meth:`SimBackend.set_friction` writes; empty means all geoms."""
  ground_added: bool
  """Whether :func:`compile_model` added a floor the MJCF did not have."""
  sources: dict[str, str] = field(default_factory=dict)
  """Where each answer came from, by topic: ``joints``, ``gains``,
  ``effort_limit``, ``default_pose``, ``base``, ``contacts``, ``contact_geoms``."""

  def describe(self) -> str:
    """One paragraph a person can check: what was found, what was declared."""
    s = self.sources
    base = f"floating base '{self.base_body}'" if self.floating else "fixed base"
    contacts = ", ".join(self.contact_names) if self.contact_names else "none"
    return (
        f"{len(self.joint_names)} joints ({s.get('joints', '?')}); "
        f"gains {s.get('gains', '?')}; effort limit {s.get('effort_limit', '?')}; "
        f"default pose {s.get('default_pose', '?')}; {base} ({s.get('base', '?')}); "
        f"contacts {contacts} ({s.get('contacts', '?')}); "
        f"{len(self.contact_geom_ids) or 'all'} contact geoms ({s.get('contact_geoms', '?')})"
        + ("; floor added" if self.ground_added else "")
    )


def _body_aabb(model: Any, body_id: int) -> tuple[np.ndarray, np.ndarray] | None:
  """Centre and half-size, in the body frame, of a body's collision geoms."""
  import mujoco

  lo = np.full(3, np.inf)
  hi = np.full(3, -np.inf)
  found = False
  for g in _collision_geoms(model, body_id):
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


def _collision_geoms(model: Any, body_id: int) -> list[int]:
  return [
      g for g in range(model.ngeom)
      if model.geom_bodyid[g] == body_id
      and (model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0)
  ]


def _leaf_bodies(model: Any) -> list[str]:
  """Bodies with no children and at least one collision geom, in model order."""
  parents = {int(p) for p in model.body_parentid[1:]}
  return [
      model.body(b).name for b in range(1, model.nbody)
      if b not in parents and _collision_geoms(model, b)
  ]


def _has_ground(model: Any) -> bool:
  """Whether anything in the model is a floor: a plane or a height field."""
  import mujoco

  kinds = (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD))
  return any(int(model.geom_type[g]) in kinds for g in range(model.ngeom))


def _pose_keyframe(model: Any, spec: RobotSpec) -> int | None:
  """The keyframe holding the default pose, or None when there is none."""
  import mujoco

  if spec.keyframe is not None:
    k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, spec.keyframe)
    if k < 0:
      raise KeyError(
          f"RobotSpec names keyframe '{spec.keyframe}', which {spec.xml} does not "
          f"have. Keyframes: {[model.key(i).name for i in range(model.nkey)]}"
      )
    return k
  if model.nkey == 0:
    return None
  names = [model.key(i).name for i in range(model.nkey)]
  for wanted in POSE_KEYFRAMES:
    if wanted in names:
      return names.index(wanted)
  return 0


def _joint_actuator(model: Any, name: str) -> int | None:
  """The actuator driving joint ``name`` directly, if the file has one."""
  import mujoco

  for a in range(model.nu):
    if (int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        and model.joint(model.actuator_trnid[a, 0]).name == name):
      return a
  return None


def compile_model(spec: RobotSpec, options: SimOptions | None = None) -> tuple[Any, ModelLayout]:
  """The MJCF plus what the spec asks for, compiled, and where things are.

  What the file lacks is added: a position actuator per actuated joint
  (``kp`` as the gain, ``-kp`` and ``-kd`` as the biases, which is how MuJoCo
  spells a PD controller), a touch sensor per contact, a box site around a
  body named as a contact, a floor. What the file has is kept and read:
  its actuators' gains, its keyframe's pose, its free joint. Every answer
  is recorded in ``layout.sources``.
  """
  import mujoco

  options = options or SimOptions()
  plain = mujoco.MjModel.from_xml_path(spec.xml)
  mjspec = mujoco.MjSpec.from_file(spec.xml)
  sources: dict[str, str] = {}
  all_joints = [plain.joint(j).name for j in range(plain.njnt)]

  # Which joints.
  single_dof = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
  joints = list(spec.joints)
  if joints:
    sources["joints"] = "from spec"
  else:
    joints = [n for j, n in enumerate(all_joints) if int(plain.jnt_type[j]) in single_dof]
    sources["joints"] = "every single-dof joint in the file"
  for name in joints:
    if name not in all_joints:
      raise KeyError(
          f"RobotSpec names joint '{name}', which {spec.xml} does not have. "
          f"Joints: {all_joints}"
      )
  if not joints:
    raise ValueError(f"{spec.xml} has no single-dof joint to actuate.")

  # Actuators: the file's own where it drives the joint, else one per the spec.
  existing = {plain.actuator(a).name for a in range(plain.nu)}
  actuator_of: dict[str, str] = {}
  from_file: list[str] = []
  from_spec: list[str] = []
  for name in joints:
    a = _joint_actuator(plain, name)
    if a is not None and spec.stiffness is None:
      actuator_of[name] = plain.actuator(a).name
      from_file.append(name)
      continue
    if spec.stiffness is None:
      raise ValueError(
          f"Joint '{name}' has no actuator in {spec.xml} and the spec gives no "
          "stiffness. Give RobotSpec(stiffness=..., damping=...) -- a number, or "
          "{pattern: number} by joint name -- or add position actuators to the file."
      )
    kp = gain_for(spec.stiffness, name)
    kd = gain_for(spec.damping, name, default=0.0)
    limit = gain_for(spec.effort_limit, name, default=0.0)
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
    from_spec.append(name)
  if from_file and from_spec:
    sources["gains"] = f"{len(from_file)} from the file's actuators, {len(from_spec)} from spec"
  elif from_file:
    sources["gains"] = "from the file's actuators"
  else:
    sources["gains"] = "from spec"
  if spec.effort_limit is not None:
    sources["effort_limit"] = "from spec"
  elif from_file:
    sources["effort_limit"] = "the file's actuator force ranges"
  else:
    sources["effort_limit"] = "none (unclamped)"

  # Contacts: sites or bodies named by the spec, else the leaf bodies.
  site_names = {s.name for s in mjspec.sites}
  body_names = [plain.body(b).name for b in range(plain.nbody)]
  contacts = list(spec.contacts)
  if contacts:
    sources["contacts"] = "from spec"
  else:
    contacts = _leaf_bodies(plain)
    sources["contacts"] = "leaf bodies that can collide"
  touch_on = {
      plain.site(plain.sensor_objid[s]).name: plain.sensor(s).name
      for s in range(plain.nsensor)
      if int(plain.sensor_type[s]) == int(mujoco.mjtSensor.mjSENS_TOUCH)
  }
  sensor_of: dict[str, str] = {}
  site_of: dict[str, str] = {}
  contact_bodies: list[str] = []
  for name in contacts:
    site = name
    if name not in site_names:
      if name not in body_names:
        raise KeyError(
            f"RobotSpec contact '{name}' is neither a site nor a body in {spec.xml}. "
            f"Sites: {sorted(site_names)}; bodies: {body_names[1:]}"
        )
      box = _body_aabb(plain, body_names.index(name))
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
    site_of[name] = site
    site_id = mujoco.mj_name2id(plain, mujoco.mjtObj.mjOBJ_SITE, site)
    body_id = int(plain.site_bodyid[site_id]) if site_id >= 0 else body_names.index(name)
    contact_bodies.append(body_names[body_id])

  # Contact geoms, and the contact tuning on them.
  geom_names = [g.name for g in mjspec.geoms]
  if spec.contact_geoms is None:
    contact_geoms = [
        plain.geom(g).name
        for body in dict.fromkeys(contact_bodies)
        for g in _collision_geoms(plain, body_names.index(body))
        if plain.geom(g).name
    ]
    sources["contact_geoms"] = "collision geoms of the contact bodies"
  else:
    contact_geoms = list(spec.contact_geoms)
    sources["contact_geoms"] = "from spec" if contact_geoms else "all geoms"
  for name in contact_geoms:
    if name not in geom_names:
      raise KeyError(
          f"RobotSpec contact geom '{name}' is not in {spec.xml}. Geoms: {sorted(geom_names)}"
      )
  for g in mjspec.geoms:
    if g.contype == 0 and g.conaffinity == 0:
      continue
    if spec.geom_solref is not None:
      g.solref[0], g.solref[1] = float(spec.geom_solref[0]), float(spec.geom_solref[1])
    if g.name in contact_geoms or not contact_geoms:
      if spec.contact_priority is not None:
        g.priority = int(spec.contact_priority)
      if spec.contact_condim is not None:
        g.condim = int(spec.contact_condim)
      if spec.contact_friction is not None:
        g.friction[0], g.friction[1], g.friction[2] = (float(v) for v in spec.contact_friction)

  # A floor, and light to see it by.
  ground_added = bool(options.ground_plane and not _has_ground(plain))
  if ground_added:
    mjspec.worldbody.add_geom(
        name="rlmcp_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.05],
        contype=1, conaffinity=1, rgba=[0.5, 0.5, 0.55, 1.0],
    )
  if options.ground_plane and plain.nlight == 0:
    light = mjspec.worldbody.add_light(pos=[0.0, 0.0, 4.0], dir=[0.0, 0.0, -1.0])
    try:
      light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    except AttributeError:  # MuJoCo < 3.3 spells it as a flag.
      light.directional = True

  model = mjspec.compile()
  apply_options(model, options)

  # The base: the body carrying a free joint, or none.
  free = [j for j in range(model.njnt)
          if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
  if free:
    free_joint = free[0]
    base_body = spec.base_body or model.body(model.jnt_bodyid[free_joint]).name
    base_qpos, base_qvel = int(model.jnt_qposadr[free_joint]), int(model.jnt_dofadr[free_joint])
    sources["base"] = "from spec" if spec.base_body else "the free joint's body"
  else:
    if spec.base_body:
      raise ValueError(
          f"RobotSpec names base body '{spec.base_body}' but {spec.xml} has no free "
          "joint; a fixed-base robot has no floating base."
      )
    base_body, base_qpos, base_qvel = "", -1, -1
    sources["base"] = "no free joint in the file"

  qpos_adr = np.array([model.jnt_qposadr[model.joint(n).id] for n in joints], dtype=np.int64)
  qvel_adr = np.array([model.jnt_dofadr[model.joint(n).id] for n in joints], dtype=np.int64)

  # The default pose.
  if spec.default_joint_pos is not None:
    default = np.array([gain_for(spec.default_joint_pos, n) for n in joints], dtype=np.float64)
    sources["default_pose"] = "from spec"
  else:
    key = _pose_keyframe(model, spec)
    if key is not None:
      default = np.array(model.key_qpos[key][qpos_adr], dtype=np.float64)
      sources["default_pose"] = f"keyframe '{model.key(key).name}'"
    else:
      default = np.array(model.qpos0[qpos_adr], dtype=np.float64)
      sources["default_pose"] = "qpos0 (no keyframe in the file)"

  layout = ModelLayout(
      joint_names=joints,
      qpos_adr=qpos_adr,
      qvel_adr=qvel_adr,
      actuator_ids=np.array([model.actuator(actuator_of[n]).id for n in joints], dtype=np.int64),
      limits=np.array([model.jnt_range[model.joint(n).id] for n in joints], dtype=np.float64),
      default_dof_pos=default,
      floating=bool(free),
      base_body=base_body,
      base_qpos=base_qpos,
      base_qvel=base_qvel,
      contact_names=contacts,
      contact_sensor_adr=np.array(
          [model.sensor_adr[model.sensor(sensor_of[n]).id] for n in contacts], dtype=np.int64),
      contact_bodies=contact_bodies,
      contact_site_ids=np.array([model.site(site_of[n]).id for n in contacts], dtype=np.int64),
      contact_geom_ids=np.array([model.geom(n).id for n in contact_geoms], dtype=np.int64),
      ground_added=ground_added,
      sources=sources,
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
  model.opt.impratio = float(options.impratio)
  if options.ccd_iterations is not None:
    model.opt.ccd_iterations = int(options.ccd_iterations)


class FixedBase(RuntimeError):
  """Raised when a root-state operation is asked of a fixed-base robot."""


class SimBackend(ABC):
  """One batch of robots in one simulator, behind one vocabulary.

  Construct with the robot, the batch size and the timing; then per control
  step: :meth:`set_dof_targets`, :meth:`step`, read the state properties.
  State properties are current after :meth:`step` and :meth:`reset` return.
  The ``root_*`` properties, :meth:`push` and the root arguments of
  :meth:`reset` exist for a floating base; on a fixed-base robot the
  properties raise :class:`FixedBase` and the arguments are ignored.
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

  def announce(self, layout: ModelLayout) -> None:
    """Say once what was found in the file and what the spec declared."""
    print(f"[{self.name}] {Path(self.robot.xml).name}: {layout.describe()}", flush=True)

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
    """``(n_dof, 2)`` joint range, on :attr:`device`."""

  @property
  @abstractmethod
  def default_dof_pos(self) -> Tensor:
    """``(n_dof,)`` the default pose: the spec's, a keyframe's, or ``qpos0``."""

  @property
  @abstractmethod
  def floating_base(self) -> bool:
    """Whether the robot has a free joint. False for an arm on a table."""

  @property
  @abstractmethod
  def contact_names(self) -> list[str]:
    """The contacts, in the order :attr:`contact_forces` uses."""

  @property
  def mj_model(self) -> Any | None:
    """The MuJoCo model, when this backend has one. Rendering and the live
    view read it; an environment should not."""
    return None

  # State, all ``(num_envs, ...)`` on :attr:`device`.

  @property
  @abstractmethod
  def root_pos(self) -> Tensor:
    """Base position in the world, ``(N, 3)``. Floating base only."""

  @property
  @abstractmethod
  def root_quat(self) -> Tensor:
    """Base orientation ``(w, x, y, z)``, ``(N, 4)``. Floating base only."""

  @property
  @abstractmethod
  def root_lin_vel(self) -> Tensor:
    """Base linear velocity, world frame, ``(N, 3)``. Floating base only."""

  @property
  @abstractmethod
  def root_ang_vel(self) -> Tensor:
    """Base angular velocity, world frame, ``(N, 3)``. Floating base only."""

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
    """Normal contact force at each contact ``(N, n_contacts)``, newtons."""

  @property
  @abstractmethod
  def contact_site_pos(self) -> Tensor:
    """World position of each contact ``(N, n_contacts, 3)``. Height above a
    flat floor is its z; a finite difference over a control step is the
    contact's velocity."""

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
      dof_pos: Tensor,
      dof_vel: Tensor | None = None,
      root_pos: Tensor | None = None,
      root_quat: Tensor | None = None,
      root_lin_vel: Tensor | None = None,
      root_ang_vel: Tensor | None = None,
  ) -> None:
    """Restart ``env_ids`` at the given state. Velocities default to zero,
    the root pose to where the model spawns. Every tensor is indexed like
    ``env_ids``: ``dof_pos`` is ``(len(env_ids), n_dof)`` and so on. The
    root arguments are ignored on a fixed base."""

  @abstractmethod
  def push(self, env_ids: Tensor, lin_vel: Tensor, ang_vel: Tensor) -> None:
    """Add ``lin_vel`` and ``ang_vel`` (world frame, per env) to the base
    velocity of ``env_ids``: the instantaneous, mass-free kick used as a
    disturbance. Raises :class:`FixedBase` without a free joint."""

  # Optional.

  def set_friction(self, env_ids: Tensor, coefficient: Tensor) -> None:
    """Sliding friction of the contact geoms in ``env_ids``, one value per env."""
    raise NotImplementedError(f"{self.name} does not randomise friction.")

  def render(self, env_id: int = 0, width: int = 640, height: int = 480) -> np.ndarray:
    """An RGB frame of one environment, ``(height, width, 3)`` uint8."""
    raise NotImplementedError(f"{self.name} cannot render.")

  def close(self) -> None:  # noqa: B027 - releasing nothing is a valid answer.
    """Release simulator resources. Safe to call twice."""

  # Helpers for subclasses.

  def _require_floating(self, what: str) -> None:
    if not self.floating_base:
      raise FixedBase(
          f"{what} needs a floating base, and {Path(self.robot.xml).name} has no "
          "free joint. A fixed-base robot has joints and contacts only."
      )

  def _ids(self, env_ids: Tensor | Sequence[int] | None) -> np.ndarray:
    if env_ids is None:
      return np.arange(self.num_envs)
    if torch.is_tensor(env_ids):
      return env_ids.detach().cpu().numpy().astype(np.int64).reshape(-1)
    return np.asarray(list(env_ids), dtype=np.int64).reshape(-1)


__all__ = [
    "POSE_KEYFRAMES",
    "FixedBase",
    "Gains",
    "ModelLayout",
    "RobotSpec",
    "SimBackend",
    "SimOptions",
    "apply_options",
    "compile_model",
    "gain_for",
]
