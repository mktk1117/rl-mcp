"""Playing a policy bundle in plain MuJoCo: the reference every other player follows.

A bundle (:mod:`rlmcp.bundle`) is a policy and everything needed to run it
without the training stack: ``policy.onnx``, ``model.mjb`` (the compiled
MuJoCo model, meshes included), and ``spec.json`` -- which observations the
policy reads, in which order, with what scale, clip and history; how its
outputs become actuator targets; how often it acts. This module turns those
three files into a robot moving, using nothing but ``mujoco`` and ``numpy``
(and ``onnxruntime`` when the policy is an ONNX file rather than a callable).

It is deliberately small and deliberately literal. A browser player -- MuJoCo
compiled to WebAssembly, ONNX Runtime Web -- is a port of this file, term by
term, and :func:`rlmcp.bundle.export_bundle` proves at export time that these
term functions reproduce the training environment's observations to floating
point precision. So the set of terms here *is* the set a bundle may use: a
policy whose environment reads something this module cannot compute is refused
at export with the term's name, never published and then found to play wrong.

Each term function takes the player and the term's entry from the spec and
returns a flat float array. The pipeline per term is the training stack's:
compute → clip → scale → history (oldest to newest, flattened). Noise is a
training aid and is not applied.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SPEC_VERSION = 1
SPEC_FILE = "spec.json"
MODEL_FILE = "model.mjb"
POLICY_FILE = "policy.onnx"
MANIFEST_FILE = "manifest.json"


class BundleError(RuntimeError):
  """A bundle that cannot be played, with the reason."""


# ── values the spec carries ─────────────────────────────────────────────


def decode_value(value: Any) -> Any:
  """Undo :func:`rlmcp.adapters.manager_based.term_capture.encode_value`.

  Maps and tuples come back as dicts and tuples; a dataclass comes back as a
  plain dict of its non-default fields (its class is not needed to *play*);
  anything unrenderable stays as its repr string.
  """
  if isinstance(value, dict):
    if "__map__" in value:
      return {k: decode_value(v) for k, v in value["__map__"].items()}
    if "__tuple__" in value:
      return tuple(decode_value(v) for v in value["__tuple__"])
    if "__obj__" in value:
      obj = value["__obj__"]
      return {"__class__": f"{obj.get('module')}.{obj.get('name')}",
              **{k: decode_value(v) for k, v in (obj.get("fields") or {}).items()}}
    if "__slice_all__" in value:
      return None
    if "__ref__" in value:
      ref = value["__ref__"]
      return f"{ref.get('module')}.{ref.get('qualname')}"
    if "__repr__" in value:
      return value["__repr__"]
    return {k: decode_value(v) for k, v in value.items()}
  if isinstance(value, list):
    return [decode_value(v) for v in value]
  return value


def resolve_names(patterns: Sequence[str] | str | None, names: Sequence[str]) -> list[int]:
  """Indices of ``names`` matching the regex ``patterns``, in natural order --
  the same rule mjlab's ``find_joints`` applies with ``preserve_order=False``.
  ``None`` means every name."""
  if patterns is None:
    return list(range(len(names)))
  if isinstance(patterns, str):
    patterns = [patterns]
  compiled = [re.compile(f"^(?:{p})$") for p in patterns]
  found = [i for i, name in enumerate(names) if any(c.match(name) for c in compiled)]
  if not found:
    raise BundleError(f"no name matches {list(patterns)} among {list(names)}")
  return found


# ── quaternions (w, x, y, z), MuJoCo's convention ───────────────────────


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate ``v`` by the inverse of unit quaternion ``q``: world → body."""
  w, x, y, z = (float(c) for c in q)
  qv = np.array([x, y, z])
  return v * (2.0 * w * w - 1.0) - 2.0 * w * np.cross(qv, v) + 2.0 * qv * float(np.dot(qv, v))


# ── the terms ───────────────────────────────────────────────────────────


def _entity(player: Player, term: dict[str, Any]) -> dict[str, Any]:
  asset = (term.get("params") or {}).get("asset_cfg") or {}
  name = asset.get("name") if isinstance(asset, dict) else None
  return player.entity(name or player.spec.get("default_entity") or "robot")


def _joint_subset(player: Player, term: dict[str, Any], entity: dict[str, Any]) -> list[int]:
  asset = (term.get("params") or {}).get("asset_cfg") or {}
  patterns = asset.get("joint_names") if isinstance(asset, dict) else None
  return resolve_names(patterns, entity["joint_names"])


def term_builtin_sensor(player: Player, term: dict[str, Any]) -> np.ndarray:
  name = (term.get("params") or {}).get("sensor_name")
  sensor = player.spec["sensors"].get(name)
  if sensor is None:
    raise BundleError(f"sensor '{name}' is not in the model")
  start, dim = int(sensor["adr"]), int(sensor["dim"])
  return player.data.sensordata[start:start + dim].copy()


def term_projected_gravity(player: Player, term: dict[str, Any]) -> np.ndarray:
  entity = _entity(player, term)
  return quat_rotate_inverse(player.root_quat(entity), player.gravity)


def term_joint_pos_rel(player: Player, term: dict[str, Any]) -> np.ndarray:
  entity = _entity(player, term)
  ids = _joint_subset(player, term, entity)
  q = player.data.qpos[np.asarray(entity["joint_qpos_adr"])[ids]]
  return q - np.asarray(entity["default_joint_pos"])[ids]


def term_joint_vel_rel(player: Player, term: dict[str, Any]) -> np.ndarray:
  entity = _entity(player, term)
  ids = _joint_subset(player, term, entity)
  v = player.data.qvel[np.asarray(entity["joint_qvel_adr"])[ids]]
  default = np.asarray(entity.get("default_joint_vel") or np.zeros(len(entity["joint_names"])))
  return v - default[ids]


def term_last_action(player: Player, term: dict[str, Any]) -> np.ndarray:
  name = (term.get("params") or {}).get("action_name")
  if name:
    return player.last_action_of(name)
  return player.last_action.copy()


def term_base_lin_vel(player: Player, term: dict[str, Any]) -> np.ndarray:
  entity = _entity(player, term)
  adr = np.asarray(entity["root_qvel_adr"])
  # A free joint's linear velocity is the body origin's, in world coordinates.
  return quat_rotate_inverse(player.root_quat(entity), player.data.qvel[adr[:3]])


def term_base_ang_vel(player: Player, term: dict[str, Any]) -> np.ndarray:
  entity = _entity(player, term)
  adr = np.asarray(entity["root_qvel_adr"])
  # And its angular velocity is already in the body frame.
  return player.data.qvel[adr[3:6]].copy()


def term_generated_commands(player: Player, term: dict[str, Any]) -> np.ndarray:
  name = (term.get("params") or {}).get("command_name")
  return player.command(name)


TERMS: dict[str, Callable[[Player, dict[str, Any]], np.ndarray]] = {
    "builtin_sensor": term_builtin_sensor,
    "projected_gravity": term_projected_gravity,
    "joint_pos_rel": term_joint_pos_rel,
    "joint_vel_rel": term_joint_vel_rel,
    "last_action": term_last_action,
    "base_lin_vel": term_base_lin_vel,
    "base_ang_vel": term_base_ang_vel,
    "generated_commands": term_generated_commands,
}
"""What this player can compute, by the term function's own name. The export
refuses a bundle whose policy reads anything else."""


def term_key(term: dict[str, Any]) -> str:
  return str(term.get("func") or "").rsplit(".", 1)[-1]


def unsupported_terms(spec: dict[str, Any]) -> list[str]:
  """Every term in the spec this module cannot compute, named ``group.term``."""
  out = []
  for group in spec.get("observation", {}).get("groups") or []:
    for term in group.get("terms") or []:
      key = term_key(term)
      why = ""
      if key not in TERMS:
        why = f"{key} is not a term this player knows"
      elif key == "last_action" and (term.get("params") or {}).get("action_name") \
          and (term.get("params") or {}).get("action_name") not in {
              a["name"] for a in spec.get("actions") or []}:
        why = "names an action term the spec lacks"
      elif int(term.get("delay_max_lag") or 0) > 0:
        why = "observation delay is not supported"
      if why:
        out.append(f"{group['name']}.{term['name']}: {why}")
  for action in spec.get("actions") or []:
    if action.get("kind") != "joint_position":
      out.append(f"action {action.get('name')}: kind {action.get('kind')!r} is not supported")
  return out


# ── the player ──────────────────────────────────────────────────────────


class Player:
  """One robot, one policy, stepping in plain MuJoCo.

  ``policy`` maps a flat float32 observation (1, obs_dim) to raw actions
  (1, act_dim), the way the ONNX file does. ``reset`` takes an optional state
  so the training environment's own reset can be copied in for a comparison.
  """

  def __init__(self, model: Any, spec: dict[str, Any],
               policy: Callable[[np.ndarray], np.ndarray]):
    import mujoco

    self.mujoco = mujoco
    self.model = model
    self.spec = spec
    self.policy = policy
    self.data = mujoco.MjData(model)
    problems = unsupported_terms(spec)
    if problems:
      raise BundleError("this bundle cannot be played here: " + "; ".join(problems))
    self.groups = spec["observation"]["groups"]
    self.actions = spec["actions"]
    self.timing = spec["timing"]
    gravity = np.asarray(model.opt.gravity, dtype=float)
    norm = float(np.linalg.norm(gravity))
    self.gravity = gravity / norm if norm > 0 else np.array([0.0, 0.0, -1.0])
    self._commands: dict[str, np.ndarray] = {
        name: np.asarray(cmd.get("default") or np.zeros(int(cmd["dim"])), dtype=float)
        for name, cmd in (spec.get("commands") or {}).items()}
    self._last: dict[str, np.ndarray] = {}
    self._history: dict[tuple[str, str], deque] = {}
    self.last_action = np.zeros(sum(int(a["dim"]) for a in self.actions))
    self.steps = 0
    self.reset()

  # State.

  def entity(self, name: str) -> dict[str, Any]:
    try:
      return self.spec["entities"][name]
    except KeyError as exc:
      raise BundleError(f"the spec has no entity '{name}'") from exc

  def root_quat(self, entity: dict[str, Any]) -> np.ndarray:
    adr = entity.get("root_qpos_adr") or []
    if len(adr) < 7:
      # A fixed-base entity has no free joint; its root body's orientation is
      # the world's as far as gravity and velocities are concerned.
      return np.array([1.0, 0.0, 0.0, 0.0])
    return self.data.qpos[np.asarray(adr)[3:7]]

  def command(self, name: str) -> np.ndarray:
    if name not in self._commands:
      raise BundleError(f"the spec has no command '{name}'")
    return self._commands[name].copy()

  def set_command(self, name: str, value: Sequence[float]) -> None:
    current = self.command(name)
    given = np.asarray(value, dtype=float).reshape(-1)
    if given.shape != current.shape:
      raise BundleError(f"command '{name}' takes {current.shape[0]} values, not {given.shape[0]}")
    self._commands[name] = given

  def last_action_of(self, name: str) -> np.ndarray:
    return self._last.get(name, np.zeros(self._action_dim(name))).copy()

  def _action_dim(self, name: str) -> int:
    for action in self.actions:
      if action["name"] == name:
        return int(action["dim"])
    raise BundleError(f"the spec has no action '{name}'")

  def reset(self, qpos: Sequence[float] | None = None,
            qvel: Sequence[float] | None = None) -> None:
    """Back to the model's initial state, or to a given one, with empty
    action memory and history buffers that will backfill from the first
    observation -- the training buffer's own rule."""
    self.mujoco.mj_resetData(self.model, self.data)
    if qpos is not None:
      self.data.qpos[:] = np.asarray(qpos, dtype=float)
    else:
      for entity in self.spec.get("entities", {}).values():
        adr = entity.get("root_qpos_adr") or []
        init = entity.get("initial_root_pose")
        if len(adr) == 7 and init:
          self.data.qpos[np.asarray(adr)] = np.asarray(init, dtype=float)
        jadr = entity.get("joint_qpos_adr") or []
        default = entity.get("default_joint_pos")
        if jadr and default:
          self.data.qpos[np.asarray(jadr)] = np.asarray(default, dtype=float)
    if qvel is not None:
      self.data.qvel[:] = np.asarray(qvel, dtype=float)
    self.mujoco.mj_forward(self.model, self.data)
    self.last_action[:] = 0.0
    self._last = {}
    self._history = {}
    self.steps = 0

  # Observing.

  def term(self, group: str, term: dict[str, Any]) -> np.ndarray:
    """One term through the training stack's pipeline."""
    value = np.asarray(TERMS[term_key(term)](self, term), dtype=float).reshape(-1)
    clip = term.get("clip")
    if clip:
      value = np.clip(value, float(clip[0]), float(clip[1]))
    scale = term.get("scale")
    if scale is not None:
      value = value * np.asarray(scale, dtype=float)
    history = int(term.get("history_length") or 0)
    if history > 0:
      key = (group, str(term["name"]))
      buffer = self._history.get(key)
      if buffer is None:
        buffer = deque([value.copy()] * history, maxlen=history)
        self._history[key] = buffer
      else:
        buffer.append(value.copy())
      value = np.concatenate(list(buffer))
    return value

  def observe(self) -> np.ndarray:
    """The policy's input: every group's terms, in order, concatenated."""
    parts = [self.term(group["name"], term)
             for group in self.groups for term in group["terms"]]
    return np.concatenate(parts).astype(np.float32)

  # Acting.

  def act(self, obs: np.ndarray) -> np.ndarray:
    raw = np.asarray(self.policy(obs[None].astype(np.float32)), dtype=float).reshape(-1)
    clip = self.spec.get("clip_actions")
    if clip:
      raw = np.clip(raw, -float(clip), float(clip))
    return raw

  def apply(self, raw: np.ndarray) -> None:
    """Raw policy output → actuator targets, and remember it for `last_action`."""
    self.last_action = raw.copy()
    offset = 0
    for action in self.actions:
      dim = int(action["dim"])
      piece = raw[offset:offset + dim]
      offset += dim
      self._last[action["name"]] = piece.copy()
      scale = np.asarray(action["scale"], dtype=float)
      target = piece * scale + np.asarray(action["offset"], dtype=float)
      clip = action.get("clip")
      if clip:
        lo, hi = np.asarray(clip[0], dtype=float), np.asarray(clip[1], dtype=float)
        target = np.clip(target, lo, hi)
      self.data.ctrl[np.asarray(action["ctrl_ids"])] = target

  def step(self) -> np.ndarray:
    """One control step: observe, act, apply, advance the physics."""
    obs = self.observe()
    self.apply(self.act(obs))
    for _ in range(int(self.timing["decimation"])):
      self.mujoco.mj_step(self.model, self.data)
    self.steps += 1
    return obs

  @property
  def control_dt(self) -> float:
    return float(self.timing["physics_dt"]) * int(self.timing["decimation"])


# ── loading a bundle from disk ──────────────────────────────────────────


def read_spec(bundle_dir: Path | str) -> dict[str, Any]:
  path = Path(bundle_dir) / SPEC_FILE
  try:
    spec = json.loads(path.read_text())
  except OSError as exc:
    raise BundleError(f"no {SPEC_FILE} in {bundle_dir}: {exc}") from exc
  except json.JSONDecodeError as exc:
    raise BundleError(f"{path} is not JSON: {exc}") from exc
  if int(spec.get("spec_version") or 0) > SPEC_VERSION:
    raise BundleError(f"{path} is spec version {spec.get('spec_version')}; "
                      f"this rlmcp reads up to {SPEC_VERSION}")
  return spec


def load_model(bundle_dir: Path | str) -> Any:
  import mujoco

  path = Path(bundle_dir) / MODEL_FILE
  if not path.exists():
    raise BundleError(f"no {MODEL_FILE} in {bundle_dir}")
  return mujoco.MjModel.from_binary_path(str(path))


def onnx_policy(path: Path | str) -> Callable[[np.ndarray], np.ndarray]:
  """The ONNX file as a callable, through onnxruntime on the CPU."""
  try:
    import onnxruntime as ort
  except ImportError as exc:
    raise BundleError("playing an ONNX policy needs onnxruntime: "
                      "`pip install 'rl-mcp[bundle]'`") from exc
  session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  name = session.get_inputs()[0].name

  def policy(obs: np.ndarray) -> np.ndarray:
    return session.run(None, {name: obs.astype(np.float32)})[0]

  return policy


def load_player(bundle_dir: Path | str,
                policy: Callable[[np.ndarray], np.ndarray] | None = None) -> Player:
  bundle_dir = Path(bundle_dir)
  spec = read_spec(bundle_dir)
  model = load_model(bundle_dir)
  return Player(model, spec, policy or onnx_policy(bundle_dir / POLICY_FILE))


def rollout(bundle_dir: Path | str, seconds: float = 5.0,
            policy: Callable[[np.ndarray], np.ndarray] | None = None,
            commands: dict[str, Sequence[float]] | None = None) -> dict[str, Any]:
  """Play the bundle from its initial state and say what happened.

  The summary is what a portal page or a smoke test needs: how many control
  steps ran, how long they took, and the root's height over the rollout --
  a policy that fell shows as a height that collapsed.
  """
  player = load_player(bundle_dir, policy)
  for name, value in (commands or {}).items():
    player.set_command(name, value)
  steps = max(1, round(seconds / player.control_dt))
  entity = player.entity(player.spec.get("default_entity") or "robot")
  adr = entity.get("root_qpos_adr") or []
  heights: list[float] = []
  started = time.perf_counter()
  for _ in range(steps):
    player.step()
    if len(adr) >= 3:
      heights.append(float(player.data.qpos[adr[2]]))
  elapsed = time.perf_counter() - started
  out: dict[str, Any] = {
      "steps": steps, "seconds": round(steps * player.control_dt, 3),
      "compute_seconds": round(elapsed, 3),
      "realtime_factor": round((steps * player.control_dt) / elapsed, 1) if elapsed > 0 else None,
      "obs_dim": int(player.observe().shape[0]), "action_dim": int(player.last_action.shape[0]),
  }
  if heights:
    out["root_height"] = {"start": round(heights[0], 3), "min": round(min(heights), 3),
                          "end": round(heights[-1], 3)}
  return out


__all__ = [
  "MANIFEST_FILE",
  "MODEL_FILE",
  "POLICY_FILE",
  "SPEC_FILE",
  "SPEC_VERSION",
  "TERMS",
  "BundleError",
  "Player",
  "decode_value",
  "load_player",
  "onnx_policy",
  "quat_rotate_inverse",
  "read_spec",
  "resolve_names",
  "rollout",
  "unsupported_terms",
]
