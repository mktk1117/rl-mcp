"""A policy bundle: the trained policy, playable without the training stack.

``rlmcp bundle export`` takes a checkpoint and writes a directory that plays
in plain MuJoCo -- and therefore in a browser -- with nothing from mjlab,
torch or the task package installed:

    bundle/
      policy.onnx     the actor, observation normalisation folded in
      model.mjb       the compiled MuJoCo model, meshes included
      model.xml       the same scene as MJCF, for reading (no assets)
      spec.json       what the policy reads and emits, and how often
      manifest.json   what this is, what it was checked against, the result
      README.md       the same, for a person

A bundle is not a recipe. ``rlmcp recipe build`` makes a run *reproducible*:
the package at its commit, the ladder, the config, launchable. A bundle makes
a run's result *usable* anywhere: it carries no code at all, and what it
carries instead is a declaration of the policy's interface precise enough for
:mod:`rlmcp.bundle_play` to compute the observations itself.

**Checked, not claimed.** The export builds the task, loads the checkpoint,
and runs the environment for some hundreds of steps with the real policy
while the plain-MuJoCo player recomputes every observation from the same
state. The largest difference is in ``manifest.json``, and a bundle whose
observations do not match to tolerance -- or whose environment reads a term
the player cannot compute -- is refused with the term named. What is
published plays the way it trained, or is not published.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from rlmcp.bundle_play import (
  MANIFEST_FILE,
  MODEL_FILE,
  POLICY_FILE,
  SPEC_FILE,
  SPEC_VERSION,
  BundleError,
  Player,
  decode_value,
  rollout,
  term_key,
  unsupported_terms,
)

DEFAULT_STEPS = 300
DEFAULT_TOLERANCE = 1e-4
"""How far the player's observation may sit from the environment's, per
element, before the bundle is refused. The stack computes in float32 and the
player in float64; a real mismatch (a wrong frame, a missing offset) is
orders of magnitude larger than this."""

XML_FILE = "model.xml"
README_FILE = "README.md"


def export_bundle(checkpoint: str | Path, out: str | Path, task: str = "",
                  task_packages: list[str] | None = None, device: str = "cuda:0",
                  steps: int = DEFAULT_STEPS, tolerance: float = DEFAULT_TOLERANCE,
                  seconds: float = 5.0) -> dict[str, Any]:
  """Write a bundle for ``checkpoint`` into ``out`` and check it plays.

  Returns the manifest, with ``ok`` saying whether the bundle passed its
  checks. Everything is written either way, so a refused bundle can be read;
  ``unsupported`` and ``check`` say why it was refused.
  """
  from rlmcp.play import (
    PlayConfig,
    PlayError,
    _choose_gl_backend,
    build_env,
    checkpoint_iteration,
    find_checkpoint,
    session_for,
    task_for,
  )

  checkpoint = find_checkpoint(checkpoint)
  trained_session = session_for(checkpoint)
  task = task or task_for(trained_session)
  if not task:
    raise PlayError("No task: this checkpoint's session does not name one; pass --task.")
  out = Path(out).expanduser().resolve()
  out.mkdir(parents=True, exist_ok=True)

  cfg = PlayConfig(
      checkpoint=str(checkpoint), task=task, mode="hold", num_envs=1, device=device,
      task_package=list(task_packages or []), replay=False, quiet=True,
      session_dir=tempfile.mkdtemp(prefix="rlmcp-bundle-"))
  _choose_gl_backend(cfg)
  env, _controller, agent_cfg, vec_env = build_env(cfg, task, trained_session)
  # `build_env` hands back the rlmcp wrapper; the managers, the scene and the
  # simulation are on the environment it wraps.
  lab = getattr(env, "unwrapped", env)
  try:
    runner, policy_module, obs_groups = _load_runner(cfg, task, vec_env, checkpoint, agent_cfg)
    runner.export_policy_to_onnx(str(out), POLICY_FILE)
    _silence_noise(lab, obs_groups)
    spec = build_spec(lab, agent_cfg, obs_groups, task, checkpoint)
    _write_model(lab, out)
    spec["spec_version"] = SPEC_VERSION
    (out / SPEC_FILE).write_text(json.dumps(spec, indent=1, default=str) + "\n")

    manifest: dict[str, Any] = {
        "kind": "rlmcp-policy-bundle", "spec_version": SPEC_VERSION,
        "task": task, "checkpoint": str(checkpoint),
        "iteration": checkpoint_iteration(checkpoint),
        "session": str(trained_session) if trained_session else "",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "obs_groups": obs_groups,
        "obs_dim": int(sum(int(np.prod(t["dim"])) for g in spec["observation"]["groups"]
                          for t in g["terms"])),
        "action_dim": int(sum(int(a["dim"]) for a in spec["actions"])),
        "unsupported": unsupported_terms(spec),
        "check": None, "rollout": None, "ok": False,
    }
    if not manifest["unsupported"]:
      manifest["check"] = compare_observations(lab, vec_env, runner, spec, out, obs_groups,
                                               steps=steps, tolerance=tolerance)
      manifest["ok"] = bool(manifest["check"]["ok"])
      if manifest["ok"]:
        policy = _numpy_policy(policy_module)
        try:
          manifest["rollout"] = rollout(out, seconds=seconds, policy=policy)
        except BundleError as exc:
          manifest["rollout"] = {"error": str(exc)}
    manifest["files"] = {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}
  finally:
    with contextlib.suppress(Exception):   # the export is written; closing is courtesy
      env.close()
  (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1, default=str) + "\n")
  (out / README_FILE).write_text(render_readme(manifest, spec))
  return {**manifest, "path": str(out)}


def _load_runner(cfg: Any, task: str, vec_env: Any, checkpoint: Path,
                 agent_cfg: Any) -> tuple[Any, Any, list[str]]:
  """The runner with the checkpoint loaded, its policy model, and the
  observation groups that model concatenates -- in the order it does."""
  from mjlab.rl import MjlabOnPolicyRunner
  from mjlab.tasks.registry import load_runner_cls

  from rlmcp.play import PlayError

  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  with tempfile.TemporaryDirectory(prefix="rlmcp-bundle-runner-") as tmp:
    runner = runner_cls(vec_env, dataclasses.asdict(agent_cfg), tmp, cfg.device)
    try:
      runner.load(str(checkpoint))
    except Exception as exc:
      raise PlayError(
          f"Could not load {checkpoint.name} into a '{task}' runner: {exc}. "
          "A checkpoint only fits the task it was trained on.") from exc
  model = runner.alg.get_policy()
  groups = list(getattr(model, "obs_groups", None) or ["actor"])
  return runner, model, groups


def _silence_noise(lab: Any, groups: list[str]) -> None:
  """Observation noise is a training aid: the player never applies it, so the
  comparison must not either. The manager reads each term's noise per call."""
  manager = lab.observation_manager
  for group in groups:
    for name in list(manager.active_terms.get(group, []) or []):
      manager.get_term_cfg(group, name).noise = None


def _numpy_policy(policy_module: Any) -> Any:
  """The actor as a numpy callable, through its own ONNX-export module, so the
  rollout needs no onnxruntime and runs exactly what was exported."""
  import torch

  module = policy_module.as_onnx(verbose=False).to("cpu").eval()

  def policy(obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
      return module(torch.as_tensor(obs, dtype=torch.float32)).numpy()

  return policy


# ── the spec ────────────────────────────────────────────────────────────


def build_spec(lab: Any, agent_cfg: Any, obs_groups: list[str], task: str,
               checkpoint: Path) -> dict[str, Any]:
  """Everything the player needs, read off the built environment."""
  import mujoco

  from rlmcp.adapters.manager_based.term_capture import capture_env_terms

  snapshot = capture_env_terms(lab)
  captured = snapshot.get("observations") or {}
  groups = []
  for name in obs_groups:
    if name not in captured:
      raise BundleError(f"the policy reads observation group '{name}', which the "
                        f"environment does not have (it has {sorted(captured)})")
    terms = []
    for term in captured[name].get("terms") or []:
      func = term.get("func") or {}
      terms.append({
          "name": term["name"],
          "func": f"{func.get('module')}.{func.get('qualname')}",
          "params": decode_value(term.get("params") or {}),
          "dim": list(term.get("dim") or []),
          "scale": _plain(decode_value(term.get("scale"))),
          "clip": _plain(decode_value(term.get("clip"))),
          "history_length": int(decode_value(term.get("history_length")) or 0),
          "delay_max_lag": int(_term_attr(lab, name, term["name"], "delay_max_lag") or 0),
      })
    groups.append({"name": name, "terms": terms})

  mjm = lab.sim.mj_model
  actions, entity_names = _actions(lab, mjm)
  for group in groups:
    for term in group["terms"]:
      asset = (term["params"] or {}).get("asset_cfg")
      if isinstance(asset, dict) and asset.get("name"):
        entity_names.append(str(asset["name"]))
  if "robot" in _entity_names(lab):
    entity_names.append("robot")
  entities = {name: _entity(lab, name) for name in dict.fromkeys(entity_names)}

  sensors = {}
  for s in range(mjm.nsensor):
    name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_SENSOR, s)
    sensors[name] = {"adr": int(mjm.sensor_adr[s]), "dim": int(mjm.sensor_dim[s])}

  commands = {}
  manager = getattr(lab, "command_manager", None)
  for name in list(getattr(manager, "active_terms", []) or []):
    try:
      value = manager.get_command(name)[0].detach().cpu().numpy().reshape(-1)
    except Exception:
      continue
    commands[name] = {"dim": int(value.shape[0]), "default": [0.0] * int(value.shape[0]),
                      "sample": [float(v) for v in value]}

  return {
      "task": task,
      "checkpoint": checkpoint.name,
      "timing": {"physics_dt": float(lab.physics_dt), "decimation": int(lab.cfg.decimation)},
      "default_entity": "robot" if "robot" in entities else next(iter(entities), ""),
      "entities": entities,
      "observation": {"groups": groups},
      "actions": actions,
      "commands": commands,
      "sensors": sensors,
      "clip_actions": getattr(agent_cfg, "clip_actions", None),
      "terms_problems": snapshot.get("problems") or [],
  }


def _term_attr(lab: Any, group: str, name: str, attr: str) -> Any:
  try:
    return getattr(lab.observation_manager.get_term_cfg(group, name), attr, None)
  except Exception:
    return None


def _plain(value: Any) -> Any:
  """Scales and clips as JSON lists, or None; a torch tensor as a list."""
  if value is None:
    return None
  if hasattr(value, "detach"):
    return value.detach().cpu().numpy().reshape(-1).tolist()
  if isinstance(value, tuple):
    return [float(v) for v in value]
  if isinstance(value, (int, float)):
    return float(value)
  return value


def _entity_names(lab: Any) -> list[str]:
  scene = lab.scene
  for attr in ("entities", "_entities"):
    found = getattr(scene, attr, None)
    if isinstance(found, dict):
      return list(found)
  return []


def _entity(lab: Any, name: str) -> dict[str, Any]:
  ent = lab.scene[name]
  idx = ent.indexing
  qpos = lab.sim.data.qpos[0].detach().cpu().numpy()
  root_q = idx.free_joint_q_adr.detach().cpu().numpy().reshape(-1).tolist()
  out: dict[str, Any] = {
      "joint_names": list(ent.joint_names),
      "joint_qpos_adr": idx.joint_q_adr.detach().cpu().numpy().reshape(-1).tolist(),
      "joint_qvel_adr": idx.joint_v_adr.detach().cpu().numpy().reshape(-1).tolist(),
      "root_qpos_adr": root_q,
      "root_qvel_adr": idx.free_joint_v_adr.detach().cpu().numpy().reshape(-1).tolist(),
      "root_body_id": int(idx.root_body_id),
  }
  default_pos = getattr(ent.data, "default_joint_pos", None)
  if default_pos is not None:
    out["default_joint_pos"] = default_pos[0].detach().cpu().numpy().reshape(-1).tolist()
  default_vel = getattr(ent.data, "default_joint_vel", None)
  if default_vel is not None:
    out["default_joint_vel"] = default_vel[0].detach().cpu().numpy().reshape(-1).tolist()
  if len(root_q) == 7:
    out["initial_root_pose"] = [float(v) for v in qpos[np.asarray(root_q)]]
  return out


def _actions(lab: Any, mjm: Any) -> tuple[list[dict[str, Any]], list[str]]:
  """Every action term, as the player applies it, plus the entities they drive."""
  import mujoco

  manager = lab.action_manager
  ctrl_of_joint = {int(mjm.actuator_trnid[a, 0]): a for a in range(mjm.nu)
                   if mjm.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
  out: list[dict[str, Any]] = []
  entities: list[str] = []
  names = list(manager.active_terms)
  dims = list(getattr(manager, "action_term_dim", []) or [])
  for index, name in enumerate(names):
    term = manager.get_term(name)
    kind = type(term).__name__
    entry: dict[str, Any] = {
        "name": name,
        "kind": "joint_position" if kind == "JointPositionAction" else f"unsupported:{kind}",
        "dim": int(dims[index]) if index < len(dims) else int(getattr(term, "action_dim", 0)),
    }
    entity_name = str(getattr(getattr(term, "cfg", None), "entity_name", "") or "robot")
    entities.append(entity_name)
    if entry["kind"] == "joint_position":
      ent = lab.scene[entity_name]
      target_local = [int(i) for i in term.target_ids]
      joint_ids = ent.indexing.joint_ids.detach().cpu().numpy().reshape(-1).tolist()
      try:
        ctrl_ids = [ctrl_of_joint[joint_ids[j]] for j in target_local]
      except KeyError:
        entry["kind"] = "unsupported:JointPositionAction-without-joint-actuator"
        ctrl_ids = []
      entry.update({
          "entity": entity_name,
          "joint_local_ids": target_local,
          "joint_names": list(term.target_names),
          "scale": _per_joint(term.scale, entry["dim"]),
          "offset": _per_joint(term.offset, entry["dim"]),
          "ctrl_ids": ctrl_ids,
      })
      clip = getattr(term, "_clip", None)
      if clip is not None and hasattr(clip, "detach"):
        arr = clip.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[-1] == 2:
          entry["clip"] = [arr[0, :, 0].tolist(), arr[0, :, 1].tolist()]
    out.append(entry)
  return out, entities


def _per_joint(value: Any, dim: int) -> list[float]:
  if hasattr(value, "detach"):
    arr = value.detach().cpu().numpy()
    arr = arr[0] if arr.ndim == 2 else arr.reshape(-1)
    return [float(v) for v in arr]
  return [float(value)] * dim


def _write_model(lab: Any, out: Path) -> None:
  import mujoco

  mujoco.mj_saveModel(lab.sim.mj_model, str(out / MODEL_FILE))
  # The MJB is the one that plays; the XML is for reading, and some scenes cannot write it.
  with contextlib.suppress(Exception):
    (out / XML_FILE).write_text(lab.scene.spec.to_xml())


# ── the check ───────────────────────────────────────────────────────────


def compare_observations(lab: Any, vec_env: Any, runner: Any, spec: dict[str, Any],
                         bundle_dir: Path, obs_groups: list[str], steps: int,
                         tolerance: float) -> dict[str, Any]:
  """Run the environment with the real policy; recompute every observation
  from the same state with the plain player; report the largest difference.

  A reset mid-way resets the player too (its history and action memory),
  the way the environment's own buffers reset.
  """
  import mujoco
  import torch

  model = mujoco.MjModel.from_binary_path(str(bundle_dir / MODEL_FILE))
  player = Player(model, spec, policy=lambda o: np.zeros((1, spec_action_dim(spec))))
  policy = runner.get_inference_policy(device=lab.device)
  # The observation the wrapper cached at construction was computed with the
  # task's noise still on; a reset after `_silence_noise` is the first clean
  # one, and the player starts its own memory from the same point.
  reset = vec_env.reset()
  obs = reset[0] if isinstance(reset, tuple) else reset
  player.reset()
  worst, worst_term, worst_step = 0.0, "", -1
  layout = [(g["name"], t["name"], int(np.prod(t["dim"])))
            for g in spec["observation"]["groups"] for t in g["terms"]]
  for step in range(int(steps)):
    qpos = lab.sim.data.qpos[0].detach().cpu().numpy()
    qvel = lab.sim.data.qvel[0].detach().cpu().numpy()
    player.data.qpos[:] = qpos
    player.data.qvel[:] = qvel
    mujoco.mj_forward(player.model, player.data)
    mine = player.observe().astype(np.float64)
    theirs = torch.cat([obs[g] for g in obs_groups], dim=-1)[0]
    theirs = theirs.detach().cpu().numpy().astype(np.float64)
    if mine.shape != theirs.shape:
      return {"ok": False, "steps": step, "tolerance": tolerance,
              "error": f"the player's observation has {mine.shape[0]} elements, "
                       f"the environment's {theirs.shape[0]}"}
    diff = np.abs(mine - theirs)
    if diff.max() > worst:
      worst = float(diff.max())
      worst_step = step
      i, acc = int(diff.argmax()), 0
      for group, name, n in layout:
        if i < acc + n:
          worst_term = f"{group}.{name}[{i - acc}]"
          break
        acc += n
    with torch.no_grad():
      act = policy(obs)
    player.apply(act[0].detach().cpu().numpy().astype(np.float64))
    obs, _rew, dones, _extras = vec_env.step(act)
    if bool(dones[0]):
      player.reset()
  return {"ok": worst <= tolerance, "steps": int(steps), "tolerance": tolerance,
          "max_abs_diff": worst, "worst_term": worst_term, "worst_step": worst_step}


def spec_action_dim(spec: dict[str, Any]) -> int:
  return int(sum(int(a["dim"]) for a in spec.get("actions") or []))


# ── words ───────────────────────────────────────────────────────────────


def render_readme(manifest: dict[str, Any], spec: dict[str, Any]) -> str:
  lines = [f"# Policy bundle: {manifest.get('task')}", ""]
  lines.append(f"Checkpoint `{Path(str(manifest.get('checkpoint'))).name}`"
               f" (iteration {manifest.get('iteration')}), exported {manifest.get('created')}.")
  lines.append("")
  lines.append("Plays with `rlmcp bundle play <this directory>` -- plain MuJoCo and the "
               "ONNX policy, no training stack -- or in a browser player that follows "
               "`rlmcp/bundle_play.py`.")
  lines.append("")
  check = manifest.get("check") or {}
  if manifest.get("unsupported"):
    lines.append("## Refused")
    lines.append("")
    lines.append("This policy's environment reads something the player cannot compute:")
    lines.extend(f"- {u}" for u in manifest["unsupported"])
  elif check:
    verdict = "passed" if check.get("ok") else "FAILED"
    lines.append(f"## Check: {verdict}")
    lines.append("")
    if check.get("error"):
      lines.append(check["error"])
    else:
      lines.append(f"Over {check.get('steps')} environment steps with the real policy, the "
                   f"player's observations differed from the environment's by at most "
                   f"{check.get('max_abs_diff'):.2e} (tolerance {check.get('tolerance'):.0e}), "
                   f"at `{check.get('worst_term')}`.")
  roll = manifest.get("rollout") or {}
  if roll and not roll.get("error"):
    height = roll.get("root_height") or {}
    lines.append("")
    lines.append(f"A {roll.get('seconds')} s rollout from the initial state ran at "
                 f"{roll.get('realtime_factor')}x real time on the CPU"
                 + (f"; root height {height.get('start')} → {height.get('end')} m "
                    f"(min {height.get('min')})." if height else "."))
  lines.append("")
  lines.append("## What the policy reads")
  lines.append("")
  for group in spec.get("observation", {}).get("groups") or []:
    lines.append(f"Group `{group['name']}`:")
    lines.append("")
    lines.append("| term | function | dim | scale | clip | history |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for t in group["terms"]:
      lines.append(f"| {t['name']} | `{term_key(t)}` | {int(np.prod(t['dim']))} | "
                   f"{t.get('scale') or ''} | {t.get('clip') or ''} | "
                   f"{t.get('history_length') or ''} |")
    lines.append("")
  lines.append("## What it emits")
  lines.append("")
  hz = 1.0 / (spec["timing"]["physics_dt"] * spec["timing"]["decimation"])
  for a in spec.get("actions") or []:
    lines.append(f"- `{a['name']}`: {a['kind']}, {a['dim']} values → actuator targets "
                 f"`raw * scale + offset` at {hz:.0f} Hz")
  if spec.get("commands"):
    lines.append("")
    lines.append("## Commands")
    lines.append("")
    for name, cmd in spec["commands"].items():
      lines.append(f"- `{name}`: {cmd['dim']} values, default {cmd['default']} "
                   f"(the environment sampled {[round(v, 3) for v in cmd.get('sample', [])]})")
  lines.append("")
  return "\n".join(lines)


def describe(payload: dict[str, Any]) -> str:
  if payload.get("unsupported"):
    return ("Refused: the player cannot compute " + "; ".join(payload["unsupported"])
            + f". Files are in {payload.get('path')} for reading.")
  check = payload.get("check") or {}
  if not payload.get("ok"):
    return (f"Refused: {check.get('error') or 'observations did not match'} "
            f"(max diff {check.get('max_abs_diff')}, at {check.get('worst_term')}).")
  roll = payload.get("rollout") or {}
  return (f"Bundle written to {payload.get('path')}: obs {payload.get('obs_dim')}, "
          f"actions {payload.get('action_dim')}; observations match to "
          f"{check.get('max_abs_diff'):.1e} over {check.get('steps')} steps"
          + (f"; a {roll.get('seconds')} s rollout ran at {roll.get('realtime_factor')}x"
             if roll.get("realtime_factor") else "") + ".")


__all__ = ["DEFAULT_STEPS", "DEFAULT_TOLERANCE", "build_spec", "compare_observations",
           "describe", "export_bundle", "render_readme"]
