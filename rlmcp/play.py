"""``rlmcp play``: watch a saved checkpoint, as a clip or in a viewer.

Everything else in rlmcp talks to a *live* trainer. That leaves a gap: a run
ends -- normally, or because its falsifier fired -- and the one thing nobody can
do any more is look at it. The evidence that would settle the question is a
checkpoint on disk and no way to play it.

This is that way. It rebuilds the environment, restores the conditions the
policy was trained under (see :mod:`rlmcp.core.replay` for why that matters more
than it sounds), loads the weights, and then either records a clip or opens a
viewer:

* ``--mode video`` renders offscreen to an mp4. No display needed, so it works
  over ssh, in a cron job, and on a machine whose GPU is busy training -- pass
  ``--device cpu`` and it will not touch the card.
* ``--mode native`` opens MuJoCo's own viewer, for looking closely at a robot
  that is in front of you.
* ``--mode viser`` serves a viewer in the browser, for a robot that is not.

Neither viewer is required to be installed: a build without them says so and
points at ``--mode video``, rather than failing on import.

The environment is wrapped exactly as a training run is, so a play session is
itself steerable -- ``rlmcp --session <play dir> run <command>`` changes the
task under the policy while you watch it. It writes itself down as a *play*
session, which session discovery skips, so it never becomes the answer to
"the latest session here".

Nothing above the ``run_play`` entry point imports a simulator, torch or an
encoder, so ``rlmcp --help`` stays as cheap as it has always been.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rlmcp.core.replay import (
    Conditions,
    apply_conditions,
    read_conditions,
    with_overrides,
)

MODES = ("video", "native", "viser")

TASK_PACKAGES_ENV = "RLMCP_TASK_PACKAGES"
"""Comma-separated packages to import before reading the task registry.

``--task-package`` is the explicit form and is what ``rlmcp train`` takes. This
is the same thing said once for a whole shell, which is what you want when
every run in a project draws its tasks from the same package.
"""

# Checkpoints are conventionally model_<iteration>.pt / model_final_<n>.pt, but
# nothing here depends on that name -- only on the suffix.
_CHECKPOINT_GLOB = "*.pt"

# How far above a given path to look for them. A play session sits three deep
# inside a run (<run>/rlmcp/play/<stamp>); beyond that we would start climbing
# out of the run and into whatever else the log root holds.
_SEARCH_DEPTH = 5


class PlayError(RuntimeError):
  """Something about the request cannot work, stated in terms the caller can fix."""


@dataclass
class PlayConfig:
  """Everything ``play`` needs, with defaults that suit the common case.

  The common case is "the run just ended and I want to see it": point at a run
  directory, take the last checkpoint, replay the stage it died in, write eight
  seconds of mp4 next to the run's other evidence.
  """

  checkpoint: str = ""
  """A checkpoint file, or a run/session directory to take the latest from."""
  task: str = ""
  """Defaults to the task recorded in the session this checkpoint belongs to."""
  mode: str = "video"
  seconds: float = 8.0
  num_envs: int = 1
  device: str = "cuda:0"
  out: str = ""
  """Video path. Defaults into the training session's artifacts directory."""
  fps: Optional[int] = None
  """Defaults to the environment's own control rate, so the clip runs in real time."""
  stage: str = ""
  """Curriculum stage to restore. Defaults to the last one the run entered."""
  replay: bool = True
  """Restore trained-under conditions at all. Off gives the task's play config."""
  overrides: Dict[str, Any] = field(default_factory=dict)
  """Parameter edits applied after the replay, so they win."""
  task_package: List[str] = field(default_factory=list)
  render_width: int = 960
  render_height: int = 720
  extra_envs: Optional[int] = 0
  """Neighbouring envs composited into the frame. 0 -- an unambiguous close-up."""
  session_dir: str = ""
  """Where the play session publishes itself. Defaults under the run directory."""
  allow_partial: bool = False
  """Render even when some conditions could not be restored. Rarely what you want."""
  quiet: bool = False


# Finding things.


def find_checkpoint(target: Path | str) -> Path:
  """Resolve a file, a run directory or a session directory to one checkpoint.

  Picking "the latest" by iteration rather than by mtime, because a run that
  saved ``model_final_4375.pt`` and then had ``model_900.pt`` copied back in
  for comparison should still play the one it ended on.
  """
  path = Path(target).expanduser()
  if path.is_file():
    return path
  if not path.exists():
    raise PlayError(f"No such checkpoint or run directory: {path}")

  # Checkpoints live in the run directory, but the path in hand is often
  # something inside it -- the session (…/rlmcp), or a play session nested
  # deeper still. Walk outwards until a directory has some, so pointing at any
  # part of a run finds the run's checkpoints.
  found: List[Path] = []
  for directory in [path, *path.parents][:_SEARCH_DEPTH]:
    found = [p for p in directory.glob(_CHECKPOINT_GLOB) if p.is_file()]
    found += list((directory / "checkpoints").glob(_CHECKPOINT_GLOB))
    if found:
      break
  if not found:
    raise PlayError(
        f"No .pt checkpoints in {path} or the directories above it. Point "
        "--checkpoint at the file itself if it lives somewhere else."
    )
  return max(found, key=lambda p: (checkpoint_iteration(p), p.stat().st_mtime))


def checkpoint_iteration(path: Path) -> int:
  """The iteration in ``model_4300.pt`` / ``model_final_4375.pt``; -1 if unnamed."""
  digits = [part for part in path.stem.replace("-", "_").split("_") if part.isdigit()]
  return int(digits[-1]) if digits else -1


def session_for(checkpoint: Path) -> Optional[Path]:
  """The rlmcp session a checkpoint belongs to, if it has one.

  A trainer's checkpoints sit in the run directory and the session is
  ``<run>/rlmcp``; a checkpoint saved through ``rlmcp checkpoint`` sits one
  deeper, inside ``<session>/checkpoints``.
  """
  for candidate in (
      checkpoint.parent / "rlmcp",
      checkpoint.parent.parent / "rlmcp",
      checkpoint.parent,
      checkpoint.parent.parent,
  ):
    if (candidate / "session.json").exists():
      return candidate
  return None


def session_info(session_dir: Optional[Path]) -> Dict[str, Any]:
  """What the session recorded about itself, or {} if there is nothing to read."""
  if session_dir is None:
    return {}
  import json

  try:
    info = json.loads((session_dir / "session.json").read_text())
  except (OSError, ValueError):
    return {}
  return info if isinstance(info, dict) else {}


def task_for(session_dir: Optional[Path]) -> str:
  """The task id the session recorded, or '' if there is nothing to read."""
  return str(session_info(session_dir).get("task") or "")


def packages_to_import(cfg: PlayConfig) -> List[str]:
  """Packages that must be imported before the task registry knows the task.

  A project's tasks register on import, exactly as the simulator's own do --
  the same contract ``rlmcp train --task-package`` uses. Nothing here guesses
  at a package name or goes looking for one on disk: a replay months later
  imports what it was told to import, and says so plainly when that is nothing.
  """
  found = list(cfg.task_package)
  for name in os.environ.get(TASK_PACKAGES_ENV, "").split(","):
    name = name.strip()
    if name and name not in found:
      found.append(name)
  return found


# Playing.


def run_play(cfg: PlayConfig) -> Dict[str, Any]:
  """Build the environment, restore conditions, load the policy, and show it.

  Returns a result payload; in video mode it carries ``video_path``, which is
  what makes the CLI put the clip on screen.
  """
  if cfg.mode not in MODES:
    raise PlayError(f"Unknown mode '{cfg.mode}'. Choose one of: {', '.join(MODES)}")

  checkpoint = find_checkpoint(cfg.checkpoint or Path.cwd())
  session_dir = session_for(checkpoint)
  task = cfg.task or task_for(session_dir)
  if not task:
    raise PlayError(
        f"Could not tell which task {checkpoint.name} was trained on: there is "
        "no session.json near it. Pass --task."
    )

  _choose_gl_backend(cfg)
  env, lab, agent_cfg, vec_env = _build_env(cfg, task, session_dir)

  conditions, restored = _restore_conditions(cfg, lab, session_dir)
  policy = _load_policy(cfg, task, vec_env, checkpoint, agent_cfg)

  result: Dict[str, Any] = {
      "mode": cfg.mode,
      "checkpoint": str(checkpoint),
      "iteration": checkpoint_iteration(checkpoint),
      "task": task,
      "device": cfg.device,
      "num_envs": cfg.num_envs,
      "trained_session": str(session_dir) if session_dir else None,
      "play_session": str(lab.session.dir),
      "conditions": {
          "replayed": cfg.replay,
          "stage": conditions.stage or None,
          **restored,
      },
  }

  try:
    if cfg.mode == "video":
      result.update(_record(cfg, env, vec_env, policy, checkpoint, session_dir))
    else:
      result.update(_view(cfg, env, vec_env, policy))
  finally:
    try:
      vec_env.close()
    except Exception:
      pass
  return result


def _choose_gl_backend(cfg: PlayConfig) -> None:
  """Pick a GL backend before anything imports mujoco, because it is read once.

  Offscreen rendering wants EGL, which needs no display and is the only thing
  that works over ssh. The native viewer wants a real window, so EGL would make
  it fail with an error about the frame buffer rather than about the display.
  """
  os.environ.setdefault("MUJOCO_GL", "glfw" if cfg.mode == "native" else "egl")
  if cfg.device.startswith("cuda"):
    index = cfg.device.split(":")[-1] if ":" in cfg.device else "0"
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", index)


def _build_env(
    cfg: PlayConfig, task: str, session_dir: Optional[Path]
) -> Tuple[Any, Any, Any, Any]:
  import importlib

  import mjlab.tasks  # noqa: F401  (populates the task registry)

  for module in packages_to_import(cfg):
    try:
      importlib.import_module(module)
    except ImportError as exc:
      raise PlayError(
          f"Could not import task package '{module}': {exc}. It is on the "
          "PYTHONPATH of whatever trained this run, but not of this shell."
      ) from exc

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
  from mjlab.utils.torch import configure_torch_backends

  import rlmcp
  from rlmcp.session import PLAY_SESSION_KIND

  registered = list_tasks()
  if task not in registered:
    raise PlayError(
        f"Unknown task '{task}'. Nothing imported here registers it. Pass "
        f"--task-package <module> (repeatable) or set {TASK_PACKAGES_ENV}, "
        "naming the package whose import registers this task -- the same "
        "package `rlmcp train --task-package` was given. Registered here: "
        + (", ".join(registered) or "(nothing)")
    )

  configure_torch_backends()

  # play=True is the right starting point -- it turns off the randomisation
  # that exists to make training robust and would only make a clip noisy -- but
  # it is also rung zero of any curriculum, which _restore_conditions fixes.
  env_cfg = load_env_cfg(task, play=True)
  agent_cfg = load_rl_cfg(task)
  env_cfg.scene.num_envs = max(1, int(cfg.num_envs))
  env_cfg.viewer.width = cfg.render_width
  env_cfg.viewer.height = cfg.render_height
  if cfg.extra_envs is not None and hasattr(env_cfg.viewer, "max_extra_envs"):
    # Neighbouring envs composited into one frame read as duplicate objects,
    # which is the same trap `RlMcpEnvWrapper` warns about at startup. Off by
    # default here: a clip of one policy should show one policy.
    env_cfg.viewer.max_extra_envs = int(cfg.extra_envs)

  # render_mode stays None in every mode: rlmcp builds the offscreen renderer
  # on the first frame, which is the path the whole project renders through and
  # the only one that knows how to point at a chosen environment.
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device, render_mode=None)

  play_session = (
      Path(cfg.session_dir) if cfg.session_dir
      else _default_session(session_dir, cfg.mode)
  )
  env = rlmcp.wrap(
      env,
      session_dir=play_session,
      curriculum=None,  # A replay is not a ladder; nothing here promotes.
      task_id=task,
      session_kind=PLAY_SESSION_KIND,
  )
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  return env, env.rlmcp, agent_cfg, vec_env


def _default_session(trained_session: Optional[Path], mode: str) -> Path:
  stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
  root = trained_session if trained_session else Path.cwd() / "rlmcp_session"
  return root / "play" / f"{stamp}_{mode}"


def _restore_conditions(
    cfg: PlayConfig, lab: Any, session_dir: Optional[Path]
) -> Tuple[Conditions, Dict[str, Any]]:
  """Put the environment back where the checkpoint left it."""
  if not cfg.replay:
    conditions = with_overrides(Conditions(), cfg.overrides)
  elif session_dir is None:
    conditions = with_overrides(
        Conditions(
            warnings=(
                "No rlmcp session found next to this checkpoint, so there is "
                "nothing to replay: the environment runs at the task's own play "
                "configuration, which is rung zero of any curriculum.",
            )
        ),
        cfg.overrides,
    )
  else:
    conditions = with_overrides(
        read_conditions(session_dir, cfg.stage or None), cfg.overrides
    )

  restored = apply_conditions(lab, conditions)
  if restored["errors"] and not cfg.allow_partial:
    raise PlayError(_cannot_restore(restored))
  if not cfg.quiet:
    stage = conditions.stage or "(no curriculum)"
    print(
        f"[rlmcp-play] conditions: stage '{stage}', "
        f"{len(restored['parameters'])} parameters, {len(restored['calls'])} commands",
        flush=True,
    )
    for message in restored["warnings"] + restored["errors"]:
      print(f"[rlmcp-play]   ! {message}", flush=True)
  return conditions, restored


def _cannot_restore(restored: Dict[str, Any]) -> str:
  """Say what went wrong, and what to do about that particular thing.

  The advice matters more than the list. "Unknown command" and "the command no
  longer takes this argument" are different problems with different fixes, and
  a message that guesses at the wrong one sends whoever is reading it to check
  something that was never broken.
  """
  listed = "\n  ".join(restored["errors"])
  kinds = set(restored.get("error_kinds") or ())
  advice: List[str] = []
  if "missing_command" in kinds:
    advice.append(
        "A command that does not exist here means the package defining this "
        "task's vocabulary was not imported -- pass --task-package."
    )
  if "changed_command" in kinds:
    advice.append(
        "A command that no longer takes these arguments means the task's code "
        "has moved on since this run. There is no way to restore exactly what "
        "was trained; --allow-partial renders without that command, and --set "
        "<parameter>=<value> puts the underlying knob back by hand."
    )
  if "parameter" in kinds:
    advice.append(
        "A refused parameter is one this build of the task no longer has, or "
        "no longer accepts that value for."
    )
  if not advice:
    advice.append("--allow-partial renders anyway.")
  return (
      "Could not restore the conditions this checkpoint was trained under, so "
      "a clip would show the policy attempting a different task than the one "
      f"it learned:\n  {listed}\n" + "\n".join(advice)
  )


def _load_policy(
    cfg: PlayConfig, task: str, vec_env: Any, checkpoint: Path, agent_cfg: Any
) -> Any:
  import tempfile
  from dataclasses import asdict

  from mjlab.rl import MjlabOnPolicyRunner
  from mjlab.tasks.registry import load_runner_cls

  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  # The runner writes a log directory it will never use here; a temporary one
  # keeps a play from leaving an empty run beside the real ones.
  with tempfile.TemporaryDirectory(prefix="rlmcp-play-") as tmp:
    runner = runner_cls(vec_env, asdict(agent_cfg), tmp, cfg.device)
    try:
      runner.load(str(checkpoint))
    except Exception as exc:
      raise PlayError(
          f"Could not load {checkpoint.name} into a '{task}' runner: {exc}. "
          "A checkpoint only fits the task it was trained on -- if the task's "
          "observations or actions have changed since, this is that."
      ) from exc
  return runner.get_inference_policy(device=cfg.device)


def _record(
    cfg: PlayConfig,
    env: Any,
    vec_env: Any,
    policy: Any,
    checkpoint: Path,
    session_dir: Optional[Path],
) -> Dict[str, Any]:
  """Roll the policy out and encode what it did."""
  import imageio.v2 as imageio
  import numpy as np
  import torch

  lab = env.rlmcp
  step_dt = float(getattr(env.unwrapped, "step_dt", 0.02))
  steps = max(1, int(round(cfg.seconds / step_dt)))
  fps = int(cfg.fps or round(1.0 / step_dt))

  frames: List[Any] = []
  obs, _ = vec_env.reset()
  with torch.no_grad():
    for _ in range(steps):
      obs, _, _, _ = vec_env.step(policy(obs))
      try:
        frames.append(np.asarray(lab.sim.render(0)).astype(np.uint8))
      except Exception as exc:
        # A build with no offscreen renderer cannot make a clip, and finding
        # that out 200 steps in is worse than finding it out on the first one.
        raise PlayError(
            f"This environment could not render a frame ({type(exc).__name__}: "
            f"{exc}). --mode native or --mode viser still work if a viewer is "
            "installed; a headless machine needs an EGL-capable mujoco build."
        ) from exc

  if not frames:
    raise PlayError(f"Rendered no frames in {steps} steps.")

  out = Path(cfg.out) if cfg.out else _default_out(checkpoint, session_dir, cfg)
  out.parent.mkdir(parents=True, exist_ok=True)
  # +faststart puts the moov atom in front of the media data. Without it a
  # player has to fetch the whole file before showing anything, so the clip
  # silently never starts in a preview pane -- the file is fine, it just does
  # not play where anyone looks at it.
  imageio.mimwrite(
      str(out),
      frames,
      fps=fps,
      codec="libx264",
      macro_block_size=1,
      output_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"],
  )
  if not cfg.quiet:
    print(f"[rlmcp-play] wrote {out} ({len(frames)} frames @ {fps} fps)", flush=True)
  return {
      "video_path": str(out),
      "num_frames": len(frames),
      "fps": fps,
      "seconds": round(len(frames) / max(fps, 1), 2),
      "size_mb": round(out.stat().st_size / 1e6, 2),
      "metrics": _headline(env),
  }


def _default_out(
    checkpoint: Path, session_dir: Optional[Path], cfg: PlayConfig
) -> Path:
  """Beside the run's other evidence, named for what it shows."""
  stem = f"play_{checkpoint.stem}"
  if cfg.stage:
    stem += f"_{cfg.stage}"
  if session_dir is not None:
    return session_dir / "artifacts" / f"{stem}.mp4"
  return checkpoint.parent / f"{stem}.mp4"


def _headline(env: Any) -> Dict[str, float]:
  """Whatever the run publishes about itself, measured over the clip.

  Numbers beside a video stop it being a vibe. These are the same metrics the
  training run reported, so a clip is comparable with the curves.
  """
  try:
    metrics = env.rlmcp.telemetry.get_latest_metrics()
  except Exception:
    return {}
  return {
      key: round(float(value), 5)
      for key, value in (metrics or {}).items()
      if isinstance(value, (int, float)) and key.startswith("rlmcp/")
  }


def _view(cfg: PlayConfig, env: Any, vec_env: Any, policy: Any) -> Dict[str, Any]:
  """Hand the environment to one of the backend's interactive viewers.

  Both block until the window (or the browser tab) is closed, which is the
  point: this is a person looking at a robot. Neither is a hard dependency, so
  an install without them is told what it has rather than shown a traceback.
  """
  try:
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
  except ImportError as exc:
    raise PlayError(
        f"This install has no interactive viewers ({exc}). --mode video still "
        "works and needs nothing but an offscreen renderer."
    ) from exc

  if not cfg.quiet:
    print(
        f"[rlmcp-play] steer it from another shell:\n"
        f"[rlmcp-play]   rlmcp --session {env.rlmcp.session.dir} status",
        flush=True,
    )
  viewer_cls = NativeMujocoViewer if cfg.mode == "native" else ViserPlayViewer
  try:
    viewer = viewer_cls(vec_env, policy)
  except ImportError as exc:
    # viser in particular is an optional install of its own, and only says so
    # when the viewer is actually built.
    raise PlayError(
        f"--mode {cfg.mode} needs a package this environment does not have "
        f"({exc}). --mode video still works."
    ) from exc
  viewer.run()
  return {"viewer": cfg.mode, "closed": True}


# Command line.


def add_arguments(parser: Any) -> Any:
  """The play options. Attached to ``rlmcp play`` by :mod:`rlmcp.cli`."""
  parser.add_argument(
      "checkpoint", nargs="?", default="",
      help="Checkpoint .pt, or a run directory to take the latest one from",
  )
  parser.add_argument("--mode", default="video", choices=list(MODES),
                      help="video: render an mp4. native/viser: open a viewer.")
  parser.add_argument("--seconds", type=float, default=8.0)
  parser.add_argument("--out", default="", help="Video path (video mode)")
  parser.add_argument("--fps", type=int, default=None)
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--device", default="cuda:0",
                      help="'cpu' plays without touching a GPU that is training")
  parser.add_argument("--task", default="",
                      help="Defaults to the task the session recorded")
  parser.add_argument(
      "--stage", default="",
      help="Curriculum stage to restore; defaults to the last one the run entered",
  )
  parser.add_argument(
      "--no-replay", dest="replay", action="store_false",
      help="Do not restore trained-under conditions. The clip then shows the "
           "task's own play configuration, which for a curriculum run is rung "
           "zero -- a late policy will look broken on it.",
  )
  parser.add_argument(
      "--set", action="append", default=[], metavar="KEY=VALUE",
      help="Parameter override applied after the replay, so it wins. Repeatable.",
  )
  parser.add_argument("--task-package", action="append", default=[], metavar="MODULE",
                      help="Import this module before reading the task registry, "
                           "so a project's own tasks register. Repeatable.")
  parser.add_argument("--render-width", type=int, default=960)
  parser.add_argument("--render-height", type=int, default=720)
  parser.add_argument("--extra-envs", type=int, default=0,
                      help="Neighbouring envs composited into the frame")
  parser.add_argument("--session-dir", default="",
                      help="Where the play session publishes itself")
  parser.add_argument(
      "--allow-partial", action="store_true",
      help="Render even if some trained-under conditions could not be "
           "restored. The result is a policy shown doing a task it was not "
           "trained on, which reads as a bad policy -- so this is off.",
  )
  return parser


def config_from_args(args: Any, overrides: Dict[str, Any]) -> PlayConfig:
  return PlayConfig(
      checkpoint=args.checkpoint,
      task=args.task,
      mode=args.mode,
      seconds=args.seconds,
      num_envs=args.num_envs,
      device=args.device,
      out=args.out,
      fps=args.fps,
      stage=args.stage,
      replay=args.replay,
      overrides=overrides,
      task_package=list(args.task_package or []),
      render_width=args.render_width,
      render_height=args.render_height,
      extra_envs=args.extra_envs,
      session_dir=args.session_dir,
      allow_partial=args.allow_partial,
  )


__all__ = [
    "MODES",
    "PlayConfig",
    "PlayError",
    "add_arguments",
    "config_from_args",
    "find_checkpoint",
    "run_play",
    "session_for",
    "task_for",
]
