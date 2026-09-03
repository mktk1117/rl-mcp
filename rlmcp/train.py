"""``rlmcp-train``: run any registered mjlab task under rlmcp supervision.

This is a convenience entrypoint, not a required one -- the library way is to
call :func:`rlmcp.wrap` from your own training script (see
``examples/train_g1_rough_curriculum.py``). It exists so the whole system can be
exercised with a single command::

    rlmcp-train Mjlab-Velocity-Rough-Unitree-G1 --num-envs 2048 --curriculum terrain

While it runs, drive it from another shell::

    rlmcp status
    rlmcp view --on                       # watch it in a browser, live
    rlmcp diagnose --seconds 4 --terrain pyramid_stairs
    rlmcp set reward.action_rate_l2.weight -0.2 --why "knees buzzing"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _prog_name(subcommand: str) -> str:
  """Name this the way it was actually invoked.

  The same code is reachable as `rlmcp train` and as the `rlmcp-train` console
  script, and a usage line that names the other one is confusing exactly when
  the reader is looking up how to type it.
  """
  invoked = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else ""
  return invoked if invoked.startswith("rlmcp-") else subcommand


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(prog=_prog_name("rlmcp train"),
                                   description=__doc__.splitlines()[0])
  parser.add_argument("task", nargs="?", default="",
                      help="Registered mjlab task id (taken from --recipe when given)")
  parser.add_argument(
      "--recipe", default="", metavar="DIR",
      help="A directory written by `rlmcp recipe build`. Its recipe.json fills "
           "in the task, --task-package, --config-json, --curriculum-json, "
           "--seed, --num-envs, --max-iterations and --code-root; any of them "
           "given here wins. Without --record-run, opens the record for the "
           "rerun as recipe-<run>.",
  )
  parser.add_argument("--num-envs", type=int, default=None)
  parser.add_argument("--max-iterations", type=int, default=None)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--log-root", default="logs/rsl_rl")
  parser.add_argument("--run-name", default="")
  parser.add_argument(
      "--curriculum", default="terrain", choices=["terrain", "none"],
      help="'terrain' unlocks terrain groups flat-first; 'none' keeps the task's own config",
  )
  parser.add_argument(
      "--curriculum-json", default="", metavar="PATH",
      help="A ladder written by `rlmcp recipe build` (curriculum.json), or any "
           "StageSchedule.to_dict() saved as JSON. Starts it from its first rung. "
           "Overrides --curriculum.",
  )
  parser.add_argument(
      "--config-json", default="", metavar="PATH",
      help="Parameter values to set before the first batch, as {key: value} -- "
           "a recipe's config.json. Keys the task does not have are reported, "
           "not fatal.",
  )
  parser.add_argument("--stage-min-iterations", type=int, default=150)
  parser.add_argument("--stage-hold-iterations", type=int, default=20)
  parser.add_argument("--no-auto-promote", action="store_true",
                      help="Set up the stages but require an agent to advance them")
  parser.add_argument("--render-width", type=int, default=640)
  parser.add_argument("--render-height", type=int, default=480)
  parser.add_argument("--eager-render", action="store_true",
                      help="Build the offscreen renderer at startup instead of on "
                           "the first screenshot. Costs GPU memory for the whole "
                           "run; only needed if something outside rlmcp calls "
                           "env.render() before rlmcp does.")
  parser.add_argument(
      "--video-every", default=None, metavar="CADENCE",
      help="How often the run films itself, starting at iteration 0, with "
           "every clip attached to the run record. Default 'double': clips at "
           "0, 50, 100, 200, 400 ... each gap twice the last, never more than "
           "2000 apart. Also takes 'double:<first>:<cap>' to move the first "
           "gap and the cap, or a flat interval like '200'. Pass 'off' for no "
           "clips at all ('none', 'never' and '0' mean the same).",
  )
  parser.add_argument("--video-seconds", type=float, default=4.0,
                      help="Length of each progress clip")
  parser.add_argument("--video-budget-mb", type=float, default=None,
                      metavar="MB",
                      help="Stop taking progress clips once they have used "
                           "this much disk (default 200; 0 for no limit)")
  parser.add_argument(
      "--viser", dest="viser", action="store_true", default=True,
      help="Serve a live 3-D view of the run in a browser, over viser. On by "
           "default: it needs no renderer and costs nothing at all while no "
           "browser is open or the view is paused, so the only thing an "
           "unwatched one uses is a port.",
  )
  parser.add_argument(
      "--no-viser", dest="viser", action="store_false",
      help="Do not serve a live view, and do not bind a port for one. It can "
           "still be attached to the run later with `rlmcp view --on`.",
  )
  parser.add_argument("--viser-port", type=int, default=None, metavar="PORT",
                      help="First port to try for the live view (default 8740; "
                           "busy ports are skipped and the one taken is "
                           "reported in `rlmcp status`)")
  parser.add_argument("--viser-fps", type=float, default=None, metavar="HZ",
                      help="Frames per second the live view pushes while "
                           "somebody is watching (default 20)")
  parser.add_argument("--viser-env-id", type=int, default=0,
                      help="Which environment the live view shows")
  parser.add_argument(
      "--viser-realtime", action="store_true",
      help="Play the live view back at the speed the robot actually moves: "
           "the run records a few seconds of itself and the tab plays that "
           "window at 1x, with pause, single-step and a speed control. "
           "Without it the view shows the current step, at the run's pace.",
  )
  parser.add_argument("--viser-buffer-seconds", type=float, default=None,
                      metavar="S",
                      help="Sim time one realtime window holds (default 4)")
  parser.add_argument("--save-interval", type=int, default=None)
  parser.add_argument("--resume", default="", help="Checkpoint path to resume from")
  parser.add_argument(
      "--motion-file", default="",
      help="Reference motion .npz, required by tracking tasks",
  )
  parser.add_argument(
      "--task-package", action="append", default=[], metavar="MODULE",
      help="Import this module before reading the registry, so a project's own "
           "tasks register (e.g. --task-package smp.rl.tasks). Repeatable.",
  )
  parser.add_argument("--record-run", default="",
                      help="Record id this run is testing (see `rlmcp record new`)")
  parser.add_argument("--records-root", default="", help="Records directory")
  parser.add_argument("--record-slot", default="", help="Resource slot to claim")
  parser.add_argument("--record-strict", action="store_true",
                      help="Refuse to launch without a registered record")
  parser.add_argument(
      "--code-root", default=None, metavar="DIR",
      help="Task package to stamp on the record at launch (default: the "
           "directory you launched from). Pass '' to record no code snapshot.",
  )
  parser.add_argument("--logger", default="tensorboard",
                      choices=["tensorboard", "wandb", "neptune"])
  return parser.parse_args(argv)


def load_curriculum_json(path: str) -> Any:
  """The ladder in ``path`` as a fresh :class:`StageSchedule`, or None.

  Fresh on purpose: a file saved from a live schedule carries the rung it was
  on and the history behind it, and a new run starts on the first rung with
  none of that.
  """
  if not path:
    return None
  from rlmcp.core.curriculum import StageSchedule

  raw = json.loads(Path(path).read_text())
  if isinstance(raw, list):
    raw = {"stages": raw}
  if not isinstance(raw, dict) or not raw.get("stages"):
    raise ValueError(f"--curriculum-json {path}: no 'stages' in it.")
  return StageSchedule.from_dict({"stages": raw["stages"],
                                  "auto_promote": raw.get("auto_promote", True)})


def load_config_json(path: str) -> dict[str, Any]:
  """``{key: value}`` from ``path``, or ``{}`` when no path was given."""
  if not path:
    return {}
  raw = json.loads(Path(path).read_text())
  if not isinstance(raw, dict):
    raise ValueError(f"--config-json {path}: expected a JSON object of key: value.")
  return dict(raw)


def apply_recipe(args: argparse.Namespace) -> dict[str, Any] | None:
  """Fill the parser's blanks from ``--recipe``, and open its record.

  Explicit flags win over the manifest, so a rerun with more envs or a
  different device is one flag away. The recipe's ``package/`` goes first on
  ``sys.path`` and becomes the code root, so the task package that trains is
  the one the original run had, not whichever one is installed.
  """
  if not args.recipe:
    return None
  from rlmcp.records.recipe import load_manifest

  manifest = load_manifest(args.recipe)
  where = Path(manifest["dir"])
  args.task = args.task or manifest["task"]
  for package in manifest.get("task_packages") or []:
    if package not in args.task_package:
      args.task_package.append(package)
  if not args.config_json and (where / manifest.get("config", "config.json")).exists():
    args.config_json = str(where / manifest.get("config", "config.json"))
  if not args.curriculum_json and (
      where / manifest.get("curriculum", "curriculum.json")).exists():
    args.curriculum_json = str(where / manifest.get("curriculum", "curriculum.json"))
  if args.seed is None and manifest.get("seed") is not None:
    args.seed = int(manifest["seed"])
  if args.num_envs is None and manifest.get("num_envs"):
    args.num_envs = int(manifest["num_envs"])
  if args.max_iterations is None and manifest.get("iterations"):
    args.max_iterations = int(manifest["iterations"])
  package = manifest.get("package")
  if package and (where / package).is_dir():
    sys.path.insert(0, str(where / package))
    if args.code_root is None:
      args.code_root = str(where / package)
  if not args.record_run:
    from rlmcp.records import open_store
    from rlmcp.records.recipe import open_reproduction_record

    store = open_store(args.records_root or None)
    record = open_reproduction_record(store, manifest, where)
    args.record_run = record.id
    print(f"[rlmcp-train] opened record {record.id} for the rerun of "
          f"{manifest.get('from_run')} (records: {getattr(store, 'root', '')})")
  return manifest


def main(argv: list[str] | None = None) -> int:
  args = _parse_args(argv)
  try:
    apply_recipe(args)
  except (OSError, ValueError, KeyError, TypeError) as exc:
    print(f"[rlmcp-train] --recipe: {exc}")
    return 2
  if not args.task:
    print("[rlmcp-train] name a task, or pass --recipe <dir>.")
    return 2

  # Checked before anything expensive starts: a mistyped cadence should cost a
  # line, not a traceback out of the middle of environment construction.
  from rlmcp.core.progress_video import Cadence, CadenceError

  try:
    Cadence.parse(args.video_every)
  except CadenceError as exc:
    print(f"[rlmcp-train] --video-every: {exc}")
    return 2

  try:
    schedule = load_curriculum_json(args.curriculum_json)
    launch_config = load_config_json(args.config_json)
  except (OSError, ValueError, KeyError, TypeError) as exc:
    print(f"[rlmcp-train] {exc}")
    return 2

  # Must precede the first mujoco import so the GL backend is picked correctly.
  os.environ.setdefault("MUJOCO_GL", "egl")
  if args.device.startswith("cuda"):
    device_index = args.device.split(":")[-1] if ":" in args.device else "0"
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", device_index)

  import importlib

  import mjlab.tasks  # noqa: F401  (populates the task registry)

  # A project's tasks register on import, exactly as mjlab's own do.
  for module in args.task_package:
    try:
      importlib.import_module(module)
    except ImportError as exc:
      print(f"[rlmcp-train] could not import task package '{module}': {exc}")
      return 2
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends

  import rlmcp
  from rlmcp.adapters.mjlab import TrainingStopped

  if args.task not in list_tasks():
    print(f"Unknown task '{args.task}'.\nAvailable:\n  " + "\n  ".join(list_tasks()))
    return 2

  env_cfg = load_env_cfg(args.task)
  agent_cfg = load_rl_cfg(args.task)

  if args.num_envs:
    env_cfg.scene.num_envs = args.num_envs
  if args.max_iterations:
    agent_cfg.max_iterations = args.max_iterations
  if args.save_interval:
    agent_cfg.save_interval = args.save_interval
  if args.seed is not None:
    agent_cfg.seed = args.seed
  env_cfg.seed = agent_cfg.seed
  agent_cfg.logger = args.logger
  env_cfg.viewer.width = args.render_width
  env_cfg.viewer.height = args.render_height

  # Tracking tasks need a reference motion. Say so plainly rather than failing
  # later inside the motion loader with an empty-path error.
  motion_cmd = (env_cfg.commands or {}).get("motion")
  if motion_cmd is not None and hasattr(motion_cmd, "motion_file"):
    if args.motion_file:
      motion_cmd.motion_file = args.motion_file
    if not motion_cmd.motion_file or not Path(motion_cmd.motion_file).exists():
      print(
          f"[rlmcp-train] '{args.task}' is a motion tracking task and needs a "
          "reference motion. Pass --motion-file /path/to/motion.npz "
          "(or fetch one from your W&B registry with mjlab's own train script)."
      )
      return 2

  configure_torch_backends()

  log_root = (Path(args.log_root) / agent_cfg.experiment_name).resolve()
  # Local time deliberately: this becomes a directory name somebody has to
  # find in a listing.
  run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ005
  if args.run_name:
    run_dir += f"_{args.run_name}"
  log_dir = log_root / run_dir
  log_dir.mkdir(parents=True, exist_ok=True)
  (log_dir / "params").mkdir(exist_ok=True)
  (log_dir / "params" / "env.yaml").write_text(json.dumps(asdict(env_cfg), indent=2, default=str))

  print(f"[rlmcp-train] task={args.task} envs={env_cfg.scene.num_envs} device={args.device}")
  print(f"[rlmcp-train] logs: {log_dir}")

  # render_mode stays None: rlmcp creates the renderer on the first screenshot,
  # so a run that never takes one pays no GPU memory for the option.
  env = ManagerBasedRlEnv(
      cfg=env_cfg,
      device=args.device,
      render_mode="rgb_array" if args.eager_render else None,
  )

  if schedule is not None:
    (log_dir / "params" / "curriculum.json").write_text(
        json.dumps([s.to_dict() for s in schedule.stages], indent=2))

  env = rlmcp.wrap(
      env,
      session_dir=log_dir / "rlmcp",
      curriculum=schedule if schedule is not None
      else None if args.curriculum == "none" else "terrain",
      task_id=args.task,
      task_packages=list(args.task_package),
      seed=agent_cfg.seed,
      parameters=launch_config,
      service_every_steps=agent_cfg.num_steps_per_env,
      record_run=args.record_run or None,
      records_root=args.records_root or None,
      record_slot=args.record_slot,
      record_strict=args.record_strict,
      code_root=args.code_root,
      video_every=args.video_every,
      video_seconds=args.video_seconds,
      video_budget_mb=args.video_budget_mb,
      viser=args.viser,
      viser_port=args.viser_port,
      viser_fps=args.viser_fps,
      viser_env_id=args.viser_env_id,
      viser_realtime=args.viser_realtime,
      viser_buffer_seconds=args.viser_buffer_seconds,
      curriculum_kwargs={
          "min_iterations": args.stage_min_iterations,
          "hold_iterations": args.stage_hold_iterations,
          "auto_promote": not args.no_auto_promote,
      },
  )

  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(vec_env, asdict(agent_cfg), str(log_dir), args.device)
  if args.resume:
    print(f"[rlmcp-train] resuming from {args.resume}")
    runner.load(args.resume)

  env.attach_runner(runner)

  stop_reason = "completed"
  try:
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
    )
  except TrainingStopped as stopped:
    stop_reason = str(stopped)
    print(f"[rlmcp-train] stopping: {stop_reason}")
  except KeyboardInterrupt:
    stop_reason = "interrupted"
    print("[rlmcp-train] interrupted")

  final = log_dir / f"model_final_{runner.current_learning_iteration}.pt"
  try:
    runner.save(str(final))
    print(f"[rlmcp-train] saved {final}")
  except Exception as exc:
    print(f"[rlmcp-train] final save failed: {exc}")

  env.rlmcp.session.append_event("train_exit", {"reason": stop_reason,
                                                 "final_checkpoint": str(final)})
  vec_env.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
