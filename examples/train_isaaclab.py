"""Train an IsaacLab task under rlmcp supervision.

IsaacLab launches its own app before anything else can be imported, so this is
its train script with one line added:

    env = rlmcp.adapters.isaaclab.wrap(env, session_dir=log_dir / "rlmcp", ...)

Run it with the interpreter your IsaacLab install uses::

    python examples/train_isaaclab.py --task Isaac-Cartpole-v0 \
        --num_envs 64 --max_iterations 50 --headless

Then, from another shell::

    rlmcp status --session <log_dir>/rlmcp
    rlmcp params --contains reward
    rlmcp set reward.alive.weight 2.0 --why "it is dying too cheaply"

Frames (`shot`, `video`, progress clips) additionally need the app launched
with ``--enable_cameras``; everything else works without them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", required=True, help="Registered IsaacLab task id")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--log-root", default="logs/rlmcp_isaaclab")
parser.add_argument("--record-run", default="", help="Record id this run fills")
parser.add_argument("--records-root", default="")
parser.add_argument("--video-every", default=None,
                    help="Progress-clip cadence; needs --enable_cameras")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# The app must exist before isaaclab.envs and the task registry can be imported.
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import rlmcp.adapters.isaaclab as rlmcp_isaaclab
from rlmcp.adapters.isaaclab import TrainingStopped


def _migrate_agent_cfg(agent_cfg):
  """Bring a task's agent config in line with the installed rsl_rl.

  Older IsaacLab releases have no migration to do and no function to do it
  with, so its absence is not an error.
  """
  try:
    from importlib import metadata

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
  except ImportError:
    return agent_cfg
  return handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))


def main() -> int:
  env_cfg = parse_env_cfg(args.task, device=args.device,
                          num_envs=args.num_envs or 1)
  agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
  # IsaacLab ships one agent config per task and migrates it to whichever
  # rsl_rl is installed. Its own train script does this; skipping it is how you
  # get `MLPModel.__init__() got an unexpected keyword argument 'stochastic'`.
  agent_cfg = _migrate_agent_cfg(agent_cfg)
  if args.max_iterations:
    agent_cfg.max_iterations = args.max_iterations
  if args.seed is not None:
    agent_cfg.seed = args.seed
    env_cfg.seed = agent_cfg.seed

  log_dir = Path(args.log_root).resolve() / args.task
  log_dir.mkdir(parents=True, exist_ok=True)

  # render_mode follows the app: asking for rgb_array without cameras is how
  # you get a crash at the first frame instead of a run with no frames.
  render_mode = "rgb_array" if getattr(args, "enable_cameras", False) else None
  if render_mode:
    # rlmcp reads eye and lookat the way IsaacLab reads them for a per-env
    # origin: as an offset from the robot a frame is following. The stock
    # (7.5, 7.5, 7.5) is set to take in the whole grid, which leaves one robot
    # a speck; a picture of a gait wants to be closer than that.
    env_cfg.viewer.eye = (2.2, 2.2, 1.2)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.3)
  env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)

  env = rlmcp_isaaclab.wrap(
      env,
      session_dir=log_dir / "rlmcp",
      task_id=args.task,
      service_every_steps=agent_cfg.num_steps_per_env,
      record_run=args.record_run or None,
      records_root=args.records_root or None,
      video_every=args.video_every if render_mode else 0,
  )

  vec_env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
  runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=str(log_dir),
                          device=agent_cfg.device)
  env.attach_runner(runner)

  reason = "completed"
  try:
    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=True)
  except TrainingStopped as stopped:
    reason = str(stopped)
    print(f"[rlmcp] stopping: {reason}")
  except KeyboardInterrupt:
    reason = "interrupted"

  final = log_dir / f"model_final_{runner.current_learning_iteration}.pt"
  try:
    runner.save(str(final))
  except Exception as exc:
    print(f"[rlmcp] final save failed: {exc}")
  env.rlmcp.session.append_event("train_exit", {"reason": reason,
                                                "final_checkpoint": str(final)})
  env.close()
  return 0


if __name__ == "__main__":
  code = main()
  simulation_app.close()
  sys.exit(code)
