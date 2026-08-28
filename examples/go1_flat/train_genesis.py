"""Go1 flat locomotion on Genesis, steerable from another shell.

    python examples/go1_flat/train_genesis.py --num-envs 4096 --max-iterations 300

Then, from a second shell, the same commands you would use on any backend:

    rlmcp status
    rlmcp set action.scale 0.20 --why "calves buzzing at 9 Hz"
    rlmcp diagnose --seconds 2

Genesis's own `examples/locomotion/go2_train.py` must be importable -- it holds
the PPO config this shares. Point at it with $GENESIS_LOCOMOTION, or run from a
directory that has it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if os.environ.get("GENESIS_LOCOMOTION"):
  # Appended, not prepended: Genesis's locomotion directory is only here to
  # supply go2_train's PPO config, and putting it first lets any same-named
  # module there shadow this example's own. That silently picked up the wrong
  # get_cfgs the first time it was run.
  sys.path.append(os.environ["GENESIS_LOCOMOTION"])

import genesis as gs  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import rlmcp.adapters.genesis as rlmcp_genesis  # noqa: E402
from genesis_env import QuadrupedEnv  # noqa: E402
from go1_cfg import get_cfgs  # noqa: E402


def add_camera_before_build(rendered_envs):
  """Genesis cameras must exist before `scene.build()`, and the environment
  builds inside `__init__` -- so the camera goes in by patching `build` rather
  than by editing the environment. Without one, everything works except `shot`,
  `video` and progress clips, and rlmcp says so at wrap time."""
  original = gs.Scene.build

  def build(self, *args, **kwargs):
    self.add_camera(res=(640, 480), pos=(2.5, 1.5, 1.6),
                    lookat=(0.0, 0.0, 0.35), GUI=False, debug=True)
    context = getattr(getattr(self, "_visualizer", None), "_context", None)
    if context is not None and hasattr(context, "rendered_envs_idx"):
      context.rendered_envs_idx = list(rendered_envs)
    return original(self, *args, **kwargs)

  gs.Scene.build = build


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--max-iterations", type=int, default=300)
  parser.add_argument("--log-dir", type=str, default="logs/go1-flat")
  parser.add_argument("--rendered-envs", type=int, nargs="*", default=[0])
  parser.add_argument("--action-scale", type=float, default=None,
                      help="0.20 is the tuned value; see README.md. Default is "
                           "Genesis's stock 0.25.")
  parser.add_argument("--action-rate-weight", type=float, default=None)
  args = parser.parse_args()

  from go2_train import get_train_cfg

  env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs(
      action_scale=args.action_scale,
      action_rate_weight=args.action_rate_weight,
  )
  train_cfg = get_train_cfg("go1-flat")
  log_dir = Path(args.log_dir)

  gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=1)
  add_camera_before_build(args.rendered_envs)

  env = QuadrupedEnv(num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                     reward_cfg=reward_cfg, command_cfg=command_cfg)

  # The one line.
  env = rlmcp_genesis.wrap(env, session_dir=log_dir / "rlmcp", task_id="go1-flat")

  runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
  env.attach_runner(runner)

  try:
    runner.learn(num_learning_iterations=args.max_iterations,
                 init_at_random_ep_len=True)
  except rlmcp_genesis.TrainingStopped as stop:
    print(f"[rlmcp] stopped on request: {stop}", flush=True)


if __name__ == "__main__":
  main()
