"""Go1 flat locomotion on mjlab, steerable from another shell.

    python examples/go1_flat/train_mjlab.py --num-envs 4096 --max-iterations 300

mjlab ships this task, so there is nothing to configure -- the whole example is
`gym`-style construction plus the one rlmcp line. Compare with
`train_genesis.py`, which needs an environment because Genesis ships a Go2 and
no Go1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

TASK = "Mjlab-Velocity-Flat-Unitree-Go1"


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--max-iterations", type=int, default=300)
  parser.add_argument("--log-dir", type=str, default="logs/go1-flat-mjlab")
  parser.add_argument("--device", type=str, default="cuda:0")
  args = parser.parse_args()

  import mjlab.tasks  # noqa: F401  (registers the tasks)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.torch import configure_torch_backends
  from rsl_rl.runners import OnPolicyRunner

  import rlmcp

  configure_torch_backends()
  log_dir = Path(args.log_dir)

  env_cfg = load_env_cfg(TASK)
  agent_cfg = load_rl_cfg(TASK)
  env_cfg.scene.num_envs = args.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)

  # The one line.
  env = rlmcp.wrap(env, session_dir=log_dir / "rlmcp", task_id=TASK)

  vec_env = RslRlVecEnvWrapper(env)
  runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), str(log_dir),
                          device=args.device)
  env.attach_runner(runner)

  try:
    runner.learn(num_learning_iterations=args.max_iterations)
  except rlmcp.TrainingStopped as stop:
    print(f"[rlmcp] stopped on request: {stop}", flush=True)


if __name__ == "__main__":
  main()
