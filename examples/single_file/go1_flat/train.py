"""Train the single-file Go1 task, on whichever backend you have.

    python examples/single_file/go1_flat/train.py --backend mjwarp --num-envs 4096
    python examples/single_file/go1_flat/train.py --backend mjbatch --num-envs 1024 --device cpu
    python examples/single_file/go1_flat/train.py --backend genesis --num-envs 2048

Then, from a second shell, the same commands as on any other backend::

    rlmcp status
    rlmcp params --contains reward
    rlmcp set reward.action_rate.weight -0.2 --why "knees chattering"
    rlmcp video --seconds 4

The training loop is written out here rather than hidden in a runner, which
is the whole idea: what happens per iteration is on this page. rlmcp needs
two lines of it -- ``attach_algorithm`` for knobs and checkpoints, and
``service`` at the iteration boundary, where edits land between rollouts.

Set ``MJLAB_GO1_XML`` to mjlab's ``go1.xml`` if mjlab is not importable here,
and ``MUJOCO_GL=glfw`` (or ``egl``) for frames from the MuJoCo backends.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

import rlmcp.adapters.single_file as rlmcp_single_file

# env.py and ppo.py live next to this script, not on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import EnvConfig, Go1FlatEnv
from ppo import PPO, Actor, Critic, ModelConfig, PPOConfig, RolloutStorage


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--backend", default="mjwarp", choices=("mjwarp", "mjbatch", "genesis"))
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--max-iterations", type=int, default=1500)
  parser.add_argument("--steps-per-env", type=int, default=24)
  parser.add_argument("--save-every", type=int, default=100)
  parser.add_argument("--log-every", type=int, default=10)
  parser.add_argument("--log-dir", default="logs/go1-flat-single-file")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--video-every", default=None,
                      help="progress-clip cadence for rlmcp: a flat interval like 200, "
                           "'double' (0, 50, 100, 200, ... -- the default), or 'off'")
  parser.add_argument("--no-viser", action="store_true", help="do not serve the live view")
  args = parser.parse_args()

  torch.manual_seed(args.seed)
  device = torch.device(args.device)
  log_dir = Path(args.log_dir) / args.backend
  log_dir.mkdir(parents=True, exist_ok=True)

  cfg = EnvConfig(backend=args.backend, num_envs=args.num_envs, device=args.device)
  env = Go1FlatEnv(cfg)
  env = rlmcp_single_file.wrap(
      env, session_dir=log_dir / "rlmcp", task_id=f"go1-flat-{args.backend}",
      seed=args.seed, viser=False if args.no_viser else None,
      **({} if args.video_every is None else {"video_every": args.video_every}),
  )

  actor = Actor(env.obs_dim, env.action_dim, ModelConfig()).to(device)
  critic = Critic(env.obs_dim, ModelConfig()).to(device)
  storage = RolloutStorage(env.num_envs, args.steps_per_env, env.obs_dim, env.action_dim, device)
  ppo = PPO(actor, critic, storage, PPOConfig())
  env.attach_algorithm(ppo, log_dir=str(log_dir))

  obs = env.reset()
  total_steps, start = 0, time.time()
  episode_rewards: list[float] = []
  episode_lengths: list[float] = []
  print(f"training {args.max_iterations} iterations of {args.num_envs} envs on {args.backend}; "
        f"logs in {log_dir}", flush=True)

  try:
    for iteration in range(1, args.max_iterations + 1):
      ppo.train_mode()
      t0 = time.time()
      with torch.inference_mode():
        for _ in range(args.steps_per_env):
          actions = ppo.act(obs)
          obs, rewards, dones, info = env.step(actions)
          ppo.process_env_step(rewards, dones, info.get("time_outs"))
          total_steps += env.num_envs
          episode_rewards += info["episode_rewards"].tolist()
          episode_lengths += info["episode_lengths"].tolist()
        ppo.compute_returns(obs)
      collect = time.time() - t0

      t0 = time.time()
      losses = ppo.update()
      learn = time.time() - t0

      episode_rewards, episode_lengths = episode_rewards[-200:], episode_lengths[-200:]
      metrics = {f"losses/{k}": v for k, v in losses.items()}
      if episode_rewards:
        metrics["Train/mean_reward"] = sum(episode_rewards) / len(episode_rewards)
        metrics["Train/mean_episode_length"] = sum(episode_lengths) / len(episode_lengths)
      metrics["perf/fps"] = env.num_envs * args.steps_per_env / max(1e-6, collect + learn)

      # The iteration boundary: rlmcp applies edits, answers commands, may stop us.
      env.service(iteration, metrics=metrics)

      if iteration % args.log_every == 0:
        reward = metrics.get("Train/mean_reward", float("nan"))
        length = metrics.get("Train/mean_episode_length", float("nan"))
        print(f"it {iteration:5d} | reward {reward:8.3f} | ep len {length:6.1f} | "
              f"fps {metrics['perf/fps']:8.0f} | lr {losses['learning_rate']:.1e} | "
              f"{' '.join(f'{k}={v:.3f}' for k, v in info['reward_terms'].items())}",
              flush=True)
      if iteration % args.save_every == 0:
        torch.save({**ppo.save(), "iteration": iteration}, log_dir / f"model_{iteration}.pt")
        torch.save({**ppo.save(), "iteration": iteration}, log_dir / "model_latest.pt")
  except rlmcp_single_file.TrainingStopped as stop:
    print(f"[rlmcp] stopped on request: {stop}", flush=True)
    torch.save({**ppo.save(), "iteration": iteration}, log_dir / "model_latest.pt")
  finally:
    env.close()

  print(f"done: {total_steps:,} env steps in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
  main()
