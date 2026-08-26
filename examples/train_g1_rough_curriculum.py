"""G1 rough-terrain locomotion, trained flat-first under rlmcp.

This is the library usage pattern: take the training script you already have,
wrap the environment, hand rlmcp the runner. Everything else is stock mjlab.

Run it::

    python examples/train_g1_rough_curriculum.py --num-envs 4096

Then, from another shell (or from an LLM agent over MCP), watch and steer it::

    rlmcp status
    rlmcp curriculum                       # which terrains are unlocked
    rlmcp diagnose --seconds 4             # is the gait smooth? is it tracking?
    rlmcp shot --where terrain=pyramid_stairs   # a robot on the stairs
    rlmcp set reward.action_rate_l2.weight -0.25 --why "ankles chattering"
    rlmcp curriculum advance --why "flat is solved, unlock rough"

The curriculum below is written out explicitly rather than using
``curriculum="terrain"`` so it is clear what a stage actually controls. A stage
speaks in two vocabularies only -- parameter edits, and commands -- so the same
structure drives a manipulation task; the ``set_terrain`` command here comes
from the terrain extension, which loads only because this scene has a terrain
grid.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


def build_curriculum():
  """Flat first, then rough ground, then slopes, then stairs.

  Penalties that punish a policy which has not yet learned to walk (foot
  clearance, slip) start soft and tighten once the gait exists. Promotion needs
  both survival (episodes running at least half their length) and progress
  (mjlab's own terrain-level curriculum pushing robots up the rows).
  """
  from rlmcp.extensions.terrain import METRIC_TERRAIN_LEVEL_FRAC
  from rlmcp.core.curriculum import (
      METRIC_EPISODE_LENGTH_FRAC,
      Action,
      Condition,
      CurriculumStage,
      StageSchedule,
  )

  def promote(episode_frac: float = 0.5, level_frac: float = 0.6):
    return [
        Condition(METRIC_EPISODE_LENGTH_FRAC, ">=", episode_frac),
        Condition(METRIC_TERRAIN_LEVEL_FRAC, ">=", level_frac),
    ]

  return StageSchedule(
      [
          CurriculumStage(
              name="0_flat",
              # `apply` runs any command this run accepts. `set_terrain` comes
              # from the terrain extension, which loads because this environment
              # has a terrain grid; on a task without one it simply would not.
              apply=[
                  Action("set_terrain", {"terrains": ["flat"], "max_level": 2}),
                  Action("set_parameter", {
                      "key": "command.twist.ranges.lin_vel_x", "value": [-1.0, 1.0]}),
              ],
              parameters={
                  # Learn to stand and move before paying for foot styling.
                  "reward.foot_clearance.weight": -0.5,
                  "reward.foot_slip.weight": -0.05,
              },
              promote_when=promote(episode_frac=0.55, level_frac=0.5),
              min_iterations=200,
              hold_iterations=20,
              notes="Flat ground only. Goal: a gait that survives a full episode.",
          ),
          CurriculumStage(
              name="1_rough",
              apply=[
                  Action("set_terrain", {
                      "terrains": ["flat", "random_rough", "wave_terrain"],
                      "weights": {"flat": 1.0, "random_rough": 1.5,
                                  "wave_terrain": 1.0},
                      "max_level": 5,
                  })
              ],
              parameters={
                  "reward.foot_clearance.weight": -1.0,
                  "reward.foot_slip.weight": -0.1,
              },
              promote_when=promote(),
              min_iterations=250,
              hold_iterations=20,
              notes="Uneven ground. Goal: keep tracking velocity when the floor moves.",
          ),
          CurriculumStage(
              name="2_slopes",
              apply=[
                  Action("set_terrain", {
                      "terrains": ["flat", "random_rough", "wave_terrain",
                                   "hf_pyramid_slope", "hf_pyramid_slope_inv"],
                      "weights": {"hf_pyramid_slope": 1.5,
                                  "hf_pyramid_slope_inv": 1.5},
                      "max_level": 8,
                  })
              ],
              parameters={"reward.foot_clearance.weight": -2.0},
              promote_when=promote(),
              min_iterations=300,
              hold_iterations=20,
              notes="Slopes. Goal: lean into gradients without stalling or sliding.",
          ),
          CurriculumStage(
              name="3_stairs",
              apply=[
                  Action("set_terrain", {
                      "terrains": None,  # Everything the scene has.
                      "weights": {"pyramid_stairs": 2.0,
                                  "pyramid_stairs_inv": 2.0},
                      "max_level": 10,
                  })
              ],
              parameters={
                  "reward.foot_clearance.weight": -2.0,
                  "reward.foot_swing_height.weight": -0.5,
              },
              min_iterations=400,
              notes="Full terrain set including stairs, at maximum difficulty.",
          ),
      ]
  )


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--max-iterations", type=int, default=8000)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--log-root", default="logs/rsl_rl")
  parser.add_argument("--task", default="Mjlab-Velocity-Rough-Unitree-G1")
  parser.add_argument(
      "--logger", default="tensorboard", choices=["tensorboard", "wandb", "neptune"],
      help="rsl_rl logging backend; wandb needs credentials in the environment",
  )
  args = parser.parse_args()

  # Set before mujoco is imported so the offscreen renderer picks a GL backend.
  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", args.device.split(":")[-1])

  import mjlab.tasks  # noqa: F401  (registers the tasks)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends

  import rlmcp
  from rlmcp.adapters.mjlab import TrainingStopped

  env_cfg = load_env_cfg(args.task)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg.max_iterations = args.max_iterations
  agent_cfg.logger = args.logger
  env_cfg.seed = agent_cfg.seed
  # Frames the agent will look at; also the size of recorded clips.
  env_cfg.viewer.width, env_cfg.viewer.height = 640, 480

  configure_torch_backends()
  log_dir = (
      Path(args.log_root) / agent_cfg.experiment_name
      / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_g1_rough")
  ).resolve()
  log_dir.mkdir(parents=True, exist_ok=True)

  # render_mode="rgb_array" is what makes screenshots and video possible.
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode="rgb_array")

  # --- the rlmcp part -------------------------------------------------------
  env = rlmcp.wrap(
      env,
      session_dir=log_dir / "rlmcp",
      curriculum=build_curriculum(),
      task_id=args.task,
      service_every_steps=agent_cfg.num_steps_per_env,
  )
  # ---------------------------------------------------------------------------

  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(vec_env, asdict(agent_cfg), str(log_dir), args.device)

  # Gives rlmcp the PPO knobs, checkpoint save/load, and exact iteration
  # boundaries to apply changes on.
  env.attach_runner(runner)

  try:
    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=True)
  except TrainingStopped as stopped:
    print(f"[example] stopped on request: {stopped}")

  runner.save(str(log_dir / f"model_final_{runner.current_learning_iteration}.pt"))
  vec_env.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
