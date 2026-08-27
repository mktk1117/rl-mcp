# Driving an IsaacLab run

rlmcp watches and steers IsaacLab the same way it does mjlab: one line in the
training script, then a live run you can query, tune and record from another
shell. This page is only what is different about IsaacLab.

## The one line

IsaacLab launches its own app before anything else can be imported, so the wrap
goes after `gym.make`:

```python
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args)          # IsaacLab's own launch, unchanged
simulation_app = app_launcher.app

import gymnasium as gym
import rlmcp.adapters.isaaclab as rlmcp_isaaclab
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")
env = rlmcp_isaaclab.wrap(env, session_dir=log_dir / "rlmcp", task_id=task,
                          service_every_steps=agent_cfg.num_steps_per_env)

vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=str(log_dir),
                        device=agent_cfg.device)
env.attach_runner(runner)                 # PPO knobs, checkpoints, iterations
runner.learn(num_learning_iterations=agent_cfg.max_iterations)
```

[`examples/train_isaaclab.py`](../examples/train_isaaclab.py) is that script,
runnable as it stands:

```bash
python examples/train_isaaclab.py --task Isaac-Cartpole-v0 \
    --num_envs 64 --max_iterations 200 --headless
```

Then, from another shell:

```bash
rlmcp --session <log_dir>/rlmcp status
rlmcp --session <log_dir>/rlmcp params --contains reward
rlmcp --session <log_dir>/rlmcp set reward.pole_pos.weight -2.0 --why "pole angle is cheap"
```

Nothing about the task has to be described to rlmcp: a stock `Isaac-Cartpole-v0`
comes up with 22 tunable parameters — reward weights, their function
parameters, termination bounds, the reset event ranges, the action scale —
found by walking the environment's own manager configs.

## Which Isaac Sim, and which driver

These have to agree, and when they do not the failure is a segfault with no
message — inside `librtx.scenedb.plugin.so`, at stage creation, before any
rlmcp code runs:

| Isaac Sim | validated Linux driver | Python |
| --- | --- | --- |
| 5.1.0 | 580.65.06 | 3.11 |
| 6.0.1 | 595.58.03 | 3.12 |

A driver from the R590 branch (595.x) crashes Isaac Sim 5.1's RTX renderer, so
cameras — and therefore `shot`, `video` and progress clips — are unavailable
there while everything else works headless. Isaac Sim 6.0.1 validates that
branch and renders normally. Verified on an RTX 3090 with driver 595.84: 5.1
crashes with cameras enabled and 21 GB of the card free; 6.0.1 does not.

Two install notes for a machine without root, both learned the hard way:
`./isaaclab.sh --install` shells out to `sudo apt-get`, so the source packages
go in with `uv pip install --no-build-isolation --editable source/<pkg>`
instead; and a dependency's `setup.py` imports `pkg_resources`, so the venv
needs `setuptools<81` first.

## Frames need cameras, and that is decided at launch

`shot`, `video` and [progress clips](tools.md#progress-clips-the-ones-you-do-not-have-to-ask-for)
render through the Kit app rather than through a renderer rlmcp can build on
demand, so they need **both**:

* the app launched with `--enable_cameras`, and
* the environment built with `render_mode="rgb_array"`.

A run without them says so at wrap time rather than at the first `shot`, and
everything else — parameters, metrics, traces, curricula, records — works
regardless.

**Which env a frame shows** takes more than `cfg.viewer.env_index`, and how
much more depends on the version. A run with a viewport has a
`viewport_camera_controller`, and the index does nothing there until the
camera's anchor moves off `origin_type="world"`, which is the default. A
headless IsaacLab 6 run has no controller at all: its frames come from
`env.video_recorder`, whose camera is placed once at construction and never
consults `cfg.viewer` again. rlmcp moves whichever of the two makes the frame,
onto the robot in the env you asked for, and puts it back afterwards. Without
that, `shot --env-id 7` and `shot --where …` answer with the same overview of
the whole grid every time — a picture that looks like an answer and is not one.

**`eye` and `lookat` are read as an offset** from the robot being followed,
which is how IsaacLab reads them for a per-env origin. The stock
`(7.5, 7.5, 7.5)` is framed to take in the whole grid and leaves a single robot
a speck, so a task you intend to watch is worth a closer pair:

```python
env_cfg.viewer.eye = (2.2, 2.2, 1.2)
env_cfg.viewer.lookat = (0.0, 0.0, 0.3)
```

## What lands where

| what you touch | where it is |
| --- | --- |
| a reward weight, a termination bound, an event range | `env.cfg.<manager>.<term>` — the object IsaacLab re-reads |
| the PPO knobs, checkpoints, `stop` | the rsl_rl runner, through `attach_runner` |
| "which env is env 7" for a frame | `cfg.viewer.env_index` **and** `origin_type`, both put back after each render |

## Known differences from mjlab

* **The robot is named.** IsaacLab keeps articulations in their own mapping, so
  rlmcp takes `robot` by convention, or the only articulation there is. A scene
  with two and no convention is asked about, not guessed:
  `wrap(robot_name="hand")`.
* **The iteration hook is on the runner.** mjlab's runner owns a logger object;
  plain rsl_rl logs from a method on the runner itself. rlmcp finds either, so
  commands are answered on iteration boundaries in both.
* **Terrain commands are absent.** The terrain extension is mjlab-shaped; an
  IsaacLab locomotion task simply reports fewer commands rather than broken
  ones. Anything IsaacLab has that mjlab does not belongs in an
  [extension](extensions.md).
