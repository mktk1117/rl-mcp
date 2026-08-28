# Go1 flat locomotion, on three backends

The same task — a Unitree Go1 tracking a velocity command on flat ground —
trained under mjlab, IsaacLab and Genesis, each wrapped in one line of rlmcp.
The point is that the second shell is identical whichever one is running:

```bash
rlmcp status
rlmcp params --contains reward
rlmcp set action.scale 0.20 --why "calves are buzzing at 9 Hz"
rlmcp diagnose --seconds 2
rlmcp video --seconds 4
```

Same commands, same parameter keys, same artifacts in the run record. What
differs is underneath, and that is the part you should not have to think about.

## Running each

```bash
# mjlab
python examples/go1_flat/train_mjlab.py --num-envs 4096 --max-iterations 300

# Genesis
python examples/go1_flat/train_genesis.py --num-envs 4096 --max-iterations 300

# IsaacLab -- no script here, the existing one already takes any task id
python examples/train_isaaclab.py --task Isaac-Velocity-Flat-Unitree-Go1-v0 \
    --num_envs 4096 --max_iterations 300 --headless --enable_cameras
```

There is deliberately no `train_isaaclab.py` in this directory. IsaacLab's
example is already task-agnostic, so a Go1 copy of it would be a second file
that has to be kept in step with the first for no gain. mjlab gets its own
script because its construction differs enough to be worth showing; Genesis
gets one because it needs an environment at all.

## What has actually been run

| script | status |
| --- | --- |
| `train_genesis.py` | run end to end on genesis-world 1.3.3, RTX 3090. Trains, and the tuning campaign below was done through it. |
| `train_mjlab.py` | trains — smoke-run for 3 iterations on mjlab 1.5.3. Progress clips fail on that machine with an OpenGL/EGL error from MuJoCo's offscreen renderer, which is a GL setup problem on the box rather than anything about this script; pass `--video-every 0` or fix `MUJOCO_GL` if you hit it. |
| `../train_isaaclab.py` | run for 30 iterations on Isaac Sim 6 / IsaacLab 6.1.17, `Isaac-Velocity-Flat-Unitree-Go1-v0`. 71 parameters discovered, cameras built, progress clip written. Isaac Sim needs `OMNI_KIT_ACCEPT_EULA=YES` to launch non-interactively. |

## Where the robot comes from

| backend | task | asset |
| --- | --- | --- |
| mjlab | `Mjlab-Velocity-Flat-Unitree-Go1` | its own asset zoo (`go1.xml`) |
| IsaacLab | `Isaac-Velocity-Flat-Unitree-Go1-v0` | `UNITREE_GO1_CFG` (USD) |
| Genesis | this directory | **mjlab's `go1.xml`**, loaded as MJCF |

Genesis ships a Go2 and no Go1, so the Genesis environment here loads mjlab's
Go1 description directly — Genesis reads MJCF as readily as URDF. One robot
description feeding two backends beats two that drift apart.

## What is not the same

The three tasks are the same *task*, not the same *numbers*. Each framework's
stock Go1 config has its own reward weights, gains and episode lengths, and this
directory does not try to unify them — a run under one is not a controlled
comparison against another. What transfers is the vocabulary: a reward weight is
`reward.<term>.weight` on all three, so an agent that has driven one can drive
the others.

Two capabilities are missing on Genesis and say so by name rather than failing
oddly: `rlmcp view` (no scene to hand viser) and `rlmcp play` (no task registry
to rebuild from — IsaacLab cannot play either). See
[docs/genesis.md](../../docs/genesis.md).

## A tuned starting point for the Genesis run

`train_genesis.py` defaults to Genesis's stock reward weights. The campaign in
`runs/` on this machine found the stock settings walk but chatter — joints
carrying 16% of their velocity power above 8 Hz — and that the useful lever is
the action scale rather than the action-rate penalty:

| | `hf_share_median` | jerk RMS | speed (cmd 0.5) |
| --- | --- | --- | --- |
| stock | 0.161 | 1384 | 0.489 |
| action-rate 4x | 0.107 | 1279 | 0.484 |
| action-rate 8x | 0.097 | **9390** | 0.464 |
| action scale 0.20 | **0.075** | 1692 | 0.479 |

**What this means:** doubling the action-rate penalty past 4x stops buying
smoothness and starts making the motion worse — the high-frequency share barely
moved while jerk went up sevenfold. Lower the action scale instead. Pass
`--action-scale 0.20` to start from the better setting.
