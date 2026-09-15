# Go1 flat locomotion, in one file, on three backends

`go1_flat/env.py` is the whole task: the config dataclasses at the top, then
`reset()`, `step()`, the observations, the rewards and the terminations, in
the order they happen. `ppo.py` is the algorithm, copied here so a task that
wants a different one edits its copy. `train.py` is the loop, written out.

The same `env.py` runs on MuJoCo Warp, mjbatch and Genesis. `--backend` is the
only thing that changes:

```bash
export MJLAB_GO1_XML=/path/to/mjlab/src/mjlab/asset_zoo/robots/unitree_go1/xmls/go1.xml
export MUJOCO_GL=glfw   # or egl; frames from the MuJoCo backends need a GL context

python examples/single_file/go1_flat/train.py --backend mjwarp  --num-envs 4096
python examples/single_file/go1_flat/train.py --backend mjbatch --num-envs 1024 --device cpu
python examples/single_file/go1_flat/train.py --backend genesis --num-envs 2048
```

Then, from a second shell, the same commands as on any other backend:

```bash
rlmcp status
rlmcp params --contains reward
rlmcp set reward.action_rate.weight -0.2 --why "knees chattering"
rlmcp set command.lin_vel_x '[0.5, 1.5]'
rlmcp diagnose --seconds 2
rlmcp video --seconds 4
```

`rlmcp set env.num_envs 8192` is refused with the reason (`at_startup`): the
config says so with `Static[int]`, and the whole point of declaring it is that
the answer is a sentence rather than a silent no-op.

## Installing the backends

Everything here is `pip`-installable; a Python 3.12 environment holds all
three at once (mjbatch pins `mujoco==3.13.0`, which mujoco_warp and Genesis
both accept, and ships wheels for 3.10-3.12 only):

```bash
pip install torch mujoco==3.13.0 mjbatch genesis-world warp-lang \
    'mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp'
```

The robot comes from mjlab's asset zoo (`go1.xml`, plain MJCF, read by every
backend). mjlab itself is not needed: point `MJLAB_GO1_XML` at the file.

## Run notes

Recorded on an RTX 3090 (shared with another job), 32 CPU threads,
genesis-world 1.3.3, mujoco 3.13.0, mjbatch 0.1.1, mujoco_warp 3.13.0.

| backend | envs | iterations | tracking reward | episode length | wall time |
| --- | --- | --- | --- | --- | --- |
| mjbatch (CPU) | 2048 | 400 | 0.94 | 996 / 1000 | 482 s |
| mjwarp | 4096 | 300 | 0.96 | 962 / 1000 | 934 s |
| genesis | 1024 | 300 | 0.90 | 933 / 1000 | 553 s |

Same file, same PPO, same defaults; a gait by iteration 100 on each. mjbatch
and mjwarp are the same physics and agree to seven digits on the standing
robot, so a run on one reproduces on the other. Genesis is a different engine
and its numbers are its own.

Before `action_clip` existed the example clamped actions to `[-1, 1]` and
every backend settled into standing (tracking 0.22, flat through 1500
iterations). A quarter radian per joint is not a step. See
[docs/single-file.md](../../docs/single-file.md) for how it was found.
