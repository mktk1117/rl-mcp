# Go1 flat locomotion, in one file, on three backends

`go1_flat/env.py` is the whole task: the config dataclasses at the top, the
`State` of every variable `step()` writes, then `reset()`, `step()`, the
observation groups, one method per reward term and the terminations, in the
order they happen. It is built from `rlmcp.adapters.single_file.blocks` and inherits
`rlmcp.adapters.single_file.SingleFileEnv`, which is the contract rlmcp reads
it through (see [docs/single-file.md](../../docs/single-file.md)): every
config leaf and every observation-pipe stage is a parameter
(`actor_obs.joint_vel.uniform_noise.half_width`), every variable is traced. `ppo.py` is the algorithm,
copied here so a task that wants a different one edits its copy. `train.py`
is the loop, written out.

The physics is `rlmcp.backends`: one robot-level contract (`RobotSpec`,
`SimBackend`) with MuJoCo Warp, mjbatch and Genesis behind it, which finds
the joints, gains, default pose, base and contacts in the MJCF and reports
what it found. It knows nothing about legs; this task tells it which sites
are feet and asks for mjlab's foot contact tuning, and that is the whole of
what is Go1-shaped outside `env.py`.

The same `env.py` runs on all three. `--backend` is the only thing that
changes:

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

The task is mjlab's `Mjlab-Velocity-Flat-Unitree-Go1`, term for term (see
[docs/single-file.md](../../docs/single-file.md) for the one-to-one list and
what differs). Recorded on an RTX 3090 shared with another job, 32 CPU
threads, genesis-world 1.3.3, mujoco 3.13.0, mjbatch 0.1.1, mujoco_warp 3.13.0.

| backend | envs | iterations | linear / angular tracking | posture | episode length | wall time |
| --- | --- | --- | --- | --- | --- | --- |
| mjbatch (CPU) | 2048 | 1500 | 0.86 / 0.61 | 0.84 | 994 / 1000 | 45 min |
| mjwarp | 4096 | 1500 | 0.87 / 0.71 | 0.87 | 1000 / 1000 | 49 min |
| genesis | 2048 | 400 | 0.30 / 0.80 | 0.89 | 937 / 1000 | 16 min |

The two MuJoCo backends trot by iteration 250; at 1024 envs on CPU the
same file starts walking late and slowly. Genesis stands and turns
under mjlab's symmetric commands and walks (0.85 by iteration 200) once the
commands are forward-biased; the ablations behind that sentence are in the
docs page, along with the `--set` line to start a Genesis run with.

Two earlier lessons: actions must not be clipped to `[-1, 1]` before the
scale (a quarter radian per joint is not a step, and every backend stood
still for 1500 iterations), and mujoco_warp's per-step line-search warning
has to be silenced or the log grows by gigabytes an hour.
