# Driving a single-file environment

> **Status: verified on real runs.** The Go1 example in
> [`examples/single_file/go1_flat/`](../examples/single_file/go1_flat/) trains
> on MuJoCo Warp, on mjbatch and on Genesis, from the same `env.py`, with rlmcp
> wrapped around it and every command driven from another shell. What follows
> is only what is different about this family.

A single-file environment is an `env.py` an agent can read top to bottom: the
configuration is one dataclass at the top of the file, and `step()`, `reset()`,
the rewards and the terminations are written out below it. There is no manager
and no base class. rlmcp watches and steers it the same way it does an mjlab or
IsaacLab run -- one `wrap()`, then `rlmcp status`, `rlmcp set`, `rlmcp video`
from a second shell -- because the file *declares* what rlmcp needs to know.

## The two lines

```python
import rlmcp.adapters.single_file as rlmcp_single_file

env = Go1FlatEnv(EnvConfig(backend="mjwarp", num_envs=4096))
env = rlmcp_single_file.wrap(env, session_dir=log_dir / "rlmcp", task_id="go1-flat")
env.attach_algorithm(ppo)                    # hyperparameters + checkpoints

for iteration in range(1, max_iterations + 1):
  ...rollout with env.step(), then losses = ppo.update()...
  env.service(iteration, metrics=losses)     # the iteration boundary
```

`attach_algorithm` is the counterpart of `attach_runner` for a loop that owns
its own PPO: the object's `learning_rate`, `entropy_coef`, `clip_param` and
friends become `rl.*` parameters, and `save()`/`load()` become checkpoints.
`service` is where parameter edits land, commands are answered and a pause
blocks -- after the update, before the next rollout, the one point where
nothing is mid-step. It raises `TrainingStopped` when an agent asks the run to
stop, so the loop can save and exit.

There is no runner to hook, so the loop says where its boundary is instead of
rlmcp guessing at one. Until `attach_algorithm` or `service` is called the
wrapper services on a step cadence, so a script that forgets both still
answers commands, just not on an iteration boundary.

## Declaring the config

The config is a plain dataclass tree on `env.cfg`. Two markers from
`rlmcp.declare` say what a dataclass cannot:

```python
from rlmcp.declare import Static, Term, term

@dataclass
class Rewards:
  tracking_lin_vel: Term = term(2.0, sigma=0.5)
  upright: Term = term(1.0, sigma=0.4472)
  action_rate: Term = term(-0.1)

@dataclass
class Commands:
  lin_vel_x: tuple[float, float] = (0.3, 1.0)
  lin_vel_y: tuple[float, float] = (-0.3, 0.3)
  ang_vel_yaw: tuple[float, float] = (-0.5, 0.5)

@dataclass
class EnvConfig:
  backend: Static[str] = "mjwarp"
  num_envs: Static[int] = 4096
  action_scale: float = 0.25
  reward: Rewards = field(default_factory=Rewards)
  command: Commands = field(default_factory=Commands)
  termination: Termination = field(default_factory=Termination)
```

**`Static[T]`** marks a value the environment reads once, at construction:
sizes, timesteps, the asset, anything copied into a tensor in `__init__`.
rlmcp lists it with liveness `at_startup` and refuses a write with that
reason, instead of reporting success and changing nothing. Everything
unmarked is live, because a single-file environment reads its config on every
step or resample by construction -- that is the point of the shape. A nested
dataclass marked `Static` makes everything under it static.

**`term(weight, **params)`** declares one reward term. It returns a dataclass
field, so every `EnvConfig()` gets its own `Term` and a weight rlmcp changes on
one run cannot leak into the next config built in the same process. The reward
loop reads the table with `terms()`:

```python
from rlmcp.declare import terms

for name, t in terms(self.cfg.reward).items():
  value = computed[name] if name in computed else t.func(self, **t.params)
  total += t.weight * value * self.control_dt
```

The `func` branch is what makes `rlmcp add-reward` work: an appended term
carries its function and scores from the next step, under
`reward.<name>.weight` like the ones the task shipped with.

## What the keys look like

The tree decides the vocabulary, so the same commands work as on mjlab:

| key | from |
| --- | --- |
| `reward.upright.weight`, `reward.upright.params.sigma` | a `term(...)` in the reward group |
| `command.lin_vel_x` (a `[low, high]` range) | a field of the `command` group |
| `termination.max_tilt_rad`, `noise.dof_pos` | a field of any other nested group |
| `env.action_scale` | a scalar at the top level of the config |
| `rl.learning_rate`, `rl.entropy_coef` | the attached algorithm |

Categories follow group names -- `reward`, `command` (curriculum),
`termination`, `action`, `noise`/`randomization` (domain randomization),
`sim`/`physics` -- and anything else is `other`, still fully tunable.

The `command` group has one convention: its `[low, high]` fields, in
declaration order, are the columns of the `commands` buffer. That is how the
trace sampler knows whether the buffer holds a plane velocity, which decides
whether `rlmcp diagnose` may measure tracking against it.

## What the environment has to have

Detection is by shape, at wrap time, and a mismatch is refused by name:

* `env.cfg` -- the declared dataclass;
* `env.num_envs`, `env.reset(env_ids)` -- `None` restarts everything;
* the conventional state buffers, which are legged_gym's names and what the
  shared trace sampler reads: `dof_pos`, `dof_vel`, `actions`, `last_actions`,
  `base_pos`, `base_lin_vel`, `base_ang_vel`, `projected_gravity`, `commands`;
* `env.sim` -- the physics backend (below), which is where frames come from.

An environment that spells these differently says so with
`wrap(spec=SingleFileSpec(...))`.

`step()` returns `(obs, rewards, dones, info)`. The wrapper parks `rewards` on
the environment as `rew_buf` for the trace, and reads `info["reward_terms"]`
(per-term means), `info["episode_rewards"]` and `info["episode_lengths"]` into
the per-iteration telemetry.

## Backends, and swapping them

The environment never touches a simulator API. It builds one object from a
`RobotSpec` and reads robot state out of it:

```python
from rlmcp.backends import RobotSpec, make_backend

robot = RobotSpec(xml=go1_xml, stiffness={"*_calf_joint": 35.76, "*": 15.89},
                  damping={"*_calf_joint": 2.28, "*": 1.01},
                  effort_limit={"*_calf_joint": 35.55, "*": 23.7},
                  contact_sites=("FR", "FL", "RR", "RL", "trunk"))
sim = make_backend(cfg.backend, robot, cfg.num_envs, dt=0.005, decimation=4)

sim.set_dof_targets(targets); sim.step()
sim.root_pos, sim.root_quat, sim.root_lin_vel, sim.root_ang_vel
sim.dof_pos, sim.dof_vel, sim.dof_torque, sim.contact_forces
sim.reset(env_ids, root_pos, root_quat, dof_pos)
```

Three backends answer to that contract, and `cfg.backend` is the only thing
that changes between them:

| backend | what it is | device |
| --- | --- | --- |
| `mjwarp` | [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp): every env on the GPU, CUDA graphs for step/reset | CUDA |
| `mjbatch` | [mjbatch](https://github.com/kevinzakka/mjbatch): one C MuJoCo `MjData` per env on a thread pool | CPU |
| `genesis` | [Genesis](https://github.com/Genesis-Embodied-AI/Genesis), reading the same MJCF | CUDA |

The contract sits at the level of a robot, not of an `MjData`, because that is
the level the three share: Genesis has no `qpos` layout and no sensors. So the
vocabulary is root pose and velocity (world frame; MuJoCo's body-frame angular
velocity is rotated for you), joint position, velocity and torque in the order
`RobotSpec.joints` gives (model order when empty), and a normal contact force
per named site. Quaternions are `(w, x, y, z)` everywhere.

What a `RobotSpec` asks for, the MuJoCo backends compile into the model with
`MjSpec`: a position actuator per actuated joint (`kp`, `-kp`, `-kd` -- how
MuJoCo spells a PD controller), a touch sensor per contact site, a box site
around a *body* named as a contact site so the whole body is watched, and a
ground plane when the MJCF has none. Genesis gets the same gains through
`set_dofs_kp`/`set_dofs_kv` and reads contact from link forces. mjlab's
`go1.xml` ships with none of those, and needs none added to the file.

`mjbatch` and `mjwarp` are the same physics and agree to floating-point
noise on the standing Go1 (the test suite checks the first fifty steps). Genesis
is a different engine and lands a centimetre away with slightly different
foot loads; a policy trained on one is a starting point on another, not a
transfer.

Frames come from the backend too: `mujoco.Renderer` on a CPU copy of one env's
state for the MuJoCo backends (set `MUJOCO_GL`; on this machine `glfw` works
where `egl` does not), and the observing camera the Genesis backend adds
before it builds the scene. Genesis builds every environment at the same place
and draws environment 0, so a frame of `env_id` 7 is a view of the pile, not of
robot 7.

## What has actually been run

Recorded on an RTX 3090 shared with another training job, 32 CPU threads,
genesis-world 1.3.3, mujoco 3.13.0, mjbatch 0.1.1, mujoco_warp 3.13.0. The
same `env.py`, the same PPO, the same defaults; only `--backend` changes.

| backend | envs | iterations | tracking reward at the end | episode length | wall time |
| --- | --- | --- | --- | --- | --- |
| mjbatch (CPU) | 2048 | 400 | 0.94 | 996 / 1000 | 482 s |
| mjwarp | 4096 | 300 | 0.96 | 962 / 1000 | 934 s |
| genesis | 1024 | 300 | 0.90 | 933 / 1000 | 553 s |

A gait appears by iteration 100 on every backend; the progress clips in each
session's `artifacts/` show it. Genesis lands a little lower on angular
tracking (0.60 against 0.73 and 0.79), which is a different contact model
under the same weights rather than anything the file does differently.

The one finding worth carrying: the first version of the example clamped
actions to `[-1, 1]` before the 0.25 rad scale, and every backend converged
to a *standing* policy -- tracking reward flat at 0.22 through 1500
iterations, unmoved by raising the tracking weight live from 2.0 to 5.0. A
quarter radian per joint is not enough for a step. `action_clip` is now a
live parameter with legged_gym's 100 as the default, and the plateau was
diagnosed the way rlmcp intends: `status` showed commanded speed 0.67 against
achieved 0.10 while the reward climbed.

Everything in the rlmcp surface was driven from a second shell against these
runs: `status`, `params`, `set` (applied live), a `Static` write (refused with
the reason), `shot`, `diagnose`, `add-reward` (scored from the next batch and
tunable as `reward.foot_slip.weight`), and the progress clips.

## Where things live

```
rlmcp/declare.py                 Static, Term, term, terms -- stdlib only
rlmcp/adapters/single_file/      the family: spec, access, sim adapter, wrapper, algorithm adapter
rlmcp/backends/                  RobotSpec, SimBackend, mjwarp, mjbatch, genesis
examples/single_file/go1_flat/   env.py, ppo.py, train.py -- the worked example
```

The markers in `rlmcp.declare` import nothing outside the standard library and
are detected by shape, not identity, so a project that would rather not depend
on rlmcp can vendor the twenty lines and still be driven by it.
