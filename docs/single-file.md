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

## The example is mjlab's flat Go1 task, term for term

`examples/single_file/go1_flat/env.py` is `Mjlab-Velocity-Flat-Unitree-Go1`
written out flat, so a number in it means what it means there and an agent
that knows the mjlab task's levers can drive this one:

* **Observations.** Actor: base linear and angular velocity (body frame),
  projected gravity, joint positions relative to default with a per-env
  encoder bias, joint velocities, last actions, the command -- with mjlab's
  uniform noise half-widths (0.5, 0.2, 0.05, 0.01, 1.5). Critic: the same
  unbiased and clean, plus foot air time, foot contact and log1p contact
  force. Both networks normalise their inputs with a running mean and std.
* **Actions.** Joint targets around the default pose, scaled per joint by
  0.25 x effort limit / stiffness (0.373 hip and thigh, 0.249 calf), unclipped.
* **Commands.** mjlab's `twist`: planar velocity in [-1, 1], yaw rate in
  [-0.5, 0.5], resampled every 3 to 8 s; 10% of envs stand, 30% steer toward
  a target heading (yaw rate = 0.5 x heading error), 20% are sent straight
  ahead.
* **Rewards.** Linear and angular tracking (2.0, std 0.5 and 0.707), upright
  (1.0, std 0.447), variable posture (1.0, mjlab's per-regime stds), soft
  joint limits (-1.0, 0.9 of the range), action rate (-0.1), foot clearance
  (-2.0, 0.1 m), swing height at landing (-0.25), foot slip (-0.1), soft
  landing (-1e-5); air time and body angular velocity at 0.0 as mjlab ships
  them. Every term is multiplied by the control timestep.
* **Disturbances and randomisation.** A velocity kick every 1 to 3 s
  (mjlab's `push_by_setting_velocity` ranges); foot friction in [0.3, 1.5]
  and encoder bias in +-0.015 rad drawn once per env at startup.
* **Termination.** 70 degrees of tilt, or 20 s.
* **Physics.** dt 0.005, decimation 4, Newton with 10 and 20 iterations,
  elliptic cone, impratio 10, feet with contact priority and mjlab's foot
  friction, all collision geoms hardened to solref (0.01, 1).
* **PPO.** rsl_rl's, with mjlab's Go1 runner config: (512, 256, 128) ELU,
  unit initial std, observation normalisation, 24 steps per env.

What differs, and why: foot height is the foot site's z above the flat floor
and foot velocity is a finite difference over the control step, where mjlab
has a terrain height sensor and site velocities; the angular-momentum term
(weight 0 in mjlab) and the base centre-of-mass offset randomisation are
omitted, since neither has a backend-neutral spelling yet; mjlab's
command-range curriculum at 5000 and 10000 iterations is not built in --
that is what an rlmcp `StageSchedule` on `command.lin_vel_x` is for.

## What has actually been run

Recorded on an RTX 3090 shared with another training job, 32 CPU threads,
genesis-world 1.3.3, mujoco 3.13.0, mjbatch 0.1.1, mujoco_warp 3.13.0. The
same `env.py`, the same PPO, mjlab's defaults; only `--backend` changes.

| backend | envs | iterations | linear / angular tracking | posture | episode length | wall time |
| --- | --- | --- | --- | --- | --- | --- |
| mjbatch (CPU) | 2048 | 1500 | 0.86 / 0.61 | 0.84 | 994 / 1000 | 45 min |
| mjwarp | 4096 | 1500 | 0.87 / 0.71 | 0.87 | 1000 / 1000 | 49 min |
| genesis | 2048 | 400 | 0.30 / 0.80 | 0.89 | 937 / 1000 | 16 min |

The tracking terms are exp(-error / std^2), so 0.86 linear is an RMS
velocity error of about 0.27 m/s against commands up to 1 m/s in any planar
direction, with pushes every 1 to 3 s and a fifth of the commands pointing
backwards or sideways. Both MuJoCo backends trot by iteration 250 and are
still improving at 1500; the progress clips in each session's `artifacts/`
show it.

Genesis under the same file learns to stand and turn in place: angular
tracking and posture match the other two, linear tracking does not move off
0.3. Foot heights, contact forces and pushes read identically on Genesis and
mjbatch (checked side by side), so the harness is not the difference. Ten
short runs, every one launched with `--set` overrides through rlmcp's
launch config, narrow it down:

| run (200 iterations unless noted) | linear tracking at the end |
| --- | --- |
| genesis, 1024 envs, as configured | 0.29 |
| genesis, pushes off | 0.30 |
| genesis, pushes and observation noise off | 0.32 |
| genesis, foot clearance / swing / slip / landing terms off | 0.28 |
| genesis, **forward-biased commands** (x in [0.3, 1], y in +-0.3, no heading, standing or forward envs) | **0.85** |
| genesis, 2048 envs, 400 iterations | 0.30 |
| mjbatch, 1024 envs, 400 iterations | 0.31 |
| mjbatch, 2048 envs, 400 iterations | 0.60 |

Two things are true at once. Standing still is a strong attractor under
mjlab's symmetric command distribution -- a third of the commands point
backwards or sideways and a tenth are zero -- and escaping it needs batch:
mjbatch at 1024 envs is as stuck as Genesis at 1024. And Genesis has a
higher barrier of its own: at 2048 envs it stays put where mjbatch walks,
though it walks readily once the commands are forward-biased, so its contact
dynamics make the first steps costlier rather than impossible. Genesis
ignores the MuJoCo-side contact tuning (foot priority, condim, solref,
elliptic cone), which is the obvious place to look next. For a Genesis run
today, start it with

```bash
python examples/single_file/go1_flat/train.py --backend genesis \
    --set 'command.lin_vel_x=[0.3, 1.0]' --set 'command.lin_vel_y=[-0.3, 0.3]' \
    --set command.rel_heading_envs=0 --set command.rel_standing_envs=0
```

and widen the ranges with `rlmcp set` once it is walking -- which is what a
curriculum stage is for.

Two findings from the first version of the example are worth carrying:

* It clamped actions to `[-1, 1]` before the 0.25 rad scale, and every
  backend converged to a standing policy -- tracking flat at 0.22 through 1500
  iterations, unmoved by raising the tracking weight live from 2.0 to 5.0. A
  quarter radian per joint is not a step; mjlab does not clip.
* With mjlab's solver settings mujoco_warp warns about the line-search budget
  on every step, which at 50 Hz is a log that grows by gigabytes an hour. The
  backend silences it the way mjlab does.

Everything in the rlmcp surface was driven from a second shell against these
runs: `status`, `params` (73 declared), `set` (applied live), a `Static`
write (refused with the reason), `shot`, `diagnose`, `add-reward` (scored
from the next batch and tunable as `reward.foot_slip.weight`), progress
clips, and `--set key=value` launch overrides for the ablations.

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
