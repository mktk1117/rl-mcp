# Writing and driving a single-file environment

> **Status: verified on real runs.** The Go1 example in
> [`examples/single_file/go1_flat/`](../examples/single_file/go1_flat/) trains
> on MuJoCo Warp, on mjbatch and on Genesis from the same `env.py`, with rlmcp
> wrapped around it and every MCP tool driven against it from another shell.
> What follows is what is different about this family.

A single-file environment is an `env.py` an agent can read top to bottom: the
configuration is one dataclass at the top of the file, and `step()`, `reset()`,
the rewards and the terminations are written out below it. There is no
manager and no registry. rlmcp watches and steers it the same way it does an
mjlab or IsaacLab run -- one `wrap()`, then `rlmcp status`, `rlmcp set`,
`rlmcp video` from a second shell, or the same tools over MCP -- because the
file *declares* what rlmcp needs to know and inherits one base class that
says where the rest is.

## When to choose this route

Pick a single-file environment when:

* the task is new and you would rather write it than configure it: a hundred
  lines of `step()` you can read beat a manager stack you have to trace;
* the simulator is not one rlmcp has a manager-based adapter for, or you want
  the same task to run on more than one;
* an agent is going to edit the environment itself, not only its numbers --
  everything it can change is in one file, and everything rlmcp can change is
  declared at the top of it.

Pick a manager-based task (mjlab, IsaacLab) when one already exists for the
robot and the terrain and observation machinery it brings is what you need.
Both routes get the same rlmcp surface once wrapped; nothing in the tools
knows which family is underneath.

## The blocks: `rlmcp.adapters.single_file.blocks`

An `env.py` is built from four kinds of thing, and rlmcp reaches all four
without the file listing them by hand:

| block | what it is | how rlmcp uses it |
| --- | --- | --- |
| **config** | a dataclass tree on `env.cfg`, declared with `Static[...]` and `term(...)` | every numeric leaf is a parameter: listed, set live, refused with the reason when static |
| **variables** | a `Variables` subclass on `env.state`: every tensor `step()` writes, with a name, a shape and labels | every one is sampled into a trace; the conventional names feed the diagnostics |
| **observations** | `Obs` groups: each term a source and a pipe of stages (`Noise`, `Delay`, `Scale`, `Clip`, `Offset`) | a stage's fields are parameters: `actor_obs.joint_vel.uniform_noise.half_width`, `actor_obs.joint_vel.delay.steps` |
| **reward terms** | a `term(weight, **params)` per row of `cfg.reward`, each naming a method of the environment | `weight` and `params.<p>` are parameters; a term added at runtime is scored by the same loop |

```python
from rlmcp.adapters.single_file.blocks import Delay, Noise, Obs, Offset, Static, Term, Variables, term, variable

class State(Variables):                                   # the variables
  base_lin_vel: Tensor = variable(3)
  joint_pos: Tensor = variable("joint")
  foot_contact: Tensor = variable("foot", dtype=torch.bool)
  command: Tensor = variable(("lin_vel_x", "lin_vel_y", "ang_vel_z"))
  reward: Tensor = variable()
  episode_length: Tensor = variable(dtype=torch.long)

@dataclass
class Rewards:                                       # the reward table
  track_linear_velocity: Term = term(2.0, std=0.5)
  action_rate_l2: Term = term(-0.1)

class MyEnv(SingleFileEnv):
  def __init__(self, cfg):
    self.cfg = cfg
    self.state = State(cfg.num_envs, cfg.device, joint=joint_names, foot=("FR", "FL"))
    self.actor_obs = Obs(                            # the observations
        base_lin_vel=("base_lin_vel", UniformNoise(0.5)),
        joint_pos=(lambda s: s.joint_pos - default_pos, Offset(encoder_bias), UniformNoise(0.01)),
        joint_vel=("joint_vel", Delay(2), GaussianNoise(0.5)),
        command="command",
    )
    ...

  def reset(self, env_ids=None):
    self.state.reset(env_ids)                        # every variable to zero
    self.reset_blocks(env_ids)                       # every delay forgets the old episode
    ...
    return self.actor_obs(self.state)

  def step(self, actions):
    s = self.state
    ...
    s.reward[:], terms = self.compute_reward(scale=self.control_dt)
    info = self.step_info(terms, episode_rewards, episode_lengths, time_outs)
    return self.actor_obs(s), s.reward, done, info

  def track_linear_velocity(self, std):              # one method per term
    s = self.state
    err = torch.sum(torch.square(s.command[:, :2] - s.base_lin_vel[:, :2]), dim=1)
    return torch.exp(-err / std ** 2)

  def action_rate_l2(self):
    return torch.sum(torch.square(self.state.action - self.state.last_action), dim=1)
```

**Variables.** `variable()` is one number per environment, `variable(3)`
three, `variable("joint")` as many as the `joint` dimension passed at
construction, and `variable(("x", "y"))` two with those labels. A dimension
passed as a list of names labels that axis, so `rlmcp trace` plots
`joint_pos` by joint name. `state.reset(env_ids)` zeroes every variable for
those environments, which is the right start for bookkeeping and harmless
for state the next read from the simulator overwrites. Every variable goes
into the trace under its own name, and the library attaches no meaning to
a name. The one convention is the trace vocabulary in
`rlmcp/adapters/base.py`: a variable named `joint_pos`, `joint_vel`,
`joint_torque`, `action`, `base_pos`, `base_lin_vel`, `base_ang_vel`,
`projected_gravity`, `foot_contact`, `command` or `reward` also feeds the
diagnostics and summary metrics that read that channel. `command` is a
plane velocity `[vx, vy(, wz)]` by that definition; a goal pose or a
motion target goes under another name and is traced under it. A fixed-base
arm does not declare the base ones, and the channels they feed are dropped
rather than faked.

**Pipes.** A term of an `Obs` group is a source -- the name of a variable, or
a function of the state -- followed by stages: `UniformNoise(half_width)`,
`GaussianNoise(std)`, `Delay`, `Scale`, `Clip`, `Offset`. Each stage is a
small dataclass, so its fields are parameters served under the group's
attribute name and the stage's class name in snake case:
`actor_obs.joint_pos.uniform_noise.half_width`. `Delay(steps, max_steps=)`
keeps `max_steps` of history (read once) and `steps` is live within it; a
write outside that range is refused with the range. Repeated stages in one
pipe are numbered (`noise`, `noise_2`). A bare `Pipe(Clip(-1, 1), Scale(0.25))`
assigned to an attribute is served the same way (`action_pipe.clip.high`).
`Obs.slices()` says where each term sits in the concatenated vector.

**Reward terms.** A `term(weight, **params)` in the table names a method of
the environment; `compute_reward()` calls it as `method(**params)` and sums
`weight * value` over the table. A term `rlmcp add-reward` appends carries
its own `func(env, **params)` and goes through the same loop, so the file
never has to know a term was added. A term with no method and no function is
refused by name.

## The contract: `SingleFileEnv`

Inherit `rlmcp.adapters.single_file.SingleFileEnv`. It holds no physics and
no task; it names what rlmcp reads and owns the reward loop.

The checklist, which `wrap()` also checks at construction and refuses by name:

| what | where | required |
| --- | --- | --- |
| the declared config | `env.cfg`, a dataclass instance | yes |
| batch size and device | `env.num_envs`, `env.device` | yes |
| timing | `env.control_dt`, `env.max_episode_steps` | yes |
| the variables | `env.state`, a `Variables` (or plain attributes under the legged_gym names) | yes |
| joint state and actions | variables `joint_pos`, `joint_vel`, `action` | yes |
| base state, for a floating base | variables `base_pos`, `base_quat`, `base_lin_vel`, `base_ang_vel`, `projected_gravity` | no |
| the command, for a commanded task | a variable `command` holding `[vx, vy(, wz)]` | no |
| observation groups and pipes | any `Obs` or `Pipe` attribute | no |
| physics | `env.sim`, with `render(env_id)` for frames | no |
| the reward table | `cfg.reward`, a dataclass of `term(...)` fields, one method per term | for reward tuning |

`compute_reward()` is why the base class exists: it weights every term in the
table, calling the method the term names, or the function a term appended at
runtime carries. `step_info()` builds the `info` dict the wrapper reads into
per-iteration telemetry: per-term means, the totals of the episodes that
ended this step, and which `done` envs timed out rather than failed.
`reset_blocks(env_ids)` clears every delay's history for those environments.

A file that cannot inherit -- vendored, or a class hierarchy of its own --
is still accepted when it has the same shape; the base class is the
documented path, not the only one.

## The two lines in the training loop

```python
import rlmcp.adapters.single_file as rlmcp_single_file

env = MyEnv(EnvConfig(num_envs=4096))
env = rlmcp_single_file.wrap(env, session_dir=log_dir / "rlmcp", task_id="my-task")
env.attach_algorithm(ppo)                    # hyperparameters + checkpoints

for iteration in range(1, max_iterations + 1):
  ...rollout with env.step(), then losses = ppo.update()...
  env.service(iteration, metrics=losses)     # the iteration boundary
```

`attach_algorithm` is the counterpart of `attach_runner` for a loop that owns
its own PPO. `service` is where parameter edits land, commands are answered
and a pause blocks -- after the update, before the next rollout, the one
point where nothing is mid-step. It raises `TrainingStopped` when an agent
asks the run to stop, so the loop can save and exit. Until `attach_algorithm`
or `service` is called the wrapper services on a step cadence, so a script
that forgets both still answers commands, just not on an iteration boundary.

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
one run cannot leak into the next config built in the same process.

The markers import nothing outside the standard library and are detected by
shape, not identity, so a project that would rather not depend on rlmcp for
its config can vendor the twenty lines.

## What the keys look like

The tree decides the vocabulary, so the same commands work as on mjlab:

| key | from |
| --- | --- |
| `reward.upright.weight`, `reward.upright.params.sigma` | a `term(...)` in the reward group |
| `command.lin_vel_x` (a `[low, high]` range) | a field of the `command` group |
| `termination.max_tilt_rad` | a field of any other nested group |
| `randomization.startup.foot_friction` | a field two levels down; `at_startup` if any level is `Static` |
| `env.action_scale` | a scalar at the top level of the config |
| `actor_obs.joint_pos.uniform_noise.half_width`, `actor_obs.joint_vel.delay.steps` | a stage field in an `Obs` group on the environment |
| `rl.learning_rate`, `rl.entropy_coef` | the attached algorithm's `cfg` |

Categories follow group names -- `reward`, `command` (curriculum),
`termination`, `action`, `noise`/`randomization` (domain randomization),
`sim`/`physics` -- observation groups and pipes are `observation`, and
anything else is `other`, still fully tunable.

Two things to know when reaching for these from an agent:

* **Numbers, bools and `[low, high]` pairs are parameters. Strings and dicts
  are not.** A per-joint gain table kept as a dict is not served at all; put
  what should be tunable in a field of its own.
* **There is no `get_parameter` MCP tool.** A single read is
  `run_command("get_parameter", {"key": ...})` or
  `list_parameters(contains=...)`; `set_parameter` answers with the old and
  new value anyway.

The `command` variable has one convention: it is a plane velocity, because
that is what the trace channel of that name means to `rlmcp diagnose`,
which measures tracking against it. A goal or a motion target is declared
under another name.

## Declaring the algorithm

The algorithm is read the way the environment is. A dataclass on
`algorithm.cfg` declares the hyperparameters; every numeric leaf is served as
`rl.<name>`, `Static[...]` marks one read once (a network width), and a
write lands on the config field and on the attribute of the same name when
the algorithm mirrors it. Then, if the algorithm defines it,
`on_hyperparameter_change(name, value)` runs for whatever must follow the
write:

```python
@dataclass
class PPOConfig:
  learning_rate: float = 1e-3
  entropy_coef: float = 0.01
  num_learning_epochs: int = 5

class PPO:
  def __init__(self, ..., cfg: PPOConfig):
    self.cfg = cfg
    self.learning_rate = cfg.learning_rate      # mirrored: read where it is used
    ...

  def on_hyperparameter_change(self, name, value):
    if name == "learning_rate":                  # the optimizer has its own copy
      for group in self.optimizer.param_groups:
        group["lr"] = value
      self.schedule = "fixed"                    # or adaptive overwrites it next update

  def metrics(self):                             # optional: joins the telemetry
    return {"Policy/mean_std": float(self.actor.output_std.mean())}

  def save(self) -> dict: ...                    # checkpoints
  def load(self, state: dict): ...
```

Nothing is hand-listed in rlmcp, so a knob added to the config is tunable the
moment it is declared. An algorithm with no `cfg` declares nothing, and a
write says so.

## Physics: `rlmcp.backends`, or your own

The environment builds whatever simulator it wants and keeps the handle on
`env.sim`. The family asks it for `render(env_id)` when a frame is wanted
and for nothing else; a backend without it gets "frames are not available on
it" rather than a crash.

What rlmcp ships for that slot is `rlmcp.backends`: one contract for an
articulated robot, with MuJoCo Warp, mjbatch and Genesis behind it, so one
`env.py` runs on all three. The contract knows joints, an optional floating
base, and contacts -- not legs. A quadruped, an arm on a table and a hand
are the same thing to it.

```python
from rlmcp.backends import RobotSpec, make_backend

sim = make_backend("mjwarp", RobotSpec(xml="robot.xml"), num_envs=4096, dt=0.005, decimation=4)
# [mjwarp] robot.xml: 7 joints (every single-dof joint in the file); gains from the
# file's actuators; effort limit the file's actuator force ranges; default pose
# keyframe 'home'; fixed base (no free joint in the file); contacts lf_down, rf_down
# (leaf bodies that can collide); 22 contact geoms (collision geoms of the contact
# bodies); floor added

sim.reset(ids, dof_pos=sim.default_dof_pos.expand(n, -1))   # root pose too, on a floating base
sim.set_dof_targets(targets); sim.step()
sim.dof_pos, sim.dof_vel, sim.dof_torque
sim.contact_forces, sim.contact_site_pos                      # (N, n_contacts), (N, n_contacts, 3)
sim.root_pos, sim.root_quat, sim.root_lin_vel, sim.root_ang_vel   # floating base only
sim.push(ids, lin_vel, ang_vel)                               # floating base only
sim.set_friction(ids, coefficient)                            # the contact geoms
```

Only the MJCF is required. Everything else is found in the file when it is
there and declared only when it is not, or when the task wants otherwise --
and the line printed at construction says which was which, so a wrong guess
is a sentence to read rather than a run to debug:

| what | found as | declare with |
| --- | --- | --- |
| actuated joints | every single-dof joint, in model order | `joints=` |
| PD gains and effort limits | the file's position actuators | `stiffness=`, `damping=`, `effort_limit=` (a number, or `{pattern: number}` by joint name) |
| default pose | a keyframe named `home`/`init`/`default`/`standing`/`rest`, else the first, else `qpos0` | `default_joint_pos=`, `keyframe=` |
| floating base | the body carrying the free joint; none means a fixed base | `base_body=` |
| contacts | the leaf bodies of the tree that can collide: feet, fingertips | `contacts=` (sites or bodies) |
| contact geoms | the collision geoms of the contact bodies | `contact_geoms=` |
| a floor | added when the file has no plane or height field | `SimOptions(ground_plane=False)` |

Contact tuning -- `contact_friction`, `contact_condim`, `contact_priority`,
`geom_solref` -- is left at what the file says unless the task asks; the
Go1 example asks for mjlab's values. A joint with no gains in either place
is refused by name with the two ways to supply them. A fixed-base robot
gets no root state and no pushes; asking raises `FixedBase` with the file's
name in it.

The MuJoCo backends compile the spec into the model with `MjSpec` -- a
position actuator per joint the file does not drive, a touch sensor per
contact, a box site around a body named as a contact -- and Genesis reads
the same gains and limits off that compiled model, so the three agree by
construction. `mjbatch` and `mjwarp` are the same physics and agree to
floating-point noise; Genesis is a different engine. Frames come from
`mujoco.Renderer` on a CPU copy of one environment for the MuJoCo backends
(set `MUJOCO_GL`) and from the observing camera the Genesis backend adds
before it builds its scene. The live 3-D view (`rlmcp view`) is not offered
on this family today; clips and screenshots are.

## The example is mjlab's flat Go1 task, term for term

`examples/single_file/go1_flat/env.py` is `Mjlab-Velocity-Flat-Unitree-Go1`
written out flat, so a number in it means what it means there and an agent
that knows the mjlab task's levers can drive this one: the same observations
(and the critic's privileged ones), mjlab's `twist` command with heading
control, the same reward table with the same weights and widths, the same
pushes and startup randomisation, the same termination, mjlab's solver
settings, and rsl_rl's PPO with mjlab's Go1 runner config. What differs and
why is in the file's own docstring and comments. Its shape is the four
blocks above: a `State` of 30 variables, two `Obs` groups (the actor's with
mjlab's noise widths as `UniformNoise` stages), and one method per reward term.

### What has actually been run

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
direction. Both MuJoCo backends trot by iteration 250. Genesis under the same
file stands and turns in place; forward-biased commands get it walking (0.85
by iteration 200), and batch size matters as much as the engine: mjbatch at
1024 envs is as stuck as Genesis at 1024. For a Genesis run today, start with

```bash
python examples/single_file/go1_flat/train.py --backend genesis \
    --set 'command.lin_vel_x=[0.3, 1.0]' --set 'command.lin_vel_y=[-0.3, 0.3]' \
    --set command.rel_heading_envs=0 --set command.rel_standing_envs=0
```

and widen the ranges with `rlmcp set` once it is walking, which is what a
curriculum stage is for.

Every MCP tool was then driven against a live mjbatch run of the example:
status, listing, live writes, the `Static` refusal, `add_reward` scoring from
the next batch, metrics and plots, screenshots, clips, motion diagnosis and
joint traces, resets, pause and resume, checkpoints, notes, events and
feedback, and a stop through the server with the post-mortem tools still
answering from disk. `tests/test_single_file_mcp_roundtrip.py` keeps the
parameter half of that promise in CI: every leaf the example's config
declares is listed, read, written and reset through the real server.

Two findings from earlier versions worth carrying:

* Actions must not be clipped to `[-1, 1]` before the scale. A quarter
  radian per joint is not a step, and every backend stood still for 1500
  iterations; mjlab does not clip.
* With mjlab's solver settings mujoco_warp warns about the line-search budget
  on every step, which at 50 Hz is a log that grows by gigabytes an hour. The
  backend silences it the way mjlab does.

## Where things live

```
rlmcp/declare.py                     Static, Term, term, terms -- stdlib only
rlmcp/adapters/single_file/      the family: SingleFileEnv, providers built from the config tree,
                                 the adapter, the wrapper with attach_algorithm/service
                                     wrapper, algorithm adapter
examples/single_file/backends/       RobotSpec, SimBackend, mjwarp, mjbatch, genesis
examples/single_file/go1_flat/       env.py, ppo.py, train.py -- the worked example
tests/test_single_file_env.py        the family against a fake env
tests/test_single_file_mcp_roundtrip.py  every declared parameter through the server
tests/test_backends.py               the backend contract on every simulator present
```
