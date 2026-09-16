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

## The contract: `SingleFileEnv`

Inherit `rlmcp.adapters.single_file.SingleFileEnv`. It holds no physics and
no task; it names what rlmcp reads and owns the one loop every file would
otherwise copy.

```python
from rlmcp.adapters.single_file import SingleFileEnv

class MyEnv(SingleFileEnv):
  def __init__(self, cfg):
    self.cfg = cfg                       # the declared dataclass
    self.num_envs = cfg.num_envs
    self.device = torch.device(cfg.device)
    self.control_dt = cfg.dt * cfg.decimation
    self.max_episode_steps = ...
    self.sim = ...                       # optional; render(env_id) gives frames
    self.dof_pos = self.dof_vel = self.actions = ...   # (num_envs, n) tensors

  def reset(self, env_ids=None): ...     # None restarts everything; returns obs
  def step(self, actions):               # -> (obs, reward, done, info)
    ...
    reward, terms = self.compute_reward(scale=self.control_dt)
    info = self.step_info(terms, episode_rewards, episode_lengths, time_outs)
    return obs, reward, done, info

  def compute_reward_terms(self):        # {name: (num_envs,) tensor}, unweighted
    return {"upright": ..., "action_rate": ...}
```

The checklist, which `wrap()` also checks at construction and refuses by name:

| what | where | required |
| --- | --- | --- |
| the declared config | `env.cfg`, a dataclass instance | yes |
| batch size and device | `env.num_envs`, `env.device` | yes |
| timing | `env.control_dt`, `env.max_episode_steps` | yes |
| joint state and actions | `dof_pos`, `dof_vel`, `actions` (`last_actions` for the rate metric) | yes |
| base state, for a floating base | `base_pos`, `base_quat`, `base_lin_vel`, `base_ang_vel`, `projected_gravity` | no |
| the command buffer, for a commanded task | `commands` | no |
| physics | `env.sim`, with `render(env_id)` for frames | no |
| the reward table | `cfg.reward`, a dataclass of `term(...)` fields | for reward tuning |

The buffer names are legged_gym's, which is why the trace sampler and the
summary metrics are shared with the Genesis family. A buffer that is absent
drops its channel rather than being faked: a fixed-base hand has no
`base_lin_vel`, and `rlmcp diagnose` simply skips the tracking section for
it. The names are fixed; an environment that keeps its config, reward table
or reset under other names says so with `wrap(spec=SingleFileSpec(...))`.

`compute_reward()` is why the base class exists: it weights every term in the
table, and a term the table has that `compute_reward_terms()` did not score is
one `rlmcp add-reward` appended at runtime, carrying its own function. The
file never has to know a term was added. `step_info()` builds the `info` dict
the wrapper reads into per-iteration telemetry: per-term means, the totals of
the episodes that ended this step, and which `done` envs timed out rather
than failed.

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
| `termination.max_tilt_rad`, `noise.dof_pos` | a field of any other nested group |
| `randomization.startup.foot_friction` | a field two levels down; `at_startup` if any level is `Static` |
| `env.action_scale` | a scalar at the top level of the config |
| `rl.learning_rate`, `rl.entropy_coef` | the attached algorithm's `cfg` |

Categories follow group names -- `reward`, `command` (curriculum),
`termination`, `action`, `noise`/`randomization` (domain randomization),
`sim`/`physics` -- and anything else is `other`, still fully tunable.

Two things to know when reaching for these from an agent:

* **Numbers, bools and `[low, high]` pairs are parameters. Strings and dicts
  are not.** A per-joint gain table kept as a dict is not served at all; put
  what should be tunable in a field of its own.
* **There is no `get_parameter` MCP tool.** A single read is
  `run_command("get_parameter", {"key": ...})` or
  `list_parameters(contains=...)`; `set_parameter` answers with the old and
  new value anyway.

The `command` group has one convention: its `[low, high]` fields, in
declaration order, are the columns of the `commands` buffer. That is how the
trace sampler knows whether the buffer holds a plane velocity, which decides
whether `rlmcp diagnose` may measure tracking against it.

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

## Physics is the file's business

The environment builds whatever simulator it wants and keeps the handle on
`env.sim`. rlmcp asks it for `render(env_id)` when a frame is wanted, and for
nothing else; a backend without it gets "frames are not available on it"
rather than a crash. The live 3-D view (`rlmcp view`) is not offered on this
family today; clips and screenshots are.

The Go1 example runs on three simulators from one file because its physics
sits behind a small robot-level contract -- root pose and velocity, joint
state and torque, contact force and position per named site, position
targets, reset, push -- with MuJoCo Warp, mjbatch and Genesis behind it.
That contract and those three backends live next to the example, in
[`examples/single_file/backends/`](../examples/single_file/backends/), as a
thing a task copies. They are locomotion-shaped (a floating base, feet with
their own friction) and are not part of rlmcp; a manipulation task writes its
own or talks to its simulator directly.

## The example is mjlab's flat Go1 task, term for term

`examples/single_file/go1_flat/env.py` is `Mjlab-Velocity-Flat-Unitree-Go1`
written out flat, so a number in it means what it means there and an agent
that knows the mjlab task's levers can drive this one: the same observations
(and the critic's privileged ones), mjlab's `twist` command with heading
control, the same reward table with the same weights and widths, the same
pushes and startup randomisation, the same termination, mjlab's solver
settings, and rsl_rl's PPO with mjlab's Go1 runner config. What differs and
why is in the file's own docstring and comments.

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
rlmcp/adapters/single_file/          SingleFileEnv, spec, access, sim adapter,
                                     wrapper, algorithm adapter
examples/single_file/backends/       RobotSpec, SimBackend, mjwarp, mjbatch, genesis
examples/single_file/go1_flat/       env.py, ppo.py, train.py -- the worked example
tests/test_single_file_env.py        the family against a fake env
tests/test_single_file_mcp_roundtrip.py  every declared parameter through the server
tests/test_backends.py               the backend contract on every simulator present
```
