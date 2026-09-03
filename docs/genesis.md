# Driving a Genesis run

> **Status: verified on a real run.** genesis-world 1.3.3, Genesis's own
> `examples/locomotion/go2_env.py` trained on an RTX 3090 with rlmcp wrapped
> around it, every command driven from another shell. Four bugs came out of
> that run that no test against a fake could have found; they are fixed and
> each has a regression. See *What the real run changed*, below.

rlmcp watches and steers Genesis the same way it does mjlab and IsaacLab: one
line in the training script, then a live run you can query, tune and record
from another shell. This page is only what is different about Genesis.

## The one line

```python
import rlmcp.adapters.genesis as rlmcp_genesis

env = Go2Env(num_envs=4096, env_cfg=env_cfg, obs_cfg=obs_cfg,
             reward_cfg=reward_cfg, command_cfg=command_cfg)
env = rlmcp_genesis.wrap(env, session_dir=log_dir / "rlmcp", task_id="go2-walk")

runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
env.attach_runner(runner)
runner.learn(num_learning_iterations=...)
```

Same call, same keywords and the same CLI and MCP commands as the other two
backends — see [tools.md](tools.md). What differs is underneath.

## Which environments this covers

Not "Genesis" — Genesis ships examples, not a task framework. What the adapter
targets is the **shape** those examples have, which the community forks copy:

* the tunables live in plain dicts on the env instance (`env_cfg`, `reward_cfg`,
  `command_cfg`) rather than in manager term configs,
* `reward_scales` maps a term name to a number, and
* `reward_functions` binds `_reward_<name>` for each of those keys, once, at
  construction.

That is why the shared code lives in `rlmcp/adapters/legged_gym_style/` and the
Genesis package on top of it is thin. An env of this shape on any simulator gets
the same treatment.

## Four things that are genuinely Genesis's own

### Reward scales are pre-multiplied by `dt`

`Go2Env.__init__` does `self.reward_scales[name] *= self.dt` before training
starts, so the live dict holds `-0.0001` where the config said `-0.005`. Worse,
`self.reward_scales` **is** `reward_cfg["reward_scales"]` — the same dict object
— so the original number is gone and no amount of inspection recovers it.

rlmcp therefore never exposes the raw dict. Each scale is a synthetic parameter
whose getter divides by `dt` and whose setter multiplies by it, so
`rlmcp params` shows the number that was written in the config and
`rlmcp set reward.action_rate.weight -0.01` means what it says. The wrapper
prints the assumption once at startup, because it is an assumption: by the time
rlmcp exists, the multiplication has already happened.

### Command ranges are baked into tensors at construction

This is the trap worth knowing about. `command_cfg` carries
`lin_vel_x_range`, `lin_vel_y_range` and `ang_vel_range`, but `__init__` reads
them once into `self.commands_limits`, a pair of tensors, and
`_resample_commands` samples from *those*. Writing `command_cfg[...]` during a
run reports success and changes nothing.

Command ranges are the main curriculum lever, so getting this wrong would be a
silent failure in exactly the place it hurts most. The command provider writes
through `commands_limits` and keeps `command_cfg` in step, so the value an agent
reads back is the value the sampler is actually using.

### `_reset_idx` takes a mask, not indices

It is private, it takes a boolean mask of shape `(num_envs,)`, and `None` means
every environment. `reset_envs` converts the id list the controller hands it —
including whatever `--where` resolved to — into that mask.

### Frames need a camera, and that is decided before the scene is built

Genesis has two visual paths and only one of them is a frame source.

The **viewer** (`gs.Scene(show_viewer=True, viewer_options=...)`) is an
interactive window. It needs a display — Genesis raises "No display detected"
without one — and it draws rather than returning pixels, so `shot` and `video`
cannot come from it.

**Cameras** are the offscreen path: `scene.add_camera(...)` returns a camera
whose `render()` hands back arrays, headless, over ssh. That is what rlmcp
records through, and it has to already exist:

```python
cam = scene.add_camera(res=(640, 480), pos=(2.0, 0.0, 2.5),
                       lookat=(0.0, 0.0, 0.5), GUI=False, debug=True)
scene.build(n_envs=num_envs)
```

`add_camera` carries `@gs.assert_unbuilt`, and `Go2Env.__init__` builds the
scene before it returns, so by the time `wrap()` runs the door is shut. rlmcp
cannot make one for you, which is why a run without one is told at wrap time
rather than at the first `shot`.

**Use `debug=True`.** Genesis documents debug cameras as ones that record the
simulation "without being part of the 'sensors'" and without interfering with
what the robots perceive — which is what an observing camera should be. A plain
camera joins the robot's sensor set, and a camera that changes what the policy
sees is not an observation of the run.

**Watching one environment among thousands** needs `rendered_envs_idx` too. It
defaults to `None`, meaning all of them, and it is fixed before the build, so a
run that means to draw a particular set has to say so at construction:

```python
vis_options=gs.options.VisOptions(rendered_envs_idx=[0, 7])
```

The startup check reports both — whether there is a camera, and which
environments it can be pointed at — because a `shot --env-id 7` that silently
answers with env 0 is a picture that looks like an answer and is not one. That
list is read from the visualizer's **context**, not from `VisOptions`: the
options hold what was asked for, the context holds what is drawn, and on a real
run they disagreed.

**One caveat on `--env-id`, and it is the task's, not rlmcp's.** Go2Env builds
every parallel environment at the same `base_init_pos`, so the copies are
superimposed — a frame shows a pile of robots and aiming at env 7 frames the
same pile as aiming at env 0. To look at one robot on a task like that, have
the renderer draw one. A task that spaces its environments apart gets what
`--env-id` promises.

Everything else — parameters, metrics, traces, curricula, records — works
without any of this.

## What lands where

| what you touch | where it is | liveness |
| --- | --- | --- |
| a reward weight | `reward_scales[name]`, through the `dt` conversion | live |
| `tracking_sigma`, `base_height_target` | `reward_cfg`, read inside the reward functions each step | live |
| `action_scale`, `clip_actions` | `env_cfg`, read in `step()` | live |
| pitch / roll termination limits | `env_cfg`, read in `step()` | live |
| a command range | `commands_limits`, **not** `command_cfg` | live |
| `episode_length_s` | baked into `max_episode_length` at init | refused |
| a reward term the task never had | `rlmcp add-reward` writes it into `reward_functions`, `reward_scales` and `episode_sums`, which `step()` walks | live |
| the PPO knobs, checkpoints, `stop` | the rsl_rl runner, through `attach_runner` | live |

## What the real run changed

Four things were wrong, and every one of them passed against a fake:

* **`reset-envs` failed outright.** rsl_rl collects rollouts inside
  `torch.inference_mode()`, which makes the environment's buffers *inference
  tensors*; Genesis's reset writes them in place, and torch refuses that from
  outside inference mode — which is exactly where rlmcp services commands. The
  reset now runs inside inference mode.
* **`rlmcp/action_rate_rms` was always exactly 0.** Go2Env ends `step()` with
  `last_actions.copy_(actions)`, so at the iteration boundary the two buffers
  are identical. That is not "the policy is smooth", it is "the question cannot
  be asked here" — and it read as a perfectly smooth policy on a robot that was
  visibly buzzing. The metric is omitted when the environment has already
  synced them; per-step action rate is still measured by `trace` and `diagnose`.
* **`shot --env-id 2` was falsely refused** for an environment the renderer was
  drawing, because the check read `VisOptions` instead of the visualizer
  context. See above.
* **`diagnose` gave advice in mjlab's vocabulary**, telling the operator to
  "raise `action_rate_l2` or lower `action.joint_pos.scale_gain`" — keys that do
  not exist on Genesis, where they are `reward.action_rate.weight` and
  `action.scale`. The advice now names the lever rather than one backend's key.

What did hold up: 30 parameters discovered from the stock Go2 config with
nothing described to rlmcp; reward weights reading back as the numbers the
config was written with rather than their `dt`-scaled values; and a write to
`command.lin_vel_x` moving the measured `commanded_speed_mean` from 0.5 to
1.18, which is the `commands_limits` write-through doing exactly what it was
built for.

## Known differences from mjlab

* **No live browser view** (`live_view`). `rlmcp view` mirrors a MuJoCo scene
  into viser, which a Genesis run has nothing to hand over; asking for one is
  refused by name rather than served empty. Screenshots, clips and progress
  clips all work once the scene has a camera. Future work — this is the single
  entry in `GENESIS_UNSUPPORTED` in
  [tests/test_backend_conformance.py](../tests/test_backend_conformance.py),
  and deleting it is how the capability lands.
* **`rlmcp play` is mjlab's.** `play` rebuilds an environment from a task
  registry, and Genesis has none — the env is a class you instantiate. This is
  a pre-existing limit rather than a Genesis one: IsaacLab cannot `play` either.
  Future work, tracked separately.
* **No foot contacts, no joint torques.** `Go2Env` keeps neither buffer, so
  those trace channels are omitted and the gait and effort sections of
  `diagnose` are skipped. An env that keeps them gets them.
* **Terrain commands are absent**, as on IsaacLab: the terrain extension is
  mjlab-shaped, and a flat-ground task reports fewer commands rather than
  broken ones.

## Tests

Everything below runs in stock CI: no Genesis, no GPU, no display.

| file | pins |
| --- | --- |
| `tests/test_flat_env_access.py` | discovery from the dicts, the `dt` round trip, the `commands_limits` write-through, what is refused and why |
| `tests/test_genesis_adapter.py` | the robot, trace channels and labels, `_reset_idx` masking, camera absence and per-env framing, wrap-time messages |
| `tests/test_backend_conformance.py` | every command answers on every backend, or is in that backend's declared unsupported set |

The conformance file is what makes "the same commands" checkable rather than
promised, and it covers mjlab and IsaacLab too.
