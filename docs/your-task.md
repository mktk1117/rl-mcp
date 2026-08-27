# Using rlmcp on your own task

Everything in the quickstart drives a task that ships with mjlab. Your own task
is not one of those, and it does not belong in the rl-mcp repository.

## Two repositories

```
rl-mcp/                  the tooling: harness, adapters, CLI, MCP server

your-tasks/              your work
  mytask/                an ordinary mjlab task package: env cfg, mdp terms, assets
    rlmcp_ext.py           the task's vocabulary, as an rlmcp Extension
    curriculum.py          the ladder, as a StageSchedule
    train_curriculum.py    the launcher that calls rlmcp.wrap
  records/               the run records, written and read by `rlmcp record`
```

rlmcp is a dependency of your repository, never the other way round.

They are split because they have different lifecycles. A tool's history should
be about the tool. Run records carry training logs, plots and video: large,
regenerable, and silent about the harness. Four runs of one task came to 24 MB.

`$RLMCP_RECORDS` holds that line at runtime:

```bash
pip install rl-mcp                    # the tooling
export RLMCP_RECORDS=$PWD/records     # point the harness at your records
```

The examples below use `mytask/` for the package and `mytask_ext.py` for its
extension. Nothing about the arrangement is task-specific.

---

## 1. Wrap the environment

One call, between building the environment and building the runner.

```python
import rlmcp
from rlmcp.adapters.mjlab import TrainingStopped

import mytask                 # registers the task with mjlab
import mytask.rlmcp_ext       # registers the extension: importing is what registers
from mytask.curriculum import build_curriculum

env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
env = rlmcp.wrap(
    env,
    session_dir=log_dir / "rlmcp",
    curriculum=build_curriculum(),
    task_id=args.task,
    service_every_steps=agent_cfg.num_steps_per_env,   # service once per iteration
    record_run=args.record_run or None,                # the record this run fills
)

vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = runner_cls(vec_env, asdict(agent_cfg), str(log_dir), args.device)
env.attach_runner(runner)      # PPO knobs, checkpoints, iteration boundaries

try:
    runner.learn(num_learning_iterations=agent_cfg.max_iterations)
except TrainingStopped as stopped:
    print(f"stopping: {stopped}")   # `rlmcp stop` arrives as this exception
runner.save(str(log_dir / f"model_final_{runner.current_learning_iteration}.pt"))
```

### `wrap` options

| argument | default | what it does |
| --- | --- | --- |
| `session_dir` | `./rlmcp_session` | where the run publishes itself |
| `curriculum` | `None` | a `StageSchedule`, or `"terrain"` for the built-in ladder |
| `service_every_steps` | `24` | how often commands are serviced. Set to your steps per iteration |
| `task_id` | `""` | the task name, recorded in the session |
| `extensions` / `exclude_extensions` | auto | override which capabilities load |
| `record_run` | `None` | the record id this run fills |
| `records_root` | `$RLMCP_RECORDS` | where records live |
| `record_slot` | `""` | resource lease to claim, e.g. `gpu0` |
| `video_every` | `"double"` | the [progress-clip](tools.md#progress-clips-the-ones-you-do-not-have-to-ask-for) cadence: `"double"` (0, 50, 100, 200, 400 …), a flat `200`, or `0` to turn them off |
| `video_seconds` | `4.0` | how long each progress clip is |
| `video_budget_mb` | `200` | disk the clips may use before the schedule stops itself |
| `robot_name`, `trace_capacity` | auto, `6000` | rarely needed |

### Three things worth saying out loud

**Keep a name bound to the rlmcp wrapper.** `attach_runner` lives on it, and
`RslRlVecEnvWrapper` does not forward attribute lookups, so the order above
matters. Without `attach_runner` you still get parameters, metrics, video and
curricula. You lose the PPO knobs, checkpointing and `stop`.

**`render_mode=None` is fine.** rlmcp renders through the offscreen renderer for
`shot` and `video` and does not need the environment's own render mode. Note
that progress clips build that renderer at iteration 0 of every run, so its GPU
memory is now part of a normal run's footprint; `video_every=0` opts out.

**`rlmcp stop` arrives as an exception.** rsl_rl's `learn()` has no stop hook, so
the request is raised as `TrainingStopped` inside it. Catch it and save: an
aborted run that left no checkpoint has nothing to replay, and a clip of the
final policy is the deliverable.

(`TrainingStopped` is a `rlmcp.core.controller.SessionStopped`, the same signal
named for what it is rather than for the loop it usually interrupts. A play
session's viewer loop is stopped by the same object. Catching either name works.)

Launch it with your package importable:

```bash
PYTHONPATH=. MUJOCO_GL=egl uv run --project ../rl-mcp --extra mjlab \
    python mytask/train_curriculum.py --num-envs 2048 --record-run 015
```

---

## 2. Give the task a vocabulary

The core knows parameters, metrics, traces, frames and stages. It does not know
what a "goal" is, and it should not. That vocabulary goes in an `Extension`
beside your task.

Full guide: [extensions.md](extensions.md).

---

## 3. Write the ladder as a curriculum

Anything of the form "when metric X passes Y, change weight Z" is a
`CurriculumStage`, not a note to yourself.

Full guide: [curriculum.md](curriculum.md).

---

## 4. Drive the run

From another shell, while it trains. None of this pauses training.

```bash
rlmcp sessions                       # what is running
rlmcp status                         # iteration, stage, headline metrics
rlmcp metrics --list                 # every metric name this run publishes
rlmcp metrics rlmcp/goal_rate_per_min --last-n 50
rlmcp plot rlmcp/goal_rate_per_min --smooth 20

rlmcp commands                       # every verb, including your extension's
rlmcp run goal_status                # what the task itself thinks is happening
rlmcp shot --where holding=true      # look at an env that has the object
rlmcp video --seconds 8
rlmcp diagnose --seconds 4           # jerk, chatter, effort, posture, a verdict

rlmcp set reward.drop_penalty.weight -150 --why "it is bailing out early"
rlmcp note "reviewer says the gripper should approach from above"

rlmcp checkpoint before-experiment   # and `rlmcp load <path>` to undo
rlmcp pause / rlmcp resume
rlmcp stop --why "the falsifier fired"
```

Every write lands in the session event log with the reason you gave, so
`rlmcp events` afterwards reconstructs what you did and why. That includes
`note`.

Full list: [tools.md](tools.md).

---

## 5. Open a record before you launch, close it after

One session is one run. The records are the record across runs.

Full guide: [records.md](records.md).

---

## Working with an agent

Point your agent at the repository and let it read [tools.md](tools.md),
[mcp-server.md](mcp-server.md) and [tuning.md](tuning.md) — the last one is
the workflow it should follow: verify the task before training, watch the
right numbers, diagnose before touching a weight. The short version:

- Do not parse `metrics.jsonl` or `status.json` by hand. `status`, `metrics` and
  `plot` exist.
- Do not write a loop that pokes weights at milestones. That is a curriculum.
- Do not add a task-specific verb by editing core code. That is an extension.
- Do not judge smoothness from reward curves. `diagnose` measures it.
- Do not tell the user "it improved". Show them a `video` or a `plot`.
