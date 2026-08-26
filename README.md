# rl-mcp

> ### 🚀 How to use
>
> Give your coding agent one line:
>
> ```text
> Wrap my mjlab env with rlmcp (github.com/mktk1117/rl-mcp) and show me how to use it.
> ```
>
> It reads the docs here, adds the wrapper to your training script, and drives
> the run for you from there.

**An MCP server for RL training.** Point an agent at a running job and it can
**watch, diagnose and steer** the run — read the metrics, look at the robot,
change a reward weight, roll back — without restarting.

Everything the MCP server exposes is also a shell command, so `rlmcp` works
just as well with you driving. Built for [mjlab](https://github.com/mujocolab/mjlab)
today; the adapter seam is backend-neutral.

**This is still an experimental project.**

Training a policy with RL is a long feedback loop: launch, wait an hour, squint
at TensorBoard, guess which reward weight was wrong, relaunch. rlmcp shortens it.
Attach it to a run and you can ask the live process what the robot is doing, look
at it, measure its jerkiness, change a reward weight, unlock harder terrain, and
roll back if that made things worse — without restarting.

```python
from mjlab.envs import ManagerBasedRlEnv
import rlmcp

env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
env = rlmcp.wrap(env, session_dir=log_dir / "rlmcp", curriculum="terrain")

vec_env = RslRlVecEnvWrapper(env)
runner  = VelocityOnPolicyRunner(vec_env, agent_cfg, str(log_dir), device)
env.attach_runner(runner)          # PPO knobs, checkpoints, iteration boundaries

runner.learn(num_learning_iterations=agent_cfg.max_iterations)
```

That is the whole integration.

---

## What you can do to a running job
Examples:

```bash
rlmcp status                                   # iteration, stage, headline metrics
rlmcp diagnose --seconds 4                     # is the gait smooth? is it tracking?
rlmcp shot --where terrain=pyramid_stairs      # look at a robot on the stairs
rlmcp video --seconds 5 --where terrain=random_rough
rlmcp set reward.action_rate_l2.weight -0.25 --why "ankles chattering at 15 Hz"
rlmcp commands                                 # what this run accepts
rlmcp run set_terrain terrains='["flat","random_rough"]' max_level=4
rlmcp curriculum advance --why "flat is solved"
rlmcp reset-envs --where terrain=pyramid_stairs  # fresh episodes, on the stairs only
rlmcp checkpoint before-experiment             # ... and `rlmcp load` to undo
rlmcp pause / rlmcp resume / rlmcp stop
```

### It prints for whoever is reading

The CLI has two readers with opposite needs, and it tells them apart by whether
stdout is a terminal:

* **A pipe** — an agent shelling out, the MCP server, a redirect into a file —
  gets the JSON it has always got. That output is a parsing contract and does
  not move.
* **A terminal** gets the same payload laid out to be read: aligned columns,
  wrapped prose, timestamps as wall-clock. Nothing is truncated — text mode
  changes presentation, never content.

On top of that, a command that produced a picture *shows* it. `rlmcp shot`,
`plot`, `video`, `diagnose` and `record graph` hand their artifact to the best
thing available: inline in the terminal if it can draw images (kitty, Ghostty,
WezTerm, or `chafa` / `imgcat` / `viu` if installed), otherwise the desktop's
own viewer via `xdg-open`. A piped run never opens a window — an agent gets the
path and reads the file itself.

```bash
rlmcp status                 # a table if you are a person, JSON if you are a pipe
rlmcp --json status          # force JSON from a terminal
rlmcp --text status | less   # force the formatted view into a pager
rlmcp shot --no-open         # write the PNG, print the path, open nothing
```

`RLMCP_OUTPUT=json|text` and `RLMCP_OPEN=auto|never|always` set the same
things for a whole shell.

The same surface is exposed as MCP tools, so an LLM agent can do all of it —
including *seeing* the screenshots and plots, which is the point. Capabilities
specific to a kind of task arrive through extensions and are reachable via
`list_commands` / `run_command`, so the tool list adapts to the environment
instead of the environment having to match the tool list.

Video, trace and diagnose requests are deferred jobs — they collect data over
the coming rollout steps and answer when done. `cancel_job` abandons one
mid-flight, and a deferred request made while training is paused is refused
with an explanation: no steps happen while paused, so the job could never
finish.

## Curriculum: stages of parameters and commands

A stage says two things: what changes on entry, and what has to be true before
the run leaves. It says them in the only two vocabularies the core has —
parameter edits and commands — so the same structure drives a locomotion ladder
and a manipulation one:

```python
from rlmcp import Action, Condition, CurriculumStage

CurriculumStage(
    name="1_rough",
    parameters={"reward.foot_clearance.weight": -1.0},
    apply=[Action("set_terrain", {"terrains": ["flat", "random_rough"],
                                  "max_level": 5})],
    promote_when=[
        Condition("rlmcp/episode_length_frac", ">=", 0.5),   # it survives
        Condition("rlmcp/terrain_level_frac",  ">=", 0.6),   # it is progressing
    ],
    min_iterations=250,
    hold_iterations=20,
)
```

Promotion is earned, not scheduled: every condition must hold for 20 consecutive
iterations *and* the stage floor must have passed. An agent can override any of
it (`curriculum_advance`, `curriculum_goto`, `curriculum_auto false`), and a
manual command wins over the curriculum within the same iteration.

`set_terrain` above is not a core concept — it arrives from the terrain
extension. Which brings us to the part that keeps this library small.

## Extensions: capabilities the core does not know about

rlmcp's core knows about parameters, metrics, traces, frames and stages. It
knows nothing about terrain, object sets or goal distributions, because a
library that hardcodes one task's vocabulary makes every other task carry dead
weight.

A capability lives in `rlmcp/extensions/` and adds itself to a run:

| hook | what it adds |
| --- | --- |
| `commands()` | new verbs, reachable from CLI, MCP and curriculum stages |
| `metrics()` | scalars merged into telemetry, usable in promotion conditions |
| `select_envs()` | answers `where={"terrain": "stairs"}` — the core never parses it |
| `snapshot()` / `restore()` | state saved and restored with the policy |
| `bind()` | receives the controller's `ExtensionContext` after registration |
| `on_iteration()` / `close()` | a call every learning iteration, and one at shutdown |

The `ExtensionContext` — `write_artifact`, `telemetry`, `append_event`,
`submit_job`, `pending_jobs` — is how an extension reaches controller facilities. A command
handler may return a `DeferredJob` (from `rlmcp.core.controller`) to answer
over the coming simulation steps — "watch the robot for N steps and report" —
serviced exactly like the built-in video and trace commands. Extensions take
just the environment to construct; the old `(env, plot_sink)` constructor
still builds, but is deprecated in favour of `bind`.

### Adding one

Drop a module in `rlmcp/extensions/`, subclass `Extension`, decorate it, and
import the module from the package's `__init__.py` — importing is what runs the
decorator. No command table or tool list to edit:

```python
from rlmcp.core.extensions import Extension
from rlmcp.extensions import register

@register
class ObjectSet(Extension):
    """Which objects appear in a manipulation scene."""

    name = "objects"

    def available(self) -> bool:
        return hasattr(self.env, "object_pool")

    def commands(self):
        return {"set_objects": self.cmd_set_objects}

    def metrics(self):
        return {"rlmcp/object_variety": float(len(self.env.object_pool))}

    def cmd_set_objects(self, names: list[str]) -> dict:
        """Choose which objects the scene spawns."""
        self.env.object_pool = list(names)
        return {"objects": self.env.object_pool}
```

`set_objects` is now a verb: `rlmcp run set_objects names='["cube","mug"]'`,
an MCP `run_command`, and something a curriculum stage can call in `apply`.
`rlmcp/extensions/terrain.py` is the worked example.

### Keeping it in your own repo

An extension outside rlmcp is found without editing anything here — declare an
entry point and it registers on import:

```toml
[project.entry-points."rlmcp.extensions"]
my_capability = "my_project.rlmcp_ext"
```

### What loads, and when

Every extension decides for itself whether an environment supports it, so the
same call adapts:

```
$ rlmcp extensions                   # what is installed
$ rlmcp-train Mjlab-Velocity-Rough-Unitree-G1 ...
[rlmcp] extensions: terrain          →  set_terrain, terrain_status, plot_terrain

$ rlmcp-train Mjlab-Lift-Cube-Yam ...
[rlmcp] extensions: none             →  core verbs only, nothing broken
```

Pass `extensions=[...]` or `exclude_extensions=[...]` to `rlmcp.wrap` to
override that. A capability that fails to import, fails to construct, or reports
itself unavailable is left out with a note — never fatal to a training run.

## Seeing what the policy is actually doing

Iteration metrics tell you *that* a policy is bad. Traces tell you *why*.
`diagnose` records per-step joint signals for one environment and reports:

```json
{"smoothness": {"joint_jerk_rms": 8596, "chatter_measured": true,
                "hf_share_median": 0.469, "whole_body_hf": true,
                "num_buzzing_joints": 0,
                "worst_hf_joints": [{"name": "right_shoulder_yaw_joint",
                                     "hf_power_share": 0.659, "peak_hf_hz": 9.75}]},
 "tracking":   {"lin_vel_error_rms": 0.285, "commanded_speed_mean": 0.672},
 "effort":     {"torque_rms": 11.3, "hardest_working_joints": [...]},
 "posture":    {"tilt_deg_mean": 4.2, "base_height_std": 0.044},
 "gait":       {"contact_fraction": 0.51, "mean_air_time_s": 0.32},
 "verdict":    ["The whole body carries high-frequency motion (median 0.47 of
                 joint-velocity power above 8.0 Hz across 29 joints) ... raise
                 action_rate_l2 or lower action.joint_pos.scale_gain rather than
                 chasing one joint",
                "Linear velocity tracking is reasonable (RMS error 0.28 m/s)"]}
```

Chatter is measured as the share of joint-velocity power *above* the gait band,
not the dominant frequency, because buzzing rides on top of a healthy 1–3 Hz gait
instead of replacing it. A joint is named only when it stands out against the
robot's own median — on a real 50 Hz humanoid every joint carries some
high-frequency content, and a list of 27 "bad" joints is worth nothing. When the
median itself is high, that is reported as a whole-body problem with a different
fix; and when a trace is too short or too slow to support the measurement at
all, the report says `chatter_measured: false` with the reason rather than
reading as clean. Plots (`plot_joint_trace`), raw `.npz` traces, screenshots and MP4 clips all
land in the session's `artifacts/` directory; `rlmcp analyze <trace.npz>` re-runs
the analysis after the trainer has exited. Traces are saved pickle-free; analyzing
a trace from an older rlmcp needs `--allow-legacy`, which is an explicit opt-in
because unpickling a file you did not write runs code.

Clips are captured during ordinary rollout steps, so a video shows real training
behaviour, not a separate evaluation.

## Playing a finished run

Everything above talks to a *live* trainer. Once the run has exited, the only
evidence left is a checkpoint on disk — and `rlmcp play` is how you look at it:

```bash
rlmcp play                                     # newest run, last checkpoint, 8s clip
rlmcp play logs/rsl_rl/my_run/model_final_4375.pt --device cpu
rlmcp play --stage 2_hardest --seconds 12      # the rung this checkpoint was on
rlmcp play --mode native                       # MuJoCo's own viewer
rlmcp play --mode viser                        # a viewer in the browser
```

`--mode video` renders offscreen, so it works over ssh and in a cron job, and
`--device cpu` lets it run while the GPU is busy training. The clip lands in the
run's `artifacts/` directory and is opened for you in a terminal, exactly like
`rlmcp video`. `--mode native` and `--mode viser` need a viewer to be installed;
without one they say so and point back at `--mode video`.

**The part that matters is what `play` does before it renders.** A checkpoint
remembers weights and an iteration number. It does not remember the curriculum
rung it was climbing — and a task's `play=True` config is rung zero by
construction. Replaying a late checkpoint against a fresh config therefore shows
a policy failing at a task it was never asked to do, which looks exactly like a
bad policy. That is the worst kind of wrong evidence: legible, confident, and
accusing the wrong thing.

So `play` folds the session's event log first. Every curriculum entry recorded
the parameters it set and the commands it ran; every live `rlmcp set` recorded
its key and new value. Replaying those in order puts the environment back where
the policy left it, and the result payload says exactly what was restored.
`--stage <name>` stops the fold at the end of a named rung, which is what you
want for a checkpoint saved during it; `--set key=value` overrides afterwards,
so it wins; `--no-replay` turns the whole thing off.

If something cannot be restored — a command this build of the task no longer
has, a parameter it no longer accepts — `play` refuses to render and says which
fix applies to which problem, because "unknown command" and "the command changed
its arguments" send you to look in different places. `--allow-partial` renders
anyway when you have decided the difference does not matter.

The reconstruction lives in `rlmcp/core/replay.py` and imports no simulator, so
it is a plain parse-and-fold over the event log.

Two smaller things worth knowing. The task has to be registered before its
config can be loaded, so pass `--task-package <module>` exactly as you would to
`rlmcp train` — or set `RLMCP_TASK_PACKAGES` once for the shell. And a play
session is itself a session: it publishes into `<run>/rlmcp/play/<stamp>/` and
can be steered from another shell while you watch it, but it is marked as a play
session so it never becomes the answer to "the latest session here".

### Steering the thing you are watching

A play session answers the same commands a training run does — `set`, `run`,
`shot`, `video`, `trace`, `diagnose`, `metrics`, `pause`, `resume` — against
the directory it printed at startup. Three of them are what make watching a
policy an investigation rather than a screening:

```bash
rlmcp --session <play dir> reset-envs                    # see the opening again
rlmcp --session <play dir> run load_policy checkpoint=model_2000.pt
rlmcp --session <play dir> stop                          # close it from a shell
```

`reset-envs` restarts episodes, so you can watch the start of a behaviour
instead of waiting for the robot to fail into one; it is a core command and is
just as useful mid-training. `stop` ends the session the way closing the window
does — the viewer unwinds, the session records its end, and nobody is shown a
traceback for having asked.

`load_policy` is the interesting one. The point is comparison: the conditions
you restored and the camera you set up survive, and only the weights change.
It refuses without touching the running policy if the file is missing or if the
weights load but cannot act on this environment — a swap that half-applies is
worse than one that does not happen. And because a checkpoint from another run
may have been trained at another rung, the response always names the stage it
trained at next to the stage the environment is in. When those disagree the
conditions are deliberately *not* changed — comparing two policies needs the
environment to hold still — and the payload says so loudly. Pass `replay=true`
to restore that checkpoint's own conditions instead. `run list_policies` names
the checkpoints sitting beside the one currently acting.

## What can be tuned live

`list_parameters` discovers everything from the running environment — for the G1
rough task that is 97 knobs:

| category | examples |
| --- | --- |
| `reward` | `reward.foot_slip.weight`, `reward.foot_clearance.params.target_height` |
| `termination` | `termination.fell_over.params.limit_angle` |
| `domain_randomization` | `event.interval.push_robot.params.velocity_range.x` (a `[min, max]` range) |
| `curriculum` | `command.twist.ranges.lin_vel_x` |
| `action` | `action.joint_pos.scale_gain` — a direct lever on jerkiness |
| `rl` | `rl.learning_rate`, `rl.entropy_coef`, `rl.clip_param`, `rl.desired_kl` |

Values are shape- and type-checked, applied between rollout batches (never
mid-step), and every change is written to the session event log with the
rationale the caller gave. `reset_parameters` restores the startup values.
(`rlmcp reset-envs` is the other reset and does an unrelated thing: it starts
fresh *episodes*, leaving every parameter where it is. Narrow it with
`--env-id` or with the same `--where` query `shot` takes.)
Magnitudes are *not* second-guessed. The only bounds the adapter declares are
the ones true by a parameter's definition — `gamma` and `lam` live in `[0, 1]`,
a learning rate or a gradient-norm ceiling cannot be negative. Everything else
is unbounded, because only the task knows the scale it works at: a reward
weight of 600, an action gain of 10 and a clip epsilon of 0.8 are all
legitimate experiments, and a limit invented here would refuse them while
catching nothing.

Every parameter also carries a *liveness*: `live` values take effect next
batch; `at_reset` values are re-read on episode reset, and the write's response
says so; writes to `at_startup` or `inert` knobs — values the environment will
never re-read during this run — are refused with an explanation instead of
reporting a success that changes nothing.

> Setting `rl.learning_rate` also switches PPO's schedule to `fixed` — the
> adaptive schedule would otherwise overwrite the value within one iteration. The
> tool says so in its response rather than silently doing nothing.

## How it fits together

```
training process (torch, mujoco)          session directory            agent process (no simulator)
┌──────────────────────────────┐          ┌────────────────┐          ┌──────────────────────────┐
│ RlMcpEnvWrapper             │          │ status.json    │          │ MCP server (34 tools)    │
│  ├ on_step()   trace frames  │ ───────► │ metrics.jsonl  │ ◄──────► │ rlmcp CLI               │
│  └ service()   run commands  │          │ events.jsonl   │          │ your own scripts         │
│     ├ SimAdapter  (env)      │ ◄─────── │ inbox/ outbox/ │          └──────────────────────────┘
│     ├ RunnerAdapter (PPO)    │          │ artifacts/     │
│     └ Extensions (optional)  │          │                │
└──────────────────────────────┘          └────────────────┘
```

The two processes talk through a plain directory of JSON files. That is a
deliberate choice:

* the MCP server never imports torch or mujoco, and starting or killing it cannot
  disturb training;
* commands execute **between rollout batches**, so a parameter edit can never race
  the simulator;
* everything is inspectable with `cat` when something goes wrong;
* a run leaves a complete, replayable record behind after it exits.

Commands that need the robot to move — video, traces, diagnosis — are deferred:
the request stays open, `on_step` feeds it frames, and the response is written when
the job completes. Everything else answers within one iteration.

## Records: the runs, and what they showed

One session is one run. `rlmcp record` keeps the record *across* runs: every run
gets a numbered record with its hypothesis, config snapshot, lineage (which run
it evolved from, which checkpoint it warm-started from), and a close-out verdict
backed by measurements.

The records themselves do not live in this repository. They are a directory that
belongs beside your own task packages, and `$RLMCP_RECORDS` is what points every
`rlmcp record` command at it — see
[Using it on your own task](#using-it-on-your-own-task).

```bash
rlmcp record new "lower action scale stops the arm buzz" --parent 011
rlmcp record close 012 validated --outcome "buzz gone at matched tracking" \
    --metric "hf_share_median=0.12"
rlmcp record graph     # the lineage, as an interactive page
```

A run can pre-register a **falsifier** — the condition that would disprove its
hypothesis, and the iteration before which reading it means nothing. Training
prints the moment it fires; close-out records it honestly (`FIRED`, `held`, or
`not evaluable yet` when the run died too early). Verdicts need evidence: a
`validated` close without a measurement is refused, and a warm-started run
cannot claim `validated` at all. Records are plain files — the store assigns
ids transactionally, never reuses them, and two writers cannot silently
overwrite each other.

### Feedback: the steering, kept

Most of what a human contributes to a training run is said, not measured — "the
motion looks jittery near the end of an episode", "stop tuning the entropy
coefficient", "never let the torque limit past 80% again". Said in a chat
window, it is gone when the window closes. `rlmcp` records it against the run
instead:

```bash
rlmcp feedback "it looks jittery near the end" --kind observe   # to the live trainer
rlmcp record feedback 012 "stop tuning the entropy coefficient" --kind correct
rlmcp record answer 012 0 "checked it; already at the default" --no-change
rlmcp record timeline --markdown            # the whole ledger, oldest first
```

Feedback is append-only, and each entry carries a slot for what was done about
it. Six kinds — `steer`, `correct`, `reject`, `approve`, `observe`, `constrain`
— of which the first four ask for something, so an entry of that kind with no
recorded response is *unanswered* and says so: in the timeline, in `REPORT.md`,
and in `rlmcp record check`. `--no-change` is a real answer; "looked into it,
nothing needed changing" is not the same as ignoring it, and the ledger keeps
the two apart.

`rlmcp record timeline --markdown` renders the ledger: a table of every remark
with what it was read as and what changed, then the same in full. It is
generated from the records every time, so it cannot drift from them.

## The lineage graph, and the page that draws it

A list of runs tells you what you did. It does not tell you what you learned,
because the thing you learned lives in the *differences* between runs. So the
records are kept as a graph, and `rlmcp record graph` draws it:

```bash
rlmcp record graph                         # writes records/lineage.html, opens it
rlmcp record graph --png --out tree.png    # a picture an agent reads in one call
```

The HTML is one self-contained file. No server, no build step — open it from a
path. Click a node and you get what that run claimed, what it measured, and the
**recipe**: every config change folded from the root down to that node. The
recipe is never stored, only computed, so it cannot drift from the edges.

Two kinds of edge, because they are two different claims:

| edge | means |
| --- | --- |
| **config** (`--parent`) | this run's settings are the parent's, plus a listed change |
| **warm start** (`--weights`) | this run started from that run's policy weights |

They are drawn separately because they often disagree. A run can take its config
from the run before it and its weights from six runs back, and a tool that draws
one edge from the other data will draw it wrong.

### What the graph shows

The value is not the arrows, it is the verdicts on them. An illustrative shape,
of the kind these records tend to take:

```
001 baseline ............... falsified   the policy found a degenerate optimum
 └─ 002 reshaped reward .... falsified   optimum broken, but it overshot the other way
     └─ 003 re-priced ...... falsified   so pricing was never the binding constraint
         └─ 004 new actuation interface ... validated   ← the unlock
             └─ 005 tighter tolerance ..... interrupted  stopped to fix the gate
                 └─ 006 realistic tolerance .... validated   solved
```

**What this means:** four of these six runs did not work, and the graph is what
made that affordable.

Runs **001, 002 and 003** are the interesting part. Each was falsified, and
together they are worth more than any one success would have been. The first
found a degenerate optimum; the second broke it and overshot the opposite way;
the third priced the trade in between and *also* failed. Three failures, one
conclusion the graph makes unavoidable — the reward was never the binding
constraint — and 003 had pre-registered exactly that alternative in its
falsifier. 004 changed only that, and the task opened up.

That conclusion is not visible in any single record. It lives in the shape of
three siblings, which is the argument for keeping runs as a graph rather than a
list.

`interrupted` is not a failure of anything, which is why it is a separate
verdict: a run stopped for a setup mistake or a deliberate change of plan must
not be counted as a hypothesis that lost. And `falsified` is a good verdict — a
run that kills its hypothesis in twenty minutes is information-dense, and the
graph is what turns three of them in a row into an answer.

## Install

```bash
pip install -e .                # training side: numpy, matplotlib, pillow, imageio
pip install -e '.[server]'      # adds the MCP SDK for the agent side
```

The MCP SDK is optional on purpose — the training process does not need it, and
the server does not need a simulator. Both `mcp>=2` (`MCPServer`) and `mcp` 1.x
(`FastMCP`) are supported.

Register the server with Claude Code:

```bash
claude mcp add rlmcp -- rlmcp-server --root /path/to/logs
```

[docs/mcp-server.md](docs/mcp-server.md) walks the whole tool surface with
example calls and a worked steering session.

With no `--session`, the CLI attaches to the newest session under its roots on
each invocation. The server instead **pins** one session — the newest under
`--root` at startup — and never silently changes runs: `switch_session` is the
one way to move it, and `list_sessions` shows what exists. After the trainer
exits, the server's data tools keep answering from the files the run left
behind (only commands that need a live process refuse, and say so), so a
post-mortem works like a live inspection.

## Try it

```bash
# train, with the terrain ladder driving itself
rlmcp-train Mjlab-Velocity-Rough-Unitree-G1 --num-envs 4096

# or the explicit-curriculum example
python examples/train_g1_rough_curriculum.py --num-envs 4096
```

then, from another shell:

```bash
rlmcp sessions      # what is running
rlmcp status
rlmcp curriculum
rlmcp diagnose --seconds 4
```

## Using it on your own task

Everything above drives a task that ships with mjlab. Your own task is not one
of those, and it does not belong in this repository. The arrangement that works
is two repositories:

```
rl-mcp/                  the tooling — harness, adapters, CLI, MCP server (this repo)

your-tasks/              your work
  mytask/                an ordinary mjlab task package: env cfg, mdp terms, assets
    rlmcp_ext.py           the task's vocabulary, as an rlmcp Extension
    curriculum.py          the ladder, as a StageSchedule
    train_curriculum.py    the launcher that calls rlmcp.wrap
  records/               the run records — written and read by `rlmcp record`
```

rlmcp is a dependency of your repository, never the other way round. The two
are split because they have different lifecycles: a tool's history should be
about the tool, and run records carry training logs, plots and video — large,
regenerable, and silent about the harness. Four runs of one task came to 24 MB.

`$RLMCP_RECORDS` is what holds that line at runtime. The records root resolves to
the explicit argument first (`rlmcp record --records-root`, `rlmcp.wrap(records_root=)`),
then `$RLMCP_RECORDS`, then `./records`. Set the variable once and every `rlmcp record`
command writes into the same records from any directory; leave it unset and you
get a scratch `records/` wherever you happened to be standing.

```bash
pip install rl-mcp                    # the tooling
export RLMCP_RECORDS=$PWD/records     # point the harness at your records
```

The examples below use `mytask/` for the task package and `mytask_ext.py` for
its extension. Nothing about the arrangement is specific to a task: any mjlab
task package works the same way.

### 1. Wrap the environment

One call, between building the environment and building the runner. A whole
launcher is about a hundred lines; this is one with the argument parsing cut
out:

```python
import rlmcp
from rlmcp.adapters.mjlab import TrainingStopped

import mytask                 # registers the task with mjlab
import mytask.rlmcp_ext      # registers the extension — importing is what registers
from mytask.curriculum import build_curriculum

env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
env = rlmcp.wrap(
    env,
    session_dir=log_dir / "rlmcp",
    curriculum=build_curriculum(),
    task_id=args.task,
    service_every_steps=agent_cfg.num_steps_per_env,   # service once per iteration
    record_run=args.record_run or None,                      # the record this run fills
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

Three things there are worth saying out loud.

**Keep a name bound to the rlmcp wrapper.** `attach_runner` lives on it, and
`RslRlVecEnvWrapper` does not forward attribute lookups, so the order above
matters. Without `attach_runner` you still get parameters, metrics, video and
curricula; you lose the PPO knobs, checkpointing and `stop`.

**`render_mode=None` is fine.** rlmcp renders through the offscreen renderer
for `shot` and `video` and does not need the environment's own render mode.

**`rlmcp stop` arrives as an exception.** rsl_rl's `learn()` has no stop hook,
so the request is raised as `TrainingStopped` inside it. Catch it and save: an
aborted run that left no checkpoint has nothing to replay, and a clip of the
final policy is the deliverable. (`TrainingStopped` is a
`rlmcp.core.controller.SessionStopped`, which is the same signal named for what
it is rather than for the loop it usually interrupts — a play session's viewer
loop is stopped by the same object. Catching either name works.)

Launch it with your package importable:

```bash
PYTHONPATH=. MUJOCO_GL=egl uv run --project ../rlmcp --extra mjlab \
    python mytask/train_curriculum.py --num-envs 2048 --record-run 015
```

### 2. Give the task a vocabulary

The core knows parameters, metrics, traces, frames and stages. It does not know
what a "goal" is, and it should not — so that vocabulary goes in an `Extension`
beside your task, not in here. A worked example, for a task where the robot
moves objects to goal poses:

```python
from rlmcp.core.extensions import Extension
from rlmcp.extensions import register

@register
class Goals(Extension):
  """Difficulty control and task-level metrics for a goal-reaching task."""

  name = "goals"

  def available(self) -> bool:
    return "goal" in self.env.unwrapped.command_manager._terms

  def commands(self):
    return {"set_goal_difficulty": self.cmd_set_goal_difficulty,
            "goal_status": self.cmd_goal_status}

  def metrics(self):
    # Per minute of simulated time, not per episode -- see below.
    return {"rlmcp/goal_rate_per_min": self._goal_rate()}

  def select_envs(self, **criteria):
    """Understands `holding=true`."""
    holding = criteria.pop("holding", None)
    if criteria or holding is None:
      return None                       # not our vocabulary; let others answer
    want = str(holding).lower() in ("1", "true", "yes")
    return [i for i, held in enumerate(self._holding()) if held == want]
```

Those three hooks buy three different things at once, from one class:

* `commands()` — `rlmcp run set_goal_difficulty tolerance=0.2 mode=random`
  from a shell, `run_command` over MCP, **and** an `Action` a curriculum stage
  can apply on entry. Values are parsed as JSON, so quote lists:
  `rlmcp run set_objects names='["cube","mug"]'`.
* `metrics()` — merged into telemetry, so they appear in `rlmcp metrics`,
  `rlmcp plot`, and in curriculum promotion conditions.
* `select_envs()` — `rlmcp shot --where holding=true`. The core never parses a
  `where` key; your extension is the only thing that knows what one means.

`select_envs` is dispatched **by signature**: the registry tries to bind the
`--where` keys as keyword arguments and silently moves on to the next extension
when they do not fit. So declare it as `**criteria`, pop what you understand,
and return `None` if anything is left over. A handler written as
`select_envs(self, where: dict)` never binds and is skipped without a word.
`rlmcp/extensions/terrain.py` is the reference.

Write the docstrings for a reader who cannot see the code. A command's docstring
is what `rlmcp commands` prints, and it is often all an agent has before
calling it.

Two ways to make an extension load. Importing the module registers it, which is
what the launcher above does — enough for a script you control. To have it found
without any import, publish an entry point from your `pyproject.toml`:

```toml
[project.entry-points."rlmcp.extensions"]
mytask = "mytask.rlmcp_ext"
```

Either way `available()` decides whether it actually attaches, so an extension
that does not fit the running environment is left out with a note instead of
breaking the run. `rlmcp extensions` lists what is installed.

### 3. Write the ladder as a curriculum

If you catch yourself writing "when metric X passes Y, change weight Z" into a
notes file, that is a `CurriculumStage`. Encoding it means the run drives itself,
the ladder is saved with the run (`params/curriculum.json`), and a human can
still override it live.

A non-terrain rung, from a manipulation ladder that starts with the object
already in the gripper and works up to picking it off a table:

```python
CurriculumStage(
    name="1_hold_and_place",
    apply=[Action("set_goal_difficulty",
                  {"spawn_mode": "in_gripper", "tolerance": 0.15})],
    parameters={"reward.reach_goal.weight": 300.0,
                "reward.grasp_stability.weight": 60.0,
                "reward.drop_penalty.weight": -120.0},
    promote_when=[
        Condition("Metrics/goals/goals_per_min", ">=", 25.0),
        Condition("Metrics/goals/episode_dropped", "<=", 0.5),
    ],
    min_iterations=500,
    hold_iterations=20,
    notes="Goal: keep hold of it and place it, before learning to pick it up.",
)
```

`apply` speaks your extension's verbs; `parameters` speaks the keys `rlmcp
params` discovered. Both are written to the event log when the stage is entered,
which is what lets the lineage page draw what changed and when.

**Pick promotion metrics the policy cannot buy.** Two rules, both usually
learned the expensive way:

* *Task-external, not the reward being optimised.* A reward term rises when the
  policy games it. Successes per minute, success rate and holding fraction do
  not.
* *Normalised — rates and fractions, never per-episode counts.* An episode that
  ends early on a failure is shorter for reasons unrelated to skill, so a raw
  count promotes whatever survives longest. Divide counts by
  `episode_length_buf * step_dt` and publish the rate.

A stage promotes only when every condition has held for `hold_iterations`
consecutive iterations *and* `min_iterations` has passed. You can override that
from a shell at any time:

```bash
rlmcp curriculum                     # which rung, and what it is waiting for
rlmcp curriculum advance --why "the easy tolerance is solved"
rlmcp curriculum goto 3_from_table --why "the warm start already places reliably"
rlmcp curriculum auto-off            # stop it promoting itself
```

### 4. Drive the run

From another shell, while it trains. None of this pauses training: video, traces
and diagnosis are deferred jobs serviced between rollout batches, and parameter
edits apply between batches too, so nothing here can race the simulator.

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

Every one of those writes lands in the session event log with the rationale you
gave, so `rlmcp events` after the fact reconstructs what you did and why. That
includes `note` — a remark a human made about the run belongs in the record, not
in a side file.

### 5. Open a record before you launch, close it after

One session is one run; the records are the record across runs. Open it *before*
launching, while the prediction still costs something to write down:

```bash
rlmcp record new "fade the assist on hand tracking not catches" \
    --hypothesis "catch rate is the helper's achievement while assist is on" \
    --prediction "assist reaches 0 by iteration 1500, tracking error under 0.10 m" \
    --falsifier "assist is still above 0.5 at iteration 2000" \
    --falsify-when rlmcp/assist '>=' 0.5 --check-after 2000 \
    --parent 014
```

The positional argument is a slug — it names the record's directory, so keep it
short. `--falsify-when METRIC OP VALUE` is the machine-checked twin of
`--falsifier`: training prints the moment it fires, and close-out records it as
`FIRED`, `held`, or `not evaluable yet` if the run died first. `--check-after`
is not optional in spirit — every policy is bad at iteration zero, and a
falsifier with no floor fires during the warm-up.

Pass the id to the launcher (`--record-run 015` above, reaching
`rlmcp.wrap(record_run=...)`) so the config snapshot, events and metrics land on
the record. Then close it:

```bash
rlmcp record close 015 validated \
    --outcome "assist faded to 0 by iteration 1400; catch rate held at 62/min" \
    --metric "rlmcp/assist=0.0" "rlmcp/catch_rate_per_min=62.1"
rlmcp record asset 015 run.mp4 --kind videos --caption "final policy"
rlmcp record graph          # the lineage, as an interactive page
```

`--metric` takes every measurement in one go; a second `--metric` flag replaces
the first rather than adding to it. A `validated` verdict with no measurement is
refused, and a warm-started run cannot claim `validated` at all — its result is
not attributable to its own change. The rest of the vocabulary is `falsified`,
`provisional`, `control`, `superseded`, `best` and `interrupted`, with `planned`
and `running` meaning the run has not produced a result yet.

Attach the clip. A video that stays in `logs/` is not a deliverable — `record asset`
copies it into the records' own media store, so the record still has it after
the training logs are cleaned, and the lineage page has something to show.

## How parameters are found

Nothing hand-lists them. mjlab's manager term configs are ordinary dataclasses
and dicts, so rlmcp walks them and emits a key for every leaf that is a number,
a bool or a `[min, max]` pair:

```
rlmcp/adapters/mjlab/
  sim_adapter.py          # thin: implements SimAdapter by delegating
  access/                 # parameters: discovery, reads, writes
    paths.py              #   the reflective core — walk, resolve, coerce
    base.py               #   what a provider supplies
    rewards.py  terminations.py  events.py  commands.py  actions.py
  state/                  # live state: sampling, metrics, rendering
```

Capabilities live outside the adapter entirely, in `rlmcp/extensions/`, since
they are not specific to one backend's parameter plumbing.

A provider answers one question — *which objects hold tunable values, and what
are they called* — in 20–90 lines. Adding a family means adding a file and one
entry in `PROVIDER_TYPES`; discovery, get/set, bounds, the CLI and the MCP tools
all pick it up unchanged.

Two consequences worth knowing:

- **New config fields appear on their own.** A task that adds a reward term, or
  an mjlab release that adds a field, shows up without touching rlmcp. On the
  G1 rough task this surfaced 97 parameters where the previous hand-written
  version exposed 60.
- **Keys escape dots.** mjlab keys the G1 pose-std dict by joint-name regex, so
  that parameter is `reward.pose.params.std_walking.\.*knee\.*`. Segments are
  escaped on the way out and split on unescaped dots on the way back.

Providers can hide fields that change *semantics* rather than magnitude — a
termination's `time_out` flag tells the algorithm whether to bootstrap the
value, and is a trap rather than a knob — via `skip_keys`.

## Using it with a different simulator

rlmcp targets **manager-based environments whose config is live** — mjlab,
IsaacLab, and anything built the same way: term configs are ordinary
dataclasses and dicts that the running environment re-reads, which is what
makes "change a weight and it takes effect next batch" true.

Within that family: implement `SimAdapter` (and optionally `RunnerAdapter`)
from `rlmcp/adapters/base.py`. Only `discover_parameters`, `get_parameter` and
`set_parameter` are required; everything else — rendering, traces, terrain
control — has a default that reports "not supported by this backend", so tools
degrade with an explanation instead of crashing. `rlmcp/adapters/mjlab/` is the
reference implementation, and its `access/` package is reusable for any backend
whose config is dataclasses and dicts. Anything your simulator has and others do
not belongs in an `Extension` rather than in the contract.

## Tests

```bash
pytest tests -q     # 385 tests, ~4s, no GPU and no simulator required
```

The suite runs the real controller against a fake simulator, covering the command
protocol, promotion rules, chatter detection and checkpoint round-trips; the
MCP-server tests (13 of them) skip unless the optional `mcp` package is
installed.

## License

Apache-2.0 — see [LICENSE](LICENSE).
