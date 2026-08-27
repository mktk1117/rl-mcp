# Tool reference

Every rlmcp capability, one entry each. Each entry gives the shell command, the
MCP tool an agent calls, what comes back, and the traps.

## Three ways to call the same thing

| you are | you use | example |
| --- | --- | --- |
| a person at a terminal | the `rlmcp` CLI | `rlmcp status` |
| an LLM agent | an MCP tool | `get_training_status {}` |
| anything, for a verb the core does not know | `run` / `run_command` | `rlmcp run set_terrain max_level=4` |

They all end up in the same place: a request file in the session directory that
the training process picks up between rollout batches. Nothing here can race
the simulator.

## Global CLI flags

These work on every command.

| flag | meaning |
| --- | --- |
| `--session PATH` | talk to this session |
| `--root DIR` | where to search for the newest session |
| `--json` / `--text` | force machine output or human output |
| `--open` / `--no-open` | show images and clips, or just print their paths |
| `--timeout SECONDS` | how long to wait for the trainer (default 120) |

### Which run a bare command talks to

`--session` wins, then `$RLMCP_SESSION`. Failing both, it takes the newest
session under the first root that has one: `--root`, then `$RLMCP_ROOT`, then
`./logs`, `./rlmcp_session`, `.`. Those are the same two variables the MCP
server reads, so one export steers both. When nothing is found, the error names
the directory it searched and where that directory came from.

### Environment variables

| variable | does |
| --- | --- |
| `RLMCP_SESSION` | pin one session directory |
| `RLMCP_ROOT` | where to search for sessions |
| `RLMCP_OUTPUT` | `json` or `text`, overriding the guess |
| `RLMCP_OPEN` | `auto`, `never` or `always` |
| `RLMCP_RECORDS` | the records directory |
| `RLMCP_TASK_PACKAGES` | modules to import so their tasks register (`play`) |

### What a pipe gets

Output mode is guessed from stdout: a terminal gets tables and wrapped text, a
pipe gets JSON. `--text` never truncates anything, it only lays it out.

Each command's JSON is a parsing contract and does not move. It is one contract
*per command*, though, not one shape across the CLI. The top level follows where
the answer came from:

| where the answer comes from | top level | commands |
| --- | --- | --- |
| session files on disk | the payload itself — an object, or a list for `sessions` and `events` | `status`, `info`, `sessions`, `events`, `params`, `metrics --list`, `extensions`, `record list`, `record timeline` |
| the training process, or an offline stand-in for it | `{"ok": true, "result": …}`, or `{"ok": false, "error": "…"}` plus hints | `get`, `set`, `metrics`, `plot`, `shot`, `video`, `curriculum`, `run`, … and `play`, `analyze` |
| the record store | `"ok"` beside the record's own keys | the other `record …` subcommands |

**For a parser:** look for a top-level `"ok"`. Absent, the payload is the whole
output. Present and the command is not `record`, take `"result"` on success and
`"error"` on failure. Every command's family is pinned by
`tests/test_cli_output.py`, so none of it can drift silently.

## Quick index

| what you want | command | MCP tool |
| --- | --- | --- |
| what runs exist | [`sessions`](#sessions) | `list_sessions` |
| point at another run | (use `--session`) | `switch_session` |
| how is it doing | [`status`](#status) | `get_training_status` |
| numbers over time | [`metrics`](#metrics) | `list_metrics`, `get_metrics` |
| a graph of them | [`plot`](#plot) | `plot_metrics` |
| a picture of the robot | [`shot`](#shot) | `take_screenshot` |
| a clip of the robot | [`video`](#video) | `record_video` |
| clips taken automatically | [`video --every`](#video) | `set_progress_video` |
| why does it move badly | [`diagnose`](#diagnose) | `diagnose_motion` |
| raw per-step signals | [`trace`](#trace), [`plot-trace`](#plot-trace), [`analyze`](#analyze) | `record_trace`, `plot_joint_trace` |
| what can I tune | [`params`](#params), [`get`](#get) | `list_parameters` |
| change a weight | [`set`](#set), [`reset`](#reset) | `set_parameter`, `reset_parameters` |
| restart episodes | [`reset-envs`](#reset-envs) | `reset_environments` |
| task-specific verbs | [`commands`](#commands), [`run`](#run) | `list_commands`, `run_command` |
| the stage ladder | [`curriculum`](#curriculum) | `curriculum_status`, `curriculum_advance`, `curriculum_goto`, `curriculum_auto` |
| undo an experiment | [`checkpoint`](#checkpoint), [`load`](#load) | `save_checkpoint`, `list_checkpoints`, `rollback_to_checkpoint` |
| pause, resume, stop | [`pause`](#pause-resume-step-once) | `pause_training`, `resume_training`, `stop_training` |
| what was done to this run | [`events --interventions`](#events) | `get_events` |
| write something down | [`note`](#note), [`feedback`](#feedback), [`events`](#events) | `add_note`, `record_feedback`, `get_events` |
| watch a finished run | [`play`](#play) | (CLI only) |
| what code a run used | [`record code`](records.md#what-the-code-was-the-other-half-of-a-recipe) | (CLI only) |
| the record across runs | [`record …`](records.md) | `attach_feedback`, `answer_feedback`, `get_feedback_timeline`, `set_record_headline` |

---

# Watching a run

## `sessions`

List runs under the root, newest first, with `running` / `stalled` / `dead`.

```bash
rlmcp sessions
```

**MCP:** `list_sessions()` — same rows, plus which one this server is pinned to.
`switch_session({"target": "newest"})` or a directory path moves the pin. The
server never changes runs on its own; the CLI attaches to the newest run on
every invocation.

## `status`

The one command to start with.

```bash
rlmcp status
```

**MCP:** `get_training_status()`

Returns: `iteration`, `total_env_steps`, `num_envs`, `paused`, `headline_metrics`
(everything under `rlmcp/` plus mean reward and episode length), `curriculum`
(stage and promotion progress), `pending_jobs`, `extensions`, and a liveness
`state`.

It reads the trainer's published heartbeat, so it answers even while the
training loop is busy, and after the run has died.

## `info`

Static facts about the session: task id, device, step time, when it started.

```bash
rlmcp info
```

## `metrics`

```bash
rlmcp metrics --list                          # every name this run publishes
rlmcp metrics --list --contains reward
rlmcp metrics Train/mean_reward --last-n 50
rlmcp metrics --offline                       # read metrics.jsonl, skip the trainer
```

**MCP:** `list_metrics({"contains": "..."})`, `get_metrics({"names": [...], "last_n": 30})`

`get_metrics` returns recent values plus a trend summary per metric, so you do
not have to eyeball a list of floats.

## `plot`

Draw metrics to a PNG.

```bash
rlmcp plot Train/mean_reward rlmcp/episode_length_frac --smooth 20 --last-n 400
```

**MCP:** `plot_metrics({"names": [...], "last_n": 400, "smooth": 5, "title": "..."})`
returns the image inline plus its path.

In a terminal the PNG opens by itself: inline if your terminal can draw images
(kitty, Ghostty, WezTerm, or `chafa` / `imgcat` / `viu`), otherwise through
`xdg-open`. Piped output never opens a window.

## `events`

The run's audit trail: every parameter change with its reason, every stage
change, checkpoint, note and falsifier event.

```bash
rlmcp events --last-n 50
rlmcp events --interventions     # only what somebody *did*, with the reasons
```

`--interventions` is the log with the bookkeeping taken out. A run's events are
mostly things that *happened* -- a clip rendered, a job finished, a telemetry
key dropped -- and what you usually want is the handful of things somebody
*decided*: parameter edits, stage changes, restored checkpoints, restarted
environments, notes and feedback, each phrased in one line with the reason
given at the time:

```json
{"iteration": 900, "kind": "set_parameter", "layer": "parameter",
 "what": "rl.entropy_coef: 0.005 → 0.01", "why": "entropy had collapsed"}
```

That is the third layer of a run's history -- the one that is neither its code
nor its config, and the one that most often explains a result.

**MCP:** `get_events({"last_n": 25})`

## artifacts

Everything the run has written: PNGs, MP4s, `.npz` traces.

**MCP:** `list_artifacts()`. From a shell, look in `<session>/artifacts/`.

---

# Looking at the robot

All four commands below record during normal rollout steps, so what you see is
real training behaviour, not a separate evaluation. `video`, `trace` and
`diagnose` are **deferred jobs**: they collect data over the coming steps and
answer when done.

Every one takes `--env-id N` or `--where key=value` to pick which robot. The
`where` vocabulary comes from the run's extensions — the core does not know what
`terrain=pyramid_stairs` means, the terrain extension does. See
[extensions.md](extensions.md).

## `shot`

One frame, right now.

```bash
rlmcp shot
rlmcp shot --where terrain=pyramid_stairs level=2
rlmcp shot --no-open                     # write the PNG, print the path
```

**MCP:** `take_screenshot({"where": {"terrain": "pyramid_stairs"}})` returns the
image inline plus the path. Large images are downscaled to fit MCP message
limits; if one still does not fit you get the path and a note.

## `video`

```bash
rlmcp video --seconds 8 --where terrain=random_rough
```

**MCP:** `record_video({"seconds": 4.0, "where": {...}})`

### Progress clips: the ones you do not have to ask for

Every training run films itself. A clip is taken at **iteration 0**, and after
that at gaps that double -- 50, 100, 200, 400, 800, 1600 -- so the clips are
minutes apart while the behaviour is changing fastest and thin out on their own
later. Each is copied into the run record captioned with its iteration, so a
finished record holds the run's trajectory as something you can watch, in
order, without anyone having remembered to type `rlmcp video` at the right
moment.

```
0   50   100   200   400   800   1600   3200   5200   7200 …
└────────── gaps double ─────────┘      └─── every 2000 ──┘
```

The gap stops doubling at 2 000 so a long run still shows what it is doing
*now* rather than a clip from days ago.

```bash
rlmcp video --schedule            # cadence, next iteration, clips taken, MB spent
rlmcp video --every 50            # flat: every 50 iterations from here
rlmcp video --every double:200    # doubling, starting at 200
rlmcp video --every 0             # off
rlmcp video --budget-mb 500       # let the clips use more disk
rlmcp-train <task> --video-every double --video-seconds 4      # at launch
```

`--video-every` / `--every` / `wrap(video_every=…)` / `set_progress_video` all
take the same value: `double`, `double:<first>:<cap>`, a flat interval like
`200`, or `0`. Anything else is refused with the reason rather than rounded
into a different schedule.

**What it costs.** Frames are rendered during the training rollout, one per
environment step, so a clip shows the actual training environment --
exploration noise, curriculum stage and all -- and costs a render per step only
while it is collecting. On disk, doubling keeps a 50 000-iteration run to about
thirty clips; on top of that a **200 MB budget** per run stops the schedule and
says so (`--video-budget-mb`, `0` for no limit).

A scheduled clip never takes the last deferred-job slot from a command you are
waiting on; if it cannot start it says so in the events
(`progress_clip_skipped`) and waits for the next gap.

## `diagnose`

The most useful command in this list. It records per-step joint signals and
tells you *why* the motion is bad, not just that it is.

```bash
rlmcp diagnose --seconds 4
```

**MCP:** `diagnose_motion({"seconds": 4.0})`

Returns a report with five sections and a plain-English verdict:

```json
{"smoothness": {"joint_jerk_rms": 8596, "chatter_measured": true,
                "hf_share_median": 0.469, "whole_body_hf": true,
                "worst_hf_joints": [{"name": "right_shoulder_yaw_joint",
                                     "hf_power_share": 0.659, "peak_hf_hz": 9.75}]},
 "tracking": {"lin_vel_error_rms": 0.285, "commanded_speed_mean": 0.672},
 "effort":   {"torque_rms": 11.3, "hardest_working_joints": ["..."]},
 "posture":  {"tilt_deg_mean": 4.2, "base_height_std": 0.044},
 "gait":     {"contact_fraction": 0.51, "mean_air_time_s": 0.32},
 "verdict":  ["The whole body carries high-frequency motion (median 0.47 of
               joint-velocity power above 8.0 Hz across 29 joints) ... raise
               action_rate_l2 or lower action.joint_pos.scale_gain rather than
               chasing one joint",
              "Linear velocity tracking is reasonable (RMS error 0.28 m/s)"]}
```

Three things make the report trustworthy:

- **Chatter is the share of joint-velocity power above the gait band**, not the
  dominant frequency. Buzzing sits on top of a healthy 1–3 Hz gait instead of
  replacing it, so the dominant frequency would miss it.
- **A joint is named only if it stands out against the robot's own median.** On a
  50 Hz humanoid every joint carries some high-frequency content, and a list of
  27 "bad" joints tells you nothing. When the median itself is high, the report
  calls it a whole-body problem, which has a different fix.
- **When the measurement cannot be made** (trace too short, control rate too low)
  you get `chatter_measured: false` and the reason. A missing finding never
  reads as "clean". A diverged (NaN) policy leads with `DIVERGENCE:` instead of
  tidy-looking zeros.

## `trace`

Record the raw signals without analysing them. Lands as an `.npz` in
`artifacts/`.

```bash
rlmcp trace --seconds 6 --env-id 0
```

**MCP:** `record_trace({"seconds": 4.0})`

## `plot-trace`

Re-plot the last trace.

```bash
rlmcp plot-trace --channels joint_vel --components knee ankle
```

**MCP:** `plot_joint_trace({"channels": [...], "components": [...]})`

## `analyze`

Re-run the diagnosis on a saved trace, with no live trainer. Useful after the
run has exited.

```bash
rlmcp analyze artifacts/trace_env0_it001520.npz --plot
```

Traces are saved without pickle. A trace from an older rlmcp needs
`--allow-legacy`, which is an explicit opt-in because unpickling a file you did
not write runs code from it.

## Deferred job rules

- Several jobs at once is fine. Past about four in flight, new requests are
  refused; free a slot with `rlmcp run cancel_job req_id=<id>` (ids are in
  `status`'s `pending_jobs`).
- A deferred request **while training is paused** is refused with an
  explanation. No steps happen while paused, so the job could never finish.
- Jobs time out (~90 s) and fail honestly instead of hanging your call.
- Queued commands expire after 120 s. A retry after a timeout can never cause a
  double execution.

---

# Changing a run

## `params`

Discover what can be tuned. Nothing is hand-listed; rlmcp walks the environment
config. The G1 rough task exposes 97 knobs.

```bash
rlmcp params --contains foot
rlmcp params --category reward
rlmcp params --live                     # ask the trainer instead of params.json
```

**MCP:** `list_parameters({"category": "reward", "contains": "foot"})`

| category | examples |
| --- | --- |
| `reward` | `reward.foot_slip.weight`, `reward.foot_clearance.params.target_height` |
| `termination` | `termination.fell_over.params.limit_angle` |
| `domain_randomization` | `event.interval.push_robot.params.velocity_range.x` (a `[min, max]` pair) |
| `curriculum` | `command.twist.ranges.lin_vel_x` |
| `action` | `action.joint_pos.scale_gain` — a direct lever on jerkiness |
| `rl` | `rl.learning_rate`, `rl.entropy_coef`, `rl.clip_param`, `rl.desired_kl` |

Each parameter carries a **liveness**, and this is the field to read before you
write:

| liveness | what happens |
| --- | --- |
| `live` | takes effect next rollout batch |
| `at_reset` | applied now, takes effect at each environment's next reset. The response says so |
| `at_startup`, `inert` | the write is **refused**. These are only read when the environment is built, so a write would report success and change nothing |

## `get`

```bash
rlmcp get reward.action_rate_l2.weight
```

## `set`

```bash
rlmcp set reward.action_rate_l2.weight -0.25 --why "ankles chattering at 15 Hz"
rlmcp set event.interval.push_robot.params.velocity_range.x '[-1.5, 1.5]'
```

**MCP:** `set_parameter({"key": "...", "value": -0.25, "rationale": "..."})`

Always pass a reason. It costs one clause and turns the event log into something
you can read back later.

Values are checked before anything is touched. A scalar written to a `[low, high]`
range, an inverted range, or a float that would truncate on an int knob is
refused with the expected shape named, and the live config stays intact.

**Magnitudes are not second-guessed.** The only bounds are the ones true by
definition: `gamma` and `lam` in `[0, 1]`, no negative learning rate. Everything
else is unbounded, because only the task knows its own scale. A reward weight of
600, an action gain of 10 and a clip epsilon of 0.8 are all legitimate
experiments, and an invented limit would block them while catching nothing.

> Setting `rl.learning_rate` also switches PPO's schedule to `fixed`. The
> adaptive schedule would otherwise overwrite your value within one iteration.
> The response says so rather than silently doing nothing.

## `reset`

Put parameters back to their startup values.

```bash
rlmcp reset                              # all of them
rlmcp reset reward.foot_slip.weight      # just these
```

**MCP:** `reset_parameters({"keys": [...]})`

## `reset-envs`

The other reset, and it does an unrelated thing: it starts fresh **episodes** and
leaves every parameter where it is.

```bash
rlmcp reset-envs                                  # every environment
rlmcp reset-envs --where terrain=pyramid_stairs   # only those
rlmcp reset-envs --env-id 0 --env-id 7
```

**MCP:** `reset_environments({"where": {...}, "rationale": "..."})`

Use it to clear a stuck state, or to watch the start of a behaviour instead of
waiting for the robot to fail into one.

## `commands`

What verbs this run accepts, core plus whatever its extensions added.

```bash
rlmcp commands
```

**MCP:** `list_commands()`

## `run`

Call any of them. Values are parsed as JSON, so quote lists.

```bash
rlmcp run set_terrain terrains='["flat","random_rough"]' max_level=4
rlmcp run terrain_status
rlmcp run cancel_job req_id=abc123
```

**MCP:** `run_command({"cmd": "set_terrain", "args": {"max_level": 4}})`

This is how a task's own vocabulary reaches you without the MCP tool list ever
changing. `rlmcp raw` is an alias.

## `curriculum`

```bash
rlmcp curriculum                                    # stage, and what it waits for
rlmcp curriculum advance --why "flat is solved"
rlmcp curriculum goto 3_from_table --why "warm start already places reliably"
rlmcp curriculum auto-off                           # stop it promoting itself
rlmcp curriculum auto-on
```

**MCP:** `curriculum_status()`, `curriculum_advance({"reason": "..."})`,
`curriculum_goto({"stage": "...", "reason": "..."})`, `curriculum_auto({"enabled": false})`

`curriculum_status` shows each promotion condition with its current value and the
streak. A manual command beats the automatic schedule within the same iteration.
Writing a ladder is [curriculum.md](curriculum.md).

---

# Safety net and lifecycle

## `checkpoint`

```bash
rlmcp checkpoint before-experiment --note "trying 2x push events"
rlmcp checkpoints
```

**MCP:** `save_checkpoint({"tag": "...", "note": "..."})`, `list_checkpoints()`

Saves weights **and** parameters, curriculum stage, and extension state (terrain
unlocks and the like).

## `load`

The undo button.

```bash
rlmcp load before-experiment
```

**MCP:** `rollback_to_checkpoint({"path": "before-experiment"})`

Restores weights, parameters (checkpointed values beat stage defaults),
curriculum stage and extension state, and reports what it actually found. A
checkpoint with no extension state says so instead of claiming success.

## `pause`, `resume`, `step-once`

```bash
rlmcp pause
rlmcp step-once        # advance exactly one iteration while paused
rlmcp resume
```

**MCP:** `pause_training()`, `resume_training()`

Other tools keep working while paused, except deferred jobs, which are refused
because nothing steps.

## `stop`

```bash
rlmcp stop --why "the falsifier fired"
```

**MCP:** `stop_training({"reason": "..."})`

In the training process this arrives as a `TrainingStopped` exception inside
`runner.learn()`. Catch it and save a checkpoint — see [your-task.md](your-task.md).
`stop` also closes a `play` session.

---

# Writing things down

## `note`

```bash
rlmcp note "reviewer says the gripper should approach from above"
```

**MCP:** `add_note({"text": "..."})`

Goes into the event log next to every parameter change. A remark about the run
belongs with the run, not in a side file.

## `feedback`

What a human said, kept against the run and stamped with the iteration.

```bash
rlmcp feedback "it looks jittery near the end" --kind observe
rlmcp feedback "stop tuning the entropy coefficient" --kind correct \
    --read-as "leave rl.entropy_coef alone for the rest of this run"
```

**MCP:** `record_feedback({"text": "...", "kind": "steer", "interpretation": "..."})`

Six kinds: `steer`, `correct`, `reject`, `approve`, `observe`, `constrain`. The
first four ask for something, so an entry of that kind with no recorded response
counts as **unanswered** and says so everywhere it appears. Answering is
[`rlmcp record answer`](records.md#feedback-the-steering-kept).

---

# Playing a finished run

## `play`

Everything above talks to a live trainer. Once the run has exited, the only
evidence left is a checkpoint, and `play` is how you look at it.

```bash
rlmcp play                                      # newest run, last checkpoint, 8s clip
rlmcp play logs/rsl_rl/my_run/model_final_4375.pt --device cpu
rlmcp play --stage 2_hardest --seconds 12
rlmcp play --mode native                        # MuJoCo's own viewer
rlmcp play --mode viser                         # a viewer in the browser
```

| flag | meaning |
| --- | --- |
| `--mode video\|native\|viser` | render an mp4 (works over ssh), or open a viewer |
| `--device cpu` | play without touching a GPU that is training |
| `--stage NAME` | restore conditions as of the end of that stage |
| `--set KEY=VALUE` | override a parameter after the replay, so it wins. Repeatable |
| `--no-replay` | do not restore conditions at all |
| `--policy zero\|random` | no checkpoint at all: hold zero actions, or sample random ones |
| `--allow-partial` | render even if some conditions could not be restored |
| `--task-package MODULE` | import this first so your tasks register. Repeatable |
| `--num-envs`, `--extra-envs` | how many robots, and how many composited into the frame |
| `--seconds`, `--fps`, `--out`, `--render-width/-height` | clip options |

### Why `play` replays the run first

A checkpoint remembers weights and an iteration number. It does not remember the
curriculum rung it was climbing, and a task's `play=True` config is rung zero by
construction. So replaying a late checkpoint against a fresh config shows a
policy failing at a task it was never asked to do. That looks exactly like a bad
policy, which is the worst kind of wrong evidence: clear, confident and blaming
the wrong thing.

So `play` folds the session's event log first. Every curriculum entry recorded
the parameters it set and the commands it ran; every live `rlmcp set` recorded
its key and new value. Replaying those in order puts the environment back where
the policy left it, and the result says exactly what was restored.

If something cannot be restored — a command this build of the task no longer
has, a parameter it no longer accepts — `play` refuses to render and says which
fix applies to which problem. "Unknown command" and "the command changed its
arguments" send you to different places to look.

The reconstruction lives in `rlmcp/core/replay.py` and imports no simulator. It
is a plain parse-and-fold over the event log.

`--policy zero|random` is the exception, and it is not really one: with no
checkpoint there is no run whose conditions went missing, so nothing is
restored and nothing is said about it. The environment runs at the task's own
play configuration, which is what you want when the task, not a policy, is the
thing being looked at. `--set` still applies on top.

### Steering a play session

A play session is itself a session. It publishes into `<run>/rlmcp/play/<stamp>/`
and answers the same commands a training run does. It is marked as a play
session, so it never becomes the answer to "the latest session here".

```bash
rlmcp --session <play dir> reset-envs                  # see the opening again
rlmcp --session <play dir> run list_policies           # what else is next to it
rlmcp --session <play dir> run load_policy checkpoint=model_2000.pt
rlmcp --session <play dir> stop
```

`load_policy` is the interesting one: the conditions you restored and the camera
you set up survive, and only the weights change. That is what makes a comparison
fair. It refuses without touching the running policy if the file is missing or
if the weights load but cannot act on this environment — a swap that half-applies
is worse than one that does not happen.

Because a checkpoint from another run may have trained at another rung, the
response always names the stage it trained at next to the stage the environment
is in. When those disagree the conditions are deliberately **not** changed, and
the payload says so loudly. Pass `replay=true` to restore that checkpoint's own
conditions instead.

---

# Records

`rlmcp record …` is the record *across* runs: hypotheses, verdicts, ancestry,
feedback. It has its own page: [records.md](records.md).
