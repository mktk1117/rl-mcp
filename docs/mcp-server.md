# Using the rlmcp MCP server

The MCP server is the agent side of rlmcp. It runs in its own process with no
simulator and no torch, talks to the training process through the session
directory, and exposes the whole control surface — watching, diagnosing,
steering, checkpointing — as MCP tools an LLM agent can call. Killing the
server never disturbs training.

This document shows how to set it up and what a real steering session looks
like. All JSON responses below are illustrative — shapes are exact, numbers are
examples.

## Setup

```bash
pip install -e '.[server]'      # adds the MCP SDK; mcp>=2 and 1.x both work
```

Register it with Claude Code (or any MCP client):

```bash
claude mcp add rlmcp -- rlmcp-server --root /path/to/logs
```

- `--root` is where the server looks for sessions. At startup it pins the
  **newest** session under the root and stays on it — it never silently
  retargets, even if a newer run appears or the pinned one exits.
- `--session /path/to/run/rlmcp` pins an explicit session instead.
- Every tool response carries a `"session"` key naming the pinned run, so an
  agent always knows which run it is talking about.

To move to a different run mid-conversation, call the tool for it:

```json
switch_session {"target": "newest"}
→ {"ok": true, "session": "2026-08-23_10-11-12_smp/rlmcp",
   "task": "Smp-Steering-Rough-G1", "started_at": "...", "state": "running"}
```

## The session model: running, stalled, dead

The server derives a liveness state from the trainer's pid and heartbeat:

| state | meaning |
| --- | --- |
| `running` | pid alive, heartbeat fresh |
| `stalled` | pid alive, no heartbeat for >600 s — advisory only; one long iteration can legitimately do this. Live commands are still attempted. |
| `dead` | pid gone (or heartbeat >24 h — presumed pid reuse, the note says so) |

When the trainer is **dead**, the data tools keep working from disk —
`get_training_status`, `get_metrics`, `list_metrics`, `plot_metrics`,
`list_parameters`, `get_events`, `list_artifacts` — marked `"live": false` with
the source file named. Tools that need a live trainer refuse with an error that
names the death, lists what still works, and points at `switch_session`.
Nothing is ever queued at a dead run.

## A steering session, end to end

### 1. Look at the run

```json
get_training_status {}
→ {"ok": true, "session": "…", "iteration": 1520, "num_envs": 4096,
   "paused": false, "headline_metrics": {"Train/mean_reward": 21.4,
   "rlmcp/episode_length_frac": 0.62, "rlmcp/terrain_level_frac": 0.41},
   "curriculum": {"stage": "1_rough", "streak": 7, "…": "…"},
   "pending_jobs": []}
```

### 2. Diagnose before touching anything

`diagnose_motion` records a few seconds of per-step joint signals from a live
environment and analyzes smoothness, tracking, effort, posture and gait:

```json
diagnose_motion {"seconds": 4}
→ {"ok": true, "report": {
     "smoothness": {"chatter_measured": true, "hf_share_median": 0.47,
                    "whole_body_hf": true, "worst_hf_joints": ["…"]},
     "tracking": {"lin_vel_error_rms": 0.28},
     "verdict": ["The whole body carries high-frequency motion …
                  raise action_rate_l2 or lower action.joint_pos.scale_gain
                  rather than chasing one joint", "…"]},
   "trace_path": "…/artifacts/trace_env0_it001520.npz"}
```

The verdict strings are written to be trusted: a diverged (NaN) policy leads
with `DIVERGENCE:` instead of clean-looking zeros, and when chatter cannot be
measured (trace too short, control rate too low) the report says
`chatter_measured: false` with the reason — absence of a finding is never
presented as evidence of smoothness.

Pick a specific robot with `where` — the vocabulary comes from the run's
extensions, e.g. `{"where": {"terrain": "pyramid_stairs"}}` on a locomotion
task. `take_screenshot` and `record_video` accept the same selector and return
the image inline **plus** the full data payload (images are downscaled to fit
MCP message limits; if one still doesn't fit you get the path and a note).

### 3. Change something, with the reason on the record

```json
set_parameter {"key": "reward.action_rate_l2.weight", "value": -0.25,
               "rationale": "ankles chattering at 15 Hz per diagnose"}
→ {"ok": true, "key": "reward.action_rate_l2.weight",
   "old_value": -0.1, "new_value": -0.25, "applied": true,
   "liveness": "live"}
```

Every parameter carries a **liveness** you can see in `list_parameters`:

- `live` — takes effect next rollout batch.
- `at_reset` — applied now, takes effect from each environment's next reset;
  the response says so in a `note`.
- `at_startup` / `inert` — the write is **refused** with an explanation: these
  knobs are only read at construction (or cached by the term), so a write would
  report success and change nothing. The harness does not pretend.

```json
set_parameter {"key": "event.startup.foot_friction.params.ranges", "value": [0.4, 1.0]}
→ {"ok": false, "error": "Cannot set '…': liveness is 'at_startup'. mjlab applies
   startup events exactly once when the environment is constructed … Change the
   task config and restart training instead."}
```

Values are validated before anything is touched — a scalar written to a
`[low, high]` range, an inverted range, or a truncating float on an int knob is
refused with the expected shape named, and the live config stays intact.
`reset_parameters` restores startup values; `reset_environments` is the
unrelated one that starts fresh episodes and leaves every parameter alone.

### 4. Experiment safely with checkpoints

```json
save_checkpoint {"tag": "before-experiment", "note": "trying 2x push events"}
run_command {"cmd": "set_terrain", "args": {"terrains": ["flat","random_rough"], "max_level": 6}}
… watch a few hundred iterations …
rollback_to_checkpoint {"path": "before-experiment"}
→ {"ok": true, "weights": true, "parameters_restored": 14,
   "extensions_restored": 1, "curriculum_stage": "1_rough"}
```

Rollback restores weights, parameters (checkpointed values win over stage
defaults), curriculum stage, and extension state (e.g. terrain unlocks), and
reports exactly what it found — a checkpoint missing extension state says so
instead of claiming success.

### 5. Curriculum: watch it, override it

`curriculum_status` shows the current stage, each promotion condition with its
live value, and the streak. `curriculum_advance` / `curriculum_goto` /
`curriculum_auto` override it; a manual command wins over the automatic
schedule within the same iteration. Extension verbs (like `set_terrain`) are
discovered with `list_commands` and invoked with `run_command`.

### 6. Deferred jobs: the rules

Video, trace and diagnose collect data over the coming rollout steps, so they
answer when the job completes rather than immediately:

- concurrent jobs are fine (each records privately); past ~4 in flight the
  request is refused and `run_command {"cmd": "cancel_job", "args":
  {"req_id": "…"}}` frees a slot (req ids are listed in the status payload's
  `pending_jobs`);
- a deferred request made **while training is paused** is refused with an
  explanation — no steps happen while paused, so the job could never finish;
- jobs carry a wall-clock timeout (~90 s default) and fail truthfully instead
  of hanging your call;
- queued commands expire: if the trainer doesn't service a request within
  120 s of submission it is refused as expired rather than executed late — a
  retry after a timeout can never cause a double execution.

### 7. Notes and the event log

`add_note {"text": "..."}` writes into the session's event log next to every
parameter change (with its rationale), curriculum transition, checkpoint and
falsifier event. `get_events` reads it back — this is the run's audit trail,
and it survives the trainer.

## Post-mortem: when the run has exited

Attach the server (or keep it attached) after training ends:

```json
get_metrics {"names": ["Train/mean_reward"], "last_n": 100}
→ {"ok": true, "live": false, "source": "metrics.jsonl", "metrics": {"…": "…"}}

set_parameter {"key": "…", "value": 0}
→ {"ok": false, "error": "training process 12345 has exited …
   status, metrics, plots, parameters, events and artifacts still answer from
   disk; use switch_session to attach to a newer run.", "last_status": {"…": "…"}}
```

`rlmcp analyze <trace.npz>` re-runs diagnosis offline; traces from current
rlmcp load pickle-free, and `--allow-legacy` is the explicit opt-in for old
pickled traces.

## MCP resources

Three resources mirror the hot files for clients that prefer resources over
tools: the current status payload, the full parameter schema (with liveness),
and the recent event log.

## Tool reference

| group | tools |
| --- | --- |
| sessions | `list_sessions`, `switch_session` |
| observe | `get_training_status`, `list_metrics`, `get_metrics`, `plot_metrics`, `get_events`, `list_artifacts` |
| see the robot | `take_screenshot`, `record_video`, `diagnose_motion`, `record_trace`, `plot_joint_trace` |
| tune | `list_parameters`, `set_parameter`, `reset_parameters` |
| task verbs | `list_commands`, `run_command` |
| curriculum | `curriculum_status`, `curriculum_advance`, `curriculum_goto`, `curriculum_auto` |
| checkpoints | `save_checkpoint`, `list_checkpoints`, `rollback_to_checkpoint` |
| lifecycle | `pause_training`, `resume_training`, `reset_environments`, `stop_training`, `add_note` |

`cancel_job` and any extension verb reach the run through `run_command`; the
full per-run command list comes from `list_commands`.

## Prompting tips for the agent driving this

- **Diagnose before tuning.** The verdict strings name the lever to pull; a
  reward tweak justified by a measurement beats one justified by a hunch.
- **Always pass `rationale`.** It costs nothing and turns the event log into a
  paper you can re-read: what was changed, when, and why.
- **Trust refusals.** A refused write is the harness telling the truth about a
  knob that cannot work; the error says what to do instead.
- **Checkpoint before experiments**, and treat `rollback_to_checkpoint` as the
  undo button it is.
- **After a crash, stay attached.** The post-mortem tools plus the event log
  are usually enough to explain what happened without relaunching anything.
