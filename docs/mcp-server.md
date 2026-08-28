# The MCP server (for AI agents)

The MCP server is the agent side of rlmcp. It runs in its own process with no
simulator and no torch, talks to the training process through the session
directory, and exposes the whole control surface as 35 MCP tools. Killing the
server never disturbs training.

Per-tool details are in [tools.md](tools.md). This page covers setup, the rules
an agent needs to know, and a worked session.

## Setup

```bash
pip install -e '.[server]'      # adds the MCP SDK; mcp>=2 and 1.x both work
```

Register it with Claude Code, or any MCP client:

```bash
claude mcp add rlmcp -- rlmcp-server --root /path/to/logs
```

- `--root` is where the server looks for sessions. At startup it pins the
  **newest** session under the root and stays on it. It never retargets on its
  own, even if a newer run appears or the pinned one exits.
- `--session /path/to/run/rlmcp` pins an explicit session instead.
- Every response carries a `"session"` key naming the pinned run, so an agent
  always knows which run it is talking about.

To move to a different run mid-conversation, call the tool for it:

```json
switch_session {"target": "newest"}
→ {"ok": true, "session": "2026-08-23_10-11-12_smp/rlmcp",
   "task": "Smp-Steering-Rough-G1", "started_at": "...", "state": "running"}
```

The CLI works differently on purpose: with no `--session` it attaches to the
newest session on every invocation.

## Running, stalled, dead

The server derives a liveness state from the trainer's pid and heartbeat.

| state | meaning |
| --- | --- |
| `running` | pid alive, heartbeat fresh |
| `stalled` | pid alive, no heartbeat for >600 s. Advisory only. One long iteration can legitimately do this, so live commands are still attempted |
| `dead` | pid gone, or heartbeat >24 h (presumed pid reuse; the note says so) |

When the trainer is **dead**, the data tools keep working from disk:
`get_training_status`, `get_metrics`, `list_metrics`, `plot_metrics`,
`list_parameters`, `get_events`, `list_artifacts`. They are marked
`"live": false` with the source file named.

Tools that need a live trainer refuse with an error that names the death, lists
what still works, and points at `switch_session`. Nothing is ever queued at a
dead run.

## The tools

| group | tools |
| --- | --- |
| sessions | `list_sessions`, `switch_session` |
| observe | `get_training_status`, `list_metrics`, `get_metrics`, `plot_metrics`, `get_events`, `list_artifacts` |
| see the robot | `take_screenshot`, `record_video`, `live_view`, `diagnose_motion`, `record_trace`, `plot_joint_trace` |
| tune | `list_parameters`, `set_parameter`, `reset_parameters` |
| task verbs | `list_commands`, `run_command` |
| curriculum | `curriculum_status`, `curriculum_advance`, `curriculum_goto`, `curriculum_auto` |
| checkpoints | `save_checkpoint`, `list_checkpoints`, `rollback_to_checkpoint` |
| lifecycle | `pause_training`, `resume_training`, `reset_environments`, `stop_training`, `add_note` |
| records | `record_feedback`, `attach_feedback`, `answer_feedback`, `get_feedback_timeline`, `set_record_headline` |

`cancel_job` and every extension verb reach the run through `run_command`. The
per-run command list comes from `list_commands`, so the tool list never has to
change when an environment gains a capability.

Three MCP **resources** mirror the hot files for clients that prefer resources
over tools: the current status payload, the full parameter schema with liveness,
and the recent event log.

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

The verdict strings are written to be trusted. See
[tools.md](tools.md#diagnose) for how chatter is measured and when the report
refuses to claim anything.

Pick a specific robot with `where`. The vocabulary comes from the run's
extensions, e.g. `{"where": {"terrain": "pyramid_stairs"}}` on a locomotion task.
`take_screenshot` and `record_video` take the same selector and return the image
inline **plus** the data payload. Images are downscaled to fit MCP message
limits; if one still does not fit, you get the path and a note.

`live_view` takes it too, and answers with a URL rather than an image: it
attaches a browser view to the run that keeps updating on its own. Hand the URL
to whoever asked to see the robot -- that is the deliverable, and you cannot
look at it yourself. Pass `realtime: true` when the question is about the gait
rather than about what the robot is doing this second: training steps far
faster than life, so the default view is a fast-forward.

### 3. Change something, with the reason on the record

```json
set_parameter {"key": "reward.action_rate_l2.weight", "value": -0.25,
               "rationale": "ankles chattering at 15 Hz per diagnose"}
→ {"ok": true, "key": "reward.action_rate_l2.weight",
   "old_value": -0.1, "new_value": -0.25, "applied": true, "liveness": "live"}
```

A write to a knob the environment will never re-read is refused, not faked:

```json
set_parameter {"key": "event.startup.foot_friction.params.ranges", "value": [0.4, 1.0]}
→ {"ok": false, "error": "Cannot set '…': liveness is 'at_startup'. mjlab applies
   startup events exactly once when the environment is constructed … Change the
   task config and restart training instead."}
```

### 4. Experiment safely

```json
save_checkpoint {"tag": "before-experiment", "note": "trying 2x push events"}
run_command {"cmd": "set_terrain", "args": {"terrains": ["flat","random_rough"], "max_level": 6}}
… watch a few hundred iterations …
rollback_to_checkpoint {"path": "before-experiment"}
→ {"ok": true, "weights": true, "parameters_restored": 14,
   "extensions_restored": 1, "curriculum_stage": "1_rough"}
```

### 5. Deferred jobs

Video, trace and diagnose collect data over the coming rollout steps, so they
answer when the job completes rather than immediately.

- Concurrent jobs are fine, each records privately. Past about four in flight the
  request is refused; `run_command {"cmd": "cancel_job", "args": {"req_id": "…"}}`
  frees a slot. Req ids are in the status payload's `pending_jobs`.
- A deferred request **while training is paused** is refused with an explanation.
  No steps happen while paused, so the job could never finish.
- Jobs carry a wall-clock timeout (~90 s) and fail truthfully instead of hanging
  your call.
- Queued commands expire. If the trainer does not service a request within 120 s
  it is refused as expired rather than executed late, so a retry after a timeout
  can never cause a double execution.

## Post-mortem: when the run has exited

Stay attached. The data tools keep answering.

```json
get_metrics {"names": ["Train/mean_reward"], "last_n": 100}
→ {"ok": true, "live": false, "source": "metrics.jsonl", "metrics": {"…": "…"}}

set_parameter {"key": "…", "value": 0}
→ {"ok": false, "error": "training process 12345 has exited …
   status, metrics, plots, parameters, events and artifacts still answer from
   disk; use switch_session to attach to a newer run.", "last_status": {"…": "…"}}
```

To look at the policy itself after the run has exited, use `rlmcp play` from a
shell. It restores the conditions the checkpoint trained under first. See
[tools.md](tools.md#play).

`rlmcp analyze <trace.npz>` re-runs the diagnosis offline.

## Prompting tips for the agent driving this

The long version of these rules — what to verify before launch, which numbers
to trust, and how to read a symptom into a lever — is [tuning.md](tuning.md).

- **Diagnose before tuning.** The verdict names the lever to pull. A reward tweak
  justified by a measurement beats one justified by a hunch.
- **Always pass `rationale`.** It costs nothing and turns the event log into
  something you can re-read: what changed, when, and why.
- **Trust refusals.** A refused write is the harness telling the truth about a
  knob that cannot work. The error says what to do instead.
- **Checkpoint before experiments**, and treat `rollback_to_checkpoint` as the
  undo button it is.
- **Show, do not assert.** "It improved" is worth less than a plot or a clip.
- **After a crash, stay attached.** The post-mortem tools plus the event log are
  usually enough to explain what happened without relaunching anything.
