# rl-mcp

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

Values are bounds-checked, applied between rollout batches (never mid-step), and
every change is written to the session event log with the rationale the caller
gave. `reset_parameters` restores the startup values.

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
│ RlMcpEnvWrapper             │          │ status.json    │          │ MCP server (29 tools)    │
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
