# How rlmcp works

## Two processes, one directory

```
      training process                   session directory                agent process
       torch + mujoco                     plain JSON files                 no simulator
┌──────────────────────────┐             ┌───────────────┐             ┌──────────────────┐
│ RlMcpEnvWrapper          │             │ status.json   │             │ MCP server       │
│ ├ on_step()  frames      │ ──writes──► │ metrics.jsonl │ ───reads──► │ (35 tools)       │
│ └ service()  commands    │ ◄──reads─── │ events.jsonl  │ ◄──writes── │ rlmcp CLI        │
│    ├ SimAdapter (env)    │             │ artifacts/    │             │ your own scripts │
│    ├ RunnerAdapter (PPO) │             │ inbox/        │             │                  │
│    └ Extensions          │             │ outbox/       │             │                  │
└──────────────────────────┘             └───────────────┘             └──────────────────┘
```

Telemetry flows left to right: the wrapper publishes status, metrics and
artifacts. Commands flow right to left: the CLI and the MCP server drop requests
into `inbox/`, and the wrapper answers into `outbox/` the next time it services
them. Neither side imports the other, and neither blocks the other.

That buys four things:

- The MCP server never imports torch or mujoco. Starting or killing it cannot
  disturb training.
- Commands execute **between rollout batches**, so a parameter edit can never
  race the simulator.
- Everything is inspectable with `cat` when something goes wrong.
- A run leaves a complete, replayable record behind after it exits.

Commands that need the robot to move (video, traces, diagnosis) are deferred:
the request stays open, `on_step` feeds it frames, and the response is written
when the job completes. Everything else answers within one iteration.

## The client surface is the protocol

The directory is how the two sides talk *today*. What they are allowed to say
is a smaller thing, and it has a name: `SessionClient` in
[`rlmcp/session.py`](../rlmcp/session.py). Seventeen names, listed in
`WIRE_SURFACE`:

```
address  key  name                       what to call this run
info  status  params                     static, live, tunable
metrics  metrics_count  events           history
list_artifacts  read_artifact            what it produced
submit  poll  wait  call                 commands
liveness  liveness_info                  is anyone home
```

`Session` is one implementation of it — the local one, where reaching a run
means reading its directory. It is the only one today, and on a single machine
it is the right default: no socket, nothing to crash, and `cat` still works
when something is wrong.

It exists because it will not be the only one. A run on a GPU box that the
reader cannot see needs a second implementation over a connection, and the cost
of writing that is decided here rather than then: **the CLI, the MCP server and
rl-mcp-studio are written against these names and no others**, so a second
transport changes one layer instead of every caller.

Which is why two things a filesystem gives away for free are methods:
`list_artifacts` and `read_artifact`. A caller that reaches a plot by joining
`session.dir / "artifacts"` compiles, works locally, and cannot be made to work
at all once the file is on another machine. The same reasoning makes `address`
opaque: it is a path today and a URL later, and nothing may parse it.

The second implementation exists now: [`rlmcp/wire.py`](../rlmcp/wire.py)'s
`WireSession` reaches a run through `rlmcp hostd`
([`rlmcp/hostd.py`](../rlmcp/hostd.py)) on the machine the run is on, over
plain HTTP from the standard library, and answers the same seventeen names.
`wire.connect(address)` is where the two transports meet: a directory gives a
`Session`, a URL (`http://host:8740/v1/sessions/<run>/<session>`) a
`WireSession`. The daemon adds the two faces a filesystem never needed --
**jobs** (start a process on the host, poll it, stop it, read its log) and
**host** (who this machine is, its GPUs, its disk) -- and nothing else; it
never imports torch, and if it dies, telemetry delivery stops and training
does not. One bearer token per host, checked on every request, required to
bind beyond localhost. A session key is a name, never a path: the daemon
resolves it by walking what exists under its root.

Three streams carry a `seq`: `status.json`, and every row of `metrics.jsonl`
and `events.jsonl`. It is what lets a reader ask for *what is new* --
`metrics(since_seq=n)`, `events(since_seq=n)` -- instead of refetching a log
that only grows, which is the difference between polling a directory and
polling a network. Numbers are contiguous and one-based, so the last row's
`seq` is the row count and a cursor read costs two backward block reads rather
than a scan. A trainer restarted onto the same directory continues the count
instead of replaying it, because a reader holding `seq=7` must never be handed
a *different* row 8.

`seq` in a metrics row is bookkeeping sitting in a row of measurements, which
is a trap: every reader that asks what a run logged walks the row's keys. So
the non-measurements are named once, in `RESERVED_METRIC_KEYS`, and the four
places that walked those keys use it. Adding a field to a metrics row without
adding it there puts that field on the CLI's metric list, on a plot axis, and
in the studio's headline.

What is still path-shaped, honestly: the trainer side (which owns its
directory, and should), the registry (machine-local by definition),
`read_ladder` in `core/replay.py` (a run's `curriculum.json`, which is config
rather than telemetry and has no home on the protocol yet), `task_from_session`
in `records/link.py` (deliberate -- it runs on the training process's critical
path, where a half-written session file must cost the field and not the run),
and `play`, which needs a real checkpoint file. Those are the next things to
move, not things that are fine.

## Where the code is

```
rlmcp/core/          parameters, telemetry, traces, curriculum, controller — no backend
rlmcp/adapters/      SimAdapter / RunnerAdapter; mjlab/ is the reference implementation
rlmcp/extensions/    capabilities the core does not know about; terrain.py is the model
rlmcp/records/       plans, outcomes, ancestry, the rendered tree
rlmcp/server/        the MCP server (imports no simulator, by design)
```

The layering rule: **the core never learns a task's vocabulary, and the adapter
never learns a capability.**

## How parameters are found

Nothing hand-lists them. mjlab's manager term configs are ordinary dataclasses
and dicts, so rlmcp walks them and emits a key for every leaf that is a number, a
bool or a `[min, max]` pair.

```
rlmcp/adapters/manager_based/
  access/                 # parameters: discovery, reads, writes
    paths.py              #   the reflective core: walk, resolve, coerce
    base.py               #   what a provider supplies
    rewards.py  terminations.py  events.py  commands.py  actions.py
  sampling.py  metrics.py # live state, read the same way on either backend
  env_wrapper.py          # servicing, telemetry, curricula, records, clips
rlmcp/adapters/mjlab/
  sim_adapter.py          # thin: implements SimAdapter by delegating
  state/                  # what is mjlab's own: rendering, terrain, the live view
```

None of that is mjlab's, which is why it does not live there: term configs are
dataclasses in IsaacLab too, and the same walk finds the same kinds of leaf.

A provider answers one question — *which objects hold tunable values, and what
are they called* — in 20 to 90 lines. Adding a family means adding a file and one
entry in `PROVIDER_TYPES`. Discovery, get/set, bounds, the CLI and the MCP tools
all pick it up unchanged.

Two consequences:

**New config fields appear on their own.** A task that adds a reward term, or an
mjlab release that adds a field, shows up without touching rlmcp. On the G1 rough
task this surfaced 97 parameters where the previous hand-written version exposed
60.

**Keys escape dots.** mjlab keys the G1 pose-std dict by joint-name regex, so
that parameter is `reward.pose.params.std_walking.\.*knee\.*`. Segments are
escaped on the way out and split on unescaped dots on the way back.

Providers can hide fields that change *semantics* rather than magnitude, via
`skip_keys`. A termination's `time_out` flag tells the algorithm whether to
bootstrap the value. That is a trap, not a knob.

## Using it with a different simulator

rlmcp targets **manager-based environments whose config is live**: mjlab,
IsaacLab, and anything built the same way. Term configs are ordinary dataclasses
and dicts that the running environment re-reads, which is what makes "change a
weight and it takes effect next batch" true.

Within that family, implement `SimAdapter` (and optionally `RunnerAdapter`) from
`rlmcp/adapters/base.py`. Only `discover_parameters`, `get_parameter` and
`set_parameter` are required. Everything else — rendering, traces, terrain
control — has a default that reports "not supported by this backend", so tools
degrade with an explanation instead of crashing.

Two backends ship, and what they share is where most of the code lives:

```
rlmcp/adapters/manager_based/   parameter access, trace sampling, summary
                                metrics, the env wrapper — written against the
                                shape both backends have, not against either
rlmcp/adapters/rsl_rl_runner.py the RunnerAdapter; both drive the same library
rlmcp/adapters/mjlab/           MjlabSimAdapter + its offscreen renderer
rlmcp/adapters/isaaclab/        IsaacLabSimAdapter + its Kit-app renderer
```

A backend adapter is only what is genuinely its own: how a robot is found in
the scene, how a frame is rendered, and anything its simulator has that the
others do not. `rlmcp/adapters/isaaclab/` is about 230 lines including both,
which is the number to expect for a third.

Anything your simulator has and others do not belongs in an
[Extension](extensions.md), not in the adapter contract.

## Tests

```bash
pytest tests -q     # 766 tests, ~6s, no GPU and no simulator required
```

The suite runs the real controller against a fake simulator, covering the command
protocol, promotion rules, chatter detection and checkpoint round-trips. It needs
`torch` (for tensor-shaped fakes) but no mujoco and no GPU. The 14 MCP-server
tests skip unless the optional `mcp` package is installed.
