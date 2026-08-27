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
  state/                  # what is mjlab's own: rendering, terrain
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
