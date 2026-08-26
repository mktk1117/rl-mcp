# Working in this repo (agent guide)

rlmcp exists so an agent can *watch, diagnose and steer a live training run*.
The failure mode this file is written against is real and was observed: an agent
re-implemented half of rlmcp by hand — its own metrics parser, its own frame
tiling, its own stage ladder in a scratch file — because the entry point it
happened to read first did not mention the feature. Read this before deciding
how to drive a run.

This file is the canonical guide. `CLAUDE.md` is a symlink to it, so an
assistant that looks for that name finds the same text; there is one copy to
maintain and nothing here assumes a particular agent product.

## This repo is the tool. The records are somewhere else.

Two repositories, on purpose:

* **this one** — the harness, adapters, CLI and MCP server. Its history should
  be about the tool.
* **your records** — your mjlab task packages and `records/`, the run records.
  That repository holds a package per task (env config, `mdp/` terms, its
  `rlmcp` extension, its curriculum, its launcher), the records those runs
  filled, and its own guidance for *running* experiments: simulator traps,
  verification checks, the run-record lifecycle.

rlmcp is a dependency of that repo, never the other way round. Nothing about a
task — no env config, no reward term, no run record — belongs in this tree. If
you are about to add one, you are in the wrong repository.

`$RLMCP_RECORDS` is what separates them at runtime. The records root resolves to
the explicit argument (`rlmcp record --records-root`, `rlmcp.wrap(records_root=...)`),
then `$RLMCP_RECORDS`, then `./records` — see `open_store` in `rlmcp/records/filestore.py`.
A stray `records/` appearing in this checkout means somebody ran a record command
without the variable set; it is a scratch directory, not the records.

## If you are about to hand-roll something, check here first

| you are about to… | use instead |
| --- | --- |
| parse `metrics.jsonl` / `status.json` yourself | `rlmcp status`, `rlmcp metrics`, `rlmcp plot` |
| write a loop that pokes weights at milestones | a **curriculum** (`StageSchedule`) — see below |
| add a task-specific verb by editing core code | an **extension** (`rlmcp/extensions/`, or your own package) |
| screenshot by rendering + slicing video frames | `rlmcp shot` (add `--where key=value` to pick envs) |
| judge smoothness from reward curves | `rlmcp diagnose` (measures HF power share, jerk, effort) |
| tell the user "it improved" | `rlmcp video` / `rlmcp plot`, then show them |
| look at a policy whose run has already exited | `rlmcp play` — it restores the conditions the checkpoint trained under first |
| compare a metric across several runs | `rlmcp record compare`, or `rlmcp record graph` for the lineage |
| keep the record of an experiment in a scratch file | `rlmcp record new` / `rlmcp record close` — it belongs **in** the records |

## Where things are

```
rlmcp/core/          parameters, telemetry, traces, curriculum, controller — no backend
rlmcp/adapters/      SimAdapter / RunnerAdapter; mjlab/ is the reference implementation
rlmcp/extensions/    capabilities the core does not know about; terrain.py is the model
rlmcp/records/           the records: records, lineage, verdicts, the rendered tree
rlmcp/server/        the MCP server (imports no simulator, by design)
```

The layering rule is one sentence: **the core never learns a task's vocabulary,
and the adapter never learns a capability.** Anything a particular simulator has
and others do not belongs in an `Extension`; anything one task has belongs in an
extension in *that task's* repo.

## Curricula are NOT terrain-only

`rlmcp-train --curriculum {terrain,none}` is a convenience entry point, and its
two choices are the whole of what *that CLI* offers. The library is not so
limited: a curriculum is an ordered list of stages, each of which applies
parameter edits and/or commands on entry and promotes when metric conditions
hold for N consecutive iterations past a floor. Nothing about it is terrain.

```python
from rlmcp.core.curriculum import Action, Condition, CurriculumStage, StageSchedule

schedule = StageSchedule([
    CurriculumStage(
        name="0_learn_to_turn",
        parameters={"reward.fall_penalty.weight": -40.0},
        apply=[Action("set_goal_difficulty", {"threshold": 0.6})],
        promote_when=[Condition("rlmcp/goal_rate_per_min", ">=", 8.0)],
        min_iterations=400, hold_iterations=20,
    ),
    ...
])
env = rlmcp.wrap(env, session_dir=..., curriculum=schedule)
```

A manipulation ladder is the clearest non-terrain example: rungs that start
with the object already held and work up to picking it off a table, each rung
setting reward weights and calling the task extension's own verbs on entry.

**Rule of thumb:** if you are writing down "when metric X passes Y, change
weight Z", that is a `CurriculumStage`, not a note to yourself. Encoding it
means the run drives itself, the ladder is saved with the run
(`params/curriculum.json`), and a human can still override it live with
`rlmcp curriculum advance|goto|auto-off`.

## Task vocabulary belongs in an extension

The core knows parameters, metrics, traces, frames, stages. It does not know
what a "goal", "terrain" or "object set" is. Add those through an `Extension`
(`commands()`, `metrics()`, `select_envs()`, `snapshot()`/`restore()`), which
makes them reachable from the CLI, from MCP `run_command`, **and from a
curriculum stage's `apply`** — all three at once. `rlmcp/extensions/terrain.py`
is the reference for one that ships here; a task package's own extension
module is the task-side equivalent.

An extension in another package registers itself through an entry point in the
`rlmcp.extensions` group, so nothing in this repo has to be edited to add one.

Extension `metrics()` merge into telemetry, so they are usable directly in
promotion conditions. Prefer task-external metrics there (success rate, goals
per minute, holding fraction) over the reward being optimised.

### `select_envs` is dispatched by signature

`ExtensionRegistry.select_envs` calls each extension as
`extension.select_envs(**criteria)` and treats a `TypeError` as "different
vocabulary, not an error". So an extension that declares

```python
def select_envs(self, where: dict): ...      # WRONG: never binds
```

is skipped for every `--where` query, and `rlmcp shot --where holding=true`
fails with *"No extension understands {'holding': 'true'}. This run has: …"* —
naming, in that list, the very extension that was meant to answer. Declare it as
`**criteria`, pop the keys you understand, and return `None` if anything is
left over:

```python
def select_envs(self, **criteria):
    holding = criteria.pop("holding", None)
    if criteria or holding is None:
        return None                          # not our vocabulary
    ...
```

## Promotion metrics: normalise them

Per-episode counts are not comparable across runs whose episodes end at
different lengths. Publish rates (per minute of simulated time) or fractions.
A metric that rises simply because episodes got longer will promote a policy
that has not improved.

## The record is the deliverable — and it lives in the tasks repo

`rlmcp record new` before launching (hypothesis, prediction, falsifier,
`--falsify-when` for the machine-checked version), `rlmcp record close` after,
with measurements. A `validated` verdict without evidence is refused, and that
is deliberate. `rlmcp record check` validates every record; `rlmcp record graph`
draws the lineage.

All of it writes to `$RLMCP_RECORDS`, which points at the records repository. When
you are working *in this repo* — fixing the harness, adding an adapter — you are
usually not filling a record at all. When you are running an experiment, do it
from the records and follow that repo's `AGENTS.md`.

## Inspection is concurrent

`shot`/`video`/`trace`/`diagnose` are deferred jobs serviced between rollout
batches inside the training process; parameter edits apply between batches, so
they cannot race the simulator. Never pause training to look at it. A separate
probe process (its own small env) is also fine and does not touch the run.

## Tests

See the Tests section of `README.md` for the current incantation. The suite runs
the real controller against a fake simulator — no GPU and no simulator needed —
and the MCP-server tests skip unless the optional `mcp` package is installed.
