# Extensions: teaching rlmcp your task's words

rlmcp's core knows about parameters, metrics, traces, frames and stages. It
knows nothing about terrain, object sets or goal distributions, and it should
not. A library that hardcodes one task's vocabulary makes every other task carry
dead weight.

An extension adds that vocabulary. One class buys you three things at once: a
shell verb, an MCP `run_command`, and something a curriculum stage can call.

## The hooks

| hook | what it adds |
| --- | --- |
| `commands()` | new verbs, reachable from CLI, MCP and curriculum stages |
| `metrics()` | scalars merged into telemetry, usable in promotion conditions |
| `select_envs()` | answers `where={"terrain": "stairs"}`. The core never parses it |
| `snapshot()` / `restore()` | state saved and restored with the policy |
| `bind()` | receives the controller's `ExtensionContext` after registration |
| `on_iteration()` / `close()` | a call every learning iteration, and one at shutdown |
| `available()` | whether this extension fits the running environment |

`ExtensionContext` is how an extension reaches controller facilities:
`write_artifact`, `telemetry`, `append_event`, `submit_job`, `pending_jobs`.

A command handler may return a `DeferredJob` (from `rlmcp.core.controller`) to
answer over the coming simulation steps — "watch the robot for N steps and
report" — serviced exactly like the built-in video and trace commands.

Extensions take just the environment to construct. The old `(env, plot_sink)`
constructor still builds but is deprecated in favour of `bind`.

## Writing one

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
    # Per minute of simulated time, not per episode. See curriculum.md.
    return {"rlmcp/goal_rate_per_min": self._goal_rate()}

  def select_envs(self, **criteria):
    """Understands `holding=true`."""
    holding = criteria.pop("holding", None)
    if criteria or holding is None:
      return None                       # not our vocabulary; let others answer
    want = str(holding).lower() in ("1", "true", "yes")
    return [i for i, held in enumerate(self._holding()) if held == want]

  def cmd_set_goal_difficulty(self, tolerance: float, mode: str = "random") -> dict:
    """Set how close counts as reaching the goal, and how goals are sampled."""
    ...
```

That gets you:

```bash
rlmcp run set_goal_difficulty tolerance=0.2 mode=random
rlmcp shot --where holding=true
rlmcp metrics rlmcp/goal_rate_per_min
```

plus `run_command` over MCP, plus `Action("set_goal_difficulty", {...})` inside a
curriculum stage.

Values are parsed as JSON, so quote lists:
`rlmcp run set_objects names='["cube","mug"]'`.

**Write the docstrings for a reader who cannot see the code.** A command's
docstring is what `rlmcp commands` prints, and it is often all an agent has
before calling it.

## The `select_envs` trap

The registry calls each extension as `extension.select_envs(**criteria)` and
treats a `TypeError` as "different vocabulary, not an error". So this:

```python
def select_envs(self, where: dict): ...      # WRONG: never binds
```

is skipped for every `--where` query, silently. `rlmcp shot --where holding=true`
then fails with *"No extension understands {'holding': 'true'}. This run has: …"*
naming, in that list, the very extension that was meant to answer.

Declare it as `**criteria`, pop the keys you understand, and return `None` if
anything is left over.

## Getting it loaded

Two ways.

**Import the module.** Importing is what runs the `@register` decorator. Enough
for a launcher script you control:

```python
import mytask.rlmcp_ext      # registers the extension
```

**Publish an entry point**, and it is found without anyone importing it:

```toml
[project.entry-points."rlmcp.extensions"]
mytask = "mytask.rlmcp_ext"
```

Either way, `available()` decides whether it actually attaches. An extension that
does not fit the running environment is left out with a note instead of breaking
the run. Same for one that fails to import or fails to construct: never fatal to
training.

```
$ rlmcp extensions                   # what is installed
$ rlmcp extensions --available       # what this run actually loaded

$ rlmcp-train Mjlab-Velocity-Rough-Unitree-G1 ...
[rlmcp] extensions: terrain          →  set_terrain, terrain_status, plot_terrain

$ rlmcp-train Mjlab-Lift-Cube-Yam ...
[rlmcp] extensions: none             →  core verbs only, nothing broken
```

Pass `extensions=[...]` or `exclude_extensions=[...]` to `rlmcp.wrap` to override
the automatic choice.

## Where to put it

An extension for **a task** goes in that task's package, in your own repository.
An extension for **a capability many tasks share** can live in
`rlmcp/extensions/`. `rlmcp/extensions/terrain.py` is the worked example and the
one to copy from.

The layering rule in one sentence: *the core never learns a task's vocabulary,
and the adapter never learns a capability.*
