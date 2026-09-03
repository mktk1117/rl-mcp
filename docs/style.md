# Style

`ruff check` is the style guide. `[tool.ruff]` in
[`pyproject.toml`](../pyproject.toml) is where it is written down; this page is
the long form — what each rule is doing there, and why the ones that are off
are off.

The short version:

```bash
uvx ruff@0.16.4 check rlmcp tests examples        # what CI runs
uvx ruff@0.16.4 check --fix rlmcp tests examples  # and what fixes most of it
```

Two-space indent, four for a continuation, 100 columns as a wall. There is no
formatter, on purpose.

## Two spaces, and no formatter

The style here was read off the tree rather than chosen for it: 99 of 102
modules indent by two, and the 95th-percentile line is 79 characters. So
`line-length = 100` is a wall to catch the runaway line, not a target to write
up to — keep writing to about 80.

`ruff format` and black are 4-space tools. Black cannot emit two spaces at all,
and either one rewrites 99 of the 102 modules here, re-wrapping line breaks that
were placed by hand to keep a signature or a dict readable. Those breaks carry
intent that a formatter cannot see. So the check is a linter, and the layout
stays a judgement call.

A file that obeys all of it looks like this:

```python
"""One-line summary, then a blank line."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from typing import Any

from rlmcp.core.parameters.spec import ParameterSpec


class TelemetryBuffer:
  """Ring buffer for streaming scalar metrics during RL training."""

  def __init__(
      self,
      maxlen: int = 5000,
      on_drop: Callable[[str, Any], None] | None = None,
  ):
```

Block indent is two; a continuation is four, so a `def` at column 2 puts its
parameters at column 6. Ruff enforces the indent width. The four-space
continuation is convention it cannot check — follow the file you are in.

## Why each rule is in the list

### The two that were already CI's job

**`F` — pyflakes.** `F821` (undefined name) is the rule this repo's lint job was
originally built around: it catches a `NameError` that byte-compiling cannot see
and no test reached. `F401` (unused import) matters extra here because the
import block *is* the layering contract — `rlmcp/server/` importing a simulator
would be a design violation, and a stale import hides which dependencies are
real. `F811` catches a second `def` of the same name quietly winning.

**`E4`, `E7`, `E9`.** `E9` is syntax and IO errors. `E7` is the comparison
mistakes (`== None`, `type(x) == y`). `E4` is import placement, which paid for
itself in `mcp_server.py`, where `from rlmcp.session import ...` sat below the
optional-SDK try/except ladder and made the module's real dependencies invisible.

### Spelling: one way to say each thing

**`UP` — pyupgrade.** The largest group by far. `requires-python = ">=3.10"`, so
`List[int]` and `list[int]` are the same type with two spellings, and a reader
who meets both reasonably wonders whether the difference means something.
`typing.List` is deprecated upstream, and `from __future__ import annotations`
is already in these files, so there is no runtime difference to weigh. Pure
consistency at no risk — which is why it was safe to apply mechanically.

**`I` — import order.** Deterministic order means a diff shows the *new*
dependency instead of a reshuffle. In a repo whose central rule is "the core
never learns a task's vocabulary", the import block is where you see that rule
being broken.

**`W`.** Trailing whitespace. A line that differs only by an invisible character
costs a reviewer real time.

**`E501`, at 100.** The wall.

**`C4`, `PIE`.** `dict()` becomes `{}`, `set([...])` becomes `{...}`,
`**{"k": v}` becomes `k=v`. Marginally fewer allocations; mostly one spelling.

### Bugs, not style

**`B` — bugbear.** `B905` is the strongest one here: `zip(weights, a, b)` in
`palette.py` silently truncates RGBA against three weights. That truncation is
correct and intended — but nothing said so, and the next person could not have
known. It now says `strict=False`. `B904` was dropping the original exception
behind "the file there does not parse". `B008` (a call in a default argument) is
evaluated once at import and shared by every call.

**`RET` — return paths.** `RET503` earned its place outright. This repo passes
mutation callbacks to `update_record`, whose contract is "return `False` to
abort", tested with `is False`. Six callbacks fell off the end — which means
`None`, which means "write it" — and you had to know the store's internals to
read that. They say `return None` now, with the reason.

**`RUF`.** `RUF100` (unused noqa) matters most: a suppression that outlives its
violation is a lie about the code. `RUF012` (mutable class default) is shared
across instances. `RUF059` caught `env, lab, agent_cfg, vec_env = build_env(...)`
with `env` unused — usually the sign a tuple shape changed and the unpack did not.

**`PLW`.** `PLW2901`: after `line = line.strip()` the original is gone, and a
later reader cannot tell which one they are holding. `PLW1510` (subprocess
without `check`) is silent failure.

**`PLE`.** Pylint's outright-wrong bucket — an `__all__` that is not a list of
strings, and the like.

**`ISC`.** The comma somebody forgot: `["alpha" "beta"]` is one string, silently.
This repo builds a lot of prose message lists, so that is a live risk.

**`SIM`.** `SIM105`: `try/except/pass` is a four-line shape that reads like an
unfinished edit, where `with contextlib.suppress(Exception):` says "ignoring
this is deliberate" in one line. `SIM115` catches a leaked file handle.

**`TRY300`.** A `return` inside `try` widens the guard over code that was never
meant to be guarded. Moving it to `else` keeps the `try` body down to the thing
that can actually fail.

**`DTZ`.** A stored timestamp without a zone is a bug: records get compared
across runs and machines, and two naive timestamps from different zones sort
wrongly. The rule is on so that *stored* times stay zone-aware; the two display
sites carry a noqa.

**`LOG`, `G`.** Logging misuse — an f-string in a log call formats even when the
level is off, `logging.warn` is deprecated. These have never fired. They are a
guard, because this is library code that warns into somebody else's application.

## The escape hatch, and how to spell it

A `noqa` is fine. A bare one is not: say which rule, and say why above the line.

```python
# The lambda is load-bearing, not decoration: it is what puts the attribute
# lookup inside _try's guard. Unwrapped, it is evaluated eagerly at the
# call and an adapter that lacks the attribute raises instead of
# reporting "unknown".
"max_episode_length_s": _try(lambda: lab.sim.max_episode_length()),  # noqa: PLW0108
```

That one is load-bearing for real: applying the "fix" automatically broke five
gates in `tests/test_check.py`. The same pattern covers the other deliberate
ones — local time in a directory name a person has to find (`DTZ005`), and an
any-exception assertion where the point is that *whatever* matplotlib raises,
the figure still closes (`B017`).

Do not write a comment that begins with `# noqa` unless it is one. Ruff parses
it as a malformed directive and warns.

## What is off, and why

Nobody should have to re-derive these.

| rule | hits | why it is off |
| --- | --- | --- |
| `BLE001` blind except | 78 | Catching everything is the contract for inspection code: a failed screenshot must not take the training run down with it. |
| `ARG00x` unused argument | 131 | Adapters implement a protocol. An implementation that ignores an argument is doing its job, not forgetting one. |
| `T201` print | 45 | All in the CLI, which prints for a living. |
| `PLR2004` magic value | 231 | Nearly all in test assertions, where `3` means three. |
| `TRY003` long message at the raise | 169 | Long, explanatory messages at the raise site are the house style — an error should name the fix. The rule would push every one into a custom exception class and make them worse. |
| `TRY004` `TypeError` for a bad type | 3 | These helpers are documented to raise `ValueError`, and the CLI turns that into a message for the user. Switching is a public behaviour change, not a lint fix. |
| `PLW0603` `global` | 7 | Module-level singletons are the design here. The alternative is a class with exactly one instance. |
| `E501` in `records/views.py` | 4 | The long lines are inside the record graph's embedded JavaScript. A noqa cannot go on a line that is inside a string literal, and re-wrapping another language to satisfy a Python linter is worse than the long line. |

### The one that is off and should not stay off

Complexity — `C901`, `PLR0912`, `PLR0913`, `PLR0915` — is a real maintainability
rule and it is not enforceable yet. Eleven functions score above 15 on mccabe,
topping out at 62 (`_dispatch` in `cli.py`), and one signature takes 27
arguments. Any threshold that fits what is here would rubber-stamp it, so
nothing is set rather than something dishonest. This is tracked as its own
issue.

## Where it runs

Three places, one pinned version, one config, so they cannot disagree.

| | what it does |
| --- | --- |
| CI (`.github/workflows/ci.yml`) | A `lint` job on every push and pull request, before the test matrix. The gate that blocks a merge. |
| `.claude/settings.json` | A `PostToolUse` hook. An agent that writes a `.py` file gets it linted and auto-fixed on the spot; anything left comes back as an error instead of as a red pull request. |
| `.pre-commit-config.yaml` | For a human who has run `pre-commit install`. |

The version is pinned to `0.16.4` in all three. Ruff adds rules in minor
releases, and a lint job that goes red because of an upgrade nobody made is a
job people learn to ignore. Bump it in all three together, in a commit that also
fixes whatever the new release found.
