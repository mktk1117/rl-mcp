"""What was *done* to a run, in order, with the reasons given at the time.

A run differs from another run in three ways: its code, its config, and what
somebody did to it while it was training. The first two are recorded as
snapshots; the third is the layer nobody looks at, and it is the one that most
often explains a result -- "the entropy coefficient was raised at iteration 900
because it had collapsed" is the answer, and it is sitting in ``events.jsonl``
already, next to the reason the agent typed.

So this is a reader, not a recorder. It selects the events that *changed the
run or recorded a judgement about it* out of a log that also carries job
bookkeeping, telemetry warnings and clip notices, and phrases each one in a
line a person can read. Everything else in the log stays where it is.

The vocabulary lives here because the events are rlmcp's own: a viewer that
picks its own list of "interesting kinds" goes stale the moment a command is
added, and a task repo that greps for ``set_parameter`` learns nothing about
the curriculum advancing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

KINDS: dict[str, str] = {
    "set_parameter": "parameter",
    "reset_parameters": "parameter",
    "curriculum_stage": "curriculum",
    "curriculum_auto": "curriculum",
    "reset_envs": "environment",
    "load_checkpoint": "weights",
    "checkpoint": "weights",
    "stop_requested": "run",
    "note": "said",
    "feedback": "said",
}
"""Event kinds that are interventions, and the layer each one belongs to.

Deliberately not "everything with an iteration": a rendered clip, a finished
job and a dropped telemetry key are things that *happened*, not things anybody
decided.
"""


def _phrase(event: dict[str, Any]) -> str:
  """One line saying what was done, in the run's own vocabulary."""
  kind = event.get("kind")
  if kind == "set_parameter":
    line = f"{event.get('key')}: {event.get('old')} → {event.get('new')}"
    return line if event.get("applied", True) else f"{line} (refused)"
  if kind == "reset_parameters":
    keys = event.get("keys") or []
    return f"reset {len(keys)} parameter(s) to their launch values"
  if kind == "curriculum_stage":
    return f"stage → {event.get('stage') or event.get('name') or '?'}"
  if kind == "curriculum_auto":
    return f"auto-promotion {'on' if event.get('enabled') else 'off'}"
  if kind == "reset_envs":
    return f"restarted {event.get('num_reset', 'some')} environment(s)"
  if kind == "load_checkpoint":
    return f"loaded weights from {event.get('path')}"
  if kind == "checkpoint":
    return f"checkpoint '{event.get('tag') or 'untagged'}'"
  if kind == "stop_requested":
    return "stop requested"
  if kind in ("note", "feedback"):
    return str(event.get("text") or "")
  return str(kind)


def _why(event: dict[str, Any]) -> str:
  for key in ("rationale", "reason", "why"):
    value = event.get(key)
    if value:
      return str(value)
  return ""


def from_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
  """The interventions in an event log, oldest first.

  Each is ``{iteration, kind, layer, what, why, at}`` -- enough for a timeline
  beside a metric chart, and the same shape a recipe distiller reads to turn a
  history of ad-hoc edits into a ladder.
  """
  out: list[dict[str, Any]] = []
  for event in events:
    if not isinstance(event, dict):
      continue
    kind = event.get("kind")
    if kind not in KINDS:
      continue
    what = _phrase(event)
    if not what:
      continue
    out.append({
        "iteration": int(event.get("iteration") or 0),
        "kind": kind,
        "layer": KINDS[kind],
        "what": what,
        "why": _why(event),
        "at": float(event.get("t") or 0.0),
    })
  out.sort(key=lambda i: (i["iteration"], i["at"]))
  return out


def from_session(session_dir: Any, reader: Callable[..., Any] | None = None
                 ) -> list[dict[str, Any]]:
  """The interventions of one run, read through rlmcp's own session reader."""
  from rlmcp.session import Session

  if reader is None:
    session = Session.open(session_dir)
    reader = session.events
  return from_events(reader())
