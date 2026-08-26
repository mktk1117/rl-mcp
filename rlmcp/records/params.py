"""What each parameter did, along the tree.

A run record says *what changed* in prose. This says *what the number was*, at
every iteration of every run, which is the question you actually have when a
ladder of ten runs has been tuning the same four reward weights and you want to
know whether the thing that worked was the weight or the story around it.

Three sources, in order of authority:

* ``record.config`` -- the resolved snapshot at launch. The starting value.
* ``curriculum_stage`` events -- ``applied.parameters`` at the promotion's
  iteration. The ladder moving a knob on its own.
* ``set_parameter`` events -- a live edit, with the rationale that was typed.

The x-axis is the whole point and the easy thing to get wrong. Every run starts
its own iteration counter at zero, so plotting raw iteration stacks ten runs on
top of each other and hides the lineage. Instead each run is *offset* by where
its config parent finished, which makes a fork read as a fork: two children of
005 both start at the x where 005 stopped, and diverge from there. ``offset``
is carried in the payload so a viewer can label either axis honestly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rlmcp.records.lineage import Graph
from rlmcp.records.record import RunRecord
from rlmcp.session import Session

# Parameters whose value is a number are plottable; a list-valued range or a
# string mode is still worth *listing* as changed, and is kept out of the
# series rather than coerced into one.
Number = (int, float)


def _numeric(value: Any) -> Optional[float]:
  if isinstance(value, bool) or not isinstance(value, Number):
    return None
  return float(value)


def _describe(value: Any) -> str:
  if isinstance(value, float):
    return f"{value:g}"
  return str(value)


def _session_facts(session_dir: Optional[str]) -> Tuple[List[Dict[str, Any]], int, bool]:
  """Parameter edits with their iterations, how far the run got, and liveness.

  Returns ``([], 0, False)`` for a session that is gone -- a lineage whose logs
  were cleaned up still draws, it just draws the launch snapshots.
  """
  if not session_dir:
    return [], 0, False
  try:
    session = Session.open(session_dir)
  except (FileNotFoundError, OSError):
    return [], 0, False

  edits: List[Dict[str, Any]] = []
  for event in session.events():
    kind = event.get("kind")
    iteration = event.get("iteration")
    if kind == "set_parameter" and event.get("applied", True):
      edits.append({
          "iteration": int(iteration or 0),
          "key": event.get("key"),
          "value": event.get("new"),
          "why": event.get("rationale", ""),
          "source": "manual",
      })
    elif kind == "curriculum_stage":
      applied = (event.get("applied") or {}).get("parameters") or {}
      for key, value in applied.items():
        edits.append({
            "iteration": int(iteration or 0),
            "key": key,
            "value": value,
            "why": f"stage {event.get('to')}: {event.get('reason', '')}".strip(": "),
            "source": "curriculum",
            "stage": event.get("to"),
        })
  edits.sort(key=lambda e: (e["iteration"], e["key"]))

  # How long the run ran. status.json is the live number and survives the
  # trainer exiting; the last edit is the floor when it does not.
  status = session.status()
  iterations = int(status.get("iteration") or 0)
  if not iterations:
    rows = session.metrics(last_n=1)
    iterations = int(rows[-1].get("iteration") or 0) if rows else 0
  if edits:
    iterations = max(iterations, edits[-1]["iteration"])
  # "Still training" is a fact about the process, not about the verdict: a run
  # whose trainer exited an hour ago is not an active path just because nobody
  # has closed the record yet.
  try:
    alive = session.liveness() != "dead"
  except Exception:  # noqa: BLE001 -- liveness is a nicety here.
    alive = False
  return edits, iterations, alive


def build_history(graph: Graph) -> Dict[str, Any]:
  """Per-run parameter traces, laid out along the lineage.

  The payload is deliberately viewer-shaped: a list of runs in draw order, each
  with its x offset and a ``series`` of step points per key, plus the index of
  keys that ever moved so a chooser can be built without walking every run.
  """
  runs: List[Dict[str, Any]] = []
  by_id: Dict[str, Dict[str, Any]] = {}
  keys: Dict[str, Dict[str, Any]] = {}

  for node_id in graph.order:
    record: RunRecord = graph.nodes[node_id].record
    edits, iterations, alive = _session_facts(record.session)

    parent = by_id.get(record.parent or "")
    offset = parent["end"] if parent else 0

    # Launch values. A key the run never touched still has a value, and a flat
    # line at the inherited value is information: it says nobody moved it.
    config = {k: v for k, v in (record.config or {}).items()}
    series: Dict[str, List[Dict[str, Any]]] = {}
    for key, value in config.items():
      number = _numeric(value)
      if number is None:
        continue
      series[key] = [{"x": 0, "v": number, "source": "launch"}]

    changed: List[str] = []
    non_numeric: List[str] = []
    for edit in edits:
      key = edit["key"]
      if not key:
        continue
      number = _numeric(edit["value"])
      if number is None:
        if key not in non_numeric:
          non_numeric.append(key)
        continue
      point = {"x": int(edit["iteration"]), "v": number,
               "source": edit["source"], "why": edit.get("why", "")}
      if edit.get("stage"):
        point["stage"] = edit["stage"]
      points = series.setdefault(key, [])
      # An edit at iteration 0 -- the curriculum's opening stage, applied
      # before the first rollout -- is what the run actually trained with. The
      # config snapshot is taken a moment earlier, so keeping both draws a
      # vertical spike at a value no policy ever saw. It replaces the launch
      # point instead, and the launch value survives as ``config_value``.
      if point["x"] == 0 and points and points[0]["source"] == "launch":
        point["config_value"] = points[0]["v"]
        points[0] = point
        if key not in changed and point["v"] != point["config_value"]:
          changed.append(key)
        continue
      # A stage that re-applies the value it already has is not a change; it
      # is the ladder restating itself, and drawing it as a step is a lie.
      if points and points[-1]["v"] == number:
        continue
      points.append(point)
      if key not in changed:
        changed.append(key)

    # Carry the line to the end of the run, so a step chart has something to
    # draw between the last edit and where the run actually stopped.
    for key, points in series.items():
      if points and points[-1]["x"] < iterations:
        points.append({"x": iterations, "v": points[-1]["v"], "source": "end"})

    entry = {
        "id": record.id,
        "slug": record.slug,
        "verdict": record.verdict,
        "parent": record.parent if record.parent in by_id else None,
        "offset": offset,
        "iterations": iterations,
        "end": offset + iterations,
        "series": series,
        "changed": changed,
        "unplottable": non_numeric,
        "live": record.verdict == "running" and alive,
        "open": record.verdict == "running",
    }
    runs.append(entry)
    by_id[record.id] = entry

    for key in series:
      slot = keys.setdefault(key, {"key": key, "runs": [], "changed_in": []})
      slot["runs"].append(record.id)
      if key in changed:
        slot["changed_in"].append(record.id)

  # A parameter that never moved anywhere is noise in a chooser built to answer
  # "what did we tune?", so the index is ordered by how much it moved -- and a
  # relaunch counts, because a weight the next run was configured with moved
  # just as surely as one edited mid-run.
  index = []
  for key, slot in keys.items():
    spans = [p["v"] for r in runs for p in r["series"].get(key, [])]
    live_edits, relaunched, previous = 0, [], None
    for run in runs:
      points = run["series"].get(key) or []
      if not points:
        continue
      if previous is not None and points[0]["v"] != previous:
        relaunched.append(run["id"])
      live_edits += len([p for p in points[1:]
                         if p["source"] in ("manual", "curriculum")])
      previous = points[-1]["v"]
    index.append({
        "key": key,
        "changed_in": slot["changed_in"],
        "relaunched_in": relaunched,
        "changes": live_edits + len(relaunched),
        "min": min(spans) if spans else 0.0,
        "max": max(spans) if spans else 0.0,
        "first": _describe(next((p["v"] for r in runs for p in r["series"].get(key, [])), "")),
        "last": _describe(next(
            (r["series"][key][-1]["v"] for r in reversed(runs) if r["series"].get(key)), "")),
    })
  index.sort(key=lambda e: (-e["changes"], e["key"]))

  return {"runs": runs, "index": index,
          "span": max([r["end"] for r in runs], default=0)}


def leaf_paths(graph: Graph) -> Dict[str, List[str]]:
  """The root-to-leaf chains, keyed by leaf.

  The highlight in the viewer is a path, not a node: "how did we get to the
  current best" is a question about the chain of runs that produced it. Live
  runs each contribute one, which is what makes "show all active paths" a
  lookup rather than a traversal in the browser.
  """
  paths: Dict[str, List[str]] = {}
  for node_id, node in graph.nodes.items():
    if node.children:
      continue
    chain, cursor, seen = [], node_id, set()
    while cursor and cursor in graph.nodes and cursor not in seen:
      seen.add(cursor)
      chain.append(cursor)
      cursor = graph.nodes[cursor].record.parent or ""
    paths[node_id] = list(reversed(chain))
  return paths
