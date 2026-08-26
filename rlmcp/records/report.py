"""PLAN.md and REPORT.md, generated from what the run actually recorded.

The discipline these documents encode is not new, and neither are their sections.
What is new is that rlmcp was attached to the run, so the evidence half does not
have to be typed from memory: the metrics, the parameter changes with the
rationale their author gave, the curriculum transitions and the artifacts are all
already on disk in the session.

The generator fills what it can measure and leaves the rest as headings with a
prompt. A report is an argument; only the author can make it. What this removes
is the excuse that assembling the evidence was too much work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from rlmcp.records.record import RunRecord, fold_recipe
from rlmcp.session import Session

_PLAN_PROMPTS = {
    "hypothesis": "One sentence, mechanistic. Not \"this should work better\".",
    "prediction": "The number you expect to move, how far, and by when.",
    "falsifier": "What observation would prove this wrong? If you cannot name "
                 "one, this is not an experiment.",
    "change": "Exactly what differs from the parent run. One conceptual change.",
}


def render_plan(record: RunRecord, recipe: Optional[Sequence] = None) -> str:
  """The pre-registration document."""
  lines = [f"# {record.id} — {record.slug.replace('_', ' ')}", ""]

  if record.parent:
    origin = f"Parent: `{record.parent}`"
    if record.weights:
      origin += f" · warm start from `{record.weights.describe()}`"
    else:
      origin += " · from scratch"
    lines += [origin, ""]

  lines += ["## Hypothesis", ""]
  lines += [record.hypothesis or f"_{_PLAN_PROMPTS['hypothesis']}_", ""]

  lines += ["## Prediction", ""]
  lines += [record.prediction or f"_{_PLAN_PROMPTS['prediction']}_", ""]

  lines += ["## Falsifier", ""]
  lines += [record.falsifier.prose or f"_{_PLAN_PROMPTS['falsifier']}_", ""]
  if record.falsifier.conditions:
    lines += ["Checked automatically:", ""]
    lines += [f"- `{c.describe()}`" for c in record.falsifier.conditions]
    lines += [""]

  lines += ["## Change", ""]
  if record.change:
    lines += [f"- {c}" for c in record.change]
  else:
    lines += [f"_{_PLAN_PROMPTS['change']}_"]
  lines += [""]

  if record.weights is not None:
    lines += [
        "> **Warm start.** This run can show the change *preserves* a behaviour,",
        "> not that it *creates* one. Its verdict caps at `provisional` until a",
        "> from-scratch run promotes it.",
        "",
    ]

  if recipe:
    lines += ["## Recipe at this node", "",
              "_The fold of every change from the root — computed, not stored._", ""]
    for rid, changes in recipe:
      for change in changes or ["(no change recorded)"]:
        lines += [f"- `{rid}` — {change}"]
    lines += [""]

  if record.config:
    lines += ["## Resolved config at launch", "",
              f"{len(record.config)} parameters snapshotted; see `meta.json`.", ""]

  return "\n".join(lines)


def gather_evidence(
    session_dir: Optional[str], window: int = 50
) -> Dict[str, Any]:
  """Pull the run's own record of itself out of its session directory.

  ``metrics.jsonl`` is tail-read: only the last ``window`` rows are parsed, and
  ``final_metrics`` is the newest value per metric across them. The final row
  alone would read any metric logged less often than every iteration as absent
  -- "undecidable" at close-out -- when the run did measure it. The window
  actually merged is recorded in the evidence as ``metrics_window``.
  """
  if not session_dir or not Path(session_dir).exists():
    return {}
  try:
    session = Session.open(session_dir)
  except FileNotFoundError:
    return {}

  rows = session.metrics(last_n=window)
  events = session.events()
  final = rows[-1] if rows else {}

  final_values: Dict[str, float] = {}
  for row in rows:  # oldest first, so the newest logged value wins.
    for key, value in row.items():
      if isinstance(value, (int, float)) and key not in ("iteration", "t"):
        final_values[key] = value

  try:
    with session.metrics_file.open() as f:
      num_rows = sum(1 for line in f if line.strip())
  except OSError:
    num_rows = len(rows)

  interventions = [
      {
          "iteration": e.get("iteration"),
          "key": e.get("key"),
          "old": e.get("old"),
          "new": e.get("new"),
          "rationale": e.get("rationale", ""),
      }
      for e in events
      if e.get("kind") == "set_parameter"
  ]
  stages = [
      {"iteration": e.get("iteration"), "to": e.get("to"), "notes": e.get("notes", "")}
      for e in events
      if e.get("kind") == "curriculum_stage"
  ]
  notes = [e.get("text", "") for e in events if e.get("kind") == "note"]

  artifacts = sorted(p.name for p in session.artifacts.glob("*")) if session.artifacts.exists() else []

  return {
      "iterations": final.get("iteration"),
      "final_metrics": final_values,
      "metrics_window": len(rows),
      "num_metric_rows": num_rows,
      "interventions": interventions,
      "stages": stages,
      "notes": notes,
      "artifacts": artifacts,
      "session": str(session.dir),
  }


def render_report(
    record: RunRecord,
    evidence: Optional[Dict[str, Any]] = None,
    falsifier_result: Optional[Dict[str, Any]] = None,
) -> str:
  """The close-out document, with the measurable half already filled in."""
  evidence = evidence or {}
  lines = [f"# {record.id} — REPORT", ""]

  lines += ["## Outcome", ""]
  lines += [record.outcome or "_Prediction met, missed, or ambiguous — say which in the first line._", ""]
  lines += [f"**Verdict:** `{record.verdict}`", ""]

  if falsifier_result:
    lines += ["## Falsifier", ""]
    if record.falsifier.prose:
      lines += [f"> {record.falsifier.prose}", ""]
    if falsifier_result.get("too_early"):
      lines += [f"**Not evaluable yet** — the run closed at iteration "
                f"{falsifier_result.get('evaluated_at')}, before the "
                f"pre-registered read-point (check_after "
                f"{falsifier_result.get('check_after')}). Neither fired nor "
                "held: the run ended before its falsifier meant anything.", ""]
    elif falsifier_result.get("fired"):
      lines += ["**It fired.** The hypothesis is dead, which is a result.", ""]
    elif falsifier_result.get("undecidable"):
      lines += ["**Undecidable** — the metric it names was never logged. That is "
                "not survival; the run could not test its own hypothesis.", ""]
    elif record.falsifier.conditions:
      lines += ["**It did not fire.**", ""]
    for check in falsifier_result.get("checks", []):
      mark = "FIRED" if check["fired"] else ("n/a" if not check["measurable"] else "held")
      lines += [f"- `{check['condition']}` — {mark} (current {check['current']})"]
    lines += [""]

  lines += ["## Evidence", ""]
  if record.metrics:
    lines += ["| measurement | value |", "| --- | --- |"]
    lines += [f"| {name} | {value} |" for name, value in record.metrics]
    lines += [""]
  final = evidence.get("final_metrics") or {}
  if final:
    logged = f"{evidence.get('num_metric_rows')} logged"
    window = evidence.get("metrics_window")
    if window and window > 1:
      logged += f"; final values are the newest per metric over the last {window} rows"
    lines += [f"Final iteration: **{evidence.get('iterations')}** ({logged}).", ""]
    headline = {k: v for k, v in final.items() if k.startswith(("Train/", "rlmcp/"))}
    if headline:
      lines += ["| metric | final |", "| --- | --- |"]
      lines += [f"| `{k}` | {v:.5g} |" for k, v in sorted(headline.items())]
      lines += [""]

  if evidence.get("stages"):
    lines += ["### Curriculum", ""]
    lines += [f"- iteration {s['iteration']}: → `{s['to']}` {s['notes']}".rstrip()
              for s in evidence["stages"]]
    lines += [""]

  if evidence.get("interventions"):
    lines += ["### Live interventions", "",
              "_Changed mid-run, so the final config is not the launch config._", "",
              "| iteration | parameter | from | to | why |", "| --- | --- | --- | --- | --- |"]
    lines += [
        f"| {i['iteration']} | `{i['key']}` | {i['old']} | {i['new']} | {i['rationale']} |"
        for i in evidence["interventions"]
    ]
    lines += [""]

  if record.assets:
    lines += ["### Artifacts", ""]
    for kind, entries in record.assets.items():
      for entry in entries:
        caption = entry[1] if len(entry) > 1 else ""
        lines += [f"- {kind}: `{entry[0]}` {caption}".rstrip()]
    lines += [""]
  elif evidence.get("artifacts"):
    lines += ["### Artifacts in the session (not yet registered)", ""]
    lines += [f"- `{name}`" for name in evidence["artifacts"][:12]]
    lines += [""]

  if evidence.get("notes"):
    lines += ["### Notes recorded during the run", ""]
    lines += [f"- {n}" for n in evidence["notes"]]
    lines += [""]

  lines += ["## Diagnosis", "",
            "_For a failure: the mechanism, backed by a measurement. "
            "\"Self-collision was the dominant termination at 1.83 vs 0.75 for falls\" "
            "beats \"it seemed unstable\"._", ""]
  lines += ["## Belief update", "",
            "_What you now think that you did not before, and what it invalidates "
            "in earlier runs._", ""]
  lines += ["## Next plan", "",
            "_The single next change and its falsifier._", ""]

  if record.weights is not None:
    lines += ["---", "",
              f"_Warm-started from {record.weights.describe()}: this result shows the "
              "change preserves a behaviour, not that it creates one._", ""]

  return "\n".join(lines)


def write_plan(store: Any, record: RunRecord) -> Optional[Path]:
  """Render and persist PLAN.md beside the record."""
  records = {r.id: r for r in store.list_records()}
  try:
    recipe = fold_recipe(record.id, records)
  except ValueError:
    recipe = None
  return store.write_document(record.id, "PLAN.md", render_plan(record, recipe))


def write_report(
    store: Any,
    record: RunRecord,
    falsifier_result: Optional[Dict[str, Any]] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
  """Render and persist REPORT.md, gathering evidence from the session.

  A caller that already gathered evidence (the close-out does, to evaluate the
  falsifier) passes it in rather than paying for a second read.
  """
  if evidence is None:
    evidence = gather_evidence(record.session)
  return store.write_document(
      record.id, "REPORT.md", render_report(record, evidence, falsifier_result)
  )


def final_metrics(session_dir: Optional[str]) -> Dict[str, float]:
  """The newest value per metric at the end of the run (see gather_evidence)."""
  evidence = gather_evidence(session_dir)
  return evidence.get("final_metrics", {})
