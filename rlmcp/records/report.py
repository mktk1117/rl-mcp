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

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rlmcp.records.record import OWED_A_RESPONSE, RunRecord, fold_recipe
from rlmcp.session import RESERVED_METRIC_KEYS, Session

_PLAN_PROMPTS = {
    "hypothesis": "One sentence, mechanistic. Not \"this should work better\".",
    "prediction": "The number you expect to move, how far, and by when.",
    "falsifier": "What observation would prove this wrong? If you cannot name "
                 "one, this is not an experiment.",
    "change": "Exactly what differs from the parent run. One conceptual change.",
}


def render_plan(record: RunRecord, recipe: Sequence | None = None) -> str:
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
    session_dir: str | None, window: int = 50
) -> dict[str, Any]:
  """Pull the run's own record of itself out of its session directory.

  ``metrics.jsonl`` is tail-read: only the last ``window`` rows are parsed, and
  ``final_metrics`` is the newest value per metric across them. The final row
  alone would read any metric logged less often than every iteration as absent
  -- "undecidable" at close-out -- when the run did measure it. The window
  actually merged is recorded in the evidence as ``metrics_window``.
  """
  if not session_dir:
    return {}
  try:
    session = Session.open(session_dir)
  except (FileNotFoundError, OSError):
    return {}

  rows = session.metrics(last_n=window)
  events = session.events()
  final = rows[-1] if rows else {}

  final_values: dict[str, float] = {}
  for row in rows:  # oldest first, so the newest logged value wins.
    for key, value in row.items():
      if isinstance(value, (int, float)) and key not in RESERVED_METRIC_KEYS:
        final_values[key] = value

  num_rows = session.metrics_count() or len(rows)

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
  # Feedback said to the trainer while it was running. The session knows the
  # iteration; the record does not, which is the whole reason to capture it
  # here rather than reconstructing it at close-out from memory.
  feedback = [
      {
          "iteration": e.get("iteration"),
          "at": e.get("t"),
          "kind": e.get("feedback_kind", "steer"),
          "author": e.get("author", "user"),
          "text": e.get("text", ""),
          "interpretation": e.get("interpretation", ""),
          "response": e.get("response", ""),
      }
      for e in events
      if e.get("kind") == "feedback"
  ]

  artifacts = sorted(a["name"] for a in session.list_artifacts())

  return {
      "iterations": final.get("iteration"),
      "final_metrics": final_values,
      "metrics_window": len(rows),
      "num_metric_rows": num_rows,
      "interventions": interventions,
      "stages": stages,
      "notes": notes,
      "feedback": feedback,
      "artifacts": artifacts,
      "session": session.address,
  }


def render_report(
    record: RunRecord,
    evidence: dict[str, Any] | None = None,
    falsifier_result: dict[str, Any] | None = None,
) -> str:
  """The close-out document, with the measurable half already filled in."""
  evidence = evidence or {}
  lines = [f"# {record.id} — REPORT", ""]

  lines += ["## Outcome", ""]
  lines += [record.outcome
            or "_Prediction met, missed, or ambiguous — say which in the first line._", ""]
  lines += [f"**Verdict:** `{record.verdict}`", ""]

  if falsifier_result:
    lines += ["## Falsifier", ""]
    if record.falsifier.prose:
      lines += [f"> {record.falsifier.prose}", ""]
    if falsifier_result.get("too_early"):
      lines += [(f"**Not evaluable yet** — the run closed at iteration "
                 f"{falsifier_result.get('evaluated_at')}, before the "
                 f"pre-registered read-point (check_after "
                 f"{falsifier_result.get('check_after')}). Neither fired nor "
                 "held: the run ended before its falsifier meant anything."), ""]
    elif falsifier_result.get("fired"):
      lines += ["**It fired.** The hypothesis is dead, which is a result.", ""]
    elif falsifier_result.get("undecidable"):
      lines += [("**Undecidable** — the metric it names was never logged. That is "
                 "not survival; the run could not test its own hypothesis."), ""]
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

  if record.feedback:
    lines += ["## What the humans said", "",
              ("_Verbatim, in the order it arrived, with what was done about it. "
               "Several of the load-bearing corrections in a project never appear "
               "in a metric._"), ""]
    for index, entry in enumerate(record.feedback):
      when = f"iteration {entry.iteration}" if entry.iteration is not None else "off-run"
      lines += [f"**[{index}] {entry.kind}** — {entry.author}, {when}", "",
                f"> {entry.text}", ""]
      if entry.interpretation:
        lines += [f"Read as: {entry.interpretation}", ""]
      if entry.answered:
        moved = "changed something" if entry.changed else "changed nothing"
        lines += [f"Response ({moved}): {entry.response}", ""]
      elif entry.kind in OWED_A_RESPONSE:
        lines += [("**No recorded response.** This instruction was heard and "
                   "not answered."), ""]
      else:
        # An observation or an approval asked for nothing, so an empty
        # response slot is not a dropped ball and must not read like one.
        lines += ["_No response recorded, and none was owed._", ""]
      if entry.affects:
        lines += [f"Also changed: {', '.join(entry.affects)}", ""]
      if entry.artifacts:
        lines += ["Artifacts: " + ", ".join(f"`{a}`" for a in entry.artifacts), ""]

  lines += ["## Diagnosis", "",
            ("_For a failure: the mechanism, backed by a measurement. "
             "\"Self-collision was the dominant termination at 1.83 vs 0.75 for falls\" "
             "beats \"it seemed unstable\"._"), ""]
  lines += ["## Belief update", "",
            ("_What you now think that you did not before, and what it invalidates "
             "in earlier runs._"), ""]
  lines += ["## Next plan", "",
            "_The single next change and its falsifier._", ""]

  if record.weights is not None:
    lines += ["---", "",
              (f"_Warm-started from {record.weights.describe()}: this result shows the "
               "change preserves a behaviour, not that it creates one._"), ""]

  return "\n".join(lines)


def render_feedback_ledger(rows: Sequence[dict[str, Any]]) -> str:
  """Every remark in the store as one readable ledger.

  Generated, never hand-maintained. A ledger kept by hand beside the records is
  a second source of truth that drifts the first time someone is in a hurry --
  the same reason the recipe is folded rather than stored. This reads the
  records and renders; the records stay the truth.
  """
  lines = ["# Feedback ledger", "",
           ("_Generated by `rlmcp record timeline --markdown`. Every remark lives "
            "on the run record it was said to; this is the fold of them, oldest "
            "first. Edit the records, not this file._"), ""]
  if not rows:
    lines += ["No feedback recorded yet.", ""]
    return "\n".join(lines)

  changed = sum(1 for r in rows if r.get("changed"))
  outstanding = [r for r in rows
                 if r["kind"] in OWED_A_RESPONSE and not (r.get("response") or "").strip()]
  by_kind: dict[str, int] = {}
  for row in rows:
    by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1

  lines += [f"**{len(rows)} remark{'' if len(rows) == 1 else 's'}** — "
            + ", ".join(f"{n} {k}" for k, n in sorted(by_kind.items()))
            + f". {changed} changed something.", ""]
  if outstanding:
    lines += [f"**{len(outstanding)} unanswered**: "
              + ", ".join(f"{r['run']}[{r['index']}]" for r in outstanding) + ".", ""]

  lines += ["| # | run | kind | said | what changed |", "| --- | --- | --- | --- | --- |"]
  for n, row in enumerate(rows, 1):
    said = _cell(row.get("text", ""))
    did = _cell(row.get("response", "")) or "_no recorded response_"
    if row.get("response") and not row.get("changed"):
      did += " _(nothing changed)_"
    lines.append(f"| {n} | `{row['run']}[{row['index']}]` | {row['kind']} | {said} | {did} |")
  lines.append("")

  lines += ["## In full", ""]
  for n, row in enumerate(rows, 1):
    when = (f"iteration {row['iteration']}" if row.get("iteration") is not None
            else "off-run")
    lines += [(f"### {n}. {row['run']}[{row['index']}] — {row['kind']} "
               f"({row.get('author', 'user')}, {when})"), "",
              f"> {row.get('text', '')}", ""]
    if row.get("interpretation"):
      lines += [f"**Read as:** {row['interpretation']}", ""]
    if (row.get("response") or "").strip():
      lines += [f"**Done:** {row['response']}", ""]
      if not row.get("changed"):
        lines += [("Nothing changed as a result, and that is recorded rather "
                   "than quietly omitted."), ""]
    else:
      lines += ["**No recorded response.**", ""]
    if row.get("affects"):
      lines += [f"**Also changed:** {', '.join(row['affects'])}", ""]
    if row.get("artifacts"):
      lines += ["**Artifacts:** " + ", ".join(f"`{a}`" for a in row["artifacts"]), ""]
  return "\n".join(lines)


def _cell(text: str, limit: int = 110) -> str:
  """One table cell: no pipes, no newlines, not the whole paragraph."""
  flat = " ".join(str(text).split()).replace("|", "\\|")
  return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def write_plan(store: Any, record: RunRecord) -> Path | None:
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
    falsifier_result: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
) -> Path | None:
  """Render and persist REPORT.md, gathering evidence from the session.

  A caller that already gathered evidence (the close-out does, to evaluate the
  falsifier) passes it in rather than paying for a second read.
  """
  if evidence is None:
    evidence = gather_evidence(record.session)
  return store.write_document(
      record.id, "REPORT.md", render_report(record, evidence, falsifier_result)
  )


def final_metrics(session_dir: str | None) -> dict[str, float]:
  """The newest value per metric at the end of the run (see gather_evidence)."""
  evidence = gather_evidence(session_dir)
  return evidence.get("final_metrics", {})
