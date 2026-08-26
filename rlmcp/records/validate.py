"""The epistemic rules, as code rather than as prose.

Every check here was paid for by a real failure in the project this method comes
from. Writing them down in a README makes them advice; making them a build error
makes them a property of the records.

The whole module is a pure function over a list of records -- no store, no
filesystem. That is deliberate: it runs client-side today, and unchanged inside a
service later, when a hundred clients each enforcing their own rules would mean
no rules at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from rlmcp.records.record import (
    FEEDBACK_KINDS,
    OPEN_VERDICTS,
    TERMINAL_VERDICTS,
    RunRecord,
    VERDICTS,
)

REQUIRED_FOR_RESULT = ("outcome", "metrics")
"""Fields a record must fill once it claims a result.

A claim without a measurement is not knowledge, so a terminal verdict with an
empty ``metrics`` list is an error rather than an omission.
"""

RESULT_VERDICTS = ("validated", "falsified", "provisional", "best")
"""Verdicts that claim a result, and therefore owe evidence."""


def missing_result_evidence(
    record: RunRecord, verdict: Optional[str] = None
) -> List[str]:
  """The evidence fields a result claim lacks.

  The one implementation of "a claim without a measurement is not knowledge",
  shared by the whole-records check and the close-time gate so the two cannot
  drift apart. ``verdict`` is the claim being judged; it defaults to the one
  already on the record.

  ``falsifier`` rows do not count as measurements. The close-out appends one
  mechanically, and it records the falsifier's *state*: a held falsifier is
  absence of disproof, not affirmative evidence for the claim.
  """
  claimed = record.verdict if verdict is None else verdict
  if claimed not in RESULT_VERDICTS:
    return []
  missing = []
  for name in REQUIRED_FOR_RESULT:
    value = getattr(record, name)
    if name == "metrics":
      value = [row for row in value if row and row[0] != "falsifier"]
    if not value:
      missing.append(name)
  return missing


@dataclass
class Problem:
  """One failed check."""

  level: str  # "error" or "warning"
  record_id: Optional[str]
  rule: str
  message: str

  def describe(self) -> str:
    where = f"{self.record_id}: " if self.record_id else ""
    return f"[{self.level}] {where}{self.message}"

  def to_dict(self) -> Dict[str, str]:
    return {
        "level": self.level,
        "record_id": self.record_id or "",
        "rule": self.rule,
        "message": self.message,
    }


@dataclass
class Report:
  """What validation found."""

  errors: List[Problem] = field(default_factory=list)
  warnings: List[Problem] = field(default_factory=list)

  @property
  def ok(self) -> bool:
    return not self.errors

  def to_dict(self) -> Dict[str, object]:
    return {
        "ok": self.ok,
        "errors": [p.to_dict() for p in self.errors],
        "warnings": [p.to_dict() for p in self.warnings],
    }

  def describe(self) -> str:
    if self.ok and not self.warnings:
      return "All records pass."
    lines = [p.describe() for p in self.errors + self.warnings]
    return "\n".join(lines)


def validate(
    records: Sequence[RunRecord],
    slots: int = 1,
    media_exists: Optional[Dict[str, bool]] = None,
) -> Report:
  """Check a whole records.

  Args:
    records: every record in the store.
    slots: how many runs may hold a lease at once. One machine, one trainer, one
      slot -- which reproduces the original single-``running`` mutex exactly.
    media_exists: optional ``{asset_key: bool}`` so asset checks can run without
      this module touching a filesystem.
  """
  report = Report()
  by_id: Dict[str, RunRecord] = {}

  def error(record_id: Optional[str], rule: str, message: str) -> None:
    report.errors.append(Problem("error", record_id, rule, message))

  def warn(record_id: Optional[str], rule: str, message: str) -> None:
    report.warnings.append(Problem("warning", record_id, rule, message))

  # Identity.
  seen_ids, seen_seq = set(), {}
  for record in records:
    if record.id in seen_ids:
      error(record.id, "unique-id", f"Duplicate record id '{record.id}'.")
    seen_ids.add(record.id)
    by_id[record.id] = record
    if record.seq in seen_seq and record.seq:
      error(
          record.id, "unique-seq",
          f"seq {record.seq} is already used by '{seen_seq[record.seq]}'.",
      )
    seen_seq[record.seq] = record.id

  for record in records:
    # Vocabulary.
    if record.verdict not in VERDICTS:
      error(
          record.id, "verdict-enum",
          f"Unknown verdict '{record.verdict}'. Use one of: {', '.join(VERDICTS)}.",
      )
    if not record.slug:
      error(record.id, "required", "slug is empty.")

    # Referential integrity.
    for field_name, ref in (
        ("parent", record.parent),
        ("prior", record.prior),
        ("weights.run", record.weights.run if record.weights else None),
    ):
      if ref is not None and ref not in by_id and ref not in seen_ids:
        error(
            record.id, "reference",
            f"{field_name} points at '{ref}', which is not a record here.",
        )

    # A claim without a measurement is not knowledge.
    for name in missing_result_evidence(record):
      error(
          record.id, "result-needs-evidence",
          f"verdict '{record.verdict}' with no {name} evidence -- "
          "a claim without a measurement is not knowledge.",
      )

    # The warm-start cap. The single highest-value rule in the method.
    if record.verdict in ("validated", "best") and record.weights is not None:
      error(
          record.id, "warm-start-cap",
          f"verdict '{record.verdict}' on a run warm-started from "
          f"{record.weights.describe()}. A warm start shows a change PRESERVES a "
          "behaviour, not that it CREATES one -- cap it at 'provisional' and "
          "promote it with a from-scratch run.",
      )

    # An experiment needs something that could kill it.
    if record.verdict in ("planned", "running"):
      if not record.hypothesis:
        warn(record.id, "hypothesis", f"'{record.verdict}' without a hypothesis.")
      if record.falsifier.is_empty():
        warn(
            record.id, "falsifier",
            f"'{record.verdict}' without a falsifier -- if you cannot name an "
            "observation that would prove this wrong, it is not an experiment.",
        )

    # One conceptual change per run; two make an ambiguous result.
    if len(record.change) > 1 and record.verdict not in ("planned", "superseded"):
      warn(
          record.id, "multi-change",
          f"{len(record.change)} changes in one run -- the outcome is "
          "directional, not attributable to any single change.",
      )

    # Feedback that was heard and dropped.
    #
    # A warning rather than an error: the answer legitimately arrives after the
    # remark, so an open run with an outstanding instruction is just work in
    # progress. Once the run is closed, nobody is coming back to it -- an
    # instruction with no recorded response at that point was dropped, and the
    # records should say so out loud rather than quietly keeping the text.
    for index, entry in record.outstanding_feedback():
      if record.verdict in OPEN_VERDICTS:
        continue
      warn(
          record.id, "feedback-unanswered",
          f"feedback[{index}] ({entry.kind}, {entry.author}) has no recorded "
          f"response on a closed run: {entry.text[:70]!r}. Answer it with "
          f"`rlmcp record answer {record.id} {index} \"...\"` -- or say plainly "
          "that nothing was done.",
      )
    for index, entry in enumerate(record.feedback):
      if entry.kind not in FEEDBACK_KINDS:
        warn(
            record.id, "feedback-kind",
            f"feedback[{index}] has kind '{entry.kind}', which is not one of: "
            f"{', '.join(FEEDBACK_KINDS)}. Views that group by kind will not "
            "know where to put it.",
        )
      for run_id in entry.affects:
        if run_id not in by_id and run_id not in seen_ids:
          warn(
              record.id, "feedback-reference",
              f"feedback[{index}] says it affects '{run_id}', which is not a "
              "record here.",
          )

    # Assets.
    if media_exists is not None:
      for kind, entries in (record.assets or {}).items():
        for entry in entries:
          key = entry[0] if entry else ""
          if key and not media_exists.get(key, False):
            warn(
                record.id, "missing-asset",
                f"{kind} asset '{key}' is recorded but not present.",
            )

  report.errors.extend(_cycle_problems(records, by_id))
  report.errors.extend(_lease_problems(records, slots))
  return report


def _cycle_problems(
    records: Sequence[RunRecord], by_id: Dict[str, RunRecord]
) -> List[Problem]:
  """Detect loops in the config lineage.

  Absent from the method this ports, where a cycle would hang the viewer walking
  the chain. Cheap to check, so check.
  """
  problems: List[Problem] = []
  for record in records:
    seen: List[str] = []
    current: Optional[RunRecord] = record
    while current is not None:
      if current.id in seen:
        problems.append(
            Problem(
                "error", record.id, "parent-cycle",
                f"parent chain loops: {' -> '.join(seen + [current.id])}.",
            )
        )
        break
      seen.append(current.id)
      current = by_id.get(current.parent) if current.parent else None
  return problems


def _lease_problems(records: Sequence[RunRecord], slots: int) -> List[Problem]:
  """More live claims than the machine has slots.

  With ``slots=1`` this is the original mutex: one machine, one trainer. It
  generalises without changing meaning, which the record-counting version could
  not do.
  """
  live = [r for r in records if r.lease is not None and not r.lease.dead()]
  problems: List[Problem] = []

  by_slot: Dict[str, List[str]] = {}
  for record in live:
    assert record.lease is not None
    by_slot.setdefault(record.lease.slot, []).append(record.id)
  for slot, holders in by_slot.items():
    if len(holders) > 1:
      problems.append(
          Problem(
              "error", holders[0], "slot-contention",
              f"slot '{slot}' is claimed by {holders} at once.",
          )
      )

  if len(live) > slots:
    problems.append(
        Problem(
            "error", None, "slots-oversubscribed",
            f"{len(live)} runs hold a live lease but only {slots} slot(s) are "
            f"configured: {[r.id for r in live]}.",
        )
    )
  return problems


def check_verdict_change(
    record: RunRecord, verdict: str, records: Sequence[RunRecord]
) -> Optional[str]:
  """Why ``record`` may not take ``verdict``, or None if it may.

  Used at close-out so the rules are enforced when a verdict is *set*, not
  discovered later by a whole-records check. The caller applies the close's
  outcome and measurements to ``record`` first: the rules judge the record as
  it would be persisted, which is what keeps this function and :func:`validate`
  in agreement. ``records`` is the rest of the records, for rules that need
  context beyond one record.
  """
  if verdict not in VERDICTS:
    return f"Unknown verdict '{verdict}'. Use one of: {', '.join(VERDICTS)}."
  if verdict in OPEN_VERDICTS:
    return (
        f"'{verdict}' is an open state, not a close-out. Closing takes one of: "
        f"{', '.join(TERMINAL_VERDICTS)}."
    )
  # ``interrupted`` is what the reaper leaves behind -- "a job that died is not
  # a result" -- so it alone among the terminal verdicts still owes a real
  # close-out. Everything else closed stays closed.
  reclosable = record.is_open or (
      record.verdict == "interrupted" and verdict != "interrupted"
  )
  if not reclosable:
    return (
        f"Run {record.id} is already closed: verdict '{record.verdict}'. "
        "Closing it again would duplicate its evidence rows. The files are the "
        "truth in these records -- to genuinely reopen the run, edit the "
        "verdict in its meta.json back to 'running' first."
    )
  if verdict in ("validated", "best") and record.weights is not None:
    return (
        f"'{verdict}' is not available to a run warm-started from "
        f"{record.weights.describe()}. A warm start shows the change preserves a "
        "behaviour, not that it creates one. Use 'provisional', and promote it "
        "with a from-scratch run."
    )
  missing = missing_result_evidence(record, verdict)
  if missing:
    return (
        f"'{verdict}' claims a result, and this record has no "
        f"{' and no '.join(missing)}. A claim without a measurement is not "
        "knowledge -- close with an outcome and at least one measurement "
        "(the falsifier row is not one: held is absence of disproof)."
    )
  return None
