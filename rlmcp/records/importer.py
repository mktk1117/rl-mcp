"""Reading someone else's records in.

rlmcp's record schema is deliberately the one from agentic-mjlab, so an
existing records can be read without transcription: ``runs/NNN-slug/meta.json``
and the flat ``runs/legacy/<ID>.json`` files both load as they are.

Two differences are handled here rather than silently dropped:

* their ``falsifier`` lives in ``PLAN.md`` prose, not in the record, so it is
  read out of the plan when one sits next to the record;
* ids are preserved exactly. They are cited by hand across skill files, commit
  messages and conversation, so renumbering them on import would break every
  reference in the surrounding prose.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from rlmcp.records.record import Falsifier, RunRecord

_FALSIFIER_SECTION = re.compile(
    r"^##\s*Falsifier.*?$(.*?)(?=^##\s|\Z)", re.MULTILINE | re.DOTALL
)
_PREDICTION_SECTION = re.compile(
    r"^##\s*Prediction.*?$(.*?)(?=^##\s|\Z)", re.MULTILINE | re.DOTALL
)


def _section(text: str, pattern: re.Pattern) -> str:
  match = pattern.search(text or "")
  if not match:
    return ""
  body = " ".join(line.strip() for line in match.group(1).strip().splitlines())
  body = re.sub(r"\s+", " ", body).strip(" -*")
  return body[:800]


def read_record(path: Path) -> Optional[RunRecord]:
  """Load one ``meta.json``, enriching it from a sibling ``PLAN.md``."""
  try:
    payload = json.loads(path.read_text())
  except (json.JSONDecodeError, OSError):
    return None
  if not isinstance(payload, dict) or "id" not in payload:
    return None

  record = RunRecord.from_dict(payload)

  plan = path.parent / "PLAN.md"
  if plan.exists():
    text = plan.read_text(errors="ignore")
    if record.falsifier.is_empty():
      prose = _section(text, _FALSIFIER_SECTION)
      if prose:
        record.falsifier = Falsifier(prose=prose)
    if not record.prediction:
      record.prediction = _section(text, _PREDICTION_SECTION)
    record.links.setdefault("plan", str(plan))

  report = path.parent / "REPORT.md"
  if report.exists():
    record.links.setdefault("report", str(report))
  return record


def find_records(source: Path) -> List[Path]:
  """Every record file under ``source``, in both layouts."""
  return sorted(source.glob("*/meta.json")) + sorted(source.glob("legacy/*.json"))


def import_records(
    store: Any, source: str | Path, dry_run: bool = False
) -> Dict[str, Any]:
  """Copy records from another store into this one, ids intact."""
  source = Path(source).expanduser().resolve()
  if not source.exists():
    return {"ok": False, "error": f"No such directory: {source}"}

  paths = find_records(source)
  if not paths:
    return {"ok": False, "error": f"No meta.json or legacy/*.json under {source}"}

  existing = {r.id for r in store.list_records()}
  imported, skipped, failed = [], [], []

  for path in paths:
    record = read_record(path)
    if record is None:
      failed.append(str(path.relative_to(source)))
      continue
    if record.id in existing:
      skipped.append(record.id)
      continue
    if not dry_run:
      store.put_record(record)
      existing.add(record.id)
    imported.append(record.id)

  return {
      "ok": True,
      "source": str(source),
      "dry_run": dry_run,
      "imported": len(imported),
      "ids": imported,
      "skipped_existing": skipped,
      "unreadable": failed,
  }
