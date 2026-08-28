"""Run records: what was tried, what it changed, and what killed it.

A training run that leaves no record is an island. This package is the memory --
records with a hypothesis and a falsifier written before the run and an outcome
written after, linked into a graph whose edges say *config* or *warm start*,
and validated by rules that refuse the claims a warm start cannot support.

Stdlib only, so the whole layer works in an interpreter with no simulator:

    from rlmcp.records import open_store
    store = open_store("./records")
    run = store.new_record("faster_commands", hypothesis="...", ...)
"""

from rlmcp.records.clips import caption_for, iteration_of
from rlmcp.records.clips import from_record as clips_of
from rlmcp.records.filestore import FileStore, open_store
from rlmcp.records.record import (
    FEEDBACK_KINDS,
    OPEN_VERDICTS,
    OWED_A_RESPONSE,
    TERMINAL_VERDICTS,
    VERDICTS,
    Falsifier,
    Feedback,
    Lease,
    RunRecord,
    Weights,
    ancestors,
    children,
    fold_recipe,
    slugify,
)
from rlmcp.records.store import (
    ConflictError,
    MediaStore,
    RecordStore,
    SlotUnavailable,
    StoreError,
)
from rlmcp.records.validate import Problem, Report, check_verdict_change, validate

__all__ = [
    "FEEDBACK_KINDS",
    "OPEN_VERDICTS",
    "OWED_A_RESPONSE",
    "TERMINAL_VERDICTS",
    "VERDICTS",
    "ConflictError",
    "Falsifier",
    "Feedback",
    "FileStore",
    "Lease",
    "MediaStore",
    "Problem",
    "RecordStore",
    "Report",
    "RunRecord",
    "SlotUnavailable",
    "StoreError",
    "Weights",
    "ancestors",
    "caption_for",
    "check_verdict_change",
    "children",
    "clips_of",
    "fold_recipe",
    "iteration_of",
    "open_store",
    "slugify",
    "validate",
]
