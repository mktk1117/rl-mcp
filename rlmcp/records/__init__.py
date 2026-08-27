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

from rlmcp.records.clips import caption_for, from_record as clips_of, iteration_of
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
    RecordStore,
    MediaStore,
    SlotUnavailable,
    StoreError,
)
from rlmcp.records.validate import Problem, Report, check_verdict_change, validate

__all__ = [
    "ConflictError",
    "FEEDBACK_KINDS",
    "FileStore",
    "Falsifier",
    "Feedback",
    "RecordStore",
    "Lease",
    "MediaStore",
    "OPEN_VERDICTS",
    "OWED_A_RESPONSE",
    "Problem",
    "Report",
    "RunRecord",
    "SlotUnavailable",
    "StoreError",
    "TERMINAL_VERDICTS",
    "VERDICTS",
    "Weights",
    "ancestors",
    "caption_for",
    "clips_of",
    "iteration_of",
    "check_verdict_change",
    "children",
    "fold_recipe",
    "open_store",
    "slugify",
    "validate",
]
