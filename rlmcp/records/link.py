"""Binding a live training run to its record.

The record is written before the run and closed after it. In between, the run
itself has three jobs: say which session holds its evidence, snapshot the config
it actually launched with, and keep its lease alive so that a machine which dies
releases its slot instead of blocking the next launch.

Pre-registration is a warning by default, not a gate. A library that refuses to
start training is a library people stop wrapping — and an unregistered run that
says so loudly on every launch is a better outcome than a disciplined one nobody
uses. ``strict=True`` turns the warning into a refusal.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from rlmcp.records.record import RunRecord
from rlmcp.records.store import StoreError, summarize_lease

UNREGISTERED_WARNING = (
    "[rlmcp] This run has no lab record: no hypothesis, no falsifier, and "
    "nothing linking its results to what you were testing.\n"
    "[rlmcp]   rlmcp record new <slug> --hypothesis ... --falsifier ...\n"
    "[rlmcp] then pass record_run=<id> to wrap()."
)


def task_from_session(session_dir: Any) -> str:
  """The task id the live session says it is training, or ``""``.

  Read straight out of ``session.json`` rather than through
  :class:`rlmcp.session.Session`, because this runs at wrap time on the
  training process's critical path and a half-written or absent session file
  must cost the field, never the run.
  """
  try:
    info = json.loads((Path(str(session_dir)) / "session.json").read_text())
  except Exception:  # noqa: BLE001 -- no session fact is worth stalling a launch.
    return ""
  task = info.get("task")
  return str(task).strip() if task else ""


class RecordLink:
  """A live run's half of the record."""

  def __init__(
      self,
      store: Any,
      record_id: Optional[str] = None,
      slot: str = "",
      strict: bool = False,
      heartbeat_seconds: float = 60.0,
      ttl_seconds: float = 900.0,
  ):
    self.store = store
    self.record_id = record_id
    self.slot = slot or os.environ.get("RLMCP_SLOT") or "local"
    self.strict = strict
    self.heartbeat_seconds = heartbeat_seconds
    self.ttl_seconds = ttl_seconds
    self.record: Optional[RunRecord] = None
    self.notes: list = []
    self._last_heartbeat = 0.0
    self._config_pending = False
    self._config_warned = False

    if record_id:
      self.record = store.get_record(record_id)
      if self.record is None:
        message = (
            f"No lab record '{record_id}'. Create it first with "
            f"`rlmcp record new <slug>`, or pass record_run=None."
        )
        if strict:
          raise StoreError(message)
        self._warn(message)
    elif strict:
      raise StoreError(
          "record_strict=True but no record_run was given. Pre-register the run with "
          "`rlmcp record new <slug> --hypothesis ... --falsifier ...` first."
      )
    else:
      self._warn(UNREGISTERED_WARNING)

  # Lifecycle.

  def start(self, session_dir: str, config: Optional[Dict[str, Any]] = None) -> None:
    """Attach the session and take a slot. Called once, at wrap time."""
    if self.record is None:
      return
    self.record.session = str(session_dir)
    # The session already knows what it is training, and a task that only a
    # flag can set is empty on most records -- which is the same as not having
    # the field at all. An explicit `record new --task` wins; this fills the gap.
    if not self.record.task:
      self.record.task = task_from_session(session_dir)
    if config:
      self.record.config = dict(config)
    self._config_pending = True
    if self.record.verdict == "planned":
      self.record.verdict = "running"
    try:
      self.store.put_record(self.record)
    except Exception as exc:
      # A records hiccup must never stall training — not even at launch.
      if self.strict:
        raise
      self._warn(
          f"[rlmcp] could not write lab record {self.record.id} at start "
          f"({exc}); this run will proceed UNRECORDED. Close it out by hand "
          "when it finishes."
      )
      self.record = None
      return

    try:
      self.record = self.store.claim(
          self.record.id, slot=self.slot, holder=f"pid-{os.getpid()}",
          ttl_seconds=self.ttl_seconds,
      )
      self._last_heartbeat = time.time()
    except StoreError as exc:
      # A busy slot is a real signal — usually another trainer is already on
      # this GPU — but it must not kill a run the user explicitly launched.
      if self.strict:
        raise
      self._warn(f"[rlmcp] could not claim slot '{self.slot}': {exc}")
      self._warn(
          f"[rlmcp] run {self.record.id} is running without a lease; it will "
          "be reaped by heartbeat staleness if the process dies."
      )

  def snapshot_config(self, config: Dict[str, Any]) -> None:
    """Record the resolved config, once everything has contributed to it.

    Taken at the first iteration rather than at wrap time: the runner attaches
    after the environment, so a snapshot taken earlier is missing every PPO
    hyperparameter and is not the config the run actually used.
    """
    if self.record is None or not self._config_pending or not config:
      return
    try:
      def set_config(record: RunRecord) -> None:
        record.config = dict(config)

      # The store's read-modify-write helper: a heartbeat or reaper write in
      # between is retried through, not surfaced as a conflict.
      updated = self.store.update_record(self.record.id, set_config)
      if updated is not None:
        self.record = updated
      self._config_pending = False
    except Exception as exc:
      # Keep the snapshot pending so the next heartbeat retries it: losing the
      # launch config to one failed write would leave config={} forever.
      if not self._config_warned:
        self._config_warned = True
        self._warn(
            f"[rlmcp] could not snapshot the launch config ({exc}); "
            "will keep retrying on the heartbeat."
        )

  def heartbeat(self) -> None:
    """Renew the lease, at most once per ``heartbeat_seconds``."""
    if self.record is None:
      return
    now = time.time()
    if now - self._last_heartbeat < self.heartbeat_seconds:
      return
    self._last_heartbeat = now
    try:
      renewed = self.store.heartbeat(self.record.id)
      if renewed is not None:
        self.record = renewed
      else:
        # No lease to renew — this run is on the leaseless path a failed claim
        # leaves behind. Rewrite the record anyway: its freshness is the only
        # live signal the reaper has for a process that dies out here. Via the
        # retry helper, with the condition re-checked under its fresh read.
        def touch_leaseless(record: RunRecord) -> Any:
          if record.lease is not None or record.verdict != "running":
            return False
          record.touch()

        refreshed = self.store.update_record(self.record.id, touch_leaseless)
        if refreshed is not None:
          self.record = refreshed
    except Exception:
      pass  # A records hiccup must never stall training.

  def finish(self, reason: str = "") -> None:
    """Release the slot. The verdict stays for an explicit close-out."""
    if self.record is None:
      return
    try:
      self.store.release(self.record.id)
      if reason:
        # The retry helper, so a reaper interleaving between the release and
        # this write conflicts a retried attempt instead of dropping the exit
        # reason the way a bare get-then-put used to.
        def set_exit(record: RunRecord) -> None:
          record.links["exit"] = reason

        self.store.update_record(self.record.id, set_exit)
    except Exception:
      pass

  # Reporting.

  def status(self) -> Dict[str, Any]:
    """The lab's contribution to the live status payload."""
    if self.record is None:
      return {"registered": False, "warning": "run has no lab record"}
    return {
        "registered": True,
        "id": self.record.id,
        "slug": self.record.slug,
        "verdict": self.record.verdict,
        "hypothesis": self.record.hypothesis,
        "falsifier": self.record.falsifier.prose,
        "parent": self.record.parent,
        "warm_start": self.record.weights.describe() if self.record.weights else None,
        "lease": summarize_lease(self.record),
    }

  def check_falsifier(
      self, metrics: Dict[str, float], iteration: Optional[int] = None
  ) -> Dict[str, Any]:
    """Has the run already disproved its own hypothesis?"""
    if self.record is None:
      return {"registered": False}
    return self.record.falsifier.check(metrics, iteration=iteration)

  def _warn(self, message: str) -> None:
    self.notes.append(message)
    print(message, flush=True)


def open_link(
    record_id: Optional[str] = None,
    root: Optional[str] = None,
    slot: str = "",
    strict: bool = False,
    slots: int = 1,
) -> RecordLink:
  """Open the records and bind a record, if one was named."""
  from rlmcp.records.filestore import open_store

  return RecordLink(open_store(root, slots=slots), record_id, slot=slot, strict=strict)
