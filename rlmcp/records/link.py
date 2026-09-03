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
from typing import Any

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
  except Exception:
    return ""
  task = info.get("task")
  return str(task).strip() if task else ""


class RecordLink:
  """A live run's half of the record."""

  def __init__(
      self,
      store: Any,
      record_id: str | None = None,
      slot: str = "",
      strict: bool = False,
      heartbeat_seconds: float = 60.0,
      ttl_seconds: float = 900.0,
      code_root: str | None = None,
  ):
    self.store = store
    self.record_id = record_id
    # Where the task package lives. None means "the directory the run was
    # launched from", which is true of every launch this project has made;
    # "" means do not stamp here -- and keep a stamp the record already has,
    # because a launcher that materialized the code wrote it before this
    # process existed (KEEPS_LAUNCHER_STAMP).
    self.code_root = code_root
    self.slot = slot or os.environ.get("RLMCP_SLOT") or "local"
    self.strict = strict
    self.heartbeat_seconds = heartbeat_seconds
    self.ttl_seconds = ttl_seconds
    self.record: RunRecord | None = None
    self.notes: list = []
    self._last_heartbeat = 0.0
    self._config_pending = False
    self._config_warned = False
    self._asset_warned = False

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

  def start(self, session_dir: str, config: dict[str, Any] | None = None) -> None:
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
    self.record.code = self._stamp_code()
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

  KEEPS_LAUNCHER_STAMP = True
  """``code_root=""`` keeps a stamp the record already carries.

  A launcher that materializes the code before the process exists -- a
  studio, a fleet -- stamps the record itself and then runs the trainer from
  a plain directory. From there the trainer can see nothing truer than what
  the launcher wrote, so "do not stamp" means exactly that, not "stamp
  nothing over it". A launcher checks this attribute before relying on it.
  """

  def _stamp_code(self) -> dict[str, Any]:
    """What the package looked like at launch. Never raises, never blocks."""
    if self.record is None:
      return {}
    if self.code_root == "":
      return dict(self.record.code or {})
    from rlmcp.records.snapshot import capture

    code = capture(self.code_root or Path.cwd(), record_id=self.record.id)
    if code.get("kind") == "none":
      self._warn(f"[rlmcp] no code snapshot for this run: {code.get('reason')}")
    elif not code.get("clean"):
      dirty = code.get("dirty", {})
      self._warn(
          f"[rlmcp] {code['head']['short'] if code.get('head') else 'no commit'}"
          f" + {dirty.get('added', 0)} uncommitted lines in {code.get('root')}"
          " -- recorded as the tree that actually ran."
      )
    return code

  def snapshot_config(self, config: dict[str, Any]) -> None:
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
          return None  # Anything but False: write the record.

        refreshed = self.store.update_record(self.record.id, touch_leaseless)
        if refreshed is not None:
          self.record = refreshed
    except Exception:
      pass  # A records hiccup must never stall training.

  def attach_asset(
      self, path: str, caption: str = "", kind: str = "videos"
  ) -> str | None:
    """Copy an artifact into the record's media store and list it.

    This is how evidence produced *during* a run reaches the record without
    anyone typing ``rlmcp record asset`` afterwards -- which is the moment it
    was reliably not typed. The file is copied, so a clip survives the log
    directory being cleaned up.

    Returns the media key, or ``None`` when there is no record to attach to or
    the write failed. Never raises: a records hiccup must not fail a job whose
    output already exists on disk.
    """
    if self.record is None:
      return None
    try:
      key = self.store.media.put(self.record.id, str(path), caption, kind)

      def add_asset(record: RunRecord) -> Any:
        entries = record.assets.setdefault(kind, [])
        # Idempotent: re-filing the same key (a retried write, a re-run of the
        # same iteration after a resume) must not list the clip twice.
        if any(entry and entry[0] == key for entry in entries):
          return False
        entries.append([key, caption])
        return None  # Anything but False: write the record.

      updated = self.store.update_record(self.record.id, add_asset)
      if updated is not None:
        self.record = updated
    except Exception as exc:
      if not self._asset_warned:
        self._asset_warned = True
        self._warn(
            f"[rlmcp] could not attach {Path(path).name} to record "
            f"{self.record.id} ({exc}); the file is still in the session's "
            "artifacts directory."
        )
      return None
    else:
      return key

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

  def status(self) -> dict[str, Any]:
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
      self, metrics: dict[str, float], iteration: int | None = None
  ) -> dict[str, Any]:
    """Has the run already disproved its own hypothesis?"""
    if self.record is None:
      return {"registered": False}
    return self.record.falsifier.check(metrics, iteration=iteration)

  def _warn(self, message: str) -> None:
    self.notes.append(message)
    print(message, flush=True)


def open_link(
    record_id: str | None = None,
    root: str | None = None,
    slot: str = "",
    strict: bool = False,
    slots: int = 1,
    code_root: str | None = None,
) -> RecordLink:
  """Open the records and bind a record, if one was named."""
  from rlmcp.records.filestore import open_store

  return RecordLink(open_store(root, slots=slots), record_id, slot=slot,
                    strict=strict, code_root=code_root)
