"""What a record store backend has to do.

One protocol, so the thing that changes when this grows from one GPU to a
hundred is the transport, not the schema. Everything else in :mod:`rlmcp.records`
talks to :class:`RecordStore` and never touches a filesystem directly.

Today there is one implementation, :class:`~rlmcp.records.filestore.FileStore`:
records are files, an SQLite index makes them queryable, and the index can be
thrown away and rebuilt. Later there will be a second one that speaks HTTP to a
service holding Postgres and an object store. The test suite for this protocol
is written once and run against both -- that is what proves the seam is real.

Two rules the protocol exists to enforce:

* **The store assigns ids.** Two jobs on two machines cannot both pick ``035``.
* **Claims are leases.** A job that dies releases its slot, rather than blocking
  the next launch forever the way a stale ``running`` record does.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

from rlmcp.records.record import RunRecord


class StoreError(RuntimeError):
  """A store refused an operation."""


class SlotUnavailable(StoreError):
  """Every configured slot is held by a live lease."""


class ConflictError(StoreError):
  """A write lost its compare-and-swap: the record changed underneath it.

  The caller's copy was at ``expected`` while the store already held ``found``.
  Overwriting silently would discard whoever wrote ``found``, so the truthful
  move is to raise; re-read, re-apply, and write again -- which is exactly what
  :meth:`RecordStore.update_record` does on the caller's behalf.
  """

  def __init__(self, record_id: str, expected: int, found: int):
    super().__init__(
        f"Record '{record_id}' changed underneath this write: "
        f"held version {expected}, the store has {found}. Re-read and retry."
    )
    self.record_id = record_id
    self.expected = expected
    self.found = found


class MediaStore(Protocol):
  """Where blobs live. Paths in, keys out; never blobs in the record itself."""

  def put(self, run_id: str, path: str, caption: str = "", kind: str = "plots") -> str:
    """Copy a file into the store and return the key to record."""

  def get(self, key: str) -> Optional[str]:
    """A locally readable path for ``key``, or None if it is gone."""

  def exists(self, key: str) -> bool:
    ...


class RecordStore(Protocol):
  """The records, however it is stored."""

  # Records.

  def new_record(self, slug: str, **fields: Any) -> RunRecord:
    """Create a record with a store-assigned id and seq, and persist it."""

  def put_record(self, record: RunRecord) -> RunRecord:
    """Persist an existing record, or raise :class:`ConflictError`.

    The write is a compare-and-swap on ``record.version``: it succeeds only if
    the store still holds the version this copy was read at, and bumps the
    version on the way through. Hold a record across a long gap and someone
    else's write in between raises rather than being silently discarded.
    """

  def update_record(
      self,
      record_id: str,
      mutate: Callable[[RunRecord], Any],
      retries: int = 3,
  ) -> Optional[RunRecord]:
    """Re-read, apply ``mutate``, and persist -- retrying on conflict.

    The read-modify-write loop every mutating caller needs, so none of them
    write it wrong. ``mutate`` edits the record in place; returning ``False``
    aborts without writing (for "check again under the fresh read" callers),
    and the helper answers a conflict by re-reading and re-applying, up to
    ``retries`` more times before letting :class:`ConflictError` out.
    """

  def get_record(self, record_id: str) -> Optional[RunRecord]:
    ...

  def list_records(self) -> List[RunRecord]:
    """Every record, ordered by ``seq``."""

  def query(
      self,
      stage: Optional[str] = None,
      verdict: Optional[str] = None,
      parent: Optional[str] = None,
      proposed_by: Optional[str] = None,
      text: Optional[str] = None,
      limit: Optional[int] = None,
  ) -> List[RunRecord]:
    """Filtered listing. ``text`` matches slug, hypothesis and outcome."""

  def delete_record(self, record_id: str) -> bool:
    ...

  # Leases.

  def claim(
      self,
      record_id: str,
      slot: str,
      holder: str = "",
      ttl_seconds: float = 900.0,
      owner: Optional[str] = None,
  ) -> RunRecord:
    """Take ``slot`` for this record, or raise :class:`SlotUnavailable`.

    ``owner`` is ``"runner"`` for a claim made by the training process itself
    (pid-death frees the slot) or ``"manual"`` for a reservation made from a
    short-lived process (TTL is the only authority). ``None`` infers it: a
    claim that names a ``holder`` is the runner path, a bare claim is manual.
    """

  def heartbeat(self, record_id: str) -> Optional[RunRecord]:
    """Renew the record's lease. Returns None if it holds none."""

  def release(self, record_id: str) -> Optional[RunRecord]:
    ...

  def reap_expired(self) -> List[RunRecord]:
    """Move records whose job has died to ``interrupted``.

    That covers a dead lease, and a ``running`` record with no lease at all
    once nothing has touched it for longer than a TTL -- the shape a run
    leaves behind when it started under a warning and then died without
    closing. One verdict for every death, because ``interrupted`` is the one
    terminal state that can still be closed with real evidence afterwards;
    what the reaper actually saw is recorded in ``links["reaped"]``.
    """

  # Media.

  @property
  def media(self) -> MediaStore:
    ...


def next_display_id(existing: Iterable[str], width: int = 3) -> str:
  """The next zero-padded numeric id after the ones already present.

  Ids stay short and human-quotable (``007``) because they are cited constantly
  -- in skill files, in commit messages, in conversation. Non-numeric ids are
  ignored rather than rejected, so a store can hold imported records like ``R5``
  alongside assigned ones.
  """
  highest = 0
  for candidate in existing:
    if candidate.isdigit():
      highest = max(highest, int(candidate))
  return str(highest + 1).zfill(width)


def summarize_lease(record: RunRecord, now: Optional[float] = None) -> Dict[str, Any]:
  """Human-readable lease state for a status payload."""
  if record.lease is None:
    return {"held": False}
  now = now or time.time()
  return {
      "held": True,
      "slot": record.lease.slot,
      "holder": record.lease.holder,
      "owner": record.lease.owner,
      "age_s": round(record.lease.age_seconds(now), 1),
      "expired": record.lease.expired(now),
      "ttl_s": record.lease.ttl_seconds,
  }


def records_by_id(records: Sequence[RunRecord]) -> Dict[str, RunRecord]:
  return {r.id: r for r in records}
