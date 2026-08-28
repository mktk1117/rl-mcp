"""Records as files, SQLite as an index you can throw away.

The files are the truth. That is what keeps a record diffable, reviewable in a
pull request, and carried by the commit that produced it -- the property that
makes a record worth keeping rather than merely worth writing.

The SQLite file exists only so that "every run where the slip penalty went below
-0.2" is a query rather than a directory walk. It is derived state: delete it,
call :meth:`FileStore.reindex`, lose nothing. Nothing is ever read from the index
that is not also in a file.

The one thing SQLite additionally provides is a lock. Assigning an id, taking a
lease, and every record write happen inside a ``BEGIN IMMEDIATE`` transaction on
the index database, so two processes cannot both allocate ``007`` or both win
the same GPU. The database *serialises* those decisions; the files still *are*
them. An id is a citation, and a citation must never silently change referent,
so deleting a record appends its id to ``runs/.tombstones`` -- an append-only
file the allocator never goes below. The ``alloc`` counters in the index carry
the same memory but share the index's fate: throw both away, and the tombstone
file still holds the line.

Layout::

    <root>/
      runs/<id>-<slug>/
        meta.json      the record
        PLAN.md        hypothesis, prediction, falsifier, change
        REPORT.md      outcome, evidence, diagnosis, belief update, next plan
      media/<id>/...   assets promoted out of the session's artifacts
      index.sqlite     derived; rebuildable
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import shutil
import sqlite3
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from rlmcp.records.record import FEEDBACK_KINDS, Feedback, Lease, RunRecord, slugify
from rlmcp.records.store import ConflictError, MediaStore, SlotUnavailable, StoreError

INDEX_NAME = "index.sqlite"

TOMBSTONES_NAME = ".tombstones"
"""Append-only file under ``runs/``, one deleted id per line.

The durable half of the never-reuse guarantee: the ``alloc`` counters carry the
same memory faster, but they live in the index, and the index is documented as
disposable. A file is what survives "delete index.sqlite and reindex"."""

STALE_RUNNING_TTL = 900.0
"""How long a leaseless ``running`` record may go untouched before it is
reaped to ``interrupted``. The same default as a lease TTL: past it, silence
is evidence."""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  seq INTEGER,
  slug TEXT,
  stage TEXT,
  date TEXT,
  verdict TEXT,
  parent TEXT,
  weights_run TEXT,
  prior TEXT,
  proposed_by TEXT,
  hypothesis TEXT,
  outcome TEXT,
  session TEXT,
  tags TEXT,
  updated_at REAL,
  path TEXT
);
CREATE INDEX IF NOT EXISTS runs_seq ON runs(seq);
CREATE INDEX IF NOT EXISTS runs_verdict ON runs(verdict);
CREATE INDEX IF NOT EXISTS runs_parent ON runs(parent);
CREATE TABLE IF NOT EXISTS feedback (
  run_id TEXT,
  idx INTEGER,
  at REAL,
  kind TEXT,
  author TEXT,
  iteration INTEGER,
  answered INTEGER,
  outstanding INTEGER,
  changed INTEGER,
  text TEXT,
  PRIMARY KEY (run_id, idx)
);
CREATE INDEX IF NOT EXISTS feedback_at ON feedback(at);
CREATE INDEX IF NOT EXISTS feedback_kind ON feedback(kind);
CREATE TABLE IF NOT EXISTS alloc (
  kind TEXT PRIMARY KEY,
  next INTEGER NOT NULL
);
"""

_COMPONENT_MAX = 120
_COMPONENT_UNSAFE = frozenset("/\\\x00*?[]")


def _warn(message: str) -> None:
  print(f"[rlmcp] {message}", file=sys.stderr, flush=True)


def _safe_component(value: Any, what: str) -> str:
  """One plain path component, or a refusal.

  Every filesystem join in this store passes through here: record ids, slugs,
  document names, media kinds. Ids arrive from ``meta.json`` files and CLI
  arguments as well as from the allocator, and a separator or ``..`` in one
  must never become a path outside the lab root.
  """
  text = str(value)
  if (not text or text in (".", "..") or text.startswith(".")
      or len(text) > _COMPONENT_MAX
      or any(c in _COMPONENT_UNSAFE for c in text)):
    raise StoreError(f"Unsafe {what} {text!r}: one plain path component only.")
  return text


def _pid_alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except OSError:
    return True
  return True


def _fsync_dir(directory: Path) -> None:
  """Make a rename or append in ``directory`` survive a power cut."""
  try:
    fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
  except OSError:  # pragma: no cover - platforms without directory opens.
    return
  try:
    os.fsync(fd)
  finally:
    os.close(fd)


def _atomic_write(path: Path, text: str) -> None:
  """Replace ``path`` so that a power cut leaves either version, never neither."""
  tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
  with open(tmp, "w", encoding="utf-8") as handle:
    handle.write(text)
    handle.flush()
    os.fsync(handle.fileno())
  os.replace(tmp, path)
  # The rename itself must reach the disk, or the file can come back empty.
  _fsync_dir(path.parent)


class FileMediaStore(MediaStore):
  """Assets copied out of a session, so a record survives log cleanup."""

  def __init__(self, root: Path):
    self.root = Path(root)

  def put(self, run_id: str, path: str, caption: str = "", kind: str = "plots") -> str:
    source = Path(path)
    if not source.exists():
      raise StoreError(f"No such file to record as an asset: {source}")
    destination = (
        self.root
        / _safe_component(run_id, "record id")
        / _safe_component(kind, "media kind")
    )
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    if source.resolve() != target.resolve():
      shutil.copy2(source, target)
    return str(target.relative_to(self.root))

  def _resolve(self, key: str) -> Path | None:
    """The path for ``key``, or None when the key points outside the store."""
    root = self.root.resolve()
    candidate = (root / key).resolve()
    return candidate if candidate.is_relative_to(root) else None

  def get(self, key: str) -> str | None:
    candidate = self._resolve(key)
    return str(candidate) if candidate is not None and candidate.exists() else None

  def exists(self, key: str) -> bool:
    candidate = self._resolve(key)
    return candidate is not None and candidate.exists()


class FileStore:
  """A records on disk."""

  def __init__(self, root: Path | str, slots: int = 1):
    self.root = Path(root).expanduser().resolve()
    self.runs_dir = self.root / "runs"
    self.media_root = self.root / "media"
    self.index_path = self.root / INDEX_NAME
    self.slots = max(1, int(slots))
    self.runs_dir.mkdir(parents=True, exist_ok=True)
    self.media_root.mkdir(parents=True, exist_ok=True)
    self._media = FileMediaStore(self.media_root)
    self._warned: set = set()
    self._sweep_stale_tmp()
    self._init_index()

  # Media.

  @property
  def media(self) -> FileMediaStore:
    return self._media

  # Paths.

  def record_dir(self, record: RunRecord) -> Path:
    rid = _safe_component(record.id, "record id")
    slug = _safe_component(record.slug, "slug")
    return self.runs_dir / f"{rid}-{slug}"

  def _find_dir(self, record_id: str) -> Path | None:
    rid = _safe_component(record_id, "record id")
    matches = sorted(self.runs_dir.glob(f"{rid}-*"))
    if len(matches) > 1:
      _warn(
          f"record id {rid} matches {len(matches)} directories "
          f"({', '.join(m.name for m in matches)}); using {matches[0].name}"
      )
    return matches[0] if matches else None

  def _sweep_stale_tmp(self) -> None:
    """Remove temp files whose writer died mid-write. Opportunistic, on open."""
    for stale in self.runs_dir.rglob(".*.tmp"):
      parts = stale.name.split(".")
      try:
        pid = int(parts[-3])
      except (IndexError, ValueError):
        continue
      if _pid_alive(pid):
        continue  # A live writer on this host; its os.replace is coming.
      try:
        stale.unlink()
        _warn(f"swept a temp file left by a crashed write: {stale}")
      except OSError:
        pass

  # Index.

  def _connect(self) -> sqlite3.Connection:
    conn = sqlite3.connect(self.index_path, timeout=5.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    return conn

  def _init_index(self) -> None:
    """Create the index if it is missing, and catch it up if it is old.

    A store written by a build that predates a table opens here with the rest
    of its index intact, so the new table would sit empty and every query
    against it would answer "nothing" -- which reads as a fact rather than as
    a missing migration. Detecting the gap and rebuilding once is cheap; the
    files are the truth, so a rebuild can only agree with them.
    """
    with contextlib.closing(self._connect()) as conn:
      conn.execute("PRAGMA journal_mode=WAL")
      stale = bool(
          conn.execute(
              "SELECT 1 FROM sqlite_master WHERE type='table' AND name='runs'"
          ).fetchone()
          and not conn.execute(
              "SELECT 1 FROM sqlite_master WHERE type='table' AND name='feedback'"
          ).fetchone()
      )
      conn.executescript(_SCHEMA)
    if stale:
      self.reindex()

  @contextlib.contextmanager
  def _tx(self) -> Iterator[sqlite3.Connection]:
    """The store's write lock: one mutator at a time, across processes.

    ``BEGIN IMMEDIATE`` takes the database's single writer slot up front, so
    everything inside the block -- reading the files, deciding, writing a file,
    updating the index -- happens with every other mutator queued behind
    ``busy_timeout``. The decisions serialise; the files stay the truth.
    """
    conn = self._connect()
    try:
      conn.execute("BEGIN IMMEDIATE")
      try:
        yield conn
        conn.commit()
      except BaseException:
        conn.rollback()
        raise
    finally:
      conn.close()

  def _index(self, record: RunRecord, path: Path, conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            record.id, record.seq, record.slug, record.stage, record.date,
            record.verdict, record.parent,
            record.weights.run if record.weights else None,
            record.prior, record.proposed_by, record.hypothesis, record.outcome,
            record.session, json.dumps(record.tags), record.updated_at, str(path),
        ),
    )
    # Feedback is append-only in the file, but a response can be filled in
    # later, so the rows are replaced wholesale rather than appended to. The
    # file stays the truth; this is only what makes the timeline a query.
    conn.execute("DELETE FROM feedback WHERE run_id = ?", (record.id,))
    conn.executemany(
        "INSERT INTO feedback VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            (record.id, i, entry.at, entry.kind, entry.author, entry.iteration,
             int(entry.answered), int(entry.outstanding), int(entry.changed),
             entry.text)
            for i, entry in enumerate(record.feedback)
        ],
    )

  def reindex(self) -> int:
    """Rebuild the index from the files. The files always win.

    The ``alloc`` counters are left alone: they only ever move forward. Losing
    them with the rest of the index costs nothing -- deleted ids stay reserved
    because ``runs/.tombstones`` is a file, and the files are the truth.
    """
    count = 0
    with self._tx() as conn:
      conn.execute("DELETE FROM runs")
      conn.execute("DELETE FROM feedback")
      for meta in sorted(self.runs_dir.glob("*/meta.json")):
        record = self._read_meta(meta)
        if record is None:
          continue
        self._index(record, meta.parent, conn)
        count += 1
    return count

  # Allocation.

  def _counter(self, conn: sqlite3.Connection, kind: str) -> int:
    row = conn.execute("SELECT next FROM alloc WHERE kind = ?", (kind,)).fetchone()
    return int(row["next"]) if row else 1

  def _advance(self, conn: sqlite3.Connection, kind: str, value: int) -> None:
    conn.execute(
        "INSERT INTO alloc (kind, next) VALUES (?, ?) "
        "ON CONFLICT(kind) DO UPDATE SET next = MAX(next, excluded.next)",
        (kind, value),
    )

  def _tombstone(self, record_id: str) -> None:
    """Reserve ``record_id`` forever, durably, before its record goes."""
    path = self.runs_dir / TOMBSTONES_NAME
    with open(path, "a", encoding="utf-8") as handle:
      handle.write(f"{record_id}\n")
      handle.flush()
      os.fsync(handle.fileno())
    _fsync_dir(path.parent)

  def _tombstone_floor(self) -> int:
    floor = 0
    try:
      lines = (self.runs_dir / TOMBSTONES_NAME).read_text().splitlines()
    except FileNotFoundError:
      return 0
    for line in lines:
      candidate = line.strip()
      if candidate.isdigit():
        floor = max(floor, int(candidate))
    return floor

  def _allocate(self, conn: sqlite3.Connection) -> tuple[str, int]:
    """The next id and seq, under the write lock.

    The id clears three floors: the counter, the run directories (which is
    what lets a fresh or rebuilt index pick up where an old lab left off, and
    covers corrupt records whose directory still squats on its id), and the
    tombstone file, which remembers deleted ids even after the index -- and
    its counters -- are thrown away and rebuilt.
    """
    id_floor = 0
    for entry in self.runs_dir.iterdir():
      if not entry.is_dir():
        continue
      head = entry.name.split("-", 1)[0]
      if head.isdigit():
        id_floor = max(id_floor, int(head))
    seq_floor = max((r.seq for r in self.list_records()), default=0)
    next_id = max(self._counter(conn, "id"), id_floor + 1, self._tombstone_floor() + 1)
    next_seq = max(self._counter(conn, "seq"), seq_floor + 1)
    self._advance(conn, "id", next_id + 1)
    self._advance(conn, "seq", next_seq + 1)
    return str(next_id).zfill(3), next_seq

  # Records.

  def new_record(self, slug: str, **fields: Any) -> RunRecord:
    fields = copy.deepcopy(fields)  # The record owns its data from here on.
    with self._tx() as conn:
      record_id, seq = self._allocate(conn)
      record = RunRecord(id=record_id, slug=slugify(slug), seq=seq, **fields)
      return self._persist(record, conn)

  def put_record(self, record: RunRecord) -> RunRecord:
    with self._tx() as conn:
      return self._persist(record, conn)

  def _persist(self, record: RunRecord, conn: sqlite3.Connection) -> RunRecord:
    """Compare-and-swap write of one record, under the write lock."""
    record_id = _safe_component(record.id, "record id")
    _safe_component(record.slug, "slug")
    directory = self._find_dir(record_id) or self.record_dir(record)
    meta = directory / "meta.json"
    if meta.exists():
      try:
        found = int(json.loads(meta.read_text()).get("version", 0))
      except (KeyError, TypeError, ValueError) as exc:
        raise StoreError(
            f"Refusing to overwrite {meta}: the file there does not parse. "
            "Repair or remove it; its id stays reserved either way."
        ) from exc
      if found != record.version:
        raise ConflictError(record.id, record.version, found)
    # The record owns its mutable fields once written: a caller mutating the
    # dict it passed in must not silently diverge the record from its file.
    record.config = copy.deepcopy(record.config)
    record.links = dict(record.links)
    record.assets = copy.deepcopy(record.assets)
    record.tags = list(record.tags)
    record.change = list(record.change)
    record.metrics = [list(pair) for pair in record.metrics]
    record.feedback = list(record.feedback)
    record.version += 1
    record.touch()
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_write(meta, json.dumps(record.to_dict(), indent=2))
    self._index(record, directory, conn)
    return record

  def update_record(
      self,
      record_id: str,
      mutate: Callable[[RunRecord], Any],
      retries: int = 3,
  ) -> RunRecord | None:
    """Re-read, apply ``mutate``, persist; retry when a writer got in between.

    ``mutate`` edits the record in place; returning ``False`` aborts without
    writing. After ``retries`` conflicted attempts beyond the first, the
    :class:`ConflictError` is allowed out -- at that point the record is
    genuinely contended and the caller should know.
    """
    conflict: ConflictError | None = None
    for _ in range(max(0, int(retries)) + 1):
      record = self.get_record(record_id)
      if record is None:
        raise StoreError(f"No record '{record_id}'.")
      if mutate(record) is False:
        return None
      try:
        return self.put_record(record)
      except ConflictError as exc:
        conflict = exc
    raise conflict  # type: ignore[misc]  # At least one attempt always ran.

  def _read_meta(self, path: Path) -> RunRecord | None:
    """Parse one record file, or warn once and skip it.

    ``json.JSONDecodeError`` is a ``ValueError``; ``TypeError`` and
    ``ValueError`` are what the ``from_dict`` coercions raise on a field like
    ``"seq": "seven"``. One broken file must never brick the whole records.
    """
    try:
      return RunRecord.from_dict(json.loads(path.read_text()))
    except (KeyError, TypeError, ValueError) as exc:
      try:
        stamp = path.stat().st_mtime
      except OSError:
        stamp = 0.0
      key = (str(path), stamp)
      if key not in self._warned:
        self._warned.add(key)
        _warn(f"skipping corrupt record {path}: {exc}")
      return None

  def get_record(self, record_id: str) -> RunRecord | None:
    directory = self._find_dir(record_id)
    if directory is None:
      return None
    meta = directory / "meta.json"
    if not meta.exists():
      return None
    return self._read_meta(meta)

  def list_records(self) -> list[RunRecord]:
    records = []
    for meta in self.runs_dir.glob("*/meta.json"):
      record = self._read_meta(meta)
      if record is not None:
        records.append(record)
    return sorted(records, key=lambda r: (r.seq, r.id))

  def query(
      self,
      stage: str | None = None,
      verdict: str | None = None,
      parent: str | None = None,
      proposed_by: str | None = None,
      text: str | None = None,
      limit: int | None = None,
  ) -> list[RunRecord]:
    clauses, params = [], []
    for column, value in (
        ("stage", stage), ("verdict", verdict),
        ("parent", parent), ("proposed_by", proposed_by),
    ):
      if value is not None:
        clauses.append(f"{column} = ?")
        params.append(value)
    if text:
      clauses.append("(slug LIKE ? OR hypothesis LIKE ? OR outcome LIKE ?)")
      params.extend([f"%{text}%"] * 3)
    sql = "SELECT id FROM runs"
    if clauses:
      sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY seq"
    if limit:
      sql += f" LIMIT {int(limit)}"

    with contextlib.closing(self._connect()) as conn:
      ids = [row["id"] for row in conn.execute(sql, params)]
    # The index answers *which*; the files answer *what*.
    found = [self.get_record(rid) for rid in ids]
    return [r for r in found if r is not None]

  def delete_record(self, record_id: str) -> bool:
    directory = self._find_dir(record_id)
    if directory is None:
      return False
    with self._tx() as conn:
      # Tombstone first: a crash mid-delete leaves the id reserved twice over
      # (the directory still floors it too), never reserved zero times.
      self._tombstone(record_id)
      shutil.rmtree(directory, ignore_errors=True)
      # Ids are never reissued, so the dead run's media cannot be inherited --
      # but it should not sit there unreferenced either.
      media_dir = self.media_root / _safe_component(record_id, "record id")
      if media_dir.exists():
        shutil.rmtree(media_dir, ignore_errors=True)
      conn.execute("DELETE FROM runs WHERE id = ?", (record_id,))
      conn.execute("DELETE FROM feedback WHERE run_id = ?", (record_id,))
    return True

  # Feedback.

  def add_feedback(self, record_id: str, feedback: Feedback) -> RunRecord:
    """Append one remark, through the retry helper.

    Feedback is the write most likely to collide with another: a human types it
    while the trainer is heartbeating its lease. Going through
    :meth:`update_record` means the append is re-applied to the fresh record on
    a lost compare-and-swap rather than failing, and it is an append every time
    -- never a write of a stale list that would drop whatever landed in between.
    """
    if not feedback.text.strip():
      raise StoreError("Feedback with no text: there is nothing to record.")
    if not isinstance(feedback.kind, str) or feedback.kind not in FEEDBACK_KINDS:
      # Refused here rather than at the index, which is where it used to fail:
      # by then meta.json had already taken the entry, so the caller was told
      # the write failed while the record kept it -- and every subsequent write
      # to that record failed the same way, across restarts, with no way back
      # through this API. One malformed field destroyed a run's ledger.
      raise StoreError(
          f"Feedback kind {feedback.kind!r} is not one of "
          f"{', '.join(FEEDBACK_KINDS)}.")

    def append(fresh: RunRecord) -> None:
      fresh.feedback.append(feedback)

    updated = self.update_record(record_id, append)
    if updated is None:
      raise StoreError(f"No record '{record_id}' to attach feedback to.")
    return updated

  def answer_feedback(self, record_id: str, index: int, response: str,
                      changed: bool = True) -> RunRecord:
    """Fill in what was done about one remark."""
    if not response.strip():
      raise StoreError("An empty response is not an answer.")

    box: dict = {}

    def answer(fresh: RunRecord) -> Any:
      if not -len(fresh.feedback) <= index < len(fresh.feedback):
        box["error"] = (
            f"Run {record_id} has {len(fresh.feedback)} feedback entries; "
            f"there is no index {index}."
        )
        return False
      entry = fresh.feedback[index]
      entry.response = response
      entry.changed = bool(changed)
      return None  # Anything but False: write the record.

    updated = self.update_record(record_id, answer)
    if updated is None:
      raise StoreError(box.get("error") or f"No record '{record_id}'.")
    return updated

  def feedback_timeline(
      self,
      kind: str | None = None,
      author: str | None = None,
      outstanding: bool = False,
      limit: int | None = None,
  ) -> list[dict]:
    """Every remark in the store, oldest first."""
    clauses, params = [], []
    if kind is not None:
      clauses.append("kind = ?")
      params.append(kind)
    if author is not None:
      clauses.append("author = ?")
      params.append(author)
    if outstanding:
      clauses.append("outstanding = 1")
    sql = "SELECT run_id, idx FROM feedback"
    if clauses:
      sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY at, run_id, idx"
    if limit:
      sql += f" LIMIT {int(limit)}"

    with contextlib.closing(self._connect()) as conn:
      hits = [(row["run_id"], row["idx"]) for row in conn.execute(sql, params)]

    # The index answers *which*; the files answer *what* -- the same division
    # ``query`` uses, so a stale index can never invent a remark.
    rows = []
    cache: dict = {}
    for run_id, idx in hits:
      if run_id not in cache:
        cache[run_id] = self.get_record(run_id)
      record = cache[run_id]
      if record is None or not 0 <= idx < len(record.feedback):
        continue
      rows.append({"run": run_id, "slug": record.slug, "index": idx,
                   **record.feedback[idx].to_dict()})
    return rows

  # Documents.

  def write_document(self, record_id: str, name: str, text: str) -> Path | None:
    """Write PLAN.md / REPORT.md next to the record."""
    directory = self._find_dir(record_id)
    if directory is None:
      return None
    path = directory / _safe_component(name, "document name")
    _atomic_write(path, text)
    return path

  def read_document(self, record_id: str, name: str) -> str | None:
    directory = self._find_dir(record_id)
    if directory is None:
      return None
    path = directory / _safe_component(name, "document name")
    return path.read_text() if path.exists() else None

  # Leases.

  def claim(
      self,
      record_id: str,
      slot: str,
      holder: str = "",
      ttl_seconds: float = 900.0,
      owner: str | None = None,
  ) -> RunRecord:
    """Take ``slot`` for this record, or raise :class:`SlotUnavailable`.

    ``owner=None`` infers the kind of claim: naming a ``holder`` is what the
    trainer process does for itself (``runner`` -- its pid means something), a
    bare claim is a reservation made from a short-lived process (``manual`` --
    recording this pid would hand the slot back the moment it exited).
    """
    if owner is None:
      owner = "runner" if holder else "manual"
    if owner not in ("runner", "manual"):
      raise StoreError(f"Unknown lease owner '{owner}': use 'runner' or 'manual'.")
    self.reap_expired()
    with self._tx() as conn:
      record = self.get_record(record_id)
      if record is None:
        raise StoreError(f"No record '{record_id}'.")

      now = time.time()
      live = [
          r for r in self.list_records()
          if r.lease is not None and not r.lease.dead(now) and r.id != record_id
      ]
      taken = {r.lease.slot for r in live if r.lease}
      if slot in taken:
        holding = next(r.id for r in live if r.lease and r.lease.slot == slot)
        raise SlotUnavailable(f"Slot '{slot}' is held by run {holding}.")
      if len(live) >= self.slots:
        raise SlotUnavailable(
            f"All {self.slots} slot(s) are in use by {[r.id for r in live]}. "
            "Close a run, or raise the slot count if this machine really has more."
        )

      record.lease = Lease(
          slot=slot,
          holder=holder or (f"manual-{uuid.uuid4().hex[:8]}" if owner == "manual" else ""),
          owner=owner,
          pid=os.getpid() if owner == "runner" else 0,
          ttl_seconds=ttl_seconds,
      )
      return self._persist(record, conn)

  def heartbeat(self, record_id: str) -> RunRecord | None:
    if self.get_record(record_id) is None:
      return None

    def renew(record: RunRecord) -> Any:
      if record.lease is None:
        return False
      record.lease.renewed_at = time.time()
      return None  # Anything but False: write the record.

    return self.update_record(record_id, renew)

  def release(self, record_id: str) -> RunRecord | None:
    if self.get_record(record_id) is None:
      return None

    def drop(record: RunRecord) -> None:
      record.lease = None

    return self.update_record(record_id, drop)

  def reap_expired(self, stale_running_ttl: float = STALE_RUNNING_TTL) -> list[RunRecord]:
    """A job that died is not a result -- move it out of ``running``.

    Three shapes of death, one verdict: ``interrupted`` is the terminal state
    that stays re-closable, so the real close-out can still land once someone
    reads the wreckage. What the reaper saw goes in ``links["reaped"]``:
    ``holder-gone`` (a runner lease whose pid died), ``lease-expired``
    (nothing renewed the lease in time -- the only way a manual reservation
    dies), or ``stale-leaseless`` (a ``running`` record holding no lease at
    all, the shape left by a run that started under a claim warning and then
    vanished, reaped once untouched for ``stale_running_ttl``).
    """
    now = time.time()
    reaped = []
    for record in self.list_records():
      if record.lease is not None and record.lease.dead(now):

        def expire(fresh: RunRecord) -> Any:
          if fresh.lease is None or not fresh.lease.dead(now):
            return False  # Someone renewed or released it since we looked.
          cause = "holder-gone" if fresh.lease.holder_is_gone() else "lease-expired"
          fresh.lease = None
          if fresh.verdict == "running":
            fresh.verdict = "interrupted"
            fresh.links["reaped"] = cause
          return None  # Anything but False: write the record.

        updated = self.update_record(record.id, expire)
      elif (record.verdict == "running" and record.lease is None
            and now - record.updated_at > stale_running_ttl):

        def stale(fresh: RunRecord) -> Any:
          if (fresh.verdict != "running" or fresh.lease is not None
              or now - fresh.updated_at <= stale_running_ttl):
            return False
          fresh.verdict = "interrupted"
          fresh.links["reaped"] = "stale-leaseless"
          return None  # Anything but False: write the record.

        updated = self.update_record(record.id, stale)
      else:
        continue
      if updated is not None:
        reaped.append(updated)
    return reaped

  def __repr__(self) -> str:  # pragma: no cover - debugging aid.
    return f"FileStore({str(self.root)!r}, slots={self.slots})"


def open_store(root: Path | str | None = None, slots: int = 1) -> FileStore:
  """Open (or create) the records.

  Resolution order: the argument, ``$RLMCP_RECORDS``, then ``./records`` -- the same
  shape as session discovery, so a project can be worked in without configuration.
  """
  if root is None:
    root = os.environ.get("RLMCP_RECORDS") or "./records"
  return FileStore(root, slots=slots)
