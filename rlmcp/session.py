"""Session directory protocol shared by the trainer and the MCP server.

The trainer process owns the simulator; the MCP server (and any CLI) is a
separate process. They talk through a plain directory so that:

* the MCP server never needs to import torch / mujoco,
* the training process never needs to open a socket,
* everything is inspectable with ``cat`` when something goes wrong,
* a crashed server does not take training down with it.

Layout::

    <session_dir>/
      session.json        static run info (task, pid, num_envs, device, ...)
      status.json         live heartbeat, rewritten every service tick
      metrics.jsonl       one JSON object per learning iteration
      events.jsonl        audit trail: parameter edits, stage changes, notes
      params.json         parameter schema snapshot (updated on change)
      inbox/              pending requests written by the agent side
      outbox/             responses written by the trainer side
      artifacts/          png / mp4 / npz produced on demand
      env_terms.json      reward / observation / action terms, with their source
      rewards/            source of every reward term added mid-run
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

SCHEMA_VERSION = 1

#: Keys a metrics row carries that are not measurements. Every reader that
#: asks "what did this run log?" walks the row's keys, so bookkeeping fields
#: have to be named in one place -- otherwise the next one added shows up as a
#: metric in the CLI's list, on a plot's axis, and in the studio's headline.
RESERVED_METRIC_KEYS = ("seq", "iteration", "t")

PLAY_SESSION_KIND = "rlmcp-play-session"
"""``kind`` a play session writes into its session.json.

A play session is a real session -- it can be steered from another shell like
any other -- but it is not the run. Discovery skips it by default so that
looking at a finished run does not leave a newer, dead session sitting in front
of the training one every bare ``rlmcp`` command resolves to.
"""

_SUBMIT_SEQ = 0  # Process-local tiebreaker for same-millisecond submissions.

_SUBDIRS = ("inbox", "outbox", "artifacts")


def _atomic_write_text(path: Path, text: str) -> None:
  """Write ``text`` to ``path`` so readers never observe a partial file."""
  tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
  tmp.write_text(text)
  os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: Any) -> None:
  _atomic_write_text(
      path,
      json.dumps(_sanitize_for_json(obj), indent=2, default=_json_default,
                 allow_nan=False),
  )


def _json_default(obj: Any) -> Any:
  """Best-effort JSON coercion for numpy / torch scalars."""
  for attr in ("item", "tolist"):
    fn = getattr(obj, attr, None)
    if callable(fn):
      try:
        return fn()
      except Exception:
        pass
  return str(obj)


def _sanitize_for_json(obj: Any) -> Any:
  """Replace non-finite floats with None, recursively.

  ``json.dumps`` writes ``NaN`` / ``Infinity`` by default, which is not JSON.
  A run's metrics go NaN exactly when it diverges -- exactly when someone
  points jq, a JS client or an MCP JSON-RPC layer at these files -- so the
  writers must never emit them. A NaN metric appears as ``null``: a truthful
  "no finite value here" that strict parsers accept. Numpy / torch scalars are
  coerced first so a ``float32`` NaN cannot sneak past the float check.
  """
  if isinstance(obj, float):
    return obj if math.isfinite(obj) else None
  if obj is None or isinstance(obj, (str, int, bool)):
    return obj
  if isinstance(obj, dict):
    return {k: _sanitize_for_json(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [_sanitize_for_json(v) for v in obj]
  coerced = _json_default(obj)
  return _sanitize_for_json(coerced) if coerced is not obj else coerced


def _read_json(path: Path, default: Any = None) -> Any:
  try:
    return json.loads(path.read_text())
  except (OSError, json.JSONDecodeError):
    return default


def _append_jsonl(path: Path, obj: Any) -> None:
  with path.open("a") as f:
    f.write(json.dumps(_sanitize_for_json(obj), default=_json_default,
                       allow_nan=False) + "\n")


#: Chunk size for backward tail reads. Small enough that asking for one line
#: of a 100MB metrics log costs one block, large enough that a typical
#: ``last_n=400`` request finishes in a handful of reads.
_TAIL_BLOCK_BYTES = 64 * 1024


def _tail_lines(path: Path, last_n: int) -> list[str]:
  """The last ``last_n`` physical lines of ``path``, read backward from EOF.

  Blocks are read from the end until the buffer holds ``last_n + 1`` newlines
  (or the whole file). The extra newline guarantees the buffer's first entry --
  the only one a block boundary can cut mid-line, or mid-UTF-8-character --
  stays outside the returned window, so the result matches what a whole-file
  read would have sliced.
  """
  with path.open("rb") as f:
    f.seek(0, os.SEEK_END)
    pos = f.tell()
    buf = b""
    while pos > 0 and buf.count(b"\n") < last_n + 1:
      step = min(_TAIL_BLOCK_BYTES, pos)
      pos -= step
      f.seek(pos)
      buf = f.read(step) + buf
  return buf.decode("utf-8", errors="replace").splitlines()[-last_n:]


def read_jsonl(path: Path, last_n: int | None = None) -> list[Any]:
  """Read a JSONL file, skipping any torn trailing line.

  With ``last_n`` only the tail of the file is read (backward, in blocks), so
  asking for the recent rows of a day-old metrics log costs kilobytes rather
  than the whole file. ``last_n=None`` reads everything; ``last_n=0`` reads
  nothing. As with a whole-file read, ``last_n`` counts physical lines: a tail
  torn by a mid-write kill occupies the last line and is then skipped.
  """
  if last_n is not None and last_n <= 0:
    return []
  try:
    lines = (path.read_bytes().decode("utf-8", errors="replace").splitlines()
             if last_n is None else _tail_lines(path, last_n))
  except OSError:
    return []
  out: list[Any] = []
  for raw in lines:
    line = raw.strip()
    if not line:
      continue
    try:
      out.append(json.loads(line))
    except json.JSONDecodeError:
      continue
  return out


def last_row(path: Path) -> dict[str, Any] | None:
  """The last well-formed JSON object in a JSONL file, or None.

  One backward block read, so "where had I got to" costs the same on a
  day-old metrics log as on an empty one.
  """
  for raw in reversed(_tail_lines(path, 2) if path.exists() else []):
    line = raw.strip()
    if not line:
      continue
    try:
      row = json.loads(line)
    except json.JSONDecodeError:
      continue  # A torn last line; the one before it still counts.
    if isinstance(row, dict):
      return row
  return None


def rows_since(path: Path, since_seq: int) -> list[dict[str, Any]]:
  """Rows whose ``seq`` is greater than ``since_seq``, oldest first.

  The cursor a reader that is not on this machine needs: it holds the last
  ``seq`` it saw and asks for what came after, instead of refetching a log
  that only ever grows.

  Sequences are contiguous and one-based, so the last row's ``seq`` is the
  row count, and the answer is the last ``last - since`` lines -- two backward
  reads, no scan. Rows are filtered again after reading, because a file
  written by an older rlmcp has no ``seq`` at all: that history is returned
  once, to a reader starting from the beginning, and never again to one
  holding a cursor.
  """
  last = last_row(path) or {}
  latest = last.get("seq")
  if isinstance(latest, int):
    want = max(0, latest - max(0, since_seq))
    rows = read_jsonl(path, last_n=want) if want else []
  else:
    rows = read_jsonl(path)  # No cursor to seek by; read it and filter.
  out = []
  for row in rows:
    seq = row.get("seq")
    if isinstance(seq, int):
      if seq > since_seq:
        out.append(row)
    elif since_seq <= 0:
      out.append(row)
  return out


@dataclass
class Request:
  """A command from the agent side to the training loop."""

  cmd: str
  args: dict[str, Any] = field(default_factory=dict)
  req_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
  created_at: float = field(default_factory=time.time)

  def to_dict(self) -> dict[str, Any]:
    return {
        "req_id": self.req_id,
        "cmd": self.cmd,
        "args": self.args,
        "created_at": self.created_at,
    }

  @staticmethod
  def from_dict(d: dict[str, Any]) -> Request:
    created = d.get("created_at")
    return Request(
        cmd=d["cmd"],
        args=d.get("args") or {},
        req_id=d["req_id"],
        # A request without a numeric timestamp is treated as freshly
        # submitted: writers that predate the field must keep working.
        created_at=created if isinstance(created, (int, float)) else time.time(),
    )


@dataclass
class Response:
  """The training loop's answer to a :class:`Request`."""

  req_id: str
  ok: bool
  result: Any = None
  error: str | None = None
  finished_at: float = field(default_factory=time.time)

  def to_dict(self) -> dict[str, Any]:
    return {
        "req_id": self.req_id,
        "ok": self.ok,
        "result": self.result,
        "error": self.error,
        "finished_at": self.finished_at,
    }

  @staticmethod
  def from_dict(d: dict[str, Any]) -> Response:
    return Response(
        req_id=d["req_id"],
        ok=d["ok"],
        result=d.get("result"),
        error=d.get("error"),
        finished_at=d.get("finished_at", time.time()),
    )


#: Every name a client of a run may use. Frozen on purpose: a reader that
#: stays inside this list keeps working when the run is on another machine,
#: and adding to it is a decision about the wire, not a convenience.
WIRE_SURFACE = (
    "address", "key", "name",
    "info", "status", "params",
    "metrics", "metrics_count", "events",
    "list_artifacts", "read_artifact",
    "submit", "poll", "wait", "call",
    "liveness", "liveness_info",
)


@runtime_checkable
class SessionClient(Protocol):
  """What reaching a run requires, and the whole of it.

  :class:`Session` is one implementation -- the local one, where reaching a run
  means reading its directory. It is the only one today and the right default
  on a single machine: no socket, nothing to crash, and ``cat`` still works
  when something is wrong.

  This exists because it will not be the only one. A run on a GPU box the
  reader cannot see needs a second implementation over a connection, and the
  cost of that is decided here: **everything above this protocol -- the CLI,
  the MCP server, rl-mcp-studio -- is written against these names and no
  others**, so the second implementation changes one layer instead of every
  caller.

  Which is why the two things a filesystem gives away for free are methods
  rather than paths. ``list_artifacts`` and ``read_artifact`` exist because a
  caller that reaches a plot by joining ``session.dir / "artifacts"`` compiles
  fine, works locally, and cannot be made to work at all when the file is on
  another machine.

  ``address`` is the string that names this run to a person and to
  ``--session``; today a path, later a URL. Nothing may parse it.
  """

  @property
  def address(self) -> str:
    """Where this run is, in whatever way the transport addresses it."""

  @property
  def key(self) -> str:
    """Short identity carried in payloads: ``<run>/<session>``."""

  @property
  def name(self) -> str:
    """The run's own name, for titles."""

  def info(self) -> dict[str, Any]: ...
  def status(self) -> dict[str, Any]: ...
  def params(self) -> dict[str, Any]: ...
  def metrics(self, last_n: int | None = ...,
              since_seq: int | None = ...) -> list[dict[str, Any]]: ...
  def metrics_count(self) -> int: ...
  def events(self, last_n: int | None = ...,
             since_seq: int | None = ...) -> list[dict[str, Any]]: ...
  def list_artifacts(self) -> list[dict[str, Any]]: ...
  def read_artifact(self, name: str) -> bytes: ...
  def submit(self, cmd: str, **args: Any) -> Request: ...
  def poll(self, req_id: str, consume: bool = ...) -> Response | None: ...
  def wait(self, req_id: str, timeout: float = ..., interval: float = ...) -> Response: ...
  def call(self, cmd: str, timeout: float = ..., **args: Any) -> Response: ...
  def liveness(self) -> str: ...
  def liveness_info(self) -> dict[str, Any]: ...


class Session:
  """Filesystem handle on one training session.

  Both sides construct this. The trainer calls :meth:`create` once and then the
  ``publish_*`` / :meth:`pop_requests` methods; the agent side calls
  :meth:`open` and then :meth:`submit` / :meth:`status`.
  """

  #: Queued requests older than this are refused, not executed. It matches the
  #: default client-side :meth:`wait` timeout, so a command is never serviced
  #: after every well-behaved waiter has already given up on it. This bounds
  #: queue age only: several MCP tools wait up to 600s for deferred jobs, and a
  #: request claimed before expiry may legitimately take that long to answer.
  REQUEST_MAX_AGE_S: float = 120.0

  #: Heartbeat age beyond which a live pid is reported as "stalled". Truthful,
  #: not fatal: a single very long iteration (a terrain rebuild, a checkpoint
  #: to slow storage) can legitimately exceed it.
  STALL_AFTER_S: float = 600.0

  #: Heartbeat age beyond which an existing pid is no longer believed. After a
  #: day without a status write, "the pid exists" almost always means the id
  #: was recycled by an unrelated process, and believing it would suppress
  #: offline fallbacks for days. Judgment call; :meth:`liveness_info` says so
  #: in its payload when this rule fires.
  PRESUMED_DEAD_AFTER_S: float = 24 * 3600.0

  #: :meth:`publish_status` prunes old outbox responses at most this often.
  OUTBOX_PRUNE_INTERVAL_S: float = 60.0

  #: Responses older than this are deleted by the publish-cadence prune. Long
  #: past every waiter's timeout; only unconsumed responses (a client that
  #: died mid-wait, refusals nobody polled) ever reach this age.
  OUTBOX_KEEP_S: float = 3600.0

  def __init__(self, session_dir: Path | str):
    self.dir = Path(session_dir).expanduser().resolve()
    self._cached_pid: int | None = None
    self._last_outbox_prune = 0.0
    #: Next sequence number for each stream this process writes, filled in
    #: lazily from what is already on disk so a trainer restarted onto the
    #: same directory continues the count instead of replaying it.
    self._next_seq: dict[str, int] = {}

  # Identity. Three strings, because callers want three different things and
  # a path answers all of them only while the run is on this machine.

  @property
  def address(self) -> str:
    """The directory, as the string ``--session`` takes."""
    return str(self.dir)

  @property
  def key(self) -> str:
    """``<run>/<session>`` -- what payloads name the session by.

    Short enough to read in a table, specific enough to tell two runs apart,
    and it survives the run moving: the last two segments are the run's, not
    the machine's.
    """
    parent = self.dir.parent.name
    return f"{parent}/{self.dir.name}" if parent else self.dir.name

  @property
  def name(self) -> str:
    """The run's own name -- the log directory, not the ``rlmcp`` inside it."""
    return self.dir.parent.name or self.dir.name

  # Paths.

  @property
  def session_file(self) -> Path:
    return self.dir / "session.json"

  @property
  def status_file(self) -> Path:
    return self.dir / "status.json"

  @property
  def metrics_file(self) -> Path:
    return self.dir / "metrics.jsonl"

  @property
  def events_file(self) -> Path:
    return self.dir / "events.jsonl"

  @property
  def params_file(self) -> Path:
    return self.dir / "params.json"

  @property
  def inbox(self) -> Path:
    return self.dir / "inbox"

  @property
  def outbox(self) -> Path:
    return self.dir / "outbox"

  @property
  def artifacts(self) -> Path:
    return self.dir / "artifacts"

  @property
  def env_terms_file(self) -> Path:
    """The captured reward / observation / action terms of this run.

    Written once at startup and refreshed when a term is added. It is what
    lets a checkpoint be paired with the environment it trained under after
    the training process is gone -- see ``rlmcp env export``.
    """
    return self.dir / "env_terms.json"

  @property
  def rewards(self) -> Path:
    """Source of reward terms added during the run, one file per term.

    Kept out of ``artifacts`` because it is not an output to look at: it is
    what the run was optimising, and the only copy of a term that existed
    nowhere before the agent wrote it. Created on first write, so a run that
    adds none has no empty directory.
    """
    return self.dir / "rewards"

  # Lifecycle.

  def create(self, info: dict[str, Any]) -> Session:
    """Initialise the directory. Called once by the training process.

    Any inbox backlog left behind by a previous process -- pending requests as
    well as ones it claimed but never finished -- is refused here with error
    responses rather than executed: a fresh run must never service commands
    that were aimed at a dead one.
    """
    self.dir.mkdir(parents=True, exist_ok=True)
    for sub in _SUBDIRS:
      (self.dir / sub).mkdir(exist_ok=True)
    self._sweep_stale_inbox()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "pid": os.getpid(),
        "started_at": time.time(),
        **info,
    }
    _atomic_write_json(self.session_file, payload)
    self._cached_pid = None  # session.json now names a new owner.
    # Announce the session machine-wide, so a bare `rlmcp status` in another
    # shell can find this run without knowing its directory. Best-effort by
    # contract: registry.register never raises.
    from rlmcp import registry

    registry.register(registry.KIND_TRAINER, session_dir=self.dir,
                      session_kind=payload.get("kind"))
    return self

  @staticmethod
  def open(session_dir: Path | str) -> Session:
    s = Session(session_dir)
    if not s.session_file.exists():
      raise FileNotFoundError(
          f"No rlmcp session at '{s.dir}' (missing session.json). "
          "Pass the directory printed by the trainer at startup."
      )
    return s

  @staticmethod
  def find_latest(root: Path | str) -> Session | None:
    """Return the most recently started session under ``root``, if any."""
    return next(iter_sessions(root), None)

  # Trainer side.

  def _seq(self, stream: str, resume_from: Callable[[], int]) -> int:
    """The next sequence number for ``stream``, counted from disk once."""
    nxt = self._next_seq.get(stream)
    if nxt is None:
      nxt = resume_from() + 1
    self._next_seq[stream] = nxt + 1
    return nxt

  def _resume_status_seq(self) -> int:
    seq = self.status().get("seq")
    return seq if isinstance(seq, int) else 0

  def _resume_log_seq(self, path: Path) -> int:
    """Where a log left off: its last ``seq``, else its length."""
    row = last_row(path)
    if row is None:
      return 0
    seq = row.get("seq")
    return seq if isinstance(seq, int) else len(read_jsonl(path))

  def publish_status(self, status: dict[str, Any]) -> None:
    seq = self._seq("status", self._resume_status_seq)
    _atomic_write_json(self.status_file,
                       {"updated_at": time.time(), "seq": seq, **status})
    # Piggyback outbox hygiene on the heartbeat: responses are deleted by the
    # waiter that consumes them, so anything old enough to prune belongs to a
    # client that vanished. Throttled -- the pause loop publishes ~7x/s.
    now = time.time()
    if now - self._last_outbox_prune >= self.OUTBOX_PRUNE_INTERVAL_S:
      self._last_outbox_prune = now
      self.prune_outbox(self.OUTBOX_KEEP_S)

  def publish_params(self, schema: dict[str, Any]) -> None:
    _atomic_write_json(self.params_file, schema)

  def publish_env_terms(self, terms: dict[str, Any]) -> None:
    _atomic_write_json(self.env_terms_file, terms)

  def env_terms(self) -> dict[str, Any]:
    return _read_json(self.env_terms_file, {}) or {}

  def append_metrics(self, iteration: int, metrics: dict[str, float]) -> None:
    seq = self._seq("metrics", lambda: self._resume_log_seq(self.metrics_file))
    _append_jsonl(self.metrics_file,
                  {"seq": seq, "iteration": iteration, "t": time.time(), **metrics})

  def append_event(self, kind: str, detail: dict[str, Any]) -> None:
    seq = self._seq("events", lambda: self._resume_log_seq(self.events_file))
    _append_jsonl(self.events_file,
                  {"seq": seq, "t": time.time(), "kind": kind, **detail})

  def pop_requests(self, max_age_s: float | None = None) -> list[Request]:
    """Claim every pending request, oldest first.

    Claiming renames the file into ``outbox`` territory first so a duplicate
    reader (or a restarted trainer) cannot execute the same command twice.

    Requests older than ``max_age_s`` (default :attr:`REQUEST_MAX_AGE_S`) are
    refused instead of returned: every well-behaved waiter has given up on
    them, and a late ``stop_training`` or ``load_checkpoint`` is a side effect
    nobody asked for anymore. Each refusal writes an error response and an
    event. Requests without a timestamp are treated as fresh.
    """
    if max_age_s is None:
      max_age_s = self.REQUEST_MAX_AGE_S
    if not self.inbox.exists():
      return []
    files = sorted(self.inbox.glob("*.json"), key=lambda p: p.name)
    requests: list[Request] = []
    for path in files:
      claimed = path.with_suffix(".claimed")
      try:
        os.replace(path, claimed)
      except OSError:
        continue  # Someone else got it.
      payload = _read_json(claimed)
      with contextlib.suppress(OSError):
        claimed.unlink()
      if not isinstance(payload, dict) or "cmd" not in payload:
        continue
      try:
        request = Request.from_dict(payload)
      except KeyError:
        continue
      age = time.time() - request.created_at
      if age > max_age_s:
        self._refuse_request(
            request.req_id,
            request.cmd,
            error=(
                f"expired before execution: submitted {age:.0f}s ago, limit "
                f"{max_age_s:.0f}s. The trainer was not servicing commands "
                "while this sat queued; resubmit it if it is still wanted."
            ),
            detail={"age_s": round(age, 3), "max_age_s": max_age_s, "reason": "ttl"},
        )
        continue
      requests.append(request)
    return requests

  def _refuse_request(
      self, req_id: str, cmd: str | None, error: str, detail: dict[str, Any]
  ) -> None:
    """Answer a request with an error instead of executing it, and log why."""
    self.respond(Response(req_id=req_id, ok=False, error=error))
    self.append_event("request_expired", {"req_id": req_id, "cmd": cmd, **detail})

  def _sweep_stale_inbox(self) -> None:
    """Refuse whatever a previous process left in the inbox, executing nothing.

    Covers pending ``*.json`` requests and ``*.claimed`` files orphaned by a
    crash between claiming and executing. Each readable request gets an error
    response, so a client still waiting on it unblocks with the truth.
    """
    if not self.inbox.exists():
      return
    leftovers = sorted(self.inbox.glob("*.json")) + sorted(self.inbox.glob("*.claimed"))
    for path in leftovers:
      payload = _read_json(path)
      try:
        path.unlink()
      except OSError:
        continue  # Someone else got it.
      if not isinstance(payload, dict) or "req_id" not in payload:
        continue
      created = payload.get("created_at")
      age = max(0.0, time.time() - created) if isinstance(created, (int, float)) else None
      submitted = f"submitted {age:.0f}s ago, " if age is not None else ""
      self._refuse_request(
          payload["req_id"],
          payload.get("cmd"),
          error=(
              f"expired before execution: {submitted}queued for a previous "
              "training process. A new run starts with an empty inbox; "
              "resubmit the command if it is still wanted."
          ),
          detail={"age_s": age, "reason": "startup_sweep"},
      )

  def respond(self, response: Response) -> None:
    self.outbox.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(self.outbox / f"{response.req_id}.json", response.to_dict())

  def artifact_path(self, name: str) -> Path:
    self.artifacts.mkdir(parents=True, exist_ok=True)
    return self.artifacts / name

  # Agent side.

  def info(self) -> dict[str, Any]:
    return _read_json(self.session_file, {}) or {}

  def status(self) -> dict[str, Any]:
    return _read_json(self.status_file, {}) or {}

  def params(self) -> dict[str, Any]:
    return _read_json(self.params_file, {}) or {}

  def metrics(self, last_n: int | None = None,
              since_seq: int | None = None) -> list[dict[str, Any]]:
    """Metric rows, newest last. ``since_seq`` reads only what is new."""
    if since_seq is not None:
      return rows_since(self.metrics_file, since_seq)
    return read_jsonl(self.metrics_file, last_n=last_n)

  def events(self, last_n: int | None = None,
             since_seq: int | None = None) -> list[dict[str, Any]]:
    """Session events, oldest first. ``since_seq`` reads only what is new."""
    if since_seq is not None:
      return rows_since(self.events_file, since_seq)
    return read_jsonl(self.events_file, last_n=last_n)

  def metrics_count(self) -> int:
    """How many metric rows the run has logged.

    A separate question from ``len(metrics(last_n=N))``: a report wants the
    total while reading only a window of it, and counting lines here keeps the
    caller from opening the file to find out.
    """
    try:
      with self.metrics_file.open("rb") as f:
        return sum(1 for line in f if line.strip())
    except OSError:
      return 0

  def list_artifacts(self) -> list[dict[str, Any]]:
    """Files this run produced -- newest first.

    ``name``, ``bytes`` and ``modified_at`` are the contract. ``path`` is
    present only because this transport has one, and a caller that requires it
    is a caller that cannot read a run on another machine.
    """
    rows: list[dict[str, Any]] = []
    if not self.artifacts.exists():
      return rows
    for path in self.artifacts.iterdir():
      try:
        stat = path.stat()
      except OSError:
        continue
      if not path.is_file():
        continue
      rows.append({
          "name": path.name,
          "path": str(path),
          "bytes": stat.st_size,
          "modified_at": stat.st_mtime,
      })
    rows.sort(key=lambda r: r["modified_at"], reverse=True)
    return rows

  def read_artifact(self, name: str) -> bytes:
    """The bytes of one artifact, by the ``name`` :meth:`list_artifacts` gave.

    Only a name: no separators, no ``..``, no absolute paths. The check is
    here rather than in each caller because this method is the one a remote
    transport re-implements, and there the same argument crosses a network
    from wherever the studio got it.
    """
    if not name or name != Path(name).name or name in (".", ".."):
      raise ValueError(
          f"'{name}' is not an artifact name. Pass the `name` from "
          "list_artifacts(), not a path."
      )
    return (self.artifacts / name).read_bytes()

  def submit(self, cmd: str, **args: Any) -> Request:
    """Queue a command for the training loop (does not wait)."""
    self.inbox.mkdir(parents=True, exist_ok=True)
    req = Request(cmd=cmd, args=args)
    # Timestamp orders across processes; the counter keeps FIFO within one, where
    # two submits can land in the same millisecond.
    global _SUBMIT_SEQ
    _SUBMIT_SEQ += 1
    name = f"{int(req.created_at * 1000):015d}-{_SUBMIT_SEQ:06d}-{req.req_id}.json"
    _atomic_write_json(self.inbox / name, req.to_dict())
    return req

  def poll(self, req_id: str, consume: bool = False) -> Response | None:
    """Read the response to ``req_id`` if it has arrived.

    With ``consume`` the response file is deleted after a successful read.
    Only the requester should consume: a response has exactly one intended
    reader, and consuming is what keeps the outbox from accumulating one file
    per command for the life of the run.
    """
    path = self.outbox / f"{req_id}.json"
    payload = _read_json(path)
    if not isinstance(payload, dict):
      return None
    # Parse before unlinking: valid JSON that is not a Response must return
    # None and stay on disk as evidence, not be consumed and then crash the
    # reader. The publish-cadence prune sweeps it eventually.
    try:
      response = Response.from_dict(payload)
    except (KeyError, TypeError):
      return None
    if consume:
      with contextlib.suppress(OSError):
        path.unlink()
    return response

  def wait(self, req_id: str, timeout: float = 120.0, interval: float = 0.1) -> Response:
    """Block until the trainer answers ``req_id`` or ``timeout`` elapses.

    The caller is the response's only consumer, so a response read here is
    deleted from the outbox; polling the same ``req_id`` again later returns
    nothing. A dead trainer is reported on the first poll that sees it -- after
    one confirming re-poll, because the answer may land in the gap between
    looking at the outbox and looking at the pid.
    """
    deadline = time.time() + timeout
    while True:
      resp = self.poll(req_id, consume=True)
      if resp is not None:
        return resp
      if not self.is_alive():
        # The trainer may have answered and exited between the poll and the
        # pid check; its writes precede its death, so one re-poll settles it.
        resp = self.poll(req_id, consume=True)
        if resp is not None:
          return resp
        if not self.is_alive():
          return Response(
              req_id=req_id,
              ok=False,
              error="Training process is not running; command will not be serviced.",
          )
      if time.time() >= deadline:
        break
      time.sleep(interval)
    return Response(
        req_id=req_id,
        ok=False,
        error=(
            f"Timed out after {timeout:.0f}s. The trainer services commands "
            "between rollout batches; a command it does not service within "
            f"{self.REQUEST_MAX_AGE_S:.0f}s of submission is refused as "
            "expired rather than executed late."
        ),
    )

  def call(self, cmd: str, timeout: float = 120.0, **args: Any) -> Response:
    """Submit a command and wait for its response."""
    req = self.submit(cmd, **args)
    return self.wait(req.req_id, timeout=timeout)

  # Liveness.

  def is_alive(self) -> bool:
    """True when the pid that created this session still exists.

    The pid is parsed from session.json once and then cached on this instance:
    a wait loop probes liveness ten times a second, and the answer to "which
    process owns this directory" does not change while that process lives. A
    dead answer clears the cache, so a trainer restarted onto the same
    directory (which rewrites session.json) is picked up on the next call.

    This is a bare existence check; a recycled pid passes it. Callers deciding
    between live commands and disk fallbacks should use :meth:`liveness`,
    which folds in the heartbeat.
    """
    pid = self._cached_pid
    if pid is None:
      pid = self.info().get("pid")
      if not isinstance(pid, int):
        return False
      self._cached_pid = pid
    try:
      os.kill(pid, 0)
    except ProcessLookupError:
      self._cached_pid = None
      return False
    except PermissionError:
      return True
    return True

  def seconds_since_heartbeat(self) -> float | None:
    updated = self.status().get("updated_at")
    if not isinstance(updated, (int, float)):
      return None
    return max(0.0, time.time() - float(updated))

  def liveness(self) -> str:
    """``"running"``, ``"stalled"`` or ``"dead"`` -- see :meth:`liveness_info`."""
    return self.liveness_info()["state"]

  def liveness_info(self) -> dict[str, Any]:
    """The liveness verdict plus the evidence it rests on.

    Returns ``{"state", "pid_alive", "heartbeat_age_s"[, "note"]}`` where
    ``state`` is one of:

    * ``"dead"`` -- the recorded pid is gone; or it exists but the heartbeat is
      older than :attr:`PRESUMED_DEAD_AFTER_S`, in which case the pid is
      presumed recycled and the note says so.
    * ``"stalled"`` -- the pid exists but the heartbeat is older than
      :attr:`STALL_AFTER_S`. Not fatal, and sometimes not even wrong: the note
      reminds the reader that one long iteration can look like this.
    * ``"running"`` -- pid exists and the heartbeat is fresh, or there is no
      status.json yet to judge by (a run still starting up).
    """
    pid_alive = self.is_alive()
    age = self.seconds_since_heartbeat()
    out: dict[str, Any] = {
        "pid_alive": pid_alive,
        "heartbeat_age_s": None if age is None else round(age, 1),
    }
    if not pid_alive:
      out["state"] = "dead"
    elif age is not None and age > self.PRESUMED_DEAD_AFTER_S:
      out["state"] = "dead"
      out["note"] = (
          f"pid exists but the last status write was {age / 3600.0:.1f}h ago "
          f"(limit {self.PRESUMED_DEAD_AFTER_S / 3600.0:.0f}h); treating the "
          "run as dead -- a pid this stale usually belongs to some unrelated "
          "process that recycled the id."
      )
    elif age is not None and age > self.STALL_AFTER_S:
      out["state"] = "stalled"
      out["note"] = (
          f"pid is alive but nothing has been published for {age:.0f}s "
          f"(threshold {self.STALL_AFTER_S:.0f}s); a single long iteration "
          "(terrain rebuild, slow checkpoint) can legitimately look like this."
      )
    else:
      out["state"] = "running"
    return out

  def prune_outbox(self, keep_seconds: float = OUTBOX_KEEP_S) -> int:
    """Delete old responses so the directory does not grow without bound.

    :meth:`wait` consumes responses as it reads them, so what ages out here is
    the residue of clients that died mid-wait and refusals nobody polled.
    :meth:`publish_status` calls this on the trainer's heartbeat cadence,
    throttled to once per :attr:`OUTBOX_PRUNE_INTERVAL_S`.
    """
    if not self.outbox.exists():
      return 0
    cutoff = time.time() - keep_seconds
    removed = 0
    for path in self.outbox.glob("*.json"):
      try:
        if path.stat().st_mtime < cutoff:
          path.unlink()
          removed += 1
      except OSError:
        pass
    return removed

  def __repr__(self) -> str:  # pragma: no cover - debugging aid.
    return f"Session({str(self.dir)!r})"


#: Directory names discovery never descends into: bulky run outputs and tool
#: trees that sit next to (or inside) session directories but never contain
#: one. Hidden directories (``.git``, ``.venv*``, caches) are pruned by rule.
_DISCOVERY_PRUNE = {"artifacts", "checkpoints", "media", "node_modules", "__pycache__"}


def _walk_session_files(root: Path) -> Iterator[Path]:
  """Yield every session.json under ``root`` without crawling junk.

  ``rglob`` would descend into artifacts/, checkpoints/, .git and virtualenvs
  -- tens of thousands of files on a long run, for an answer that lives two
  levels up. Pruning happens after a directory's own files are listed, so a
  session's session.json is always seen even though its artifacts/ subtree is
  skipped; only a session.json *nested inside* a pruned directory (where none
  belongs) goes unfound.
  """
  for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [
        d for d in dirnames if d not in _DISCOVERY_PRUNE and not d.startswith(".")
    ]
    if "session.json" in filenames:
      yield Path(dirpath) / "session.json"


def iter_sessions(root: Path | str, include_play: bool = False) -> Iterator[Session]:
  """Yield every rlmcp session under ``root``, newest first by ``started_at``.

  Play sessions are left out unless asked for: they are newer than the run they
  replay and would otherwise become the answer to "the latest session here",
  which is never what somebody looking at a run means. A command whose job is
  to say what exists -- ``rlmcp sessions`` -- passes ``include_play=True``.
  """
  root = Path(root).expanduser()
  if not root.exists():
    return
  found = []
  for session_file in _walk_session_files(root):
    info = _read_json(session_file)
    if isinstance(info, dict) and "schema_version" in info:
      if not include_play and info.get("kind") == PLAY_SESSION_KIND:
        continue
      started = info.get("started_at")
      found.append((
          started if isinstance(started, (int, float)) else 0.0,
          session_file.parent,
      ))
  found.sort(key=lambda item: (-item[0], str(item[1])))
  for _, session_dir in found:
    yield Session(session_dir)
