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

import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

SCHEMA_VERSION = 1

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


def _tail_lines(path: Path, last_n: int) -> List[str]:
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


def read_jsonl(path: Path, last_n: Optional[int] = None) -> List[Any]:
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
    lines = path.read_text().splitlines() if last_n is None else _tail_lines(path, last_n)
  except OSError:
    return []
  out: List[Any] = []
  for line in lines:
    line = line.strip()
    if not line:
      continue
    try:
      out.append(json.loads(line))
    except json.JSONDecodeError:
      continue
  return out


@dataclass
class Request:
  """A command from the agent side to the training loop."""

  cmd: str
  args: Dict[str, Any] = field(default_factory=dict)
  req_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
  created_at: float = field(default_factory=time.time)

  def to_dict(self) -> Dict[str, Any]:
    return {
        "req_id": self.req_id,
        "cmd": self.cmd,
        "args": self.args,
        "created_at": self.created_at,
    }

  @staticmethod
  def from_dict(d: Dict[str, Any]) -> "Request":
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
  error: Optional[str] = None
  finished_at: float = field(default_factory=time.time)

  def to_dict(self) -> Dict[str, Any]:
    return {
        "req_id": self.req_id,
        "ok": self.ok,
        "result": self.result,
        "error": self.error,
        "finished_at": self.finished_at,
    }

  @staticmethod
  def from_dict(d: Dict[str, Any]) -> "Response":
    return Response(
        req_id=d["req_id"],
        ok=d["ok"],
        result=d.get("result"),
        error=d.get("error"),
        finished_at=d.get("finished_at", time.time()),
    )


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
    self._cached_pid: Optional[int] = None
    self._last_outbox_prune = 0.0

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

  def create(self, info: Dict[str, Any]) -> "Session":
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
  def open(session_dir: Path | str) -> "Session":
    s = Session(session_dir)
    if not s.session_file.exists():
      raise FileNotFoundError(
          f"No rlmcp session at '{s.dir}' (missing session.json). "
          "Pass the directory printed by the trainer at startup."
      )
    return s

  @staticmethod
  def find_latest(root: Path | str) -> Optional["Session"]:
    """Return the most recently started session under ``root``, if any."""
    return next(iter_sessions(root), None)

  # Trainer side.

  def publish_status(self, status: Dict[str, Any]) -> None:
    _atomic_write_json(self.status_file, {"updated_at": time.time(), **status})
    # Piggyback outbox hygiene on the heartbeat: responses are deleted by the
    # waiter that consumes them, so anything old enough to prune belongs to a
    # client that vanished. Throttled -- the pause loop publishes ~7x/s.
    now = time.time()
    if now - self._last_outbox_prune >= self.OUTBOX_PRUNE_INTERVAL_S:
      self._last_outbox_prune = now
      self.prune_outbox(self.OUTBOX_KEEP_S)

  def publish_params(self, schema: Dict[str, Any]) -> None:
    _atomic_write_json(self.params_file, schema)

  def publish_env_terms(self, terms: Dict[str, Any]) -> None:
    _atomic_write_json(self.env_terms_file, terms)

  def env_terms(self) -> Dict[str, Any]:
    return _read_json(self.env_terms_file, {}) or {}

  def append_metrics(self, iteration: int, metrics: Dict[str, float]) -> None:
    _append_jsonl(self.metrics_file, {"iteration": iteration, "t": time.time(), **metrics})

  def append_event(self, kind: str, detail: Dict[str, Any]) -> None:
    _append_jsonl(self.events_file, {"t": time.time(), "kind": kind, **detail})

  def pop_requests(self, max_age_s: Optional[float] = None) -> List[Request]:
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
    requests: List[Request] = []
    for path in files:
      claimed = path.with_suffix(".claimed")
      try:
        os.replace(path, claimed)
      except OSError:
        continue  # Someone else got it.
      payload = _read_json(claimed)
      try:
        claimed.unlink()
      except OSError:
        pass
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
      self, req_id: str, cmd: Optional[str], error: str, detail: Dict[str, Any]
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

  def info(self) -> Dict[str, Any]:
    return _read_json(self.session_file, {}) or {}

  def status(self) -> Dict[str, Any]:
    return _read_json(self.status_file, {}) or {}

  def params(self) -> Dict[str, Any]:
    return _read_json(self.params_file, {}) or {}

  def metrics(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
    return read_jsonl(self.metrics_file, last_n=last_n)

  def events(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
    return read_jsonl(self.events_file, last_n=last_n)

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

  def poll(self, req_id: str, consume: bool = False) -> Optional[Response]:
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
      try:
        path.unlink()
      except OSError:
        pass
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

  def seconds_since_heartbeat(self) -> Optional[float]:
    updated = self.status().get("updated_at")
    if not isinstance(updated, (int, float)):
      return None
    return max(0.0, time.time() - float(updated))

  def liveness(self) -> str:
    """``"running"``, ``"stalled"`` or ``"dead"`` -- see :meth:`liveness_info`."""
    return self.liveness_info()["state"]

  def liveness_info(self) -> Dict[str, Any]:
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
    out: Dict[str, Any] = {
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
