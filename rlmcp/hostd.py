"""``rlmcp hostd`` -- one training machine, on the wire.

The daemon a studio talks to when a run is on another box (design.md, "the
client surface is the protocol"). It speaks the host protocol and nothing
else: every session under its root over the seventeen names of
:data:`~rlmcp.session.WIRE_SURFACE`, plus two faces the filesystem never
needed -- **jobs** (start a process here, poll it, stop it, read its log)
and **host** (who this machine is, its GPUs, its disk). It never imports
torch, never touches a trainer's process except to start or stop one, and
if it dies, telemetry delivery stops and training does not -- the same
property rl-mcp's directory protocol has always had, kept on purpose.

Standard library only: ``http.server`` on a thread pool, JSON bodies. A
studio behind a VPN, a rented GPU box, or ``localhost`` for honesty's sake.
Dialling out through a relay is a layer on top; nothing here changes for it.

One bearer token per host (``--token`` or ``$RLMCP_HOST_TOKEN``), checked on
every request. Binding anything but localhost without a token is refused:
this daemon starts processes.

    rlmcp hostd --root logs --port 8740 --token "$(openssl rand -hex 16)"
"""

from __future__ import annotations

import argparse
import contextlib
import hmac
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from rlmcp.session import Session, iter_sessions

VERSION = 1
MAX_WAIT_S = 600.0
LOG_TAIL = 40


class HostError(Exception):
  def __init__(self, status: int, message: str):
    super().__init__(message)
    self.status = status
    self.message = message


# ── jobs: what the host executes ────────────────────────────────────────────


class Job:
  """One process this host started for somebody else."""

  def __init__(self, job_id: str, argv: list[str], cwd: str, label: str, log: Path):
    self.id = job_id
    self.argv = argv
    self.cwd = cwd
    self.label = label
    self.log = log
    self.proc: subprocess.Popen | None = None
    self.submitted_at = time.time()
    self.ended_at: float | None = None
    self.exit_code: int | None = None
    self.error = ""
    self.cancelled = False

  @property
  def state(self) -> str:
    if self.error and self.proc is None:
      return "failed"
    if self.proc is None:
      return "queued"
    code = self.proc.poll()
    if code is None:
      return "running"
    if self.cancelled:
      return "cancelled"
    return "succeeded" if code == 0 else "failed"

  def describe(self, tail: int = 0) -> dict[str, Any]:
    if self.proc is not None and self.exit_code is None and self.proc.poll() is not None:
      self.exit_code = self.proc.returncode
      self.ended_at = self.ended_at or time.time()
    out: dict[str, Any] = {
        "id": self.id, "state": self.state, "label": self.label, "argv": self.argv,
        "cwd": self.cwd, "submitted_at": self.submitted_at, "ended_at": self.ended_at,
        "exit_code": self.exit_code, "error": self.error,
        "pid": self.proc.pid if self.proc else None,
    }
    if tail:
      try:
        out["log"] = self.log.read_text(errors="replace").splitlines()[-tail:]
      except OSError:
        out["log"] = []
    return out


class Jobs:
  def __init__(self, log_root: Path):
    self.log_root = log_root
    self._jobs: dict[str, Job] = {}
    self._lock = threading.Lock()

  def submit(self, argv: list[str], cwd: str, env: dict[str, str], label: str) -> Job:
    if not argv or not all(isinstance(a, str) for a in argv):
      raise HostError(400, "argv must be a non-empty list of strings")
    job_id = f"job-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    self.log_root.mkdir(parents=True, exist_ok=True)
    job = Job(job_id, argv, cwd, label, self.log_root / f"{job_id}.log")
    with self._lock:
      self._jobs[job_id] = job
    try:
      with open(job.log, "wb") as handle:
        job.proc = subprocess.Popen(
            argv, cwd=cwd or None, env={**os.environ, **env},
            stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True)  # its own group: stopping it never signals hostd
    except OSError as exc:
      # Either the log could not be opened or the process could not start;
      # both are the job's failure to report, never the daemon's to raise.
      job.error = f"could not start it: {exc}"
    return job

  def get(self, job_id: str) -> Job:
    job = self._jobs.get(job_id)
    if job is None:
      raise HostError(404, f"no job {job_id}")
    return job

  def cancel(self, job_id: str) -> Job:
    job = self.get(job_id)
    if job.proc is not None and job.proc.poll() is None:
      job.cancelled = True
      with contextlib.suppress(ProcessLookupError):
        os.killpg(job.proc.pid, signal.SIGTERM)
    return job

  def all(self) -> list[Job]:
    return list(self._jobs.values())


# ── the host ────────────────────────────────────────────────────────────────


def host_id(root: Path) -> str:
  """Stable per machine and root: kept beside the sessions, minted once."""
  marker = root / ".rlmcp-host-id"
  try:
    current = marker.read_text().strip()
    if current:
      return current
  except OSError:
    pass
  minted = uuid.uuid4().hex[:12]
  try:
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(minted + "\n")
  except OSError:
    pass
  return minted


def gpus() -> list[dict[str, Any]]:
  if shutil.which("nvidia-smi") is None:
    return []
  try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
        check=False)
  except (OSError, subprocess.TimeoutExpired):
    return []
  rows = []
  for line in out.stdout.splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 5:
      rows.append({"index": int(parts[0]), "name": parts[1], "memory_used_mb": int(parts[2]),
                   "memory_total_mb": int(parts[3]), "utilization": int(parts[4])})
  return rows


class Host:
  """Everything the daemon knows, behind the HTTP layer so a test can call it."""

  def __init__(self, root: Path, token: str = "", log_root: Path | None = None):
    self.root = Path(root).expanduser().resolve()
    self.token = token
    self.id = host_id(self.root)
    self.started_at = time.time()
    self.jobs = Jobs(log_root or (self.root / ".rlmcp-hostd" / "jobs"))

  def describe(self) -> dict[str, Any]:
    try:
      usage = shutil.disk_usage(self.root)
      disk = {"free_gb": round(usage.free / 1e9, 1), "total_gb": round(usage.total / 1e9, 1)}
    except OSError:
      disk = {}
    return {"v": VERSION, "host_id": self.id, "root": str(self.root),
            "started_at": self.started_at, "gpus": gpus(), "disk": disk,
            "sessions": sum(1 for _ in iter_sessions(self.root))}

  def sessions(self, include_play: bool = False) -> list[dict[str, Any]]:
    rows = []
    for session in iter_sessions(self.root, include_play=include_play):
      info = session.info()
      live = session.liveness_info()
      rows.append({"key": session.key, "name": session.name, "task": info.get("task"),
                   "started_at": info.get("started_at"), "state": live["state"],
                   "iteration": session.status().get("iteration")})
    return rows

  def session(self, key: str) -> Session:
    # A key is `<run>/<session>`: the last two path segments of a session
    # directory. Resolved by walking what exists, never by joining the key
    # onto the root -- a key is a name, not a path, and must not become one.
    for session in iter_sessions(self.root, include_play=True):
      if session.key == key:
        return session
    raise HostError(404, f"no session {key} under {self.root}")


# ── HTTP ────────────────────────────────────────────────────────────────────


def _handler(host: Host):
  class Handler(BaseHTTPRequestHandler):
    server_version = f"rlmcp-hostd/{VERSION}"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet by default; stderr is the trainer's
      if os.environ.get("RLMCP_HOSTD_LOG"):
        super().log_message(fmt, *args)

    # -- plumbing --

    def _authorized(self) -> bool:
      if not host.token:
        return True
      given = self.headers.get("authorization", "")
      return given.startswith("Bearer ") and hmac.compare_digest(given[7:], host.token)

    def _send(self, status: int, body: Any = None, raw: bytes | None = None,
              content_type: str = "application/json") -> None:
      data = raw if raw is not None else json.dumps(body).encode()
      self.send_response(status)
      self.send_header("content-type", content_type)
      self.send_header("content-length", str(len(data)))
      self.end_headers()
      self.wfile.write(data)

    def _body(self) -> dict[str, Any]:
      length = int(self.headers.get("content-length") or 0)
      if not length:
        return {}
      try:
        payload = json.loads(self.rfile.read(length).decode())
      except (ValueError, UnicodeDecodeError):
        raise HostError(400, "the body is not JSON") from None
      return payload if isinstance(payload, dict) else {}

    def _route(self, method: str) -> None:
      if not self._authorized():
        self._send(401, {"error": "a bearer token is required"})
        return
      url = urllib.parse.urlsplit(self.path)
      parts = [urllib.parse.unquote(p) for p in url.path.strip("/").split("/") if p]
      query = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query).items()}
      try:
        self._dispatch(method, parts[1:], query) if parts and parts[0] == "v1" \
            else self._send(404, {"error": "no such route"})
      except HostError as exc:
        self._send(exc.status, {"error": exc.message})
      except (FileNotFoundError, ValueError) as exc:
        self._send(404 if isinstance(exc, FileNotFoundError) else 400, {"error": str(exc)})
      except Exception as exc:
        self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_GET(self):
      self._route("GET")

    def do_POST(self):
      self._route("POST")

    # -- routes --

    def _dispatch(self, method: str, parts: list[str], q: dict[str, str]) -> None:
      if parts == ["host"] and method == "GET":
        return self._send(200, host.describe())
      if parts == ["sessions"] and method == "GET":
        return self._send(200, host.sessions(include_play=q.get("include_play") in ("1", "true")))
      if len(parts) >= 3 and parts[0] == "sessions":
        # `<run>/<session>` is one segment when quoted, two when not.
        if len(parts) >= 4 and parts[1] and "/" not in parts[1]:
          key, rest = parts[1] + "/" + parts[2], parts[3:]
        else:
          key, rest = parts[1], parts[2:]
        return self._session(method, host.session(key), rest, q)
      if parts == ["jobs"]:
        if method == "GET":
          return self._send(200, [j.describe() for j in host.jobs.all()])
        body = self._body()
        job = host.jobs.submit(body.get("argv") or [], str(body.get("cwd") or ""),
                               {str(k): str(v) for k, v in (body.get("env") or {}).items()},
                               str(body.get("label") or ""))
        return self._send(200 if job.state != "failed" else 500, job.describe(tail=LOG_TAIL))
      if len(parts) == 2 and parts[0] == "jobs" and method == "GET":
        tail = int(q.get("tail") or LOG_TAIL)
        return self._send(200, host.jobs.get(parts[1]).describe(tail=tail))
      if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel" and method == "POST":
        return self._send(200, host.jobs.cancel(parts[1]).describe(tail=LOG_TAIL))
      raise HostError(404, "no such route")

    def _session(self, method: str, session: Session, rest: list[str], q: dict[str, str]) -> None:
      what = rest[0] if rest else ""
      if method == "GET":
        if what == "info":
          return self._send(200, session.info())
        if what == "status":
          return self._send(200, session.status())
        if what == "params":
          return self._send(200, session.params())
        if what == "liveness_info":
          return self._send(200, session.liveness_info())
        if what == "metrics_count":
          return self._send(200, {"count": session.metrics_count()})
        if what in ("metrics", "events"):
          last_n = int(q["last_n"]) if q.get("last_n") else None
          since = int(q["since_seq"]) if q.get("since_seq") else None
          rows = (session.metrics if what == "metrics" else session.events)(
              last_n=last_n, since_seq=since)
          return self._send(200, rows)
        if what == "artifacts" and len(rest) == 1:
          # `path` is this transport's business, not the wire's.
          return self._send(200, [{k: v for k, v in row.items() if k != "path"}
                                  for row in session.list_artifacts()])
        if what == "artifacts" and len(rest) == 2:
          data = session.read_artifact(rest[1])
          return self._send(200, raw=data, content_type="application/octet-stream")
        if what == "poll":
          resp = session.poll(q.get("req_id", ""), consume=q.get("consume") in ("1", "true"))
          return self._send(200, resp.to_dict() if resp else None)
        if what == "wait":
          timeout = min(float(q.get("timeout") or 120.0), MAX_WAIT_S)
          return self._send(200, session.wait(q.get("req_id", ""), timeout=timeout).to_dict())
      if method == "POST" and what == "submit":
        body = self._body()
        cmd = body.get("cmd")
        if not isinstance(cmd, str) or not cmd:
          raise HostError(400, "submit needs a `cmd`")
        args = body.get("args") or {}
        if not isinstance(args, dict):
          raise HostError(400, "`args` must be an object")
        return self._send(200, session.submit(cmd, **args).to_dict())
      raise HostError(404, f"no such session route: {method} {'/'.join(rest)}")

  return Handler


class Server:
  """The daemon as an object: start it in a thread for a test, or block on it."""

  def __init__(self, host: Host, bind: str = "127.0.0.1", port: int = 8740):
    if bind not in ("127.0.0.1", "localhost", "::1") and not host.token:
      raise ValueError("refusing to bind beyond localhost without --token: hostd starts processes")
    self.host = host
    self._server = ThreadingHTTPServer((bind, port), _handler(host))
    self._server.daemon_threads = True
    self._thread: threading.Thread | None = None

  @property
  def url(self) -> str:
    bind, port = self._server.server_address[:2]
    return f"http://{bind}:{port}"

  def start(self) -> Server:
    self._thread = threading.Thread(target=self._server.serve_forever, name="rlmcp-hostd",
                                    daemon=True)
    self._thread.start()
    return self

  def stop(self) -> None:
    self._server.shutdown()
    self._server.server_close()

  def serve_forever(self) -> None:
    try:
      self._server.serve_forever()
    finally:
      self._server.server_close()


def add_arguments(p: argparse.ArgumentParser) -> None:
  p.add_argument("--root", default=os.environ.get("RLMCP_ROOT") or "logs",
                 help="Where this machine keeps its sessions (default: logs, or $RLMCP_ROOT)")
  p.add_argument("--bind", default="127.0.0.1", help="Address to listen on (default: localhost)")
  p.add_argument("--port", type=int, default=8740)
  p.add_argument("--token", default=os.environ.get("RLMCP_HOST_TOKEN", ""),
                 help="Bearer token every request must carry ($RLMCP_HOST_TOKEN). "
                      "Required to bind beyond localhost.")


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(prog="rlmcp hostd", description=__doc__.split("\n\n")[0])
  add_arguments(parser)
  args = parser.parse_args(argv)
  host = Host(Path(args.root), token=args.token)
  try:
    server = Server(host, bind=args.bind, port=args.port)
  except (ValueError, OSError) as exc:
    print(f"[rlmcp hostd] {exc}", file=sys.stderr)
    return 2
  print(f"rlmcp hostd  {server.url}  host {host.id}  root {host.root}"
        + ("" if args.token else "  (no token: localhost only)"), flush=True)
  with contextlib.suppress(KeyboardInterrupt):
    server.serve_forever()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
