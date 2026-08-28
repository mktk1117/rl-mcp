"""A per-user notebook of where rlmcp things are, so discovery survives cwd.

Sessions are found by searching the current directory, which makes "rlmcp
status shows nothing" a property of the shell you typed it in rather than of
the machine. This module is the other half of discovery: a directory of small
JSON files under ``$XDG_STATE_HOME/rlmcp`` (one per registrant) that running
pieces write and the CLI reads, so a bare command can find the run -- or the
MCP server watching it -- from anywhere.

Who writes:

* the training process, in :meth:`rlmcp.session.Session.create`;
* ``rlmcp serve`` / ``rlmcp-server``, with its root at startup and again with
  the session it pins;
* the CLI, with whatever session a bare command resolved -- a run looked at
  once stays findable, like an editor's recent-files list.

Every write is best-effort and every read distrusts what it finds. A failed
registration must never take training down, so :func:`register` swallows
everything and returns ``None``. A stale entry must never outlive its
usefulness, so :func:`entries` deletes what no longer points at anything --
no daemon, no lock, no maintenance command. Files are written atomically,
one per registrant, in the same file discipline as the session directory.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

#: Registrant kinds. "seen" is the CLI's: not a process announcing itself but
#: a resolution being remembered, which is why its file is keyed by the
#: session's path rather than by a pid -- a monitor polling ``rlmcp status``
#: every few seconds must overwrite one entry, not mint one per invocation.
KIND_TRAINER = "trainer"
KIND_SERVER = "server"
KIND_SEEN = "seen"

#: Newest files kept at read time. The registry is a notebook, not a database;
#: whatever a machine is really running fits in far fewer.
KEEP_NEWEST = 64


def state_dir() -> Path:
  """``$XDG_STATE_HOME/rlmcp``, defaulting to ``~/.local/state/rlmcp``."""
  base = os.environ.get("XDG_STATE_HOME")
  root = Path(base).expanduser() if base else Path.home() / ".local" / "state"
  return root / "rlmcp"


def register(
    kind: str,
    *,
    session_dir: Path | str | None = None,
    root: Path | str | None = None,
    session_kind: str | None = None,
) -> Path | None:
  """Record that ``kind`` exists and what it is looking at. Never raises.

  Returns the file written, or ``None`` when it could not be -- an unwritable
  state directory costs the machine-wide listing, not the training run.
  """
  try:
    directory = state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "pid": os.getpid(),
        "registered_at": time.time(),
    }
    if session_dir is not None:
      payload["session_dir"] = str(Path(session_dir).expanduser().resolve())
    if root is not None:
      payload["root"] = str(Path(root).expanduser().resolve())
    if session_kind:
      payload["session_kind"] = session_kind
    if kind == KIND_SEEN:
      slot = hashlib.sha1(str(payload.get("session_dir")).encode()).hexdigest()[:12]
    else:
      slot = str(os.getpid())
    path = directory / f"{kind}-{slot}.json"
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)
    return path
  except Exception:
    return None


def _pid_alive(pid: Any) -> bool:
  if not isinstance(pid, int) or pid <= 0:
    return False
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    return True
  except OSError:
    return False
  return True


def entries() -> list[dict[str, Any]]:
  """Every registration, newest first, pruned as it is read. Never raises.

  An entry earns its keep by still pointing at something: a session directory
  that exists, a root that exists, or a pid that is still running. Anything
  else is deleted on sight, as is everything beyond the newest
  :data:`KEEP_NEWEST` files. Each returned row carries the stored payload
  plus three judgments made now: ``pid_alive``, ``session_exists`` and
  ``root_exists``.
  """
  try:
    files = sorted(
        state_dir().glob("*.json"),
        key=lambda p: -(p.stat().st_mtime if p.exists() else 0.0),
    )
  except OSError:
    return []
  out: list[dict[str, Any]] = []
  for index, path in enumerate(files):
    try:
      payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
      payload = None
    if not isinstance(payload, dict):
      _discard(path)
      continue
    session_dir = payload.get("session_dir")
    root = payload.get("root")
    session_exists = (
        isinstance(session_dir, str)
        and (Path(session_dir) / "session.json").exists()
    )
    root_exists = isinstance(root, str) and Path(root).is_dir()
    alive = _pid_alive(payload.get("pid"))
    if index >= KEEP_NEWEST or not (session_exists or root_exists or alive):
      _discard(path)
      continue
    out.append({
        **payload,
        "pid_alive": alive,
        "session_exists": session_exists,
        "root_exists": root_exists,
    })
  out.sort(key=lambda row: -(row.get("registered_at") or 0.0))
  return out


def _discard(path: Path) -> None:
  with contextlib.suppress(OSError):
    path.unlink()
