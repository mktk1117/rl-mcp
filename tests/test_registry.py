"""The registry: best-effort writes, distrustful reads, no maintenance.

The two contracts worth pinning are the ones the rest of the system leans on:
``register`` may fail but must never raise (a broken state dir cannot take a
training run down), and ``entries`` prunes whatever no longer points at
anything (nobody will ever run a cleanup command).
"""

from __future__ import annotations

import json
import os

from rlmcp import registry
from rlmcp.session import Session

#: Above any Linux pid_max (2**22), so liveness probes always say "gone".
_DEAD_PID = 9_999_999


def make_session_dir(run_dir):
  session = run_dir / "rlmcp"
  session.mkdir(parents=True)
  (session / "session.json").write_text(json.dumps({
      "schema_version": 1, "pid": _DEAD_PID, "started_at": 1000.0,
  }))
  return session


def write_entry(name, payload):
  directory = registry.state_dir()
  directory.mkdir(parents=True, exist_ok=True)
  (directory / name).write_text(json.dumps(payload))


def test_register_and_read_back(tmp_path):
  session = make_session_dir(tmp_path / "run-001")

  path = registry.register(registry.KIND_TRAINER, session_dir=session)

  assert path is not None and path.exists()
  rows = registry.entries()
  assert len(rows) == 1
  row = rows[0]
  assert row["kind"] == "trainer"
  assert row["session_dir"] == str(session)
  assert row["pid"] == os.getpid()
  assert row["pid_alive"] is True
  assert row["session_exists"] is True


def test_seen_entries_are_keyed_by_path_not_pid(tmp_path):
  """A monitor polling status every few seconds must overwrite one entry,
  not mint one per invocation and churn real registrations out of the cap."""
  session = make_session_dir(tmp_path / "run-001")

  first = registry.register(registry.KIND_SEEN, session_dir=session)
  second = registry.register(registry.KIND_SEEN, session_dir=session)

  assert first == second
  assert len(registry.entries()) == 1


def test_entries_prune_what_points_at_nothing(tmp_path):
  gone = tmp_path / "deleted-run" / "rlmcp"
  write_entry("trainer-1.json", {
      "kind": "trainer", "pid": _DEAD_PID, "registered_at": 1000.0,
      "session_dir": str(gone),
  })
  write_entry("garbage.json", {"not": "an entry"})

  assert registry.entries() == []
  assert list(registry.state_dir().glob("*.json")) == []


def test_a_dead_pid_with_a_real_session_survives(tmp_path):
  """A finished run is still worth finding; the pid is provenance, not value."""
  session = make_session_dir(tmp_path / "run-001")
  write_entry("trainer-2.json", {
      "kind": "trainer", "pid": _DEAD_PID, "registered_at": 1000.0,
      "session_dir": str(session),
  })

  rows = registry.entries()
  assert len(rows) == 1
  assert rows[0]["pid_alive"] is False
  assert rows[0]["session_exists"] is True


def test_the_cap_bounds_the_file_count(tmp_path):
  session = make_session_dir(tmp_path / "run-001")
  for index in range(registry.KEEP_NEWEST + 8):
    write_entry(f"trainer-{index}.json", {
        "kind": "trainer", "pid": os.getpid(),
        "registered_at": float(index), "session_dir": str(session),
    })

  assert len(registry.entries()) == registry.KEEP_NEWEST
  assert len(list(registry.state_dir().glob("*.json"))) == registry.KEEP_NEWEST


def test_register_never_raises(tmp_path, monkeypatch):
  """An unwritable state dir costs the listing, never the caller."""
  blocker = tmp_path / "not-a-directory"
  blocker.write_text("a file where a directory must go")
  monkeypatch.setenv("XDG_STATE_HOME", str(blocker))

  assert registry.register(registry.KIND_TRAINER,
                           session_dir=tmp_path) is None


def test_session_create_registers_the_run(tmp_path):
  session = Session(tmp_path / "run-001" / "rlmcp").create({
      "kind": "rlmcp-training-session", "task": "Fake-Task-v0",
  })

  rows = registry.entries()
  assert [r["session_dir"] for r in rows] == [str(session.dir)]
  assert rows[0]["session_kind"] == "rlmcp-training-session"


def test_serve_registers_root_and_pin(tmp_path, monkeypatch):
  """`rlmcp serve` announces itself even before any tool call arrives."""
  from rlmcp.server import mcp_server

  class FakeServer:
    def __init__(self, name):
      self.name = name

    def tool(self):
      return lambda fn: fn

    def resource(self, uri):
      return lambda fn: fn

    def run(self):
      return None

  monkeypatch.setattr(mcp_server, "MCPServer", FakeServer)
  session = make_session_dir(tmp_path / "logs" / "run-001")

  assert mcp_server.main(["--root", str(tmp_path / "logs")]) == 0

  rows = [r for r in registry.entries() if r["kind"] == "server"]
  assert len(rows) == 1
  assert rows[0]["root"] == str(tmp_path / "logs")
  assert rows[0]["session_dir"] == str(session)
