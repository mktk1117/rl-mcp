"""Where the CLI looks for a session, and what it says when it finds none.

Discovery is what makes a bare ``rlmcp status`` mean "the run I am working
on". These tests pin the search down: the explicit flags, the environment,
the cwd defaults. The MCP server and the CLI must read the same variables --
an ``RLMCP_ROOT`` that steers one and is ignored by the other turns "works
in this shell" into a property of the directory you happened to start in.
"""

from __future__ import annotations

import json

import pytest

from rlmcp.cli import main

#: Above any Linux pid_max (2**22), so liveness probes always say "gone".
_DEAD_PID = 9_999_999


def make_session(run_dir, *, started_at: float, iteration: int = 7):
  """A finished run's session directory, as the trainer leaves it on disk."""
  session = run_dir / "rlmcp"
  session.mkdir(parents=True)
  (session / "session.json").write_text(json.dumps({
      "schema_version": 1,
      "pid": _DEAD_PID,
      "started_at": started_at,
      "kind": "rlmcp-training-session",
      "task": "Fake-Task-v0",
      "num_envs": 4,
  }))
  (session / "status.json").write_text(json.dumps({
      "iteration": iteration,
      "updated_at": started_at,
  }))
  return session


@pytest.fixture
def elsewhere(tmp_path, monkeypatch):
  """Run from a directory with no sessions anywhere beneath it."""
  cwd = tmp_path / "elsewhere"
  cwd.mkdir()
  monkeypatch.chdir(cwd)
  return cwd


def test_rlmcp_root_env_points_status_at_sessions_elsewhere(
    tmp_path, elsewhere, monkeypatch, capsys):
  """The server reads RLMCP_ROOT; the CLI must honor the same variable."""
  logs = tmp_path / "far-away" / "logs"
  session = make_session(logs / "run-001", started_at=1000.0)
  monkeypatch.setenv("RLMCP_ROOT", str(logs))

  assert main(["--json", "status"]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["session"] == str(session)
  assert payload["iteration"] == 7


def test_rlmcp_root_env_feeds_the_sessions_listing(
    tmp_path, elsewhere, monkeypatch, capsys):
  logs = tmp_path / "far-away" / "logs"
  make_session(logs / "run-001", started_at=1000.0)
  make_session(logs / "run-002", started_at=2000.0)
  monkeypatch.setenv("RLMCP_ROOT", str(logs))

  assert main(["--json", "sessions"]) == 0
  rows = json.loads(capsys.readouterr().out)
  assert [r["started_at"] for r in rows] == [2000.0, 1000.0]


def test_the_root_flag_beats_rlmcp_root(
    tmp_path, elsewhere, monkeypatch, capsys):
  """Explicit scope stays explicit, matching the server's precedence."""
  env_logs = tmp_path / "env-logs"
  flag_logs = tmp_path / "flag-logs"
  make_session(env_logs / "run-env", started_at=2000.0)
  chosen = make_session(flag_logs / "run-flag", started_at=1000.0)
  monkeypatch.setenv("RLMCP_ROOT", str(env_logs))

  assert main(["--json", "--root", str(flag_logs), "status"]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["session"] == str(chosen)


def test_not_found_error_is_free_of_list_reprs(elsewhere, monkeypatch):
  """The advice must be typable: no Python repr, and name the actual cwd."""
  monkeypatch.delenv("RLMCP_ROOT", raising=False)
  monkeypatch.delenv("RLMCP_SESSION", raising=False)

  with pytest.raises(SystemExit) as caught:
    main(["--text", "status"])

  message = str(caught.value)
  assert "['" not in message
  assert str(elsewhere) in message
  assert "RLMCP_ROOT" in message
