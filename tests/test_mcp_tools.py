"""Every MCP tool, called at least once.

The MCP server is the product's headline surface -- 34 tools an agent reaches
over stdio -- and nothing tested it. `CLAUDE.md` even claimed "the MCP-server
tests skip unless the optional `mcp` package is installed", describing a file
that did not exist.

The registry is the fixture, as in `test_cli_dispatch`: parametrising over
`list_tools()` means a tool added later is covered the day it is added.

These assert dispatch, not behaviour. The pinned session has no trainer, so the
right answer for a live tool is a truthful refusal -- an `ok: false` payload is
a pass. What must not happen is the tool failing to resolve, or raising the
`NameError`/`AttributeError`/`TypeError` that a half-applied rename leaves.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

pytest.importorskip("mcp", reason="the MCP server needs the optional 'mcp' package")

from rlmcp.server.mcp_server import create_mcp_server

# Arguments that make a tool well-formed. Anything absent takes none.
_ARGS: dict[str, dict[str, Any]] = {
    "switch_session": {"session_dir": "/nonexistent/session"},
    "get_metrics": {"names": ["Train/mean_reward"]},
    "plot_metrics": {"names": ["Train/mean_reward"]},
    "set_parameter": {"key": "reward.foot_slip.weight", "value": -0.2, "why": "test"},
    "get_parameter": {"key": "reward.foot_slip.weight"},
    "run_command": {"command": "get_status"},
    "record_video": {"seconds": 1},
    "diagnose_motion": {"seconds": 1},
    "record_trace": {"seconds": 1},
    "save_checkpoint": {},
    "load_checkpoint": {"path": "/nonexistent/ckpt.pt"},
    "add_note": {"text": "a note"},
    "request_stop": {"why": "test"},
    "curriculum_goto": {"stage": "0_flat"},
    "curriculum_auto": {"enabled": False},
    "new_record": {"slug": "a-slug"},
    "close_record": {"record_id": "001", "verdict": "falsified", "outcome": "x"},
    "show_record": {"record_id": "001"},
    "attach_asset": {"record_id": "001", "path": "/nonexistent/x.png"},
    "compare_runs": {"record_ids": ["001"]},
    "record_feedback": {"text": "they said something"},
    "attach_feedback": {"record_id": "001", "text": "they said something"},
    "answer_feedback": {"record_id": "001", "index": 0, "response": "done"},
    "set_record_headline": {"record_id": "001", "text": "one sentence"},
}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
  """A server pinned at a root with no sessions in it.

  ``records_root`` is pinned too: left unset the record tools would fall back
  to ``./records``, and a test suite must not write a store into whatever
  directory it happened to be run from.
  """
  root = tmp_path_factory.mktemp("no_sessions")
  return create_mcp_server(root=str(root),
                           records_root=str(root / "records"))


@pytest.fixture(scope="module")
def tool_names(server) -> list[str]:
  return sorted(t.name for t in asyncio.run(server.list_tools()))


def test_the_tool_registry_is_not_empty(tool_names):
  """Guards the guard: a refactor that registers nothing must show up here."""
  assert len(tool_names) > 20


def test_the_documented_headline_tools_are_present(tool_names):
  """The README and CLAUDE.md promise these by name."""
  for name in (
      "get_training_status",
      "list_parameters",
      "set_parameter",
      "take_screenshot",
      "record_video",
      "diagnose_motion",
      "run_command",
  ):
    assert name in tool_names, f"'{name}' is documented but not registered"


def _call(server, name: str) -> Any:
  return asyncio.run(server.call_tool(name, _ARGS.get(name, {})))


def _text(result: Any) -> str:
  """The payload an agent actually sees, out of the transport wrapper."""
  content = getattr(result, "content", None) or (
      result[0] if isinstance(result, tuple) else result
  )
  if isinstance(content, list):
    return "\n".join(str(getattr(c, "text", c)) for c in content)
  return str(content)


# The signatures of a half-applied rename, as opposed to a tool that simply
# needs something this environment does not have.
_ROT = (NameError, AttributeError, TypeError)


def _rot_in(exc: BaseException) -> BaseException | None:
  """Find rot anywhere in the cause chain.

  The MCP framework re-raises whatever a tool throws as ``UnexpectedToolError``,
  so catching ``NameError`` at the call site catches nothing. The original is
  kept on ``__cause__``/``__context__``; this walks there. Written after a
  deliberately broken tool sailed past the first version of this test.
  """
  seen = set()
  while exc is not None and id(exc) not in seen:
    seen.add(id(exc))
    if isinstance(exc, _ROT):
      return exc
    exc = exc.__cause__ or exc.__context__
  return None


def test_every_registered_tool_is_callable(server, tool_names):
  """Walk the whole registry: each tool resolves and answers."""
  broken: list[str] = []
  for name in tool_names:
    try:
      _call(server, name)
    except Exception as exc:
      rot = _rot_in(exc)
      if rot is not None:
        broken.append(f"{name}: {rot!r}")
      # Otherwise: needs a live trainer, a real file, or a record that is not
      # there. That is behaviour; the tool resolved and ran.
  assert not broken, "tools broken rather than merely unavailable:\n  " + "\n  ".join(broken)


def test_a_live_tool_refuses_truthfully_when_the_trainer_is_dead(tmp_path):
  """The failure mode that matters: say so, do not hang or lie.

  Pinned at a session that exists but whose process is gone -- the common case
  after a run ends, and the one the CLI answers with ``ok: false`` plus a hint.
  """
  session = tmp_path / "session"
  session.mkdir()
  (session / "status.json").write_text(
      json.dumps({"iteration": 7, "num_envs": 4, "pid": 999999, "updated_at": 0})
  )
  (session / "session.json").write_text(
      json.dumps({"task": "Fake-Task-v0", "num_envs": 4, "started_at": 0})
  )
  dead = create_mcp_server(session_dir=str(session))

  result = asyncio.run(dead.call_tool("take_screenshot", {}))

  payload = json.loads(_text(result))
  assert payload["ok"] is False
  error = payload["error"].lower()
  assert "dead" in error or "not running" in error
  assert "trainer" in error or "training" in error


def test_reading_status_off_disk_still_works_when_the_trainer_is_dead(tmp_path):
  """Data tools fall back to the files; that is the documented contract."""
  session = tmp_path / "session"
  session.mkdir()
  (session / "status.json").write_text(
      json.dumps({"iteration": 11, "num_envs": 4, "pid": 999999, "updated_at": 0})
  )
  (session / "session.json").write_text(
      json.dumps({"task": "Fake-Task-v0", "num_envs": 4, "started_at": 0})
  )
  dead = create_mcp_server(session_dir=str(session))

  result = asyncio.run(dead.call_tool("get_training_status", {}))
  payload = json.loads(_text(result))
  assert payload["iteration"] == 11


def test_an_unknown_tool_is_refused(server):
  """A typo from an agent gets an error, not a silent success."""
  try:
    result = _call(server, "no_such_tool")
  except Exception:
    return
  assert "unknown tool" in _text(result).lower()


def test_the_feedback_tools_round_trip_against_a_real_store(tmp_path):
  """The record tools answer without a trainer: a remark, then what was done
  about it, then the fold across the whole store."""
  from rlmcp.records import open_store

  store = open_store(tmp_path / "records")
  record = store.new_record("steered")
  offline = create_mcp_server(root=str(tmp_path),
                              records_root=str(tmp_path / "records"))

  attached = json.loads(_text(asyncio.run(offline.call_tool(
      "attach_feedback",
      {"record_id": record.id, "text": "Try a smaller step.", "kind": "steer"}))))
  assert attached["ok"] is True
  assert attached["outstanding"] is True

  answered = json.loads(_text(asyncio.run(offline.call_tool(
      "answer_feedback",
      {"record_id": record.id, "index": attached["index"],
       "response": "Halved it.", "changed": True}))))
  assert answered["feedback"]["response"] == "Halved it."

  timeline = json.loads(_text(asyncio.run(offline.call_tool(
      "get_feedback_timeline", {}))))
  assert timeline["count"] == 1
  assert timeline["feedback"][0]["run"] == record.id
  assert json.loads(_text(asyncio.run(offline.call_tool(
      "get_feedback_timeline", {"outstanding": True}))))["count"] == 0


def test_a_headline_set_over_mcp_shows_up_on_the_record(tmp_path):
  from rlmcp.records import open_store

  store = open_store(tmp_path / "records")
  record = store.new_record("summarised", hypothesis="A longer warmup helps.")
  offline = create_mcp_server(root=str(tmp_path),
                              records_root=str(tmp_path / "records"))

  payload = json.loads(_text(asyncio.run(offline.call_tool(
      "set_record_headline",
      {"record_id": record.id, "text": "The entropy floor is the constraint."}))))

  assert payload["derived"] is False
  assert store.get_record(record.id).headline == "The entropy floor is the constraint."


def test_a_feedback_tool_refuses_an_unknown_record_rather_than_raising(tmp_path):
  offline = create_mcp_server(root=str(tmp_path),
                              records_root=str(tmp_path / "records"))

  payload = json.loads(_text(asyncio.run(offline.call_tool(
      "attach_feedback", {"record_id": "999", "text": "into the void"}))))

  assert payload["ok"] is False
  assert "999" in payload["error"]
