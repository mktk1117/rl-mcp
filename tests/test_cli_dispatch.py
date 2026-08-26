"""Every CLI subcommand, dispatched at least once.

The suite tests the layers a command calls, but nothing walked the command
table itself. That is where a rename hides: `args.lab_root` survives a
`--lab-root` -> `--records-root` rename, `compileall` is happy, and the flag
only explodes when somebody runs it.

So the table is the fixture. `test_every_subcommand_dispatches` is
parametrised over `build_parser()`, which means a subcommand added later is
covered the day it is added, and one whose handler stops resolving fails here
rather than in a shell.

These assert dispatch, not behaviour: against a session with no trainer the
right answer is a truthful refusal, so a non-zero exit is a pass. What must not
happen is `NameError`, `AttributeError` or `TypeError` -- the signatures of a
half-applied rename.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import pytest

from rlmcp.cli import build_parser, main

# Arguments that make a subcommand well-formed. Anything absent needs none.
_REQUIRED_ARGS: Dict[str, List[str]] = {
    "get": ["reward.foot_slip.weight"],
    "set": ["reward.foot_slip.weight", "-0.2"],
    "run": ["get_status"],
    "raw": ["get_status"],
    "note": ["a note"],
    "load": ["/nonexistent/checkpoint.pt"],
    "analyze": ["/nonexistent/trace.npz"],
    "plot-trace": [],
    "record": ["list"],
}

# Launchers: `main` hands the rest of the line to another program before
# argparse sees it, so calling them here would start a trainer or block on
# stdio. Their interception is tested separately, below.
_LAUNCHERS = {"train", "serve"}


def subcommands() -> List[str]:
  sub = next(
      a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction)
  )
  return sorted(sub.choices)


SUBCOMMANDS = [c for c in subcommands() if c not in _LAUNCHERS]


@pytest.fixture
def dead_session(tmp_path):
  """A session directory whose trainer is not running."""
  session = tmp_path / "session"
  session.mkdir()
  (session / "status.json").write_text(
      json.dumps({"iteration": 7, "num_envs": 4, "pid": 999999, "updated_at": 0})
  )
  (session / "session.json").write_text(
      json.dumps({"task": "Fake-Task-v0", "num_envs": 4, "started_at": 0})
  )
  (session / "metrics.jsonl").write_text(
      json.dumps({"iteration": 7, "metrics": {"Train/mean_reward": 1.0}}) + "\n"
  )
  return session


def test_the_command_table_is_not_empty():
  """Guards the guard: a parser rewrite that yields nothing must show up here."""
  assert len(SUBCOMMANDS) > 15


@pytest.mark.parametrize("command", SUBCOMMANDS, ids=SUBCOMMANDS)
def test_every_subcommand_dispatches(command, dead_session, tmp_path, monkeypatch):
  """The command parses, reaches its handler, and returns rather than raising."""
  monkeypatch.setenv("RLMCP_RECORDS", str(tmp_path / "records"))
  monkeypatch.setenv("RLMCP_OPEN", "never")
  argv = [
      "--session", str(dead_session),
      "--timeout", "1",
      command,
      *_REQUIRED_ARGS.get(command, []),
  ]

  try:
    result = main(argv)
  except SystemExit as exc:            # argparse's own exit is a real answer
    assert exc.code is None or isinstance(exc.code, int)
    return
  except (NameError, AttributeError, TypeError) as exc:
    pytest.fail(f"'{command}' is broken, not merely unavailable: {exc!r}")
  except Exception:
    # A command that needs a live trainer, a real checkpoint or a file that is
    # not there may raise its own error type. That is behaviour, not rot.
    return

  assert isinstance(result, int)


@pytest.mark.parametrize("command", sorted(_LAUNCHERS))
def test_launchers_hand_off_without_argparse_touching_their_flags(command, monkeypatch):
  """`rlmcp train --num-envs 4` must reach the trainer, flags intact."""
  seen: Dict[str, Any] = {}

  def fake_main(argv=None):
    seen["argv"] = argv
    return 0

  target = {
      "train": "rlmcp.train.main",
      "serve": "rlmcp.server.mcp_server.main",
  }[command]
  module_path, attribute = target.rsplit(".", 1)
  monkeypatch.setattr(f"{module_path}.{attribute}", fake_main, raising=True)

  assert main([command, "--num-envs", "4", "--unknown-to-us"]) == 0
  assert seen["argv"] == ["--num-envs", "4", "--unknown-to-us"]


def test_an_unknown_subcommand_is_refused():
  with pytest.raises(SystemExit):
    main(["no-such-command"])


def test_record_subcommands_all_dispatch(tmp_path, monkeypatch):
  """The record group has its own table; walk that too."""
  monkeypatch.setenv("RLMCP_OPEN", "never")
  root = tmp_path / "records"

  sub = next(
      a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction)
  )
  record_parser = sub.choices["record"]
  record_sub = next(
      a for a in record_parser._actions if isinstance(a, argparse._SubParsersAction)
  )

  args_for = {
      "new": ["a-slug"],
      "show": ["001"],
      "close": ["001", "falsified", "--outcome", "x"],
      "asset": ["001", str(tmp_path / "nope.png")],
      "compare": ["001"],
      "import": [str(tmp_path / "other")],
      "claim": ["001"],
      "release": ["001"],
  }

  assert len(record_sub.choices) > 5
  for name in sorted(record_sub.choices):
    argv = ["record", "--records-root", str(root), name, *args_for.get(name, [])]
    try:
      main(argv)
    except SystemExit:
      pass
    except (NameError, AttributeError, TypeError) as exc:
      pytest.fail(f"'record {name}' is broken, not merely unavailable: {exc!r}")
    except Exception:
      pass
