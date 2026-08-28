"""The surface the docs promise: exported names, console scripts, version.

Every check here failed at least once in a real install. They are cheap, they
need no simulator, and each one pins a promise the README makes to a reader who
cannot see the source: that a name in an example imports, that a command in a
setup line exists, and that the version a bug report quotes is the installed one.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

import rlmcp

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _pyproject_text() -> str:
  return PYPROJECT.read_text()


def _console_scripts() -> dict:
  """Parse ``[project.scripts]`` without tomllib, which is 3.11+."""
  block = re.search(r"^\[project\.scripts\]\n(.*?)(?=^\[|\Z)",
                    _pyproject_text(), re.S | re.M)
  assert block, "pyproject.toml has no [project.scripts] table"
  entries = {}
  for line in block.group(1).splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
      continue
    name, _, target = line.partition("=")
    entries[name.strip()] = target.strip().strip('"')
  return entries


@pytest.mark.parametrize("name", sorted(rlmcp.__all__))
def test_every_exported_name_actually_imports(name):
  """``__all__`` is built from a lazy map, so a typo there is invisible until use."""
  assert getattr(rlmcp, name) is not None


@pytest.mark.parametrize("name", ["CurriculumStage", "StageSchedule", "Condition", "Action"])
def test_the_curriculum_vocabulary_is_importable_from_the_top_level(name):
  """The README's stage example uses all four bare.

  `Condition` and `Action` were missing, so copying that example raised
  ImportError on the reader's first `promote_when`.
  """
  module = importlib.import_module("rlmcp")
  assert getattr(module, name).__name__ == name


@pytest.mark.parametrize("script", ["rlmcp", "rlmcp-train", "rlmcp-server"])
def test_documented_console_scripts_are_declared(script):
  assert script in _console_scripts()


@pytest.mark.parametrize("script,target", sorted(_console_scripts().items()))
def test_every_console_script_points_at_a_callable(script, target):
  """A declared entry point that does not resolve fails at the user's shell.

  `rlmcp-server` in particular is registered into an MCP client config, where a
  broken target surfaces later, in another process, as a server that will not
  start.
  """
  module_path, _, attribute = target.partition(":")
  module = importlib.import_module(module_path)
  assert callable(getattr(module, attribute))


def test_version_matches_the_installed_distribution():
  """One source of truth. A literal here drifted from pyproject.toml before."""
  import importlib.metadata as metadata

  try:
    installed = metadata.version("rl-mcp")
  except metadata.PackageNotFoundError:
    pytest.skip("rl-mcp is not installed in this environment")
  assert rlmcp.__version__ == installed
  assert re.search(r'^version = "%s"$' % re.escape(installed),
                   _pyproject_text(), re.M), (
      "the installed distribution disagrees with pyproject.toml; "
      "reinstall, or the version was bumped without a release"
  )


def test_version_falls_back_honestly_when_not_installed(monkeypatch):
  """An uninstalled source tree must say so, not report a stale number."""
  import importlib.metadata as metadata

  def _not_installed(name):
    raise metadata.PackageNotFoundError(name)

  monkeypatch.setattr(metadata, "version", _not_installed)
  try:
    assert importlib.reload(rlmcp).__version__ == "0.0.0+unknown"
  finally:
    monkeypatch.undo()
    importlib.reload(rlmcp)


@pytest.mark.parametrize("argv0,expected", [
    ("/usr/local/bin/rlmcp-train", "rlmcp-train"),
    ("/usr/local/bin/rlmcp", "rlmcp train"),
    ("", "rlmcp train"),
])
def test_the_trainer_names_itself_after_the_command_that_was_typed(
    argv0, expected, monkeypatch):
  """One code path, two documented names; the usage line must not name the other."""
  from rlmcp import train

  monkeypatch.setattr("sys.argv", [argv0])
  assert train._prog_name("rlmcp train") == expected


def test_wrap_and_the_exception_it_raises_are_exported_together():
  """`wrap` tells the training entrypoint to catch `TrainingStopped`.

  Exporting one without the other means a script can call the thing at the top
  level and then has to reach into `rlmcp.adapters.<backend>` to catch what it
  raises. Found writing the Go1 example, which did exactly what the docstring
  says and failed on `rlmcp.TrainingStopped`.
  """
  assert issubclass(rlmcp.TrainingStopped, Exception)
  assert issubclass(rlmcp.TrainingStopped, rlmcp.SessionStopped)
