"""rlmcp with somebody else's simulator missing.

Most installs have exactly one backend. A user who has mjlab must never see
Genesis or Isaac Sim mentioned as a failure, and the same in every other
direction -- so nothing rlmcp imports at module level may belong to a
simulator, and every capability that needs one has to say so as a *reason*
rather than as a traceback.

The blocker below is the whole trick: a ``meta_path`` finder that raises
``ModuleNotFoundError`` for every simulator reproduces a single-backend install
exactly, on a machine that happens to have several. It runs in a subprocess
because it is process-wide and would otherwise leak into the rest of the suite.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

SIMULATORS = ("mjlab", "genesis", "isaaclab", "isaacsim", "viser", "mjviser")
"""Everything rlmcp can drive or draw with, and nothing it requires."""

BLOCKER = f"""
import sys
HIDDEN = {SIMULATORS!r}


class Blocker:
  def find_spec(self, name, path=None, target=None):
    if name.split(".")[0] in HIDDEN:
      raise ModuleNotFoundError(f"No module named {{name!r}}")
    return None


sys.meta_path.insert(0, Blocker())
"""


def in_a_bare_install(body: str) -> subprocess.CompletedProcess:
  """Run ``body`` with every simulator unimportable."""
  return subprocess.run(
      [sys.executable, "-c", BLOCKER + textwrap.dedent(body)],
      capture_output=True, text=True, cwd=ROOT, timeout=120, check=False,
  )


def backend_packages() -> list[str]:
  """The backend packages that exist, read off the tree rather than listed.

  A backend added without a line here would otherwise be untested for exactly
  the property this file is about.
  """
  adapters = ROOT / "rlmcp" / "adapters"
  families = {"manager_based", "legged_gym_style", "access"}
  return sorted(
      p.name for p in adapters.iterdir()
      if p.is_dir() and (p / "__init__.py").exists()
      and not p.name.startswith("_") and p.name not in families
  )


@pytest.mark.parametrize("package", backend_packages())
def test_a_backend_package_imports_without_its_simulator(package: str):
  """Importing the adapter must not import the thing it adapts.

  This is what lets `rlmcp.adapters.genesis` be importable on an mjlab-only
  machine, which in turn is what lets the conformance table and the CLI talk
  about a backend the user does not have without falling over.
  """
  done = in_a_bare_install(f"""
      import rlmcp.adapters.{package} as backend
      assert callable(backend.wrap)
  """)
  assert done.returncode == 0, done.stderr


def test_the_library_imports_with_no_simulator_at_all():
  done = in_a_bare_install("""
      import rlmcp
      assert rlmcp.__all__
      assert rlmcp.__version__
  """)
  assert done.returncode == 0, done.stderr


def test_the_cli_loads_with_no_simulator_at_all():
  """`rlmcp --help` on a fresh install, before any simulator is chosen."""
  done = in_a_bare_install("""
      import rlmcp.cli
      assert callable(rlmcp.cli.main)
  """)
  assert done.returncode == 0, done.stderr


def test_reading_a_session_does_not_reach_for_a_simulator():
  """`status`, `params`, `metrics` read files a trainer wrote.

  None of them should need the simulator that wrote them -- reading a run from
  a laptop is the case this protects.
  """
  done = in_a_bare_install("""
      import sys
      import rlmcp.cli, rlmcp.session  # noqa: F401
      from rlmcp.core.controller import RlMcp  # noqa: F401
      touched = [m for m in sys.modules if m.split(".")[0] in
                 ("mjlab", "genesis", "isaaclab", "isaacsim")]
      assert not touched, f"imported a simulator just to read: {touched}"
  """)
  assert done.returncode == 0, done.stderr


def test_every_backend_has_a_row_in_the_task_table():
  """`rlmcp tasks` answers "which backends are here" and must know them all.

  Genesis shipped without a row, so a Genesis-only machine was told about the
  two backends it did not have and nothing about the one it did. An empty task
  list and an absent backend are different answers, and the table is where that
  distinction is kept.
  """
  from rlmcp.tasks import BACKENDS

  listed = {row["backend"] for row in BACKENDS}
  missing = sorted(set(backend_packages()) - listed)
  assert not missing, (
      f"backends with no row in rlmcp.tasks.BACKENDS: {missing}. A user whose "
      "only backend is one of these is told about the others and not about "
      "theirs."
  )


def test_an_absent_backend_is_a_reason_not_a_crash():
  """Every backend row answers, and says why when it has nothing to say."""
  done = in_a_bare_install("""
      from rlmcp.tasks import registered
      reply = registered()
      rows = {row["backend"]: row for row in reply["backends"]}
      assert rows, "no backend rows at all"
      for name, row in rows.items():
        assert row["available"] is False, f"{name} claims to be available"
        assert "not installed" in row["reason"], (name, row["reason"])
      assert reply["tasks"] == []
  """)
  assert done.returncode == 0, done.stderr
