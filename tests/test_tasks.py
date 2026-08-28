"""Listing what could run, with no simulator installed.

``rlmcp tasks`` exists to be asked before anything has run, so the interesting
cases are the empty and broken ones: no backend, a package that will not
import, a task that appeared because of an import rather than because the
backend shipped it. None of those needs mjlab, and a test that needed mjlab
could not check the case where mjlab is absent at all.

So the backend table is the seam: it is replaced with fakes here, exactly as a
second simulator would replace it for real.
"""

from __future__ import annotations

import sys
import types

import pytest

from rlmcp import tasks

#: The list a fake task package registers into. It has to live in a module a
#: generated package can import by name, because the whole point of the
#: attribution test is that registration happens *during* the import.
STORE = "rlmcp_test_registry_store"


@pytest.fixture
def fake_backend(monkeypatch):
  """A backend whose registry is a list this test owns."""
  store = types.ModuleType(STORE)
  store.REGISTRY = []
  monkeypatch.setitem(sys.modules, STORE, store)

  def describe(task):
    return {"experiment": task.lower()}

  monkeypatch.setattr(tasks, "BACKENDS", (
      {"backend": "fake", "ids": lambda: list(store.REGISTRY),
       "describe": describe, "note": ""},
  ))
  return store.REGISTRY


@pytest.fixture
def package(tmp_path, monkeypatch):
  """Write a real importable package that registers on import.

  Not a module put straight into ``sys.modules``: importlib hands back what is
  already there without executing it, so a pre-inserted fake registers before
  the snapshot is taken and every task looks like the backend's own -- which is
  precisely the bug this fixture would have hidden.
  """
  monkeypatch.syspath_prepend(str(tmp_path))
  made: list[str] = []

  def make(name: str, registers: list[str]):
    (tmp_path / f"{name}.py").write_text(
        f"import {STORE} as store\n"
        f"store.REGISTRY.extend({registers!r})\n"
    )
    made.append(name)

  yield make
  for name in made:
    sys.modules.pop(name, None)


def test_the_env_var_and_the_flag_are_both_read(monkeypatch):
  monkeypatch.setenv(tasks.TASK_PACKAGES_ENV, "from_env, also_env ")
  assert tasks.packages_to_import(["explicit"]) == ["explicit", "from_env", "also_env"]


def test_a_package_named_twice_is_imported_once(monkeypatch):
  monkeypatch.setenv(tasks.TASK_PACKAGES_ENV, "same")
  assert tasks.packages_to_import(["same"]) == ["same"]


def test_nothing_is_discovered_that_was_not_named(monkeypatch):
  """A listing that goes looking for packages runs code nobody asked it to."""
  monkeypatch.delenv(tasks.TASK_PACKAGES_ENV, raising=False)
  assert tasks.packages_to_import() == []


def test_a_package_that_will_not_import_is_reported_not_raised(fake_backend):
  answer = tasks.registered(["no_such_package_anywhere"])

  assert answer["tasks"] == []
  failed = answer["packages"][0]
  assert failed["module"] == "no_such_package_anywhere"
  assert failed["imported"] is False
  assert "ModuleNotFoundError" in failed["error"]


def test_a_task_is_attributed_to_the_import_that_registered_it(
    fake_backend, package, monkeypatch):
  """Which package to name in $RLMCP_TASK_PACKAGES is the point of the list."""
  monkeypatch.delenv(tasks.TASK_PACKAGES_ENV, raising=False)
  fake_backend.append("Shipped-With-The-Backend")
  package("my_tasks", ["Mine-Rough-G1"])

  rows = {row["task"]: row for row in tasks.registered(["my_tasks"])["tasks"]}

  assert rows["Mine-Rough-G1"]["package"] == "my_tasks"
  # Already there before we imported anything: the backend's own, and saying
  # "my_tasks" would send somebody to set an environment variable that changes
  # nothing.
  assert rows["Shipped-With-The-Backend"]["package"] == ""


def test_an_absent_backend_is_a_reason_rather_than_an_empty_list(monkeypatch):
  def missing():
    raise ImportError("No module named 'mjlab'")

  monkeypatch.setattr(tasks, "BACKENDS", (
      {"backend": "mjlab", "ids": missing, "describe": lambda t: {}, "note": ""},
  ))
  monkeypatch.delenv(tasks.TASK_PACKAGES_ENV, raising=False)

  answer = tasks.registered()

  assert answer["tasks"] == []
  row = answer["backends"][0]
  assert row["available"] is False
  assert "mjlab" in row["reason"]


def test_one_unreadable_entry_does_not_lose_the_others(monkeypatch):
  def describe(task):
    if task == "Broken":
      raise KeyError("gone")
    return {"experiment": "fine"}

  monkeypatch.setattr(tasks, "BACKENDS", (
      {"backend": "fake", "ids": lambda: ["Broken", "Fine"],
       "describe": describe, "note": ""},
  ))
  monkeypatch.delenv(tasks.TASK_PACKAGES_ENV, raising=False)

  rows = {row["task"]: row for row in tasks.registered()["tasks"]}

  assert "KeyError" in rows["Broken"]["error"]
  assert rows["Fine"]["experiment"] == "fine"


def test_contains_narrows_the_list_case_insensitively(fake_backend, monkeypatch):
  monkeypatch.delenv(tasks.TASK_PACKAGES_ENV, raising=False)
  fake_backend.extend(["Mjlab-Velocity-Rough-G1", "Mjlab-Cartpole-Balance"])

  found = [row["task"] for row in tasks.registered(contains="cartpole")["tasks"]]

  assert found == ["Mjlab-Cartpole-Balance"]


def test_the_backends_are_named_even_when_the_task_list_is_full(fake_backend, monkeypatch):
  """A caller has to be able to tell "no tasks" from "no backend"."""
  monkeypatch.delenv(tasks.TASK_PACKAGES_ENV, raising=False)
  fake_backend.append("Something")

  answer = tasks.registered()

  assert [b["backend"] for b in answer["backends"]] == ["fake"]
  assert answer["backends"][0]["tasks"] == 1


def test_the_real_backend_table_covers_every_simulator():
  """The table is the seam a backend is added to, so it is pinned.

  Genesis was the third, and it arrived without a row -- which left a
  Genesis-only machine being told about the two backends it did not have and
  nothing about the one it did. tests/test_backend_isolation.py checks the
  general rule against the adapter packages on disk; this pins the order and
  the exact membership.
  """
  assert [spec["backend"] for spec in tasks.BACKENDS] == ["mjlab", "isaaclab", "genesis"]
  for spec in tasks.BACKENDS:
    assert callable(spec["ids"]) and callable(spec["describe"])


def test_the_cli_lists_tasks_without_a_session(monkeypatch, capsys, tmp_path):
  """`rlmcp tasks` must not resolve a session -- there may not be one yet."""
  from rlmcp.cli import main

  monkeypatch.setattr(tasks, "BACKENDS", (
      {"backend": "fake", "ids": lambda: ["Mjlab-Cartpole-Balance"],
       "describe": lambda t: {"experiment": "cartpole"}, "note": ""},
  ))
  monkeypatch.delenv(tasks.TASK_PACKAGES_ENV, raising=False)
  monkeypatch.chdir(tmp_path)          # nothing here is a session

  code = main(["--json", "tasks"])

  assert code == 0
  printed = capsys.readouterr().out
  assert "Mjlab-Cartpole-Balance" in printed
  assert "cartpole" in printed


def test_the_cli_says_what_to_do_when_nothing_registers_a_task(
    monkeypatch, capsys, tmp_path):
  from rlmcp.cli import main

  monkeypatch.setattr(tasks, "BACKENDS", (
      {"backend": "fake", "ids": list, "describe": lambda t: {}, "note": ""},
  ))
  monkeypatch.delenv(tasks.TASK_PACKAGES_ENV, raising=False)
  monkeypatch.chdir(tmp_path)

  code = main(["--json", "tasks"])

  assert code == 1
  err = capsys.readouterr().err
  assert tasks.TASK_PACKAGES_ENV in err
