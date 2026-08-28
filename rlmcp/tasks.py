"""What this environment could run, before any of it has ever run.

Every other command in rlmcp starts from a *session*: a run that exists, on
disk, with a task id already written into it. That leaves one question with
nowhere to go — **what could I run?** — and it is the first question anybody
building a task asks, because a task being built has no runs by definition and
therefore no session to be found by.

The lookup itself is not new. ``train`` and ``play`` both do it already,
privately, at the point of use: each imports the backend's registry to check
the id it was handed and to list the alternatives when that id is wrong. All
this module does is give that lookup a name and an answer worth reading, so it
can be asked without launching anything.

Three things it reports that the bare id does not:

* **which package registered it.** A task exists because importing a package
  registered it, and the package doing the registering is exactly what
  ``$RLMCP_TASK_PACKAGES`` names. Rather than guess it from a class or reach
  into the registry's internals, the registry is compared *before and after*
  each import: whatever appeared is that package's. Tasks already registered
  before we imported anything are the backend's own, and say so.
* **which backend it belongs to.** There are two now. A listing that quietly
  covered one would read as "you have no IsaacLab tasks" on a machine that has
  plenty.
* **where its runs land** — ``experiment``, the name the RL config gives the
  log directory, which is what turns a task id into a directory to look in.
  Only mjlab answers this one; IsaacLab keeps it somewhere that would have to
  be loaded, and a blank is better than a guess.

Nothing here imports a simulator until it is called, so ``rlmcp --help`` stays
as cheap as it has always been.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from rlmcp.play import TASK_PACKAGES_ENV

__all__ = [
  "BACKENDS",
  "TASK_PACKAGES_ENV",
  "import_packages",
  "packages_to_import",
  "registered",
]


def packages_to_import(explicit: list[str] | tuple[str, ...] = ()) -> list[str]:
  """Packages to import before asking, in the order they were given.

  ``--task-package`` first, then ``$RLMCP_TASK_PACKAGES``, deduplicated. The
  same resolution ``play`` uses, and deliberately no discovery: nothing here
  goes looking for a package on disk, because a listing that imports whatever
  it finds is a listing that can execute code nobody asked it to.
  """
  found = [name for name in explicit if name]
  for raw in os.environ.get(TASK_PACKAGES_ENV, "").split(","):
    name = raw.strip()
    if name and name not in found:
      found.append(name)
  return found


def import_packages(names: list[str]) -> list[dict[str, Any]]:
  """Import each package, and report what happened to each one.

  A package that will not import is the single most common reason a task id is
  missing, and it is invisible from the id alone -- the task simply is not
  there. So the failure is carried back with its message rather than raised:
  the other packages still register, and the answer says which one did not.
  """
  import importlib

  report: list[dict[str, Any]] = []
  for name in names:
    try:
      importlib.import_module(name)
    except Exception as exc:
      report.append({"module": name, "imported": False, "error": f"{type(exc).__name__}: {exc}"})
    else:
      report.append({"module": name, "imported": True, "error": ""})
  return report


# ── backends ──────────────────────────────────────────────────────────────
#
# One entry per simulator rlmcp can drive. `ids` returns the registered task
# ids, or raises if this backend is not installed here; `describe` says what is
# known about one of them without building it; `note` is what a reader needs to
# know to interpret an empty list. Adding a backend is an entry in this table,
# and nothing above this line learns its name.


def _mjlab_ids() -> list[str]:
  from mjlab.tasks.registry import list_tasks

  return list(list_tasks())


def _mjlab_describe(task: str) -> dict[str, Any]:
  """What the registry can say without constructing anything.

  Deliberately one field. The configs are already built -- mjlab registers
  instances, not factories -- so a dozen more could be read off them for free,
  and every one of them would be a column pushing the task id off the screen.
  ``experiment`` earns its place because it is the missing half of the id: it
  is the directory this task's runs land in, so it turns "which task is this"
  into "where are its runs".
  """
  from mjlab.tasks.registry import _REGISTRY

  entry = _REGISTRY[task]
  return {"experiment": str(getattr(entry.rl_cfg, "experiment_name", "") or "")}


def _isaaclab_ids() -> list[str]:
  """IsaacLab registers into gymnasium, and only once its app is running.

  Listing them means having started the simulation app, which a listing must
  not do -- it takes tens of seconds and a GPU. So this reports what is in the
  gym registry *if* something already put it there, and otherwise nothing;
  the backend row says which of those two happened.
  """
  import gymnasium as gym

  return sorted(spec for spec in gym.registry
                if spec.startswith(("Isaac-", "Isaaclab-")))


def _isaaclab_describe(task: str) -> dict[str, Any]:
  """IsaacLab keeps the log directory in its agent config, not in the registry,
  and reading that means loading the config -- so the honest answer is that
  this backend does not say."""
  return {"experiment": ""}


BACKENDS: tuple[dict[str, Any], ...] = (
    {"backend": "mjlab", "ids": _mjlab_ids, "describe": _mjlab_describe, "note": ""},
    {"backend": "isaaclab", "ids": _isaaclab_ids, "describe": _isaaclab_describe,
     "note": "IsaacLab tasks register into gymnasium when its app starts, and a "
             "listing must not start one -- so this sees only what something "
             "else already registered in this process."},
)


def _ids_or_reason(lister: Callable[[], list[str]]) -> tuple[list[str], str]:
  try:
    return list(lister()), ""
  except ImportError as exc:
    return [], f"not installed here ({exc})"
  except Exception as exc:
    return [], f"{type(exc).__name__}: {exc}"


def registered(packages: list[str] | tuple[str, ...] = (),
               contains: str = "") -> dict[str, Any]:
  """Every task id this environment can drive, and where each one came from.

  ``packages`` are imported first, in order, and each task is attributed to the
  import that made it appear. The reply always names the packages and the
  backends as well as the tasks: "no tasks" and "no backend installed" are
  different answers, and a caller that cannot tell them apart will report the
  wrong one to somebody.
  """
  wanted = packages_to_import(packages)

  # Snapshot per backend before importing anything, so what is already there is
  # attributable to the backend rather than to us.
  owner: dict[tuple[str, str], str] = {}
  seen: dict[str, set] = {}
  for spec in BACKENDS:
    ids, _reason = _ids_or_reason(spec["ids"])
    seen[spec["backend"]] = set(ids)

  imports = import_packages(wanted)
  for entry in imports:
    if not entry["imported"]:
      continue
    for spec in BACKENDS:
      name = spec["backend"]
      ids, _reason = _ids_or_reason(spec["ids"])
      fresh = set(ids) - seen[name]
      for task in fresh:
        owner[(name, task)] = entry["module"]
      seen[name] |= fresh

  rows: list[dict[str, Any]] = []
  backends: list[dict[str, Any]] = []
  needle = contains.lower()
  for spec in BACKENDS:
    name = spec["backend"]
    ids, reason = _ids_or_reason(spec["ids"])
    backends.append({
        "backend": name,
        "available": not reason,
        "reason": reason,
        "tasks": len(ids),
        "note": spec["note"],
    })
    for task in ids:
      if needle and needle not in task.lower():
        continue
      row = {"task": task, "backend": name, "package": owner.get((name, task), "")}
      try:
        row.update(spec["describe"](task))
      except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
      rows.append(row)

  rows.sort(key=lambda r: (r["backend"], r["task"]))
  return {"tasks": rows, "packages": imports, "backends": backends}
