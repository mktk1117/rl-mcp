"""The example script's rlmcp imports, checked against the library.

The example cannot run here -- it needs mjlab and a GPU -- so nothing else
exercises its import lines, and a renamed symbol has shipped broken before.
This parses the script and asserts that every ``import rlmcp...`` target
still exists. Heavy dependencies (torch, mjlab) missing from the environment
skip the check; a missing rlmcp module or name fails it.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").rglob("train_*.py"))
"""Every example, found rather than listed -- a new one is checked the day it
lands rather than the day somebody remembers to add it here."""

_HEAVY = ("torch", "mjlab")


def rlmcp_imports(path: Path) -> list[tuple[str, list[str]]]:
  """Every rlmcp import in ``path``, as ``(module, imported names)`` pairs."""
  out: list[tuple[str, list[str]]] = []
  for node in ast.walk(ast.parse(path.read_text())):
    if isinstance(node, ast.Import):
      for alias in node.names:
        if alias.name == "rlmcp" or alias.name.startswith("rlmcp."):
          out.append((alias.name, []))
    elif (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
          and (node.module == "rlmcp" or node.module.startswith("rlmcp."))):
      out.append((node.module, [alias.name for alias in node.names]))
  return out


def _import(module: str):
  """Import ``module``, skipping when only a heavy dependency is missing."""
  try:
    return importlib.import_module(module)
  except ModuleNotFoundError as exc:
    root = (exc.name or "").split(".")[0]
    if root in _HEAVY:
      pytest.skip(f"'{module}' needs {root}, which is not installed here")
    raise


IMPORTS = [
    (path.name, module, names)
    for path in EXAMPLES
    for module, names in rlmcp_imports(path)
]
"""``(example, module, imported names)`` for every rlmcp import in every
example."""


def test_every_example_imports_rlmcp():
  """Guards the guard.

  An example that imports no rlmcp at all is either a rewrite that dropped it
  or a file in the wrong directory, and either way nothing below would notice.
  """
  covered = {example for example, _, _ in IMPORTS}
  missing = sorted(p.name for p in EXAMPLES if p.name not in covered)
  assert not missing, f"examples with no rlmcp import: {missing}"


@pytest.mark.parametrize(
    "example, module, names", IMPORTS,
    ids=[f"{e}:{m}" for e, m, _ in IMPORTS])
def test_example_rlmcp_imports_resolve(example: str, module: str, names: list[str]):
  """Each ``import rlmcp...`` line in the example names things that exist."""
  mod = _import(module)
  for name in names:
    try:
      getattr(mod, name)
      continue
    except AttributeError:
      pass
    except ModuleNotFoundError as exc:
      # A lazy attribute (rlmcp/__init__ resolves on access) pulled in a
      # heavy dependency; that is the environment's absence, not rot.
      root = (exc.name or "").split(".")[0]
      if root in _HEAVY:
        pytest.skip(f"'{module}.{name}' needs {root}, which is not installed here")
      raise
    # A from-import may also name a submodule rather than an attribute.
    _import(f"{module}.{name}")
