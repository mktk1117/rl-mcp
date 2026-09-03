"""Pulling a function out of its module with the names it depends on.

A term captured as ``inspect.getsource(func)`` is not runnable on its own: a
real term references its module's globals -- a ``_DEFAULT_ASSET_CFG`` default
argument, a ``_gaussian_tolerance`` helper, an ``Entity`` it imported -- and a
default argument is evaluated at ``def`` time, so a module made of bare term
functions does not even import.

So the export takes the *closure* instead: the term, every top-level
definition in its module the term reaches (transitively), and the import
statements that bind the names those reach. Statements come out in the order
the module had them, with relative imports made absolute so they work from
anywhere. What the module imported from elsewhere is still imported, and is
listed so the reader knows what has to be installed -- a backend such as mjlab
is one thing, the task package itself is another, and the README says which.

This is a bundler for one module at a time and deliberately no more: it does
not chase imports into other modules of the task package. The recipe's
``package/`` is the answer when the whole package is wanted.
"""

from __future__ import annotations

import ast
import importlib.util
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Bundled:
  """The statements one module contributes, and what they still import."""

  statements: list[str] = field(default_factory=list)
  """Source text of every kept statement, in the module's order."""

  defined: list[str] = field(default_factory=list)
  """Top-level names the kept statements bind (definitions and imports)."""

  imported_modules: list[str] = field(default_factory=list)
  """Modules the kept import statements load, absolute."""

  missing: list[str] = field(default_factory=list)
  """Names a term references that nothing in the module binds."""


def bundle_module(source: str, package: str | None, wanted: list[str]) -> Bundled:
  """The closure of ``wanted`` (top-level names) inside ``source``.

  ``package`` is the module's ``__package__``, used to make relative imports
  absolute. Raises ``SyntaxError`` if ``source`` does not parse; the caller
  decides what to do about a module it cannot read.
  """
  tree = ast.parse(source)
  lines = source.splitlines(keepends=True)
  binds: dict[str, int] = {}
  for index, node in enumerate(tree.body):
    for name in _bound_names(node):
      binds.setdefault(name, index)

  keep: set[int] = set()
  missing: list[str] = []
  queue = list(wanted)
  seen: set[str] = set()
  while queue:
    name = queue.pop()
    if name in seen:
      continue
    seen.add(name)
    index = binds.get(name)
    if index is None:
      if name in wanted:
        missing.append(name)
      continue
    if index in keep:
      continue
    keep.add(index)
    for referenced in _referenced_names(tree.body[index]):
      if referenced not in seen:
        queue.append(referenced)

  out = Bundled(missing=missing)
  for index in sorted(keep):
    node = tree.body[index]
    out.statements.append(_statement_text(node, lines, package))
    out.defined.extend(_bound_names(node))
    out.imported_modules.extend(_imported_modules(node, package))
  out.imported_modules = list(dict.fromkeys(out.imported_modules))
  return out


# What a statement binds and what it reads.


def _bound_names(node: ast.AST) -> list[str]:
  """Top-level names a statement introduces into the module namespace."""
  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
    return [node.name]
  if isinstance(node, ast.Import):
    return [(alias.asname or alias.name).split(".")[0] for alias in node.names]
  if isinstance(node, ast.ImportFrom):
    return [alias.asname or alias.name for alias in node.names if alias.name != "*"]
  if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: list[str] = []
    for target in targets:
      names.extend(_target_names(target))
    return names
  if isinstance(node, (ast.If, ast.Try, ast.With)):
    # `if TYPE_CHECKING:` and `try: import x` blocks bind whatever is inside.
    names = []
    for child in ast.iter_child_nodes(node):
      if isinstance(child, ast.stmt):
        names.extend(_bound_names(child))
    for body in getattr(node, "orelse", []) or []:
      names.extend(_bound_names(body))
    return names
  return []


def _target_names(target: ast.AST) -> list[str]:
  if isinstance(target, ast.Name):
    return [target.id]
  if isinstance(target, (ast.Tuple, ast.List)):
    names: list[str] = []
    for element in target.elts:
      names.extend(_target_names(element))
    return names
  if isinstance(target, ast.Starred):
    return _target_names(target.value)
  return []


def _referenced_names(node: ast.AST) -> set[str]:
  """Every bare name a statement reads, minus the ones it binds locally.

  Coarse on purpose: a function's parameters and locals come out too, and
  cost nothing, since a name the module does not bind is simply skipped. The
  point is never to *miss* a module-level name.
  """
  names: set[str] = set()
  for child in ast.walk(node):
    if isinstance(child, ast.Name):
      names.add(child.id)
    elif isinstance(child, ast.Attribute):
      base = child
      while isinstance(base, ast.Attribute):
        base = base.value
      if isinstance(base, ast.Name):
        names.add(base.id)
  return names


# Rendering a kept statement.


def _statement_text(node: ast.stmt, lines: list[str], package: str | None) -> str:
  if isinstance(node, ast.ImportFrom) and node.level:
    return _absolute_import(node, package)
  start = node.lineno
  decorators = getattr(node, "decorator_list", None) or []
  if decorators:
    start = min([start, *[d.lineno for d in decorators]])
  end = node.end_lineno or start
  return "".join(lines[start - 1:end]).rstrip("\n") + "\n"


def _absolute_import(node: ast.ImportFrom, package: str | None) -> str:
  """``from .utils import x`` as ``from task.mdp.utils import x``."""
  target = "." * node.level + (node.module or "")
  try:
    absolute = importlib.util.resolve_name(target, package or "")
  except (ImportError, ValueError):
    # No package to resolve against: keep it as written and let the reader
    # see the dot. Better than inventing a module name.
    return ast.unparse(node) + "\n"
  rewritten = ast.ImportFrom(module=absolute, names=node.names, level=0)
  return ast.unparse(rewritten) + "\n"


def _imported_modules(node: ast.stmt, package: str | None) -> list[str]:
  if isinstance(node, ast.Import):
    return [alias.name for alias in node.names]
  if isinstance(node, ast.ImportFrom):
    if node.level:
      target = "." * node.level + (node.module or "")
      try:
        return [importlib.util.resolve_name(target, package or "")]
      except (ImportError, ValueError):
        return [target]
    return [node.module or ""]
  if isinstance(node, (ast.If, ast.Try, ast.With)):
    out: list[str] = []
    for child in ast.iter_child_nodes(node):
      if isinstance(child, ast.stmt):
        out.extend(_imported_modules(child, package))
    return out
  return []


def top_level_package(module: str) -> str:
  return (module or "").split(".")[0]


__all__: list[Any] = ["Bundled", "bundle_module", "top_level_package"]
