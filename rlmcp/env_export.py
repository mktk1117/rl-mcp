"""The environment a policy trained under, written out beside it.

A checkpoint on its own is not a thing you can run. To use it you need the
observations it expects in the order it expects them, the actions it emits, and
-- to keep training it or judge it -- what it was being paid for. That lives in
a task package that has since moved on, at a commit nobody wrote down.

``rlmcp env export`` turns the run's captured terms into a directory that pairs
with the checkpoint:

    exported/
      mdp_terms.py   every term's implementation, inlined
      env_cfg.py     RewardsCfg / ObservationsCfg / ActionsCfg over those
      README.md      what this is, what it pairs with, what did not survive

**Inlined, with what each term reaches.** A term's source alone does not run:
it reads its module's constants and helpers and the names its module imported.
So each term is cut out of its module with that closure -- see
:mod:`rlmcp.source_bundle` -- and what the module imported from *elsewhere* is
still imported, and listed in the README so the reader knows what has to be
installed: a backend such as mjlab is one thing, a sibling module of the task
package is another. The cost is honest and worth stating: this is a *fork*,
not a reference. It will not pick up later fixes to those terms, and it is a
snapshot for pairing with one checkpoint rather than a package to develop in.
When what you want is the living package at the run's commit, that is what
``rlmcp recipe build`` gives you.

**Weights are the ones the run ended on.** A term added at 0.5 and tuned to 3.0
is exported at 3.0, because 3.0 is what the checkpoint was trained under. The
originally-configured value is kept as a comment.

**What could not be rendered is named, never guessed.** A param holding an
object that cannot be reconstructed from its fields is written as a commented
`repr` with the term marked in the README, because a config that quietly
differs from the one that trained the policy is worse than one that says where
it stopped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rlmcp.session import Session

TERMS_STEM = "mdp_terms"
CFG_STEM = "env_cfg"

#: Managers this exports, in the order they appear in the generated config.
SECTIONS = ("rewards", "observations", "actions")


@dataclass
class Rendered:
  """The text of an export, plus what it could not do."""

  terms_module: str = ""
  config_module: str = ""
  readme: str = ""
  imports: list[str] = field(default_factory=list)
  unrendered: list[str] = field(default_factory=list)
  missing_source: list[str] = field(default_factory=list)
  symbols: dict[tuple[str, str], str] = field(default_factory=dict)
  """``(module, name)`` of a captured term -> the name it has in mdp_terms.py.
  Differs from ``name`` only when two modules defined the same name."""
  still_imports: list[str] = field(default_factory=list)
  """Modules the export imports because the terms' modules did."""
  no_context: list[str] = field(default_factory=list)
  """Modules whose source was not captured, so their terms are inlined bare."""
  unbound: list[str] = field(default_factory=list)
  """Names a term reaches that its module does not bind (bundler could not follow)."""


class _Imports:
  """Collects `from x import Y` lines, deduplicated and ordered."""

  def __init__(self) -> None:
    self._pairs: set = set()

  def add(self, module: str | None, name: str | None) -> None:
    if module and name and module != "builtins":
      self._pairs.add((module, name, None))

  def add_alias(self, module: str, name: str, alias: str) -> None:
    self._pairs.add((module, name, alias))

  def lines(self) -> list[str]:
    out = []
    for module, name, alias in sorted(self._pairs):
      suffix = f" as {alias}" if alias else ""
      out.append(f"from {module} import {name}{suffix}")
    return out


# Rendering a captured value back into source.


def render_value(
    value: Any, imports: _Imports, unrendered: list[str], *, where: str = ""
) -> str:
  """Turn :func:`~rlmcp.adapters.manager_based.term_capture.encode_value`
  output back into constructible Python, collecting the imports it needs."""
  if value is None or isinstance(value, (bool, int, float, str)):
    return repr(value)
  if isinstance(value, list):
    inner = ", ".join(
        render_value(v, imports, unrendered, where=where) for v in value)
    return f"[{inner}]"
  if not isinstance(value, dict):
    unrendered.append(f"{where}: a bare {type(value).__name__}")
    return f"None  # unrendered: {type(value).__name__}"

  if "__tuple__" in value:
    items = [render_value(v, imports, unrendered, where=where)
             for v in value["__tuple__"]]
    if len(items) == 1:
      return f"({items[0]},)"
    return "(" + ", ".join(items) + ")"
  if "__map__" in value:
    inner = ", ".join(
        f"{k!r}: {render_value(v, imports, unrendered, where=where)}"
        for k, v in value["__map__"].items())
    return "{" + inner + "}"
  if "__slice_all__" in value:
    return "slice(None)"
  if "__ref__" in value:
    ref = value["__ref__"]
    name = (ref.get("qualname") or "").split(".")[0]
    imports.add(ref.get("module"), name)
    return ref.get("qualname") or "None"
  if "__obj__" in value:
    obj = value["__obj__"]
    imports.add(obj.get("module"), obj.get("name"))
    fields = obj.get("fields") or {}
    inner = ", ".join(
        f"{k}={render_value(v, imports, unrendered, where=where)}"
        for k, v in fields.items())
    return f"{obj.get('name')}({inner})"
  if value.get("__unrenderable__"):
    reason = value.get("reason", "unknown")
    unrendered.append(f"{where}: {reason}")
    return f"None  # unrendered ({reason}): {value.get('__repr__', '')}"

  # A plain dict that carried no marker: encode_value always marks maps, so
  # this is data from an older schema. Render it as a dict rather than fail.
  inner = ", ".join(
      f"{k!r}: {render_value(v, imports, unrendered, where=where)}"
      for k, v in value.items())
  return "{" + inner + "}"


# Reading the run.


def _final_reward_values(session: Session) -> dict[str, Any]:
  """Reward parameters as the run left them, keyed by their dotted path."""
  out: dict[str, Any] = {}
  for key, spec in (session.params() or {}).items():
    if not key.startswith("reward."):
      continue
    value = spec.get("current") if isinstance(spec, dict) else spec
    out[key] = value
  return out


def _overlay_weight(
    term: dict[str, Any], final: dict[str, Any]) -> tuple[float, float | None]:
  """The weight to export, and the configured one when tuning moved it."""
  configured = term.get("weight")
  key = f"reward.{term.get('name')}.weight"
  live = final.get(key)
  if isinstance(live, (int, float)) and not isinstance(live, bool):
    live = float(live)
    if configured is None or live != float(configured):
      return live, (float(configured) if configured is not None else None)
    return live, None
  return (float(configured) if configured is not None else 0.0), None


def _source_of(term: dict[str, Any], missing: list[str], label: str) -> str | None:
  info = term.get("func") or {}
  if info.get("available") and info.get("source"):
    return info["source"]
  reason = info.get("reason", "no source captured")
  qualified = ".".join(
      p for p in (info.get("module"), info.get("qualname")) if p)
  missing.append(f"{label}: {qualified or 'unknown'} ({reason})")
  return None


# The two generated modules.


def render_terms_module(
    snapshot: dict[str, Any], rendered: Rendered) -> str:
  """Every implementation, with the module-level names it reaches.

  Terms are grouped by the module they came from, and each module contributes
  the closure of its terms: the functions, the constants and helpers they use,
  and the imports that bind the rest. Two modules that define the same name
  differently do not shadow each other -- the second is renamed with its
  module's slug, and the config refers to the renamed symbol.
  """
  lines = [
      '"""Implementations of every term this policy trained under.',
      "",
      "Written by `rlmcp env export`. Each function is the source that was",
      "running in the training process, captured from the live managers, with",
      "the module-level names it reaches. What its module imported from",
      "elsewhere is imported here too; README.md lists what that needs.",
      "",
      "It is a snapshot, not a package: it will not pick up later fixes to",
      "these terms. Pair it with the checkpoint it was exported beside.",
  ]
  if snapshot.get("task"):
    lines += ["", f"task: {snapshot['task']}"]
  lines += ['"""', "", "from __future__ import annotations", "",
            "import torch  # noqa: F401 - agent-written terms are compiled against it.",
            ""]

  # Which top-level names each module has to contribute. Actions are absent:
  # an action term is a backend class named by its config, not a function
  # with source of its own.
  wanted: dict[str, list[str]] = {}
  bare: list[tuple[str, dict[str, Any]]] = []
  for label, term in _source_terms(snapshot):
    info = term.get("func") or {}
    if not info.get("available") or not info.get("source"):
      _source_of(term, rendered.missing_source, label)
      continue
    module = str(info.get("module") or "")
    top = str(info.get("qualname") or info.get("name") or "").split(".")[0]
    if info.get("origin") == "agent" or module not in (snapshot.get("modules") or {}):
      bare.append((label, term))
      continue
    wanted.setdefault(module, [])
    if top and top not in wanted[module]:
      wanted[module].append(top)

  body: list[str] = []
  emitted: dict[str, tuple[str, str]] = {}   # name -> (module, statement text)
  modules = snapshot.get("modules") or {}
  for module, names in wanted.items():
    _emit_module(module, names, modules[module], body, emitted, rendered)
  for label, term in bare:
    _emit_bare(label, term, body, emitted, rendered, snapshot)

  if not body:
    body = ["# No term source was captured for this run.", ""]
  return "\n".join([*lines, "", *body]).rstrip() + "\n"


def _source_terms(snapshot: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
  """Every reward and observation term, with a label for the README."""
  out: list[tuple[str, dict[str, Any]]] = []
  for term in _iter_terms(snapshot, "rewards"):
    out.append((f"reward '{term.get('name')}'", term))
  for group, spec in (snapshot.get("observations") or {}).items():
    for term in spec.get("terms") or []:
      out.append((f"observation {group} '{term.get('name')}'", term))
  return out


def _slug(module: str) -> str:
  return "".join(c if c.isalnum() else "_" for c in module)


def _emit_module(
    module: str, names: list[str], captured: dict[str, Any], body: list[str],
    emitted: dict[str, tuple[str, str]], rendered: Rendered) -> None:
  """One module's closure into the body, renaming on collision."""
  from rlmcp.source_bundle import bundle_module

  try:
    bundle = bundle_module(captured.get("source") or "", captured.get("package"),
                           list(names))
  except SyntaxError as exc:
    rendered.no_context.append(f"{module}: its source does not parse ({exc})")
    for name in names:
      rendered.symbols[(module, name)] = name
    return
  rendered.unbound += [f"{module}: {name}" for name in bundle.missing]
  rendered.still_imports += [
      m for m in bundle.imported_modules if m not in rendered.still_imports]

  # A definition that another module already contributed under the same name
  # with different text is renamed. An *import* of the same name is not: two
  # modules importing `Entity` from the backend mean the same object, and a
  # rename would rewrite the import line itself into a name that does not
  # exist.
  renames: dict[str, str] = {}
  statements = list(bundle.statements)
  for text in statements:
    if _is_import(text):
      continue
    for bound in _bound_in(text):
      previous = emitted.get(bound)
      if previous is not None and previous[0] != module and previous[1] != text:
        renames[bound] = f"{bound}__{_slug(module)}"
  if renames:
    statements = [_rename(text, renames) for text in statements]

  section: list[str] = []
  for text in statements:
    bound = _bound_in(text)
    if bound and all(b in emitted and emitted[b][1] == text for b in bound):
      continue  # the same statement, already there from another module
    for b in bound:
      emitted.setdefault(b, (module, text))
    section.append(text.rstrip("\n"))
  for name in names:
    rendered.symbols[(module, name)] = renames.get(name, name)
  if section:
    body.append(f"# ---- from {module} ----")
    body += [s + "\n" if not s.endswith("\n") else s for s in section]
    body.append("")


def _emit_bare(
    label: str, term: dict[str, Any], body: list[str],
    emitted: dict[str, tuple[str, str]], rendered: Rendered,
    snapshot: dict[str, Any]) -> None:
  """A term whose module was not captured: its own source, and a note."""
  info = term.get("func") or {}
  module = str(info.get("module") or "")
  name = str(info.get("qualname") or info.get("name") or "").split(".")[0]
  source = str(info.get("source") or "")
  if info.get("origin") != "agent":
    note = f"{module}: not captured, so {label} is inlined without its module's names"
    if note not in rendered.no_context:
      rendered.no_context.append(note)
  previous = emitted.get(name)
  if previous is not None and previous[1] == source:
    rendered.symbols[(module, name)] = name
    return
  symbol = name
  if previous is not None:
    symbol = f"{name}__{_slug(module)}"
    source = _rename(source, {name: symbol})
  emitted[symbol] = (module, source)
  rendered.symbols[(module, name)] = symbol
  origin = " (written by the agent during the run)" if info.get(
      "origin") == "agent" else ""
  body.append(f"# from {module}{origin}")
  body.append(source.rstrip())
  body.append("")


def _is_import(statement: str) -> bool:
  """Whether a statement's text is an import (bare, or inside an if/try)."""
  import ast

  try:
    tree = ast.parse(statement)
  except SyntaxError:
    return False
  for node in tree.body:
    inner = [node]
    if isinstance(node, (ast.If, ast.Try, ast.With)):
      inner = [c for c in ast.walk(node) if isinstance(c, ast.stmt) and c is not node]
    if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in inner):
      return True
  return False


def _bound_in(statement: str) -> list[str]:
  """Top-level names one statement's text binds."""
  import ast

  from rlmcp.source_bundle import _bound_names

  try:
    tree = ast.parse(statement)
  except SyntaxError:
    return []
  names: list[str] = []
  for node in tree.body:
    names.extend(_bound_names(node))
  return names


def _rename(text: str, renames: dict[str, str]) -> str:
  """Rename bare identifiers in one module's statements.

  Textual, on word boundaries and not after a dot, so `obj.track` is left
  alone and `track(`, `def track`, `= track` are renamed. Good enough for the
  rare collision this exists for; the README names every rename.
  """
  import re

  for old, new in renames.items():
    text = re.sub(rf"(?<![\w.]){re.escape(old)}\b", new, text)
  return text


def _iter_terms(snapshot: dict[str, Any], section: str) -> list[dict[str, Any]]:
  value = snapshot.get(section)
  return list(value) if isinstance(value, list) else []


def _emit_source(
    term: dict[str, Any], label: str, body: list[str], seen: set,
    rendered: Rendered) -> None:
  info = term.get("func") or {}
  key = (info.get("module"), info.get("qualname"))
  source = _source_of(term, rendered.missing_source,
                      f"{label} '{term.get('name')}'")
  if source is None or key in seen:
    return
  seen.add(key)
  origin = " (written by the agent during the run)" if info.get(
      "origin") == "agent" else ""
  body.append(f"# from {info.get('module')}{origin}")
  body.append(source.rstrip())
  body.append("")


def render_config_module(snapshot: dict[str, Any], session: Session,
                         rendered: Rendered) -> str:
  """The config half: one cfg class per manager, over ``mdp_terms``."""
  imports = _Imports()
  unrendered = rendered.unrendered
  final = _final_reward_values(session)

  blocks: list[str] = []
  blocks.append(_render_rewards(snapshot, final, imports, unrendered, rendered))
  blocks.append(_render_observations(snapshot, imports, unrendered, rendered))
  blocks.append(_render_actions(snapshot, imports, unrendered))

  # The backend's own term-config classes, recorded at capture. Emitting the
  # real import beats offering the reader two commented guesses: the run knew
  # which backend it was.
  cfg_types = snapshot.get("term_cfg_types") or {}
  for key, alias in (("reward", "RewTerm"), ("observation", "ObsTerm")):
    entry = cfg_types.get(key) or {}
    if entry.get("module") and entry.get("name"):
      imports.add_alias(entry["module"], entry["name"], alias)

  header = [
      '"""The environment config this policy trained under.',
      "",
      "Written by `rlmcp env export`, over the implementations in",
      f"{TERMS_STEM}.py. Weights are the ones in force when the run ended,",
      "which is what the checkpoint was actually trained against.",
      '"""',
      "",
      "from __future__ import annotations",
      "",
      "from dataclasses import dataclass, field",
      "",
  ]
  import_lines = imports.lines()
  if import_lines:
    header += [*import_lines, ""]
  # Works dropped into a package and as a loose directory on sys.path, which
  # are the two ways this actually gets used.
  header += [
      "try:",
      f"  from . import {TERMS_STEM}",
      "except ImportError:  # a loose directory rather than a package",
      f"  import {TERMS_STEM}  # type: ignore[no-redef]",
      "", "",
  ]
  rendered.imports = import_lines
  return "\n".join(header + blocks).rstrip() + "\n"


def _render_rewards(snapshot, final, imports, unrendered, rendered) -> str:
  terms = _iter_terms(snapshot, "rewards")
  lines = ["@dataclass", "class RewardsCfg:",
           '  """What the policy was paid for."""', ""]
  if not terms:
    lines.append("  pass")
    return "\n".join(lines) + "\n\n"
  for term in terms:
    name = term.get("name")
    weight, configured = _overlay_weight(term, final)
    params = render_value(term.get("params") or {}, imports, unrendered,
                          where=f"reward '{name}' params")
    if configured is not None:
      lines.append(
          f"  # configured at {configured!r}; the run ended at {weight!r}.")
    func = _symbol(term, rendered)
    extra = f", params={params}" if params not in ("{}", "None") else ""
    line = (f"  {name}: RewTerm = field(default_factory=lambda: RewTerm("
            f"func={TERMS_STEM}.{func}, weight={weight!r}{extra}))")
    if func is None:
      # No source to point at. A line naming a function that is not there
      # would make RewardsCfg() raise; a commented one says what is missing.
      reason = (term.get("func") or {}).get("reason", "no source captured")
      lines.append(f"  # {name}: source unavailable ({reason}); supply it by hand:")
      lines.append("  #" + line.replace(f"{TERMS_STEM}.None", f"{TERMS_STEM}.<{name}>")[1:])
      continue
    lines.append(line)
  if all(row.lstrip().startswith("#") or not row.strip() for row in lines[4:]):
    lines.append("  pass")
  return "\n".join(lines) + "\n\n"


def _symbol(term: dict[str, Any], rendered: Rendered) -> str | None:
  """The name in mdp_terms.py this term's func has, or None when it has none."""
  info = term.get("func") or {}
  if not info.get("available") or not info.get("source"):
    return None
  module = str(info.get("module") or "")
  top = str(info.get("qualname") or info.get("name") or "").split(".")[0]
  return rendered.symbols.get((module, top), top)


def _render_observations(snapshot, imports, unrendered, rendered) -> str:
  groups = snapshot.get("observations") or {}
  lines = ["@dataclass", "class ObservationsCfg:",
           '  """What the policy saw. Order is the input vector\'s order."""',
           ""]
  if not groups:
    lines.append("  pass")
    return "\n".join(lines) + "\n\n"
  bodies: list[str] = []
  for group, spec in groups.items():
    cls = _group_class_name(group)
    body = ["  @dataclass", f"  class {cls}:"]
    terms = spec.get("terms") or []
    if not terms:
      body.append("    pass")
    for term in terms:
      name = term.get("name")
      func = _symbol(term, rendered)
      if func is None:
        reason = (term.get("func") or {}).get("reason", "no source captured")
        body.append(f"    # {name}: source unavailable ({reason}); supply it by hand.")
        continue
      pieces = [f"func={TERMS_STEM}.{func}"]
      params = render_value(term.get("params") or {}, imports, unrendered,
                            where=f"observation '{group}.{name}' params")
      if params not in ("{}", "None"):
        pieces.append(f"params={params}")
      for extra in ("noise", "clip", "scale", "history_length"):
        if extra in term:
          pieces.append(f"{extra}=" + render_value(
              term[extra], imports, unrendered,
              where=f"observation '{group}.{name}'.{extra}"))
      dim = term.get("dim")
      comment = f"  # dim {tuple(dim)}" if dim else ""
      body.append(
          f"    {name}: ObsTerm = field(default_factory=lambda: ObsTerm("
          + ", ".join(pieces) + f")){comment}")
    body.append("")
    body.append(f"    concatenate_terms: bool = {bool(spec.get('concatenate', True))!r}")
    body.append("")
    bodies.append("\n".join(body))
  lines += bodies
  lines.append("  " + "\n  ".join(
      f"{group}: {_group_class_name(group)} = field("
      f"default_factory={_group_class_name(group)})" for group in groups))
  return "\n".join(lines) + "\n\n"


def _group_class_name(group: str) -> str:
  return "".join(part.capitalize() for part in str(group).split("_")) + "Cfg"


def _render_actions(snapshot, imports, unrendered) -> str:
  terms = _iter_terms(snapshot, "actions")
  lines = ["@dataclass", "class ActionsCfg:",
           '  """What the policy\'s outputs meant."""', ""]
  if not terms:
    lines.append("  pass")
    return "\n".join(lines) + "\n"
  for term in terms:
    name = term.get("name")
    cfg = render_value(term.get("cfg"), imports, unrendered,
                       where=f"action '{name}'")
    dim = term.get("dim")
    comment = f"  # dim {dim}" if dim else ""
    # An annotated default_factory, not a bare class attribute: an action cfg
    # is mutable, and `name = field(...)` without an annotation is not a
    # dataclass field at all -- it leaves a Field object on the class.
    annotation = ((term.get("_cfg_type") or {}).get("name")) or "Any"
    if annotation == "Any":
      imports.add("typing", "Any")
    lines.append(
        f"  {name}: {annotation} = field(default_factory=lambda: {cfg})"
        f"{comment}")
  return "\n".join(lines) + "\n"


def render_readme(snapshot: dict[str, Any], session: Session,
                  rendered: Rendered) -> str:
  task = snapshot.get("task") or "(unknown task)"
  rewards = len(_iter_terms(snapshot, "rewards"))
  groups = snapshot.get("observations") or {}
  obs = sum(len(g.get("terms") or []) for g in groups.values())
  actions = len(_iter_terms(snapshot, "actions"))
  lines = [
      f"# Environment export — {task}",
      "",
      f"Exported from `{session.dir}`.",
      "",
      f"- **{rewards}** reward terms",
      f"- **{obs}** observation terms across {len(groups)} group(s): "
      + ", ".join(sorted(groups)) if groups else "- no observation groups",
      f"- **{actions}** action term(s)",
      "",
      "## What this is",
      "",
      f"`{TERMS_STEM}.py` holds the implementation of every term, inlined as",
      "the source that was running during the run. `" + CFG_STEM + ".py` holds",
      "the config over them, with reward weights as the run left them.",
      "",
      "It is a snapshot for pairing with one checkpoint, not a package to",
      "develop in — it will not pick up later fixes to these terms. For the",
      "living package at the run's commit, use `rlmcp recipe build`.",
      "",
      "## Before you trust it",
      "",
      "Nothing here was executed. Import it once against your backend before",
      "pairing it with a checkpoint.",
  ]
  if rendered.still_imports:
    task_top = {str((t.get("func") or {}).get("module") or "").split(".")[0]
                for _, t in _source_terms(snapshot)} - {""}
    own = [m for m in rendered.still_imports if m.split(".")[0] in task_top]
    other = [m for m in rendered.still_imports if m not in own]
    lines += ["", "## What it still imports", "",
              "The terms' modules imported these, and the export imports them",
              "too. They have to be importable where this runs:", ""]
    if other:
      lines += [f"- `{m}`" for m in other]
    if own:
      lines += ["", "From the task package itself, which therefore still has",
                "to be installed (the recipe's `package/` is the version that ran):", ""]
      lines += [f"- `{m}`" for m in own]
  if rendered.no_context:
    lines += ["", "## Terms inlined without their module", "",
              "The module's source was not captured, so only the function is",
              "here; a name it reads from its module will be undefined:", ""]
    lines += [f"- {entry}" for entry in rendered.no_context]
  if rendered.unbound:
    lines += ["", "## Names the bundler could not follow", "",
              "Read by a term but bound nowhere in its module (a builtin, or",
              "injected at runtime):", ""]
    lines += [f"- {entry}" for entry in rendered.unbound]
  if rendered.missing_source:
    lines += ["", "## Terms whose source could not be captured", "",
              "These are **commented out in the config** and must be supplied by",
              "hand, from the task package:", ""]
    lines += [f"- {entry}" for entry in rendered.missing_source]
  if rendered.unrendered:
    lines += ["", "## Values that could not be rendered", "",
              "Written as commented `repr` in the config rather than guessed:",
              ""]
    lines += [f"- {entry}" for entry in rendered.unrendered]
  problems = snapshot.get("problems") or []
  if problems:
    lines += ["", "## Problems noted while capturing", ""]
    lines += [f"- {entry}" for entry in problems]
  return "\n".join(lines).rstrip() + "\n"


def render_export(session: Session) -> Rendered:
  """Render all three files without writing anything."""
  snapshot = session.env_terms()
  rendered = Rendered()
  if not snapshot:
    return rendered
  rendered.terms_module = render_terms_module(snapshot, rendered)
  rendered.config_module = render_config_module(snapshot, session, rendered)
  rendered.readme = render_readme(snapshot, session, rendered)
  return rendered


def export_env(session: Session, out_dir: Path | str) -> dict[str, Any]:
  """Write the export into ``out_dir``.

  Returns a payload naming what was written and what could not be, or
  ``{"ok": False, ...}`` when the run captured no terms at all -- which is
  what a run predating the capture, or a backend that has no managers, looks
  like.
  """
  snapshot = session.env_terms()
  if not snapshot:
    return {
        "ok": False,
        "session": str(session.dir),
        "error": (
            "This run captured no environment terms, so there is nothing to "
            "export. Runs started before `env_terms.json` existed, and "
            "backends with no reward/observation/action managers, both look "
            "like this."
        ),
    }
  rendered = render_export(session)
  directory = Path(out_dir).expanduser()
  directory.mkdir(parents=True, exist_ok=True)
  terms_path = directory / f"{TERMS_STEM}.py"
  cfg_path = directory / f"{CFG_STEM}.py"
  readme_path = directory / "README.md"
  terms_path.write_text(rendered.terms_module)
  cfg_path.write_text(rendered.config_module)
  readme_path.write_text(rendered.readme)

  groups = snapshot.get("observations") or {}
  return {
      "ok": True,
      "session": str(session.dir),
      "task": snapshot.get("task"),
      "out_dir": str(directory),
      "implementation": str(terms_path),
      "config": str(cfg_path),
      "readme": str(readme_path),
      "counts": {
          "rewards": len(_iter_terms(snapshot, "rewards")),
          "observation_groups": len(groups),
          "observations": sum(len(g.get("terms") or []) for g in groups.values()),
          "actions": len(_iter_terms(snapshot, "actions")),
      },
      "missing_source": rendered.missing_source,
      "unrendered": rendered.unrendered,
      "still_imports": rendered.still_imports,
      "no_context": rendered.no_context,
      "problems": snapshot.get("problems") or [],
  }


def describe(payload: dict[str, Any]) -> str:
  """A human summary for the CLI."""
  if not payload.get("ok"):
    return payload.get("error", "Nothing to export.")
  counts = payload["counts"]
  lines = [
      (f"Exported {counts['rewards']} reward, {counts['observations']} "
      f"observation and {counts['actions']} action term(s) to "
      f"{payload['out_dir']}"),
      f"  implementations: {payload['implementation']}",
      f"  config:          {payload['config']}",
  ]
  if payload.get("missing_source"):
    lines.append(
        f"  {len(payload['missing_source'])} term(s) had no capturable "
        "source and are MISSING from the export -- see README.md")
  if payload.get("unrendered"):
    lines.append(
        f"  {len(payload['unrendered'])} value(s) could not be rendered and "
        "are commented out -- see README.md")
  lines.append("Nothing here was executed; import it once before trusting it.")
  return "\n".join(lines)


__all__ = [
  "CFG_STEM",
  "TERMS_STEM",
  "Rendered",
  "describe",
  "export_env",
  "render_export",
]
