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

**Self-contained on purpose.** The implementations are inlined rather than
imported, so the export runs without the task package installed at the version
the run used. The cost is honest and worth stating: this is a *fork*, not a
reference. It will not pick up later fixes to those terms, and it is a snapshot
for pairing with one checkpoint rather than a package to develop in. When what
you want is the living package at the run's commit, that is what
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
  """Every implementation, inlined once even when several terms share it."""
  lines = [
      '"""Implementations of every term this policy trained under.',
      "",
      "Written by `rlmcp env export`. Each function is the source that was",
      "running in the training process, captured from the live managers -- not",
      "an import, so this module does not need the task package installed.",
      "",
      "It is a snapshot, not a package: it will not pick up later fixes to",
      "these terms. Pair it with the checkpoint it was exported beside.",
  ]
  if snapshot.get("task"):
    lines += ["", f"task: {snapshot['task']}"]
  lines += ['"""', "", "from __future__ import annotations", "",
            "import torch  # noqa: F401 - terms are written against it.", ""]

  body: list[str] = []
  seen: set = set()
  # Actions are deliberately absent: an action term is a backend class named
  # by its config, not a function with source of its own, so there is nothing
  # here to inline and nothing missing when there is no `func`.
  for term in _iter_terms(snapshot, "rewards"):
    _emit_source(term, "reward", body, seen, rendered)
  for group, spec in (snapshot.get("observations") or {}).items():
    for term in spec.get("terms") or []:
      _emit_source(term, f"observation {group}", body, seen, rendered)

  if not body:
    body = ["# No term source was captured for this run.", ""]
  return "\n".join([*lines, "", *body]).rstrip() + "\n"


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
  blocks.append(_render_rewards(snapshot, final, imports, unrendered))
  blocks.append(_render_observations(snapshot, imports, unrendered))
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


def _render_rewards(snapshot, final, imports, unrendered) -> str:
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
    func = (term.get("func") or {}).get("name") or name
    extra = f", params={params}" if params not in ("{}", "None") else ""
    lines.append(
        f"  {name}: RewTerm = field(default_factory=lambda: RewTerm("
        f"func={TERMS_STEM}.{func}, weight={weight!r}{extra}))")
  return "\n".join(lines) + "\n\n"


def _render_observations(snapshot, imports, unrendered) -> str:
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
      func = (term.get("func") or {}).get("name") or name
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
  if rendered.missing_source:
    lines += ["", "## Terms whose source could not be captured", "",
              "These are **missing from the export** and must be supplied by",
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
