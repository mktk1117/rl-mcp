"""Reading a running environment's terms into something that outlives it.

A checkpoint is half an answer. The other half is the environment it trained
under -- which observations it saw in which order, what its actions meant, what
it was being paid for -- and that half lives only as Python objects inside a
process that is about to exit. Afterwards there is a `.pt` file and a task
package that has since moved on.

So this module reads the three managers that define the policy's world and
writes them down as data:

* **rewards** -- what it was optimising,
* **observations** -- what it saw, per group, in order, with the noise, scale
  and clipping applied to each term,
* **actions** -- what its outputs meant.

Terminations and events are deliberately absent: they shape *training*, not the
policy's interface, and a checkpoint pairs with the interface.

Every term's ``func`` is captured **as source**, via ``inspect.getsource``, not
as an import path. An import path is only as good as the package still being
installed at the version the run used; source is the thing itself. Where source
cannot be read -- a builtin, a lambda in a REPL -- the term is recorded as
unavailable with its module and qualname, and nothing pretends otherwise.

Values are encoded rather than repr'd. A reward param is routinely a
``SceneEntityCfg``, which is a dataclass, and ``repr`` of one is neither
guaranteed to round-trip nor to carry the import it needs. :func:`encode_value`
turns such an object into ``{"__obj__": …}`` naming its class, module and
non-default fields, which the exporter can render back into constructible
source with the right import. The resolved ``*_ids`` fields are dropped when
the matching ``*_names`` is set: those are indices the manager filled in
against one particular scene, and carrying them into a fresh env would pin it
to that scene's ordering.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Dict, List

from rlmcp.core.reward_source import SOURCE_ATTR

SCHEMA_VERSION = 1

MAX_SOURCE_BYTES = 200_000
"""Ceiling on one term's captured source. A term whose source is larger than
this is almost certainly a whole module misidentified as a function; recording
it would bloat every session for no gain."""


def capture_env_terms(env: Any) -> Dict[str, Any]:
  """Snapshot the reward, observation and action terms of ``env``.

  Never raises for a manager that is absent or shaped unexpectedly: the
  section is simply missing or empty, and ``problems`` says why. A capture is
  a convenience taken at run start, and it must not be able to stop a run.
  """
  problems: List[str] = []
  snapshot: Dict[str, Any] = {
      "schema_version": SCHEMA_VERSION,
      "rewards": _capture_rewards(env, problems),
      "observations": _capture_observations(env, problems),
      "actions": _capture_actions(env, problems),
      "problems": problems,
  }
  # The exact term-config classes this backend uses, so the export can import
  # them rather than make the reader choose between two commented guesses.
  snapshot["term_cfg_types"] = _capture_cfg_types(snapshot)
  return snapshot


def _capture_cfg_types(snapshot: Dict[str, Any]) -> Dict[str, Any]:
  """Where each manager's term-config class came from, read off a real term."""
  out: Dict[str, Any] = {}
  for section, key in (("rewards", "reward"), ("actions", "action")):
    for term in snapshot.get(section) or []:
      captured = term.get("_cfg_type")
      if captured:
        out[key] = captured
        break
  for spec in (snapshot.get("observations") or {}).values():
    for term in spec.get("terms") or []:
      captured = term.get("_cfg_type")
      if captured:
        out["observation"] = captured
        break
    if "observation" in out:
      break
  return out


def _cfg_type_of(cfg: Any) -> Dict[str, str]:
  return {"module": type(cfg).__module__, "name": type(cfg).__name__}


# Per-manager capture.


def _capture_rewards(env: Any, problems: List[str]) -> List[Dict[str, Any]]:
  manager = getattr(env, "reward_manager", None)
  if manager is None:
    problems.append("no reward_manager on this environment")
    return []
  out: List[Dict[str, Any]] = []
  for name in list(getattr(manager, "active_terms", []) or []):
    try:
      cfg = manager.get_term_cfg(name)
    except Exception as exc:
      problems.append(f"reward term '{name}' could not be read: {exc}")
      continue
    out.append({
        "name": name,
        "_cfg_type": _cfg_type_of(cfg),
        "weight": _plain_number(getattr(cfg, "weight", None)),
        "params": encode_value(getattr(cfg, "params", {}) or {}),
        "func": capture_callable(getattr(cfg, "func", None), problems,
                                 label=f"reward '{name}'"),
    })
  return out


def _capture_observations(env: Any, problems: List[str]) -> Dict[str, Any]:
  manager = getattr(env, "observation_manager", None)
  if manager is None:
    problems.append("no observation_manager on this environment")
    return {}
  groups: Dict[str, Any] = {}
  active = getattr(manager, "active_terms", {}) or {}
  concatenate = getattr(manager, "group_obs_concatenate", {}) or {}
  dims = getattr(manager, "group_obs_term_dim", {}) or {}
  for group, names in active.items():
    terms: List[Dict[str, Any]] = []
    for index, name in enumerate(list(names or [])):
      try:
        cfg = manager.get_term_cfg(group, name)
      except Exception as exc:
        problems.append(
            f"observation term '{group}.{name}' could not be read: {exc}")
        continue
      entry: Dict[str, Any] = {
          "name": name,
          "_cfg_type": _cfg_type_of(cfg),
          "params": encode_value(getattr(cfg, "params", {}) or {}),
          "func": capture_callable(getattr(cfg, "func", None), problems,
                                   label=f"observation '{group}.{name}'"),
      }
      # Order matters here in a way it does not for rewards: the policy's
      # input vector is these terms concatenated, so a term's dimension is
      # recorded to make a mismatch obvious rather than mysterious.
      group_dims = dims.get(group) or []
      if index < len(group_dims):
        entry["dim"] = list(group_dims[index])
      for field in ("noise", "clip", "scale", "history_length",
                    "flatten_history_dim"):
        value = getattr(cfg, field, None)
        if value not in (None, 0, False):
          entry[field] = encode_value(value)
      terms.append(entry)
    groups[group] = {
        "concatenate": bool(concatenate.get(group, True)),
        "terms": terms,
    }
  return groups


def _capture_actions(env: Any, problems: List[str]) -> List[Dict[str, Any]]:
  manager = getattr(env, "action_manager", None)
  if manager is None:
    problems.append("no action_manager on this environment")
    return []
  out: List[Dict[str, Any]] = []
  for name in list(getattr(manager, "active_terms", []) or []):
    cfg = None
    try:
      term = manager.get_term(name)
      cfg = getattr(term, "cfg", None)
    except Exception as exc:
      problems.append(f"action term '{name}' could not be read: {exc}")
      continue
    entry: Dict[str, Any] = {"name": name, "cfg": encode_value(cfg)}
    if cfg is not None:
      entry["_cfg_type"] = _cfg_type_of(cfg)
    try:
      entry["dim"] = int(manager.action_term_dim[
          list(manager.active_terms).index(name)])
    except Exception:
      pass
    out.append(entry)
  return out


# Callables and values.


def capture_callable(
    func: Any, problems: List[str], *, label: str = "") -> Dict[str, Any]:
  """Describe a term's ``func``: where it came from and, ideally, its source.

  A class-based term arrives here as the *instance* the manager built, so the
  class is what gets captured -- the instance's source is its class's.
  """
  if func is None:
    return {"available": False, "reason": "term has no func"}

  # A term an agent wrote was compiled from a string, so its "file" is a
  # pseudo-name that inspect cannot read. It carries its own source instead --
  # see rlmcp.core.reward_source -- which makes the snapshot self-sufficient
  # rather than something the exporter has to patch up afterwards.
  carried = getattr(func, SOURCE_ATTR, None)
  if isinstance(carried, str) and carried.strip():
    return {
        "module": getattr(func, "__module__", None),
        "qualname": getattr(func, "__qualname__", getattr(func, "__name__", None)),
        "name": getattr(func, "__name__", None),
        "kind": "function",
        "origin": "agent",
        "available": True,
        "source": _dedent(carried),
    }

  target = func
  kind = "function"
  if not (inspect.isfunction(func) or inspect.isclass(func)
          or inspect.ismethod(func)):
    target = type(func)
    kind = "class"
  elif inspect.isclass(func):
    kind = "class"

  info: Dict[str, Any] = {
      "module": getattr(target, "__module__", None),
      "qualname": getattr(target, "__qualname__", getattr(target, "__name__", None)),
      "name": getattr(target, "__name__", None),
      "kind": kind,
  }
  try:
    source = inspect.getsource(target)
  except (OSError, TypeError) as exc:
    info["available"] = False
    info["reason"] = f"{type(exc).__name__}: {exc}"
    if label:
      problems.append(f"{label}: source unavailable ({info['reason']})")
    return info
  if len(source) > MAX_SOURCE_BYTES:
    info["available"] = False
    info["reason"] = f"source is {len(source)} bytes, over the cap"
    return info
  info["available"] = True
  info["source"] = _dedent(source)
  return info


def _dedent(source: str) -> str:
  """Left-align a def that was captured from inside a class or a closure."""
  import textwrap

  return textwrap.dedent(source).rstrip() + "\n"


def _plain_number(value: Any) -> Any:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return value
  return float(value)


#: Fields the manager resolves against one particular scene. Dropped when the
#: matching ``*_names`` field is set, so the exported cfg re-resolves rather
#: than carrying another scene's indices.
_RESOLVED_SUFFIX = "_ids"


def encode_value(value: Any, _depth: int = 0) -> Any:
  """Turn a config value into JSON-safe data the exporter can render back.

  Tuples keep their tuple-ness, dataclasses keep their class and their
  non-default fields, callables become references, and anything else is kept
  as its ``repr`` and flagged so the exporter can say it could not render it
  rather than emitting something that does not run.
  """
  if _depth > 12:
    return {"__repr__": "...", "__unrenderable__": True,
            "reason": "nesting deeper than 12"}
  if value is None or isinstance(value, (bool, int, float, str)):
    return value
  if isinstance(value, tuple):
    return {"__tuple__": [encode_value(v, _depth + 1) for v in value]}
  if isinstance(value, list):
    return [encode_value(v, _depth + 1) for v in value]
  if isinstance(value, dict):
    return {"__map__": {str(k): encode_value(v, _depth + 1)
                        for k, v in value.items()}}
  if isinstance(value, slice):
    # `slice(None)` is the unresolved default of every SceneEntityCfg id field.
    if value == slice(None):
      return {"__slice_all__": True}
    return {"__repr__": repr(value), "__unrenderable__": True,
            "reason": "a non-trivial slice"}
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return _encode_dataclass(value, _depth)
  if inspect.isfunction(value) or inspect.isclass(value) or inspect.ismethod(value):
    return {"__ref__": {
        "module": getattr(value, "__module__", None),
        "qualname": getattr(value, "__qualname__", getattr(value, "__name__", None)),
    }}
  return {"__repr__": repr(value), "__unrenderable__": True,
          "reason": f"a {type(value).__name__}"}


def _encode_dataclass(value: Any, depth: int) -> Dict[str, Any]:
  fields: Dict[str, Any] = {}
  names_set = {
      f.name for f in dataclasses.fields(value)
      if f.name.endswith("_names") and getattr(value, f.name, None) is not None
  }
  for field in dataclasses.fields(value):
    if not field.init:
      continue
    current = getattr(value, field.name, None)
    # An id field the manager resolved, whose names counterpart is set: those
    # indices belong to the scene that resolved them, not to this config.
    if field.name.endswith(_RESOLVED_SUFFIX):
      counterpart = field.name[: -len(_RESOLVED_SUFFIX)] + "_names"
      if counterpart in names_set:
        continue
    if _is_default(field, current):
      continue
    fields[field.name] = encode_value(current, depth + 1)
  return {"__obj__": {
      "module": type(value).__module__,
      "name": type(value).__name__,
      "fields": fields,
  }}


def _is_default(field: Any, current: Any) -> bool:
  if field.default is not dataclasses.MISSING:
    try:
      return bool(current == field.default)
    except Exception:
      return False
  if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
    try:
      return bool(current == field.default_factory())
    except Exception:
      return False
  return False


__all__ = [
    "SCHEMA_VERSION",
    "SOURCE_ATTR",
    "capture_callable",
    "capture_env_terms",
    "encode_value",
]
