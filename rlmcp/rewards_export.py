"""Getting a reward term the agent invented out of the session and into the task.

A term added mid-run lives in two places while the run is going: as a function
object inside the training process, and as text in ``<session>/rewards/``. When
the process exits, only the text is left -- and text in a session directory is
not a reward the next run has. The task package still does not define the term,
its config still does not list it, and the improvement is a footnote in an
event log.

This module closes that gap. It reads the session's ``add_reward_term`` events
and the sources beside them, and writes the two things a task package actually
needs:

* an **implementation** module -- the functions, ready to be imported by (or
  pasted into) the task's ``mdp`` package;
* a **config** snippet -- one ``RewTerm`` line per term, with the weight and
  params the run actually used, ready to go into its ``RewardsCfg``.

It stops there, deliberately. Editing the task's own files is a job for
whoever owns that repository: rlmcp does not know which of several packages a
term belongs to, which ``mdp`` module re-exports it, or whether the run that
produced it was any good. What it does know is exactly what the term was, and
that is what it hands over -- see ``AGENTS.md`` on why nothing task-shaped is
allowed to live in this tree.

The weights written are the ones **in force at export**, not the ones the term
was added with: a term added at 0.5 and tuned to 3.0 over the next hundred
iterations should arrive in the config at 3.0, since that is the run being
reproduced. The originally-added weight stays in the header comment.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rlmcp.session import Session

MODULE_STEM = "added_rewards"
CFG_STEM = "added_rewards_cfg"


@dataclass
class AddedReward:
  """One term an agent added during a run, as the export needs it."""

  name: str
  func_name: str
  weight: float
  params: Dict[str, Any] = field(default_factory=dict)
  digest: str = ""
  iteration: int = 0
  rationale: str = ""
  source: str = ""
  added_weight: Optional[float] = None
  """The weight it was added with, when tuning has since moved it."""


def _strip_header(text: str) -> str:
  """Drop the module docstring rlmcp wrote, leaving the agent's own source.

  Parsed rather than pattern-matched: the header is a docstring, and a
  docstring that happens to contain ``\"\"\"`` in the rationale would defeat
  anything cruder.
  """
  try:
    tree = ast.parse(text)
  except SyntaxError:
    return text
  body = tree.body
  if not body:
    return text
  first = body[0]
  is_docstring = (
      isinstance(first, ast.Expr)
      and isinstance(getattr(first, "value", None), ast.Constant)
      and isinstance(first.value.value, str)
  )
  if not is_docstring or len(body) < 2:
    return text
  start = body[1].lineno - 1
  # Keep any decorator, which sits above the def it decorates.
  decorators = getattr(body[1], "decorator_list", []) or []
  if decorators:
    start = min(start, min(d.lineno for d in decorators) - 1)
  return "\n".join(text.splitlines()[start:]).strip() + "\n"


def collect_added_rewards(session: Session) -> List[AddedReward]:
  """Every reward term added during this run, oldest first.

  Later events for the same name win, which matters only for a session that
  was replayed into: the term the run ended with is the term to export.
  """
  live_weights = _live_weights(session)
  by_name: Dict[str, AddedReward] = {}
  for event in session.events() or []:
    if event.get("kind") != "add_reward_term":
      continue
    detail = event.get("detail") or event
    name = detail.get("name")
    if not name:
      continue
    added_weight = float(detail.get("weight", 0.0) or 0.0)
    weight = live_weights.get(f"reward.{name}.weight", added_weight)
    source = ""
    path = detail.get("source_path")
    if path:
      candidate = Path(path)
      if not candidate.exists():
        # A session directory that moved: the file is still beside the events.
        candidate = session.rewards / f"{name}.py"
      if candidate.exists():
        source = _strip_header(candidate.read_text())
    by_name[name] = AddedReward(
        name=name,
        func_name=detail.get("function") or name,
        weight=float(weight),
        params=dict(detail.get("params") or {}),
        digest=detail.get("digest", ""),
        iteration=int(detail.get("iteration", 0) or 0),
        rationale=detail.get("rationale", ""),
        source=source,
        added_weight=added_weight if added_weight != weight else None,
    )
  return list(by_name.values())


def _live_weights(session: Session) -> Dict[str, float]:
  """Reward weights as the parameter snapshot last saw them."""
  out: Dict[str, float] = {}
  for key, spec in (session.params() or {}).items():
    if not key.startswith("reward.") or not key.endswith(".weight"):
      continue
    value = spec.get("current") if isinstance(spec, dict) else spec
    if isinstance(value, (int, float)) and not isinstance(value, bool):
      out[key] = float(value)
  return out


def render_module(rewards: List[AddedReward], *, task: str = "") -> str:
  """The implementation file: every added function, in one module."""
  lines = [
      '"""Reward terms added by an agent during a training run.',
      "",
      "Written by `rlmcp rewards export` from the run's session. Move these",
      "into the task's own mdp package -- they are the implementation half of",
      f"the terms listed in {CFG_STEM}.py.",
  ]
  if task:
    lines += ["", f"task: {task}"]
  # `torch` is in scope when a term is compiled, so agent source uses it
  # without importing it. The exported module has no such help, and a file
  # that NameErrors on import is not an implementation anybody can move.
  lines += ['"""', "", "from __future__ import annotations", "",
            "import torch  # noqa: F401 - in scope when these were compiled.",
            "", ""]
  for reward in rewards:
    lines.append(f"# Reward term '{reward.name}', added at iteration "
                 f"{reward.iteration} (digest {reward.digest}).")
    if reward.rationale:
      lines.append(f"# {reward.rationale}")
    lines.append(reward.source.strip())
    lines += ["", ""]
  return "\n".join(lines).rstrip() + "\n"


def render_cfg(rewards: List[AddedReward], *, module: str = MODULE_STEM) -> str:
  """The config half: one ``RewTerm`` line per term, weights as they stand."""
  lines = [
      '"""Config lines for the reward terms in '
      f'{module}.py.',
      "",
      "Paste these fields into the task's RewardsCfg. The import is written",
      f"against `{module}` as a sibling module; point it at wherever the",
      "implementations ended up.",
      "",
      "Uncomment the RewTerm import your backend uses -- the two managers take",
      "the same three fields, so the lines below are identical either way.",
      '"""',
      "",
      "# mjlab:",
      "#   from mjlab.managers.manager_term_config import RewardTermCfg as RewTerm",
      "# IsaacLab:",
      "#   from isaaclab.managers import RewardTermCfg as RewTerm",
      "",
      f"from . import {module}",
      "",
      "",
  ]
  for reward in rewards:
    if reward.added_weight is not None:
      lines.append(
          f"# '{reward.name}' was added at weight {reward.added_weight!r} and "
          f"tuned to {reward.weight!r} during the run.")
    params = f", params={reward.params!r}" if reward.params else ""
    lines.append(
        f"{reward.name} = RewTerm(func={module}.{reward.func_name}, "
        f"weight={reward.weight!r}{params})")
  return "\n".join(lines).rstrip() + "\n"


def export_added_rewards(
    session: Session,
    out_dir: Path | str,
    *,
    task: str = "",
) -> Dict[str, Any]:
  """Write the implementation module and the config snippet into ``out_dir``.

  Returns a payload naming what was written, or ``{"count": 0}`` with no files
  when the run added no terms -- which is the common case and not an error.
  """
  rewards = collect_added_rewards(session)
  if not rewards:
    return {"count": 0, "rewards": [], "files": [],
            "session": str(session.dir)}

  directory = Path(out_dir).expanduser()
  directory.mkdir(parents=True, exist_ok=True)
  module_path = directory / f"{MODULE_STEM}.py"
  cfg_path = directory / f"{CFG_STEM}.py"
  module_path.write_text(render_module(rewards, task=task))
  cfg_path.write_text(render_cfg(rewards))

  return {
      "count": len(rewards),
      "session": str(session.dir),
      "implementation": str(module_path),
      "config": str(cfg_path),
      "files": [str(module_path), str(cfg_path)],
      "rewards": [
          {
              "name": r.name,
              "function": r.func_name,
              "weight": r.weight,
              "added_weight": r.added_weight,
              "params": r.params,
              "iteration": r.iteration,
              "digest": r.digest,
              "rationale": r.rationale,
          }
          for r in rewards
      ],
  }


def describe(payload: Dict[str, Any]) -> str:
  """A human line for the CLI, since the JSON is for the agent."""
  if not payload.get("count"):
    return "No reward terms were added during this run; nothing to export."
  names = ", ".join(r["name"] for r in payload["rewards"])
  return (
      f"{payload['count']} added reward term(s) -- {names}\n"
      f"  implementation: {payload['implementation']}\n"
      f"  config lines:   {payload['config']}\n"
      "Move the implementation into the task's mdp package and paste the "
      "config lines into its RewardsCfg."
  )


__all__ = [
    "AddedReward",
    "CFG_STEM",
    "MODULE_STEM",
    "collect_added_rewards",
    "describe",
    "export_added_rewards",
    "render_cfg",
    "render_module",
]
