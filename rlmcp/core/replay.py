"""What a checkpoint does not remember: the task it was trained on.

A saved policy is weights and an iteration number. It says nothing about the
rung of the curriculum it was climbing, and a task's ``play`` configuration is
by construction rung zero -- so replaying a late checkpoint against a fresh
config shows a policy failing at a task it was never asked to do. That failure
looks exactly like a bad policy, which is the worst kind of wrong evidence: it
is legible, it is confident, and it accuses the wrong thing.

The session already wrote down everything needed to reconstruct the conditions.
Every curriculum entry logged the parameters it set and the commands it ran;
every live edit logged its key and its new value. Replaying that log in order
puts the environment back where the policy left it.

This module only reads and orders. It has no simulator, no array library and no
backend import, so the reconstruction can be tested for what it is -- a parse
and a fold over an event log -- without a GPU in the room.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Step:
  """One thing that was done to the environment, and when.

  ``kind`` is ``"parameter"`` (set ``key`` to ``value``) or ``"command"`` (call
  ``key`` with ``value`` as keyword arguments). Keeping both in one ordered
  sequence matters: a stage that sets a reward weight and then runs a command
  that reads it is not the same run as one that does it the other way round.
  """

  kind: str
  key: str
  value: Any
  iteration: int = 0
  stage: str = ""
  source: str = ""

  def describe(self) -> str:
    if self.kind == "parameter":
      return f"{self.key} = {self.value!r}"
    args = ", ".join(f"{k}={v!r}" for k, v in (self.value or {}).items())
    return f"{self.key}({args})"


@dataclass(frozen=True)
class Conditions:
  """The environment state a checkpoint was trained under, as replayable steps."""

  steps: tuple[Step, ...] = ()
  stage: str = ""
  """The curriculum stage this reconstruction stops at ('' if the run had none)."""
  stage_names: tuple[str, ...] = ()
  iteration: int = 0
  """Iteration of the last event folded in -- how far through the run this is."""
  warnings: tuple[str, ...] = ()
  """Things the log could not answer. Never raised: a partial replay beats none."""

  @property
  def parameters(self) -> dict[str, Any]:
    """Final value of every parameter the log touched, last write winning."""
    out: dict[str, Any] = {}
    for step in self.steps:
      if step.kind == "parameter":
        out[step.key] = step.value
    return out

  @property
  def calls(self) -> list[tuple[str, dict[str, Any]]]:
    """Every command the log replays, in order."""
    return [(s.key, dict(s.value or {})) for s in self.steps if s.kind == "command"]

  def summary(self) -> dict[str, Any]:
    return {
        "stage": self.stage or None,
        "stage_names": list(self.stage_names),
        "through_iteration": self.iteration,
        "parameters": self.parameters,
        "calls": [{"cmd": cmd, "args": args} for cmd, args in self.calls],
        "num_steps": len(self.steps),
        "warnings": list(self.warnings),
    }


def parse_action(text: str) -> tuple[str, dict[str, Any]]:
  """Recover a command and its arguments from a logged stage action.

  A stage's actions are written to the event log as ``Action.describe()``
  prose -- ``set_difficulty(level=2, mode='hard')``. That prose happens to be a
  Python call with literal arguments, so it parses exactly rather than
  approximately, which is what makes an old log replayable at all.

  Raises ``ValueError`` on anything that is not a call of literals.
  """
  cleaned = text.strip()
  if not cleaned:
    raise ValueError("Empty action.")
  if "(" not in cleaned:
    # A no-argument command describes as its bare name; anything else with no
    # arguments is not a command at all, and passing it on would only turn a
    # parse problem into an "unknown command" further down.
    if not cleaned.isidentifier():
      raise ValueError(f"Not a command name: {text!r}")
    return cleaned, {}

  try:
    node = ast.parse(cleaned, mode="eval").body
  except SyntaxError as exc:
    raise ValueError(f"Not a parseable call: {text!r} ({exc.msg})") from exc
  if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
    raise ValueError(f"Not a plain command call: {text!r}")
  if node.args:
    # describe() only ever emits keywords, so positionals mean this string came
    # from somewhere else and we would be guessing at the parameter names.
    raise ValueError(f"Positional arguments cannot be replayed: {text!r}")

  args: dict[str, Any] = {}
  for keyword in node.keywords:
    if keyword.arg is None:
      raise ValueError(f"**kwargs cannot be replayed: {text!r}")
    try:
      args[keyword.arg] = ast.literal_eval(keyword.value)
    except ValueError as exc:
      raise ValueError(
          f"Argument '{keyword.arg}' in {text!r} is not a literal: {exc}"
      ) from exc
  return node.func.id, args


def read_events(session_dir: Path | str) -> list[dict[str, Any]]:
  """Load a session's event log, skipping any line that is not whole JSON.

  A run killed mid-write leaves a torn last line. That is not a reason to
  refuse to reconstruct the 4000 iterations in front of it.
  """
  path = Path(session_dir) / "events.jsonl"
  if not path.exists():
    return []
  events: list[dict[str, Any]] = []
  for raw in path.read_text(errors="replace").splitlines():
    line = raw.strip()
    if not line:
      continue
    try:
      event = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(event, dict):
      events.append(event)
  return events


def stage_names(session_dir: Path | str) -> list[str]:
  """Every stage this run entered, in the order it entered them."""
  seen: list[str] = []
  for event in read_events(session_dir):
    if event.get("kind") != "curriculum_stage":
      continue
    name = str(event.get("to") or "")
    if name and name not in seen:
      seen.append(name)
  return seen


# Where a run's stage ladder is written down, best first: beside the session,
# or in the run directory's ``params/`` alongside the rest of its config.
_LADDER_PATHS = (
    ("curriculum.json", False),
    ("params/curriculum.json", True),
)


def read_ladder(session_dir: Path | str) -> dict[str, dict[str, Any]]:
  """The run's curriculum stages as data, keyed by name.

  The ladder is what was *planned*; the event log is what *happened*. This is
  the better source for a stage's arguments -- it is JSON rather than a repr --
  but only the log knows which stages the run actually entered, and in what
  order, so the two are used together rather than one instead of the other.
  """
  session_dir = Path(session_dir)
  for name, from_parent in _LADDER_PATHS:
    path = (session_dir.parent if from_parent else session_dir) / name
    if not path.exists():
      continue
    try:
      loaded = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
      continue
    # Written either as a bare list of stages or as StageSchedule.to_dict().
    stages = loaded.get("stages") if isinstance(loaded, dict) else loaded
    if not isinstance(stages, list):
      continue
    return {
        str(stage["name"]): stage
        for stage in stages
        if isinstance(stage, dict) and stage.get("name")
    }
  return {}


def read_conditions(
    session_dir: Path | str,
    stage: str | None = None,
    *,
    include_parameter_edits: bool = True,
) -> Conditions:
  """Fold a session's log into the environment state at a point in the run.

  ``stage`` stops the fold at the end of the named curriculum stage, which is
  what you want when replaying a checkpoint saved during it. The default folds
  the whole log, giving the state the run ended in.

  Live parameter edits are included by default because a run that was steered
  by hand was steered for a reason, and a clip that ignores those edits shows a
  policy under conditions nobody trained it on.
  """
  events = read_events(session_dir)
  ladder = read_ladder(session_dir)
  names = [
      str(e.get("to") or "")
      for e in events
      if e.get("kind") == "curriculum_stage" and e.get("to")
  ]
  ordered_names = list(dict.fromkeys(names))

  if stage and stage not in ordered_names:
    raise KeyError(
        f"This run never entered a stage named '{stage}'. It entered: "
        + (", ".join(ordered_names) if ordered_names else "(no stages)")
    )

  steps: list[Step] = []
  warnings: list[str] = []
  current_stage = ""
  last_iteration = 0

  for event in events:
    kind = event.get("kind")
    iteration = int(event.get("iteration") or 0)

    if kind == "curriculum_stage":
      if stage and current_stage == stage:
        break  # We are leaving the stage we were asked to stop at.
      current_stage = str(event.get("to") or "")
      last_iteration = iteration
      applied = event.get("applied") or {}
      for key, value in (applied.get("parameters") or {}).items():
        steps.append(Step("parameter", key, value, iteration, current_stage, "stage"))
      for cmd, args in _stage_calls(applied, current_stage, ladder, warnings):
        steps.append(Step("command", cmd, args, iteration, current_stage, "stage"))
      for key, message in (applied.get("parameter_errors") or {}).items():
        warnings.append(
            f"stage '{current_stage}' could not set {key} during training: {message}"
        )
      for cmd, message in (applied.get("action_errors") or {}).items():
        warnings.append(
            f"stage '{current_stage}' could not run {cmd} during training: {message}"
        )
    elif kind == "set_parameter" and include_parameter_edits:
      if not event.get("applied", True):
        continue  # A refused write changed nothing.
      key = str(event.get("key") or "")
      if not key:
        continue
      last_iteration = max(last_iteration, iteration)
      steps.append(
          Step("parameter", key, event.get("new"), iteration, current_stage, "edit")
      )

  if not events:
    warnings.append(
        f"No event log at {Path(session_dir) / 'events.jsonl'}; the environment "
        "will run at the task's own play configuration, which is rung zero of "
        "any curriculum."
    )

  return Conditions(
      steps=tuple(steps),
      stage=stage or current_stage,
      stage_names=tuple(ordered_names),
      iteration=last_iteration,
      warnings=tuple(warnings),
  )


def _stage_calls(
    applied: dict[str, Any],
    stage: str,
    ladder: dict[str, dict[str, Any]],
    warnings: list[str],
) -> list[tuple[str, dict[str, Any]]]:
  """A stage's actions as data, from the best source that has them.

  Three sources, each a fallback for the last. The event log's ``calls`` is
  exact -- it is what ran, arguments and all. The ladder is nearly as good and
  is real JSON, so it survives arguments whose repr is not a literal. Parsing
  the logged prose is the last resort, and the only one that can fail; it is
  also the only one available for a run recorded before either of the others
  existed, which is why it is here at all.
  """
  structured = applied.get("calls")
  if isinstance(structured, list) and structured:
    return [
        (str(entry["cmd"]), dict(entry.get("args") or {}))
        for entry in structured
        if isinstance(entry, dict) and entry.get("cmd")
    ]

  logged = [str(text) for text in applied.get("actions") or []]
  if not logged:
    return []

  planned = (ladder.get(stage) or {}).get("apply")
  if isinstance(planned, list) and len(planned) == len(logged):
    # Same count means the log and the plan describe the same actions; a
    # mismatch means the run diverged from its ladder and the log is the only
    # honest account, so fall through to parsing it.
    return [
        (str(entry["cmd"]), dict(entry.get("args") or {}))
        for entry in planned
        if isinstance(entry, dict) and entry.get("cmd")
    ]

  out: list[tuple[str, dict[str, Any]]] = []
  for text in logged:
    try:
      out.append(parse_action(text))
    except ValueError as exc:
      warnings.append(f"stage '{stage}': {exc}")
  return out


def apply_conditions(lab: Any, conditions: Conditions) -> dict[str, Any]:
  """Replay conditions onto a wrapped environment, in order.

  ``lab`` is an :class:`~rlmcp.core.controller.RlMcp`. Failures are collected
  rather than raised: a parameter this build of the task no longer has is worth
  saying out loud, but it is not worth throwing away the clip.
  """
  applied_parameters: dict[str, Any] = {}
  applied_calls: list[str] = []
  errors: list[str] = []
  kinds: set = set()

  for step in conditions.steps:
    if step.kind == "parameter":
      # The registry refuses a write it cannot honour either by returning
      # False or by raising -- an unregistered key, a value out of range, a
      # parameter whose liveness means the write could never take effect. All
      # three mean the same thing here: this condition was not restored.
      try:
        ok = lab.parameters.set_value(step.key, step.value)
        detail = "refused by the parameter registry"
      except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}"
      if ok:
        applied_parameters[step.key] = step.value
      else:
        errors.append(f"{step.describe()}: {detail}")
        kinds.add("parameter")
      continue

    try:
      lab.run_command(step.key, **(step.value or {}))
      applied_calls.append(step.describe())
    except KeyError as exc:
      # The command does not exist here at all: the package that defines this
      # task's vocabulary was never imported.
      errors.append(f"{step.describe()}: {exc}")
      kinds.add("missing_command")
    except TypeError as exc:
      # The command exists but no longer takes these arguments -- the code has
      # moved on since the run. Show what it takes now, so the difference is
      # visible rather than merely asserted.
      errors.append(f"{step.describe()}: {exc}{_signature_of(lab, step.key)}")
      kinds.add("changed_command")
    except Exception as exc:
      errors.append(f"{step.describe()}: {type(exc).__name__}: {exc}")
      kinds.add("failed_command")

  return {
      "parameters": applied_parameters,
      "calls": applied_calls,
      "errors": errors,
      "error_kinds": sorted(kinds),
      "warnings": list(conditions.warnings),
  }


def _signature_of(lab: Any, cmd: str) -> str:
  """What the command accepts today, for comparing against what was logged."""
  import inspect

  try:
    handler = lab._handlers[cmd]
    accepted = [
        name for name in inspect.signature(handler).parameters if name != "self"
    ]
  except Exception:
    return ""
  return f" (it now takes: {', '.join(accepted) or 'no arguments'})"


def parse_overrides(items: Sequence[str] | None) -> dict[str, Any]:
  """Turn ``key=value`` strings into typed parameter edits.

  JSON first, so ``reward.effort.weight=-0.2``, ``enabled=false`` and
  ``command.twist.ranges.lin_vel_x=[1.1,1.7]`` all mean what they look like;
  anything else stays a string.
  """
  out: dict[str, Any] = {}
  for item in items or []:
    if "=" not in item:
      raise ValueError(f"Expected key=value, got '{item}'")
    key, raw = item.split("=", 1)
    try:
      out[key.strip()] = json.loads(raw)
    except json.JSONDecodeError:
      out[key.strip()] = raw
  return out


def with_overrides(conditions: Conditions, overrides: dict[str, Any]) -> Conditions:
  """Append explicit parameter edits, which therefore win over the replay."""
  if not overrides:
    return conditions
  extra = tuple(
      Step("parameter", key, value, conditions.iteration, conditions.stage, "override")
      for key, value in overrides.items()
  )
  return Conditions(
      steps=conditions.steps + extra,
      stage=conditions.stage,
      stage_names=conditions.stage_names,
      iteration=conditions.iteration,
      warnings=conditions.warnings,
  )
