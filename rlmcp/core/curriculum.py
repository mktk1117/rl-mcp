"""Staged curriculum: what the policy faces, and when it graduates.

A stage says two things: what changes when the run enters it, and what has to
be true before the run leaves it.

What changes is expressed in the only two vocabularies the core has -- parameter
edits, and commands. Since commands include whatever the environment's
extensions contribute, a locomotion stage that unlocks terrain and a
manipulation stage that widens the object set are the same kind of object here.
Nothing in this module knows which is which.

Promotion is condition-based rather than iteration-based: the run advances when
the policy has actually earned it, and an agent watching the run can override
the decision at any point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
}


@dataclass
class Condition:
  """A single promotion test against a published metric."""

  metric: str
  op: str
  value: float

  def check(self, metrics: Dict[str, float]) -> Tuple[bool, Optional[float]]:
    current = metrics.get(self.metric)
    if not isinstance(current, (int, float)):
      return False, None
    fn = _OPS.get(self.op)
    if fn is None:
      raise ValueError(f"Unsupported comparison '{self.op}' (use one of {sorted(_OPS)})")
    return bool(fn(float(current), float(self.value))), float(current)

  def describe(self) -> str:
    return f"{self.metric} {self.op} {self.value}"

  def to_dict(self) -> Dict[str, Any]:
    return {"metric": self.metric, "op": self.op, "value": self.value}

  @staticmethod
  def from_dict(d: Dict[str, Any]) -> "Condition":
    return Condition(metric=d["metric"], op=d.get("op", ">="), value=float(d["value"]))


@dataclass
class Action:
  """A command to run when a stage is entered.

  ``cmd`` is any command the controller knows, including ones contributed by an
  extension -- which is how a stage changes something the core has no concept
  of.
  """

  cmd: str
  args: Dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> Dict[str, Any]:
    return {"cmd": self.cmd, "args": self.args}

  @staticmethod
  def from_dict(d: Dict[str, Any]) -> "Action":
    return Action(cmd=d["cmd"], args=dict(d.get("args") or {}))

  def describe(self) -> str:
    if not self.args:
      return self.cmd
    rendered = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
    return f"{self.cmd}({rendered})"


@dataclass
class CurriculumStage:
  """One rung of the ladder."""

  name: str
  parameters: Dict[str, Any] = field(default_factory=dict)
  """Parameter edits applied once on entry, e.g. ``{"reward.foot_slip.weight": -0.2}``."""
  apply: List[Action] = field(default_factory=list)
  """Commands run once on entry, in order."""
  promote_when: List[Condition] = field(default_factory=list)
  min_iterations: int = 100
  """Never promote before this many iterations in the stage."""
  hold_iterations: int = 20
  """Conditions must hold for this many consecutive iterations."""
  notes: str = ""

  def to_dict(self) -> Dict[str, Any]:
    return {
        "name": self.name,
        "parameters": self.parameters,
        "apply": [a.to_dict() for a in self.apply],
        "promote_when": [c.to_dict() for c in self.promote_when],
        "min_iterations": self.min_iterations,
        "hold_iterations": self.hold_iterations,
        "notes": self.notes,
    }

  @staticmethod
  def from_dict(d: Dict[str, Any]) -> "CurriculumStage":
    return CurriculumStage(
        name=d["name"],
        parameters=dict(d.get("parameters") or {}),
        apply=[Action.from_dict(a) for a in d.get("apply") or []],
        promote_when=[Condition.from_dict(c) for c in d.get("promote_when") or []],
        min_iterations=int(d.get("min_iterations", 100)),
        hold_iterations=int(d.get("hold_iterations", 20)),
        notes=d.get("notes", ""),
    )


class StageSchedule:
  """Ordered stages plus the bookkeeping that decides when to move on."""

  def __init__(self, stages: Sequence[CurriculumStage], auto_promote: bool = True):
    if not stages:
      raise ValueError("A curriculum needs at least one stage.")
    self.stages: List[CurriculumStage] = list(stages)
    self.auto_promote = bool(auto_promote)
    self.index = 0
    self.entered_at_iteration = 0
    self.streak = 0
    self.history: List[Dict[str, Any]] = []

  # Access.

  @property
  def current(self) -> CurriculumStage:
    return self.stages[self.index]

  @property
  def is_final(self) -> bool:
    return self.index >= len(self.stages) - 1

  def find(self, name: str) -> Optional[int]:
    for i, stage in enumerate(self.stages):
      if stage.name == name:
        return i
    return None

  # Evaluation.

  def evaluate(self, iteration: int, metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Advance a stage if this iteration's metrics say the policy has earned it.

    Returns a description of the transition, or ``None`` when staying put.
    """
    stage = self.current
    progress = self.check(iteration, metrics)
    if progress["satisfied"]:
      self.streak += 1
    else:
      self.streak = 0

    if not self.auto_promote or self.is_final:
      return None
    if iteration - self.entered_at_iteration < stage.min_iterations:
      return None
    if self.streak < max(1, stage.hold_iterations):
      return None
    return self.advance(iteration, reason="auto: promotion conditions met")

  def check(self, iteration: int, metrics: Dict[str, float]) -> Dict[str, Any]:
    """Report on each promotion condition without changing state."""
    stage = self.current
    checks = []
    satisfied = True
    for cond in stage.promote_when:
      ok, current = cond.check(metrics)
      satisfied = satisfied and ok
      checks.append(
          {
              "condition": cond.describe(),
              "current": None if current is None else round(current, 4),
              "met": ok,
          }
      )
    if not stage.promote_when:
      satisfied = False  # No conditions means "never auto-promote".
    return {
        "stage": stage.name,
        "iterations_in_stage": iteration - self.entered_at_iteration,
        "min_iterations": stage.min_iterations,
        "hold_iterations": stage.hold_iterations,
        "streak": self.streak,
        "satisfied": satisfied,
        "checks": checks,
    }

  # Transitions.

  def advance(self, iteration: int, reason: str = "") -> Optional[Dict[str, Any]]:
    if self.is_final:
      return None
    return self.goto(self.index + 1, iteration, reason=reason or "advance")

  def retreat(self, iteration: int, reason: str = "") -> Optional[Dict[str, Any]]:
    """Step back one stage. Not wired to a command; agents use ``curriculum_goto``."""
    if self.index == 0:
      return None
    return self.goto(self.index - 1, iteration, reason=reason or "retreat")

  def goto(self, index: int, iteration: int, reason: str = "") -> Dict[str, Any]:
    if not 0 <= index < len(self.stages):
      raise IndexError(f"Stage index {index} out of range (0..{len(self.stages) - 1})")
    previous = self.current.name
    self.index = index
    self.entered_at_iteration = iteration
    self.streak = 0
    transition = {
        "from": previous,
        "to": self.current.name,
        "iteration": iteration,
        "reason": reason,
    }
    self.history.append(transition)
    return transition

  def goto_named(self, name: str, iteration: int, reason: str = "") -> Dict[str, Any]:
    index = self.find(name)
    if index is None:
      raise KeyError(
          f"No stage named '{name}'. Available: {[s.name for s in self.stages]}"
      )
    return self.goto(index, iteration, reason=reason or f"manual jump to {name}")

  # Serialisation.

  def status(self, iteration: Optional[int] = None) -> Dict[str, Any]:
    stage = self.current
    out: Dict[str, Any] = {
        "stage": stage.name,
        "stage_index": self.index,
        "num_stages": len(self.stages),
        "is_final": self.is_final,
        "auto_promote": self.auto_promote,
        "entered_at_iteration": self.entered_at_iteration,
        "streak": self.streak,
        "applies": [a.describe() for a in stage.apply],
        "parameters": stage.parameters,
        "notes": stage.notes,
        "promote_when": [c.describe() for c in stage.promote_when],
        "stage_names": [s.name for s in self.stages],
        "history": self.history[-10:],
    }
    if iteration is not None:
      out["iterations_in_stage"] = iteration - self.entered_at_iteration
    return out

  def to_dict(self) -> Dict[str, Any]:
    return {
        "stages": [s.to_dict() for s in self.stages],
        "auto_promote": self.auto_promote,
        "index": self.index,
        "entered_at_iteration": self.entered_at_iteration,
        "history": self.history,
    }

  @staticmethod
  def from_dict(d: Dict[str, Any]) -> "StageSchedule":
    schedule = StageSchedule(
        [CurriculumStage.from_dict(s) for s in d["stages"]],
        auto_promote=bool(d.get("auto_promote", True)),
    )
    schedule.index = int(d.get("index", 0))
    schedule.entered_at_iteration = int(d.get("entered_at_iteration", 0))
    schedule.history = list(d.get("history") or [])
    return schedule


# Metric names rlmcp publishes for itself, so promotion rules do not depend on
# the simulator's own logging conventions. Extensions may publish more.
METRIC_EPISODE_LENGTH_FRAC = "rlmcp/episode_length_frac"
