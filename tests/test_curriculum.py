"""Stage promotion rules."""

from __future__ import annotations

import pytest

from rlmcp.core.curriculum import (
    METRIC_EPISODE_LENGTH_FRAC,
    Action,
    Condition,
    CurriculumStage,
    StageSchedule,
)

PROGRESS = "task/progress"
PASSING = {METRIC_EPISODE_LENGTH_FRAC: 0.9, PROGRESS: 0.9}
FAILING = {METRIC_EPISODE_LENGTH_FRAC: 0.1, PROGRESS: 0.1}

STAGE_NAMES = ["0_easy", "1_medium", "2_hard", "3_full"]


def _plan(**kwargs) -> StageSchedule:
  """A four-rung ladder in the core's own vocabulary: parameters and commands."""
  defaults = dict(min_iterations=5, hold_iterations=2, auto_promote=True)
  defaults.update(kwargs)
  auto = defaults.pop("auto_promote")
  stages = []
  for i, name in enumerate(STAGE_NAMES):
    last = i == len(STAGE_NAMES) - 1
    stages.append(
        CurriculumStage(
            name=name,
            parameters={"reward.penalty.weight": -0.1 * (i + 1)},
            apply=[Action("widen_task", {"level": i + 1})],
            promote_when=[]
            if last
            else [
                Condition(METRIC_EPISODE_LENGTH_FRAC, ">=", 0.5),
                Condition(PROGRESS, ">=", 0.6),
            ],
            **defaults,
        )
    )
  return StageSchedule(stages, auto_promote=auto)


def test_a_stage_carries_parameters_and_commands():
  plan = _plan()
  stage = plan.current
  assert stage.parameters == {"reward.penalty.weight": -0.1}
  assert stage.apply[0].cmd == "widen_task"
  assert stage.apply[0].args == {"level": 1}


def test_actions_render_readably_for_a_log():
  assert Action("set_terrain", {"max_level": 4}).describe() == "set_terrain(max_level=4)"
  assert Action("pause").describe() == "pause"


def test_promotion_needs_both_min_iterations_and_a_streak():
  plan = _plan(min_iterations=5, hold_iterations=3)

  # Metrics pass from the start, but the stage floor has not been reached.
  assert plan.evaluate(0, PASSING) is None
  assert plan.evaluate(1, PASSING) is None
  assert plan.evaluate(4, PASSING) is None
  assert plan.current.name == "0_easy"

  transition = plan.evaluate(5, PASSING)  # Floor reached, streak already long.

  assert transition is not None and transition["to"] == "1_medium"


def test_a_long_stage_floor_delays_a_ready_policy():
  plan = _plan(min_iterations=20, hold_iterations=1)
  for iteration in range(20):
    assert plan.evaluate(iteration, PASSING) is None
  assert plan.evaluate(20, PASSING) is not None


def test_a_bad_iteration_resets_the_streak():
  plan = _plan(min_iterations=1, hold_iterations=3)
  plan.evaluate(5, PASSING)
  plan.evaluate(6, PASSING)
  plan.evaluate(7, FAILING)

  assert plan.streak == 0
  assert plan.evaluate(8, PASSING) is None  # Must build the streak again.


def test_auto_promote_off_means_never_advance_by_itself():
  plan = _plan(min_iterations=1, hold_iterations=1, auto_promote=False)
  for it in range(50):
    assert plan.evaluate(it, PASSING) is None
  assert plan.current.name == "0_easy"


def test_final_stage_never_promotes():
  plan = _plan(min_iterations=1, hold_iterations=1)
  for it in range(200):
    plan.evaluate(it, PASSING)
  assert plan.is_final and plan.current.name == STAGE_NAMES[-1]


def test_a_stage_with_an_unknown_command_is_still_a_valid_stage():
  """The core does not validate command names; the controller reports failures."""
  stage = CurriculumStage(name="x", apply=[Action("no_such_command")])
  assert StageSchedule([stage]).current.apply[0].cmd == "no_such_command"


def test_missing_metric_blocks_promotion():
  plan = _plan(min_iterations=1, hold_iterations=1)
  for it in range(20):
    assert plan.evaluate(it, {METRIC_EPISODE_LENGTH_FRAC: 0.9}) is None


def test_manual_jump_forward_and_back():
  plan = _plan()
  plan.goto_named("3_full", iteration=10, reason="operator")
  assert plan.current.name == "3_full"

  plan.retreat(iteration=20, reason="too hard")
  assert plan.current.name == "2_hard"
  assert plan.history[-1]["reason"] == "too hard"

  with pytest.raises(KeyError):
    plan.goto_named("does_not_exist", iteration=21)


def test_round_trips_through_a_dict():
  plan = _plan()
  plan.evaluate(10, PASSING)
  plan.goto_named("2_hard", iteration=11)

  restored = StageSchedule.from_dict(plan.to_dict())

  assert restored.current.name == plan.current.name
  assert restored.current.parameters == plan.current.parameters
  assert [a.to_dict() for a in restored.current.apply] == [
      a.to_dict() for a in plan.current.apply
  ]
  assert restored.entered_at_iteration == plan.entered_at_iteration


def test_check_reports_each_condition():
  plan = StageSchedule(
      [
          CurriculumStage(
              name="a",
              promote_when=[Condition(PROGRESS, ">=", 0.5)],
              min_iterations=0,
          ),
          CurriculumStage(name="b"),
      ]
  )
  report = plan.check(3, {PROGRESS: 0.2})

  assert report["checks"][0]["met"] is False
  assert report["checks"][0]["current"] == 0.2
  assert report["satisfied"] is False


def test_a_stage_with_no_conditions_holds_forever():
  plan = StageSchedule(
      [CurriculumStage(name="a", min_iterations=0), CurriculumStage(name="b")]
  )
  for it in range(50):
    assert plan.evaluate(it, PASSING) is None
