"""Reconstructing the conditions a checkpoint was trained under.

A checkpoint is weights and an iteration; the curriculum rung it was climbing
lives only in the session's event log. Everything here is a fold over that log,
so it is testable for exactly what it is -- a parse and an ordering -- with no
simulator, no GPU and no session from anybody's machine.

The cases worth pinning are the ones where a wrong answer still looks like an
answer: a stage boundary off by one, a live edit silently dropped, a refused
write replayed as if it had taken, a torn last line taking the whole run with
it.
"""

from __future__ import annotations

import json

import pytest

from rlmcp.core.replay import (
    Conditions,
    Step,
    apply_conditions,
    parse_action,
    parse_overrides,
    read_conditions,
    read_events,
    read_ladder,
    stage_names,
    with_overrides,
)


def _session(tmp_path, events, name: str = "session", ladder=None, torn: str = ""):
  """A session directory holding just an event log, and maybe a ladder."""
  d = tmp_path / name
  d.mkdir(parents=True, exist_ok=True)
  text = "".join(json.dumps(e) + "\n" for e in events) + torn
  (d / "events.jsonl").write_text(text)
  if ladder is not None:
    (d / "curriculum.json").write_text(json.dumps(ladder))
  return d


def _stage(to: str, iteration: int, **applied):
  return {"kind": "curriculum_stage", "iteration": iteration, "to": to,
          "applied": applied}


def _edit(key: str, new, iteration: int, applied: bool = True):
  return {"kind": "set_parameter", "iteration": iteration, "key": key,
          "new": new, "applied": applied}


# The fold.


def test_the_last_stage_is_what_a_final_checkpoint_trained_under(tmp_path):
  """No stage asked for means the state the run ended in, which is what the
  checkpoint sitting in the run directory was saved under."""
  session = _session(tmp_path, [
      _stage("0_start", 0, parameters={"reward.a.weight": 1.0}),
      _stage("1_harder", 400, parameters={"reward.a.weight": 2.0}),
      _stage("2_hardest", 900, parameters={"reward.a.weight": 3.0}),
  ])

  conditions = read_conditions(session)

  assert conditions.stage == "2_hardest"
  assert conditions.parameters == {"reward.a.weight": 3.0}
  assert conditions.iteration == 900
  assert conditions.stage_names == ("0_start", "1_harder", "2_hardest")


def test_asking_for_a_stage_stops_at_the_end_of_it_not_the_start(tmp_path):
  """A checkpoint saved *during* a stage trained under everything that stage
  did, including the edits made while it was current. Stopping at its entry
  would replay a rung the policy had already climbed past."""
  session = _session(tmp_path, [
      _stage("0_start", 0, parameters={"reward.a.weight": 1.0}),
      _stage("1_harder", 400, parameters={"reward.a.weight": 2.0}),
      _edit("reward.b.weight", -0.5, 550),
      _stage("2_hardest", 900, parameters={"reward.a.weight": 3.0}),
  ])

  conditions = read_conditions(session, "1_harder")

  assert conditions.stage == "1_harder"
  assert conditions.parameters == {"reward.a.weight": 2.0, "reward.b.weight": -0.5}
  assert conditions.iteration == 550


def test_a_stage_the_run_never_entered_is_refused_by_name(tmp_path):
  """Naming the stages it did enter is the whole answer -- a bare KeyError
  sends somebody to go and read the log by hand."""
  session = _session(tmp_path, [_stage("0_start", 0)])

  with pytest.raises(KeyError) as caught:
    read_conditions(session, "9_imaginary")

  assert "0_start" in str(caught.value)


def test_order_is_preserved_across_parameters_and_commands(tmp_path):
  """A stage that sets a weight and then runs a command reading it is not the
  same run as one that does it the other way round."""
  session = _session(tmp_path, [
      _stage("0_start", 0,
             parameters={"reward.a.weight": 1.0},
             calls=[{"cmd": "set_difficulty", "args": {"level": 1}}]),
      _edit("reward.a.weight", 5.0, 100),
  ])

  kinds = [(s.kind, s.key) for s in read_conditions(session).steps]

  assert kinds == [
      ("parameter", "reward.a.weight"),
      ("command", "set_difficulty"),
      ("parameter", "reward.a.weight"),
  ]


def test_the_last_write_of_a_parameter_wins(tmp_path):
  session = _session(tmp_path, [
      _stage("0_start", 0, parameters={"reward.a.weight": 1.0}),
      _edit("reward.a.weight", 2.0, 100),
      _edit("reward.a.weight", 4.0, 200),
  ])

  assert read_conditions(session).parameters == {"reward.a.weight": 4.0}


def test_a_refused_live_edit_is_not_replayed(tmp_path):
  """The trainer recorded that the write did not take. Replaying it would put
  the environment somewhere the policy never was."""
  session = _session(tmp_path, [_edit("reward.a.weight", 2.0, 100, applied=False)])

  assert read_conditions(session).parameters == {}


def test_live_edits_can_be_left_out_on_purpose(tmp_path):
  """The ladder alone, for asking what the curriculum by itself would have
  produced."""
  session = _session(tmp_path, [
      _stage("0_start", 0, parameters={"reward.a.weight": 1.0}),
      _edit("reward.a.weight", 9.0, 100),
  ])

  conditions = read_conditions(session, include_parameter_edits=False)

  assert conditions.parameters == {"reward.a.weight": 1.0}


def test_a_run_that_could_not_apply_its_own_stage_says_so(tmp_path):
  """The log records what the stage failed to do during training. That is a
  warning about the reconstruction, not a reason to refuse one."""
  session = _session(tmp_path, [
      _stage("0_start", 0,
             parameter_errors={"reward.gone.weight": "not registered"},
             action_errors={"set_difficulty": "Unknown command"}),
  ])

  warnings = read_conditions(session).warnings

  assert any("reward.gone.weight" in w for w in warnings)
  assert any("set_difficulty" in w for w in warnings)


def test_a_session_with_no_event_log_reconstructs_nothing_and_explains_why(tmp_path):
  """The dangerous case: an empty reconstruction is rung zero, and a late
  policy shown at rung zero looks broken."""
  empty = tmp_path / "empty"
  empty.mkdir()

  conditions = read_conditions(empty)

  assert conditions.steps == ()
  assert any("rung zero" in w for w in conditions.warnings)


def test_a_torn_last_line_does_not_cost_the_run_in_front_of_it(tmp_path):
  """A run killed mid-write leaves half a JSON object. The 900 iterations
  already written are still the best account of what happened."""
  session = _session(tmp_path, [
      _stage("0_start", 0, parameters={"reward.a.weight": 1.0}),
      _stage("1_harder", 900, parameters={"reward.a.weight": 2.0}),
  ], torn='{"kind": "set_parameter", "iter')

  assert read_events(session) != []
  assert read_conditions(session).parameters == {"reward.a.weight": 2.0}


def test_stage_names_are_listed_once_in_the_order_entered(tmp_path):
  """A run sent back to an earlier rung entered it twice; the ladder is still
  the ordered set of rungs it touched."""
  session = _session(tmp_path, [
      _stage("0_start", 0), _stage("1_harder", 400), _stage("0_start", 700),
  ])

  assert stage_names(session) == ["0_start", "1_harder"]


# Where a stage's arguments come from.


def test_structured_calls_are_used_verbatim_when_the_log_has_them(tmp_path):
  session = _session(tmp_path, [
      _stage("0_start", 0, actions=["set_difficulty(level=1)"],
             calls=[{"cmd": "set_difficulty", "args": {"level": 7}}]),
  ])

  assert read_conditions(session).calls == [("set_difficulty", {"level": 7})]


def test_the_ladder_supplies_arguments_a_repr_could_not_carry(tmp_path):
  """An argument whose repr is not a literal -- an object, a numpy value --
  parses out of the prose as nothing. The planned ladder is real JSON and
  survives it, and matching counts is what says the two describe the same
  actions."""
  session = _session(
      tmp_path,
      [_stage("0_start", 0, actions=["set_mode(profile=<Profile object>)"])],
      ladder=[{"name": "0_start",
               "apply": [{"cmd": "set_mode", "args": {"profile": "fast"}}]}],
  )

  assert read_conditions(session).calls == [("set_mode", {"profile": "fast"})]


def test_a_run_that_diverged_from_its_ladder_trusts_the_log(tmp_path):
  """Different counts mean the plan is not what ran. The log is the only
  honest account, even though parsing it is the weaker route."""
  session = _session(
      tmp_path,
      [_stage("0_start", 0, actions=["set_difficulty(level=2)"])],
      ladder=[{"name": "0_start", "apply": [
          {"cmd": "set_difficulty", "args": {"level": 9}},
          {"cmd": "set_mode", "args": {"profile": "slow"}},
      ]}],
  )

  assert read_conditions(session).calls == [("set_difficulty", {"level": 2})]


def test_prose_that_cannot_be_parsed_becomes_a_warning_not_a_failure(tmp_path):
  """One unreadable action should cost that action, not the reconstruction."""
  session = _session(tmp_path, [
      _stage("0_start", 0, actions=["set_mode(profile=<object>)",
                                    "set_difficulty(level=3)"]),
  ])

  conditions = read_conditions(session)

  assert conditions.calls == [("set_difficulty", {"level": 3})]
  assert any("set_mode" in w for w in conditions.warnings)


def test_a_ladder_written_beside_the_run_is_found_too(tmp_path):
  """``<run>/params/curriculum.json`` is where a launcher saves the plan."""
  run = tmp_path / "run"
  (run / "params").mkdir(parents=True)
  (run / "params" / "curriculum.json").write_text(
      json.dumps({"stages": [{"name": "0_start", "apply": []}]})
  )
  session = _session(run, [], name="rlmcp")

  assert set(read_ladder(session)) == {"0_start"}


def test_an_unreadable_ladder_is_simply_absent(tmp_path):
  session = tmp_path / "session"
  session.mkdir()
  (session / "curriculum.json").write_text("{not json")

  assert read_ladder(session) == {}


# Parsing.


@pytest.mark.parametrize("text,expected", [
    ("set_difficulty(level=2)", ("set_difficulty", {"level": 2})),
    ("reset_all", ("reset_all", {})),
    ("set_mode(profile='fast', retries=3)",
     ("set_mode", {"profile": "fast", "retries": 3})),
    ("set_range(bounds=[1.1, 1.7])", ("set_range", {"bounds": [1.1, 1.7]})),
    ("set_flag(enabled=True)", ("set_flag", {"enabled": True})),
])
def test_a_logged_action_parses_back_into_the_call_that_ran(text, expected):
  assert parse_action(text) == expected


@pytest.mark.parametrize("text", [
    "",
    "not a command at all",
    "set_difficulty(2)",              # positional: we would guess the name
    "set_difficulty(**kwargs)",
    "set_mode(profile=<Profile object>)",
    "set_mode(profile=compute())",    # a call is not a literal
])
def test_anything_that_is_not_a_call_of_literals_is_refused(text):
  with pytest.raises(ValueError):
    parse_action(text)


def test_overrides_are_typed_by_json_and_fall_back_to_text():
  parsed = parse_overrides([
      "reward.a.weight=-0.2", "enabled=false", "bounds=[1.1,1.7]", "label=fast",
  ])

  assert parsed == {"reward.a.weight": -0.2, "enabled": False,
                    "bounds": [1.1, 1.7], "label": "fast"}


def test_an_override_without_a_value_is_refused():
  with pytest.raises(ValueError):
    parse_overrides(["reward.a.weight"])


def test_overrides_are_appended_so_they_win_over_the_replay():
  base = Conditions(steps=(Step("parameter", "reward.a.weight", 1.0),), stage="1")

  merged = with_overrides(base, {"reward.a.weight": 9.0})

  assert merged.parameters == {"reward.a.weight": 9.0}
  assert merged.stage == "1"
  assert base.parameters == {"reward.a.weight": 1.0}  # the original is untouched


def test_no_overrides_leaves_the_conditions_exactly_as_they_were():
  base = Conditions(steps=(Step("parameter", "reward.a.weight", 1.0),))

  assert with_overrides(base, {}) is base


# Replaying onto a controller.


class _FakeLab:
  """The two surfaces a replay touches: the parameter registry and commands."""

  def __init__(self, refuse=(), raise_on=None, handlers=None):
    self.values = {}
    self.called = []
    self._refuse = set(refuse)
    self._raise_on = raise_on or {}
    self._handlers = handlers if handlers is not None else {
        "set_difficulty": lambda level=0: None
    }
    self.parameters = self

  def set_value(self, key, value):
    if key in self._raise_on:
      raise self._raise_on[key]
    if key in self._refuse:
      return False
    self.values[key] = value
    return True

  def run_command(self, cmd, **args):
    if cmd not in self._handlers:
      raise KeyError(f"Unknown command '{cmd}'. This run has: set_difficulty")
    self.called.append((cmd, args))
    return self._handlers[cmd](**args)


def test_a_clean_replay_reports_what_it_put_back():
  lab = _FakeLab()
  conditions = Conditions(steps=(
      Step("parameter", "reward.a.weight", 2.0),
      Step("command", "set_difficulty", {"level": 3}),
  ))

  restored = apply_conditions(lab, conditions)

  assert lab.values == {"reward.a.weight": 2.0}
  assert lab.called == [("set_difficulty", {"level": 3})]
  assert restored["errors"] == []
  assert restored["parameters"] == {"reward.a.weight": 2.0}


def test_a_command_this_build_does_not_have_is_named_as_missing():
  """The fix is --task-package, and the error kind is what tells the caller
  so. Guessing at the wrong fix sends them to check something unbroken."""
  conditions = Conditions(steps=(Step("command", "set_mode", {"profile": "fast"}),))

  restored = apply_conditions(_FakeLab(), conditions)

  assert restored["error_kinds"] == ["missing_command"]


def test_a_command_whose_arguments_moved_on_reports_what_it_takes_now():
  """"Unknown command" and "the command changed" are different problems, and
  showing today's signature is what makes the difference visible."""
  lab = _FakeLab(handlers={"set_difficulty": lambda tier=0: None})
  conditions = Conditions(steps=(Step("command", "set_difficulty", {"level": 3}),))

  restored = apply_conditions(lab, conditions)

  assert restored["error_kinds"] == ["changed_command"]
  assert "tier" in restored["errors"][0]


def test_a_refused_parameter_is_an_error_not_a_silent_pass():
  lab = _FakeLab(refuse={"reward.gone.weight"})
  conditions = Conditions(steps=(Step("parameter", "reward.gone.weight", 1.0),))

  restored = apply_conditions(lab, conditions)

  assert restored["error_kinds"] == ["parameter"]
  assert restored["parameters"] == {}


def test_a_registry_that_raises_is_the_same_answer_as_one_that_refuses():
  """An unregistered key raises, an out-of-range value raises, a dead-on-write
  parameter raises. All three mean the condition was not restored."""
  lab = _FakeLab(raise_on={"reward.gone.weight": KeyError("not registered")})
  conditions = Conditions(steps=(Step("parameter", "reward.gone.weight", 1.0),))

  restored = apply_conditions(lab, conditions)

  assert restored["error_kinds"] == ["parameter"]
  assert "not registered" in restored["errors"][0]


def test_one_failure_does_not_stop_the_rest_of_the_replay():
  """A partial reconstruction is still worth reporting; throwing it away turns
  one missing knob into no evidence at all."""
  lab = _FakeLab(refuse={"reward.gone.weight"})
  conditions = Conditions(steps=(
      Step("parameter", "reward.gone.weight", 1.0),
      Step("parameter", "reward.a.weight", 2.0),
      Step("command", "set_difficulty", {"level": 1}),
  ))

  restored = apply_conditions(lab, conditions)

  assert lab.values == {"reward.a.weight": 2.0}
  assert lab.called == [("set_difficulty", {"level": 1})]
  assert len(restored["errors"]) == 1


def test_a_commands_own_exception_is_reported_as_itself():
  def explode(level=0):
    raise RuntimeError("the object set is empty")

  lab = _FakeLab(handlers={"set_difficulty": explode})
  conditions = Conditions(steps=(Step("command", "set_difficulty", {"level": 1}),))

  restored = apply_conditions(lab, conditions)

  assert restored["error_kinds"] == ["failed_command"]
  assert "the object set is empty" in restored["errors"][0]


def test_the_summary_is_json_shaped_for_a_result_payload():
  conditions = Conditions(
      steps=(Step("parameter", "reward.a.weight", 2.0),
             Step("command", "set_difficulty", {"level": 3})),
      stage="1_harder", stage_names=("0_start", "1_harder"), iteration=700,
  )

  summary = conditions.summary()

  assert json.loads(json.dumps(summary)) == summary
  assert summary["stage"] == "1_harder"
  assert summary["through_iteration"] == 700
  assert summary["calls"] == [{"cmd": "set_difficulty", "args": {"level": 3}}]


def test_read_events_takes_an_open_session_as_well_as_a_directory(tmp_path):
  """The reader a run's history goes through must not require a filesystem.

  `play` hands it a directory; anything reading a run it cannot open files in
  hands it the session, which answers the same question over whatever
  transport it has.
  """
  from rlmcp.session import Session

  session = Session(tmp_path / "sess").create({})
  session.append_event("curriculum_stage", {"to": "1_harder", "iteration": 10})

  from_dir = read_events(session.dir)
  from_session = read_events(session)

  assert [e["to"] for e in from_dir] == ["1_harder"]
  assert from_session == from_dir
  assert stage_names(session) == ["1_harder"]
