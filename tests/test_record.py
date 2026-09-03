"""The record itself: serialisation, falsifiers, and the computed recipe."""

from __future__ import annotations

import json

import pytest

from rlmcp.cli import main as cli_main
from rlmcp.core.curriculum import Condition
from rlmcp.records.filestore import FileStore
from rlmcp.records.record import (
    Falsifier,
    Feedback,
    Lease,
    RunRecord,
    Weights,
    ancestors,
    children,
    fold_recipe,
    slugify,
)
from rlmcp.session import Session


def _record(rid: str, parent=None, change=None, **kwargs) -> RunRecord:
  return RunRecord(id=rid, slug=f"run_{rid}", change=change or [f"change {rid}"],
                   parent=parent, **kwargs)


def test_slugify_makes_a_safe_name():
  assert slugify("Faster commands, v2!") == "faster_commands_v2"
  assert slugify("   ") == "run"


def test_record_round_trips_through_a_dict():
  record = RunRecord(
      id="007",
      slug="wide_scan",
      seq=7,
      stage="rough",
      verdict="provisional",
      hypothesis="the wide scan lifts the plateau",
      prediction="terrain level above 6.1",
      falsifier=Falsifier(
          prose="plateau stays at or below 6.1",
          conditions=[Condition("terrain/level", "<=", 6.1)],
      ),
      change=["scan width 1.6m"],
      outcome="plateau moved to 6.03",
      metrics=[["plateau", "6.03"], ["falsifier fired", "no"]],
      parent="006",
      weights=Weights("006", "model_10000.pt"),
      prior="P3",
      proposed_by="orchestrator",
      session="/logs/run7/rlmcp",
      config={"reward.slip.weight": -0.2},
      links={"log": "logs/007.log"},
      assets={"videos": [["media/007/tour.mp4", "3-seed tour"]]},
      lease=Lease(slot="gpu0", holder="pid-1"),
      tags=["locomotion"],
  )

  restored = RunRecord.from_dict(record.to_dict())

  assert restored.to_dict() == record.to_dict()
  assert restored.weights.describe() == "006 @ model_10000.pt"
  assert restored.falsifier.conditions[0].describe() == "terrain/level <= 6.1"
  assert restored.lease.slot == "gpu0"
  assert restored.proposed_by == "orchestrator"


def test_a_bare_string_is_accepted_as_a_falsifier():
  """What a human writes first is a sentence, not a condition list."""
  record = RunRecord.from_dict(
      {"id": "1", "slug": "x", "falsifier": "the robot never leaves the spawn"}
  )
  assert record.falsifier.prose == "the robot never leaves the spawn"
  assert record.falsifier.conditions == []


def test_falsifier_fires_when_its_condition_is_met():
  falsifier = Falsifier(
      prose="episodes stay short",
      conditions=[Condition("rlmcp/episode_length_frac", "<=", 0.2)],
  )

  alive = falsifier.check({"rlmcp/episode_length_frac": 0.8})
  dead = falsifier.check({"rlmcp/episode_length_frac": 0.1})

  assert alive["fired"] is False
  assert dead["fired"] is True
  assert dead["checks"][0]["current"] == 0.1


def test_a_missing_metric_is_undecidable_not_survival():
  """"The falsifier did not fire" and "we could not tell" are different claims."""
  falsifier = Falsifier(conditions=[Condition("never/logged", "<=", 1.0)])

  result = falsifier.check({"something/else": 1.0})

  assert result["fired"] is False
  assert result["undecidable"] is True
  assert result["checks"][0]["measurable"] is False


def test_an_empty_falsifier_knows_it_is_empty():
  assert Falsifier().is_empty()
  assert not Falsifier(prose="something").is_empty()
  assert not Falsifier(conditions=[Condition("a", ">=", 1)]).is_empty()


def test_a_lease_expires_and_ages():
  lease = Lease(slot="gpu0", ttl_seconds=100.0)
  assert not lease.expired()

  lease.renewed_at -= 200
  assert lease.expired()
  assert lease.age_seconds() > 0


def test_from_scratch_is_the_absence_of_weights():
  assert _record("1").from_scratch
  assert not _record("2", weights=Weights("1")).from_scratch


def test_recipe_is_the_fold_of_changes_from_the_root():
  records = {
      r.id: r
      for r in [
          _record("001", change=["baseline"]),
          _record("002", parent="001", change=["add slip penalty"]),
          _record("003", parent="002", change=["raise action rate cost"]),
      ]
  }

  recipe = fold_recipe("003", records)

  assert recipe == [
      ("001", ["baseline"]),
      ("002", ["add slip penalty"]),
      ("003", ["raise action rate cost"]),
  ]
  assert ancestors("003", records) == ["001", "002", "003"]


def test_a_root_folds_to_itself():
  records = {"001": _record("001")}
  assert fold_recipe("001", records) == [("001", ["change 001"])]


def test_a_parent_cycle_raises_rather_than_hanging():
  """The method this ports has no cycle check; its viewer would loop forever."""
  a = _record("001", parent="002")
  b = _record("002", parent="001")
  records = {"001": a, "002": b}

  with pytest.raises(ValueError, match="Cycle in the parent chain"):
    fold_recipe("001", records)


def test_children_finds_config_descendants_only():
  records = [
      _record("001"),
      _record("002", parent="001"),
      _record("003", parent="001"),
      _record("004", parent="002", weights=Weights("001")),  # warm from 001, child of 002
  ]

  assert [r.id for r in children("001", records)] == ["002", "003"]
  assert [r.id for r in children("002", records)] == ["004"]


def test_summary_carries_the_ancestry_shape():
  record = _record("002", parent="001", weights=Weights("001", "model_5.pt"))
  summary = record.summary()

  assert summary["parent"] == "001"
  assert summary["weights"] == "001"
  assert summary["from_scratch"] is False


# Close-out. The falsifier is read at the run's final iteration, never before
# its pre-registered read-point, and a close is written exactly once.


def _store(tmp_path) -> FileStore:
  return FileStore(tmp_path / "records", slots=1)


def _session(tmp_path, rows) -> str:
  """A finished session whose metrics.jsonl holds ``(iteration, metrics)`` rows."""
  session = Session(tmp_path / "logs" / "run" / "rlmcp").create({"task": "test"})
  for iteration, metrics in rows:
    session.append_metrics(iteration, metrics)
  return str(session.dir)


def _close(root, *argv) -> int:
  return cli_main(["record", "--records-root", str(root), "close", *argv])


def _falsifier_rows(record: RunRecord) -> list:
  return [value for name, value in record.metrics if name == "falsifier"]


def test_a_close_before_the_read_point_records_not_evaluable_never_fired(tmp_path, capsys):
  """A run killed at iteration 50 has not fired a falsifier that says
  "do not read me before 600" -- even when the condition would be met."""
  store = _store(tmp_path)
  record = store.new_record(
      "killed_early",
      falsifier=Falsifier(
          prose="episodes stay short",
          conditions=[Condition("rlmcp/episode_length_frac", "<=", 0.2)],
          check_after=600,
      ),
  )
  record.session = _session(
      tmp_path,
      [(i, {"rlmcp/episode_length_frac": 0.05}) for i in range(0, 51, 10)],
  )
  store.put_record(record)

  rc = _close(store.root, record.id, "interrupted")

  assert rc == 0
  closed = store.get_record(record.id)
  assert _falsifier_rows(closed) == [
      "not evaluable yet (closed at iteration 50 < check_after 600)"
  ]
  assert not any("FIRED" in value for _, value in closed.metrics)
  payload = json.loads(capsys.readouterr().out)
  assert payload["falsifier"]["too_early"] is True
  assert payload["falsifier"]["evaluated_at"] == 50
  assert "Not evaluable yet" in (store.read_document(record.id, "REPORT.md") or "")


def test_a_close_past_the_read_point_evaluates_and_records_the_iteration(tmp_path, capsys):
  store = _store(tmp_path)
  record = store.new_record(
      "ran_long",
      falsifier=Falsifier(
          conditions=[Condition("rlmcp/episode_length_frac", "<=", 0.2)],
          check_after=600,
      ),
  )
  record.session = _session(
      tmp_path,
      [(i, {"rlmcp/episode_length_frac": 0.05}) for i in range(600, 1201, 100)],
  )
  store.put_record(record)

  rc = _close(store.root, record.id, "falsified", "--outcome", "episodes collapsed",
              "--metric", "rlmcp/episode_length_frac=0.05")

  assert rc == 0
  closed = store.get_record(record.id)
  assert closed.verdict == "falsified"
  assert _falsifier_rows(closed) == ["FIRED (at iteration 1200)"]
  assert json.loads(capsys.readouterr().out)["falsifier"]["fired"] is True


def test_an_intermittently_logged_metric_still_evaluates_at_close(tmp_path, capsys):
  """A metric logged every N iterations is absent from the final row; the
  close-out merges the newest value per metric over the tail window instead
  of calling it undecidable."""
  store = _store(tmp_path)
  record = store.new_record(
      "sparse_metric",
      falsifier=Falsifier(
          conditions=[Condition("eval/success_rate", "<=", 0.2)],
          check_after=100,
      ),
  )
  rows = [(i, {"Train/mean_reward": 1.0}) for i in range(1150, 1201)]
  rows[10] = (1160, {"Train/mean_reward": 1.0, "eval/success_rate": 0.9})
  record.session = _session(tmp_path, rows)
  store.put_record(record)

  rc = _close(store.root, record.id, "validated",
              "--outcome", "held up", "--metric", "success=0.9")

  assert rc == 0
  assert _falsifier_rows(store.get_record(record.id)) == ["held (at iteration 1200)"]
  payload = json.loads(capsys.readouterr().out)
  assert payload["falsifier"]["undecidable"] is False
  assert payload["falsifier"]["metrics_window"] == 50  # the window is on record


def test_closing_twice_is_refused_and_duplicates_nothing(tmp_path, capsys):
  store = _store(tmp_path)
  record = store.new_record(
      "once_only",
      falsifier=Falsifier(conditions=[Condition("m/x", "<=", 0.2)]),
  )
  record.session = _session(tmp_path, [(100, {"m/x": 0.9})])
  store.put_record(record)
  assert _close(store.root, record.id, "provisional", "--outcome", "fine",
                "--metric", "plateau=6.0") == 0
  first = store.get_record(record.id).metrics
  capsys.readouterr()

  rc = _close(store.root, record.id, "provisional", "--outcome", "fine",
              "--metric", "plateau=6.0")

  assert rc == 1
  error = json.loads(capsys.readouterr().out)["error"]
  assert "already closed" in error
  assert "provisional" in error  # says what the verdict already is
  assert "meta.json" in error  # reopening means editing the file
  assert store.get_record(record.id).metrics == first  # no duplicate rows


def test_a_result_verdict_without_evidence_is_refused_at_close(tmp_path, capsys):
  store = _store(tmp_path)
  record = store.new_record("bare", hypothesis="h")

  rc = _close(store.root, record.id, "validated")

  assert rc == 1
  assert "not knowledge" in json.loads(capsys.readouterr().out)["error"]
  assert store.get_record(record.id).verdict == "planned"  # nothing persisted


def test_a_held_falsifier_alone_cannot_carry_a_validated_close(tmp_path, capsys):
  """The machine-appended row records the falsifier's state, and held is
  absence of disproof -- it must not row-count as the claim's measurement."""
  store = _store(tmp_path)
  record = store.new_record(
      "no_numbers",
      falsifier=Falsifier(conditions=[Condition("m/x", "<=", 0.2)],
                          check_after=100),
  )
  record.session = _session(tmp_path, [(1200, {"m/x": 0.9})])  # held
  store.put_record(record)

  rc = _close(store.root, record.id, "validated", "--outcome", "looks great")

  assert rc == 1
  assert "not knowledge" in json.loads(capsys.readouterr().out)["error"]
  after = store.get_record(record.id)
  assert after.verdict == "planned"
  assert after.metrics == []  # the refused close persisted nothing


def test_a_not_evaluable_falsifier_row_cannot_carry_a_validated_close(tmp_path, capsys):
  """Worse than held: the row explicitly says nothing was measured."""
  store = _store(tmp_path)
  record = store.new_record(
      "killed_then_claimed",
      falsifier=Falsifier(conditions=[Condition("m/x", "<=", 0.2)],
                          check_after=600),
  )
  record.session = _session(tmp_path, [(50, {"m/x": 0.05})])  # premature
  store.put_record(record)

  rc = _close(store.root, record.id, "validated", "--outcome", "looks great")

  assert rc == 1
  assert "not knowledge" in json.loads(capsys.readouterr().out)["error"]
  after = store.get_record(record.id)
  assert after.verdict == "planned"
  assert after.metrics == []


# Feedback: what a human said, and what was done about it. The point of the
# field is that an instruction with no recorded response stays visible, so
# these check the honest-by-default shapes as much as the round trip.


def test_feedback_round_trips_through_a_dict():
  record = RunRecord(
      id="011",
      slug="steered",
      feedback=[
          Feedback(text="It gets jittery near the end.", kind="observe",
                   author="reviewer", iteration=420),
          Feedback(text="Stop tuning the entropy coefficient.", kind="correct",
                   response="Checked; it was already at the default.",
                   changed=False, affects=["012"], artifacts=["plots/jitter.png"]),
      ],
  )

  restored = RunRecord.from_dict(json.loads(json.dumps(record.to_dict())))

  assert [f.text for f in restored.feedback] == [f.text for f in record.feedback]
  assert restored.feedback[0].kind == "observe"
  assert restored.feedback[0].iteration == 420
  assert restored.feedback[1].affects == ["012"]
  assert restored.feedback[1].artifacts == ["plots/jitter.png"]
  assert restored.feedback[1].changed is False


def test_a_record_written_before_feedback_existed_still_loads():
  """Every new field is optional; a v1 record keeps its own schema stamp."""
  old = {"schema_version": 1, "id": "003", "slug": "ancient",
         "hypothesis": "it will work", "verdict": "planned"}

  record = RunRecord.from_dict(old)

  assert record.feedback == []
  assert record.headline == ""
  assert record.task == ""
  assert record.schema_version == 1
  assert record.outstanding_feedback() == []


def test_bare_string_feedback_is_kept_rather_than_dropped():
  """A hand-edited record is still a record; losing the remark is worse."""
  record = RunRecord.from_dict(
      {"id": "004", "slug": "hand_edited", "feedback": ["just do the thing"]}
  )

  assert record.feedback[0].text == "just do the thing"
  assert record.feedback[0].kind == "steer"


def test_only_instructions_are_outstanding_without_a_response():
  """An observation asked for nothing, so silence is not a dropped ball."""
  record = RunRecord(
      id="005", slug="mixed",
      feedback=[
          Feedback(text="looks good", kind="approve"),
          Feedback(text="it looks jittery", kind="observe"),
          Feedback(text="try a smaller step", kind="steer"),
          Feedback(text="never exceed 80%", kind="constrain", response="Capped it."),
      ],
  )

  assert [i for i, _ in record.outstanding_feedback()] == [2]
  assert record.feedback_kinds() == {
      "approve": 1, "observe": 1, "steer": 1, "constrain": 1}


def test_a_written_headline_wins_over_the_derived_sentence():
  record = RunRecord(id="006", slug="summarised",
                     outcome="Reward rose to 4.2. Then it plateaued.")
  assert record.one_line() == "Reward rose to 4.2."

  record.headline = "The plateau, not the rise, is the story."
  assert record.one_line() == "The plateau, not the rise, is the story."


def test_one_line_still_falls_back_when_no_headline_is_written():
  """The viewer payload reads one_line(); clearing the headline restores it."""
  record = RunRecord(id="007", slug="open_run", hypothesis="A longer warmup helps.")
  assert record.one_line() == "A longer warmup helps."

  record.outcome = "It did not. The warmup was never the constraint."
  assert record.one_line() == "It did not."

  record.headline = "   "  # whitespace is not a headline
  assert record.one_line() == "It did not."


def test_the_summary_row_carries_the_feedback_counts():
  record = RunRecord(
      id="008", slug="counted", task="Some-Task-v0",
      feedback=[Feedback(text="do this", kind="steer"),
                Feedback(text="nice", kind="approve")],
  )

  row = record.summary()

  assert row["feedback"] == 2
  assert row["feedback_outstanding"] == 1
  assert row["task"] == "Some-Task-v0"
  # Existing keys keep their meaning: the JSON is a parsing contract.
  assert row["hypothesis"] == "" and row["outcome"] == ""


# The command line over the same data. `rlmcp record` JSON is a parsing
# contract for agents, so these check the payload shapes as well as the effect.


def _record_cli(root, *argv) -> int:
  return cli_main(["record", "--records-root", str(root), *argv])


def _payload(capsys) -> dict:
  return json.loads(capsys.readouterr().out)


def _feedback_session(tmp_path, said) -> str:
  """A finished session whose event log holds feedback rows."""
  session = Session(tmp_path / "logs" / "spoken" / "rlmcp").create({"task": "test"})
  session.append_metrics(400, {"Train/mean_reward": 1.0})
  for entry in said:
    session.append_event("feedback", entry)
  return str(session.dir)


def test_the_cli_attaches_feedback_and_says_it_is_unanswered(tmp_path, capsys):
  store = _store(tmp_path)
  record = store.new_record("steered")

  rc = _record_cli(store.root, "feedback", record.id,
                   "Stop tuning the entropy coefficient.", "--kind", "correct")

  assert rc == 0
  payload = _payload(capsys)
  assert payload["index"] == 0
  assert payload["feedback"]["kind"] == "correct"
  # The reminder names the exact command that would answer it.
  assert f"record answer {record.id} 0" in payload["reminder"]


def test_an_approval_gets_no_nagging_reminder(tmp_path, capsys):
  store = _store(tmp_path)
  record = store.new_record("approved")

  _record_cli(store.root, "feedback", record.id, "Looks right.", "--kind", "approve")

  assert "reminder" not in _payload(capsys)


def test_feedback_is_stamped_with_the_runs_current_iteration(tmp_path, capsys):
  """The iteration is the one thing only the session knows, and it is what
  places the remark on the run's timeline."""
  store = _store(tmp_path)
  record = store.new_record("live")
  session = Session(tmp_path / "logs" / "live" / "rlmcp").create({"task": "test"})
  session.publish_status({"iteration": 880, "num_envs": 4})
  record.session = str(session.dir)
  store.put_record(record)

  _record_cli(store.root, "feedback", record.id, "Something is off.")

  assert _payload(capsys)["feedback"]["iteration"] == 880


def test_feedback_on_a_record_with_no_session_has_no_iteration(tmp_path, capsys):
  """Missing is fine -- a remark on a finished write-up has no iteration --
  but guessing one is not."""
  store = _store(tmp_path)
  record = store.new_record("off_run")

  _record_cli(store.root, "feedback", record.id, "About the write-up.")

  assert _payload(capsys)["feedback"]["iteration"] is None


def test_the_cli_answers_a_remark_without_touching_its_text(tmp_path, capsys):
  store = _store(tmp_path)
  record = store.new_record("answered")
  _record_cli(store.root, "feedback", record.id, "Try a smaller step.")
  capsys.readouterr()

  rc = _record_cli(store.root, "answer", record.id, "0",
                   "Halved it; the jitter went away.")

  assert rc == 0
  entry = _payload(capsys)["feedback"]
  assert entry["text"] == "Try a smaller step."
  assert entry["changed"] is True
  assert store.get_record(record.id).outstanding_feedback() == []


def test_answering_an_index_that_is_not_there_fails_loudly(tmp_path, capsys):
  store = _store(tmp_path)
  record = store.new_record("short")

  rc = _record_cli(store.root, "answer", record.id, "3", "about the fourth")

  assert rc == 1
  assert "no index 3" in _payload(capsys)["error"]


def test_a_written_headline_replaces_the_derived_one(tmp_path, capsys):
  store = _store(tmp_path)
  record = store.new_record("summarised", hypothesis="A longer warmup helps.")

  _record_cli(store.root, "headline", record.id, "The entropy floor is the constraint.")
  payload = _payload(capsys)

  assert payload["headline"] == "The entropy floor is the constraint."
  assert payload["derived"] is False

  _record_cli(store.root, "headline", record.id)  # no text clears it
  cleared = _payload(capsys)
  assert cleared["headline"] == "A longer warmup helps."
  assert cleared["derived"] is True


def test_the_timeline_renders_a_ledger_with_the_unanswered_count(tmp_path, capsys):
  store = _store(tmp_path)
  first = store.new_record("early")
  second = store.new_record("late")
  _record_cli(store.root, "feedback", first.id, "Never exceed 80%.",
              "--kind", "constrain")
  _record_cli(store.root, "feedback", second.id, "Try a smaller step.",
              "--kind", "steer", "--did", "Halved it.")
  _record_cli(store.root, "feedback", second.id, "Stop tuning it.",
              "--kind", "correct")
  capsys.readouterr()

  rc = _record_cli(store.root, "timeline", "--markdown")

  assert rc == 0
  text = capsys.readouterr().out
  assert "# Feedback ledger" in text
  assert "**2 unanswered**" in text
  assert f"{first.id}[0]" in text
  assert "Never exceed 80%." in text          # the table
  assert "**Done:** Halved it." in text       # and again, in full


def test_the_ledger_can_be_written_to_a_file(tmp_path, capsys):
  store = _store(tmp_path)
  record = store.new_record("written_out")
  _record_cli(store.root, "feedback", record.id, "Say it once.")
  capsys.readouterr()
  out = tmp_path / "FEEDBACK.md"

  _record_cli(store.root, "timeline", "--markdown", "--out", str(out))

  assert _payload(capsys)["count"] == 1
  assert "Say it once." in out.read_text()


def test_an_empty_store_renders_a_ledger_rather_than_failing(tmp_path, capsys):
  store = _store(tmp_path)

  rc = _record_cli(store.root, "timeline", "--markdown")

  assert rc == 0
  assert "No feedback recorded yet." in capsys.readouterr().out


def test_check_reports_the_unanswered_count_without_reshaping_its_payload(
    tmp_path, capsys):
  store = _store(tmp_path)
  record = store.new_record("closed_with_a_loose_end", outcome="it happened",
                            metrics=[["measured", "1.0"]], verdict="falsified")
  store.put_record(record)
  _record_cli(store.root, "feedback", record.id, "Never exceed 80%.",
              "--kind", "constrain")
  capsys.readouterr()

  _record_cli(store.root, "check")
  payload = _payload(capsys)

  # Everything an existing parser reads is still exactly where it was.
  assert {"ok", "errors", "warnings", "records"} <= set(payload)
  assert payload["feedback"] == {
      "total": 1, "unanswered": 1, "runs_with_unanswered": [record.id]}


def test_check_on_a_store_with_no_feedback_still_answers(tmp_path, capsys):
  store = _store(tmp_path)
  store.new_record("quiet")

  _record_cli(store.root, "check")

  assert _payload(capsys)["feedback"]["unanswered"] == 0


def test_closing_folds_what_was_said_to_the_running_trainer(tmp_path, capsys):
  """Feedback said mid-run lives in the session log, which is deleted with the
  logs. The close-out is the last chance to move it into the record."""
  store = _store(tmp_path)
  record = store.new_record("spoken_to")
  record.session = _feedback_session(tmp_path, [
      {"iteration": 120, "text": "It collapses after the warmup.",
       "feedback_kind": "correct", "author": "user",
       "interpretation": "Shorten the warmup."},
  ])
  store.put_record(record)

  _close(store.root, record.id, "falsified", "--outcome", "it did not hold",
         "--metric", "x=1")
  capsys.readouterr()

  folded = store.get_record(record.id).feedback
  assert [f.text for f in folded] == ["It collapses after the warmup."]
  assert folded[0].iteration == 120
  assert folded[0].interpretation == "Shorten the warmup."


def test_a_second_close_does_not_double_the_ledger(tmp_path, capsys):
  """`interrupted` is the one terminal verdict that can be closed again."""
  store = _store(tmp_path)
  record = store.new_record("reclosed")
  record.session = _feedback_session(tmp_path, [
      {"iteration": 40, "text": "said once", "feedback_kind": "steer"},
  ])
  store.put_record(record)
  _close(store.root, record.id, "interrupted")
  capsys.readouterr()

  _close(store.root, record.id, "falsified", "--outcome", "done",
         "--metric", "x=1")
  capsys.readouterr()

  assert [f.text for f in store.get_record(record.id).feedback] == ["said once"]


def test_a_headline_given_at_close_wins_over_the_outcome(tmp_path, capsys):
  store = _store(tmp_path)
  record = store.new_record("closed_with_a_headline")

  _close(store.root, record.id, "falsified",
         "--outcome", "Reward rose to 4.2. Then it plateaued.",
         "--headline", "The plateau is the story.", "--metric", "x=1")

  assert _payload(capsys)["record"]["headline"] == "The plateau is the story."
