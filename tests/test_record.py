"""The record itself: serialisation, falsifiers, and the computed recipe."""

from __future__ import annotations

import json

import pytest

from rlmcp.cli import main as cli_main
from rlmcp.core.curriculum import Condition
from rlmcp.records.filestore import FileStore
from rlmcp.records.record import (
    Falsifier,
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


def test_summary_carries_the_lineage_shape():
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
