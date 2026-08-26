"""The rules. Each one was paid for by a real failure; each one is now a check."""

from __future__ import annotations

from rlmcp.records.record import Falsifier, Lease, RunRecord, Weights
from rlmcp.records.validate import check_verdict_change, validate


def _record(rid: str, **kwargs) -> RunRecord:
  base = {"slug": f"run_{rid}", "change": ["a change"]}
  base.update(kwargs)
  return RunRecord(id=rid, **base)


def _closed(rid: str, verdict: str = "provisional", **kwargs) -> RunRecord:
  kwargs.setdefault("outcome", "it happened")
  kwargs.setdefault("metrics", [["measured", "1.0"]])
  return _record(rid, verdict=verdict, **kwargs)


def _rules(report) -> list:
  return [p.rule for p in report.errors]


def test_a_clean_records_passes():
  report = validate([_closed("001"), _closed("002", parent="001")])
  assert report.ok
  assert "All records pass" in report.describe()


# Identity and references.


def test_duplicate_ids_are_an_error():
  assert "unique-id" in _rules(validate([_record("001"), _record("001")]))


def test_duplicate_sequence_numbers_are_an_error():
  records = [_record("001", seq=1), _record("002", seq=1)]
  assert "unique-seq" in _rules(validate(records))


def test_a_parent_that_does_not_exist_is_an_error():
  report = validate([_record("002", parent="404")])
  assert "reference" in _rules(report)
  assert "404" in report.errors[0].message


def test_weights_pointing_nowhere_is_an_error():
  report = validate([_record("002", weights=Weights("404", "model.pt"))])
  assert "reference" in _rules(report)


def test_a_parent_cycle_is_caught():
  records = [_record("001", parent="002"), _record("002", parent="001")]
  assert "parent-cycle" in _rules(validate(records))


# The warm-start cap — the highest-value rule in the method.


def test_a_warm_started_run_cannot_be_validated():
  records = [
      _closed("001", verdict="validated"),
      _closed("002", verdict="validated", parent="001",
              weights=Weights("001", "model_100.pt")),
  ]

  report = validate(records)

  assert "warm-start-cap" in _rules(report)
  assert "PRESERVES" in report.errors[0].message


def test_a_warm_started_run_may_be_provisional():
  records = [
      _closed("001", verdict="validated"),
      _closed("002", verdict="provisional", parent="001", weights=Weights("001")),
  ]
  assert validate(records).ok


def test_a_from_scratch_run_may_be_validated():
  assert validate([_closed("001", verdict="validated")]).ok


def test_close_out_refuses_the_verdict_a_warm_start_cannot_support():
  record = _closed("002", verdict="planned", weights=Weights("001", "model_100.pt"))

  refusal = check_verdict_change(record, "validated", [record])

  assert refusal is not None
  assert "provisional" in refusal
  assert check_verdict_change(record, "provisional", [record]) is None


def test_close_out_rejects_an_unknown_verdict():
  assert "Unknown verdict" in check_verdict_change(_record("1"), "great", [])


def test_close_out_rejects_an_open_state_as_a_target():
  record = _record("001")
  assert "open state" in check_verdict_change(record, "running", [record])


def test_close_out_requires_evidence_for_a_result_verdict():
  """The same rule the whole-records check enforces, applied when the verdict
  is set rather than discovered later."""
  bare = _record("001")

  refusal = check_verdict_change(bare, "validated", [bare])

  assert refusal is not None
  assert "not knowledge" in refusal
  # With outcome and a measurement in place, the same close goes through.
  assert check_verdict_change(_closed("002", verdict="planned"),
                              "validated", []) is None
  # A verdict that claims no result owes no evidence.
  assert check_verdict_change(bare, "superseded", [bare]) is None


def test_a_falsifier_row_alone_is_not_evidence():
  """Held is absence of disproof. The machine-appended row cannot carry a
  result claim -- at close time or in the whole-records check."""
  record = _record("001", outcome="looks great",
                   metrics=[["falsifier", "held (at iteration 1200)"]])

  assert check_verdict_change(record, "validated", [record]) is not None

  record.verdict = "validated"
  assert "result-needs-evidence" in _rules(validate([record]))

  # One real measurement beside it, and the same claim goes through.
  record.verdict = "planned"
  record.metrics.append(["plateau", "6.03"])
  assert check_verdict_change(record, "validated", [record]) is None
  record.verdict = "validated"
  assert validate([record]).ok


def test_close_out_refuses_a_record_that_is_already_closed():
  done = _closed("001", verdict="falsified")

  refusal = check_verdict_change(done, "validated", [done])

  assert refusal is not None
  assert "already closed" in refusal
  assert "falsified" in refusal  # says what the verdict already is
  assert "meta.json" in refusal  # the files are the truth here


def test_interrupted_is_the_one_terminal_verdict_that_may_be_closed_again():
  """A job that died is not a result -- it still owes a real close-out."""
  reaped = _closed("001", verdict="interrupted")

  assert check_verdict_change(reaped, "falsified", [reaped]) is None
  assert "already closed" in check_verdict_change(reaped, "interrupted", [reaped])


# Evidence.


def test_a_result_without_measurements_is_an_error():
  """A claim without a measurement is not knowledge."""
  report = validate([_record("001", verdict="validated", outcome="it worked")])
  assert "result-needs-evidence" in _rules(report)


def test_a_result_without_an_outcome_is_an_error():
  records = [_record("001", verdict="falsified", metrics=[["x", "1"]])]
  assert "result-needs-evidence" in _rules(records and validate(records))


def test_an_open_run_needs_no_evidence_yet():
  assert validate([_record("001", verdict="planned", hypothesis="h",
                           falsifier=Falsifier(prose="f"))]).ok


# Warnings — discipline, not gates.


def test_a_planned_run_without_a_falsifier_warns():
  report = validate([_record("001", verdict="planned", hypothesis="h")])

  assert report.ok  # a warning, not a gate
  assert any(p.rule == "falsifier" for p in report.warnings)
  assert "not an experiment" in report.warnings[0].message


def test_a_running_run_without_a_hypothesis_warns():
  report = validate([_record("001", verdict="running",
                             falsifier=Falsifier(prose="f"))])
  assert any(p.rule == "hypothesis" for p in report.warnings)


def test_two_changes_in_one_finished_run_warn():
  report = validate([_closed("001", change=["one thing", "another thing"])])

  assert report.ok
  assert any(p.rule == "multi-change" for p in report.warnings)


def test_a_planned_run_may_still_list_several_changes():
  report = validate([_record("001", verdict="planned", hypothesis="h",
                             falsifier=Falsifier(prose="f"),
                             change=["one", "two"])])
  assert not any(p.rule == "multi-change" for p in report.warnings)


def test_a_recorded_asset_that_is_gone_warns():
  record = _closed("001", assets={"videos": [["media/001/tour.mp4", "tour"]]})

  report = validate([record], media_exists={"media/001/tour.mp4": False})

  assert any(p.rule == "missing-asset" for p in report.warnings)
  assert validate([record], media_exists={"media/001/tour.mp4": True}).ok


# Slots.


def test_more_live_leases_than_slots_is_an_error():
  records = [
      _record("001", lease=Lease(slot="gpu0")),
      _record("002", lease=Lease(slot="gpu1")),
  ]

  assert "slots-oversubscribed" in _rules(validate(records, slots=1))
  assert validate(records, slots=2).ok


def test_two_runs_on_one_slot_is_an_error_at_any_capacity():
  records = [
      _record("001", lease=Lease(slot="gpu0")),
      _record("002", lease=Lease(slot="gpu0")),
  ]
  assert "slot-contention" in _rules(validate(records, slots=8))


def test_an_expired_lease_does_not_count_against_capacity():
  stale = Lease(slot="gpu0", ttl_seconds=10.0)
  stale.renewed_at -= 1000
  records = [_record("001", lease=stale), _record("002", lease=Lease(slot="gpu1"))]

  assert validate(records, slots=1).ok


def test_one_slot_reproduces_the_original_single_trainer_mutex():
  """The rule this replaces was 'more than one running record is an error'."""
  records = [
      _record("001", verdict="running", lease=Lease(slot="gpu0")),
      _record("002", verdict="running", lease=Lease(slot="gpu0")),
  ]
  report = validate(records, slots=1)

  assert not report.ok
  assert {"slot-contention", "slots-oversubscribed"} & set(_rules(report))


def test_the_report_serialises_for_a_tool_response():
  payload = validate([_record("001", verdict="nonsense")]).to_dict()

  assert payload["ok"] is False
  assert payload["errors"][0]["rule"] == "verdict-enum"
