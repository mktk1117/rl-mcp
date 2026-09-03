"""Binding a live run to its record, and the documents that come out."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from rlmcp.core.curriculum import Condition
from rlmcp.records import ConflictError  # The package-level export callers use.
from rlmcp.records.filestore import FileStore
from rlmcp.records.link import RecordLink
from rlmcp.records.record import Falsifier, Weights
from rlmcp.records.report import render_plan, render_report
from rlmcp.records.store import StoreError


@pytest.fixture
def store(tmp_path) -> FileStore:
  return FileStore(tmp_path / "records", slots=1)


@pytest.fixture
def repo_with_package(tmp_path) -> Path:
  """A task repository, one commit deep -- what a launch stamps."""
  import subprocess

  root = tmp_path / "tasks"
  (root / "shand").mkdir(parents=True)
  (root / "shand" / "task.py").write_text("weight = 1.0\n")

  def git(*args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True,
                   capture_output=True)

  subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
  git("config", "user.email", "test@example.com")
  git("config", "user.name", "Test")
  git("add", "-A")
  git("commit", "-qm", "the task")
  return root


# Registration.


def test_an_unregistered_run_says_so_loudly_but_still_runs(store, capsys):
  link = RecordLink(store)

  assert link.record is None
  assert "no lab record" in capsys.readouterr().out
  # And every hook is a no-op rather than an error.
  link.start("/tmp/session", {"a": 1})
  link.heartbeat()
  link.finish("done")
  assert link.status() == {"registered": False, "warning": "run has no lab record"}


def test_strict_refuses_to_start_without_a_record(store):
  with pytest.raises(StoreError, match="Pre-register the run"):
    RecordLink(store, strict=True)


def test_strict_refuses_an_id_that_does_not_exist(store):
  with pytest.raises(StoreError, match="No lab record"):
    RecordLink(store, record_id="404", strict=True)


def test_a_missing_id_warns_but_proceeds_when_not_strict(store, capsys):
  link = RecordLink(store, record_id="404")
  assert link.record is None
  assert "No lab record '404'" in capsys.readouterr().out


# Lifecycle.


def test_starting_attaches_the_session_and_claims_a_slot(store):
  record = store.new_record("run", hypothesis="h")
  link = RecordLink(store, record_id=record.id, slot="gpu0")

  link.start("/logs/run/rlmcp")

  live = store.get_record(record.id)
  assert live.session == "/logs/run/rlmcp"
  assert live.verdict == "running"
  assert live.lease.slot == "gpu0"


def test_the_config_snapshot_waits_for_everything_to_attach(store):
  """The runner contributes its hyperparameters after the environment does."""
  record = store.new_record("run")
  link = RecordLink(store, record_id=record.id)
  link.start("/logs/run/rlmcp", {"reward.a.weight": 1.0})

  link.snapshot_config({"reward.a.weight": 1.0, "rl.learning_rate": 0.001})

  assert store.get_record(record.id).config == {
      "reward.a.weight": 1.0, "rl.learning_rate": 0.001
  }


def test_the_config_is_snapshotted_only_once(store):
  record = store.new_record("run")
  link = RecordLink(store, record_id=record.id)
  link.start("/logs/run/rlmcp")

  link.snapshot_config({"first": 1})
  link.snapshot_config({"second": 2})  # a later iteration must not overwrite it

  assert store.get_record(record.id).config == {"first": 1}


def test_finishing_releases_the_slot_but_keeps_the_verdict_open(store):
  record = store.new_record("run")
  link = RecordLink(store, record_id=record.id, slot="gpu0")
  link.start("/logs/run/rlmcp")

  link.finish("stopped by the agent")

  live = store.get_record(record.id)
  assert live.lease is None
  assert live.verdict == "running"  # closing out is a deliberate, separate act
  assert live.links["exit"] == "stopped by the agent"


def test_a_busy_slot_warns_rather_than_killing_the_run(store, capsys):
  other = store.new_record("incumbent")
  store.claim(other.id, slot="gpu0")
  record = store.new_record("newcomer")

  RecordLink(store, record_id=record.id, slot="gpu0").start("/logs/x")

  assert "could not claim slot" in capsys.readouterr().out
  assert store.get_record(record.id).verdict == "running"


def test_strict_makes_a_busy_slot_fatal(store):
  other = store.new_record("incumbent")
  store.claim(other.id, slot="gpu0")
  record = store.new_record("newcomer")

  link = RecordLink(store, record_id=record.id, slot="gpu0", strict=True)

  with pytest.raises(StoreError):
    link.start("/logs/x")


def test_heartbeats_are_rate_limited(store):
  record = store.new_record("run")
  link = RecordLink(store, record_id=record.id, heartbeat_seconds=1e6)
  link.start("/logs/run")
  before = store.get_record(record.id).lease.renewed_at

  link.heartbeat()  # too soon to matter

  assert store.get_record(record.id).lease.renewed_at == before


def test_a_leaseless_run_says_so_and_keeps_its_record_fresh(store, capsys):
  """A failed claim leaves a running record with no lease; its freshness is
  the only live signal the reaper has, so the heartbeat must keep writing."""
  other = store.new_record("incumbent")
  store.claim(other.id, slot="gpu0")
  record = store.new_record("newcomer")
  link = RecordLink(store, record_id=record.id, slot="gpu0", heartbeat_seconds=0.0)

  link.start("/logs/x")

  out = capsys.readouterr().out
  assert "running without a lease" in out
  assert "reaped by heartbeat staleness" in out

  before = store.get_record(record.id).updated_at
  time.sleep(0.01)
  link.heartbeat()

  after = store.get_record(record.id)
  assert after.lease is None
  assert after.verdict == "running"
  assert after.updated_at > before


# Surviving a flaky records. A hiccup must never stall training, and it must
# not silently eat the record either.


class _FlakyStore:
  """Forwards to a real store, but fails record writes the next N times."""

  def __init__(self, inner, failures: int = 0):
    self.inner = inner
    self.failures = failures

  def _flake(self) -> None:
    if self.failures > 0:
      self.failures -= 1
      raise StoreError("the records is briefly unwritable")

  def put_record(self, record):
    self._flake()
    return self.inner.put_record(record)

  def update_record(self, record_id, mutate, retries: int = 3):
    self._flake()
    return self.inner.update_record(record_id, mutate, retries)

  def __getattr__(self, name):
    return getattr(self.inner, name)


def test_a_failed_config_snapshot_is_retried_not_lost(store):
  record = store.new_record("run")
  flaky = _FlakyStore(store)
  link = RecordLink(flaky, record_id=record.id)
  link.start("/logs/run/rlmcp")
  flaky.failures = 1

  link.snapshot_config({"rl.learning_rate": 0.001})  # fails; must stay pending
  link.snapshot_config({"rl.learning_rate": 0.001})  # the next tick retries

  assert store.get_record(record.id).config == {"rl.learning_rate": 0.001}
  # And a success still closes the once-only window.
  link.snapshot_config({"rl.learning_rate": 999.0})
  assert store.get_record(record.id).config == {"rl.learning_rate": 0.001}


def test_the_config_snapshot_warns_once_while_it_retries(store, capsys):
  record = store.new_record("run")
  flaky = _FlakyStore(store)
  link = RecordLink(flaky, record_id=record.id)
  link.start("/logs/run/rlmcp")
  capsys.readouterr()
  flaky.failures = 3

  for _ in range(3):
    link.snapshot_config({"a": 1})

  assert capsys.readouterr().out.count("could not snapshot the launch config") == 1
  link.snapshot_config({"a": 1})
  assert store.get_record(record.id).config == {"a": 1}


def test_a_failed_write_at_start_warns_once_and_the_run_proceeds(store, capsys):
  record = store.new_record("run")
  flaky = _FlakyStore(store, failures=1)
  link = RecordLink(flaky, record_id=record.id)
  capsys.readouterr()

  link.start("/logs/run/rlmcp")

  assert capsys.readouterr().out.count("proceed UNRECORDED") == 1
  # The training-side object keeps working, with unregistered-run semantics.
  assert link.record is None
  link.snapshot_config({"a": 1})
  link.heartbeat()
  link.finish("done")
  assert link.status() == {"registered": False, "warning": "run has no lab record"}
  # The record itself is untouched -- still the planned one, closable by hand.
  assert store.get_record(record.id).verdict == "planned"


def test_strict_makes_a_failed_write_at_start_fatal(store):
  record = store.new_record("run")
  link = RecordLink(_FlakyStore(store, failures=1), record_id=record.id, strict=True)

  with pytest.raises(StoreError, match="unwritable"):
    link.start("/logs/run/rlmcp")


class _ExitWriteConflictsOnce(FileStore):
  """The reaper interleave: the first write carrying the exit reason loses
  its compare-and-swap, exactly once."""

  def __init__(self, root):
    super().__init__(root)
    self.conflicts = 0

  def put_record(self, record):
    if record.links.get("exit") and not self.conflicts:
      self.conflicts += 1
      raise ConflictError(record.id, record.version, record.version + 1)
    return super().put_record(record)


def test_a_conflict_interleaved_finish_keeps_the_exit_reason(tmp_path):
  """finish() writes the exit reason through the store's retry helper, so a
  writer sneaking in between cannot make the reason silently vanish."""
  store = _ExitWriteConflictsOnce(tmp_path / "records")
  record = store.new_record("run")
  link = RecordLink(store, record_id=record.id, slot="gpu0")
  link.start("/logs/run/rlmcp")

  link.finish("stopped by the agent")

  live = store.get_record(record.id)
  assert store.conflicts == 1  # The interleave actually happened.
  assert live.links["exit"] == "stopped by the agent"
  assert live.lease is None


# The falsifier watch.


def test_the_falsifier_is_not_read_before_its_read_point(store):
  """Every policy is bad at iteration zero."""
  record = store.new_record(
      "run",
      falsifier=Falsifier(
          conditions=[Condition("rlmcp/episode_length_frac", "<=", 0.2)],
          check_after=100,
      ),
  )
  link = RecordLink(store, record_id=record.id)
  metrics = {"rlmcp/episode_length_frac": 0.05}

  early = link.check_falsifier(metrics, iteration=10)
  late = link.check_falsifier(metrics, iteration=150)

  assert early["fired"] is False and early["too_early"] is True
  assert late["fired"] is True


def test_status_carries_the_hypothesis_under_test(store):
  record = store.new_record(
      "run", hypothesis="flat first", parent=None,
      weights=Weights("001", "model_5.pt"),
      falsifier=Falsifier(prose="it falls over"),
  )
  status = RecordLink(store, record_id=record.id).status()

  assert status["registered"] is True
  assert status["hypothesis"] == "flat first"
  assert status["falsifier"] == "it falls over"
  assert status["warm_start"] == "001 @ model_5.pt"


# Documents.


def test_the_plan_carries_the_four_sections_and_the_recipe(store):
  record = store.new_record(
      "wide_scan",
      hypothesis="a wider scan lifts the plateau",
      prediction="level above 6.1",
      falsifier=Falsifier(prose="plateau stays flat",
                          conditions=[Condition("terrain/level", "<=", 6.1)]),
      change=["scan width 1.6m"],
  )

  plan = render_plan(record, recipe=[("001", ["scan width 1.6m"])])

  for heading in ("## Hypothesis", "## Prediction", "## Falsifier", "## Change"):
    assert heading in plan
  assert "`terrain/level <= 6.1`" in plan
  assert "Recipe at this node" in plan


def test_the_plan_prompts_for_what_is_missing(store):
  plan = render_plan(store.new_record("empty"))
  assert "not an experiment" in plan


def test_the_plan_flags_a_warm_start_before_the_run_starts(store):
  record = store.new_record("finetune", weights=Weights("001", "model_5.pt"))
  assert "caps at `provisional`" in render_plan(record)


def test_the_report_distinguishes_fired_from_undecidable(store):
  record = store.new_record("run", falsifier=Falsifier(prose="it gets worse"))
  record.verdict = "falsified"
  record.outcome = "it got worse"

  fired = render_report(record, {}, {"fired": True, "checks": []})
  unknown = render_report(record, {}, {"fired": False, "undecidable": True, "checks": []})

  assert "It fired" in fired
  assert "Undecidable" in unknown
  assert "not survival" in unknown


def test_the_report_tabulates_live_interventions(store):
  record = store.new_record("run")
  evidence = {
      "iterations": 500,
      "num_metric_rows": 500,
      "final_metrics": {"Train/mean_reward": 12.5},
      "interventions": [
          {"iteration": 100, "key": "reward.slip.weight", "old": -0.1,
           "new": -0.3, "rationale": "feet skating"},
      ],
      "stages": [{"iteration": 0, "to": "0_flat", "notes": "flat only"}],
      "notes": [],
      "artifacts": [],
  }

  report = render_report(record, evidence)

  assert "reward.slip.weight" in report
  assert "feet skating" in report
  assert "the final config is not the launch config" in report
  assert "`Train/mean_reward` | 12.5" in report


# Code provenance.


def test_a_run_stamps_the_code_it_launched_with(store, repo_with_package):
  """The other half of a recipe: config has always been recorded, code has not."""
  record = store.new_record("ema-filter", hypothesis="the filter is the unlock")
  link = RecordLink(store, record_id=record.id, code_root=str(repo_with_package))

  link.start("/tmp/session", {"reward.weight": 1.0})

  code = store.get_record(record.id).code
  assert code["kind"] == "git" and code["clean"] is True
  assert code["head"]["commit"] and code["tree"]
  assert code["ref"].endswith(record.id)


def test_a_dirty_launch_is_recorded_and_said_out_loud(store, repo_with_package, capsys):
  """A run whose tree was dirty is not reproducible from its commit alone, so
  the number that matters is said at launch rather than discovered later."""
  (repo_with_package / "shand" / "task.py").write_text("weight = 9.0\n")
  record = store.new_record("dirty-launch")
  link = RecordLink(store, record_id=record.id, code_root=str(repo_with_package))

  link.start("/tmp/session")

  code = store.get_record(record.id).code
  assert code["clean"] is False and code["dirty"]["files"] == 1
  assert "uncommitted lines" in capsys.readouterr().out


def test_stamping_can_be_declined_and_a_loose_directory_still_launches(
    store, tmp_path, capsys):
  off = store.new_record("no-stamp")
  RecordLink(store, record_id=off.id, code_root="").start("/tmp/session")
  assert store.get_record(off.id).code == {}

  loose = store.new_record("outside-git")
  RecordLink(store, record_id=loose.id, code_root=str(tmp_path)).start("/tmp/session")
  assert store.get_record(loose.id).code["kind"] == "none"
  assert "no code snapshot" in capsys.readouterr().out


def test_a_launchers_stamp_survives_a_trainer_told_not_to_stamp(store, repo_with_package):
  """A launcher that materializes the code (a studio, a fleet) stamps the
  record itself, before the process exists, and then runs the trainer from a
  plain directory with ``--code-root ''``: "do not stamp, I already did". The
  trainer must keep that stamp, not replace it with ``{}`` -- the tree the
  run actually trains from is the one the launcher wrote, and nothing the
  trainer can see from its cwd is truer."""
  from rlmcp.records.snapshot import capture

  record = store.new_record("materialized")
  record.code = capture(repo_with_package, record_id=record.id)
  store.put_record(record)
  assert record.code["kind"] == "git"

  RecordLink(store, record_id=record.id, code_root="").start("/tmp/session")

  kept = store.get_record(record.id).code
  assert kept["kind"] == "git" and kept["tree"] == record.code["tree"], \
      "the trainer erased the launcher's stamp"
  assert RecordLink.KEEPS_LAUNCHER_STAMP is True, "a launcher needs a way to ask"
