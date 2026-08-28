"""Every command, driven end to end against a real backend adapter.

The promise this file exists to keep is "the same commands on every backend".
Prose cannot keep it: IsaacLab's two gaps were written down in a doc and
nothing checked they were still the only two. So instead of describing what a
backend supports, this walks the controller's own handler table and calls every
verb on a live run -- through the session request/response path the CLI and the
MCP server use, not by poking the handler, so the envelope is exercised too.

Deferred verbs are driven to completion rather than counted as answered when
they enqueue. That distinction matters: `record_trace` returns a job object
immediately, and a check that stopped there would pass for a backend whose
tracing does not work at all.

The Genesis adapter is driven against the Go2Env-shaped fake in conftest, so
this runs in stock CI -- no Genesis, no GPU, no display.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rlmcp.adapters.genesis import GenesisSimAdapter
from rlmcp.core.controller import RlMcp
from rlmcp.core.curriculum import CurriculumStage, StageSchedule
from rlmcp.session import Session

GENESIS_UNSUPPORTED = {"live_view"}
"""What a Genesis run cannot do, in one place.

* ``live_view`` -- ``rlmcp view`` mirrors a scene into viser, and the Genesis
  backend has no scene handle to hand over. Screenshots, clips and progress
  clips all work; the live view is the one thing that does not. Future work,
  and deleting this entry is how it lands.

``rlmcp play`` is absent because it is not a controller command: it is a CLI
path that rebuilds an environment from a task registry, which is mjlab's alone
today -- IsaacLab cannot play either. Tracked separately.
"""

CALLS = {
    "help": {},
    "status": {},
    "list_parameters": {},
    "get_parameter": {"key": "reward.tracking_lin_vel.weight"},
    "set_parameter": {"key": "reward.tracking_lin_vel.weight", "value": 1.5,
                      "rationale": "conformance"},
    "reset_parameters": {},
    "reset_envs": {"env_ids": [0]},
    "list_metrics": {},
    "get_metrics": {},
    "plot_metrics": {"names": ["rlmcp/tilt_deg_mean"]},
    "screenshot": {"env_id": 0},
    "record_video": {"seconds": 0.05},
    "progress_video": {"every": 0},
    "live_view": {"enabled": True},
    "record_trace": {"seconds": 0.05},
    "plot_trace": {},
    "diagnose": {"seconds": 0.05},
    "curriculum_status": {},
    "curriculum_advance": {"reason": "conformance"},
    "curriculum_goto": {"stage": "0_slow"},  # a name, not an index
    "curriculum_auto": {"enabled": False},
    "pause": {},
    "resume": {},
    "step_once": {},
    "cancel_job": {"req_id": "no-such-job"},
    "save_checkpoint": {"tag": "conformance"},
    "list_checkpoints": {},
    "load_checkpoint": {"path": "no-such-checkpoint.pt"},
    "note": {"text": "conformance"},
    "feedback": {"text": "conformance"},
    "stop_training": {"reason": "conformance"},
}
"""One representative call per verb, chosen to be the cheapest thing that still
exercises the handler for real. A verb missing from here is caught below, so
the table cannot rot quietly as the controller grows."""

ORDER = [
    # Read-only first, while nothing has been changed underneath them.
    "help", "status", "list_parameters", "get_parameter", "list_metrics",
    "get_metrics", "curriculum_status", "list_checkpoints",
    # Then the ones that record something.
    "record_trace", "diagnose", "plot_trace", "plot_metrics",
    "screenshot", "record_video", "progress_video", "live_view",
    # Then the ones that change the run.
    "set_parameter", "reset_parameters", "reset_envs",
    "curriculum_advance", "curriculum_goto", "curriculum_auto",
    "save_checkpoint", "load_checkpoint", "note", "feedback", "cancel_job",
    # Pause blocks servicing until it is undone, so its undo follows it
    # immediately -- and stopping ends the session, so it goes last.
    "pause", "step_once", "resume",
    "stop_training",
]
"""The order these run in, which is load-bearing rather than tidy.

`plot_trace` needs a trace to plot. `pause` makes `service()` block until
something resumes it, so the nine verbs alphabetical order once put between
`pause` and `resume` hung this suite for two minutes before it was noticed.
And `stop_training` ends the session, so nothing can follow it. A test that
drives every command has to know that some of them are not independent."""

NEEDS_ITS_UNDO_QUEUED = {"pause": "resume"}
"""Verbs whose own semantics stop the servicing loop, so the test has to queue
the undo alongside them. See the note in :func:`drive`."""

WRONG_ARGUMENT_NOT_MISSING_CAPABILITY = {"cancel_job", "load_checkpoint"}
"""These are asked for something that does not exist on purpose -- an unknown
job id, an absent checkpoint -- so a refusal is the right answer and says
nothing about the backend. Every other verb is called with arguments it should
be able to satisfy."""


def plan() -> StageSchedule:
  """A two-rung ladder in Genesis's own vocabulary.

  The curriculum verbs need a schedule to act on; without one they refuse, and
  a conformance run that accepted that refusal would be reporting the fixture's
  shape as a property of the backend. The parameters are real Genesis keys, so
  advancing a stage actually writes through the adapter.
  """
  return StageSchedule([
      CurriculumStage(
          name="0_slow",
          parameters={"command.lin_vel_x": [0.0, 0.5]},
          min_iterations=1,
      ),
      CurriculumStage(
          name="1_faster",
          parameters={"command.lin_vel_x": [0.5, 1.5]},
          min_iterations=1,
      ),
  ])


@pytest.fixture
def genesis_lab(genesis_env, fake_runner, tmp_path) -> RlMcp:
  return RlMcp(
      sim_adapter=GenesisSimAdapter(genesis_env),
      runner_adapter=fake_runner,
      session_dir=tmp_path / "session",
      curriculum=plan(),
      video_every=0,
  )


def drive(lab: RlMcp, name: str, steps: int = 12):
  """Submit one verb the way the CLI does and see it through to a response.

  Deferred jobs -- traces, clips -- answer only after the simulation has run,
  so the loop below stands in for the training steps that would feed them.
  """
  client = Session.open(lab.session.dir)
  request = client.submit(name, **CALLS[name])
  if name in NEEDS_ITS_UNDO_QUEUED:
    # `service()` loops while paused, by design: a paused run is waiting for
    # somebody to resume it, and in a real session that somebody is another
    # shell. Queueing the undo before servicing is how a single-threaded test
    # plays both parts. Without it this call blocks for ever, which is the
    # correct behaviour of the command and a hang of the harness.
    client.submit(NEEDS_ITS_UNDO_QUEUED[name])
  lab.service(iteration=lab.iteration)
  response = client.poll(request.req_id)
  for _ in range(steps):
    if response is not None:
      break
    for _ in range(10):
      lab.on_step()
    lab.service(iteration=lab.iteration + 1)
    response = client.poll(request.req_id)
  return response


def refusal(lab: RlMcp, name: str) -> str:
  """"" when the verb answered; why not, when it did not."""
  response = drive(lab, name)
  if response is None:
    return "never answered: the job was accepted and never completed"
  if response.ok:
    return ""
  error = str(response.error or "")
  if name in WRONG_ARGUMENT_NOT_MISSING_CAPABILITY:
    return ""
  if "not supported" in error.lower() or "notsupported" in error.lower():
    return f"NotSupported: {error}"
  return f"failed: {error}"


# The table itself.


def test_every_command_is_covered_by_this_file(genesis_lab):
  """A verb added to the controller must be added here, or nothing checks it
  on any backend."""
  missing = set(genesis_lab._handlers) - set(CALLS)
  assert not missing, f"no conformance call for: {sorted(missing)}"


def test_no_call_is_listed_for_a_verb_that_does_not_exist(genesis_lab):
  stale = set(CALLS) - set(genesis_lab._handlers)
  assert not stale, f"conformance calls for verbs that are gone: {sorted(stale)}"


def test_the_run_order_covers_every_call():
  """Ordering is hand-written because some verbs depend on others, so a verb
  left out of it would be silently untested."""
  assert set(ORDER) == set(CALLS), (
      f"ORDER and CALLS disagree: {sorted(set(CALLS) ^ set(ORDER))}"
  )


# Genesis.


def test_every_command_answers_on_genesis_or_is_declared(genesis_lab):
  refused = {name: why for name in ORDER if (why := refusal(genesis_lab, name))}
  unexpected = set(refused) - GENESIS_UNSUPPORTED
  assert not unexpected, (
      "these commands do not work on Genesis and are not declared as such: "
      f"{ {k: refused[k] for k in sorted(unexpected)} }"
  )


def test_the_declared_gaps_are_still_real(genesis_lab):
  """The declared set has to shrink deliberately. A capability that quietly
  starts working should be claimed, not left listed as missing."""
  still_missing = {n for n in GENESIS_UNSUPPORTED if refusal(genesis_lab, n)}
  assert still_missing == GENESIS_UNSUPPORTED, (
      "these are declared unsupported but now work: "
      f"{sorted(GENESIS_UNSUPPORTED - still_missing)}. Delete them from "
      "GENESIS_UNSUPPORTED and from Known differences in docs/genesis.md."
  )


def test_the_declared_gaps_are_documented():
  """Whatever a backend cannot do, a reader should find out from its page."""
  text = (Path(__file__).resolve().parent.parent / "docs" / "genesis.md").read_text()
  for name in GENESIS_UNSUPPORTED:
    assert name in text, (
        f"{name} is unsupported on Genesis but docs/genesis.md does not name it"
    )


# The wrap surface, which is the other half of "the same API".


@pytest.mark.parametrize("module", ["mjlab", "isaaclab", "genesis"])
def test_every_backend_exports_the_same_entry_points(module):
  """One import line, one wrap(), one TrainingStopped, whatever is underneath."""
  import importlib

  backend = importlib.import_module(f"rlmcp.adapters.{module}")
  assert callable(backend.wrap)
  assert issubclass(backend.TrainingStopped, Exception)
  assert hasattr(backend.RlMcpEnvWrapper, "attach_runner")
  assert hasattr(backend.RlMcpEnvWrapper, "build_sim_adapter")


def test_the_recording_commands_leave_real_files_behind(genesis_lab):
  """Answering is not the same as working.

  `screenshot`, `record_trace` and `record_video` all return happily while
  producing nothing, so the response is not the evidence -- the artifact is.
  """
  for name in ("screenshot", "record_trace", "record_video", "plot_trace",
               "save_checkpoint"):
    assert drive(genesis_lab, name) is not None, f"{name} never answered"

  produced = {p.suffix for p in genesis_lab.session.dir.rglob("*") if p.is_file()}
  for suffix, what in ((".png", "a screenshot"), (".npz", "a trace"),
                       (".mp4", "a clip"), (".pt", "a checkpoint")):
    assert suffix in produced, f"{what} was reported but no {suffix} was written"


def test_a_trace_carries_the_channels_the_diagnostics_read(genesis_lab):
  """A trace of the wrong channels diagnoses as empty, which reads as a healthy
  run rather than as a broken adapter."""
  import numpy as np

  assert drive(genesis_lab, "record_trace") is not None
  traces = sorted(genesis_lab.session.dir.rglob("trace_*.npz"))
  assert traces, "record_trace answered but wrote no trace"
  recorded = set(np.load(traces[-1]).files)
  assert {"joint_pos", "joint_vel", "action", "base_lin_vel", "command",
          "time"} <= recorded, f"trace holds only {sorted(recorded)}"
