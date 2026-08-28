"""End-to-end command handling, driven through the session directory."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import FakeSimAdapter

from rlmcp.adapters.base import SimAdapter
from rlmcp.core.controller import RlMcp
from rlmcp.core.curriculum import (
  METRIC_EPISODE_LENGTH_FRAC,
  Action,
  Condition,
  CurriculumStage,
  StageSchedule,
)
from rlmcp.core.parameters.spec import Liveness
from rlmcp.session import Session

METRIC_TERRAIN_LEVEL_FRAC = "rlmcp/terrain_level_frac"


def _plan() -> StageSchedule:
  return StageSchedule(
      [
          CurriculumStage(
              name="0_flat",
              apply=[Action("set_terrain", {"terrains": ["flat"], "max_level": 2})],
              parameters={"reward.action_rate_l2.weight": -0.05},
              promote_when=[
                  Condition(METRIC_EPISODE_LENGTH_FRAC, ">=", 0.5),
                  Condition(METRIC_TERRAIN_LEVEL_FRAC, ">=", 0.5),
              ],
              min_iterations=2,
              hold_iterations=1,
          ),
          CurriculumStage(
              name="1_rough",
              apply=[
                  Action(
                      "set_terrain",
                      {"terrains": ["flat", "random_rough"], "max_level": 4},
                  ),
                  Action(
                      "set_parameter",
                      {"key": "command.twist.ranges.lin_vel_x", "value": [-2.0, 2.0]},
                  ),
              ],
          ),
      ]
  )


@pytest.fixture
def lab(tmp_path, fake_sim, fake_runner, fake_terrain) -> RlMcp:
  controller = RlMcp(
      sim_adapter=fake_sim,
      runner_adapter=fake_runner,
      session_dir=tmp_path / "session",
      curriculum=_plan(),
      extensions=[fake_terrain],
      # No progress clips: these fixtures assert on the deferred-job
      # queue, and a scheduled clip sitting in it is noise. Clips have
      # their own suite (tests/test_progress_video.py).
      video_every=0,
  )
  yield controller
  controller.close()


@pytest.fixture
def bare_lab(tmp_path, fake_sim, fake_runner) -> RlMcp:
  """A run whose environment supports no extensions at all."""
  controller = RlMcp(
      sim_adapter=fake_sim,
      runner_adapter=fake_runner,
      session_dir=tmp_path / "bare",
      video_every=0,
  )
  yield controller
  controller.close()


def _client(lab: RlMcp) -> Session:
  return Session.open(lab.session.dir)


def _run(lab: RlMcp, cmd: str, **args):
  """Submit a command, let the trainer service it, and return the response."""
  client = _client(lab)
  request = client.submit(cmd, **args)
  lab.service(iteration=lab.iteration)
  return client.poll(request.req_id)


def test_session_is_discoverable_and_describes_itself(lab):
  assert Session.find_latest(lab.session.dir.parent).dir == lab.session.dir
  assert _run(lab, "help").result["commands"]["set_parameter"]


def test_status_reports_stage_and_terrain(lab):
  response = _run(lab, "status")

  assert response.ok
  assert response.result["curriculum"]["stage"] == "0_flat"
  assert response.result["extensions"]["terrain"]["active_terrains"] == ["flat"]
  assert response.result["num_envs"] == 12


def test_status_says_what_where_will_accept_on_this_run(lab):
  """A client cannot offer `--where terrain=...` unless the run says the
  criterion exists and which values are live right now."""
  selectors = _run(lab, "status").result["selectors"]

  assert selectors["terrain"]["values"] == ["flat"]
  assert selectors["terrain"]["extension"] == "terrain"
  assert selectors["level"]["values"] == [0, 1]

  # Advertised and answerable are the same list, which is the whole promise.
  assert _run(lab, "screenshot", where={"terrain": "flat"}).ok


def test_opening_stage_is_applied_before_the_first_iteration(lab, fake_sim):
  # Constructing the controller does not touch the sim; the first service does.
  lab.service(iteration=0)

  assert fake_sim.enabled == ["flat"]
  assert fake_sim.level_ceiling == 2
  assert lab.parameters.get_value("reward.action_rate_l2.weight") == -0.05


def test_set_parameter_reaches_the_simulator_and_the_event_log(lab, fake_sim):
  response = _run(lab, "set_parameter", key="reward.action_rate_l2.weight",
                  value=-0.3, rationale="knees buzzing")

  assert response.ok and response.result["new_value"] == -0.3
  assert fake_sim.get_parameter("reward.action_rate_l2.weight") == -0.3
  events = [e for e in _client(lab).events() if e["kind"] == "set_parameter"]
  assert events[-1]["rationale"] == "knees buzzing"


class NotingFakeSim(FakeSimAdapter):
  """FakeSimAdapter that reports liveness and per-write notes, like mjlab.

  Mirrors MjlabSimAdapter: ``last_set_notes()`` holds the notes of the most
  recent sim write only (overwritten on every sim write, untouched by runner
  hyperparameter writes).
  """

  AT_RESET_KEY = "event.push_robot.velocity_range"

  def __init__(self):
    super().__init__()
    self._params[self.AT_RESET_KEY] = [0.0, 0.5]
    self._last_set_notes = {}

  def discover_parameters(self):
    specs = super().discover_parameters()
    for spec in specs:
      if spec.key == self.AT_RESET_KEY:
        spec.liveness = Liveness.AT_RESET
    return specs

  def set_parameter(self, key, value):
    ok = super().set_parameter(key, value)
    if key == self.AT_RESET_KEY:
      self._last_set_notes = {
          "liveness": "at_reset",
          "note": ("Applied to the live config; takes effect from each "
                   "environment's next reset."),
          "curriculum": "velocity-curriculum table rewritten",
      }
    else:
      self._last_set_notes = {}
    return ok

  def last_set_notes(self):
    return dict(self._last_set_notes)


def test_at_reset_write_reports_liveness_and_adapter_notes(tmp_path, fake_runner):
  sim = NotingFakeSim()
  controller = RlMcp(sim_adapter=sim, runner_adapter=fake_runner,
                      session_dir=tmp_path / "noting")
  try:
    response = _run(controller, "set_parameter", key=NotingFakeSim.AT_RESET_KEY,
                    value=[0.0, 1.0], rationale="stronger pushes")
    assert response.ok and response.result["applied"] is True
    assert response.result["liveness"] == "at_reset"
    assert "next reset" in response.result["note"]
    assert response.result["curriculum"] == "velocity-curriculum table rewritten"

    # A runner hyperparameter write never routes through the sim adapter, so
    # it must not pick up the sim's (still populated) note from the write above.
    hyper = _run(controller, "set_parameter", key="rl.learning_rate", value=5e-4)
    assert hyper.ok and hyper.result["liveness"] == "live"
    assert "note" not in hyper.result and "curriculum" not in hyper.result

    # A later live sim write carries no stale note either.
    live = _run(controller, "set_parameter",
                key="reward.action_rate_l2.weight", value=-0.2)
    assert live.ok and live.result["liveness"] == "live"
    assert "note" not in live.result and "curriculum" not in live.result
  finally:
    controller.close()


def test_refused_write_is_truthful_and_does_not_republish_schema(
    lab, fake_sim, monkeypatch):
  """A write the adapter refuses must say applied=false and change nothing."""
  published = []
  monkeypatch.setattr(lab.session, "publish_params",
                      lambda schema: published.append(schema))
  monkeypatch.setattr(fake_sim, "set_parameter", lambda key, value: False)

  response = _run(lab, "set_parameter", key="reward.track_linear_velocity.weight",
                  value=1.5, rationale="try anyway")

  assert response.ok and response.result["applied"] is False
  assert response.result["new_value"] == response.result["old_value"]
  assert not published  # An unchanged schema is not republished.
  events = [e for e in _client(lab).events() if e["kind"] == "set_parameter"]
  assert events[-1]["applied"] is False


def test_range_parameters_accept_two_element_lists(lab, fake_sim):
  response = _run(lab, "set_parameter", key="command.twist.ranges.lin_vel_x",
                  value=[-2.0, 3.0], rationale="faster commands")

  assert response.ok
  assert fake_sim.get_parameter("command.twist.ranges.lin_vel_x") == [-2.0, 3.0]


def test_out_of_bounds_values_are_refused(lab, fake_sim):
  response = _run(lab, "set_parameter", key="reward.action_rate_l2.weight",
                  value=1e6, rationale="oops")

  assert not response.ok and "above maximum" in response.error
  assert fake_sim.get_parameter("reward.action_rate_l2.weight") != 1e6


def test_unknown_command_lists_what_is_available(lab):
  response = _run(lab, "make_coffee")

  assert not response.ok
  assert "Unknown command" in response.error and "set_parameter" in response.error


def test_bad_arguments_do_not_kill_the_trainer(lab):
  response = _run(lab, "set_parameter", nonsense=1)

  assert not response.ok and "Bad arguments" in response.error
  assert _run(lab, "status").ok  # Still serving.


def test_reset_parameters_restores_startup_values(lab, fake_sim):
  _run(lab, "set_parameter", key="reward.track_linear_velocity.weight", value=5.0)
  response = _run(lab, "reset_parameters")

  assert response.ok
  assert fake_sim.get_parameter("reward.track_linear_velocity.weight") == 2.0


def test_set_terrain_rejects_unknown_names(lab, fake_sim):
  response = _run(lab, "set_terrain", terrains=["lava"])

  assert not response.ok and "lava" in response.error
  assert fake_sim.enabled == ["flat"]


def test_curriculum_advances_automatically_when_metrics_hold(lab, fake_sim):
  lab.service(iteration=0)
  fake_sim.terrain_levels[:] = 1  # level_frac = 1/(2-1) = 1.0 against ceiling 2

  for iteration in range(1, 6):
    lab.service(iteration=iteration)

  assert lab.curriculum.current.name == "1_rough"
  assert fake_sim.enabled == ["flat", "random_rough"]
  assert fake_sim.level_ceiling == 4
  assert fake_sim.get_parameter("command.twist.ranges.lin_vel_x") == [-2.0, 2.0]


def test_curriculum_can_be_driven_by_hand(lab, fake_sim):
  lab.service(iteration=0)
  response = _run(lab, "curriculum_advance", reason="operator says so")

  assert response.ok and response.result["to"] == "1_rough"
  assert fake_sim.enabled == ["flat", "random_rough"]

  back = _run(lab, "curriculum_goto", stage="0_flat", reason="too hard")
  assert back.ok and fake_sim.enabled == ["flat"]


def test_auto_promotion_can_be_switched_off(lab, fake_sim):
  lab.service(iteration=0)
  _run(lab, "curriculum_auto", enabled=False)
  fake_sim.terrain_levels[:] = 1

  for iteration in range(1, 10):
    lab.service(iteration=iteration)

  assert lab.curriculum.current.name == "0_flat"


def test_screenshot_can_target_an_env_by_description(lab, fake_sim):
  _run(lab, "set_terrain", terrains=["flat", "random_rough"])
  response = _run(lab, "screenshot", where={"terrain": "random_rough"})

  assert response.ok
  assert Path(response.result["image_path"]).exists()
  assert response.result["env_id"] in fake_sim.env_ids_on(terrain="random_rough")


def test_screenshot_where_nothing_matches_explains_itself(lab):
  response = _run(lab, "screenshot", where={"terrain": "pyramid_stairs"})

  assert not response.ok
  assert "No environment currently matches" in response.error


def test_a_description_no_extension_understands_says_so(lab):
  response = _run(lab, "screenshot", where={"object": "cube"})

  assert not response.ok
  assert "No extension understands" in response.error


def test_an_env_without_extensions_has_neither_the_commands_nor_the_concept(bare_lab):
  commands = _run(bare_lab, "help").result["commands"]

  assert "set_terrain" not in commands
  assert "terrain_status" not in commands
  assert "screenshot" in commands  # Core verbs are unaffected.
  assert "extensions" not in _run(bare_lab, "status").result


def test_extension_commands_are_reachable_like_any_other(lab, fake_sim):
  response = _run(lab, "set_terrain", terrains=["flat", "random_rough"], max_level=3)

  assert response.ok
  assert fake_sim.enabled == ["flat", "random_rough"]
  assert "set_terrain" in _run(lab, "help").result["commands"]


def test_extension_metrics_join_the_telemetry(lab, fake_sim):
  fake_sim.terrain_levels[:] = 1
  lab.service(iteration=1)

  metrics = _run(lab, "get_metrics", names=["rlmcp/terrain_level_mean"]).result
  assert metrics["metrics"]["rlmcp/terrain_level_mean"][-1][1] == 1.0


def test_command_name_clashes_are_first_wins_and_logged(lab, fake_sim):
  """Late extensions cannot shadow built-ins or earlier extensions' verbs."""
  from rlmcp.core.extensions import Extension

  class Impostor(Extension):
    name = "impostor"

    def __init__(self):
      super().__init__(env=None)

    def available(self):
      return True

    def commands(self):
      return {"set_terrain": self._steal, "status": self._steal,
              "impostor_only": self._steal}

    def _steal(self, **kwargs):
      """Marker handler that must never win a clash."""
      return {"stolen": True}

  assert lab.add_extension(Impostor())

  # The clashing verbs still belong to their first owners...
  terrain = _run(lab, "set_terrain", terrains=["flat", "random_rough"])
  assert terrain.ok and "stolen" not in (terrain.result or {})
  assert fake_sim.enabled == ["flat", "random_rough"]
  status = _run(lab, "status")
  assert status.ok and "stolen" not in status.result
  # ...the non-clashing verb landed...
  assert _run(lab, "impostor_only").result == {"stolen": True}
  # ...and each clash is on the record, naming both sides.
  conflicts = [e for e in _client(lab).events() if e["kind"] == "command_conflict"]
  named = {(e["command"], e["kept"], e["ignored"]) for e in conflicts}
  assert ("set_terrain", "terrain", "impostor") in named
  assert ("status", "built-in", "impostor") in named


def test_non_floatable_metric_values_are_reported_once_per_key(lab):
  """A metric that cannot be plotted must not vanish without a trace."""
  lab.service(iteration=1, metrics={"Train/notes": "diverged",
                                    "Train/mean_reward": 1.0})
  lab.service(iteration=2, metrics={"Train/notes": "still text"})

  drops = [e for e in _client(lab).events() if e["kind"] == "telemetry_drop"]
  assert len(drops) == 1
  assert drops[0]["key"] == "Train/notes"
  assert "diverged" in drops[0]["value"]
  # The floatable metric recorded normally.
  series = _run(lab, "get_metrics", names=["Train/mean_reward"]).result
  assert series["metrics"]["Train/mean_reward"]


def test_trace_completes_across_steps_and_reports(lab):
  client = _client(lab)
  request = client.submit("diagnose", seconds=0.4)
  lab.service(iteration=1)

  assert client.poll(request.req_id) is None  # Deferred: needs simulation steps.
  for _ in range(20):
    lab.on_step()
  lab.service(iteration=2)

  response = client.poll(request.req_id)
  assert response.ok
  assert response.result["report"]["num_steps"] == 20
  assert Path(response.result["trace_path"]).exists()
  assert Path(response.result["image_path"]).exists()


def _tagged_sampler(fake_sim):
  """sample_state whose values encode which env was sampled (env*100 + index).

  Any cross-env interleaving in a recorded trace is visible in the data.
  """
  samples = {"n": 0}

  def tagged_sample(env_id: int = 0):
    samples["n"] += 1
    value = float(100 * env_id + samples["n"])
    return {
        "joint_pos": np.array([value, value], dtype=np.float32),
        "joint_vel": np.array([0.5, -0.5], dtype=np.float32),
        "action": np.array([0.1, -0.1], dtype=np.float32),
        "command": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "base_lin_vel": np.array([0.9, 0.0, 0.0], dtype=np.float32),
        "base_ang_vel": np.array([0.0, 0.0, 0.05], dtype=np.float32),
    }

  fake_sim.sample_state = tagged_sample


def test_concurrent_diagnose_jobs_both_succeed_with_env_pure_traces(lab, fake_sim):
  """Each trace job owns its recorder, so concurrent diagnoses never interleave."""
  from rlmcp.core.telemetry.trace import load_npz

  _tagged_sampler(fake_sim)
  client = _client(lab)
  first = client.submit("diagnose", seconds=0.4, env_id=3)
  second = client.submit("diagnose", seconds=0.4, env_id=7)
  lab.service(iteration=1)  # Both scheduled: concurrency is legal now.

  for _ in range(20):
    lab.on_step()
  lab.service(iteration=2)

  resp3 = client.poll(first.req_id)
  resp7 = client.poll(second.req_id)
  assert resp3.ok and resp3.result["env_id"] == 3
  assert resp7.ok and resp7.result["env_id"] == 7
  assert resp3.result["report"]["num_steps"] == 20
  assert resp7.result["report"]["num_steps"] == 20

  # Each saved trace holds only its own env's samples -- never a stray one
  # from the other job, which is what the old shared recorder would have done.
  vals3 = load_npz(resp3.result["trace_path"])["data"]["joint_pos"][:, 0]
  vals7 = load_npz(resp7.result["trace_path"])["data"]["joint_pos"][:, 0]
  assert vals3.min() >= 300 and vals3.max() < 400
  assert vals7.min() >= 700 and vals7.max() < 800


def test_duplicate_diagnose_on_the_same_env_is_legal(lab, fake_sim):
  client = _client(lab)
  first = client.submit("diagnose", seconds=0.4, env_id=2)
  second = client.submit("diagnose", seconds=0.4, env_id=2)
  lab.service(iteration=1)
  for _ in range(20):
    lab.on_step()
  lab.service(iteration=2)

  for request in (first, second):
    response = client.poll(request.req_id)
    assert response.ok and response.result["env_id"] == 2
    assert response.result["report"]["num_steps"] == 20


def test_deferred_jobs_beyond_the_cap_are_refused_truthfully(lab):
  from rlmcp.core.controller import MAX_CONCURRENT_JOBS

  client = _client(lab)
  admitted = [
      client.submit("record_video", seconds=0.2)
      for _ in range(MAX_CONCURRENT_JOBS)
  ]
  over = client.submit("diagnose", seconds=0.4)
  lab.service(iteration=1)

  refusal = client.poll(over.req_id)
  assert not refusal.ok
  assert "Deferred-job limit reached" in refusal.error
  assert "cancel_job" in refusal.error and "video" in refusal.error

  # The admitted jobs are unaffected and all complete.
  for _ in range(10):
    lab.on_step()
  lab.service(iteration=2)
  for request in admitted:
    assert client.poll(request.req_id).ok


def test_a_starved_job_times_out_with_a_truthful_error(lab):
  """A deferred job whose steps never come must answer, not hang its client."""
  client = _client(lab)
  request = client.submit("diagnose", seconds=0.4)
  lab.service(iteration=1)
  assert client.poll(request.req_id) is None  # In flight, waiting on steps.

  lab._jobs[0].started_at -= 999.0  # The wall clock has long passed the budget.
  lab.service(iteration=2)

  response = client.poll(request.req_id)
  assert not response.ok
  assert "Timed out after 90s" in response.error
  assert "20 steps still to collect" in response.error
  assert lab._jobs == []  # Failed and cleared, not lingering.
  events = [e for e in _client(lab).events() if e["kind"] == "job_timeout"]
  assert events and events[-1]["req_id"] == request.req_id


def test_cancel_job_answers_the_requester_and_the_canceller(lab):
  client = _client(lab)
  video = client.submit("record_video", seconds=0.2)
  lab.service(iteration=1)

  cancelled = _run(lab, "cancel_job", req_id=video.req_id, reason="wrong env")
  assert cancelled.ok
  assert cancelled.result["cancelled"] is True
  assert cancelled.result["req_id"] == video.req_id

  original = client.poll(video.req_id)
  assert not original.ok and "Cancelled" in original.error
  assert "wrong env" in original.error
  assert lab._jobs == []

  unknown = _run(lab, "cancel_job", req_id="nope")
  assert not unknown.ok and "No in-flight job" in unknown.error


def test_new_deferred_commands_are_refused_while_paused(lab):
  """A deferred job started while paused would never progress; refuse it."""
  client = _client(lab)
  client.submit("pause")
  thread = threading.Thread(target=lambda: lab.service(iteration=1), daemon=True)
  thread.start()
  deadline = time.time() + 5.0
  while not lab.paused and time.time() < deadline:
    time.sleep(0.05)
  assert lab.paused

  try:
    for cmd in ("record_video", "diagnose", "record_trace"):
      refusal = client.call(cmd, seconds=0.4, timeout=5.0)
      assert not refusal.ok
      assert "paused" in refusal.error
      assert "Resume training" in refusal.error
  finally:
    assert client.call("resume", timeout=5.0).ok
    thread.join(timeout=5.0)


def test_step_once_feeds_an_in_flight_job_queued_before_the_pause(lab):
  """Jobs already in flight survive a pause and progress on step_once steps."""
  client = _client(lab)
  request = client.submit("diagnose", seconds=0.4)
  lab.service(iteration=1)  # Scheduled while running.

  lab.paused = True
  lab.step_once_requested = True  # What cmd_step_once sets.
  lab.service(iteration=2)  # Pause loop releases for exactly one iteration.

  for _ in range(20):
    lab.on_step()  # The stepped iteration's rollout feeds the job.
  lab.paused = False
  lab.service(iteration=3)

  response = client.poll(request.req_id)
  assert response.ok and response.result["report"]["num_steps"] == 20


def test_video_records_alongside_an_in_flight_trace(lab):
  """Video jobs keep per-job frame lists, so they never contend for the recorder."""
  client = _client(lab)
  trace_req = client.submit("record_trace", seconds=0.4)
  lab.service(iteration=1)
  video_req = client.submit("record_video", seconds=0.2)
  lab.service(iteration=1)
  for _ in range(20):
    lab.on_step()
  lab.service(iteration=2)

  trace_resp = client.poll(trace_req.req_id)
  video_resp = client.poll(video_req.req_id)
  assert trace_resp.ok and trace_resp.result["report"]["num_steps"] == 20
  assert video_resp.ok and video_resp.result["num_frames"] == 10


def test_video_encodes_captured_frames(lab):
  client = _client(lab)
  request = client.submit("record_video", seconds=0.2)
  lab.service(iteration=1)
  for _ in range(10):
    lab.on_step()
  lab.service(iteration=2)

  response = client.poll(request.req_id)
  assert response.ok and response.result["num_frames"] == 10
  assert Path(response.result["video_path"]).exists()


def test_metrics_flow_into_telemetry_and_the_session_file(lab):
  lab.service(iteration=1, metrics={"Episode_Reward/track": 0.25})
  lab.service(iteration=2, metrics={"Episode_Reward/track": 0.5})

  response = _run(lab, "get_metrics", names=["Episode_Reward/track"], last_n=5)
  assert response.result["metrics"]["Episode_Reward/track"][-1] == [2, 0.5]
  assert response.result["summary"]["Episode_Reward/track"]["trend"] == "up"

  rows = [r for r in _client(lab).metrics() if "Episode_Reward/track" in r]
  assert rows[-1]["Episode_Reward/track"] == 0.5
  # Derived from the runner's mean episode length over max_episode_length.
  assert rows[-1][METRIC_EPISODE_LENGTH_FRAC] == 0.8


def test_a_stage_reports_a_command_it_cannot_run(lab, fake_sim):
  """A stage naming a command this environment lacks must not fail silently."""
  lab.curriculum.stages[1].apply.append(Action("set_objects", {"names": ["cube"]}))
  lab.service(iteration=0)
  _run(lab, "curriculum_advance")

  events = [e for e in _client(lab).events() if e["kind"] == "curriculum_stage"]
  errors = events[-1]["applied"]["action_errors"]
  assert "set_objects" in errors
  assert "Unknown command" in errors["set_objects"]
  # The stage's other actions still applied.
  assert fake_sim.enabled == ["flat", "random_rough"]


def test_agent_commands_survive_the_curriculum_in_the_same_tick(lab, fake_sim):
  """A stage applied this iteration must not clobber a command issued with it."""
  client = _client(lab)
  client.submit("set_terrain", terrains=["flat", "random_rough"], max_level=5)

  lab.service(iteration=0)  # First tick: applies stage 0_flat *and* the command.

  assert fake_sim.enabled == ["flat", "random_rough"]
  assert fake_sim.level_ceiling == 5


def test_plot_metrics_writes_a_png(lab):
  lab.service(iteration=1, metrics={"Train/mean_reward": 1.0})
  response = _run(lab, "plot_metrics", names=["Train/mean_reward"])

  assert response.ok
  path = Path(response.result["image_path"])
  assert path.exists() and path.read_bytes()[:4] == b"\x89PNG"


def test_checkpoint_round_trip_restores_parameters_and_stage(lab, fake_sim, fake_runner):
  lab.service(iteration=0)
  _run(lab, "curriculum_advance")
  _run(lab, "set_parameter", key="reward.track_linear_velocity.weight", value=7.0)
  saved = _run(lab, "save_checkpoint", tag="good", note="before the experiment")
  assert saved.ok

  _run(lab, "set_parameter", key="reward.track_linear_velocity.weight", value=0.0)
  _run(lab, "curriculum_goto", stage="0_flat")

  restored = _run(lab, "load_checkpoint", path=saved.result["path"])

  assert restored.ok
  assert fake_sim.get_parameter("reward.track_linear_velocity.weight") == 7.0
  assert lab.curriculum.current.name == "1_rough"
  assert fake_runner.loaded == [saved.result["path"]]


def test_checkpoint_preserves_extension_state_despite_runner_env_state_swap(
    lab, fake_sim, fake_runner):
  """mjlab's runner replaces infos["env_state"] wholesale on save.

  Extension state must ride somewhere the runner preserves, survive the round
  trip, and the load response must say what was actually restored.
  """
  lab.service(iteration=0)
  _run(lab, "set_terrain", terrains=["flat", "random_rough"], max_level=3)
  saved = _run(lab, "save_checkpoint", tag="ext")
  assert saved.ok
  # The fake, like the real runner, kept only its own env_state payload.
  assert fake_runner.checkpoint_infos["env_state"] == {"common_step_counter": 0}

  _run(lab, "set_terrain", terrains=["flat"], max_level=6)
  restored = _run(lab, "load_checkpoint", path=saved.result["path"])

  assert restored.ok
  assert restored.result["env_state"] is True
  assert restored.result["extensions_restored"] == 1
  assert fake_sim.enabled == ["flat", "random_rough"]
  assert fake_sim.level_ceiling == 3


def test_legacy_checkpoint_extensions_under_env_state_still_restore(
    lab, fake_sim, fake_runner):
  """Checkpoints written before extensions moved out of env_state must load."""
  lab.service(iteration=0)
  fake_runner.checkpoint_infos = {
      "env_state": {
          "step": 42,
          "extensions": {
              "terrain": {"enabled": ["flat", "random_rough"], "ceiling": 4}
          },
      },
      "rlmcp": {"iteration": 3, "note": "", "parameters": {}, "curriculum": None},
  }

  restored = _run(lab, "load_checkpoint", path="legacy")

  assert restored.ok
  assert restored.result["env_state"] is True
  assert restored.result["extensions_restored"] == 1
  assert fake_sim.step_count == 42
  assert fake_sim.enabled == ["flat", "random_rough"]
  assert fake_sim.level_ceiling == 4


def test_a_failed_extension_restore_is_not_counted_as_restored(
    lab, fake_sim, fake_terrain, monkeypatch):
  """extensions_restored counts what actually restored, not what was found."""
  lab.service(iteration=0)
  _run(lab, "set_terrain", terrains=["flat", "random_rough"], max_level=3)
  saved = _run(lab, "save_checkpoint", tag="brittle")
  assert saved.ok

  def broken_restore(state):
    raise RuntimeError("terrain grid changed shape")

  monkeypatch.setattr(fake_terrain, "restore", broken_restore)
  restored = _run(lab, "load_checkpoint", path=saved.result["path"])

  assert restored.ok  # Weights still loaded; the response stays truthful.
  assert restored.result["extensions_restored"] == 0
  errors = [e for e in _client(lab).events() if e["kind"] == "extension_error"]
  assert errors and errors[-1]["extension"] == "terrain"
  assert errors[-1]["hook"] == "restore"
  assert "terrain grid changed shape" in errors[-1]["error"]


def test_rollback_restores_hand_tuned_values_over_stage_entry_values(lab, fake_sim):
  """Restoring a checkpoint must not let the stage's entry parameters win."""
  lab.service(iteration=0)  # Stage 0_flat sets reward.action_rate_l2.weight=-0.05.
  _run(lab, "set_parameter", key="reward.action_rate_l2.weight", value=-0.5,
       rationale="hand-tuned after stage entry")
  saved = _run(lab, "save_checkpoint", tag="tuned")
  assert saved.ok

  _run(lab, "set_parameter", key="reward.action_rate_l2.weight", value=-0.01)
  restored = _run(lab, "load_checkpoint", path=saved.result["path"])

  assert restored.ok
  assert restored.result["curriculum_stage"] == "0_flat"
  # The checkpointed value wins over the stage's entry value.
  assert fake_sim.get_parameter("reward.action_rate_l2.weight") == -0.5


def test_pause_blocks_training_but_keeps_serving_commands(lab):
  client = _client(lab)
  client.submit("pause")

  # Once paused, service() does not return -- that is how it holds the training
  # loop still -- so it has to be driven from a thread.
  returned = threading.Event()

  def drive():
    lab.service(iteration=1)
    returned.set()

  thread = threading.Thread(target=drive, daemon=True)
  thread.start()

  deadline = time.time() + 5.0
  while not lab.paused and time.time() < deadline:
    time.sleep(0.05)
  assert lab.paused
  assert not returned.is_set()

  # Commands still get answered while the loop is held.
  status = client.call("status", timeout=5.0)
  assert status.ok and status.result["paused"] is True

  assert client.call("resume", timeout=5.0).ok
  thread.join(timeout=5.0)
  assert returned.is_set() and not lab.paused


def test_step_once_releases_a_paused_loop_for_one_iteration(lab):
  client = _client(lab)
  client.submit("pause")
  thread = threading.Thread(target=lambda: lab.service(iteration=1), daemon=True)
  thread.start()

  deadline = time.time() + 5.0
  while not lab.paused and time.time() < deadline:
    time.sleep(0.05)

  assert client.call("step_once", timeout=5.0).ok
  thread.join(timeout=5.0)
  assert not thread.is_alive()
  assert lab.paused  # Still paused for the next iteration.
  lab.paused = False  # Let the fixture tear down cleanly.


def test_stop_training_is_visible_to_the_training_loop(lab, fake_runner):
  response = _run(lab, "stop_training", reason="found a bug")

  assert response.ok
  assert lab.should_stop()
  assert lab.stop_reason == "found a bug"
  assert fake_runner.stop_called


def test_notes_land_in_the_event_log(lab):
  _run(lab, "note", text="switching to stairs next")
  notes = [e for e in _client(lab).events() if e["kind"] == "note"]
  assert notes[-1]["text"] == "switching to stairs next"


# Resetting episodes, which is not resetting parameter values.


def test_reset_envs_restarts_every_environment_by_default(lab, fake_sim):
  """No narrowing means all of them. Restarting env 0 alone and calling it a
  reset would be a lie, which is why the resolver defaults to None rather than
  to the first environment the way the screenshot one does."""
  response = _run(lab, "reset_envs")

  assert response.ok
  assert response.result["scope"] == "all"
  assert response.result["num_reset"] == 12
  assert fake_sim.resets == [list(range(12))]


def test_reset_envs_takes_explicit_ids(lab, fake_sim):
  response = _run(lab, "reset_envs", env_ids=[2, 5])

  assert response.ok and response.result["env_ids"] == [2, 5]
  assert fake_sim.resets == [[2, 5]]


def test_reset_envs_narrows_by_the_extensions_own_vocabulary(lab, fake_sim):
  """`--where` is the same query `shot` and `diagnose` take, so restarting only
  the environments on one part of the task costs the core no new concepts."""
  _run(lab, "set_terrain", terrains=["flat", "random_rough"])
  expected = fake_sim.env_ids_on(terrain="random_rough")

  response = _run(lab, "reset_envs", where={"terrain": "random_rough"})

  assert response.ok
  assert response.result["scope"] == "selection"
  assert fake_sim.resets == [expected]
  assert 0 < len(expected) < 12  # A real subset, not everything.


def test_reset_envs_refuses_a_description_no_extension_understands(lab, fake_sim):
  response = _run(lab, "reset_envs", where={"holding": True})

  assert not response.ok and "terrain" in response.error
  assert fake_sim.resets == []


def test_reset_envs_refuses_a_description_that_matches_nothing(lab, fake_sim):
  response = _run(lab, "reset_envs", where={"level": 5})

  assert not response.ok
  assert fake_sim.resets == []


def test_reset_envs_refuses_an_empty_selection_rather_than_resetting_all(lab, fake_sim):
  """An empty list is a caller whose filter produced nothing, not a caller who
  meant 'everything'. Guessing the second would restart a whole run."""
  response = _run(lab, "reset_envs", env_ids=[])

  assert not response.ok
  assert fake_sim.resets == []


def test_reset_envs_says_so_when_the_backend_has_no_reset(tmp_path, fake_runner):
  """The graceful degradation the adapter contract asks for: a backend that
  cannot restart episodes says which capability is missing, and nothing else
  in the run is disturbed."""

  class NoResetSim(FakeSimAdapter):
    # Back to the contract's default, which raises NotSupported -- a backend
    # that never implemented the capability at all.
    reset_envs = SimAdapter.reset_envs

  sim = NoResetSim()
  controller = RlMcp(sim_adapter=sim, runner_adapter=fake_runner,
                     session_dir=tmp_path / "no-reset")
  try:
    response = _run(controller, "reset_envs")
  finally:
    controller.close()

  assert not response.ok
  assert "reset_envs" in response.error and "Nothing was reset" in response.error


def test_reset_envs_is_logged_with_its_rationale(lab):
  _run(lab, "reset_envs", env_ids=[1], rationale="cleared after the weight edit")

  events = [e for e in _client(lab).events() if e["kind"] == "reset_envs"]
  assert events[-1]["rationale"] == "cleared after the weight edit"
  assert events[-1]["env_ids"] == [1]


def test_resetting_episodes_and_resetting_parameters_are_different_verbs(lab, fake_sim):
  """The two names sit next to each other in the command list and mean
  unrelated things; neither may quietly do the other's job."""
  _run(lab, "set_parameter", key="reward.action_rate_l2.weight", value=-0.4,
       rationale="test")

  _run(lab, "reset_envs")
  assert fake_sim.get_parameter("reward.action_rate_l2.weight") == -0.4  # untouched

  fake_sim.resets.clear()
  _run(lab, "reset_parameters")
  assert fake_sim.resets == []  # no episode was restarted


def test_reset_envs_is_offered_to_agents_alongside_the_other_commands(lab):
  assert "reset_envs" in _run(lab, "help").result["commands"]
