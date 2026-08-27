"""The extension mechanism, and the terrain extension built on it.

The point of these tests is that the core carries no task vocabulary: an
environment gains a capability by registering an extension, and loses it by not
having one, without anything else changing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from rlmcp.core.controller import DeferredJob, RlMcp
from rlmcp.core.extensions import Extension, ExtensionRegistry
from rlmcp.extensions import registry as ext_registry
from rlmcp.session import Session


class Objects(Extension):
  """A capability from a different world entirely: manipulation object sets."""

  name = "objects"

  def __init__(self, available: bool = True):
    super().__init__(env=None)
    self._available = available
    self.object_set = ["cube"]
    self.calls: List[str] = []

  def available(self) -> bool:
    return self._available

  def commands(self):
    return {"set_objects": self.cmd_set_objects}

  def metrics(self) -> Dict[str, float]:
    return {"rlmcp/object_variety": float(len(self.object_set))}

  def select_envs(self, **criteria):
    obj = criteria.pop("object", None)
    if criteria or obj is None:
      return None
    return [7] if obj in self.object_set else []

  def describe(self) -> Dict[str, Any]:
    return {"object_set": list(self.object_set)}

  def snapshot(self) -> Dict[str, Any]:
    return {"object_set": list(self.object_set)}

  def restore(self, state: Dict[str, Any]) -> None:
    self.object_set = list(state["object_set"])

  def cmd_set_objects(self, names: List[str]) -> Dict[str, Any]:
    """Choose which objects appear in the scene."""
    self.calls.append("set_objects")
    self.object_set = list(names)
    return {"object_set": self.object_set}


class Noisy(Extension):
  """An extension that misbehaves, to check it cannot take a run down."""

  name = "noisy"

  def available(self) -> bool:
    return True

  def metrics(self):
    raise RuntimeError("sensor exploded")

  def describe(self):
    raise RuntimeError("still exploded")

  def snapshot(self):
    raise RuntimeError("and again")


def test_unavailable_extensions_are_not_registered():
  registry = ExtensionRegistry([Objects(available=False)])
  assert len(registry) == 0
  assert registry.names() == []
  assert registry.commands() == {}


def test_commands_and_metrics_are_aggregated():
  objects = Objects()
  registry = ExtensionRegistry([objects])

  assert sorted(registry.commands()) == ["set_objects"]
  assert registry.metrics() == {"rlmcp/object_variety": 1.0}

  registry.commands()["set_objects"](names=["cube", "sphere"])
  assert objects.object_set == ["cube", "sphere"]
  assert registry.metrics()["rlmcp/object_variety"] == 2.0


def test_selection_asks_each_extension_until_one_understands():
  registry = ExtensionRegistry([Objects()])

  assert registry.select_envs(object="cube") == [7]
  assert registry.select_envs(object="banana") == []  # Understood, no match.
  assert registry.select_envs(terrain="stairs") is None  # Not its vocabulary.


def test_a_broken_extension_cannot_take_the_run_down():
  registry = ExtensionRegistry([Noisy(env=None), Objects()])

  # The healthy extension still contributes; the broken one is skipped.
  assert registry.metrics() == {"rlmcp/object_variety": 1.0}
  assert registry.describe() == {"objects": {"object_set": ["cube"]}}
  assert registry.snapshot() == {"objects": {"object_set": ["cube"]}}


def test_snapshot_and_restore_round_trip():
  objects = Objects()
  registry = ExtensionRegistry([objects])
  objects.object_set = ["mug", "bowl"]

  saved = registry.snapshot()
  objects.object_set = ["cube"]
  registry.restore(saved)

  assert objects.object_set == ["mug", "bowl"]


def test_restore_reports_success_per_extension():
  """A payload whose restore raised must not be reported as restored."""

  class Brittle(Extension):
    name = "brittle"

    def available(self):
      return True

    def restore(self, state):
      raise RuntimeError("state from an incompatible run")

  registry = ExtensionRegistry([Objects(), Brittle(env=None)])
  results = registry.restore({
      "objects": {"object_set": ["mug"]},
      "brittle": {"anything": 1},
      "stranger": {"not": "registered"},
  })

  # One entry per registered extension that had a payload; the stranger's
  # payload has no extension here and is simply not part of the answer.
  assert results == {"objects": True, "brittle": False}


def test_hook_failures_are_reported_once_per_extension_and_hook():
  reports: List[tuple] = []
  registry = ExtensionRegistry([Noisy(env=None), Objects()])
  registry.set_error_sink(lambda name, hook, msg: reports.append((name, hook, msg)))

  registry.metrics()
  registry.metrics()  # Same failure again: not re-reported.
  registry.describe()
  registry.describe()

  pairs = [(name, hook) for name, hook, _ in reports]
  assert pairs.count(("noisy", "metrics")) == 1
  assert pairs.count(("noisy", "describe")) == 1
  assert all(name != "objects" for name, _ in pairs)
  assert any("sensor exploded" in msg for _, hook, msg in reports if hook == "metrics")


def test_command_clashes_are_first_wins_and_reported():
  """Two extensions claiming one verb: the first keeps it, the clash is named."""

  class Rival(Extension):
    name = "rival"

    def available(self):
      return True

    def commands(self):
      return {"set_objects": self.cmd_set_objects}

    def cmd_set_objects(self, names):
      """Marker that must never win the clash."""
      return {"rival": True}

  objects = Objects()
  reports: List[tuple] = []
  registry = ExtensionRegistry([objects, Rival(env=None)])
  registry.set_error_sink(lambda name, hook, msg: reports.append((name, hook, msg)))

  commands = registry.commands()
  result = commands["set_objects"](names=["mug"])

  assert objects.object_set == ["mug"]  # The first registration won.
  assert result != {"rival": True}
  assert any(
      hook == "commands:set_objects" and "objects" in msg and name == "rival"
      for name, hook, msg in reports
  )


def test_on_iteration_and_close_are_aggregated_with_failures_contained():
  calls: List[Any] = []

  class Lifecycle(Extension):
    name = "lifecycle"

    def available(self):
      return True

    def on_iteration(self, iteration, metrics):
      calls.append(("iter", iteration, dict(metrics)))

    def close(self):
      calls.append(("close",))

  class Dying(Extension):
    name = "dying"

    def available(self):
      return True

    def on_iteration(self, iteration, metrics):
      raise RuntimeError("mid-iteration crash")

  reports: List[tuple] = []
  registry = ExtensionRegistry([Dying(env=None), Lifecycle(env=None)])
  registry.set_error_sink(lambda name, hook, msg: reports.append((name, hook)))

  registry.on_iteration(3, {"Train/mean_reward": 1.0})
  registry.on_iteration(4, {})
  registry.close()

  assert ("iter", 3, {"Train/mean_reward": 1.0}) in calls
  assert ("iter", 4, {}) in calls  # The healthy extension kept being called.
  assert ("close",) in calls
  assert reports.count(("dying", "on_iteration")) == 1


# Extensions inside a live controller: the context, and deferred commands.


class WatchJob(DeferredJob):
  """The harness's signature move: watch the robot for N steps and report."""

  kind = "watch"

  def __init__(self, env_id: int, steps_needed: int, **kwargs):
    super().__init__(env_id=env_id, steps_needed=steps_needed, **kwargs)
    self.samples: List[Dict[str, Any]] = []

  def feed(self, lab):
    sample = lab.sim.sample_state(self.env_id)
    if sample:
      self.samples.append(sample)
    self.steps_remaining -= 1

  def complete(self, lab):
    return {
        "watched_steps": len(self.samples),
        "env_id": self.env_id,
        "iteration": lab.iteration,
    }


class Watcher(Extension):
  """An extension whose command defers, exactly like the built-in video/trace."""

  name = "watcher"

  def __init__(self):
    super().__init__(env=None)

  def available(self):
    return True

  def commands(self):
    return {"watch": self.cmd_watch}

  def cmd_watch(self, steps: int = 5):
    """Watch the robot for N steps and report."""
    return WatchJob(env_id=0, steps_needed=steps)


@pytest.fixture
def watcher_lab(tmp_path, fake_sim, fake_runner):
  watcher = Watcher()
  lab = RlMcp(
      sim_adapter=fake_sim,
      runner_adapter=fake_runner,
      session_dir=tmp_path / "watch",
      extensions=[watcher],
      video_every=0,   # this asserts on the job queue; see test_progress_video
  )
  yield lab, watcher
  lab.paused = False
  lab.close()


def test_an_extension_command_can_defer_like_the_built_ins(watcher_lab):
  """Submit via the run path, feed over steps, complete, response written."""
  lab, _ = watcher_lab
  client = Session.open(lab.session.dir)
  request = client.submit("watch", steps=6)
  lab.service(iteration=1)

  assert client.poll(request.req_id) is None  # Deferred: needs steps.
  pending = client.status()["pending_jobs"]
  assert [j["kind"] for j in pending] == ["watch"]

  for _ in range(6):
    lab.on_step()
  lab.service(iteration=2)

  response = client.poll(request.req_id)
  assert response.ok
  assert response.result == {"watched_steps": 6, "env_id": 0, "iteration": 2}


def test_extension_deferred_commands_obey_the_pause_refusal(watcher_lab):
  """The scheduling rules are the controller's, not per-command: paused refuses."""
  lab, _ = watcher_lab
  client = Session.open(lab.session.dir)
  lab.paused = True
  request = client.submit("watch", steps=3)
  lab.step_once_requested = True  # Let the pause loop release immediately.
  lab.service(iteration=1)

  refusal = client.poll(request.req_id)
  assert refusal is not None and not refusal.ok
  assert "'watch'" in refusal.error and "paused" in refusal.error


def test_bind_hands_extensions_the_controller_context(watcher_lab):
  lab, watcher = watcher_lab
  context = watcher.context
  assert context is not None

  # The artifact writer writes into the session's artifact directory.
  path = Path(context.write_artifact("watch_report", ".txt", b"all quiet"))
  assert path.read_bytes() == b"all quiet"
  assert path.parent == lab.session.artifacts

  # Telemetry is readable; events land in the session log.
  lab.service(iteration=3, metrics={"Episode_Reward/watch": 2.5})
  assert context.telemetry.get_latest_metrics()["Episode_Reward/watch"] == 2.5
  context.append_event("watch_note", {"text": "robot looks fine"})
  client = Session.open(lab.session.dir)
  notes = [e for e in client.events() if e["kind"] == "watch_note"]
  assert notes and notes[-1]["text"] == "robot looks fine"


def test_add_extension_binds_the_context_as_well(tmp_path, fake_sim, fake_runner):
  lab = RlMcp(sim_adapter=fake_sim, runner_adapter=fake_runner,
               session_dir=tmp_path / "late")
  try:
    late = Objects()
    assert lab.add_extension(late)
    assert late.context is not None
    assert "set_objects" in lab._handlers
  finally:
    lab.close()


def test_context_submitted_job_reports_through_the_event_log(watcher_lab):
  """A job with no request waiting still runs, and its outcome is on record."""
  lab, watcher = watcher_lab
  described = watcher.context.submit_job(WatchJob(env_id=0, steps_needed=3))
  assert described["kind"] == "watch"
  assert [j["kind"] for j in watcher.context.pending_jobs()] == ["watch"]

  for _ in range(3):
    lab.on_step()
  lab.service(iteration=1)

  client = Session.open(lab.session.dir)
  done = [e for e in client.events() if e["kind"] == "job_complete"]
  assert done and done[-1]["job_kind"] == "watch"
  assert done[-1]["result"]["watched_steps"] == 3


# The registry: how a capability is declared and found.


@pytest.fixture
def clean_registry():
  """Isolate the global registry so tests cannot leak into each other."""
  saved = dict(ext_registry._REGISTRY)
  yield
  ext_registry._REGISTRY.clear()
  ext_registry._REGISTRY.update(saved)


class Widget(Extension):
  """A capability that needs a plot sink."""

  name = "widget"

  def __init__(self, env, plot_sink=None):
    super().__init__(env)
    self.plot_sink = plot_sink

  def available(self) -> bool:
    return getattr(self.env, "has_widget", False)


def test_registering_makes_a_capability_discoverable(clean_registry):
  ext_registry.register(Widget)

  assert "widget" in ext_registry.registered()
  assert any(e["name"] == "widget" for e in ext_registry.catalog())


def test_registering_requires_a_name(clean_registry):
  class Nameless(Extension):
    pass

  with pytest.raises(ValueError, match="non-empty 'name'"):
    ext_registry.register(Nameless)


def test_two_capabilities_cannot_share_a_name(clean_registry):
  ext_registry.register(Widget)

  class Impostor(Extension):
    name = "widget"

  with pytest.raises(ValueError, match="already registered"):
    ext_registry.register(Impostor)


def test_registering_the_same_class_twice_is_fine(clean_registry):
  ext_registry.register(Widget)
  ext_registry.register(Widget)  # e.g. a module imported twice.
  assert ext_registry.registered()["widget"] is Widget


def test_discover_binds_only_what_the_environment_supports(clean_registry):
  ext_registry.register(Widget)

  class Env:
    has_widget = True

  class Barren:
    has_widget = False

  assert [e.name for e in ext_registry.discover(Env())] == ["widget"]
  assert ext_registry.discover(Barren()) == []


def test_discover_passes_the_plot_sink_when_accepted(clean_registry):
  ext_registry.register(Widget)

  class Env:
    has_widget = True

  sink = object()
  found = ext_registry.discover(Env(), plot_sink=sink)
  assert found[0].plot_sink is sink


def test_an_extension_that_takes_no_sink_still_builds(clean_registry):
  class Simple(Extension):
    name = "simple"

    def available(self):
      return True

  ext_registry.register(Simple)
  assert [e.name for e in ext_registry.discover(object(), plot_sink=object())] == [
      "simple"
  ]


def test_a_capability_that_explodes_on_construction_is_skipped(clean_registry):
  class Exploding(Extension):
    name = "exploding"

    def __init__(self, env):
      raise RuntimeError("bad hardware")

  class Fine(Extension):
    name = "fine"

    def available(self):
      return True

  ext_registry.register(Exploding)
  ext_registry.register(Fine)

  assert [e.name for e in ext_registry.discover(object())] == ["fine"]


def test_include_and_exclude_filter_discovery(clean_registry):
  for label in ("a", "b"):
    ext_registry.register(
        type(f"Cap{label}", (Extension,), {"name": label, "available": lambda self: True})
    )

  assert sorted(e.name for e in ext_registry.discover(object())) == ["a", "b"]
  assert [e.name for e in ext_registry.discover(object(), include=["a"])] == ["a"]
  assert [e.name for e in ext_registry.discover(object(), exclude=["a"])] == ["b"]


def test_terrain_is_registered_by_importing_the_package():
  """Built-ins self-register on import; nobody maintains a list."""
  import rlmcp.extensions  # noqa: F401

  assert "terrain" in ext_registry.registered()


# The terrain extension's plan builder: terrain-specific, and living outside
# the core because of it.


ROUGH = [
    "flat",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "random_rough",
    "wave_terrain",
]


@pytest.fixture
def terrain_module():
  return pytest.importorskip("rlmcp.extensions.terrain")


def test_terrain_groups_are_ordered_easy_to_hard(terrain_module):
  grouped = terrain_module.group_terrains(ROUGH)
  assert [label for label, _ in grouped] == ["flat", "rough", "slopes", "stairs"]


def test_unknown_terrains_are_kept_not_dropped(terrain_module):
  grouped = terrain_module.group_terrains(["flat", "mystery_terrain"])
  assert ("other", ["mystery_terrain"]) in grouped


def test_terrain_plan_is_expressed_as_ordinary_stages(terrain_module):
  """The plan is terrain-specific; the objects it produces are not."""
  plan = terrain_module.build_terrain_plan(ROUGH, num_levels=10)

  first, last = plan.stages[0], plan.stages[-1]
  assert first.apply[0].cmd == "set_terrain"
  assert first.apply[0].args["terrains"] == ["flat"]
  assert first.apply[0].args["max_level"] < 10
  assert set(last.apply[0].args["terrains"]) == set(ROUGH)
  assert last.apply[0].args["max_level"] == 10


def test_each_terrain_stage_keeps_what_came_before(terrain_module):
  plan = terrain_module.build_terrain_plan(ROUGH, num_levels=10)
  for earlier, later in zip(plan.stages, plan.stages[1:]):
    assert set(earlier.apply[0].args["terrains"]).issubset(
        set(later.apply[0].args["terrains"])
    )
    assert later.apply[0].args["max_level"] >= earlier.apply[0].args["max_level"]


def test_only_the_final_terrain_stage_has_no_exit_condition(terrain_module):
  plan = terrain_module.build_terrain_plan(ROUGH, num_levels=10)
  assert all(s.promote_when for s in plan.stages[:-1])
  assert plan.stages[-1].promote_when == []


# ── the `where` vocabulary an extension advertises ───────────────────────
def test_an_extension_advertises_what_select_envs_will_accept(fake_sim, fake_terrain):
  """The point of publishing it: a caller can offer the criteria without
  knowing the capability exists."""
  registry = ExtensionRegistry([fake_terrain])
  published = registry.selectors()

  assert set(published) == {"terrain", "level"}
  assert published["terrain"]["values"] == ["flat"]
  assert published["terrain"]["extension"] == "terrain", \
      "a criterion has to say who owns it"

  # And every advertised value actually resolves, which is what makes it a
  # menu rather than a guess.
  for terrain in published["terrain"]["values"]:
    assert registry.select_envs(terrain=terrain) != []


def test_an_extension_that_advertises_nothing_is_normal(fake_sim):
  """Publishing a vocabulary is optional; understanding one is not the same
  thing. The default extension offers no menu and still answers."""

  class Quiet(Extension):
    name = "quiet"

    def available(self) -> bool:
      return True

    def select_envs(self, **criteria):
      return [0] if criteria.get("holding") else None

  registry = ExtensionRegistry([Quiet(env=None)])
  assert registry.selectors() == {}
  assert registry.select_envs(holding=True) == [0]


def test_the_first_extension_to_claim_a_criterion_keeps_it(fake_sim, fake_terrain):
  """Same rule as select_envs, and it has to be: a menu that offers a key a
  *different* extension answers is a menu that lies."""

  class AlsoTerrain(Extension):
    name = "other"

    def available(self) -> bool:
      return True

    def selectors(self):
      return {"terrain": {"label": "somewhere else", "values": ["mars"]}}

  registry = ExtensionRegistry([fake_terrain, AlsoTerrain(env=None)])
  assert registry.selectors()["terrain"]["extension"] == "terrain"


def test_a_broken_selectors_hook_is_reported_and_skipped(fake_sim, fake_terrain):
  """A capability that cannot describe itself must not empty the menu."""

  class Broken(Extension):
    name = "broken"

    def available(self) -> bool:
      return True

    def selectors(self):
      raise RuntimeError("no idea")

  problems = []
  registry = ExtensionRegistry([Broken(env=None), fake_terrain])
  registry.set_error_sink(lambda name, hook, error: problems.append((name, hook)))

  assert set(registry.selectors()) == {"terrain", "level"}
  assert problems == [("broken", "selectors")]
