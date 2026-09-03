"""Pairing a checkpoint with the environment it trained under.

The test that matters here is not "the exporter produced some text": it is that
the text **runs**. Every case below writes the export to disk, puts it on
``sys.path`` and imports it, then checks the config objects that come out —
because an export that reads plausibly and raises on import is worth nothing,
and that is the failure mode this whole feature is exposed to.

The fakes mirror the real managers' shapes (``active_terms`` /
``get_term_cfg``, an observation manager keyed by group, an action manager of
term instances) and use dataclasses defined in *this module*, so the generated
imports name a real importable class and the round trip is genuine rather than
stubbed.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
from collections.abc import Callable
from typing import Any

import pytest
import torch

from rlmcp import env_export
from rlmcp.adapters.manager_based.term_capture import (
  capture_env_terms,
  encode_value,
)
from rlmcp.session import Session

NUM_ENVS = 4


# Stand-ins for the backend's own config classes, defined here so the export's
# generated import line names something that genuinely exists.


@dataclasses.dataclass
class SceneEntityCfg:
  name: str
  joint_names: tuple | None = None
  joint_ids: Any = dataclasses.field(default_factory=lambda: slice(None))
  preserve_order: bool = False


@dataclasses.dataclass
class RewardTermCfg:
  func: Callable[..., Any]
  weight: float
  params: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ObservationTermCfg:
  func: Callable[..., Any]
  params: dict[str, Any] = dataclasses.field(default_factory=dict)
  noise: float | None = None
  scale: float | None = None
  clip: tuple | None = None
  history_length: int = 0


@dataclasses.dataclass
class JointActionCfg:
  entity_name: str
  scale: float = 1.0
  clip: tuple | None = None


# Terms with real source, so inspect.getsource has something to read.


def alive_bonus(env, scale: float = 1.0):
  return torch.ones(env.num_envs) * scale


def joint_effort_penalty(env, asset_cfg: Any = None):
  return torch.zeros(env.num_envs)


def base_lin_vel(env):
  return torch.zeros(env.num_envs, 3)


def joint_pos_rel(env, asset_cfg: Any = None):
  return torch.zeros(env.num_envs, 12)


class _RewardManager:
  def __init__(self, terms: dict[str, RewardTermCfg]):
    self._terms = dict(terms)

  @property
  def active_terms(self) -> list[str]:
    return list(self._terms)

  def get_term_cfg(self, name: str) -> RewardTermCfg:
    return self._terms[name]


class _ObservationManager:
  def __init__(self, groups: dict[str, dict[str, ObservationTermCfg]]):
    self._groups = groups

  @property
  def active_terms(self) -> dict[str, list[str]]:
    return {g: list(t) for g, t in self._groups.items()}

  @property
  def group_obs_concatenate(self) -> dict[str, bool]:
    return dict.fromkeys(self._groups, True)

  @property
  def group_obs_term_dim(self) -> dict[str, list[tuple]]:
    return {"policy": [(3,), (12,)]}

  def get_term_cfg(self, group: str, name: str) -> ObservationTermCfg:
    return self._groups[group][name]


class _ActionTerm:
  def __init__(self, cfg: Any):
    self.cfg = cfg


class _ActionManager:
  def __init__(self, terms: dict[str, Any]):
    self._terms = dict(terms)

  @property
  def active_terms(self) -> list[str]:
    return list(self._terms)

  @property
  def action_term_dim(self) -> list[int]:
    return [12]

  def get_term(self, name: str) -> _ActionTerm:
    return self._terms[name]


class _Env:
  """A manager-based env with all three managers the export reads."""

  def __init__(self, rewards: dict[str, RewardTermCfg] | None = None):
    self.num_envs = NUM_ENVS
    self.device = "cpu"
    if rewards is None:
      rewards = {
          "alive": RewardTermCfg(func=alive_bonus, weight=1.5,
                                 params={"scale": 2.0}),
          "effort": RewardTermCfg(
              func=joint_effort_penalty, weight=-0.01,
              params={"asset_cfg": SceneEntityCfg(
                  "robot", joint_names=("hip", "knee"))}),
      }
    self.reward_manager = _RewardManager(rewards)
    self.observation_manager = _ObservationManager({
        "policy": {
            "base_lin_vel": ObservationTermCfg(func=base_lin_vel, noise=0.1,
                                               scale=2.0),
            "joint_pos": ObservationTermCfg(
                func=joint_pos_rel,
                params={"asset_cfg": SceneEntityCfg("robot")},
                clip=(-100.0, 100.0)),
        }
    })
    self.action_manager = _ActionManager({
        "joint_pos": _ActionTerm(JointActionCfg(entity_name="robot", scale=0.5))
    })


@pytest.fixture
def session(tmp_path) -> Session:
  """A session carrying a captured snapshot, as a real run would leave one."""
  s = Session(tmp_path / "session").create({"task": "Fake-Walk-v0"})
  snapshot = capture_env_terms(_Env())
  s.publish_env_terms({"task": "Fake-Walk-v0", **snapshot})
  return s


def _import_export(out_dir, name: str):
  """Import a generated module the way a user would: off sys.path.

  Only modules left over from *another* export are evicted. Dropping one this
  export already loaded would make ``env_cfg`` import a second, distinct copy
  of ``mdp_terms``, and the functions in the config would then not be the ones
  in the terms module -- an artefact of the test, not of the export.
  """
  sys.path.insert(0, str(out_dir))
  try:
    for stale in (env_export.TERMS_STEM, env_export.CFG_STEM):
      module = sys.modules.get(stale)
      if module is not None and not str(
          getattr(module, "__file__", "")).startswith(str(out_dir)):
        sys.modules.pop(stale)
    return importlib.import_module(name)
  finally:
    sys.path.remove(str(out_dir))


# Capture.


def test_capture_records_every_manager_with_its_source():
  snapshot = capture_env_terms(_Env())
  assert [t["name"] for t in snapshot["rewards"]] == ["alive", "effort"]
  assert list(snapshot["observations"]) == ["policy"]
  assert [t["name"] for t in snapshot["actions"]] == ["joint_pos"]
  assert snapshot["problems"] == []

  # Source, not an import path -- that is the whole point.
  assert "def alive_bonus(env" in snapshot["rewards"][0]["func"]["source"]
  assert snapshot["rewards"][0]["func"]["available"] is True

  # Observation order and dimensions are kept: they are the input vector.
  policy = snapshot["observations"]["policy"]["terms"]
  assert [t["name"] for t in policy] == ["base_lin_vel", "joint_pos"]
  assert policy[0]["dim"] == [3]


def test_capture_is_json_safe():
  """It is written to the session with json.dumps, so it has to survive it."""
  snapshot = capture_env_terms(_Env())
  assert json.loads(json.dumps(snapshot))["rewards"][0]["name"] == "alive"


def test_a_missing_manager_is_a_problem_not_an_exception():
  class _Bare:
    num_envs = NUM_ENVS

  snapshot = capture_env_terms(_Bare())
  assert snapshot["rewards"] == []
  assert any("reward_manager" in p for p in snapshot["problems"])


def test_resolved_scene_indices_are_dropped_when_names_are_present():
  """Those indices belong to the scene that resolved them, not to the config."""
  cfg = SceneEntityCfg("robot", joint_names=("hip",))
  cfg.joint_ids = [3, 7]  # As a manager would fill in.
  encoded = encode_value(cfg)
  assert "joint_ids" not in encoded["__obj__"]["fields"]
  assert encoded["__obj__"]["fields"]["joint_names"] == {
      "__tuple__": ["hip"]}


def test_an_unrenderable_value_is_flagged_rather_than_repr_guessed():
  encoded = encode_value(object())
  assert encoded["__unrenderable__"] is True


# Export: the generated files have to run.


def test_the_exported_config_imports_and_rebuilds_the_terms(session, tmp_path):
  out = tmp_path / "exported"
  payload = env_export.export_env(session, out)
  assert payload["ok"] is True
  assert payload["counts"] == {
      "rewards": 2, "observation_groups": 1, "observations": 2, "actions": 1}

  terms = _import_export(out, env_export.TERMS_STEM)
  assert callable(terms.alive_bonus)
  assert terms.alive_bonus(_Env()).shape == (NUM_ENVS,)

  cfg = _import_export(out, env_export.CFG_STEM)
  rewards = cfg.RewardsCfg()
  assert rewards.alive.weight == 1.5
  assert rewards.alive.params == {"scale": 2.0}
  assert rewards.alive.func is terms.alive_bonus

  # The SceneEntityCfg param survived as a constructed object, not a repr.
  assert rewards.effort.params["asset_cfg"].name == "robot"
  assert rewards.effort.params["asset_cfg"].joint_names == ("hip", "knee")

  observations = cfg.ObservationsCfg()
  assert observations.policy.base_lin_vel.noise == 0.1
  assert observations.policy.base_lin_vel.scale == 2.0
  assert observations.policy.joint_pos.clip == (-100.0, 100.0)

  actions = cfg.ActionsCfg()
  assert actions.joint_pos.entity_name == "robot"
  assert actions.joint_pos.scale == 0.5
  # A real dataclass field, so two instances do not share one mutable cfg.
  assert cfg.ActionsCfg().joint_pos is not actions.joint_pos


def test_an_action_term_is_not_reported_as_missing_source(session, tmp_path):
  """An action term is a backend class named by its config, not a function."""
  payload = env_export.export_env(session, tmp_path / "exported")
  assert payload["missing_source"] == []


def test_the_export_needs_no_task_package(session, tmp_path):
  """Implementations are inlined, so nothing has to still be installed."""
  out = tmp_path / "exported"
  env_export.export_env(session, out)
  text = (out / f"{env_export.TERMS_STEM}.py").read_text()
  assert "def alive_bonus" in text
  assert "def base_lin_vel" in text
  assert "def joint_pos_rel" in text


def test_weights_are_the_ones_the_run_ended_on(session, tmp_path):
  """A tuned weight is what the checkpoint trained under; export that."""
  session.publish_params({
      "reward.alive.weight": {"key": "reward.alive.weight", "current": 4.25,
                              "type": "float"},
  })
  out = tmp_path / "exported"
  env_export.export_env(session, out)

  cfg = _import_export(out, env_export.CFG_STEM)
  assert cfg.RewardsCfg().alive.weight == 4.25
  # And the value it was configured with is not silently lost.
  assert "configured at 1.5" in (out / f"{env_export.CFG_STEM}.py").read_text()


def test_a_term_with_no_capturable_source_is_named_not_faked(tmp_path):
  """A term that cannot be inlined must be reported, not quietly dropped."""
  env = _Env(rewards={
      "alive": RewardTermCfg(func=alive_bonus, weight=1.0),
      "builtin": RewardTermCfg(func=len, weight=1.0),
  })
  s = Session(tmp_path / "s").create({"task": "T"})
  s.publish_env_terms(capture_env_terms(env))

  out = tmp_path / "exported"
  payload = env_export.export_env(s, out)
  assert any("builtin" in entry for entry in payload["missing_source"])
  readme = (out / "README.md").read_text()
  assert "could not be captured" in readme
  assert "builtin" in readme
  # The rest of the export is still usable.
  assert "def alive_bonus" in (out / f"{env_export.TERMS_STEM}.py").read_text()


def test_a_run_that_captured_nothing_says_so_rather_than_writing_junk(tmp_path):
  s = Session(tmp_path / "s").create({"task": "T"})
  payload = env_export.export_env(s, tmp_path / "out")
  assert payload["ok"] is False
  assert "nothing to export" in payload["error"]
  assert not (tmp_path / "out").exists()
  assert "nothing to export" in env_export.describe(payload)


# A term is not its function alone. Reproduced on a real mjlab task first:
# `import mdp_terms` raised NameError on `_CART_CFG`, a module constant the
# term used as a default argument.


def _write_module(tmp_path, name: str, text: str):
  """A real importable module, so inspect.getsource has a file to read."""
  root = tmp_path / "pkgs"
  root.mkdir(exist_ok=True)
  (root / f"{name}.py").write_text(text)
  if str(root) not in sys.path:
    sys.path.insert(0, str(root))
  sys.modules.pop(name, None)
  return importlib.import_module(name)


TASK_MODULE = '''
"""A task's mdp module, the way real ones look."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from typing import Any

_SCALE = 2.0
_UNUSED = "not reached by any term"


def _tolerance(x: torch.Tensor, margin: float) -> torch.Tensor:
  return torch.exp(-(x / margin) ** 2)


def centered(env: Any, margin: float = _SCALE) -> torch.Tensor:
  """Reaches a helper, a constant (as a default) and an imported module."""
  return _tolerance(torch.zeros(env.num_envs), margin) * math.e / math.e
'''


def test_a_term_is_exported_with_the_module_names_it_reaches(tmp_path):
  mod = _write_module(tmp_path, "task_a_mdp", TASK_MODULE)
  env = _Env(rewards={"center": RewardTermCfg(func=mod.centered, weight=1.0)})
  s = Session(tmp_path / "s").create({"task": "T"})
  s.publish_env_terms(capture_env_terms(env))

  out = tmp_path / "exported"
  payload = env_export.export_env(s, out)
  terms = _import_export(out, env_export.TERMS_STEM)
  cfg = _import_export(out, env_export.CFG_STEM)

  value = cfg.RewardsCfg().center.func(env)
  assert value.shape == (NUM_ENVS,) and float(value[0]) == 1.0
  text = (out / f"{env_export.TERMS_STEM}.py").read_text()
  assert "_SCALE = 2.0" in text and "def _tolerance" in text
  assert "_UNUSED" not in text                       # only what the term reaches
  assert "import math" in text
  assert "math" in payload["still_imports"]
  assert "still imports" in (out / "README.md").read_text()
  assert terms.centered is cfg.RewardsCfg().center.func


def test_same_named_functions_from_two_modules_keep_their_own_bodies(tmp_path):
  """Two modules' `track` used to be written into one file, the second
  shadowing the first, and both config lines pointed at the survivor."""
  a = _write_module(tmp_path, "task_a",
                    "import torch\ndef track(env):\n  return torch.ones(env.num_envs)\n")
  b = _write_module(tmp_path, "task_b",
                    "import torch\ndef track(env):\n  return torch.zeros(env.num_envs)\n")
  env = _Env(rewards={"track_a": RewardTermCfg(func=a.track, weight=1.0),
                      "track_b": RewardTermCfg(func=b.track, weight=1.0)})
  s = Session(tmp_path / "s").create({"task": "T"})
  s.publish_env_terms(capture_env_terms(env))

  out = tmp_path / "exported"
  env_export.export_env(s, out)
  cfg = _import_export(out, env_export.CFG_STEM)

  rewards = cfg.RewardsCfg()
  assert float(rewards.track_a.func(env)[0]) == 1.0
  assert float(rewards.track_b.func(env)[0]) == 0.0
  assert rewards.track_b.func.__name__ == "track__task_b"


def test_a_term_with_no_source_leaves_the_config_constructible(tmp_path):
  """The config used to reference `mdp_terms.builtin_function_or_method`, so
  RewardsCfg() raised AttributeError -- the opposite of 'left out'."""
  env = _Env(rewards={
      "alive": RewardTermCfg(func=alive_bonus, weight=1.0),
      "builtin": RewardTermCfg(func=len, weight=1.0),
  })
  s = Session(tmp_path / "s").create({"task": "T"})
  s.publish_env_terms(capture_env_terms(env))

  out = tmp_path / "exported"
  env_export.export_env(s, out)
  cfg = _import_export(out, env_export.CFG_STEM)

  rewards = cfg.RewardsCfg()
  assert hasattr(rewards, "alive") and not hasattr(rewards, "builtin")
  assert "# builtin: source unavailable" in (out / f"{env_export.CFG_STEM}.py").read_text()
