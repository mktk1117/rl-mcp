"""Every declared parameter of a single-file environment, through the MCP server.

``test_single_file_env.py`` checks a handful of named keys at the adapter
level. This file checks the whole surface, and through the server rather than
the adapter, because "an agent can tune anything the config declares" is the
promise of the family and nothing else keeps it:

* the config tree is walked here, independently of the adapter, and every
  numeric leaf it finds must be a parameter ``list_parameters`` returns;
* every listed parameter must read back through ``get_parameter``;
* every live one must accept a write through ``set_parameter`` that lands in
  the config object itself (checked by resolving the key by hand, not through
  the registry) and reads back changed;
* every ``at_startup`` one must be refused with that reason and left alone;
* ``reset_parameters`` must put every value back.

Two configs go through it: the fake from ``test_single_file_env.py`` and the
real Go1 example's ``EnvConfig`` on a config-only stub, so the example's own
declared surface is covered without a simulator.

The server is the real one (``create_mcp_server``), pinned at the wrapped
run's session directory; a thread services the run the way a training loop
would, so each tool call is answered over the same request path a live agent
uses.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mcp", reason="the MCP server needs the optional 'mcp' package")

from test_single_file_env import FakePPO, FakeSingleFileEnv  # noqa: E402

from rlmcp import declare  # noqa: E402
from rlmcp.adapters.access import paths  # noqa: E402
from rlmcp.adapters.single_file import (  # noqa: E402
  AlgorithmAdapter,
  blocks,
  wrap,
)
from rlmcp.server.mcp_server import create_mcp_server  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/single_file/go1_flat/env.py"


# The two configs.


def _example_module(stem: str):
  """``env`` or ``ppo`` from the example directory, loaded by path."""
  name = f"go1_flat_{stem}_under_test"
  if name not in sys.modules:
    spec = importlib.util.spec_from_file_location(name, EXAMPLE.with_name(f"{stem}.py"))
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: the file's dataclasses carry string
    # annotations, which the dataclass machinery resolves via sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
  return sys.modules[name]


def _go1_config_class():
  return _example_module("env").EnvConfig


class ConfigOnlyEnv:
  """The example's config on an env that builds no physics.

  Enough of the conventional shape for the wrapper to accept it and for the
  servicing loop to run: the buffers exist, ``step`` is never called.
  """

  def __init__(self, cfg: Any):
    self.cfg = cfg
    self.num_envs = 2
    self.device = torch.device("cpu")
    self.control_dt = cfg.sim.dt * cfg.sim.decimation
    self.max_episode_steps = 10
    self.sim = None
    n, j = self.num_envs, 12
    self.dof_pos = torch.zeros(n, j)
    self.dof_vel = torch.zeros(n, j)
    self.actions = torch.zeros(n, j)
    self.base_pos = torch.zeros(n, 3)
    self.base_lin_vel = torch.zeros(n, 3)
    self.base_ang_vel = torch.zeros(n, 3)
    self.projected_gravity = torch.tensor([[0.0, 0.0, -1.0]] * n)
    self.commands = torch.zeros(n, 3)
    self.rew_buf = torch.zeros(n)

  def reset(self, env_ids=None):
    return torch.zeros(self.num_envs, 4)


def _fake() -> Any:
  return FakeSingleFileEnv()


def _go1() -> Any:
  return ConfigOnlyEnv(_go1_config_class()(num_envs=2, device="cpu"))


CONFIGS = {"fake": _fake, "go1_example": _go1}


# Expected keys, from the config tree alone.


def expected_keys(cfg: Any, reward_group: str = "reward", env: Any = None) -> dict[str, bool]:
  """``{key: static}`` for every numeric leaf the tree declares.

  This is the adapter's contract restated from the outside: a top-level scalar
  is ``env.<name>``, a field of a nested dataclass is ``<group>.<...>.<name>``,
  a term is ``reward.<name>.weight`` plus ``reward.<name>.params.<p>``, and a
  stage field of an observation group on the environment is
  ``<attr>.<term>.<stage>.<field>``. Strings and dicts are not leaves and are
  not expected: a dict-valued field (the example's per-joint gain tables) is
  not served at all, so the listed set and this one are the same set, which
  the first test also checks.
  """
  out: dict[str, bool] = {}
  for attr, block in blocks.blocks(env).items() if env is not None else ():
    for path, _, field_name, static in blocks.block_leaves(block):
      out[".".join((attr, *path, field_name))] = static
  for path, value, static in declare.walk(cfg):
    if not paths.is_leaf(value):
      continue
    key = paths.join_path(path) if len(path) > 1 else f"env.{path[0]}"
    out[key] = static
  for name, t in declare.terms(getattr(cfg, reward_group, None)).items():
    out[f"{reward_group}.{name}.weight"] = False
    for p, v in t.params.items():
      if paths.is_leaf(v):
        out[f"{reward_group}.{name}.params.{p}"] = False
  return out


def resolve(cfg: Any, algorithm: Any, key: str, env: Any = None) -> Any:
  """The value ``key`` names, read straight off the objects, no registry."""
  parts = paths.split_path(key)
  head, rest = parts[0], parts[1:]
  block = blocks.blocks(env).get(head) if env is not None else None
  if block is not None:
    pipe = block.terms[rest[0]].pipe if isinstance(block, blocks.Obs) else block
    stage = pipe.named_stages()[rest[-2]]
    return getattr(stage, rest[-1])
  if head == "rl":
    if hasattr(algorithm, rest[0]):
      return getattr(algorithm, rest[0])
    return getattr(algorithm.cfg, rest[0])
  if head == "env":
    return getattr(cfg, rest[0])
  obj = getattr(cfg, head)
  if declare.is_term(getattr(obj, rest[0], None)):
    term = getattr(obj, rest[0])
    if rest[1] == "weight":
      return term.weight
    assert rest[1] == "params"
    return term.params[rest[2]]
  for part in rest:
    obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
  return obj


# Driving the server.


def _text(result: Any) -> str:
  content = getattr(result, "content", None) or (
      result[0] if isinstance(result, tuple) else result
  )
  if isinstance(content, list):
    return "\n".join(str(getattr(c, "text", c)) for c in content)
  return str(content)


class Served:
  """A wrapped single-file run, serviced on a thread, behind a real server."""

  def __init__(self, env: Any, tmp_path: Path):
    self.env = env
    self.algorithm = FakePPO()
    self.wrapped = wrap(env, session_dir=tmp_path / "run", viser=False, video_every=0)
    self.wrapped.attach_algorithm(self.algorithm)
    self._stop = threading.Event()
    self._thread = threading.Thread(target=self._pump, daemon=True)
    self._thread.start()
    self.server = create_mcp_server(
        session_dir=str(self.wrapped.rlmcp.session.dir),
        records_root=str(tmp_path / "records"),
    )

  def _pump(self) -> None:
    iteration = 0
    while not self._stop.is_set():
      iteration += 1
      self.wrapped.service(iteration)
      time.sleep(0.002)

  def close(self) -> None:
    self._stop.set()
    self._thread.join(timeout=5)

  def call(self, tool: str, **args: Any) -> dict[str, Any]:
    result = asyncio.run(self.server.call_tool(tool, args))
    return json.loads(_text(result))

  def get(self, key: str) -> dict[str, Any]:
    """One parameter's live value, the way an agent reads it: there is no
    ``get_parameter`` tool, so the controller verb goes through
    ``run_command`` (``list_parameters(contains=key)`` is the other route)."""
    return self.call("run_command", cmd="get_parameter", args={"key": key})

  @property
  def cfg(self) -> Any:
    return self.env.cfg


@pytest.fixture(params=sorted(CONFIGS))
def served(request, tmp_path) -> Served:
  run = Served(CONFIGS[request.param](), tmp_path)
  try:
    yield run
  finally:
    run.close()


# Values.


def nudged(spec: dict[str, Any]) -> Any:
  """A value that differs from the current one and respects the bounds."""
  current, kind = spec["current"], spec["type"]
  lo, hi = spec.get("min"), spec.get("max")
  if kind == "bool":
    return not current
  if kind == "range":
    return [current[0] + 0.25, current[1] + 0.5]
  if kind == "int":
    candidates = [current + 1, current - 1]
  else:
    candidates = [current * 1.5 + 0.125, current * 0.5 - 0.125, current + 0.125]
  for value in candidates:
    if lo is not None and value < lo:
      continue
    if hi is not None and value > hi:
      continue
    if value != current:
      return value
  raise AssertionError(f"could not pick a new value for {spec['key']}: {spec}")


def same(a: Any, b: Any) -> bool:
  if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
    return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
  if isinstance(a, bool) or isinstance(b, bool):
    return a == b
  if isinstance(a, (int, float)) and isinstance(b, (int, float)):
    return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
  return a == b


# The tests.


def test_every_declared_leaf_is_listed_by_the_server(served):
  listed = served.call("list_parameters")
  assert listed["ok"], listed
  keys = listed["parameters"]
  expected = expected_keys(served.cfg, env=served.env)
  missing = sorted(set(expected) - set(keys))
  assert not missing, f"declared but not served: {missing}"
  unexpected = sorted(set(keys) - set(expected) - {k for k in keys if k.startswith("rl.")})
  assert not unexpected, f"served but not declared as a leaf: {unexpected}"
  wrong_liveness = sorted(
      k for k, static in expected.items()
      if (keys[k]["liveness"] == "at_startup") != static
  )
  assert not wrong_liveness, f"liveness disagrees with the declaration: {wrong_liveness}"
  # The algorithm's knobs are on the same surface.
  assert "rl.learning_rate" in keys
  assert "rl.entropy_coef" in keys


def test_every_listed_parameter_reads_back_through_the_server(served):
  keys = served.call("list_parameters")["parameters"]
  assert keys
  for key, spec in keys.items():
    got = served.get(key)
    assert got["ok"], (key, got)
    assert same(got["value"], spec["current"]), (key, got["value"], spec["current"])
    assert same(got["value"], resolve(served.cfg, served.algorithm, key, served.env)), key


def test_every_live_parameter_can_be_set_and_lands_in_the_config(served):
  keys = served.call("list_parameters")["parameters"]
  live = {k: s for k, s in keys.items() if s["liveness"] == "live"}
  assert live
  for key, spec in live.items():
    value = nudged(spec)
    out = served.call("set_parameter", key=key, value=value, rationale="round trip")
    assert out["ok"] and out["applied"], (key, value, out)
    assert same(out["new_value"], value), (key, out)
    assert same(served.get(key)["value"], value), key
    # The config object itself, not the registry's idea of it.
    assert same(resolve(served.cfg, served.algorithm, key, served.env), value), key


def test_every_static_parameter_is_refused_with_the_reason(served):
  keys = served.call("list_parameters")["parameters"]
  static = {k: s for k, s in keys.items() if s["liveness"] == "at_startup"}
  assert static, "both configs declare Static fields"
  for key, spec in static.items():
    before = resolve(served.cfg, served.algorithm, key, served.env)
    out = served.call("set_parameter", key=key, value=nudged(spec), rationale="must refuse")
    assert not out.get("ok") or not out.get("applied"), (key, out)
    reason = out.get("error", "") or ""
    assert "at_startup" in reason, (key, out)
    assert same(resolve(served.cfg, served.algorithm, key, served.env), before), key


def test_reset_parameters_puts_every_live_value_back(served):
  keys = served.call("list_parameters")["parameters"]
  live = {k: s for k, s in keys.items() if s["liveness"] == "live"}
  for key, spec in live.items():
    served.call("set_parameter", key=key, value=nudged(spec), rationale="then reset")
  out = served.call("reset_parameters")
  assert out["ok"], out
  after = served.call("list_parameters")["parameters"]
  drifted = [k for k in live if not same(after[k]["current"], keys[k]["default"])]
  assert not drifted, f"not restored: {drifted}"
  for key in live:
    assert same(resolve(served.cfg, served.algorithm, key, served.env), keys[key]["default"]), key


def test_the_example_ppo_reloads_after_an_inference_mode_rollout(tmp_path):
  """rollback_to_checkpoint failed on the real run: the observation normaliser
  rebound its std buffer under inference mode during the rollout, and a later
  load_state_dict could not copy into the inference tensor it had become."""
  ppo_module = _example_module("ppo")
  actor = ppo_module.Actor(6, 2, ppo_module.ModelConfig())
  critic = ppo_module.Critic(6, ppo_module.ModelConfig())
  storage = ppo_module.RolloutStorage(4, 3, 6, 2, torch.device("cpu"), critic_obs_dim=6)
  ppo = ppo_module.PPO(actor, critic, storage, ppo_module.PPOConfig())
  ppo.train_mode()
  with torch.inference_mode():
    for _ in range(3):
      ppo.act(torch.randn(4, 6), torch.randn(4, 6))
      ppo.process_env_step(torch.zeros(4), torch.zeros(4, dtype=torch.bool))
  state = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in ppo.save().items()}
  adapter = AlgorithmAdapter(ppo)
  path = adapter.save_checkpoint(str(tmp_path / "c.pt"))
  with torch.inference_mode():
    ppo.act(torch.randn(4, 6), torch.randn(4, 6))
  adapter.load_checkpoint(path)  # raised RuntimeError before the fix
  ppo.load(state)
  assert ppo.cfg.learning_rate == ppo.learning_rate


def test_the_go1_example_declares_the_surface_the_docs_promise():
  """The page says ``params`` listed 73 on the real run; the config-only stub
  must find at least the keys an agent is told to reach for."""
  cfg = _go1_config_class()()
  expected = expected_keys(cfg)
  for key in (
      "reward.track_linear_velocity.weight",
      "reward.track_linear_velocity.params.std",
      "reward.foot_slip.params.command_threshold",
      "command.lin_vel_x",
      "command.rel_heading_envs",
      "termination.fell_over_deg",
      "randomization.push_interval_s",
      "action.scale_calf",
      "env.episode_length_s",
  ):
    assert key in expected, key
  assert expected["env.num_envs"] is True
  assert expected["randomization.startup.foot_friction"] is True
  assert expected["sim.options.iterations"] is True
  assert expected["command.lin_vel_x"] is False
  assert dataclasses.is_dataclass(cfg)
