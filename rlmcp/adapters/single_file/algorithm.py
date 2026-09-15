"""RunnerAdapter for an algorithm object that is not an rsl_rl runner.

A single-file training script owns its loop: it builds a PPO object, rolls
out, calls ``update()`` and saves. There is no runner with a logger to hook,
so the script tells rlmcp where the iteration boundary is (``env.service``)
and hands over the algorithm object for the rest -- hyperparameters and
checkpoints -- through this adapter.

The algorithm is read by duck typing. Hyperparameters are the attributes in
:data:`rlmcp.adapters.rsl_rl_runner.ALG_PARAMS` it happens to have
(``learning_rate``, ``entropy_coef``, ...); checkpoints go through ``save()``
returning a state dict and ``load(state)`` taking one back, which is the shape
agentic-rllab's PPO has and the natural one for any hand-written loop.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from rlmcp.adapters.base import NotSupported, RunnerAdapter
from rlmcp.adapters.rsl_rl_runner import ALG_PARAMS
from rlmcp.core.parameters.spec import ParameterCategory, ParameterSpec


class AlgorithmAdapter(RunnerAdapter):
  """Live control over a hand-rolled algorithm's knobs and checkpoints."""

  def __init__(self, algorithm: Any, log_dir: str | None = None):
    self.algorithm = algorithm
    self.iteration = 0
    self._log_dir = log_dir

  def discover_hyperparameters(self) -> list[ParameterSpec]:
    specs: list[ParameterSpec] = []
    for name, meta in ALG_PARAMS.items():
      value = getattr(self.algorithm, name, None)
      if value is None or isinstance(value, bool):
        continue
      try:
        current = float(value)
      except (TypeError, ValueError):
        continue
      specs.append(
          ParameterSpec(
              key=f"rl.{name}",
              data_type="float",
              current_value=current,
              default_value=current,
              min_value=meta.get("min"),
              max_value=meta.get("max"),
              description=meta["desc"],
              category=ParameterCategory.RL_HYPERPARAMETER,
          )
      )
    return specs

  def get_hyperparameter(self, key: str) -> Any:
    name = key.split(".", 1)[-1]
    if not hasattr(self.algorithm, name):
      raise KeyError(f"The algorithm has no hyperparameter '{name}'.")
    return float(getattr(self.algorithm, name))

  def set_hyperparameter(self, key: str, value: Any) -> bool:
    name = key.split(".", 1)[-1]
    alg = self.algorithm
    if name not in ALG_PARAMS or not hasattr(alg, name):
      known = sorted(k for k in ALG_PARAMS if hasattr(alg, k))
      raise KeyError(
          f"No tunable hyperparameter '{name}' on this algorithm. Available: {known}"
      )
    if name == "learning_rate":
      lr = float(value)
      optimizer = getattr(alg, "optimizer", None)
      if optimizer is not None:
        for group in optimizer.param_groups:
          group["lr"] = lr
      alg.learning_rate = lr
      if getattr(alg, "schedule", None) == "adaptive":
        alg.schedule = "fixed"
      return True
    current = getattr(alg, name)
    setattr(alg, name, int(value) if isinstance(current, int) else float(value))
    return True

  def current_iteration(self) -> int:
    return int(self.iteration)

  def runner_metrics(self) -> dict[str, float]:
    out: dict[str, float] = {}
    alg = self.algorithm
    if getattr(alg, "learning_rate", None) is not None:
      out["Loss/learning_rate"] = float(alg.learning_rate)
    with contextlib.suppress(Exception):
      actor = getattr(alg, "actor", None)
      out["Policy/mean_std"] = float(actor.output_std.mean().item())
    return out

  # Checkpoints: save() -> state dict, load(state).

  def save_checkpoint(
      self, path: str, infos: dict[str, Any] | None = None
  ) -> str | None:
    save = getattr(self.algorithm, "save", None)
    if not callable(save):
      raise NotSupported("The algorithm does not implement save().")
    import torch

    state = save()
    if not isinstance(state, dict):
      raise NotSupported(
          "The algorithm's save() must return a state dict for rlmcp to file it."
      )
    state = dict(state)
    state["iteration"] = int(self.iteration)
    state["rlmcp_infos"] = dict(infos or {})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))
    return str(path)

  def load_checkpoint(self, path: str) -> dict[str, Any]:
    load = getattr(self.algorithm, "load", None)
    if not callable(load):
      raise NotSupported("The algorithm does not implement load().")
    if not Path(path).exists():
      raise FileNotFoundError(f"No checkpoint at '{path}'.")
    import torch

    state = torch.load(str(path), map_location="cpu", weights_only=False)
    load(state)
    infos = state.get("rlmcp_infos", {}) if isinstance(state, dict) else {}
    return infos if isinstance(infos, dict) else {}

  def log_dir(self) -> str | None:
    return self._log_dir

  def request_stop(self) -> bool:
    """No-op, like rsl_rl's: stopping rides the servicing contract instead."""
    return False


__all__ = ["AlgorithmAdapter"]
