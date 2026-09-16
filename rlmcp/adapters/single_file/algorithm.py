"""RunnerAdapter for an algorithm object that is not an rsl_rl runner.

A single-file training script owns its loop: it builds a PPO object, rolls
out, calls ``update()`` and saves. There is no runner with a logger to hook,
so the script tells rlmcp where the iteration boundary is (``env.service``)
and hands over the algorithm object for the rest -- hyperparameters and
checkpoints -- through this adapter.

The algorithm is read the way the environment is: by what it declares.

* **Hyperparameters** are the numeric leaves of the dataclass on
  ``algorithm.cfg``, served as ``rl.<name>`` (``rl.learning_rate``,
  ``rl.entropy_coef``, ...). ``Static[...]`` marks one read once -- a
  network width, say -- and a write to it is refused with that reason. The
  live value is the attribute of the same name on the algorithm when it
  mirrors the field (the common pattern, ``self.entropy_coef = cfg.entropy_coef``),
  else the config field itself; a write sets both, then calls
  ``on_hyperparameter_change(name, value)`` if the algorithm defines it, for
  whatever must follow the write -- an optimizer's learning rate, a schedule
  that would otherwise overwrite the edit. Nothing is hand-listed here, so a
  knob added to the config is tunable the moment it is declared.
* **Metrics**: ``algorithm.metrics()``, if defined, returning ``{name: float}``,
  is published alongside rlmcp's own each iteration.
* **Checkpoints** go through ``save()`` returning a state dict and
  ``load(state)`` taking one back, the natural shape for a hand-written loop.
"""

from __future__ import annotations

import contextlib
import dataclasses
from pathlib import Path
from typing import Any

from rlmcp import declare
from rlmcp.adapters.access import paths
from rlmcp.adapters.base import NotSupported, RunnerAdapter
from rlmcp.core.parameters.spec import Liveness, ParameterCategory, ParameterSpec


class AlgorithmAdapter(RunnerAdapter):
  """Live control over a hand-rolled algorithm's declared knobs and checkpoints."""

  def __init__(self, algorithm: Any, log_dir: str | None = None):
    self.algorithm = algorithm
    self.iteration = 0
    self._log_dir = log_dir

  # The declared surface.

  @property
  def cfg(self) -> Any | None:
    cfg = getattr(self.algorithm, "cfg", None)
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
      return cfg
    return None

  def _leaves(self) -> dict[str, tuple[tuple[str, ...], bool]]:
    """``{name: (path, static)}`` for every numeric leaf of the config."""
    cfg = self.cfg
    if cfg is None:
      return {}
    out = {}
    for path, value, static in declare.walk(cfg):
      if paths.is_leaf(value):
        out[".".join(path)] = (path, static)
    return out

  def _read(self, path: tuple[str, ...]) -> Any:
    """The live value: the mirrored attribute when there is one, else the field."""
    if len(path) == 1 and hasattr(self.algorithm, path[0]):
      return getattr(self.algorithm, path[0])
    node = self.cfg
    for part in path:
      node = getattr(node, part)
    return node

  def _write(self, path: tuple[str, ...], value: Any) -> None:
    node = self.cfg
    for part in path[:-1]:
      node = getattr(node, part)
    setattr(node, path[-1], value)
    if len(path) == 1 and hasattr(self.algorithm, path[0]):
      setattr(self.algorithm, path[0], value)

  def discover_hyperparameters(self) -> list[ParameterSpec]:
    specs: list[ParameterSpec] = []
    for name, (path, static) in self._leaves().items():
      value = self._read(path)
      kind = paths.leaf_kind(value)
      current = list(value) if kind == "range" else value
      specs.append(
          ParameterSpec(
              key=f"rl.{name}",
              data_type=kind,
              current_value=current,
              default_value=current,
              description=f"'{name}' of the algorithm config",
              category=ParameterCategory.RL_HYPERPARAMETER,
              liveness=Liveness.AT_STARTUP if static else Liveness.LIVE,
          )
      )
    return specs

  def _path_for(self, key: str) -> tuple[tuple[str, ...], bool]:
    name = key.split(".", 1)[-1]
    leaves = self._leaves()
    if name not in leaves:
      if self.cfg is None:
        raise KeyError(
            "The algorithm declares no `cfg` dataclass, so it has no tunable "
            "hyperparameters. Give it one (see docs/single-file.md)."
        )
      raise KeyError(
          f"No hyperparameter '{name}' on this algorithm. Available: {sorted(leaves)}"
      )
    return leaves[name]

  def get_hyperparameter(self, key: str) -> Any:
    path, _ = self._path_for(key)
    value = self._read(path)
    return list(value) if paths.is_range(value) else value

  def set_hyperparameter(self, key: str, value: Any) -> bool:
    path, static = self._path_for(key)
    name = ".".join(path)
    if static:
      raise ValueError(
          f"Hyperparameter 'rl.{name}' has liveness 'at_startup': the algorithm "
          "reads it once when it is built, so a write now would change nothing. "
          "Change the config and restart training instead."
      )
    current = self._read(path)
    coerced = paths.coerce_like(current, value)
    if isinstance(current, tuple) and isinstance(coerced, list):
      coerced = tuple(coerced)
    self._write(path, coerced)
    hook = getattr(self.algorithm, "on_hyperparameter_change", None)
    if callable(hook):
      hook(name, coerced)
    return True

  # Iteration and metrics.

  def current_iteration(self) -> int:
    return int(self.iteration)

  def runner_metrics(self) -> dict[str, float]:
    out: dict[str, float] = {}
    lr = getattr(self.algorithm, "learning_rate", None)
    if lr is not None:
      with contextlib.suppress(TypeError, ValueError):
        out["Loss/learning_rate"] = float(lr)
    metrics = getattr(self.algorithm, "metrics", None)
    if callable(metrics):
      with contextlib.suppress(Exception):
        for name, value in dict(metrics()).items():
          out[str(name)] = float(value)
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
    # Inside inference mode, for the reason the sim adapter's reset gives: a
    # buffer the loop last touched under inference mode (a running observation
    # normaliser, typically) refuses an in-place copy from outside it.
    with torch.inference_mode():
      load(state)
    infos = state.get("rlmcp_infos", {}) if isinstance(state, dict) else {}
    return infos if isinstance(infos, dict) else {}

  def log_dir(self) -> str | None:
    return self._log_dir

  def request_stop(self) -> bool:
    """No-op, like rsl_rl's: stopping rides the servicing contract instead."""
    return False


__all__ = ["AlgorithmAdapter"]
