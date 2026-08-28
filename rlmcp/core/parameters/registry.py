"""Central Parameter Registry and Schema Manager."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from rlmcp.core.parameters.spec import Liveness, ParameterCategory, ParameterSpec


class ParameterRegistry:
  """Maintains the active schema and dispatch table for dynamic parameters."""

  def __init__(self):
    self._specs: dict[str, ParameterSpec] = {}
    self._setters: dict[str, Callable[[Any], bool]] = {}
    self._getters: dict[str, Callable[[], Any]] = {}

  def register(
      self,
      spec: ParameterSpec,
      setter: Callable[[Any], bool] | None = None,
      getter: Callable[[], Any] | None = None,
  ) -> None:
    """Registers a parameter spec and its getter/setter callbacks."""
    self._specs[spec.key] = spec
    if setter:
      self._setters[spec.key] = setter
    if getter:
      self._getters[spec.key] = getter

  def get_spec(self, key: str) -> ParameterSpec | None:
    """Retrieves spec for a given parameter key."""
    return self._specs.get(key)

  def get_value(self, key: str) -> Any:
    """Gets the live value of a parameter."""
    if key in self._getters:
      val = self._getters[key]()
      if key in self._specs:
        self._specs[key].current_value = val
      return val
    if key in self._specs:
      return self._specs[key].current_value
    raise KeyError(f"Parameter '{key}' not found in registry.")

  def set_value(self, key: str, value: Any) -> bool:
    """Validates and updates the parameter value.

    Refuses writes that could never take effect this run (liveness
    'at_startup' or 'inert') before touching anything, so an agent is never
    told a dead write was applied.
    """
    if key not in self._specs:
      raise KeyError(f"Parameter '{key}' is not registered.")

    spec = self._specs[key]
    liveness = Liveness(spec.liveness)
    if liveness is Liveness.AT_STARTUP:
      raise ValueError(
          f"Parameter '{key}' has liveness 'at_startup': it is only read when "
          "the environment is constructed, so a write now would report success "
          "and change nothing for the rest of this run. Change the task config "
          "and restart training instead."
      )
    if liveness is Liveness.INERT:
      raise ValueError(
          f"Parameter '{key}' has liveness 'inert': the owning term cached "
          "this value at construction and never re-reads it, so a write can "
          "never take effect. Change the task config and restart training "
          "instead."
      )
    spec.validate(value)

    if key in self._setters:
      success = self._setters[key](value)
      if success:
        spec.current_value = value
      return success
    spec.current_value = value
    return True

  def get_all_specs(
      self, category: ParameterCategory | None = None
  ) -> list[ParameterSpec]:
    """Returns all registered parameter specs, optionally filtered by category."""
    if category is None:
      return list(self._specs.values())
    return [s for s in self._specs.values() if s.category == category]

  def get_snapshot(self) -> dict[str, Any]:
    """Returns a flat dictionary snapshot of all current parameter values."""
    snapshot = {}
    for key in self._specs:
      snapshot[key] = self.get_value(key)
    return snapshot

  def compute_diff(self, baseline_snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Computes difference between current values and a baseline snapshot."""
    diff = {}
    current = self.get_snapshot()
    for key, curr_val in current.items():
      base_val = baseline_snapshot.get(key)
      if base_val != curr_val:
        diff[key] = {"old": base_val, "new": curr_val}
    return diff

  def export_schema_json(self) -> dict[str, Any]:
    """Exports structured schema suitable for LLM context / MCP resource."""
    schema = {}
    for key, spec in self._specs.items():
      # Update live value
      if key in self._getters:
        with contextlib.suppress(Exception):
          spec.current_value = self._getters[key]()
      schema[key] = spec.to_dict()
    return schema
