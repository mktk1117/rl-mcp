"""Extensions: capabilities that only some environments have.

The core of rlmcp knows about parameters, metrics, traces, frames and stages.
It deliberately knows nothing about terrain, object sets, goal distributions or
anything else specific to one kind of task -- otherwise every environment that
is not a legged robot walking over a heightfield carries dead vocabulary.

An extension adds such a capability without the core learning about it:

* **commands** -- new verbs, reachable from the CLI and MCP exactly like the
  built-in ones, and usable in a curriculum stage,
* **metrics** -- extra scalars merged into the per-iteration telemetry, which
  curriculum promotion conditions can then reference,
* **env selection** -- answers "which environments match this description", so
  ``screenshot(where={"terrain": "stairs"})`` works without the core knowing
  what a terrain is,
* **checkpoint state** -- anything that must be saved and restored alongside
  the policy.

Extensions declare their own availability, so wrapping a manipulation env simply
yields fewer commands rather than errors.

After registration the controller hands each extension an
:class:`ExtensionContext` through :meth:`Extension.bind`: the sanctioned way to
write artifacts, read telemetry, append session events, and run deferred jobs
(a command handler that returns a :class:`~rlmcp.core.controller.DeferredJob`
is serviced exactly like the built-in video/trace commands).

Hook failures are not silent: the registry reports each failing
``(extension, hook)`` pair once -- into the session event log when a controller
is attached, as a warning otherwise -- and keeps going, so a broken extension
is visible without taking the run down.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only.
  from rlmcp.core.telemetry.buffer import TelemetryBuffer


@dataclass
class ExtensionContext:
  """What the controller offers an extension, handed over at registration.

  Replaces the old convention of probing constructors for an ``(env,
  plot_sink)`` signature: an extension that needs controller facilities
  overrides (or inherits) :meth:`Extension.bind` and uses these attributes.

  Attributes:
    write_artifact: ``(stem, suffix, payload_bytes) -> Path`` -- save bytes
      into the session's artifact directory, named with the current iteration.
    telemetry: the run's :class:`~rlmcp.core.telemetry.buffer.TelemetryBuffer`
      for *reading* recorded metrics (``get_series``, ``get_latest_metrics``);
      pushing belongs to the controller.
    append_event: ``(kind, detail_dict) -> None`` -- append to the session's
      event log.
    submit_job: schedule a :class:`~rlmcp.core.controller.DeferredJob` for
      step feeding outside a command handler; returns the job's description.
      The job's outcome is written to the session event log (a job returned
      from a command handler instead answers that command's request).
    pending_jobs: ``() -> list`` of in-flight deferred job descriptions.
  """

  write_artifact: Callable[..., Any]
  telemetry: "TelemetryBuffer"
  append_event: Callable[[str, Dict[str, Any]], None]
  submit_job: Callable[[Any], Dict[str, Any]]
  pending_jobs: Callable[[], List[Dict[str, Any]]]


class Extension:
  """A capability bound to one environment.

  Subclasses implement whichever hooks apply. Everything is optional except
  :attr:`name` and :meth:`available`.
  """

  name: str = ""

  def __init__(self, env: Any):
    self.env = env
    self.context: Optional[ExtensionContext] = None

  def available(self) -> bool:
    """Whether this environment supports the capability at all."""
    return False

  # Lifecycle.

  def bind(self, context: ExtensionContext) -> None:
    """Receive the controller's :class:`ExtensionContext` after registration.

    The default stores it as :attr:`context`. Override to eagerly wire up
    resources; call ``super().bind(context)`` to keep the attribute.
    """
    self.context = context

  def on_iteration(self, iteration: int, metrics: Dict[str, float]) -> None:
    """Called once per learning iteration with the merged metrics."""
    return None

  def close(self) -> None:
    """Called once when the run shuts down; release what you hold."""
    return None

  # Hooks.

  def commands(self) -> Dict[str, Callable[..., Any]]:
    """``{command_name: handler}``, merged into the controller's dispatch table.

    Handler docstrings become the tool descriptions an agent reads, so write
    the first line as the one-line summary. A handler may return a
    :class:`~rlmcp.core.controller.DeferredJob` to answer over the coming
    simulation steps instead of immediately.
    """
    return {}

  def metrics(self) -> Dict[str, float]:
    """Extra scalars for this iteration, conventionally prefixed ``rlmcp/``."""
    return {}

  def select_envs(self, **criteria: Any) -> Optional[List[int]]:
    """Environment indices matching ``criteria``, or None if not understood.

    Returning None means "this is not my vocabulary" and lets another extension
    answer; returning an empty list means "mine, and nothing matches".
    """
    return None

  def describe(self) -> Dict[str, Any]:
    """Short summary for the status payload."""
    return {}

  def snapshot(self) -> Dict[str, Any]:
    """State to persist with a checkpoint."""
    return {}

  def restore(self, state: Dict[str, Any]) -> None:
    """Restore what :meth:`snapshot` saved."""
    return None


class ExtensionRegistry:
  """The extensions active for one run.

  Aggregates the hooks and reports hook failures truthfully: each failing
  ``(extension, hook)`` pair is reported exactly once through the attached
  error sink (the controller routes it into the session event log), or as a
  :class:`RuntimeWarning` when no sink is attached, and the failing extension
  is skipped rather than allowed to take the run down.
  """

  def __init__(self, extensions: Optional[List[Extension]] = None):
    self._extensions: List[Extension] = []
    self._error_sink: Optional[Callable[[str, str, str], None]] = None
    self._reported: Set[Tuple[str, str]] = set()
    for extension in extensions or []:
      self.add(extension)

  def set_error_sink(self, sink: Callable[[str, str, str], None]) -> None:
    """Route hook-failure reports as ``sink(extension_name, hook, message)``."""
    self._error_sink = sink

  def _report(self, extension_name: str, hook: str, message: str) -> None:
    """Report one hook failure, once per (extension, hook)."""
    key = (extension_name, hook)
    if key in self._reported:
      return
    self._reported.add(key)
    if self._error_sink is not None:
      try:
        self._error_sink(extension_name, hook, message)
        return
      except Exception:
        pass  # A broken sink must not mask the original failure.
    warnings.warn(
        f"rlmcp extension '{extension_name}' failed in {hook}: {message}",
        RuntimeWarning,
        stacklevel=3,
    )

  def add(self, extension: Extension) -> bool:
    """Register an extension if this environment supports it."""
    try:
      if not extension.available():
        return False
    except Exception:
      return False
    self._extensions.append(extension)
    return True

  def names(self) -> List[str]:
    return [e.name for e in self._extensions]

  def __iter__(self):
    return iter(self._extensions)

  def __len__(self) -> int:
    return len(self._extensions)

  # Aggregated lifecycle.

  def bind(self, extension: Extension, context: ExtensionContext) -> None:
    """Hand ``context`` to one extension, reporting (not raising) a failure."""
    try:
      extension.bind(context)
    except Exception as exc:
      self._report(extension.name, "bind", str(exc))

  def bind_all(self, context: ExtensionContext) -> None:
    for extension in self._extensions:
      self.bind(extension, context)

  def on_iteration(self, iteration: int, metrics: Dict[str, float]) -> None:
    for extension in self._extensions:
      try:
        extension.on_iteration(iteration, metrics)
      except Exception as exc:
        self._report(extension.name, "on_iteration", str(exc))

  def close(self) -> None:
    for extension in self._extensions:
      try:
        extension.close()
      except Exception as exc:
        self._report(extension.name, "close", str(exc))

  # Aggregated hooks.

  def commands(self) -> Dict[str, Callable[..., Any]]:
    """Every extension's commands, first registration wins on a name clash.

    A clash is reported (once) naming both extensions, so a shadowed command
    is a logged fact rather than a silent surprise. The same first-wins rule
    governs :meth:`~rlmcp.core.controller.RlMcp.add_extension`.
    """
    out: Dict[str, Callable[..., Any]] = {}
    owners: Dict[str, str] = {}
    for extension in self._extensions:
      try:
        contributed = extension.commands()
      except Exception as exc:
        self._report(extension.name, "commands", str(exc))
        continue
      for name, handler in contributed.items():
        if name in out:
          self._report(
              extension.name,
              f"commands:{name}",
              f"command '{name}' already provided by extension "
              f"'{owners[name]}'; keeping the first, ignoring "
              f"'{extension.name}'s.",
          )
          continue
        out[name] = handler
        owners[name] = extension.name
    return out

  def metrics(self) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for extension in self._extensions:
      try:
        out.update(extension.metrics())
      except Exception as exc:
        self._report(extension.name, "metrics", str(exc))
        continue
    return out

  def select_envs(self, **criteria: Any) -> Optional[List[int]]:
    """Ask each extension in turn; the first that understands wins."""
    for extension in self._extensions:
      try:
        result = extension.select_envs(**criteria)
      except TypeError:
        continue  # Different vocabulary; not an error.
      if result is not None:
        return result
    return None

  def describe(self) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for extension in self._extensions:
      try:
        summary = extension.describe()
      except Exception as exc:
        self._report(extension.name, "describe", str(exc))
        continue
      if summary:
        out[extension.name] = summary
    return out

  def snapshot(self) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for extension in self._extensions:
      try:
        state = extension.snapshot()
      except Exception as exc:
        self._report(extension.name, "snapshot", str(exc))
        continue
      if state:
        out[extension.name] = state
    return out

  def restore(self, state: Dict[str, Any]) -> Dict[str, bool]:
    """Restore each extension's payload; return per-extension success.

    The returned dict has an entry per registered extension that *had* a
    payload: ``True`` if its ``restore`` ran clean, ``False`` if it raised
    (reported once, run continues). ``cmd_load_checkpoint`` counts the Trues,
    so a checkpoint whose state failed to apply is never reported as restored.
    """
    results: Dict[str, bool] = {}
    for extension in self._extensions:
      payload = (state or {}).get(extension.name)
      if not payload:
        continue
      try:
        extension.restore(payload)
        results[extension.name] = True
      except Exception as exc:
        self._report(extension.name, "restore", str(exc))
        results[extension.name] = False
    return results
