"""Adapter contracts between rlmcp and a simulator / RL runner.

rlmcp targets manager-based environments whose config is live -- mjlab,
IsaacLab, and anything built the same way: term configs are ordinary
dataclasses and dicts that the running environment re-reads, which is what
makes "change a weight and it takes effect next batch" true. This contract is
written for that family. Jitted backends (MJX, Brax) bake parameters into
compiled closures, so the live-mutation promise cannot hold there.

Only :meth:`SimAdapter.discover_parameters`, :meth:`SimAdapter.get_parameter`
and :meth:`SimAdapter.set_parameter` are mandatory. Everything else has a
default that reports "not supported by this backend", so a new simulator can be
wired up in stages and the tools degrade gracefully instead of crashing.
:mod:`rlmcp.adapters.mjlab` is the reference implementation; the fakes in
``tests/conftest.py`` are the minimal one.

The contract is deliberately free of any particular task's vocabulary: there is
nothing here about terrain, object sets or goals. Capabilities like those belong
in an :class:`~rlmcp.core.extensions.Extension`, which adds commands to a run
without every other environment inheriting the concept.

Error convention
----------------

One convention, every method, so a caller never guesses what a return means:

* Capability absent: raise :class:`NotSupported`. Never stand in an empty
  ``{}``/``None`` for a capability the backend does not have -- the controller
  catches :class:`NotSupported` and tells the agent which tool is missing.
* Capability present but the call failed: raise, with a message naming the key
  or resource involved (``KeyError`` for a name that does not exist,
  ``ValueError`` for a value that cannot be applied). Failures are exceptions,
  never falsy returns.
* Setters (:meth:`SimAdapter.set_parameter`,
  :meth:`RunnerAdapter.set_hyperparameter`) raise on failure and return
  ``True`` on success. The ``bool`` return is load-bearing: the parameter
  registry treats a falsy setter return as "not applied" and reports
  ``applied: false`` without updating its schema, so a raise-only setter that
  falls off the end -- implicitly returning ``None`` -- would report every
  *successful* write as not applied. Return ``True`` explicitly.
* ``{}`` from :meth:`SimAdapter.summary_metrics`,
  :meth:`SimAdapter.get_env_state` and :meth:`RunnerAdapter.runner_metrics`
  means "nothing to report right now", which is not a failure: these are
  periodic best-effort reads and an env with no metrics this tick is healthy.
  That is exactly why a genuine failure must raise instead of returning ``{}``.

Servicing contract
------------------

Two hooks drive rlmcp from inside a training loop, and a backend integration
must arrange both:

* ``RlMcp.on_step()`` -- once per vectorised environment step. Feeds
  in-flight trace and video recording jobs; near-free when nothing records.
* ``RlMcp.service(iteration=...)`` -- once per learning-iteration boundary,
  after the policy update. Applies parameter changes, answers agent commands,
  publishes status, honours pause. All mutations land here, so the boundary
  must be a point where the environment is not mid-rollout.

The mjlab wrapper (:class:`~rlmcp.adapters.mjlab.env_wrapper.RlMcpEnvWrapper`)
calls ``on_step()`` from its ``step()`` override and finds the iteration
boundary by monkey-patching ``logger.log`` on the attached rsl_rl runner --
rsl_rl calls that exactly once per iteration, after the update, and offers no
public hook there. With no runner attached it falls back to servicing every
``service_every_steps`` env steps. Stopping rides the same mechanism: the loop
polls ``RlMcp.should_stop()`` at each service point and unwinds (the mjlab
wrapper raises ``TrainingStopped``), because
:meth:`RunnerAdapter.request_stop` is advisory only -- see its docstring.

Trace channel vocabulary
------------------------

:meth:`SimAdapter.sample_state` returns one step's signals keyed by the
``CHANNEL_*`` constants below. The names are the contract:
:func:`rlmcp.core.diagnostics.analyze_trace` looks channels up by exactly
these strings, so a backend that invents its own spelling records traces that
diagnose as empty. Per-step shapes below are per env (the recorder stacks them
into ``(n_steps, width)``); a channel the backend cannot provide is simply
omitted and its diagnostics section is skipped.

=========================  =============================  ====================
Channel                    Per-step shape / units         Diagnostics section
=========================  =============================  ====================
CHANNEL_JOINT_POS          (n_joints,) rad                smoothness (jerk);
                                                          joint labels
CHANNEL_JOINT_VEL          (n_joints,) rad/s              smoothness (vel and
                                                          accel RMS, chatter)
CHANNEL_JOINT_ACC          (n_joints,) rad/s^2            plots only
CHANNEL_JOINT_TORQUE       (n_joints,) N*m                effort
CHANNEL_ACTION             (action_dim,) raw policy out   smoothness (rates)
CHANNEL_COMMAND            (2..3,) [vx, vy(, wz)]         tracking, against
                           m/s, m/s, rad/s                base_{lin,ang}_vel
CHANNEL_BASE_LIN_VEL       (3,) m/s, base frame           tracking
CHANNEL_BASE_ANG_VEL       (3,) rad/s, base frame         tracking (yaw);
                                                          posture
CHANNEL_BASE_POS           (3,) m, world frame            posture (height)
CHANNEL_PROJECTED_GRAVITY  (3,) unit gravity, base frame  posture (tilt)
CHANNEL_FOOT_CONTACT       (n_feet,) 0/1 flags            gait
CHANNEL_REWARD             (1,) per-step reward           reward
=========================  =============================  ====================

``CHANNEL_COMMAND`` is a *plane-velocity* command by definition: the tracking
section subtracts it from the measured base velocity component-for-component.
A command term that is not one (a motion target, a goal pose) must be recorded
under another name -- the mjlab adapter uses ``command_<term>`` -- so it still
lands in the trace without being misread as a velocity. ``CHANNEL_TIME`` is
appended by the trace recorder itself; adapters never emit it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from rlmcp.core.parameters.spec import ParameterSpec

# Trace channels: the vocabulary shared by sample_state() producers and
# rlmcp.core.diagnostics. See the module docstring for shapes, units and
# which diagnostics section reads each one.
CHANNEL_TIME = "time"
CHANNEL_JOINT_POS = "joint_pos"
CHANNEL_JOINT_VEL = "joint_vel"
CHANNEL_JOINT_ACC = "joint_acc"
CHANNEL_JOINT_TORQUE = "joint_torque"
CHANNEL_ACTION = "action"
CHANNEL_COMMAND = "command"
CHANNEL_BASE_LIN_VEL = "base_lin_vel"
CHANNEL_BASE_ANG_VEL = "base_ang_vel"
CHANNEL_BASE_POS = "base_pos"
CHANNEL_PROJECTED_GRAVITY = "projected_gravity"
CHANNEL_FOOT_CONTACT = "foot_contact"
CHANNEL_REWARD = "reward"

#: Every channel an adapter may emit from sample_state (CHANNEL_TIME excluded:
#: the recorder adds it). Custom channels beyond these are recorded and plotted
#: but no diagnostics section reads them.
TRACE_CHANNELS = (
    CHANNEL_JOINT_POS,
    CHANNEL_JOINT_VEL,
    CHANNEL_JOINT_ACC,
    CHANNEL_JOINT_TORQUE,
    CHANNEL_ACTION,
    CHANNEL_COMMAND,
    CHANNEL_BASE_LIN_VEL,
    CHANNEL_BASE_ANG_VEL,
    CHANNEL_BASE_POS,
    CHANNEL_PROJECTED_GRAVITY,
    CHANNEL_FOOT_CONTACT,
    CHANNEL_REWARD,
)


class NotSupported(RuntimeError):
  """Raised when a backend does not implement an optional capability."""


class SimAdapter(ABC):
  """Interface to a vectorised training environment."""

  # Required.

  @abstractmethod
  def discover_parameters(self) -> list[ParameterSpec]:
    """Inspect the environment and describe every tunable parameter."""

  @abstractmethod
  def get_parameter(self, key: str) -> Any:
    """Read a parameter's live value.

    Raises ``KeyError`` (naming the key and, ideally, what exists) when the
    key is unknown.
    """

  @abstractmethod
  def set_parameter(self, key: str, value: Any) -> bool:
    """Mutate a parameter in place, taking effect on the next step.

    Returns ``True`` on success -- explicitly, per the module's error
    convention: the parameter registry reads a falsy return as "not applied".
    A write that cannot land raises instead (``KeyError`` for an unknown key,
    ``ValueError`` for a value the leaf refuses), with a message the agent can
    act on; it never returns ``False`` for a failure it can explain.
    """

  # Optional: introspection.

  def num_envs(self) -> int:
    raise NotSupported("num_envs")

  def step_dt(self) -> float:
    raise NotSupported("step_dt")

  def joint_names(self) -> list[str]:
    raise NotSupported("joint_names")

  def max_episode_length(self) -> float | None:
    """Episode length cap in steps, used to normalise survival metrics."""
    return None

  def sample_state(self, env_id: int = 0) -> dict[str, np.ndarray]:
    """One step's worth of per-env signals, for trace recording.

    Keys come from the trace channel vocabulary (the ``CHANNEL_*`` constants;
    see the module docstring for shapes, units and which diagnostics section
    consumes each channel). Provide what the backend has and omit the rest --
    but never emit a known channel name with different semantics, and in
    particular emit :data:`CHANNEL_COMMAND` only for a plane-velocity command.
    """
    raise NotSupported("sample_state")

  def trace_labels(self) -> dict[str, list[str]]:
    """Component names for the channels :meth:`sample_state` emits.

    The controller attaches these to every recorded trace: diagnostics names
    offending components with them ("left_knee buzzes", not "joint[7]"), and
    plots use them as legends. Keys are channel names; each list's length
    matches that channel's width. Derive them from the live environment --
    a wrong label misdirects the agent reading the diagnosis, so when a
    component's meaning is unknown, number it honestly instead.

    Graceful default: ``{}`` -- components appear indexed, nothing breaks.
    """
    return {}

  def reward_terms(self) -> dict[str, float]:
    """What each reward term paid on the last step, by name.

    The single most useful thing a task can be asked, and the one a curve
    cannot answer: total reward went up, *because of which term?* The failure
    it exists to catch is in this project's own worked example -- a task where
    standing still paid about 2600x what making progress paid. That is
    invisible in the code, invisible in the total, and obvious the moment the
    terms are put side by side.

    Values are per-term contributions averaged over the batch, in whatever
    units the environment adds them in -- weighted, because the question is
    what each term is *paying*, not what its raw function returned. A backend
    that scales rewards by ``dt`` may report the unscaled rate, since a factor
    common to every term changes no comparison between them.

    Raises :class:`NotSupported` where terms are not a thing the environment
    has: a hand-written env with one scalar reward has nothing to break down,
    and saying so is more useful than a dict with one entry called "reward".
    """
    raise NotSupported("reward_terms")

  def termination_terms(self) -> dict[str, float]:
    """Which termination terms fired on the last step, as a fraction of envs.

    The other half of "why did the episode end". An env that terminates on
    step 1 and one that never terminates at all are both broken, both produce
    training curves, and neither says so anywhere else.

    ``0.0`` for a term that did not fire is meaningful and should be reported;
    an empty dict means the environment has no named termination terms.
    """
    raise NotSupported("termination_terms")

  def summary_metrics(self) -> dict[str, float]:
    """Cheap always-on scalars describing the current batch.

    ``{}`` means "nothing to report", not failure -- see the module's error
    convention. Omit any metric whose meaning cannot be verified rather than
    publishing a number computed from the wrong signal.
    """
    return {}

  def reset_envs(self, env_ids: Sequence[int] | None = None) -> dict[str, Any]:
    """Start fresh episodes in some or all environments.

    ``env_ids`` is ``None`` for every environment, or the subset to restart;
    the controller resolves ``--where`` into that subset through the extension
    registry, so a caller can restart only the environments on one part of the
    task without this method learning what that part is.

    Called from :meth:`~rlmcp.core.controller.RlMcp.service`, i.e. between
    steps, like every other mutation. The observation the training loop is
    holding was computed before the reset and is one step stale; the next step
    recomputes it from the restarted episodes, which is the same ordering a
    termination-driven reset already has.

    Returns a dict describing what happened -- at minimum ``num_reset`` -- and
    raises :class:`NotSupported` on a backend with no reset path, which is the
    honest default: an environment nobody can restart on demand should say so
    rather than silently do nothing.
    """
    raise NotSupported("reset_envs")

  def get_env_state(self) -> dict[str, Any]:
    """Environment-side state worth carrying through a checkpoint.

    ``{}`` means "nothing to save", not failure.
    """
    return {}

  def set_env_state(self, state: dict[str, Any]) -> None:
    return None

  # Optional: growing the reward function.

  def add_reward_term(
      self,
      name: str,
      func: Any,
      weight: float,
      params: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    """Append a reward term to the running environment.

    The counterpart of :meth:`set_parameter` for the case where no existing
    weight expresses what the agent wants: a term the task never had, scoring
    from the next batch on. ``func`` is already a callable -- compiling the
    agent's source and deciding whether to trust it happens above this layer,
    in :mod:`rlmcp.core.reward_source` and the controller.

    An implementation must be all-or-nothing: a term that cannot be installed
    leaves the manager exactly as it was, because the alternative is a run
    that dies on its next batch inside the reward manager. Validate first --
    including a trial call -- then mutate.

    Returns a dict describing the installed term (at minimum ``name`` and
    ``index``), and raises :class:`NotSupported` on a backend whose reward
    function is fixed at construction.
    """
    raise NotSupported("add_reward_term")

  def capture_env_terms(self) -> dict[str, Any]:
    """Snapshot the terms that define what a policy trained against.

    The rewards it was paid, the observations it saw and the actions it
    emitted -- with each term's implementation captured as *source*, not as an
    import path, because an import path is only as good as the package still
    being installed at the version the run used.

    Taken once at run start and kept in the session, so that a checkpoint can
    be paired with the environment it trained under after the training process
    is gone. Terminations and events are out of scope: they shape training
    rather than the policy's interface.

    Must not raise for a manager that is absent or oddly shaped -- a capture is
    a convenience and cannot be allowed to stop a run. Report what could not be
    read in the payload's ``problems`` list instead.

    Returns ``{}`` on a backend with nothing to capture.
    """
    return {}

  # Optional: rendering.

  def render(self, env_id: int = 0) -> np.ndarray:
    """Return an RGB frame of ``env_id`` with shape (H, W, 3).

    Raises :class:`NotSupported` when no renderer can exist here (and the
    message should say why -- headless install, no memory for a context).
    """
    raise NotSupported("render")

  def renderer_ready(self) -> bool:
    """Whether a renderer already exists, so a frame costs no construction.

    The controller reports this in ``status`` as ``renderer_built``, telling
    the agent whether a screenshot is instant or will pay a one-off renderer
    build (which on a full GPU can fail). Purely informational -- ``render()``
    decides for itself whether it can produce a frame.

    Graceful default: ``False``.
    """
    return False

  def open_live_view(self, server: Any, realtime: bool = False,
                     buffer_seconds: float = 4.0) -> Any:
    """Mirror this environment into a viser server, for a live browser view.

    The counterpart of :meth:`render` for watching rather than recording:
    where ``render`` produces pixels for a clip, this hands the browser the
    scene once and then feeds it state, which is why it needs no render
    context and works on a headless machine.

    ``server`` is a live :class:`viser.ViserServer` owned by
    :class:`~rlmcp.core.live_view.LiveView`; the backend adds its geometry to
    it and returns a *scene handle* with two methods:

    * ``update(env_id: int) -> None`` -- take the current state of one
      environment. It must be a copy, never a simulation step or a reset.
    * ``close() -> None`` -- release whatever the handle holds. The server
      itself is closed by the caller.
    * ``describe() -> dict`` -- optional; whatever the backend can say about
      its own playback, reported under ``playback`` in ``status``.

    ``realtime`` picks between the two things "watch the run" can mean, and
    changes what ``update`` is for:

    * ``False`` -- push the state to the browser now. Called at a bounded frame
      rate; the motion in the tab runs at the *run's* pace, which for training
      is faster than life.
    * ``True`` -- record this step into a buffer holding ``buffer_seconds`` of
      sim time, and play the window back at 1x once it is full. ``update`` is
      then called on *every* environment step, because a window with steps
      missing is not the run's motion; it must therefore be cheap enough for
      the training loop -- no device synchronisation per step. What plays the
      window back is the backend's business (mjlab drives mjlab's own viewer),
      which is why none of it is in the core.

    Raise :class:`NotSupported` when the backend has no scene to mirror -- a
    live view is a nice-to-have, and a run whose backend cannot serve one
    should keep every other capability.
    """
    raise NotSupported(
        "open_live_view: this backend cannot serve a live 3-D view of its "
        "environment. `rlmcp video` still records clips of it."
    )


class RunnerAdapter(ABC):
  """Interface to an RL training runner."""

  @abstractmethod
  def discover_hyperparameters(self) -> list[ParameterSpec]:
    """Describe algorithm and optimiser hyperparameters."""

  @abstractmethod
  def set_hyperparameter(self, key: str, value: Any) -> bool:
    """Update a hyperparameter live.

    Same setter convention as :meth:`SimAdapter.set_parameter`: raise on
    failure (``KeyError`` naming an unknown key), return ``True`` on success
    -- the registry reads a falsy return as "not applied".
    """

  def get_hyperparameter(self, key: str) -> Any:
    raise NotSupported("get_hyperparameter")

  def current_iteration(self) -> int:
    return 0

  def save_checkpoint(self, path: str, infos: dict[str, Any] | None = None) -> str | None:
    raise NotSupported("save_checkpoint")

  def load_checkpoint(self, path: str) -> dict[str, Any]:
    raise NotSupported("load_checkpoint")

  def runner_metrics(self) -> dict[str, float]:
    """Scalars the runner already tracks. ``{}`` means nothing to report."""
    return {}

  def request_stop(self) -> bool:
    """Advisory only: tell the runner a stop was requested, if it can listen.

    Stopping does not happen here. It is delivered by the servicing contract
    (module docstring): the controller sets its stop flag, the training loop
    polls ``RlMcp.should_stop()`` at the next service point and unwinds --
    the mjlab wrapper by raising ``TrainingStopped``. Implement this only for
    a runner with a native stop mechanism this call can arm, and return
    ``True`` only when one was armed. The controller ignores the return value
    and swallows :class:`NotSupported`, so the default -- not supported -- is
    the honest one for runners like rsl_rl that have no such hook.
    """
    raise NotSupported("request_stop")
