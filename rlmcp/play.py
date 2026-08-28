"""``rlmcp play``: watch a task, as a clip or in a viewer.

Usually that means watching a *saved checkpoint*, which is what the options
below are shaped around. But a task being written has no checkpoint yet, and
looking at it is the cheapest way to find out that it terminates on the first
step or that the robot spawns inside the floor. ``--policy zero`` and
``--policy random`` open the same session with no weights at all:

    rlmcp play --task Mjlab-Velocity-Flat-Unitree-G1 --policy zero --mode viser

Nothing else changes. The env is built from the task's own ``play`` config, the
session is rlmcp-wrapped exactly as a training run is, and ``load_policy``
still works -- so once a checkpoint exists you swap it into the session you are
already watching, keeping the conditions and the camera.

Everything else in rlmcp talks to a *live* trainer. That leaves a gap: a run
ends -- normally, or because its falsifier fired -- and the one thing nobody can
do any more is look at it. The evidence that would settle the question is a
checkpoint on disk and no way to play it.

This is that way. It rebuilds the environment, restores the conditions the
policy was trained under (see :mod:`rlmcp.core.replay` for why that matters more
than it sounds), loads the weights, and then either records a clip or opens a
viewer:

* ``--mode video`` renders offscreen to an mp4. No display needed, so it works
  over ssh, in a cron job, and on a machine whose GPU is busy training -- pass
  ``--device cpu`` and it will not touch the card.
* ``--mode native`` opens MuJoCo's own viewer, for looking closely at a robot
  that is in front of you.
* ``--mode viser`` serves a viewer in the browser, for a robot that is not.

Neither viewer is required to be installed: a build without them says so and
points at ``--mode video``, rather than failing on import.

The environment is wrapped exactly as a training run is, so a play session is
itself steerable -- ``rlmcp --session <play dir> run <command>`` changes the
task under the policy while you watch it. It writes itself down as a *play*
session, which session discovery skips, so it never becomes the answer to
"the latest session here".

Three of those controls are what make watching a policy an investigation rather
than a screening:

* ``rlmcp reset-envs`` starts fresh episodes, so you can see the opening of a
  behaviour again instead of waiting for the robot to fail into one. That verb
  is core, not play -- it is just as useful mid-training.
* ``rlmcp stop`` ends the session the way closing the window does: the viewer
  unwinds, the session records its end, and nobody is shown a traceback for
  having asked.
* ``rlmcp run load_policy checkpoint=<other.pt>`` swaps the acting policy
  between steps. The point is comparison: the conditions you restored and the
  camera you set up survive, and only the weights change. See
  :class:`PolicySwap` for what happens when the new checkpoint was trained
  somewhere else on the ladder.

Nothing above the ``run_play`` entry point imports a simulator, torch or an
encoder, so ``rlmcp --help`` stays as cheap as it has always been.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from rlmcp.core.extensions import Extension
from rlmcp.core.replay import (
    Conditions,
    apply_conditions,
    read_conditions,
    read_events,
    with_overrides,
)

MODES = ("video", "native", "viser")
POLICIES = ("checkpoint", "zero", "random")

TASK_PACKAGES_ENV = "RLMCP_TASK_PACKAGES"
"""Comma-separated packages to import before reading the task registry.

``--task-package`` is the explicit form and is what ``rlmcp train`` takes. This
is the same thing said once for a whole shell, which is what you want when
every run in a project draws its tasks from the same package.
"""

# Checkpoints are conventionally model_<iteration>.pt / model_final_<n>.pt, but
# nothing here depends on that name -- only on the suffix.
_CHECKPOINT_GLOB = "*.pt"

# How far above a given path to look for them. A play session sits three deep
# inside a run (<run>/rlmcp/play/<stamp>); beyond that we would start climbing
# out of the run and into whatever else the log root holds.
_SEARCH_DEPTH = 5


class PlayError(RuntimeError):
  """Something about the request cannot work, stated in terms the caller can fix."""


@dataclass
class PlayConfig:
  """Everything ``play`` needs, with defaults that suit the common case.

  The common case is "the run just ended and I want to see it": point at a run
  directory, take the last checkpoint, replay the stage it died in, write eight
  seconds of mp4 next to the run's other evidence.
  """

  checkpoint: str = ""
  """A checkpoint file, or a run/session directory to take the latest from."""
  policy: str = "checkpoint"
  """Where the actions come from: ``checkpoint``, ``zero`` or ``random``.

  ``zero`` and ``random`` need no weights, so they need no checkpoint and no
  session to have produced one -- which is what makes them the way to look at a
  task that does not train yet. Both require ``--task``: with no checkpoint
  there is nothing to infer it from.
  """
  task: str = ""
  """Defaults to the task recorded in the session this checkpoint belongs to."""
  mode: str = "video"
  seconds: float = 8.0
  num_envs: int = 1
  device: str = "cuda:0"
  out: str = ""
  """Video path. Defaults into the training session's artifacts directory."""
  fps: Optional[int] = None
  """Defaults to the environment's own control rate, so the clip runs in real time."""
  stage: str = ""
  """Curriculum stage to restore. Defaults to the last one the run entered."""
  replay: bool = True
  """Restore trained-under conditions at all. Off gives the task's play config."""
  overrides: Dict[str, Any] = field(default_factory=dict)
  """Parameter edits applied after the replay, so they win."""
  task_package: List[str] = field(default_factory=list)
  render_width: int = 960
  render_height: int = 720
  extra_envs: Optional[int] = 0
  """Neighbouring envs composited into the frame. 0 -- an unambiguous close-up."""
  session_dir: str = ""
  """Where the play session publishes itself. Defaults under the run directory."""
  allow_partial: bool = False
  """Render even when some conditions could not be restored. Rarely what you want."""
  quiet: bool = False


# Finding things.


def find_checkpoint(target: Path | str) -> Path:
  """Resolve a file, a run directory or a session directory to one checkpoint.

  Picking "the latest" by iteration rather than by mtime, because a run that
  saved ``model_final_4375.pt`` and then had ``model_900.pt`` copied back in
  for comparison should still play the one it ended on.
  """
  path = Path(target).expanduser()
  if path.is_file():
    return path
  if not path.exists():
    raise PlayError(f"No such checkpoint or run directory: {path}")

  # Checkpoints live in the run directory, but the path in hand is often
  # something inside it -- the session (…/rlmcp), or a play session nested
  # deeper still. Walk outwards until a directory has some, so pointing at any
  # part of a run finds the run's checkpoints.
  found: List[Path] = []
  for directory in [path, *path.parents][:_SEARCH_DEPTH]:
    found = [p for p in directory.glob(_CHECKPOINT_GLOB) if p.is_file()]
    found += list((directory / "checkpoints").glob(_CHECKPOINT_GLOB))
    if found:
      break
  if not found:
    raise PlayError(
        f"No .pt checkpoints in {path} or the directories above it. Point "
        "--checkpoint at the file itself if it lives somewhere else."
    )
  return max(found, key=lambda p: (checkpoint_iteration(p), p.stat().st_mtime))


def checkpoint_iteration(path: Path) -> int:
  """The iteration in ``model_4300.pt`` / ``model_final_4375.pt``; -1 if unnamed."""
  digits = [part for part in path.stem.replace("-", "_").split("_") if part.isdigit()]
  return int(digits[-1]) if digits else -1


def session_for(checkpoint: Path) -> Optional[Path]:
  """The rlmcp session a checkpoint belongs to, if it has one.

  A trainer's checkpoints sit in the run directory and the session is
  ``<run>/rlmcp``; a checkpoint saved through ``rlmcp checkpoint`` sits one
  deeper, inside ``<session>/checkpoints``.
  """
  for candidate in (
      checkpoint.parent / "rlmcp",
      checkpoint.parent.parent / "rlmcp",
      checkpoint.parent,
      checkpoint.parent.parent,
  ):
    if (candidate / "session.json").exists():
      return candidate
  return None


def session_info(session_dir: Optional[Path]) -> Dict[str, Any]:
  """What the session recorded about itself, or {} if there is nothing to read."""
  if session_dir is None:
    return {}
  import json

  try:
    info = json.loads((session_dir / "session.json").read_text())
  except (OSError, ValueError):
    return {}
  return info if isinstance(info, dict) else {}


def task_for(session_dir: Optional[Path]) -> str:
  """The task id the session recorded, or '' if there is nothing to read."""
  return str(session_info(session_dir).get("task") or "")


def stage_at_iteration(session_dir: Optional[Path], iteration: int) -> str:
  """The curriculum stage a run was on at ``iteration``; '' if it had none.

  A checkpoint's file name carries the iteration it was saved at, and the
  session's log carries the iteration at which each stage was entered. Put
  together they answer the only question a policy swap has to ask: was the
  policy about to start acting trained under the conditions currently applied?

  A checkpoint whose name says nothing (``iteration`` below zero) reads as the
  end of the run, which is where an unnamed checkpoint usually comes from.
  """
  if session_dir is None:
    return ""
  current = ""
  for event in read_events(session_dir):
    if event.get("kind") != "curriculum_stage":
      continue
    entered = int(event.get("iteration") or 0)
    if iteration >= 0 and entered > iteration:
      break
    name = str(event.get("to") or "")
    if name:
      current = name
  return current


def packages_to_import(cfg: PlayConfig) -> List[str]:
  """Packages that must be imported before the task registry knows the task.

  A project's tasks register on import, exactly as the simulator's own do --
  the same contract ``rlmcp train --task-package`` uses. Nothing here guesses
  at a package name or goes looking for one on disk: a replay months later
  imports what it was told to import, and says so plainly when that is nothing.
  """
  found = list(cfg.task_package)
  for name in os.environ.get(TASK_PACKAGES_ENV, "").split(","):
    name = name.strip()
    if name and name not in found:
      found.append(name)
  return found


# Swapping the policy.


class SwappablePolicy:
  """The callable in the rollout, pointing at whichever policy is current.

  A viewer is handed a policy once, at construction, and then calls it forever.
  So the thing it is handed is this: an object that forwards, and that can be
  made to forward somewhere else. Nothing downstream is told about the change,
  because from the loop's point of view nothing changed -- the same object is
  still being called with the same observations.

  The assignment happens inside a command handler, which the controller runs at
  a service boundary between steps, so no step ever sees half a swap.
  """

  def __init__(self, policy: Any, checkpoint: Path | str):
    self.policy = policy
    self.checkpoint = Path(checkpoint)
    self.swaps: List[Dict[str, Any]] = []

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    return self.policy(*args, **kwargs)

  def __getattr__(self, name: str) -> Any:
    # Reached only for names this object does not have, so a viewer that pokes
    # at the policy itself still finds what it is looking for.
    try:
      policy = self.__dict__["policy"]
    except KeyError:  # pragma: no cover - only during a half-built copy.
      raise AttributeError(name) from None
    return getattr(policy, name)

  def swap(self, policy: Any, checkpoint: Path | str) -> Path:
    """Point at a new policy; returns the checkpoint that was acting before."""
    previous = self.checkpoint
    self.policy = policy
    self.checkpoint = Path(checkpoint)
    return previous

  def __repr__(self) -> str:  # pragma: no cover - debugging aid.
    return f"SwappablePolicy({self.checkpoint.name})"


class UntrainedPolicy:
  """Actions for a task that has no policy yet: zeros, or samples.

  Deliberately shaped like an inference policy -- called with the observation,
  returns a batch of actions -- so everything downstream is unchanged.
  :class:`SwappablePolicy` wraps it like any other, which is what lets
  ``load_policy`` replace it with real weights mid-session without restarting:
  the env, the restored conditions and the camera all survive.
  """

  def __init__(self, action_shape: Tuple[int, ...], device: str, mode: str = "zero"):
    self.action_shape = action_shape
    self.device = device
    self.mode = mode

  def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
    import torch

    if self.mode == "random":
      # Uniform in [-1, 1]: mjlab action terms are scaled and offset from a
      # normalised action, so this is the span a policy would emit, not a
      # guess at joint limits.
      return torch.rand(self.action_shape, device=self.device) * 2.0 - 1.0
    return torch.zeros(self.action_shape, device=self.device)

  def __repr__(self) -> str:  # pragma: no cover - debugging aid.
    return f"UntrainedPolicy({self.mode})"


def _untrained_policy(cfg: PlayConfig, vec_env: Any) -> UntrainedPolicy:
  """Size an untrained actor from the env rather than from the task config."""
  num_actions = None
  for source in (
      lambda: vec_env.num_actions,
      lambda: vec_env.unwrapped.action_manager.total_action_dim,
      lambda: vec_env.action_space.shape[-1],
  ):
    try:
      num_actions = int(source())
      break
    except Exception:
      continue
  if not num_actions:
    raise PlayError(
        "Could not tell how many actions this task takes, so there is nothing "
        "to send it. This is a task-side problem: check the action manager."
    )
  return UntrainedPolicy((cfg.num_envs, num_actions), cfg.device, cfg.policy)


class PolicySwap(Extension):
  """``load_policy``: change the acting policy without restarting the session.

  Registered by ``rlmcp play`` against its own controller rather than living in
  the core, because the thing being swapped only exists in a play session: an
  inference policy inside a rollout that no runner owns. A training run's
  weights belong to its runner, and putting different ones there is
  ``load_checkpoint``, which is a different operation with different
  consequences. An :class:`~rlmcp.core.extensions.Extension` is the sanctioned
  way to contribute a verb the core does not know, and it is the right shape
  here rather than a play-owned registration: going through the extension
  registry is what makes ``load_policy`` reachable from ``rlmcp run``, from MCP
  ``run_command`` and from a curriculum stage at once, and what puts the acting
  checkpoint into ``rlmcp status`` for free.

  **Conditions.** A checkpoint from another run may have been trained at
  another rung of the curriculum, and loading its weights without its
  conditions reproduces exactly the bug :mod:`rlmcp.core.replay` exists to
  prevent -- a good policy shown failing at a task it was never asked to do.
  Three answers were possible; this one holds the conditions still and says so
  loudly, because the reason to swap a policy mid-session is to compare it
  against the last one, and a comparison whose environment changed underneath
  it measures nothing. Re-restoring is one argument away (``replay=true``), and
  every response carries the stage the checkpoint trained at next to the stage
  the environment is in, so the mismatch cannot be missed by anything reading
  the payload.

  Everything the swap needs is injected, so the command is exercisable without
  a simulator: ``load`` turns a path into a policy, ``probe`` makes a policy act
  once before anything depends on it, ``restore`` replays a set of conditions.
  """

  name = "play_policy"

  def __init__(
      self,
      holder: SwappablePolicy,
      load: Callable[[Path], Any],
      session_dir: Optional[Path | str] = None,
      stage: str = "",
      restore: Optional[Callable[[Conditions], Dict[str, Any]]] = None,
      probe: Optional[Callable[[Any], None]] = None,
  ):
    super().__init__(env=None)
    self.holder = holder
    self.session_dir = Path(session_dir) if session_dir else None
    self.stage = stage or ""
    self._load = load
    self._restore = restore
    self._probe = probe

  def available(self) -> bool:
    return True

  def commands(self) -> Dict[str, Callable[..., Any]]:
    return {
        "load_policy": self.cmd_load_policy,
        "list_policies": self.cmd_list_policies,
    }

  def describe(self) -> Dict[str, Any]:
    return {
        "checkpoint": str(self.holder.checkpoint),
        "iteration": checkpoint_iteration(self.holder.checkpoint),
        "stage": self.stage or None,
        "swaps": len(self.holder.swaps),
    }

  # Commands.

  def cmd_load_policy(
      self, checkpoint: str, replay: bool = False, stage: str = ""
  ) -> Dict[str, Any]:
    """Act with a different checkpoint's weights from the next step onwards.

    ``checkpoint`` is a .pt file, or a run directory to take the latest from.
    ``replay`` restores the conditions *that* checkpoint was trained under
    before swapping, which is what you want when it comes from another run and
    not what you want when you are comparing two policies on one environment.
    ``stage`` picks which of its stages to restore, and means nothing without
    ``replay``.

    Refuses without touching the running policy if the file is missing, if the
    weights do not fit this task, or if they load but cannot produce an action
    here. A swap that half-applies is worse than one that does not happen.
    """
    path = find_checkpoint(checkpoint)
    origin = session_for(path)
    report = self._conditions_report(path, origin)

    # Load and prove the fit first: until the last two statements of this
    # method, nothing the loop touches has moved.
    policy = self._load(path)
    self._fit_or_refuse(policy, path)

    if replay:
      wanted = stage or report["checkpoint_stage"] or ""
      report.update(self._replay_from(origin, wanted))
    elif stage:
      raise PlayError(
          "stage means nothing without replay=true: it names which of the "
          "checkpoint's own stages to restore, and without replay no "
          "conditions are restored at all."
      )

    previous = self.holder.swap(policy, path)
    entry = {
        "checkpoint": str(path),
        "iteration": checkpoint_iteration(path),
        "previous_checkpoint": str(previous),
        "trained_session": str(origin) if origin else None,
        "conditions": report,
    }
    self.holder.swaps.append(entry)
    if self.context is not None:
      self.context.append_event("load_policy", entry)
    return {"loaded": True, **entry}

  def cmd_list_policies(self) -> Dict[str, Any]:
    """Checkpoints that can be loaded here: the acting one and its siblings."""
    current = self.holder.checkpoint
    found = {current}
    parent = current.parent
    for directory in (parent, parent / "checkpoints",
                      parent.parent, parent.parent / "checkpoints"):
      try:
        found.update(p for p in directory.glob(_CHECKPOINT_GLOB) if p.is_file())
      except OSError:
        continue
    return {
        "current": str(current),
        "checkpoints": [
            {
                "path": str(p),
                "iteration": checkpoint_iteration(p),
                "current": p == current,
            }
            for p in sorted(found, key=lambda p: (checkpoint_iteration(p), p.name))
        ],
    }

  # The conditions question.

  def _conditions_report(
      self, path: Path, origin: Optional[Path]
  ) -> Dict[str, Any]:
    """Where this checkpoint was trained, against where the environment is.

    Always present in the response, mismatch or not, and so is ``warning`` --
    null when there is nothing to say. An agent reading the payload should
    never have to infer from a missing key that the two agree.
    """
    trained = stage_at_iteration(origin, checkpoint_iteration(path))
    same_run = origin is not None and self.session_dir is not None and (
        Path(origin) == self.session_dir
    )
    known = origin is not None
    report: Dict[str, Any] = {
        "replayed": False,
        "applied_stage": self.stage or None,
        "checkpoint_stage": trained or None,
        "same_run": bool(same_run),
        # None, not False: "nobody can tell" is not the same answer as "they
        # disagree", and an agent deciding what to do next needs the difference.
        "match": (trained == self.stage) if known else None,
        "warning": None,
    }
    if not known:
      report["warning"] = (
          f"There is no rlmcp session beside {path.name}, so there is no way "
          "to tell what conditions it was trained under. The environment is "
          f"left at '{self.stage or '(no curriculum)'}'; if the policy looks "
          "broken, suspect that before suspecting the policy."
      )
    elif not report["match"]:
      report["warning"] = (
          f"{path.name} was trained at curriculum stage "
          f"'{trained or '(no curriculum)'}', and this environment is set up "
          f"for '{self.stage or '(no curriculum)'}'. The weights were loaded; "
          "the conditions were NOT changed, which is deliberate -- comparing "
          "two policies needs the environment to hold still. So this is a "
          "policy doing a task it was not trained on, which looks exactly like "
          "a bad policy. Pass replay=true to restore its own conditions "
          "instead."
      )
    return report

  def _replay_from(self, origin: Optional[Path], stage: str) -> Dict[str, Any]:
    """Restore the conditions the new checkpoint's own run was in.

    Failures to restore are reported, not raised: ``apply_conditions`` is
    best-effort by design, and a stage whose command no longer exists is worth
    saying out loud rather than worth abandoning the swap over.
    """
    if origin is None:
      raise PlayError(
          "replay=true needs the run this checkpoint came from, and there is "
          "no rlmcp session beside it. Load it without replay to keep the "
          "conditions this session already has."
      )
    if self._restore is None:
      raise PlayError("This session cannot restore conditions.")
    conditions = read_conditions(origin, stage or None)
    restored = self._restore(conditions)
    self.session_dir = Path(origin)
    self.stage = conditions.stage or ""
    return {
        "replayed": True,
        "applied_stage": self.stage or None,
        "match": True,
        "restored_parameters": len(restored.get("parameters") or {}),
        "restored_calls": len(restored.get("calls") or []),
        "restore_errors": list(restored.get("errors") or []),
        "warning": None,
    }

  def _fit_or_refuse(self, policy: Any, path: Path) -> None:
    """Make the new policy act once, while the old one is still driving.

    Weights can load into a runner and still be the wrong shape for the
    environment in front of them -- a task whose observation width changed
    loads clean and fails on the first step, deep inside a viewer, with the old
    policy already discarded. Acting once here turns that into a refusal.

    A probe that cannot run (an environment that will not hand over an
    observation) is not a reason to refuse; the swap goes ahead unproven.
    """
    if self._probe is None:
      return
    try:
      self._probe(policy)
    except Exception as exc:
      raise PlayError(
          f"{path.name} loaded, but could not produce an action for this "
          f"environment ({type(exc).__name__}: {exc}). Its observations or "
          "actions do not match the task running here. The policy in the loop "
          "is unchanged."
      ) from exc


# Playing.


def run_play(cfg: PlayConfig) -> Dict[str, Any]:
  """Build the environment, restore conditions, load the policy, and show it.

  Returns a result payload; in video mode it carries ``video_path``, which is
  what makes the CLI put the clip on screen.
  """
  if cfg.mode not in MODES:
    raise PlayError(f"Unknown mode '{cfg.mode}'. Choose one of: {', '.join(MODES)}")
  if cfg.policy not in POLICIES:
    raise PlayError(
        f"Unknown policy '{cfg.policy}'. Choose one of: {', '.join(POLICIES)}")

  untrained = cfg.policy != "checkpoint"
  if untrained:
    # No weights, so nothing to find a session or a task from. Nothing to
    # replay either, and that is the same request `--no-replay` makes: run at
    # the task's own play configuration. Saying so here rather than leaving
    # `replay` true keeps the missing-session warning for the case it was
    # written for -- a real checkpoint whose run is nowhere to be found. There
    # is no checkpoint here, so nothing is missing. `--set` overrides still
    # apply; they are the only steering a policy-free preview has.
    cfg = replace(cfg, replay=False)
    checkpoint = None
    session_dir = None
    task = cfg.task
    if not task:
      raise PlayError(
          f"--policy {cfg.policy} needs --task: with no checkpoint there is no "
          "session to read the task from."
      )
  else:
    checkpoint = find_checkpoint(cfg.checkpoint or Path.cwd())
    session_dir = session_for(checkpoint)
    task = cfg.task or task_for(session_dir)
    if not task:
      raise PlayError(
          f"Could not tell which task {checkpoint.name} was trained on: there is "
          "no session.json near it. Pass --task."
      )

  _choose_gl_backend(cfg)
  env, lab, agent_cfg, vec_env = _build_env(cfg, task, session_dir)

  conditions, restored = _restore_conditions(cfg, lab, session_dir)
  policy = SwappablePolicy(
      _untrained_policy(cfg, vec_env) if untrained
      else _load_policy(cfg, task, vec_env, checkpoint, agent_cfg),
      checkpoint or Path(cfg.policy),
  )
  # Registered against this session's own controller, so `load_policy` is one
  # more command a play session answers -- see PolicySwap for why it lives here
  # and not in the core.
  lab.add_extension(
      PolicySwap(
          holder=policy,
          load=lambda path: _load_policy(cfg, task, vec_env, path, agent_cfg),
          session_dir=session_dir,
          stage=conditions.stage,
          restore=lambda restored_conditions: apply_conditions(
              lab, restored_conditions
          ),
          probe=_policy_probe(vec_env),
      )
  )

  result: Dict[str, Any] = {
      "mode": cfg.mode,
      "policy": cfg.policy,
      "checkpoint": str(checkpoint) if checkpoint else None,
      "iteration": checkpoint_iteration(checkpoint) if checkpoint else -1,
      "task": task,
      "device": cfg.device,
      "num_envs": cfg.num_envs,
      "trained_session": str(session_dir) if session_dir else None,
      "play_session": str(lab.session.dir),
      "conditions": {
          "replayed": cfg.replay,
          "stage": conditions.stage or None,
          **restored,
      },
  }

  try:
    if cfg.mode == "video":
      result.update(_record(cfg, env, vec_env, policy, checkpoint, session_dir))
    else:
      result.update(_view(cfg, env, vec_env, policy))
  finally:
    try:
      vec_env.close()
    except Exception:
      pass
  if policy.swaps:
    # The checkpoint named at the top is the one this ran up with; say which
    # one it finished on rather than leaving that to be inferred.
    result["policy_swaps"] = policy.swaps
    result["final_checkpoint"] = str(policy.checkpoint)
  return result


def _stop_state(lab: Any, error: Optional[BaseException] = None) -> str:
  """Why a stepping loop ended, or '' if nobody asked it to end.

  Two ways a requested stop reaches the loop that was running the policy, and
  they must read the same afterwards. Usually it unwinds:
  :class:`~rlmcp.core.controller.SessionStopped` is raised out of the wrapper's
  ``step`` at a service boundary and ``error`` is that exception. But a viewer
  that catches everything inside its own loop swallows it and returns normally,
  and the reason a session ended should not depend on how thoroughly somebody
  else's loop catches things -- so with no exception in hand the controller's
  own flag is consulted instead.
  """
  if error is not None:
    return str(error) or "stop requested"
  if lab.should_stop():
    return lab.stop_reason or "stop requested"
  return ""


def _policy_probe(vec_env: Any) -> Optional[Callable[[Any], None]]:
  """A callable that makes a policy act once on this environment's observations.

  Returns None when the environment cannot be asked for one, in which case
  :meth:`PolicySwap._fit_or_refuse` has nothing to check and says so by
  swapping anyway -- a check that cannot run is not evidence of a problem.
  """
  getter = getattr(vec_env, "get_observations", None)
  if not callable(getter):
    return None

  def probe(policy: Any) -> None:
    import torch

    try:
      obs = getter()
    except Exception:
      return  # The environment will not answer right now; not the policy's fault.
    if isinstance(obs, tuple):
      obs = obs[0]
    with torch.no_grad():
      policy(obs)

  return probe


def _choose_gl_backend(cfg: PlayConfig) -> None:
  """Pick a GL backend before anything imports mujoco, because it is read once.

  Offscreen rendering wants EGL, which needs no display and is the only thing
  that works over ssh. The native viewer wants a real window, so EGL would make
  it fail with an error about the frame buffer rather than about the display.
  """
  os.environ.setdefault("MUJOCO_GL", "glfw" if cfg.mode == "native" else "egl")
  if cfg.device.startswith("cuda"):
    index = cfg.device.split(":")[-1] if ":" in cfg.device else "0"
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", index)


def _build_env(
    cfg: PlayConfig, task: str, session_dir: Optional[Path]
) -> Tuple[Any, Any, Any, Any]:
  import importlib

  import mjlab.tasks  # noqa: F401  (populates the task registry)

  for module in packages_to_import(cfg):
    try:
      importlib.import_module(module)
    except ImportError as exc:
      raise PlayError(
          f"Could not import task package '{module}': {exc}. It is on the "
          "PYTHONPATH of whatever trained this run, but not of this shell."
      ) from exc

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
  from mjlab.utils.torch import configure_torch_backends

  import rlmcp
  from rlmcp.session import PLAY_SESSION_KIND

  registered = list_tasks()
  if task not in registered:
    raise PlayError(
        f"Unknown task '{task}'. Nothing imported here registers it. Pass "
        f"--task-package <module> (repeatable) or set {TASK_PACKAGES_ENV}, "
        "naming the package whose import registers this task -- the same "
        "package `rlmcp train --task-package` was given. Registered here: "
        + (", ".join(registered) or "(nothing)")
    )

  configure_torch_backends()

  # play=True is the right starting point -- it turns off the randomisation
  # that exists to make training robust and would only make a clip noisy -- but
  # it is also rung zero of any curriculum, which _restore_conditions fixes.
  env_cfg = load_env_cfg(task, play=True)
  agent_cfg = load_rl_cfg(task)
  env_cfg.scene.num_envs = max(1, int(cfg.num_envs))
  env_cfg.viewer.width = cfg.render_width
  env_cfg.viewer.height = cfg.render_height
  if cfg.extra_envs is not None and hasattr(env_cfg.viewer, "max_extra_envs"):
    # Neighbouring envs composited into one frame read as duplicate objects,
    # which is the same trap `RlMcpEnvWrapper` warns about at startup. Off by
    # default here: a clip of one policy should show one policy.
    env_cfg.viewer.max_extra_envs = int(cfg.extra_envs)

  # render_mode stays None in every mode: rlmcp builds the offscreen renderer
  # on the first frame, which is the path the whole project renders through and
  # the only one that knows how to point at a chosen environment.
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device, render_mode=None)

  play_session = (
      Path(cfg.session_dir) if cfg.session_dir
      else _default_session(session_dir, cfg.mode)
  )
  env = rlmcp.wrap(
      env,
      session_dir=play_session,
      curriculum=None,  # A replay is not a ladder; nothing here promotes.
      task_id=task,
      session_kind=PLAY_SESSION_KIND,
      # A replay is already the clip. Progress clips exist so a *training* run
      # films itself; taking them here would film a rendering of a rendering.
      video_every=0,
  )
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  return env, env.rlmcp, agent_cfg, vec_env


def _default_session(trained_session: Optional[Path], mode: str) -> Path:
  stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
  root = trained_session if trained_session else Path.cwd() / "rlmcp_session"
  return root / "play" / f"{stamp}_{mode}"


def _restore_conditions(
    cfg: PlayConfig, lab: Any, session_dir: Optional[Path]
) -> Tuple[Conditions, Dict[str, Any]]:
  """Put the environment back where the checkpoint left it."""
  if not cfg.replay:
    conditions = with_overrides(Conditions(), cfg.overrides)
  elif session_dir is None:
    conditions = with_overrides(
        Conditions(
            warnings=(
                "No rlmcp session found next to this checkpoint, so there is "
                "nothing to replay: the environment runs at the task's own play "
                "configuration, which is rung zero of any curriculum.",
            )
        ),
        cfg.overrides,
    )
  else:
    conditions = with_overrides(
        read_conditions(session_dir, cfg.stage or None), cfg.overrides
    )

  restored = apply_conditions(lab, conditions)
  if restored["errors"] and not cfg.allow_partial:
    raise PlayError(_cannot_restore(restored))
  if not cfg.quiet:
    stage = conditions.stage or "(no curriculum)"
    print(
        f"[rlmcp-play] conditions: stage '{stage}', "
        f"{len(restored['parameters'])} parameters, {len(restored['calls'])} commands",
        flush=True,
    )
    for message in restored["warnings"] + restored["errors"]:
      print(f"[rlmcp-play]   ! {message}", flush=True)
  return conditions, restored


def _cannot_restore(restored: Dict[str, Any]) -> str:
  """Say what went wrong, and what to do about that particular thing.

  The advice matters more than the list. "Unknown command" and "the command no
  longer takes this argument" are different problems with different fixes, and
  a message that guesses at the wrong one sends whoever is reading it to check
  something that was never broken.
  """
  listed = "\n  ".join(restored["errors"])
  kinds = set(restored.get("error_kinds") or ())
  advice: List[str] = []
  if "missing_command" in kinds:
    advice.append(
        "A command that does not exist here means the package defining this "
        "task's vocabulary was not imported -- pass --task-package."
    )
  if "changed_command" in kinds:
    advice.append(
        "A command that no longer takes these arguments means the task's code "
        "has moved on since this run. There is no way to restore exactly what "
        "was trained; --allow-partial renders without that command, and --set "
        "<parameter>=<value> puts the underlying knob back by hand."
    )
  if "parameter" in kinds:
    advice.append(
        "A refused parameter is one this build of the task no longer has, or "
        "no longer accepts that value for."
    )
  if not advice:
    advice.append("--allow-partial renders anyway.")
  return (
      "Could not restore the conditions this checkpoint was trained under, so "
      "a clip would show the policy attempting a different task than the one "
      f"it learned:\n  {listed}\n" + "\n".join(advice)
  )


def _load_policy(
    cfg: PlayConfig, task: str, vec_env: Any, checkpoint: Path, agent_cfg: Any
) -> Any:
  import tempfile
  from dataclasses import asdict

  from mjlab.rl import MjlabOnPolicyRunner
  from mjlab.tasks.registry import load_runner_cls

  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  # The runner writes a log directory it will never use here; a temporary one
  # keeps a play from leaving an empty run beside the real ones.
  with tempfile.TemporaryDirectory(prefix="rlmcp-play-") as tmp:
    runner = runner_cls(vec_env, asdict(agent_cfg), tmp, cfg.device)
    try:
      runner.load(str(checkpoint))
    except Exception as exc:
      raise PlayError(
          f"Could not load {checkpoint.name} into a '{task}' runner: {exc}. "
          "A checkpoint only fits the task it was trained on -- if the task's "
          "observations or actions have changed since, this is that."
      ) from exc
  return runner.get_inference_policy(device=cfg.device)


def _record(
    cfg: PlayConfig,
    env: Any,
    vec_env: Any,
    policy: Any,
    checkpoint: Optional[Path],
    session_dir: Optional[Path],
) -> Dict[str, Any]:
  """Roll the policy out and encode what it did."""
  import imageio.v2 as imageio
  import numpy as np
  import torch

  from rlmcp.core.controller import SessionStopped

  lab = env.rlmcp
  step_dt = float(getattr(env.unwrapped, "step_dt", 0.02))
  steps = max(1, int(round(cfg.seconds / step_dt)))
  fps = int(cfg.fps or round(1.0 / step_dt))

  frames: List[Any] = []
  stopped = ""
  obs, _ = vec_env.reset()
  with torch.no_grad():
    for _ in range(steps):
      try:
        obs, _, _, _ = vec_env.step(policy(obs))
      except SessionStopped as stop:
        # `rlmcp stop` landed at a service boundary inside that step. The
        # frames already recorded are still evidence, so the clip is written
        # short rather than thrown away, and nobody is shown a traceback for
        # having asked the session to end.
        stopped = _stop_state(lab, stop)
        break
      try:
        frames.append(np.asarray(lab.sim.render(0)).astype(np.uint8))
      except Exception as exc:
        # A build with no offscreen renderer cannot make a clip, and finding
        # that out 200 steps in is worse than finding it out on the first one.
        raise PlayError(
            f"This environment could not render a frame ({type(exc).__name__}: "
            f"{exc}). --mode native or --mode viser still work if a viewer is "
            "installed; a headless machine needs an EGL-capable mujoco build."
        ) from exc

  if not frames and stopped:
    raise PlayError(f"Stopped before any frame was rendered ({stopped}).")
  if not frames:
    raise PlayError(f"Rendered no frames in {steps} steps.")

  out = Path(cfg.out) if cfg.out else _default_out(checkpoint, session_dir, cfg)
  out.parent.mkdir(parents=True, exist_ok=True)
  # +faststart puts the moov atom in front of the media data. Without it a
  # player has to fetch the whole file before showing anything, so the clip
  # silently never starts in a preview pane -- the file is fine, it just does
  # not play where anyone looks at it.
  imageio.mimwrite(
      str(out),
      frames,
      fps=fps,
      codec="libx264",
      macro_block_size=1,
      output_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"],
  )
  if not cfg.quiet:
    print(f"[rlmcp-play] wrote {out} ({len(frames)} frames @ {fps} fps)", flush=True)
  return {
      "video_path": str(out),
      "num_frames": len(frames),
      "fps": fps,
      "seconds": round(len(frames) / max(fps, 1), 2),
      "size_mb": round(out.stat().st_size / 1e6, 2),
      "metrics": _headline(env),
      "stopped": bool(stopped),
      "stop_reason": stopped or None,
  }


def _default_out(
    checkpoint: Optional[Path], session_dir: Optional[Path], cfg: PlayConfig
) -> Path:
  """Beside the run's other evidence, named for what it shows.

  With no checkpoint there is no run to sit beside, so the clip lands in the
  play session's own artifacts -- which is where the studio looks anyway.
  """
  stem = f"play_{checkpoint.stem}" if checkpoint else f"play_{cfg.policy}"
  if cfg.stage:
    stem += f"_{cfg.stage}"
  if session_dir is not None:
    return session_dir / "artifacts" / f"{stem}.mp4"
  if checkpoint is not None:
    return checkpoint.parent / f"{stem}.mp4"
  return Path(cfg.session_dir or ".") / "artifacts" / f"{stem}.mp4"


def _headline(env: Any) -> Dict[str, float]:
  """Whatever the run publishes about itself, measured over the clip.

  Numbers beside a video stop it being a vibe. These are the same metrics the
  training run reported, so a clip is comparable with the curves.
  """
  try:
    metrics = env.rlmcp.telemetry.get_latest_metrics()
  except Exception:
    return {}
  return {
      key: round(float(value), 5)
      for key, value in (metrics or {}).items()
      if isinstance(value, (int, float)) and key.startswith("rlmcp/")
  }


def _view(cfg: PlayConfig, env: Any, vec_env: Any, policy: Any) -> Dict[str, Any]:
  """Hand the environment to one of the backend's interactive viewers.

  Both block until the window (or the browser tab) is closed, which is the
  point: this is a person looking at a robot. Neither is a hard dependency, so
  an install without them is told what it has rather than shown a traceback.

  ``rlmcp stop`` is the other way out, for the person who is not at the window
  -- an agent, or a shell on the far end of an ssh connection. It arrives as a
  :class:`~rlmcp.core.controller.SessionStopped` raised out of the wrapper's
  ``step`` at a service boundary, unwinds the viewer's loop, and is caught
  here: from the caller's side that is the same clean exit closing the window
  gives, and ``run_play`` closes the environment either way, which is what
  writes the session's end.
  """
  from rlmcp.core.controller import SessionStopped

  try:
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
  except ImportError as exc:
    raise PlayError(
        f"This install has no interactive viewers ({exc}). --mode video still "
        "works and needs nothing but an offscreen renderer."
    ) from exc

  lab = env.rlmcp
  if not cfg.quiet:
    session = lab.session.dir
    print(
        f"[rlmcp-play] steer it from another shell:\n"
        f"[rlmcp-play]   rlmcp --session {session} status\n"
        f"[rlmcp-play]   rlmcp --session {session} reset-envs\n"
        f"[rlmcp-play]   rlmcp --session {session} run load_policy "
        f"checkpoint=<other.pt>\n"
        f"[rlmcp-play]   rlmcp --session {session} stop",
        flush=True,
    )
  viewer_cls = NativeMujocoViewer if cfg.mode == "native" else ViserPlayViewer
  # One viser server, not two. The session's live view opens it -- it is the
  # thing that picks the port, publishes the url in `status`, and closes it
  # when the run ends -- and mjlab's viewer draws its own GUI on it. Letting
  # the viewer open its own left the published port serving a second, smaller
  # panel of the same environment, which is the one anybody reading `status`
  # would have gone to. `host()` returns None on an install without viser, and
  # then this is exactly what it was before.
  hosted = lab.live_view.host_for_viewer() if cfg.mode == "viser" else None
  extra = {"viser_server": hosted} if hosted is not None else {}
  try:
    viewer = viewer_cls(vec_env, policy, **extra)
  except ImportError as exc:
    # viser in particular is an optional install of its own, and only says so
    # when the viewer is actually built.
    raise PlayError(
        f"--mode {cfg.mode} needs a package this environment does not have "
        f"({exc}). --mode video still works."
    ) from exc

  try:
    viewer.run()
  except SessionStopped as stop:
    stopped = _stop_state(lab, stop)
  else:
    stopped = _stop_state(lab)
  if stopped and not cfg.quiet:
    print(f"[rlmcp-play] stopped: {stopped}", flush=True)
  return {
      "viewer": cfg.mode,
      "closed": True,
      "stopped": bool(stopped),
      "stop_reason": stopped or None,
  }


# Command line.


def add_arguments(parser: Any) -> Any:
  """The play options. Attached to ``rlmcp play`` by :mod:`rlmcp.cli`."""
  parser.add_argument(
      "checkpoint", nargs="?", default="",
      help="Checkpoint .pt, or a run directory to take the latest one from",
  )
  parser.add_argument("--mode", default="video", choices=list(MODES),
                      help="video: render an mp4. native/viser: open a viewer.")
  parser.add_argument(
      "--policy", default="checkpoint", choices=list(POLICIES),
      help="Where actions come from. zero/random need no checkpoint, so they "
           "are how you look at a task that does not train yet (needs --task).",
  )
  parser.add_argument("--seconds", type=float, default=8.0)
  parser.add_argument("--out", default="", help="Video path (video mode)")
  parser.add_argument("--fps", type=int, default=None)
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--device", default="cuda:0",
                      help="'cpu' plays without touching a GPU that is training")
  parser.add_argument("--task", default="",
                      help="Defaults to the task the session recorded")
  parser.add_argument(
      "--stage", default="",
      help="Curriculum stage to restore; defaults to the last one the run entered",
  )
  parser.add_argument(
      "--no-replay", dest="replay", action="store_false",
      help="Do not restore trained-under conditions. The clip then shows the "
           "task's own play configuration, which for a curriculum run is rung "
           "zero -- a late policy will look broken on it.",
  )
  parser.add_argument(
      "--set", action="append", default=[], metavar="KEY=VALUE",
      help="Parameter override applied after the replay, so it wins. Repeatable.",
  )
  parser.add_argument("--task-package", action="append", default=[], metavar="MODULE",
                      help="Import this module before reading the task registry, "
                           "so a project's own tasks register. Repeatable.")
  parser.add_argument("--render-width", type=int, default=960)
  parser.add_argument("--render-height", type=int, default=720)
  parser.add_argument("--extra-envs", type=int, default=0,
                      help="Neighbouring envs composited into the frame")
  parser.add_argument("--session-dir", default="",
                      help="Where the play session publishes itself")
  parser.add_argument(
      "--allow-partial", action="store_true",
      help="Render even if some trained-under conditions could not be "
           "restored. The result is a policy shown doing a task it was not "
           "trained on, which reads as a bad policy -- so this is off.",
  )
  return parser


def config_from_args(args: Any, overrides: Dict[str, Any]) -> PlayConfig:
  return PlayConfig(
      checkpoint=args.checkpoint,
      task=args.task,
      mode=args.mode,
      policy=args.policy,
      seconds=args.seconds,
      num_envs=args.num_envs,
      device=args.device,
      out=args.out,
      fps=args.fps,
      stage=args.stage,
      replay=args.replay,
      overrides=overrides,
      task_package=list(args.task_package or []),
      render_width=args.render_width,
      render_height=args.render_height,
      extra_envs=args.extra_envs,
      session_dir=args.session_dir,
      allow_partial=args.allow_partial,
  )


__all__ = [
    "MODES",
    "PlayConfig",
    "PlayError",
    "PolicySwap",
    "SwappablePolicy",
    "add_arguments",
    "config_from_args",
    "find_checkpoint",
    "run_play",
    "session_for",
    "stage_at_iteration",
    "task_for",
]
