"""``rlmcp check``: verify a task before spending GPU time on it.

[docs/tuning.md](../docs/tuning.md) opens with the finding that paid for this
file: across two long campaigns, ~90 recorded locomotion runs and an in-hand
manipulation notebook, **no blocking failure was ever a PPO hyperparameter.**
Every one was a broken task, a broken measurement, or a broken interface. Step
one of that document is *verify the task before training on it*, and until now
it was prose: roll zero actions, roll random ones, look at the term
magnitudes, ask the do-nothing question. Every agent that followed it wrote the
loop again in a scratch file, slightly differently.

This is that loop, once, as a command. It builds the task with no policy --
exactly as ``rlmcp play --policy zero`` does, through the same builder -- rolls
it, and answers six questions in the order they fail in:

===================  ======================================================
imports              a syntax error, a package that does not import
constructs           a term naming a body the robot does not have
steps                an env that dies on step 1, or hangs
rewards finite       a NaN, a divide by zero, an exploding scale
terminations sane    everything ending at once, or nothing ever ending
trains               a runner that will not build, or dies at iteration 0
===================  ======================================================

Those six are most of the first day of a new task and none of them needs a
GPU or a training run to find. A gate that fails stops the ones after it,
because they would be measuring an environment that is already wrong -- and a
"reward is NaN" line under "the env would not construct" sends somebody to fix
a reward that was never the problem.

The last one is here because the first five were once all that there was, and
they were not enough. A task passed every one of them and its training run
died at iteration 0: its ``rl_cfg`` was missing ``distribution_cfg``, so the
actor had no action distribution to sample from. Nothing above could have seen
it -- a zero policy is a callable returning zeros, it never constructs a
runner, so it never reads the agent config at all. Five green ticks were read,
by a GUI and by a person, as *this will train*, and it did not. ``trains``
builds what ``rlmcp train`` builds, from the same registry, and takes one
iteration on it. It is last because it is the only gate that needs the five
before it to be true to mean anything, and because it is the only one that
steps the environment with something other than zeros -- running it earlier
would leave the reward table measuring what a half-random policy paid rather
than what doing nothing pays.

What it reports beyond pass/fail is the part worth reading: **what each reward
term paid**. The worked example this project keeps coming back to is a task
where standing still paid about 2600x what making progress paid; nobody writes
that on purpose, it is invisible in the code and in the total, and it is
obvious the moment the terms are listed side by side against a zero policy that
is, by construction, doing nothing.

Nothing above ``run_check`` imports a simulator, so ``rlmcp --help`` stays
cheap.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_STEPS = 150
"""Enough to leave the reset transient and see a short episode end.

`docs/tuning.md` says ~150 passive steps, which is where the number comes
from: it is what the campaigns actually rolled."""

MAX_STEPS = 5000

DEFAULT_ENVS = 4
"""More than one, because a termination that fires in one env and not another
is information, and few, because this is meant to be cheap."""


class CheckError(RuntimeError):
  """The check could not be run at all -- distinct from a gate that failed.

  "This task does not exist" and "this task terminates on step 1" are different
  answers and lead to different places.
  """


@dataclass
class CheckConfig:
  task: str = ""
  policy: str = "zero"
  """``zero`` or ``random``. Both are the point: a task must survive doing
  nothing, and must survive being shaken."""
  steps: int = DEFAULT_STEPS
  num_envs: int = DEFAULT_ENVS
  device: str = "cpu"
  """CPU by default. This is a correctness check on a handful of environments,
  and it must not queue behind -- or evict -- a training run on the card."""
  task_package: list[str] = field(default_factory=list)
  session_dir: str = ""
  """Where the throwaway session goes. Empty means a temporary directory that
  is removed afterwards; nothing about a check belongs in the records."""
  runner: bool = True
  """Whether to build the RL runner and take one iteration -- the ``trains``
  gate. On by default, and the default is the whole argument: a check that
  skips the failure it was built for is not much of a check. It is also nearly
  free where it matters. The environment is already built and already rolled
  by the time this runs; one iteration adds ``num_steps_per_env`` (24, in
  every task here) steps on a handful of CPU envs and one PPO update over the
  ~100 transitions they produce. Cartpole pays about a second for it.

  ``--no-runner`` turns it off, for the case this cannot argue with: a task
  whose iteration really is expensive, being checked for something else."""
  quiet: bool = False


# ── the gates ─────────────────────────────────────────────────────────────
#
# Each is a dict rather than a class because it is an answer, not an object:
# it is emitted as JSON, rendered as a table, and read by an agent. `fix` is
# the field that earns its place -- rlmcp's habit everywhere else is to say
# which fix applies to which problem rather than leave that to be inferred.


def _gate(name: str, ok: bool, detail: str = "", fix: str = "", **extra: Any) -> dict[str, Any]:
  return {"gate": name, "ok": ok, "detail": detail, "fix": fix, **extra}


def _skipped(name: str, because: str) -> dict[str, Any]:
  """A gate that did not run, which is not a gate that passed.

  Reporting it as a pass is how a broken env comes back with four ticks and
  one cross; reporting it as a failure invents a second problem.
  """
  return {"gate": name, "ok": None, "detail": f"not run: {because}", "fix": ""}


def _finite(value: Any) -> bool:
  try:
    return math.isfinite(float(value))
  except (TypeError, ValueError):
    return False


def summarise(gates: list[dict[str, Any]]) -> dict[str, Any]:
  """Pass/fail for the whole check, and the first thing that went wrong."""
  failed = [g for g in gates if g["ok"] is False]
  skipped = [g for g in gates if g["ok"] is None]
  return {
      # `passed`, not `ok`: the envelope this is emitted in already uses `ok`
      # for "the command ran", and a check that ran and found a broken task is
      # both of those at once.
      "passed": not failed,
      "failed": [g["gate"] for g in failed],
      "skipped": [g["gate"] for g in skipped],
      "first_problem": failed[0]["gate"] if failed else "",
  }


def rank_terms(totals: dict[str, float], steps: int) -> list[dict[str, Any]]:
  """Reward terms, largest contribution first, with each one's share.

  Sorted by magnitude rather than by value: a penalty large enough to swamp
  every reward is the same finding as a reward that does, and sorting by
  signed value would put it at the bottom of the list.
  """
  if not totals or steps <= 0:
    return []
  means = {name: total / steps for name, total in totals.items()}
  scale = sum(abs(v) for v in means.values()) or 1.0
  rows = [{"term": name, "mean": value, "share": abs(value) / scale}
          for name, value in means.items()]
  rows.sort(key=lambda r: -abs(r["mean"]))
  return rows


def dominance(rows: list[dict[str, Any]]) -> dict[str, Any]:
  """How lopsided the reward is, in the form the failure actually takes.

  Not a verdict -- a task with one dominant term may be exactly right. It is
  the ratio somebody should look at, named, so the 2600x case cannot be
  scrolled past.
  """
  if len(rows) < 2:
    return {}
  top, second = rows[0], rows[1]
  if not abs(second["mean"]):
    return {"term": top["term"], "over": second["term"], "ratio": float("inf")}
  return {"term": top["term"], "over": second["term"],
          "ratio": abs(top["mean"]) / abs(second["mean"])}


# ── the rollout ───────────────────────────────────────────────────────────


def roll(vec_env: Any, adapter: Any, policy: Any, steps: int) -> dict[str, Any]:
  """Step the environment and collect what the gates need.

  Deliberately tolerant of an environment that answers half of this: a task
  whose adapter cannot break rewards down is still worth stepping, and the
  gate that needed the breakdown says it did not run rather than that it
  failed.
  """
  import torch

  reward_totals: dict[str, float] = {}
  termination_hits: dict[str, float] = {}
  nonfinite: list[str] = []
  total_reward = 0.0
  dones = 0
  completed = 0
  first_done_step = 0
  terms_seen = False
  terminations_seen = False
  # Why the breakdown is missing, kept once. Swallowing it entirely is how a
  # typo in an adapter becomes "this backend does not support terms" forever.
  terms_refused = ""
  terminations_refused = ""

  obs = None
  try:
    obs = vec_env.get_observations()
    if isinstance(obs, tuple):
      obs = obs[0]
  except Exception:
    obs = None

  for step in range(steps):
    action = policy(obs)
    result = vec_env.step(action)
    # Both vector APIs in the wild: (obs, reward, done, info) and the
    # five-tuple with terminated/truncated split.
    if len(result) == 5:
      obs, reward, terminated, truncated, _info = result
      done = terminated | truncated
    else:
      obs, reward, done, _info = result
    if isinstance(obs, tuple):
      obs = obs[0]

    completed = step + 1
    with torch.no_grad():
      mean_reward = float(reward.float().mean().item()) if hasattr(reward, "float") else float(reward)
      done_now = int(done.sum().item()) if hasattr(done, "sum") else int(bool(done))
    if not _finite(mean_reward):
      nonfinite.append(f"total reward at step {completed}")
    else:
      total_reward += mean_reward
    if done_now:
      dones += done_now
      first_done_step = first_done_step or completed

    try:
      terms = adapter.reward_terms()
      terms_seen = True
      for name, value in terms.items():
        if not _finite(value):
          nonfinite.append(f"reward.{name} at step {completed}")
          continue
        reward_totals[name] = reward_totals.get(name, 0.0) + float(value)
    except Exception as exc:
      terms_refused = terms_refused or f"{type(exc).__name__}: {exc}"

    try:
      fired = adapter.termination_terms()
      terminations_seen = True
      for name, value in fired.items():
        termination_hits[name] = termination_hits.get(name, 0.0) + float(value)
    except Exception as exc:
      terminations_refused = terminations_refused or f"{type(exc).__name__}: {exc}"

  return {
      "completed": completed,
      "total_reward_per_step": total_reward / completed if completed else 0.0,
      "reward_totals": reward_totals,
      "termination_totals": termination_hits,
      "nonfinite": nonfinite,
      "dones": dones,
      "first_done_step": first_done_step,
      "terms_seen": terms_seen,
      "terminations_seen": terminations_seen,
      "terms_refused": terms_refused,
      "terminations_refused": terminations_refused,
  }


def gates_from(rolled: dict[str, Any], steps: int, num_envs: int) -> list[dict[str, Any]]:
  """The three gates that can only be answered by having stepped."""
  gates: list[dict[str, Any]] = []

  completed = rolled["completed"]
  gates.append(_gate(
      "steps", completed >= steps,
      detail=f"{completed} of {steps} steps",
      fix="" if completed >= steps else
          "The environment stopped stepping. The traceback above is the task's.",
      completed=completed))

  if not rolled["terms_seen"]:
    gates.append(_skipped(
        "rewards_finite",
        rolled.get("terms_refused")
        or "this backend does not break the reward into terms"))
  elif rolled["nonfinite"]:
    first = rolled["nonfinite"][0]
    gates.append(_gate(
        "rewards_finite", False,
        detail=f"{len(rolled['nonfinite'])} non-finite value(s), first at {first}",
        fix="A NaN in one term poisons the sum, and PPO will train on it "
            "without complaining. Look for a division by a quantity that can "
            "be zero, or an exp of something unbounded.",
        offenders=rolled["nonfinite"][:8]))
  else:
    gates.append(_gate("rewards_finite", True,
                       detail=f"{len(rolled['reward_totals'])} terms, all finite"))

  # Terminations: both extremes are failures, and the whole point is that they
  # look identical from a reward curve.
  per_env = rolled["dones"] / max(1, num_envs)
  if not rolled["terminations_seen"] and not rolled["dones"]:
    gates.append(_skipped(
        "terminations",
        rolled.get("terminations_refused") or "no termination terms to read"))
  elif rolled["first_done_step"] and rolled["first_done_step"] <= 2:
    gates.append(_gate(
        "terminations", False,
        detail=f"an episode ended on step {rolled['first_done_step']}",
        fix="Something terminates immediately -- a contact term touching the "
            "ground the robot spawns on, or a pose limit the keyframe already "
            "violates. Check the terms listed under `terminations` below.",
        first_done_step=rolled["first_done_step"]))
  elif per_env > 4:
    gates.append(_gate(
        "terminations", False,
        detail=f"{per_env:.1f} episodes per env in {steps} steps",
        fix="Episodes are ending several times faster than the task's own "
            "episode length. A policy cannot learn a task it never sees the "
            "middle of.",
        episodes_per_env=per_env))
  else:
    gates.append(_gate(
        "terminations", True,
        detail=(f"{per_env:.1f} episodes per env"
                if rolled["dones"] else "nothing terminated"),
        episodes_per_env=per_env))
  return gates


# ── the gate that constructs what training constructs ─────────────────────
#
# Everything above rolls a zero policy, which is a callable returning zeros.
# It is the right thing to roll -- a task must survive doing nothing -- but it
# means nothing above here has opened the agent config, and half of what stops
# a run at iteration 0 lives in the agent config. This is the other half of
# the interface: the runner `rlmcp train` would build, against the environment
# that was just built, taking one iteration.
#
# One caveat, stated rather than hidden: the environment under it is `play`'s
# (`load_env_cfg(task, play=True)`), because that is what the rest of the
# command built. Randomisation differs from training there; observation and
# action shapes do not, which is what this gate is reading.


def load_runner(task: str) -> Any:
  """The runner class ``rlmcp train`` would use, from the same registry.

  Deliberately a separate call from the step below, so the two failures stay
  apart. An :class:`ImportError` here means there is no ``rsl_rl`` in this
  environment, which is a gate that *did not run*; anything raised while
  constructing or stepping that class belongs to the task, which is a gate
  that *failed*. Collapsing the two is how "you have not installed the RL
  library" gets reported as "your task is broken".
  """
  from mjlab.rl import MjlabOnPolicyRunner
  from mjlab.tasks.registry import load_runner_cls

  # `or MjlabOnPolicyRunner` is `rlmcp train`'s own line, not a convention
  # invented here: a task registers a runner class only when it needs a
  # special one.
  return load_runner_cls(task) or MjlabOnPolicyRunner


def train_once(runner_cls: Any, vec_env: Any, agent_cfg: Any,
               device: str) -> dict[str, Any]:
  """Construct the runner and take one iteration: a rollout and an update.

  Constructing it is not enough, and that is not a guess. The failure this
  gate exists for -- a missing ``distribution_cfg`` -- builds a runner
  perfectly happily; the actor is simply left with no output distribution,
  and the first time PPO asks it to act, it raises. Only a step reaches that
  code. The same is true of a shape mismatch that survives construction and
  of an optimiser that never sees a gradient.
  """
  import contextlib
  import io
  from dataclasses import asdict, is_dataclass

  cfg = asdict(agent_cfg) if is_dataclass(agent_cfg) else dict(agent_cfg)
  started = time.time()
  # rsl_rl prints the actor and critic modules as it builds them, and its
  # iteration table as it learns. Both belong to a training run, not to a
  # command whose whole output is a table of gates. Nothing is lost: a failure
  # comes back from here as data, not as something printed.
  with contextlib.redirect_stdout(io.StringIO()):
    # `log_dir=None` on purpose. rsl_rl opens its summary writer only when it
    # has somewhere to write, and which writer is the agent config's business
    # -- it may be W&B. A check must not open a network connection, and must
    # not leave a run behind in somebody's logs. No `attach_runner` either:
    # that is how a *training* session adopts a runner, with the telemetry
    # and checkpoints that go with it, and this one is thrown away in a
    # moment.
    runner = runner_cls(vec_env, cfg, None, device)
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
  return {
      "steps_per_env": int(cfg.get("num_steps_per_env", 0) or 0),
      "seconds": round(time.time() - started, 2),
  }


def _raised_in(exc: BaseException) -> str:
  """The innermost frame, which for this gate is most of the diagnosis.

  A runner failure surfaces deep inside the RL library, where the message on
  its own -- "'NoneType' object has no attribute 'mean'" -- names nothing a
  task author can act on. The file and function it came from do.
  """
  import traceback

  frames = traceback.extract_tb(exc.__traceback__)
  if not frames:
    return ""
  last = frames[-1]
  return f"{Path(last.filename).name}:{last.lineno} in {last.name}"


def _traceback_of(exc: BaseException) -> str:
  """The traceback with this file's own frames removed.

  The outermost two are always ``train_gate`` calling ``train_once``, which
  say nothing: what is wanted is the first line that is not rlmcp's.
  """
  import traceback

  ours = str(Path(__file__).resolve())
  frames = traceback.extract_tb(exc.__traceback__)
  theirs = [f for f in frames if str(Path(f.filename).resolve()) != ours]
  return "".join(traceback.format_list(theirs or frames)
                 + traceback.format_exception_only(type(exc), exc))


TRAINS_FIX = (
    "This is the agent config, not the environment: the env built, stepped "
    "and paid finite rewards above. Look at the task's registered `rl_cfg` -- "
    "a field the policy needs and does not have (`distribution_cfg` is the "
    "one that has bitten this project), an actor whose input does not match "
    "the observation the env returns, or an optimiser that cannot be built. "
    "`rlmcp train` fails here too, at iteration 0, after paying for the "
    "environment first."
)


def train_gate(task: str, vec_env: Any, agent_cfg: Any, device: str,
               earlier: list[dict[str, Any]], enabled: bool = True) -> dict[str, Any]:
  """Answer ``trains``: would a training run get past iteration 0?

  A pass means training *starts* and takes a step. It is not a forecast that
  the task learns anything -- no gate here is, and `docs/tuning.md` is the
  rest of that question.
  """
  if not enabled:
    return _skipped("trains", "--no-runner")
  broken = [g["gate"] for g in earlier if g["ok"] is False]
  if broken:
    # Not thrift -- correctness. PPO will take an optimiser step on a NaN
    # reward without complaining, and report the tick this gate exists to
    # stop being wrong.
    return _skipped(
        "trains",
        f"{broken[0]} failed, and one optimiser step on an environment that "
        "is already wrong proves nothing")
  if agent_cfg is None:
    return _skipped("trains", "this task registers no RL config to build from")

  try:
    # Only an ImportError from the *lookup* is a gate that did not run.
    # Everything else, from either half, is the task's answer and is reported
    # as one -- including a registry that cannot produce a runner at all,
    # which `rlmcp train` would hit at the same line.
    try:
      runner_cls = load_runner(task)
    except ImportError as exc:
      return _skipped("trains", f"no RL runner available here: {exc}")
    info = train_once(runner_cls, vec_env, agent_cfg, device)
  except Exception as exc:
    where = _raised_in(exc)
    return _gate(
        "trains", False,
        detail=f"{type(exc).__name__}: {exc}" + (f" (raised in {where})" if where else ""),
        fix=TRAINS_FIX,
        traceback=_traceback_of(exc))
  # Short prose in `detail` and only `seconds` beside it, as `constructs`
  # does. A column per fact -- the runner class, the rollout length -- widens
  # the gates table past a terminal for every task that passes, and that table
  # is the thing somebody actually reads. When it fails, the runner class is
  # named in the traceback, which is where it is wanted anyway.
  return _gate(
      "trains", True,
      detail=f"1 iteration of {info['steps_per_env']} steps/env",
      seconds=info["seconds"])


# ── the command ───────────────────────────────────────────────────────────


def run_check(cfg: CheckConfig) -> dict[str, Any]:
  """Build the task, roll it, and answer the six questions.

  The build is ``play``'s, unchanged: the same registry lookup, the same
  ``play=True`` config, the same wrapper. A check that constructed the
  environment its own way would be checking something nobody trains.
  """
  from rlmcp.play import PlayConfig, PlayError, build_env, untrained_policy

  if not cfg.task:
    raise CheckError(
        "`rlmcp check` needs --task: there is no checkpoint here to infer one "
        "from, which is the point -- this runs before anything has trained.")
  if cfg.policy not in ("zero", "random"):
    raise CheckError(
        f"--policy {cfg.policy} is for `rlmcp play`. A check has no weights to "
        "load: it is zero or random.")

  steps = max(1, min(int(cfg.steps), MAX_STEPS))
  num_envs = max(1, int(cfg.num_envs))
  temporary = ""
  if not cfg.session_dir:
    temporary = tempfile.mkdtemp(prefix="rlmcp-check-")

  play_cfg = PlayConfig(
      task=cfg.task, policy=cfg.policy, mode="video", num_envs=num_envs,
      device=cfg.device, task_package=list(cfg.task_package),
      # Nothing to replay: a task being checked has no run behind it, and
      # restoring conditions from one would be checking that run's task.
      replay=False,
      session_dir=cfg.session_dir or temporary,
      quiet=True,
  )

  gates: list[dict[str, Any]] = []
  built: dict[str, Any] = {}
  rolled: dict[str, Any] = {}
  started = time.time()

  try:
    try:
      env, lab, agent_cfg, vec_env = build_env(play_cfg, cfg.task, None)
    except PlayError as exc:
      # Everything `build_env` refuses is one of the first two gates, and it
      # already says which: an unknown task and a package that will not import
      # are import problems; anything raised while constructing is not.
      importing = "import" in str(exc).lower() or "Unknown task" in str(exc)
      gates.append(_gate(
          "imports", not importing, detail="" if not importing else str(exc),
          fix="Name the package whose import registers this task: "
              "--task-package <module>." if importing else ""))
      # All six gates are always reported, and the ones after the failure say
      # they did not run. A gate simply missing from the list is a hole a
      # reader fills in with an assumption.
      if importing:
        gates.append(_skipped("constructs", "the task could not be imported"))
      else:
        gates.append(_gate("constructs", False, detail=str(exc),
                           fix="The environment could not be built. The message "
                               "is the task's own."))
      for name in ("steps", "rewards_finite", "terminations", "trains"):
        gates.append(_skipped(name, "the environment was not built"))
      return _payload(cfg, gates, {}, {}, steps, num_envs, time.time() - started)
    except Exception as exc:
      gates.append(_gate("imports", True))
      gates.append(_gate(
          "constructs", False, detail=f"{type(exc).__name__}: {exc}",
          fix="Built from the task's own config, so this traceback is the "
              "task's. A term naming a body the robot does not have is the "
                  "usual cause."))
      for name in ("steps", "rewards_finite", "terminations", "trains"):
        gates.append(_skipped(name, "the environment was not built"))
      return _payload(cfg, gates, {}, {}, steps, num_envs, time.time() - started)

    gates.append(_gate("imports", True))
    gates.append(_gate("constructs", True,
                       detail=f"{num_envs} env(s) on {cfg.device}",
                       seconds=round(time.time() - started, 2)))
    built = _describe_env(lab, vec_env, num_envs)

    policy = untrained_policy(play_cfg, vec_env)
    try:
      rolled = roll(vec_env, lab.sim, policy, steps)
    except Exception as exc:
      gates.append(_gate(
          "steps", False, detail=f"{type(exc).__name__}: {exc}",
          fix="The environment raised while stepping. Nothing after this could "
              "be measured on it."))
      for name in ("rewards_finite", "terminations", "trains"):
        gates.append(_skipped(name, "the environment stopped stepping"))
      return _payload(cfg, gates, built, {}, steps, num_envs, time.time() - started)

    gates.extend(gates_from(rolled, steps, num_envs))
    # Last, and only now: the reward table above is what a zero policy paid,
    # and it stays that way because nothing before this point stepped the
    # environment with anything else.
    gates.append(train_gate(cfg.task, vec_env, agent_cfg, cfg.device, gates,
                            enabled=cfg.runner))
    return _payload(cfg, gates, built, rolled, steps, num_envs, time.time() - started)
  finally:
    if temporary:
      shutil.rmtree(temporary, ignore_errors=True)


def _describe_env(lab: Any, vec_env: Any, num_envs: int) -> dict[str, Any]:
  """The shape of what was built, which is worth seeing even when it passes."""
  def _try(fn, default=None):
    try:
      return fn()
    except Exception:
      return default

  return {
      "num_envs": num_envs,
      "step_dt": _try(lambda: float(lab.sim.step_dt())),
      "actions": _try(lambda: int(vec_env.num_actions)),
      "max_episode_length_s": _try(lambda: lab.sim.max_episode_length()),
      "joints": _try(lambda: len(lab.sim.joint_names() or []), 0),
  }


def _payload(cfg: CheckConfig, gates: list[dict[str, Any]], built: dict[str, Any],
             rolled: dict[str, Any], steps: int, num_envs: int,
             seconds: float) -> dict[str, Any]:
  rewards = rank_terms(rolled.get("reward_totals", {}), rolled.get("completed", 0))
  terminations = [
      {"term": name, "per_step": total / max(1, rolled.get("completed", 1))}
      for name, total in sorted(rolled.get("termination_totals", {}).items(),
                                key=lambda kv: -kv[1])
  ]
  return {
      "task": cfg.task,
      "policy": cfg.policy,
      "steps": steps,
      "num_envs": num_envs,
      "device": cfg.device,
      "seconds": round(seconds, 2),
      "env": built,
      "gates": gates,
      "rewards": rewards,
      "dominance": dominance(rewards),
      "terminations": terminations,
      "reward_per_step": rolled.get("total_reward_per_step", 0.0),
      **summarise(gates),
  }


# ── command line ──────────────────────────────────────────────────────────


def add_arguments(parser: Any) -> Any:
  parser.add_argument("--task", default="", help="Registered task id (required)")
  parser.add_argument("--policy", default="zero", choices=("zero", "random"),
                      help="zero: the task must survive doing nothing. "
                           "random: it must survive being shaken.")
  parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
  parser.add_argument("--num-envs", type=int, default=DEFAULT_ENVS)
  parser.add_argument("--device", default="cpu",
                      help="cpu by default, so a check never queues behind a run")
  parser.add_argument("--task-package", action="append", default=[], metavar="MODULE",
                      help="Import this module first, so its tasks register. Repeatable.")
  parser.add_argument("--session-dir", default="",
                      help="Keep the throwaway session here instead of a temp dir")
  parser.add_argument("--no-runner", dest="runner", action="store_false",
                      help="Skip the `trains` gate: do not build the RL runner "
                           "and do not take an optimiser step. On by default, "
                           "because it is the only gate that reads the agent "
                           "config at all.")
  return parser


def config_from_args(args: Any) -> CheckConfig:
  return CheckConfig(
      task=args.task,
      policy=args.policy,
      steps=args.steps,
      num_envs=args.num_envs,
      device=args.device,
      task_package=list(args.task_package or []),
      session_dir=args.session_dir,
      runner=getattr(args, "runner", True),
  )


__all__ = [
  "DEFAULT_ENVS",
  "DEFAULT_STEPS",
  "CheckConfig",
  "CheckError",
  "add_arguments",
  "config_from_args",
  "dominance",
  "gates_from",
  "load_runner",
  "rank_terms",
  "roll",
  "run_check",
  "summarise",
  "train_gate",
  "train_once",
]
