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
it, and answers five questions in the order they fail in:

===================  ======================================================
imports              a syntax error, a package that does not import
constructs           a term naming a body the robot does not have
steps                an env that dies on step 1, or hangs
rewards finite       a NaN, a divide by zero, an exploding scale
terminations sane    everything ending at once, or nothing ever ending
===================  ======================================================

Those five are most of the first day of a new task and none of them needs a
GPU, a policy, or a training run to find. A gate that fails stops the ones
after it, because they would be measuring an environment that is already
wrong -- and a "reward is NaN" line under "the env would not construct" sends
somebody to fix a reward that was never the problem.

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
from typing import Any, Dict, List, Optional, Tuple

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
  task_package: List[str] = field(default_factory=list)
  session_dir: str = ""
  """Where the throwaway session goes. Empty means a temporary directory that
  is removed afterwards; nothing about a check belongs in the records."""
  quiet: bool = False


# ── the gates ─────────────────────────────────────────────────────────────
#
# Each is a dict rather than a class because it is an answer, not an object:
# it is emitted as JSON, rendered as a table, and read by an agent. `fix` is
# the field that earns its place -- rlmcp's habit everywhere else is to say
# which fix applies to which problem rather than leave that to be inferred.


def _gate(name: str, ok: bool, detail: str = "", fix: str = "", **extra: Any) -> Dict[str, Any]:
  return {"gate": name, "ok": ok, "detail": detail, "fix": fix, **extra}


def _skipped(name: str, because: str) -> Dict[str, Any]:
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


def summarise(gates: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def rank_terms(totals: Dict[str, float], steps: int) -> List[Dict[str, Any]]:
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


def dominance(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def roll(vec_env: Any, adapter: Any, policy: Any, steps: int) -> Dict[str, Any]:
  """Step the environment and collect what the gates need.

  Deliberately tolerant of an environment that answers half of this: a task
  whose adapter cannot break rewards down is still worth stepping, and the
  gate that needed the breakdown says it did not run rather than that it
  failed.
  """
  import torch

  reward_totals: Dict[str, float] = {}
  termination_hits: Dict[str, float] = {}
  nonfinite: List[str] = []
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
    except Exception as exc:                      # noqa: BLE001 - reported below
      terms_refused = terms_refused or f"{type(exc).__name__}: {exc}"

    try:
      fired = adapter.termination_terms()
      terminations_seen = True
      for name, value in fired.items():
        termination_hits[name] = termination_hits.get(name, 0.0) + float(value)
    except Exception as exc:                      # noqa: BLE001 - reported below
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


def gates_from(rolled: Dict[str, Any], steps: int, num_envs: int) -> List[Dict[str, Any]]:
  """The three gates that can only be answered by having stepped."""
  gates: List[Dict[str, Any]] = []

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


# ── the command ───────────────────────────────────────────────────────────


def run_check(cfg: CheckConfig) -> Dict[str, Any]:
  """Build the task, roll it, and answer the five questions.

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

  gates: List[Dict[str, Any]] = []
  built: Dict[str, Any] = {}
  rolled: Dict[str, Any] = {}
  started = time.time()

  try:
    try:
      env, lab, _agent_cfg, vec_env = build_env(play_cfg, cfg.task, None)
    except PlayError as exc:
      # Everything `build_env` refuses is one of the first two gates, and it
      # already says which: an unknown task and a package that will not import
      # are import problems; anything raised while constructing is not.
      importing = "import" in str(exc).lower() or "Unknown task" in str(exc)
      gates.append(_gate(
          "imports", not importing, detail="" if not importing else str(exc),
          fix="Name the package whose import registers this task: "
              "--task-package <module>." if importing else ""))
      # All five gates are always reported, and the ones after the failure say
      # they did not run. A gate simply missing from the list is a hole a
      # reader fills in with an assumption.
      if importing:
        gates.append(_skipped("constructs", "the task could not be imported"))
      else:
        gates.append(_gate("constructs", False, detail=str(exc),
                           fix="The environment could not be built. The message "
                               "is the task's own."))
      for name in ("steps", "rewards_finite", "terminations"):
        gates.append(_skipped(name, "the environment was not built"))
      return _payload(cfg, gates, {}, {}, steps, num_envs, time.time() - started)
    except Exception as exc:                      # noqa: BLE001 - it is the answer
      gates.append(_gate("imports", True))
      gates.append(_gate(
          "constructs", False, detail=f"{type(exc).__name__}: {exc}",
          fix="Built from the task's own config, so this traceback is the "
              "task's. A term naming a body the robot does not have is the "
              "usual cause."))
      for name in ("steps", "rewards_finite", "terminations"):
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
    except Exception as exc:                      # noqa: BLE001 - it is the answer
      gates.append(_gate(
          "steps", False, detail=f"{type(exc).__name__}: {exc}",
          fix="The environment raised while stepping. Nothing after this could "
              "be measured on it."))
      for name in ("rewards_finite", "terminations"):
        gates.append(_skipped(name, "the environment stopped stepping"))
      return _payload(cfg, gates, built, {}, steps, num_envs, time.time() - started)

    gates.extend(gates_from(rolled, steps, num_envs))
    return _payload(cfg, gates, built, rolled, steps, num_envs, time.time() - started)
  finally:
    if temporary:
      shutil.rmtree(temporary, ignore_errors=True)


def _describe_env(lab: Any, vec_env: Any, num_envs: int) -> Dict[str, Any]:
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


def _payload(cfg: CheckConfig, gates: List[Dict[str, Any]], built: Dict[str, Any],
             rolled: Dict[str, Any], steps: int, num_envs: int,
             seconds: float) -> Dict[str, Any]:
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
  )


__all__ = [
    "CheckConfig",
    "CheckError",
    "DEFAULT_ENVS",
    "DEFAULT_STEPS",
    "add_arguments",
    "config_from_args",
    "dominance",
    "gates_from",
    "rank_terms",
    "roll",
    "run_check",
    "summarise",
]
