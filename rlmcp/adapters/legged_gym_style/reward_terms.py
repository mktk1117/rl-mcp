"""Appending a reward term to a legged-gym-shaped environment mid-run.

Go2Env, legged_gym and their forks keep the reward function as three dicts on
the instance, filled once in ``__init__`` from the keys of ``reward_scales``::

    for name in self.reward_scales:
      self.reward_scales[name] *= self.dt
      self.reward_functions[name] = getattr(self, "_reward_" + name)
      self.episode_sums[name] = torch.zeros(self.num_envs)

and ``step()`` walks them::

    for name, fn in self.reward_functions.items():
      rew = fn() * self.reward_scales[name]
      self.rew_buf += rew
      self.episode_sums[name] += rew

Nothing about that is closed. The loop reads the dicts every step, so a term
written into all three scores from the next step on, is reported by
``reset_idx`` under ``rew_<name>`` like the ones the task shipped with, and is
re-weighted through ``reward.<name>.weight`` afterwards because that key is
served straight off ``reward_scales``. What a term needs that the task's own do
not is a function: the task binds ``_reward_<name>`` methods, and an agent's
term is a plain ``func(env, **params)``, so it is bound to the environment here
and stored as a no-argument callable the loop can call.

The writes go in the order ``step()`` would notice them least: the episode sum
first, the scale second, the function last. A step that runs between two of
those writes -- there is none, servicing happens at the iteration boundary on
the training thread, but the order costs nothing -- would see a scale with no
function, which the loop never reads, rather than a function with no scale,
which raises.

**The trial call happens before any of it**, inside ``torch.inference_mode()``
because that is where ``step()`` runs under rsl_rl and where the buffers a term
reads were made. A term that raises, or returns the wrong shape, costs a
message rather than the run: the environment is untouched on every refusal.
"""

from __future__ import annotations

from typing import Any

from rlmcp.adapters.reward_terms import RewardInstallError, trial_call


def install_reward_term(
    env: Any,
    spec: Any,
    *,
    name: str,
    func: Any,
    weight: float,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Add ``name`` to the environment's reward dicts and return what it did.

  Args:
    env: the environment, of the shape ``spec`` describes.
    spec: the :class:`~rlmcp.adapters.legged_gym_style.spec.FlatEnvSpec` that
      says where the dicts are and whether the scales carry ``dt``.
    name: term name, unique among the environment's bound terms.
    func: the reward callable, ``func(env, **params) -> tensor[num_envs]``.
    weight: the weight in the units the task config uses -- the ``dt``
      conversion the environment applied to its own scales is applied here too,
      so ``reward.<name>.weight`` reads back as the number given.
    params: keyword arguments passed to ``func`` on every call.

  Returns:
    A dict describing the installed term: its index, weight, and the trial
    value's range.

  Raises:
    RewardInstallError: the environment has no reward dicts, the name is
      already bound, or the trial call failed. Nothing is changed in any case.
  """
  import torch

  params = dict(params or {})
  scales = spec.resolve(env, "reward_scales", None)
  functions = spec.resolve(env, "reward_functions", None)
  sums = spec.resolve(env, "episode_sums", None)
  if not isinstance(scales, dict) or not isinstance(functions, dict):
    raise RewardInstallError(
        f"This environment keeps no env.{spec.reward_scales} and "
        f"env.{spec.reward_functions} dicts to append to, so there is nowhere "
        "for a new term to go. If it keeps them under other names, say so "
        "with wrap(spec=FlatEnvSpec(...))."
    )
  if sums is not None and not isinstance(sums, dict):
    raise RewardInstallError(
        f"env.{spec.episode_sums} is a {type(sums).__name__}, not the dict of "
        "per-term running totals this family expects, so a new term could not "
        "be reported at episode end. Nothing was added."
    )
  if name in scales or name in functions:
    raise RewardInstallError(
        f"Reward term '{name}' already exists. Change its weight with "
        f"set_parameter('reward.{name}.weight', ...), or add the new term "
        "under a different name."
    )

  num_envs = int(spec.resolve(env, "num_envs", 0) or 0)
  with torch.inference_mode():
    value = trial_call(env, name=name, func=func, params=params,
                       num_envs=num_envs)

  # The buffer first: allocation is the only step below that can fail, and a
  # failure here leaves the dicts exactly as they were.
  new_sum = None
  if sums is not None:
    like = next(iter(sums.values()), None)
    if not isinstance(like, torch.Tensor):
      like = getattr(env, "rew_buf", None)
    if isinstance(like, torch.Tensor):
      new_sum = torch.zeros(num_envs, dtype=like.dtype, device=like.device)
    else:
      new_sum = torch.zeros(num_envs, dtype=torch.float,
                            device=getattr(env, "device", None))

  factor = 1.0
  if spec.scales_premultiplied_by_dt:
    dt = spec.resolve(env, "dt", None)
    factor = float(dt) if dt else 1.0

  def bound() -> Any:
    return func(env, **params)

  bound.__name__ = f"_reward_{name}"
  bound.__qualname__ = bound.__name__

  if new_sum is not None:
    sums[name] = new_sum
  scales[name] = float(weight) * factor
  functions[name] = bound

  return {
      "name": name,
      "index": len(functions) - 1,
      "weight": float(weight),
      "params": params,
      "class_based": False,
      "trial_value": value,
  }
