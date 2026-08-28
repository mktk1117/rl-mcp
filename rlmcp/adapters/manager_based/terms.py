"""What each manager term paid, and which one ended the episode.

Both manager-based backends compute a reward as a sum of named terms and a
termination as an OR of named terms, and both keep the per-term result of the
last step on the manager. That is the whole of this module: read those two
buffers and average them over the batch.

It lives here rather than in a backend because the shape is the shared one --
``env.reward_manager.active_terms`` and ``env.termination_manager.get_term``
mean the same thing on mjlab and on IsaacLab, and a copy in each would be a
copy that drifts.

Why the *rate* rather than the scaled reward: mjlab multiplies each term by
``dt`` before adding it, and keeps the unscaled ``weight x value`` in
``_step_reward`` -- the buffer its own viewer's bar panel reads. A factor
common to every term changes no comparison between them, and the rate is the
number that stays the same when somebody changes the control frequency.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _mean_over_envs(values: Any) -> List[float]:
  """A per-env tensor or array to plain floats, one per column."""
  if values is None:
    return []
  try:
    if hasattr(values, "dim") and values.dim() > 1:      # torch [envs, terms]
      return [float(v) for v in values.float().mean(dim=0).cpu().tolist()]
    if hasattr(values, "mean"):                          # torch/np [envs]
      return [float(values.float().mean().cpu().item())
              if hasattr(values, "cpu") else float(values.mean())]
  except Exception:
    return []
  return []


def reward_terms(env: Any) -> Dict[str, float]:
  """Per-term reward on the last step, averaged over the environments.

  Empty when the environment has no reward manager -- the caller turns that
  into :class:`~rlmcp.adapters.base.NotSupported`, because "no terms" and "a
  task with nothing to break down" are the same answer and both mean *ask
  something else*.
  """
  manager = getattr(getattr(env, "unwrapped", env), "reward_manager", None)
  if manager is None:
    return {}
  names = list(getattr(manager, "active_terms", []) or [])
  if not names:
    return {}

  # The buffer the manager fills every step, in the order of `active_terms`.
  # Documented by mjlab's own RewardManager as the unscaled rate, and the same
  # buffer its viewer reads; IsaacLab's is the same field with the same shape.
  step = getattr(manager, "_step_reward", None)
  column = _mean_over_envs(step)
  if len(column) == len(names):
    return {name: column[index] for index, name in enumerate(names)}

  # No buffer of that shape: fall back to the public per-env accessor, which
  # both backends have, and read environment zero rather than inventing a mean.
  iterable = getattr(manager, "get_active_iterable_terms", None)
  if callable(iterable):
    try:
      return {str(name): float(value[0])
              for name, value in iterable(env_idx=0) if value}
    except Exception:
      return {}
  return {}


def termination_terms(env: Any) -> Dict[str, float]:
  """Fraction of environments each termination term fired on, last step.

  Includes the terms that did not fire, as ``0.0``. A list that silently drops
  them cannot answer the question it exists for -- *nothing ever ends, which
  term should have?* -- because the term that never fires is exactly the one
  missing from it.
  """
  manager = getattr(getattr(env, "unwrapped", env), "termination_manager", None)
  if manager is None:
    return {}
  names = list(getattr(manager, "active_terms", []) or [])
  out: Dict[str, float] = {}
  for name in names:
    try:
      fired = manager.get_term(name)
    except Exception:
      continue
    column = _mean_over_envs(fired)
    if column:
      out[str(name)] = column[0]
  return out


__all__ = ["reward_terms", "termination_terms"]
