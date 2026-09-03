"""What every backend checks before it lets an agent's reward term in.

Each family appends a term differently -- a manager-based environment has a
``RewardManager`` with private lists to grow, a legged-gym-shaped one has three
dicts on the instance -- but what makes a term *safe to append* is the same
everywhere: it can be called as ``func(env, **params)``, it returns one finite
score per environment, and it did so once against the live environment before
anything was mutated. That check lives here so that a term refused on mjlab is
refused on Genesis for the same reason in the same words.

The installers themselves stay with their families:
:mod:`rlmcp.adapters.manager_based.reward_terms` and
:mod:`rlmcp.adapters.legged_gym_style.reward_terms`.
"""

from __future__ import annotations

from typing import Any


class RewardInstallError(RuntimeError):
  """A compiled term could not be installed, with the environment unchanged."""


def trial_call(env: Any, *, name: str, func: Any, params: dict[str, Any],
               num_envs: int) -> dict[str, Any]:
  """Call ``func`` once against ``env`` and check what came back.

  Returns the value's range as ``{"min", "max", "mean"}`` -- what the agent is
  shown so it can tell a term that scores from one that is silently zero.
  Raises :class:`RewardInstallError` naming the term and the reason otherwise.
  Nothing is written to ``env`` either way.
  """
  import torch

  try:
    value = func(env, **params)
  except TypeError as exc:
    raise RewardInstallError(
        f"Reward term '{name}' could not be called as func(env, **params) "
        f"with params {sorted(params)}: {exc}"
    ) from exc
  except Exception as exc:
    raise RewardInstallError(
        f"Reward term '{name}' raised on its trial call "
        f"({type(exc).__name__}: {exc}). It was not installed, and the run is "
        "unaffected."
    ) from exc

  if not isinstance(value, torch.Tensor):
    raise RewardInstallError(
        f"Reward term '{name}' returned {type(value).__name__}, but a term "
        "has to return a torch tensor of one score per environment."
    )
  if tuple(value.shape) != (int(num_envs),):
    raise RewardInstallError(
        f"Reward term '{name}' returned shape {tuple(value.shape)}, but the "
        f"environment needs ({int(num_envs)},) -- one score per environment. "
        "A term that reduces over joints usually wants a sum or mean over "
        "dim=1."
    )
  if not bool(torch.isfinite(value).all()):
    raise RewardInstallError(
        f"Reward term '{name}' returned non-finite values on its trial call. "
        "A NaN here becomes a NaN gradient; fix the term before adding it."
    )
  return {
      "min": float(value.min()),
      "max": float(value.max()),
      "mean": float(value.mean()),
  }
