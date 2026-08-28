"""Appending a reward term to a manager that is already running.

Both backends build their ``RewardManager`` once, at environment construction,
from a config dict: ``_prepare_terms`` fills ``_term_names`` and ``_term_cfgs``
and then ``__init__`` sizes the buffers -- one ``_episode_sums`` entry per term
and a ``_step_reward`` matrix of ``(num_envs, len(terms))``. Nothing about that
is re-entrant, which is why appending a term is a small piece of surgery rather
than a method call.

The surgery is four writes and has to be all four. A term appended to the name
and cfg lists but missing from ``_episode_sums`` raises ``KeyError`` on the
first ``compute()``; one missing from ``_step_reward`` writes past the end of
the matrix. So this module does the whole set, in one place, for both backends
-- they differ only in which private names exist, and every difference is
probed rather than assumed.

**The trial call happens before any of it.** A reward term that raises, or that
returns the wrong shape, would otherwise take the run down on the next batch
with a traceback pointing into the manager rather than at the term that was
just added. Instead the function is called once against the live environment,
its result checked, and the manager left untouched if either fails: a rejected
term costs a message, not the run.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class RewardInstallError(RuntimeError):
  """A compiled term could not be installed, with the manager left unchanged."""


def install_reward_term(
    env: Any,
    *,
    name: str,
    func: Any,
    weight: float,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  """Add ``name`` to the live reward manager and return what it did.

  Args:
    env: the unwrapped manager-based environment.
    name: term name, unique among the manager's active terms.
    func: the reward callable (or a class the manager will instantiate).
    weight: the term's weight, tunable afterwards as ``reward.<name>.weight``.
    params: keyword arguments passed to ``func`` on every call.

  Returns:
    A dict describing the installed term: its index, weight, whether it is
    class-based, and the trial value's range.

  Raises:
    RewardInstallError: no reward manager, a name already in use, or the
      trial call failing. The manager is unchanged in every case.
  """
  import torch

  manager = getattr(env, "reward_manager", None)
  if manager is None:
    raise RewardInstallError(
        "This environment has no reward_manager, so there is no term list to "
        "append to. Adding a reward term needs a manager-based environment."
    )

  existing = list(getattr(manager, "_term_names", []) or [])
  if name in existing:
    raise RewardInstallError(
        f"Reward term '{name}' already exists. Change its weight with "
        f"set_parameter('reward.{name}.weight', ...), or add the new term "
        "under a different name."
    )

  cfg_type = _term_cfg_type(manager)
  params = dict(params or {})
  try:
    term_cfg = cfg_type(func=func, weight=float(weight), params=params)
  except Exception as exc:
    raise RewardInstallError(
        f"Could not build a {getattr(cfg_type, '__name__', cfg_type)} for "
        f"reward term '{name}': {type(exc).__name__}: {exc}"
    ) from exc

  # mjlab and IsaacLab both use this to turn a class `func` into an instance
  # bound to its cfg, and to resolve SceneEntityCfg params against the scene.
  # A term that skips it and happens to be class-based is never instantiated.
  resolve = getattr(manager, "_resolve_common_term_cfg", None)
  if callable(resolve):
    try:
      resolve(name, term_cfg)
    except Exception as exc:
      raise RewardInstallError(
          f"Reward term '{name}' was refused while its config was being "
          f"resolved ({type(exc).__name__}: {exc}). A params entry naming a "
          "scene asset that this task does not have is the usual cause."
      ) from exc

  value = _trial_call(env, manager, name=name, term_cfg=term_cfg)

  num_envs = int(getattr(manager, "num_envs", getattr(env, "num_envs", 0)))
  device = getattr(manager, "device", getattr(env, "device", "cpu"))

  # Buffers first: allocation is the only step here that can fail, and doing
  # it before the appends keeps a failure from leaving a half-added term.
  episode_sums = getattr(manager, "_episode_sums", None)
  if episode_sums is None:
    raise RewardInstallError(
        "The reward manager has no _episode_sums, so this is not a manager "
        "shape this version of rlmcp knows how to append to."
    )
  new_sum = torch.zeros(num_envs, dtype=torch.float, device=device)
  step_reward = getattr(manager, "_step_reward", None)
  widened = None
  if step_reward is not None:
    column = torch.zeros((num_envs, 1), dtype=step_reward.dtype,
                         device=step_reward.device)
    widened = torch.cat([step_reward, column], dim=1)

  manager._term_names.append(name)
  manager._term_cfgs.append(term_cfg)
  episode_sums[name] = new_sum
  if widened is not None:
    manager._step_reward = widened
  class_terms = getattr(manager, "_class_term_cfgs", None)
  class_based = callable(getattr(term_cfg.func, "reset", None))
  if class_based and class_terms is not None:
    class_terms.append(term_cfg)
  _record_in_cfg(manager, name, term_cfg)

  return {
      "name": name,
      "index": len(manager._term_names) - 1,
      "weight": float(weight),
      "params": params,
      "class_based": bool(class_based),
      "trial_value": value,
  }


def _term_cfg_type(manager: Any) -> type:
  """The cfg class this manager's terms use, taken from a term it already has.

  Read off an existing term rather than imported, so that one code path serves
  mjlab's ``RewardTermCfg`` and IsaacLab's without this module importing either
  simulator.
  """
  for term_name in list(getattr(manager, "_term_names", []) or []):
    try:
      return type(manager.get_term_cfg(term_name))
    except Exception:
      continue
  for candidate in _cfg_entries(manager).values():
    if candidate is not None:
      return type(candidate)
  raise RewardInstallError(
      "This task has no reward terms at all, so there is no term config class "
      "to copy. Add the first term to the task's config and restart; rlmcp "
      "appends to a manager that is already running, it does not create one."
  )


def _cfg_entries(manager: Any) -> Dict[str, Any]:
  """``manager.cfg`` as a name -> term-cfg mapping, dict or dataclass."""
  cfg = getattr(manager, "cfg", None)
  if isinstance(cfg, dict):
    return cfg
  if cfg is not None and hasattr(cfg, "__dict__"):
    return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
  return {}


def _record_in_cfg(manager: Any, name: str, term_cfg: Any) -> None:
  """Put the term into ``manager.cfg`` too, so snapshots see the whole set.

  ``compute()`` reads the lists, not the cfg, so this changes no behaviour --
  but everything that reports what the run is optimising reads the cfg, and a
  term missing from it is a term missing from the record.
  """
  cfg = getattr(manager, "cfg", None)
  if isinstance(cfg, dict):
    cfg[name] = term_cfg
  elif cfg is not None:
    try:
      setattr(cfg, name, term_cfg)
    except Exception:
      pass


def _trial_call(env: Any, manager: Any, *, name: str, term_cfg: Any) -> Dict[str, Any]:
  """Call the term once and check its result, before the manager knows of it."""
  import torch

  try:
    value = term_cfg.func(env, **term_cfg.params)
  except TypeError as exc:
    raise RewardInstallError(
        f"Reward term '{name}' could not be called as func(env, **params) "
        f"with params {sorted(term_cfg.params)}: {exc}"
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
  num_envs = int(getattr(manager, "num_envs", getattr(env, "num_envs", 0)))
  if tuple(value.shape) != (num_envs,):
    raise RewardInstallError(
        f"Reward term '{name}' returned shape {tuple(value.shape)}, but the "
        f"manager needs ({num_envs},) -- one score per environment. A term "
        "that reduces over joints usually wants a sum or mean over dim=1."
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
