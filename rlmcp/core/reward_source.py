"""A reward term an agent wrote, compiled mid-run.

Tuning a weight explores the reward function a task shipped with. This is the
other move: the agent decided the task is missing a term, wrote one, and wants
it scoring from the next batch on.

The honest sentence about what happens here is that **agent-authored Python is
compiled and executed inside the training process**, with the simulator, the
policy and the optimiser in scope. There is no sandbox and this module does not
pretend to build one -- a reward term is handed the whole environment by
definition, so anything able to compute a reward is able to do anything else.

That is the same trust the rest of rlmcp already extends to whoever is driving
the run: an agent that can call ``set_parameter``, ``load_checkpoint`` and
``stop_training`` on a live process is not being held back by the absence of
``exec``. So this is not gated behind a flag; it is *recorded* instead. Every
accepted term is written to the session as its own file, and the event log
carries the digest, so "what was this run actually optimising" has an answer
after the fact that does not depend on anyone remembering.

What *is* checked, before the term reaches the manager:

* the source parses, and defines exactly one top-level function (or one named
  after the term), so "which function did you mean" is never a guess;
* the function accepts ``(env, **params)``;
* it survives a trial call and returns one finite score per environment --
  see :func:`rlmcp.adapters.manager_based.reward_terms.install_reward_term`,
  which does that part because only the backend knows the shape.

The digest is over the exact source, so a term added to two runs from the same
text is recognisably the same term.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class RewardSourceError(ValueError):
  """The source an agent supplied cannot become a reward term.

  Carries a message aimed at the agent that wrote the source, naming what was
  wrong rather than where the exception surfaced.
  """


@dataclass
class CompiledReward:
  """One reward term, ready to install.

  Attributes:
    name: the term name it will be addressed by (``reward.<name>.weight``).
    func: the callable, taking ``(env, **params)``.
    func_name: the function's own name in the source, which need not match
      ``name`` -- a task's ``mdp`` module names functions by what they measure
      and the config names terms by what they are for.
    source: the exact text that was compiled, dedented and newline-terminated.
    digest: sha256 of :attr:`source`, first 12 hex characters.
  """

  name: str
  func: Callable[..., Any]
  func_name: str
  source: str
  digest: str


def source_digest(source: str) -> str:
  return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def compile_reward_source(
    source: str,
    *,
    name: str,
    namespace: Optional[Dict[str, Any]] = None,
) -> CompiledReward:
  """Compile ``source`` into a reward function called ``name``.

  ``namespace`` is what the source is compiled against -- the task's ``mdp``
  module, ``torch``, whatever the backend offers. The source may import for
  itself; seeding the namespace only saves it from having to.

  Raises:
    RewardSourceError: with a message the agent can act on, for every failure
      mode -- a name that is not an identifier, source that does not parse, no
      function, several functions, or a signature that cannot take ``env``.
  """
  if not NAME_RE.match(name or ""):
    raise RewardSourceError(
        f"Reward term name {name!r} is not a Python identifier. The name "
        "becomes a config field and a parameter key ('reward.<name>.weight'), "
        "so it has to be one."
    )

  text = textwrap.dedent(source or "").strip()
  if not text:
    raise RewardSourceError(
        f"No source given for reward term '{name}'. Pass the text of a "
        "function that takes (env, **params) and returns one score per "
        "environment."
    )
  text += "\n"

  filename = f"<rlmcp reward '{name}'>"
  try:
    code = compile(text, filename, "exec")
  except SyntaxError as exc:
    raise RewardSourceError(
        f"Reward term '{name}' does not parse: {exc.msg} (line {exc.lineno})."
    ) from exc

  module_ns: Dict[str, Any] = dict(namespace or {})
  module_ns.setdefault("__name__", f"rlmcp_reward_{name}")
  try:
    exec(code, module_ns)  # noqa: S102 - executing agent source is the feature.
  except Exception as exc:
    raise RewardSourceError(
        f"Reward term '{name}' failed while its source was being executed "
        f"({type(exc).__name__}: {exc}). Note that this is the module body "
        "running, not the reward being computed -- an import at the top of "
        "the source is the usual cause."
    ) from exc

  func, func_name = _pick_function(module_ns, name=name)
  _check_signature(func, name=name, func_name=func_name)

  return CompiledReward(
      name=name,
      func=func,
      func_name=func_name,
      source=text,
      digest=source_digest(text),
  )


def _pick_function(module_ns: Dict[str, Any], *, name: str) -> tuple:
  """The one function the source defined, or an error naming the ambiguity."""
  defined = {
      key: value
      for key, value in module_ns.items()
      if not key.startswith("__")
      and (inspect.isfunction(value) or inspect.isclass(value))
      and getattr(value, "__module__", None) == module_ns.get("__name__")
  }
  if name in defined:
    return defined[name], name
  if len(defined) == 1:
    only = next(iter(defined))
    return defined[only], only
  if not defined:
    raise RewardSourceError(
        f"Reward term '{name}' defines no function. The source has to contain "
        "a 'def' taking (env, **params) -- an expression or a bare value is "
        "not a reward term."
    )
  raise RewardSourceError(
      f"Reward term '{name}' defines {len(defined)} top-level names "
      f"({', '.join(sorted(defined))}) and none of them is called '{name}', "
      "so which one scores the reward is a guess. Name the reward function "
      f"'{name}', or move the helpers inside it."
  )


def _check_signature(func: Any, *, name: str, func_name: str) -> None:
  """Refuse a function that cannot be called as ``func(env, **params)``.

  A class ``func`` is left alone: the manager instantiates it with its cfg and
  calls the instance, so its ``__init__`` signature says nothing about how it
  will be called.
  """
  if inspect.isclass(func):
    return
  try:
    signature = inspect.signature(func)
  except (TypeError, ValueError):  # pragma: no cover - builtins only.
    return
  positional = [
      parameter
      for parameter in signature.parameters.values()
      if parameter.kind
      in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
  ]
  takes_var_positional = any(
      parameter.kind is parameter.VAR_POSITIONAL
      for parameter in signature.parameters.values()
  )
  if not positional and not takes_var_positional:
    raise RewardSourceError(
        f"Reward term '{name}': {func_name}{signature} takes no positional "
        "argument, but a reward term is called as func(env, **params). Give "
        "it 'env' as its first parameter."
    )
