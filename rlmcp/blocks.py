"""Building blocks for a single-file environment: variables, pipes, terms.

An ``env.py`` has three kinds of things in it, and rlmcp wants to reach all
three without the file listing them by hand:

* **variables** -- the tensors ``step()`` writes, one row per environment:
  joint positions, the base velocity, foot contacts, the command, the
  reward. Declared once as a :class:`Vars` subclass, so they can be listed,
  sampled into a trace and labelled without the environment knowing which
  of its attributes rlmcp reads::

      class State(Vars):
        base_lin_vel: Tensor = var(3)
        joint_pos: Tensor = var("joint")
        foot_contact: Tensor = var("foot", dtype=torch.bool)
        commands: Tensor = var(("lin_vel_x", "lin_vel_y", "ang_vel_z"))

      self.state = State(num_envs, device, joint=joint_names, foot=("FR", "FL"))

* **parameters** -- the numbers the environment reads: the config dataclass
  (:mod:`rlmcp.declare`), a reward weight, a noise width, a delay. Every
  numeric leaf of a declared object is served by rlmcp; ``Static[...]`` marks
  the ones read once at construction.

* **pipes** -- what sits between a variable and whoever consumes it. An
  :class:`Obs` group names its terms, each a *source* (a variable, or a
  function of the state) followed by *stages* (:class:`Noise`,
  :class:`Delay`, :class:`Scale`, :class:`Clip`, :class:`Offset`), and
  concatenates the results. A stage is a small dataclass, so its fields are
  parameters like any other: ``actor_obs.joint_pos.noise.half_width``,
  ``actor_obs.joint_vel.delay.steps``::

      self.actor_obs = Obs(
          base_lin_vel=("base_lin_vel", Noise(0.5)),
          joint_pos=(self.joint_pos_rel, Offset(self.encoder_bias), Noise(0.01)),
          joint_vel=("joint_vel", Delay(2), Noise(1.5)),
          commands="commands",
      )
      obs = self.actor_obs(self.state)

Reward terms are the fourth block and live in :mod:`rlmcp.declare`: a
``term(weight, **params)`` in the config's reward table names a method of the
environment with the same signature, and
:meth:`~rlmcp.adapters.single_file.SingleFileEnv.compute_reward` sums
``weight * method(**params)`` over the table.

The blocks are plain torch; nothing here knows what a robot is. rlmcp finds
an environment's :class:`Obs` and :class:`Pipe` blocks by looking at its
attributes (:func:`blocks`), and its variables through the ``state``
attribute, so a block is served the moment it is assigned.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from rlmcp.declare import Static, Term, is_static, static_fields, term, terms

# ---------------------------------------------------------------------------
# Variables.
# ---------------------------------------------------------------------------

Dim = int | str | Sequence[str]
"""One axis of a variable: a size, the name of a dimension given at
construction (``"joint"``), or the labels of the axis themselves."""


@dataclass(frozen=True)
class VarSpec:
  """How one variable is declared: its shape after the env axis, its dtype."""

  shape: tuple[Dim, ...]
  dtype: torch.dtype
  doc: str = ""


def var(*shape: Dim, dtype: torch.dtype = torch.float32, doc: str = "") -> Any:
  """Declare one variable on a :class:`Vars` subclass.

  ``var()`` is one number per environment, ``var(3)`` three, ``var("joint")``
  as many as the ``joint`` dimension passed at construction, and
  ``var(("x", "y"))`` two with those labels. The annotation on the field is
  ``Tensor``, which is what the instance holds.
  """
  return VarSpec(tuple(shape), dtype, doc)


class Vars:
  """Named per-environment tensors, declared on the class, allocated on the
  instance.

  Subclass and declare each variable with :func:`var`; construct with the
  batch size, the device and every named dimension the declarations use,
  as a size or as the labels of that axis. Every variable starts at zero.
  ``reset(env_ids)`` zeroes every variable for those environments, which is
  the right start for bookkeeping (air time, episode length) and harmless
  for state the next read from the simulator overwrites.

  The instance is a plain namespace: ``state.joint_pos`` is the tensor.
  :meth:`vars` lists them, :meth:`labels` names the components of one, and
  that is how rlmcp samples a trace of everything the environment keeps.
  """

  def __init__(self, num_envs: int, device: torch.device | str, **dims: int | Sequence[str]):
    self.num_envs = int(num_envs)
    self.device = torch.device(device)
    self._sizes: dict[str, int] = {}
    self._dim_labels: dict[str, list[str]] = {}
    for name, dim in dims.items():
      if isinstance(dim, int):
        self._sizes[name] = dim
      else:
        labels = [str(x) for x in dim]
        self._sizes[name] = len(labels)
        self._dim_labels[name] = labels
    self._specs = self.specs()
    self._labels: dict[str, list[str] | None] = {}
    for name, spec in self._specs.items():
      shape = tuple(self._size(name, d) for d in spec.shape)
      setattr(self, name, torch.zeros(
          (self.num_envs, *shape), dtype=spec.dtype, device=self.device))
      self._labels[name] = self._label_axes(spec.shape)

  # Declaration.

  @classmethod
  def specs(cls) -> dict[str, VarSpec]:
    """``{name: spec}`` for every declared variable, base classes first."""
    out: dict[str, VarSpec] = {}
    for klass in reversed(cls.__mro__):
      for name, value in vars(klass).items():
        if isinstance(value, VarSpec):
          out[name] = value
    return out

  def _size(self, name: str, dim: Dim) -> int:
    if isinstance(dim, int):
      return dim
    if isinstance(dim, str):
      if dim not in self._sizes:
        raise KeyError(
            f"Variable '{name}' uses dimension '{dim}', which {type(self).__name__} "
            f"was not given. Known: {sorted(self._sizes) or '(none)'}."
        )
      return self._sizes[dim]
    return len(dim)

  def _label_axes(self, shape: tuple[Dim, ...]) -> list[str] | None:
    """Labels for the flattened components, when any axis has them."""
    axes: list[list[str]] = []
    labelled = False
    for dim in shape:
      if isinstance(dim, int):
        axes.append([str(i) for i in range(dim)])
      elif isinstance(dim, str):
        names = self._dim_labels.get(dim)
        labelled |= names is not None
        axes.append(names or [str(i) for i in range(self._sizes[dim])])
      else:
        labelled = True
        axes.append([str(x) for x in dim])
    if not labelled:
      return None
    out = [""]
    for axis in axes:
      out = [f"{a}.{b}" if a else b for a in out for b in axis]
    return out

  # The instance.

  def vars(self) -> dict[str, Tensor]:
    """``{name: tensor}`` in declaration order."""
    return {name: getattr(self, name) for name in self._specs}

  def names(self) -> list[str]:
    return list(self._specs)

  def labels(self, name: str) -> list[str] | None:
    """Component names of variable ``name``, or None when no axis is labelled."""
    return self._labels.get(name)

  def dims(self) -> dict[str, int]:
    return dict(self._sizes)

  def __getitem__(self, name: str) -> Tensor:
    if name not in self._specs:
      raise KeyError(f"No variable '{name}'. Declared: {self.names()}")
    return getattr(self, name)

  def __contains__(self, name: object) -> bool:
    return name in self._specs

  def __iter__(self) -> Iterator[str]:
    return iter(self._specs)

  def reset(self, env_ids: Tensor | None = None) -> None:
    """Zero every variable for ``env_ids`` (all of them when None)."""
    for tensor in self.vars().values():
      if env_ids is None:
        tensor.zero_()
      else:
        tensor[env_ids] = 0

  def __repr__(self) -> str:
    inner = ", ".join(f"{n}={tuple(t.shape[1:])}" for n, t in self.vars().items())
    return f"{type(self).__name__}(num_envs={self.num_envs}, {inner})"


# ---------------------------------------------------------------------------
# Pipe stages. Each is a dataclass, so its fields are parameters.
# ---------------------------------------------------------------------------


@dataclass
class Noise:
  """Uniform noise of ``+-half_width`` in the signal's own units. 0 is clean."""

  half_width: float

  def bounds(self) -> dict[str, tuple[float, float | None]]:
    return {"half_width": (0.0, None)}

  def __call__(self, x: Tensor) -> Tensor:
    if self.half_width <= 0.0:
      return x
    return x + (torch.rand_like(x) * 2.0 - 1.0) * self.half_width


@dataclass
class Delay:
  """The signal as it was ``steps`` policy steps ago.

  ``max_steps`` sizes the history and is read once; ``steps`` is live within
  ``[0, max_steps]``. After a reset the history of those environments is
  filled with the next value, so nothing from the previous episode leaks.
  """

  steps: int
  max_steps: Static[int] = -1
  _history: Tensor | None = field(default=None, init=False, repr=False, compare=False)
  _pending: Tensor | None = field(default=None, init=False, repr=False, compare=False)
  _ptr: int = field(default=0, init=False, repr=False, compare=False)

  def __post_init__(self) -> None:
    if self.max_steps < 0:
      self.max_steps = int(self.steps)
    if not 0 <= self.steps <= self.max_steps:
      raise ValueError(f"Delay: steps must be within [0, {self.max_steps}]; got {self.steps}.")

  def bounds(self) -> dict[str, tuple[float, float | None]]:
    return {"steps": (0, self.max_steps)}

  def __call__(self, x: Tensor) -> Tensor:
    length = self.max_steps + 1
    if self._history is None or self._history.shape[1:] != x.shape:
      self._history = x.detach().unsqueeze(0).repeat(length, *([1] * x.ndim)).clone()
      self._pending = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
      self._ptr = 0
    self._ptr = (self._ptr + 1) % length
    self._history[self._ptr] = x
    if self._pending is not None and bool(self._pending.any()):
      self._history[:, self._pending] = x[self._pending]
      self._pending.zero_()
    steps = min(max(int(self.steps), 0), self.max_steps)
    return self._history[(self._ptr - steps) % length]

  def reset(self, env_ids: Tensor | None = None) -> None:
    if self._pending is None:
      return
    if env_ids is None:
      self._pending.fill_(True)
    else:
      self._pending[env_ids] = True


@dataclass
class Scale:
  """Multiply by ``factor``."""

  factor: float

  def __call__(self, x: Tensor) -> Tensor:
    return x * self.factor


@dataclass
class Clip:
  """Clamp to ``[low, high]``."""

  low: float
  high: float

  def __call__(self, x: Tensor) -> Tensor:
    return torch.clamp(x, self.low, self.high)


@dataclass
class Offset:
  """Add ``value``: a number, or a tensor that broadcasts against the signal
  (a per-environment bias drawn once at startup, say). A tensor is not a
  parameter and is not served."""

  value: Any

  def __call__(self, x: Tensor) -> Tensor:
    return x + self.value


class Pipe:
  """A sequence of stages applied in order. Stages with a ``reset`` are told
  about episode resets."""

  def __init__(self, *stages: Any):
    self.stages: tuple[Any, ...] = tuple(stages)

  def __call__(self, x: Tensor) -> Tensor:
    for stage in self.stages:
      x = stage(x)
    return x

  def reset(self, env_ids: Tensor | None = None) -> None:
    for stage in self.stages:
      reset = getattr(stage, "reset", None)
      if callable(reset):
        reset(env_ids)

  def named_stages(self) -> dict[str, Any]:
    """``{name: stage}``: the class name in lower case, numbered on repeats."""
    out: dict[str, Any] = {}
    for stage in self.stages:
      base = type(stage).__name__.lower()
      name, n = base, 1
      while name in out:
        n += 1
        name = f"{base}_{n}"
      out[name] = stage
    return out

  def __repr__(self) -> str:
    return f"Pipe({', '.join(repr(s) for s in self.stages)})"


# ---------------------------------------------------------------------------
# Observation groups.
# ---------------------------------------------------------------------------

Source = str | Callable[[Any], Tensor]
"""Where a term's signal comes from: the name of a variable on the state, or
a function of the state."""


class ObsTerm:
  """One observation: a source, then a pipe. Flattened to ``(num_envs, k)``."""

  def __init__(self, source: Source, *stages: Any):
    self.source = source
    self.pipe = Pipe(*stages)
    self.width: int | None = None

  def __call__(self, state: Any) -> Tensor:
    if isinstance(self.source, str):
      x = state[self.source] if hasattr(state, "__getitem__") else getattr(state, self.source)
    else:
      x = self.source(state)
    if x.ndim == 1:
      x = x.unsqueeze(-1)
    elif x.ndim > 2:
      x = x.flatten(1)
    if not torch.is_floating_point(x):
      x = x.float()
    x = self.pipe(x)
    self.width = int(x.shape[-1])
    return x

  def __repr__(self) -> str:
    source = self.source if isinstance(self.source, str) else getattr(self.source, "__name__", "fn")
    return f"ObsTerm({source!r}, {self.pipe})"


class Obs:
  """A named group of observation terms, concatenated in declaration order.

  Each keyword is a term: a source alone (``commands="commands"``) or a
  tuple of the source and its stages
  (``joint_vel=("joint_vel", Delay(2), Noise(1.5))``). Calling the group with
  the state returns the ``(num_envs, dim)`` tensor; :attr:`dim` and
  :meth:`slices` are known after the first call.
  """

  def __init__(self, **terms: Source | tuple[Any, ...] | ObsTerm):
    self.terms: dict[str, ObsTerm] = {}
    for name, spec in terms.items():
      if isinstance(spec, ObsTerm):
        self.terms[name] = spec
      elif isinstance(spec, tuple):
        self.terms[name] = ObsTerm(*spec)
      else:
        self.terms[name] = ObsTerm(spec)
    self.dim: int | None = None

  def __call__(self, state: Any) -> Tensor:
    out = torch.cat([item(state) for item in self.terms.values()], dim=-1)
    self.dim = int(out.shape[-1])
    return out

  def slices(self) -> dict[str, slice]:
    """Where each term sits in the concatenated vector, after a call."""
    out, start = {}, 0
    for name, item in self.terms.items():
      if item.width is None:
        raise RuntimeError("Obs.slices(): call the group once first.")
      out[name] = slice(start, start + item.width)
      start += item.width
    return out

  def reset(self, env_ids: Tensor | None = None) -> None:
    for item in self.terms.values():
      item.pipe.reset(env_ids)

  def __repr__(self) -> str:
    return "Obs(" + ", ".join(f"{n}={t}" for n, t in self.terms.items()) + ")"


# ---------------------------------------------------------------------------
# Finding the blocks on an environment.
# ---------------------------------------------------------------------------

Block = Obs | Pipe


def blocks(env: Any) -> dict[str, Block]:
  """The :class:`Obs` and :class:`Pipe` attributes of ``env``, by name."""
  try:
    items = vars(env).items()
  except TypeError:
    return {}
  return {name: value for name, value in items if isinstance(value, (Obs, Pipe))}


def stage_leaves(stage: Any) -> Iterator[tuple[str, bool]]:
  """``(field, static)`` for every numeric field of a stage dataclass."""
  if not dataclasses.is_dataclass(stage) or isinstance(stage, type):
    return
  statics = static_fields(stage)
  for f in dataclasses.fields(stage):
    if f.name.startswith("_"):
      continue
    value = getattr(stage, f.name, None)
    if isinstance(value, (bool, int, float)):
      yield f.name, f.name in statics


def block_leaves(block: Block) -> Iterator[tuple[tuple[str, ...], Any, str, bool]]:
  """``(path, stage, field, static)`` for every parameter in a block.

  For an :class:`Obs`, ``path`` is ``(term, stage)``; for a bare
  :class:`Pipe`, ``(stage,)``.
  """
  if isinstance(block, Obs):
    for term_name, term in block.terms.items():
      for stage_name, stage in term.pipe.named_stages().items():
        for field_name, static in stage_leaves(stage):
          yield (term_name, stage_name), stage, field_name, static
  else:
    for stage_name, stage in block.named_stages().items():
      for field_name, static in stage_leaves(stage):
        yield (stage_name,), stage, field_name, static


__all__ = [
    "Clip",
    "Delay",
    "Noise",
    "Obs",
    "ObsTerm",
    "Offset",
    "Pipe",
    "Scale",
    "Static",
    "Term",
    "VarSpec",
    "Vars",
    "block_leaves",
    "blocks",
    "is_static",
    "stage_leaves",
    "static_fields",
    "term",
    "terms",
    "var",
]
