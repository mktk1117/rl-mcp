"""Physics backends a single-file environment can run on, interchangeably.

::

    from backends import RobotSpec, make_backend

    robot = RobotSpec(xml="robot.xml", stiffness=20.0, damping=0.5,
                      contact_sites=("foot_left", "foot_right", "base"))
    sim = make_backend("mjwarp", robot, num_envs=4096, dt=0.005, decimation=4)

    sim.reset(ids, root_pos, root_quat, dof_pos)
    sim.set_dof_targets(targets)
    sim.step()
    sim.root_pos, sim.dof_pos, sim.contact_forces, ...

``"mjwarp"`` (MuJoCo Warp, GPU), ``"mjbatch"`` (C MuJoCo on CPU threads) and
``"genesis"`` answer to the same :class:`~backends.base.SimBackend`
contract; see that module for the vocabulary and the frame conventions.
Nothing here imports a simulator until a backend is constructed, so the
name of one you do not have is a reason, not a traceback.
"""

from __future__ import annotations

from typing import Any

from .base import RobotSpec, SimBackend, SimOptions

BACKENDS: dict[str, tuple[str, str]] = {
    "mjwarp": (".mjwarp", "MjWarpBackend"),
    "mjbatch": (".mjbatch", "MjBatchBackend"),
    "genesis": (".genesis", "GenesisBackend"),
}
"""Backend name -> (module, class), relative to this package so it can be
copied next to any task. Adding one is a line here."""


def backend_class(name: str) -> type[SimBackend]:
  """The class behind ``name``, imported now. Raises ImportError with the
  install hint when its simulator is missing."""
  import importlib

  try:
    module_name, class_name = BACKENDS[name]
  except KeyError:
    raise KeyError(
        f"No backend '{name}'. Available: {sorted(BACKENDS)}"
    ) from None
  module = importlib.import_module(module_name, package=__package__)
  return getattr(module, class_name)


def make_backend(name: str, robot: RobotSpec, num_envs: int, dt: float,
                 decimation: int, device: Any = "cuda",
                 options: SimOptions | None = None) -> SimBackend:
  """Construct the backend called ``name`` around ``robot``."""
  return backend_class(name)(robot, num_envs, dt, decimation, device=device, options=options)


def available() -> dict[str, str]:
  """``{name: ""}`` for each backend whose simulator imports, else the reason."""
  import importlib

  out: dict[str, str] = {}
  probes = {"mjwarp": "mujoco_warp", "mjbatch": "mjbatch", "genesis": "genesis"}
  for name, module in probes.items():
    try:
      importlib.import_module(module)
      out[name] = ""
    except Exception as exc:
      out[name] = f"{type(exc).__name__}: {exc}"
  return out


__all__ = [
    "BACKENDS",
    "RobotSpec",
    "SimBackend",
    "SimOptions",
    "available",
    "backend_class",
    "make_backend",
]
