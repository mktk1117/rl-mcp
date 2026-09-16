"""Physics backends an articulated robot can run on, interchangeably.

::

    from rlmcp.backends import RobotSpec, make_backend

    sim = make_backend("mjwarp", RobotSpec(xml="robot.xml"),
                       num_envs=4096, dt=0.005, decimation=4)
    # [mjwarp] robot.xml: 7 joints (every single-dof joint in the file); gains
    # from the file's actuators; ... contacts lf_down, rf_down (leaf bodies
    # that can collide); ...

    sim.reset(ids, dof_pos=sim.default_dof_pos.expand(n, -1))
    sim.set_dof_targets(targets)
    sim.step()
    sim.dof_pos, sim.contact_forces, sim.root_pos (floating base only), ...

Only the MJCF is required: joints, gains, the default pose, the base and
the contacts are read off the file and reported, and a spec field overrides
any of them. ``"mjwarp"`` (MuJoCo Warp, GPU), ``"mjbatch"`` (C MuJoCo on CPU
threads) and ``"genesis"`` answer to the same
:class:`~rlmcp.backends.base.SimBackend` contract; see that module for the
vocabulary and the frame conventions. Nothing here imports a simulator until
a backend is constructed, so the name of one you do not have is a reason,
not a traceback.
"""

from __future__ import annotations

from typing import Any

from rlmcp.backends.base import FixedBase, RobotSpec, SimBackend, SimOptions

BACKENDS: dict[str, tuple[str, str]] = {
    "mjwarp": ("rlmcp.backends.mjwarp", "MjWarpBackend"),
    "mjbatch": ("rlmcp.backends.mjbatch", "MjBatchBackend"),
    "genesis": ("rlmcp.backends.genesis", "GenesisBackend"),
}
"""Backend name -> (module, class). Adding one is a line here."""


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
  module = importlib.import_module(module_name)
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
    "FixedBase",
    "RobotSpec",
    "SimBackend",
    "SimOptions",
    "available",
    "backend_class",
    "make_backend",
]
