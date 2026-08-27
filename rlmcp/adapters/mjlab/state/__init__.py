"""Reading and steering live environment state.

Split by what each piece touches: per-step sampling, batch metrics, the terrain
grid, offscreen rendering, the live browser view. Nothing here knows about
parameters -- that is the job of :mod:`rlmcp.adapters.manager_based.access`.
"""

from rlmcp.adapters.manager_based.sampling import StateSampler
from rlmcp.adapters.mjlab.state.terrain import TerrainControl

__all__ = ["StateSampler", "TerrainControl"]
