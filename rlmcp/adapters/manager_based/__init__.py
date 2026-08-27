"""What every manager-based environment has in common.

mjlab and IsaacLab describe a task the same way: manager term configs that are
ordinary dataclasses the running environment re-reads, a scene of named
entities, and per-env buffers on the environment object. That shared shape --
not either simulator -- is what rlmcp's parameter discovery, trace sampling and
summary metrics are actually written against, which is why they live here
rather than inside one backend's package.

A backend adapter is then only what is genuinely its own: how a robot is found
in the scene, how a frame is rendered, and whatever its simulator exposes that
the others do not.
"""

from rlmcp.adapters.manager_based.access import ParameterAccess
from rlmcp.adapters.manager_based.metrics import summary_metrics
from rlmcp.adapters.manager_based.sampling import StateSampler

__all__ = ["ParameterAccess", "StateSampler", "summary_metrics"]
