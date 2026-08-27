"""mjlab backend: sim adapter, env wrapper, and the rsl_rl runner adapter."""

from rlmcp.adapters.mjlab.env_wrapper import (
    RlMcpEnvWrapper,
    TrainingStopped,
    wrap,
)
from rlmcp.adapters.mjlab.sim_adapter import MjlabSimAdapter
from rlmcp.adapters.rsl_rl_runner import RslRlRunnerAdapter

# The runner adapter is rsl_rl's rather than mjlab's -- IsaacLab drives the same
# library -- but it was published under this name, so it stays importable.
MjlabRunnerAdapter = RslRlRunnerAdapter

__all__ = [
    "MjlabRunnerAdapter",
    "MjlabSimAdapter",
    "RlMcpEnvWrapper",
    "RslRlRunnerAdapter",
    "TrainingStopped",
    "wrap",
]
