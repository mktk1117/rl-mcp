"""Single-file backend: an ``env.py`` whose config is one declared dataclass.

The simulator underneath is whatever ``env.sim`` is -- MuJoCo Warp, mjbatch,
Genesis -- and this package does not care which. See ``docs/single-file.md``.
"""

from rlmcp.adapters.single_file.algorithm import AlgorithmAdapter
from rlmcp.adapters.single_file.env_wrapper import (
    RlMcpEnvWrapper,
    TrainingStopped,
    wrap,
)
from rlmcp.adapters.single_file.sim_adapter import SingleFileSimAdapter
from rlmcp.adapters.single_file.spec import NotASingleFileEnv, SingleFileSpec

__all__ = [
    "AlgorithmAdapter",
    "NotASingleFileEnv",
    "RlMcpEnvWrapper",
    "SingleFileSimAdapter",
    "SingleFileSpec",
    "TrainingStopped",
    "wrap",
]
