"""The single-file family: an ``env.py`` whose config is one declared dataclass.

:class:`SingleFileEnv` is the contract and :mod:`rlmcp.blocks` the parts
it is built from; ``wrap`` attaches rlmcp to an instance of it (or to
anything of the same shape). The simulator underneath is
whatever ``env.sim`` is, and this package does not care which. See
``docs/single-file.md``.
"""

from rlmcp.adapters.single_file.algorithm import AlgorithmAdapter
from rlmcp.adapters.single_file.base import SingleFileEnv
from rlmcp.adapters.single_file.env_wrapper import (
    RlMcpEnvWrapper,
    TrainingStopped,
    wrap,
)
from rlmcp.adapters.single_file.sim_adapter import SingleFileSimAdapter
from rlmcp.adapters.single_file.spec import NotASingleFileEnv, SingleFileSpec
from rlmcp.adapters.single_file.state import StateSampler

__all__ = [
    "AlgorithmAdapter",
    "NotASingleFileEnv",
    "RlMcpEnvWrapper",
    "SingleFileEnv",
    "SingleFileSimAdapter",
    "SingleFileSpec",
    "StateSampler",
    "TrainingStopped",
    "wrap",
]
