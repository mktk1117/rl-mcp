"""Where the shared wrapper used to live, re-exported so imports keep working.

The wrapper moved up to :mod:`rlmcp.adapters.env_wrapper` when it became clear
that nothing in it was manager-based: it is the same servicing, telemetry and
records work whichever simulator is underneath, which is exactly why a backend
that is *not* manager-based can subclass it unchanged. This module stays so that
anything importing it from its old home is not broken by the move.

Import from :mod:`rlmcp.adapters.env_wrapper` in new code.
"""

from rlmcp.adapters.env_wrapper import (
    CurriculumArg,
    RlMcpEnvWrapper,
    TrainingStopped,
    serve_a_live_view,
)

__all__ = [
    "CurriculumArg",
    "RlMcpEnvWrapper",
    "TrainingStopped",
    "serve_a_live_view",
]
