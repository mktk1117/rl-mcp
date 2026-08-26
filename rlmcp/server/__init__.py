"""MCP server exposing a live training session to an agent."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
  if name in ("create_mcp_server", "main"):
    import importlib

    return getattr(importlib.import_module("rlmcp.server.mcp_server"), name)
  raise AttributeError(name)
