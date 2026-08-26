"""The MCP server module without its SDK.

Deliberately *not* behind ``importorskip("mcp")`` -- these are about what
happens when the SDK is absent, so skipping them without it would skip exactly
the case they exist for. They run in both CI dependency variants.
"""

from __future__ import annotations

import pytest


def test_the_module_imports_without_the_sdk(monkeypatch):
  """Importing the server must be harmless; only using it needs the SDK.

  It used to raise `SystemExit` at import time, so anything that merely read
  the module -- a test monkeypatching `main`, a tool walking entry points --
  died instead of getting an error it could handle.
  """
  from rlmcp.server import mcp_server

  monkeypatch.setattr(mcp_server, "MCPServer", None)

  with pytest.raises(ImportError, match="No MCP server SDK"):
    mcp_server.create_mcp_server(root="/nonexistent")


def test_serve_reports_the_missing_sdk_rather_than_dying(monkeypatch, capsys):
  """`rlmcp serve` keeps its friendly message and its non-zero exit."""
  from rlmcp.server import mcp_server

  monkeypatch.setattr(mcp_server, "MCPServer", None)

  assert mcp_server.main([]) == 1
  assert "pip install mcp" in capsys.readouterr().err
