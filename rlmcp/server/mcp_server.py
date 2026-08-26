"""MCP server: hand a running training job to an LLM agent.

The server is a thin client of the session directory -- it holds no simulator
state and can be started, killed and restarted while training continues. Tools
that produce pictures return the image itself *and* its path, so an agent can
look at the frame and a human can open the same file.

One server, one run: the session is pinned at startup and every payload names
it, so mid-conversation the data can never quietly start coming from a
different run. If the trainer dies, data tools keep answering from the files
it left behind and command tools say plainly that it is gone; switching runs
is an explicit act (the ``switch_session`` tool).

The tools here are the ones every run has. Anything specific to a kind of task
-- terrain control on a legged robot, for instance -- arrives through the run's
extensions and is reachable via ``list_commands`` and ``run_command``, so this
server needs no update when an environment gains a new capability.

Run it with::

    rlmcp-server --root logs/rsl_rl          # newest session under that root
    rlmcp-server --session logs/.../rlmcp   # a specific run

or register it with Claude Code::

    claude mcp add rlmcp -- rlmcp-server --root /path/to/logs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# The decorator-based server class moved between SDK generations: it is
# ``MCPServer`` in mcp>=2, ``FastMCP`` in mcp 1.x, and the standalone ``fastmcp``
# package outside the official SDK. The surface we use (tool/resource decorators,
# Image, run) is the same in all three.
SDK_MISSING = (
    "No MCP server SDK found in this interpreter. Install one with "
    "`pip install mcp` (or `uv pip install mcp`) -- it is deliberately an "
    "optional dependency, since the training process does not need it."
)

try:
  from mcp.server.mcpserver import Image, MCPServer as _Server
except ImportError:
  try:
    from mcp.server.fastmcp import FastMCP as _Server, Image
  except ImportError:
    try:
      from fastmcp import FastMCP as _Server  # type: ignore
      from fastmcp.utilities.types import Image  # type: ignore
    except ImportError:  # pragma: no cover - depends on the install.
      # Importing this module must stay harmless: something that merely reads
      # it -- a test monkeypatching `main`, a tool listing entry points -- gets
      # no server, not a dead interpreter. `create_mcp_server` and `main` are
      # where the absence actually matters, and they say so there.
      _Server = None  # type: ignore[assignment]
      Image = None  # type: ignore[assignment]

MCPServer = _Server

from rlmcp.session import Session, iter_sessions

DEFAULT_TIMEOUT = 180.0


class _SessionHandle:
  """Holds the one session this server is pinned to.

  Resolution happens once -- an explicit ``--session``, else the newest session
  under ``--root`` the first time anything asks -- and then never changes on
  its own. A server that silently reattached to a different run would have the
  agent reading one run's metrics while steering another; when the pinned
  trainer dies, tools say so (and data tools answer from disk) rather than
  hopping to whatever run looks newest. Retargeting is only ever explicit,
  through the ``switch_session`` tool.
  """

  def __init__(self, session_dir: Optional[str], root: Optional[str]):
    self.explicit = session_dir
    self.root = root or "."
    self._pinned: Optional[Session] = None

  def get(self) -> Session:
    if self._pinned is None:
      if self.explicit:
        self._pinned = Session.open(self.explicit)
      else:
        found = Session.find_latest(self.root)
        if found is None:
          raise RuntimeError(
              f"No rlmcp session found under '{self.root}'. Start a training "
              "run wrapped with rlmcp, point the server at a session "
              "directory, or call switch_session once one exists."
          )
        self._pinned = found
    return self._pinned

  def switch(self, target: str = "newest") -> Session:
    """Explicitly re-pin: a directory path, or "newest" under the root."""
    if target == "newest":
      found = Session.find_latest(self.root)
      if found is None:
        raise RuntimeError(f"No rlmcp session found under '{self.root}'.")
      self._pinned = found
    else:
      self._pinned = Session.open(target)
    self.explicit = str(self._pinned.dir)
    return self._pinned


def _session_key(session: Session) -> str:
  """Short id stamped into every payload: enough to tell runs apart.

  Session directories are conventionally ``<run_dir>/rlmcp``, so the basename
  alone would name every run the same; the parent disambiguates.
  """
  parent = session.dir.parent.name
  return f"{parent}/{session.dir.name}" if parent else session.dir.name


def _dead_error(session: Session, live: Dict[str, Any], cmd: str) -> Dict[str, Any]:
  """What a live-command tool answers once the pinned trainer is gone."""
  out = {
      "ok": False,
      "session": _session_key(session),
      "session_dir": str(session.dir),
      "error": (
          f"The training process for this session is dead; '{cmd}' needs a "
          "live trainer and will not be serviced."
      ),
      "hint": (
          "This run's data is still on disk: get_training_status, get_metrics, "
          "list_metrics, plot_metrics, get_events, list_parameters and "
          "list_artifacts keep answering for it. To steer a different run, "
          "call switch_session(\"newest\") or switch_session(<dir>)."
      ),
      "last_status": session.status(),
  }
  if "note" in live:
    out["note"] = live["note"]
  return out


def _call(handle: _SessionHandle, cmd: str, timeout: float = DEFAULT_TIMEOUT, **args: Any) -> Dict[str, Any]:
  """Send ``cmd`` to the pinned trainer; refuse up front when it is dead.

  "Dead" here is :meth:`Session.liveness`, not the bare pid: a pid that exists
  under a day-stale heartbeat is presumed recycled. A merely *stalled* trainer
  still gets the command -- it may be one long iteration away from servicing
  it, and the request TTL refuses it truthfully if not.
  """
  session = handle.get()
  live = session.liveness_info()
  if live["state"] == "dead":
    return _dead_error(session, live, cmd)
  clean = {k: v for k, v in args.items() if v is not None}
  response = session.call(cmd, timeout=timeout, **clean)
  if response.ok:
    result = response.result if isinstance(response.result, dict) else {"result": response.result}
    # "session" is reserved: the server's identity stamp must win over any
    # same-named key a command result carries.
    return {"ok": True, **result, "session": _session_key(session)}
  return {"ok": False, "session": _session_key(session), "error": response.error}


# Post-mortem: once the trainer is dead, everything worth asking is still on
# disk, and the CLI already knows how to read it. These mirror its offline
# fallbacks so the MCP surface keeps answering; only commands that need a
# process (parameter edits, rendering, checkpoints) refuse.


def _offline_metric_names(session: Session, contains: Optional[str] = None) -> List[str]:
  rows = session.metrics(last_n=1)
  names = sorted({k for row in rows for k in row if k not in ("iteration", "t")})
  if contains:
    names = [n for n in names if contains.lower() in n.lower()]
  return names


def _offline_metrics_payload(
    session: Session, names: Optional[List[str]], last_n: int
) -> Dict[str, Any]:
  from rlmcp.cli import _default_metric_names, _offline_series

  chosen = list(names) if names else _default_metric_names(session)
  payload: Dict[str, Any] = {
      "ok": True,
      "session": _session_key(session),
      "live": False,
      "source": "metrics.jsonl",
      "metrics": _offline_series(session, chosen, last_n=last_n),
  }
  try:
    from rlmcp.core.diagnostics import summarize_metric_history

    # The summary's window math looks at the last ~40 points; a bounded tail
    # feeds it fully without re-reading the whole file.
    payload["summary"] = summarize_metric_history(
        session.metrics(last_n=max(last_n, 200)), chosen
    )
  except ImportError:
    pass  # The summary needs numpy; the series alone still answers.
  return payload


def _offline_plot_payload(
    session: Session,
    names: Optional[List[str]],
    last_n: int,
    smooth: int,
    title: Optional[str] = None,
) -> Dict[str, Any]:
  from rlmcp.cli import _default_metric_names, _offline_series

  try:
    from rlmcp.core.telemetry.plotter import plot_metric_series
  except ImportError as exc:
    return {
        "ok": False,
        "session": _session_key(session),
        "error": f"Offline plotting needs matplotlib in the server's interpreter ({exc}).",
    }
  chosen = list(names) if names else _default_metric_names(session)
  series = _offline_series(session, chosen, last_n=last_n)
  if not any(series.values()):
    return {
        "ok": False,
        "session": _session_key(session),
        "error": f"No data for {chosen}.",
        "available": _offline_metric_names(session)[:40],
    }
  markers = [
      (float(e["iteration"]), str(e.get("to", "")))
      for e in session.events(last_n=2000)
      if e.get("kind") == "curriculum_stage" and isinstance(e.get("iteration"), (int, float))
  ]
  png = plot_metric_series(
      {k: [tuple(p) for p in v] for k, v in series.items()},
      title=title or f"{session.dir.parent.name} (offline)",
      smooth_window=max(1, smooth),
      markers=markers,
  )
  path = session.artifact_path("metrics_offline.png")
  path.write_bytes(png)
  return {
      "ok": True,
      "session": _session_key(session),
      "live": False,
      "source": "metrics.jsonl",
      "image_path": str(path),
      "metrics": chosen,
  }


def _offline_parameters_payload(
    session: Session, category: Optional[str], contains: Optional[str]
) -> Dict[str, Any]:
  schema = session.params()
  items = {
      k: v
      for k, v in schema.items()
      if (not contains or contains.lower() in k.lower())
      and (not category or (isinstance(v, dict) and v.get("category") == category))
  }
  return {
      "ok": True,
      "session": _session_key(session),
      "live": False,
      "source": "params.json",
      "count": len(items),
      "parameters": items,
  }


def _status_payload(handle: _SessionHandle) -> Dict[str, Any]:
  session = handle.get()
  live = session.liveness_info()
  return {
      "ok": True,
      "session": _session_key(session),
      "session_dir": str(session.dir),
      "state": live["state"],
      "alive": live["pid_alive"],
      "heartbeat_age_s": live["heartbeat_age_s"],
      **({"liveness_note": live["note"]} if "note" in live else {}),
      **session.status(),
  }


def _events_payload(session: Session, last_n: int) -> Dict[str, Any]:
  rows = session.events(last_n=last_n)
  return {
      "ok": True,
      "session": _session_key(session),
      "count": len(rows),
      "events": rows,
  }


def _artifacts_payload(session: Session) -> Dict[str, Any]:
  rows = []
  if session.artifacts.exists():
    for path in session.artifacts.iterdir():
      try:
        stat = path.stat()
      except OSError:
        continue
      if path.is_file():
        rows.append({
            "name": path.name,
            "path": str(path),
            "bytes": stat.st_size,
            "modified_at": stat.st_mtime,
        })
  rows.sort(key=lambda r: r["modified_at"], reverse=True)
  return {
      "ok": True,
      "session": _session_key(session),
      "count": len(rows),
      "artifacts": rows,
  }


def _sessions_payload(handle: _SessionHandle) -> List[Dict[str, Any]]:
  pinned: Optional[str] = None
  if handle._pinned is not None:
    pinned = str(handle._pinned.dir)
  out = []
  for session in iter_sessions(handle.root):  # Newest first, by started_at.
    info = session.info()
    live = session.liveness_info()
    out.append({
        "session_dir": str(session.dir),
        "session": _session_key(session),
        "task": info.get("task"),
        "num_envs": info.get("num_envs"),
        "started_at": info.get("started_at"),
        "state": live["state"],
        "iteration": session.status().get("iteration"),
        "pinned": str(session.dir) == pinned,
    })
  return out


def _switch_payload(handle: _SessionHandle, target: str) -> Dict[str, Any]:
  try:
    session = handle.switch(target)
  except (FileNotFoundError, RuntimeError) as exc:
    return {"ok": False, "error": str(exc)}
  info = session.info()
  return {
      "ok": True,
      "session": _session_key(session),
      "session_dir": str(session.dir),
      "state": session.liveness(),
      "task": info.get("task"),
      "started_at": info.get("started_at"),
      "iteration": session.status().get("iteration"),
  }


# Tool bodies for the dead-or-alive data tools, kept at module level so the
# live/offline routing is plain to read (and testable without an MCP client).


def _list_metrics_impl(handle: _SessionHandle, contains: Optional[str]) -> Dict[str, Any]:
  session = handle.get()
  if session.liveness() == "dead":
    names = _offline_metric_names(session, contains=contains)
    return {"ok": True, "session": _session_key(session), "live": False,
            "count": len(names), "metrics": names}
  return _call(handle, "list_metrics", contains=contains)


def _get_metrics_impl(
    handle: _SessionHandle, names: Optional[List[str]], last_n: int
) -> Dict[str, Any]:
  session = handle.get()
  if session.liveness() == "dead":
    return _offline_metrics_payload(session, names, last_n)
  return _call(handle, "get_metrics", names=names, last_n=last_n)


def _plot_metrics_impl(
    handle: _SessionHandle,
    names: Optional[List[str]],
    last_n: int,
    smooth: int,
    title: Optional[str],
) -> Any:
  session = handle.get()
  if session.liveness() == "dead":
    payload = _offline_plot_payload(session, names, last_n, smooth, title=title)
  else:
    payload = _call(handle, "plot_metrics", names=names, last_n=last_n,
                    smooth=smooth, title=title)
  return _image_result(payload)


def _list_parameters_impl(
    handle: _SessionHandle, category: Optional[str], contains: Optional[str]
) -> Dict[str, Any]:
  session = handle.get()
  if session.liveness() == "dead":
    return _offline_parameters_payload(session, category, contains)
  return _call(handle, "list_parameters", category=category, contains=contains)


#: Images above this many bytes are downscaled before being attached to a tool
#: reply; typical MCP clients cap messages around 1MB, and base64 adds a third.
IMAGE_BYTE_LIMIT = 800_000
#: Longest side, in pixels, that an oversized image is downscaled to.
IMAGE_MAX_DIM = 1024


def _image_format(suffix: str) -> str:
  """Map a filename suffix to an SDK image format with a standard MIME type.

  The SDK builds the MIME type as ``image/<format>``, so ``jpg`` must become
  ``jpeg`` -- ``image/jpg`` is not a registered type and some clients drop it.
  """
  fmt = suffix.lower().lstrip(".")
  return {"jpg": "jpeg", "": "png"}.get(fmt, fmt)


def _prepare_image(
    path: Path, byte_limit: int = IMAGE_BYTE_LIMIT, max_dim: int = IMAGE_MAX_DIM
) -> Tuple[Optional[bytes], str, Optional[str]]:
  """Read an image, re-encoding it when it would blow the reply budget.

  Returns ``(data, format, note)``. Files at or under ``byte_limit`` pass
  through untouched. Larger ones are downscaled to ``max_dim`` on the longest
  side and re-encoded (PNG first for PNG sources, then JPEG). When even that
  cannot fit the budget, ``data`` is ``None`` and ``note`` says why, so the
  caller can fall back to the file path.
  """
  fmt = _image_format(path.suffix)
  data = path.read_bytes()
  if len(data) <= byte_limit:
    return data, fmt, None
  original_kb = len(data) // 1024
  try:
    from PIL import Image as PILImage
  except ImportError:
    return None, fmt, (
        f"image is {original_kb}KB, above the {byte_limit // 1024}KB reply "
        "budget, and Pillow is unavailable to downscale it; open image_path "
        "directly instead."
    )
  import io

  with PILImage.open(io.BytesIO(data)) as img:
    img.load()
    if max(img.size) > max_dim:
      img.thumbnail((max_dim, max_dim))
    # PNG keeps plot lines and text crisp; JPEG is the fallback when the
    # downscaled PNG is still too dense.
    attempts = [("png", "PNG")] if fmt == "png" else []
    attempts.append(("jpeg", "JPEG"))
    for out_fmt, pil_name in attempts:
      frame = img
      if pil_name == "JPEG" and img.mode not in ("RGB", "L"):
        frame = img.convert("RGB")
      buf = io.BytesIO()
      frame.save(buf, format=pil_name, **({"quality": 85} if pil_name == "JPEG" else {}))
      if buf.tell() <= byte_limit:
        return buf.getvalue(), out_fmt, None
  return None, fmt, (
      f"image is {original_kb}KB and still exceeds the {byte_limit // 1024}KB "
      "reply budget after downscaling; open image_path directly instead."
  )


def _image_result(payload: Dict[str, Any], key: str = "image_path") -> Any:
  """Attach the picture to a payload without dropping the numbers around it.

  Returns ``[payload, Image]``; every supported SDK generation converts that
  list into a JSON text block plus an image block, so the agent sees the
  structured result *and* the frame. Oversized files are downscaled first;
  when even that is not enough, or the file cannot be decoded, the payload
  comes back alone with an ``image_note`` explaining why.
  """
  path = payload.get(key)
  if not payload.get("ok") or not path:
    return payload
  file = Path(path)
  if not file.exists():
    return payload
  try:
    data, fmt, note = _prepare_image(file, byte_limit=IMAGE_BYTE_LIMIT, max_dim=IMAGE_MAX_DIM)
  except Exception as exc:  # A corrupt file must not eat the numeric result.
    return {**payload, "image_note": f"could not read {file.name}: {exc}"}
  if data is None:
    return {**payload, "image_note": note}
  return [payload, Image(data=data, format=fmt)]


def _open_records(root: Optional[str]):
  """Open the record store this server writes feedback into.

  Resolution is the CLI's: the server's ``--records-root``, then
  ``$RLMCP_RECORDS``, then ``./records``. Imported here rather than at module
  scope so a server started only to watch a live run never pays for the records
  layer.
  """
  from rlmcp.records import open_store

  return open_store(root)


def create_mcp_server(
    session_dir: Optional[str] = None,
    root: Optional[str] = None,
    name: str = "rlmcp",
    records_root: Optional[str] = None,
) -> MCPServer:
  """Build the MCP server exposing one pinned training session.

  The session is resolved once -- ``session_dir`` if given, else the newest
  session under ``root`` at first use -- and never silently changes. When the
  pinned trainer dies, data tools switch to reading its files off disk and
  live-command tools explain themselves; ``switch_session`` is the one way to
  attach to a different run.

  ``records_root`` points the feedback tools at a record store; left unset they
  fall back to ``$RLMCP_RECORDS`` and then ``./records``, the same resolution
  ``rlmcp record`` uses.
  """
  if MCPServer is None:
    raise ImportError(SDK_MISSING)

  handle = _SessionHandle(session_dir, root)
  try:
    print(f"[rlmcp-server] pinned session: {handle.get().dir}", file=sys.stderr)
  except (FileNotFoundError, RuntimeError) as exc:
    # No run yet; the first tool call (or switch_session) pins one.
    print(f"[rlmcp-server] no session pinned yet: {exc}", file=sys.stderr)
  mcp = MCPServer(name)

  # Session selection.

  @mcp.tool()
  def list_sessions() -> List[Dict[str, Any]]:
    """List rlmcp sessions under the server's root, newest first.

    Each row carries started_at, a running/stalled/dead state, and whether it
    is the session this server is pinned to.
    """
    return _sessions_payload(handle)

  @mcp.tool()
  def switch_session(target: str = "newest") -> Dict[str, Any]:
    """Re-point every later tool call at a different session -- the only way
    this server ever changes runs.

    Args:
      target: a session directory path, or "newest" for the most recently
        started session under the server's root. An explicit path is not
        bounded by --root: the server is a local operator tool with the same
        reach as the operator's own shell, and --root only scopes discovery
        ("newest", list_sessions), not addressing. Deliberate.

    Returns what it attached to: directory, task, started_at and liveness
    state, so a switch to a dead or stalled run is visible immediately.
    """
    return _switch_payload(handle, target)

  # Status and telemetry.

  @mcp.tool()
  def get_training_status() -> Dict[str, Any]:
    """Iteration, pause state, curriculum stage, headline metrics, liveness.

    Reads the trainer's published heartbeat, so it answers even while the
    training loop is busy -- and after it has died: ``state`` says
    running/stalled/dead, and the rest is the last published status.
    """
    return _status_payload(handle)

  @mcp.tool()
  def list_metrics(contains: Optional[str] = None) -> Dict[str, Any]:
    """List every metric name recorded so far, optionally filtered by substring.

    Answers from metrics.jsonl when the trainer is dead.
    """
    return _list_metrics_impl(handle, contains)

  @mcp.tool()
  def get_metrics(
      names: Optional[List[str]] = None, last_n: int = 30
  ) -> Dict[str, Any]:
    """Recent values plus a trend summary for the named metrics.

    Args:
      names: metric keys, e.g. ["Train/mean_reward", "Episode_Reward/track_linear_velocity"].
      last_n: how many recent iterations to return per metric.

    Works after the trainer has exited: a dead run's series are rebuilt from
    metrics.jsonl on disk (marked ``"live": false``).
    """
    return _get_metrics_impl(handle, names, last_n)

  @mcp.tool()
  def plot_metrics(
      names: Optional[List[str]] = None,
      last_n: int = 400,
      smooth: int = 5,
      title: Optional[str] = None,
  ) -> Any:
    """Plot metric curves and return the chart as an image.

    A live trainer renders the plot itself; for a dead run the server plots
    metrics.jsonl directly, so post-mortems keep their pictures.
    """
    return _plot_metrics_impl(handle, names, last_n, smooth, title)

  # Parameters.

  @mcp.tool()
  def list_parameters(
      category: Optional[str] = None, contains: Optional[str] = None
  ) -> Dict[str, Any]:
    """List tunable parameters with live values, bounds and descriptions.

    Categories: reward, termination, domain_randomization, curriculum, action, rl.
    For a dead run the last params.json snapshot answers instead.
    """
    return _list_parameters_impl(handle, category, contains)

  @mcp.tool()
  def set_parameter(key: str, value: Any, rationale: str) -> Dict[str, Any]:
    """Change a reward weight, randomization range or PPO hyperparameter live.

    Args:
      key: e.g. "reward.action_rate_l2.weight", "rl.entropy_coef",
        "event.interval.push_robot.params.velocity_range.x".
      value: a number, or a two-element [min, max] list for range parameters.
      rationale: why you are making this change; recorded in the event log.
    """
    return _call(handle, "set_parameter", key=key, value=value, rationale=rationale)

  @mcp.tool()
  def reset_parameters(keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Restore parameters to the values they had when training started."""
    return _call(handle, "reset_parameters", keys=keys)

  @mcp.tool()
  def reset_environments(
      env_ids: Optional[List[int]] = None,
      where: Optional[Dict[str, Any]] = None,
      rationale: str = "",
  ) -> Dict[str, Any]:
    """Start fresh episodes in some or all environments.

    Episodes, not parameter values -- ``reset_parameters`` is the other one.
    Use this after an edit that left the robots in a state worth clearing, or
    to see a policy from the start of an episode instead of mid-recovery.

    Args:
      env_ids: explicit environment indices; omit for every environment.
      where: pick them by description instead, in whatever vocabulary this
        run's extensions provide -- e.g. {"terrain": "pyramid_stairs"}. See
        ``get_training_status`` for what this environment supports.
      rationale: why; recorded in the event log.
    """
    return _call(handle, "reset_envs", env_ids=env_ids, where=where,
                 rationale=rationale)

  # Seeing the robot.

  @mcp.tool()
  def take_screenshot(
      env_id: Optional[int] = None,
      where: Optional[Dict[str, Any]] = None,
  ) -> Any:
    """Render one frame of a training environment.

    Args:
      env_id: an explicit environment index.
      where: pick one by description instead, using whatever vocabulary this
        run's extensions provide -- e.g. {"terrain": "pyramid_stairs",
        "level": 2} on a locomotion task. See ``get_training_status`` for what
        this environment supports.
    """
    return _image_result(_call(handle, "screenshot", env_id=env_id, where=where))

  @mcp.tool()
  def record_video(
      seconds: float = 4.0,
      env_id: Optional[int] = None,
      where: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """Record a clip of the policy as it trains and return the video file path.

    Frames are captured during normal rollout steps, so the clip shows real
    training behaviour rather than a separate evaluation.
    """
    return _call(
        handle, "record_video", timeout=max(DEFAULT_TIMEOUT, seconds * 20 + 90),
        seconds=seconds, env_id=env_id, where=where,
    )

  # Motion analysis.

  @mcp.tool()
  def diagnose_motion(
      seconds: float = 4.0,
      env_id: Optional[int] = None,
      where: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """Record per-step joint signals and report smoothness, tracking, effort and gait.

    Returns jerk and high-frequency chatter per joint (which joints are buzzing),
    velocity tracking error, torque load, torso tilt and contact statistics,
    plus a plot of the raw traces.
    """
    return _call(
        handle, "diagnose", timeout=max(DEFAULT_TIMEOUT, seconds * 20 + 90),
        seconds=seconds, env_id=env_id, where=where,
    )

  @mcp.tool()
  def record_trace(
      seconds: float = 4.0,
      env_id: Optional[int] = None,
      where: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """Record joint positions/velocities/torques for one env and save them as .npz."""
    return _call(
        handle, "record_trace", timeout=max(DEFAULT_TIMEOUT, seconds * 20 + 90),
        seconds=seconds, env_id=env_id, where=where,
    )

  @mcp.tool()
  def plot_joint_trace(
      channels: Optional[List[str]] = None,
      components: Optional[List[str]] = None,
      title: Optional[str] = None,
  ) -> Any:
    """Plot the last recorded trace as an image.

    Args:
      channels: which signals to draw, e.g. ["joint_pos", "joint_vel", "joint_torque"].
      components: substring filter on joint names, e.g. ["knee", "ankle"].
    """
    return _image_result(
        _call(handle, "plot_trace", channels=channels, components=components, title=title)
    )

  # Extension commands and curriculum.

  @mcp.tool()
  def list_commands() -> Dict[str, Any]:
    """Every command this run accepts, with a one-line description each.

    Beyond the built-in tools, a run exposes whatever its environment supports
    through extensions -- ``set_terrain`` and ``terrain_status`` on a legged
    locomotion task, nothing of the sort on a manipulation task. Call this to
    find out what is actually available, then invoke with ``run_command``.
    """
    return _call(handle, "help")

  @mcp.tool()
  def run_command(cmd: str, args: Optional[Dict[str, Any]] = None) -> Any:
    """Run any command this run accepts, including extension commands.

    Args:
      cmd: a name from ``list_commands``, e.g. "set_terrain".
      args: keyword arguments for it, e.g. {"terrains": ["flat"], "max_level": 4}.
    """
    result = _call(handle, cmd, timeout=600.0, **(args or {}))
    return _image_result(result)

  @mcp.tool()
  def curriculum_status() -> Dict[str, Any]:
    """Current curriculum stage and how close it is to its promotion conditions."""
    return _call(handle, "curriculum_status")

  @mcp.tool()
  def curriculum_advance(reason: str = "manual") -> Dict[str, Any]:
    """Promote to the next curriculum stage immediately."""
    return _call(handle, "curriculum_advance", reason=reason)

  @mcp.tool()
  def curriculum_goto(stage: str, reason: str = "manual") -> Dict[str, Any]:
    """Jump to a named curriculum stage, forward or backward."""
    return _call(handle, "curriculum_goto", stage=stage, reason=reason)

  @mcp.tool()
  def curriculum_auto(enabled: bool = True) -> Dict[str, Any]:
    """Turn automatic stage promotion on or off."""
    return _call(handle, "curriculum_auto", enabled=enabled)

  # Lifecycle.

  @mcp.tool()
  def pause_training() -> Dict[str, Any]:
    """Pause training between iterations; other tools keep working while paused."""
    return _call(handle, "pause")

  @mcp.tool()
  def resume_training() -> Dict[str, Any]:
    """Resume a paused run."""
    return _call(handle, "resume")

  @mcp.tool()
  def save_checkpoint(tag: str = "", note: str = "") -> Dict[str, Any]:
    """Save policy weights plus curriculum and terrain state under a tag."""
    return _call(handle, "save_checkpoint", timeout=600.0, tag=tag, note=note)

  @mcp.tool()
  def list_checkpoints() -> Dict[str, Any]:
    """List checkpoints saved through rlmcp in this session."""
    return _call(handle, "list_checkpoints")

  @mcp.tool()
  def rollback_to_checkpoint(path: str) -> Dict[str, Any]:
    """Roll back weights, parameters and curriculum state to a saved checkpoint."""
    return _call(handle, "load_checkpoint", timeout=600.0, path=path)

  @mcp.tool()
  def record_feedback(
      text: str,
      kind: str = "steer",
      author: str = "user",
      interpretation: str = "",
  ) -> Dict[str, Any]:
    """Record what a human just said about this run, stamped with the iteration.

    Call this whenever the user steers, corrects, rejects or approves what the
    run is doing -- verbatim in ``text``, your reading of it in
    ``interpretation``. ``kind`` is one of steer, correct, reject, approve,
    observe, constrain. It lands in the run record at close-out, so the reason
    behind a change stays recoverable after the conversation is gone.
    """
    return _call(handle, "feedback", text=text, kind=kind, author=author,
                 interpretation=interpretation)

  @mcp.tool()
  def attach_feedback(
      record_id: str,
      text: str,
      kind: str = "steer",
      author: str = "user",
      interpretation: str = "",
      response: str = "",
      changed: bool = True,
  ) -> Dict[str, Any]:
    """Attach a remark to a run record, live trainer or not.

    ``record_feedback`` is for a run that is still going; this is for one that
    has finished, or for a remark about a record the trainer never saw. The
    entry is appended, never edited, and its index is how a response is
    attached to it later.
    """
    from rlmcp.records.record import Feedback
    from rlmcp.records.store import StoreError

    entry = Feedback(text=text, kind=kind, author=author,
                     interpretation=interpretation, response=response,
                     changed=bool(response) and changed)
    try:
      updated = _open_records(records_root).add_feedback(record_id, entry)
    except StoreError as exc:
      return {"ok": False, "error": str(exc)}
    index = len(updated.feedback) - 1
    return {"ok": True, "record": updated.id, "index": index,
            "feedback": updated.feedback[index].to_dict(),
            "outstanding": updated.feedback[index].outstanding}

  @mcp.tool()
  def answer_feedback(
      record_id: str,
      index: int,
      response: str,
      changed: bool = True,
  ) -> Dict[str, Any]:
    """Record what was done about one remark on a run record.

    ``changed=False`` is a real answer: "looked into it, nothing needed
    changing" is not the same as ignoring it, and recording the difference is
    what keeps the ledger honest.
    """
    from rlmcp.records.store import StoreError

    try:
      updated = _open_records(records_root).answer_feedback(
          record_id, index, response, changed=changed)
    except StoreError as exc:
      return {"ok": False, "error": str(exc)}
    return {"ok": True, "record": updated.id, "index": index,
            "feedback": updated.feedback[index].to_dict()}

  @mcp.tool()
  def get_feedback_timeline(
      kind: Optional[str] = None,
      author: Optional[str] = None,
      outstanding: bool = False,
      limit: Optional[int] = None,
  ) -> Dict[str, Any]:
    """Every remark across the records, oldest first, with what came of it.

    ``outstanding=True`` narrows it to instructions nobody has recorded a
    response to -- the question worth asking before closing a run out.
    """
    rows = _open_records(records_root).feedback_timeline(
        kind=kind, author=author, outstanding=outstanding, limit=limit)
    return {"count": len(rows), "feedback": rows}

  @mcp.tool()
  def set_record_headline(record_id: str, text: str = "") -> Dict[str, Any]:
    """Set the one-sentence summary a tree or a listing shows for a run.

    An empty ``text`` clears it, falling back to the first sentence of the
    run's outcome (or its hypothesis while it is still open).
    """
    from rlmcp.records.store import StoreError

    def set_headline(fresh) -> None:
      fresh.headline = text.strip()

    try:
      updated = _open_records(records_root).update_record(record_id, set_headline)
    except StoreError as exc:
      return {"ok": False, "error": str(exc)}
    if updated is None:
      return {"ok": False, "error": f"No record '{record_id}'."}
    return {"ok": True, "record": updated.id, "headline": updated.one_line(),
            "derived": not updated.headline}

  @mcp.tool()
  def add_note(text: str) -> Dict[str, Any]:
    """Record a note in the session event log, next to parameter changes."""
    return _call(handle, "note", text=text)

  @mcp.tool()
  def get_events(last_n: int = 25) -> Dict[str, Any]:
    """Recent session events: parameter edits, stage changes, checkpoints, notes.

    Reads the event log on disk, so it answers for dead runs too.
    """
    return _events_payload(handle.get(), last_n)

  @mcp.tool()
  def list_artifacts() -> Dict[str, Any]:
    """Files this run produced -- plots, videos, traces -- newest first.

    Purely a disk read; a post-mortem can pull every picture the run left
    behind even though the trainer is gone.
    """
    return _artifacts_payload(handle.get())

  @mcp.tool()
  def stop_training(reason: str = "") -> Dict[str, Any]:
    """Ask the training loop to stop cleanly at the next iteration boundary."""
    return _call(handle, "stop_training", reason=reason)

  # Resources.

  @mcp.resource("rlmcp://status")
  def status_resource() -> str:
    """Live training status as JSON."""
    return json.dumps(handle.get().status(), indent=2)

  @mcp.resource("rlmcp://parameters")
  def parameters_resource() -> str:
    """Full parameter schema as JSON."""
    return json.dumps(handle.get().params(), indent=2)

  @mcp.resource("rlmcp://events")
  def events_resource() -> str:
    """Session event log as JSON."""
    return json.dumps(handle.get().events(last_n=200), indent=2)

  return mcp


def main(argv: Optional[List[str]] = None) -> int:
  if MCPServer is None:
    print(SDK_MISSING, file=sys.stderr)
    return 1

  # Reachable as `rlmcp serve` and as the `rlmcp-server` console script; name
  # the usage line after whichever one the reader typed.
  invoked = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else ""
  parser = argparse.ArgumentParser(
      prog=invoked if invoked.startswith("rlmcp-") else "rlmcp serve",
      description=__doc__.splitlines()[0],
  )
  parser.add_argument("--session", help="Path to a specific session directory")
  parser.add_argument(
      "--root", default=os.environ.get("RLMCP_ROOT", "."),
      help="Directory searched for the newest session (default: cwd)",
  )
  parser.add_argument("--name", default="rlmcp")
  parser.add_argument(
      "--records-root",
      help="Record store the feedback tools write to "
           "(default: $RLMCP_RECORDS, then ./records)",
  )
  args = parser.parse_args(argv)
  create_mcp_server(
      session_dir=args.session or os.environ.get("RLMCP_SESSION"),
      root=args.root,
      name=args.name,
      records_root=args.records_root,
  ).run()
  return 0


if __name__ == "__main__":
  main()
