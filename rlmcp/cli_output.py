"""How the CLI prints: raw JSON for a pipe, formatted text for a person.

The CLI has two readers with opposite needs. An agent shells out, captures
stdout and parses it -- for that reader the bytes must stay exactly what they
have always been. A person runs the same command in a terminal and wants to
read the answer, not a wall of JSON.

``sys.stdout.isatty()`` separates them without either side declaring itself: a
pipe means an agent (or the MCP server, or a redirect into a file), a terminal
means a person. ``--json`` / ``--text`` and ``RLMCP_OUTPUT`` override it for
the cases the check cannot see, like a pretty run being teed into a file.

Two rules hold this together:

* **Text mode changes presentation, never content.** Long strings wrap, they
  do not truncate. Whatever is in the JSON is on the screen somewhere.
* **Nothing here is imported in JSON mode's path.** Only the standard library
  is used at all, so the CLI keeps running in a bare interpreter -- the same
  promise the rest of ``rlmcp.cli`` makes.

:func:`render` dispatches on the *trainer* command name (``get_metrics``,
``screenshot``), not the CLI verb, so ``rlmcp shot`` and
``rlmcp run screenshot`` format identically and a future MCP-side pretty
printer can reuse the same table. Commands with no registered renderer fall
through to :func:`render_generic`, which is what makes extension-defined verbs
readable without the core knowing they exist.
"""

from __future__ import annotations

import base64
import datetime as _dt
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Suffixes worth putting in front of a person. Deliberately narrow: a
# checkpoint's ``path`` is a .pt and a close-out's ``report`` is a .md, and
# neither should launch anything.
STILL_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"})
MOTION_SUFFIXES = frozenset({".mp4", ".webm", ".mov", ".gif"})
PAGE_SUFFIXES = frozenset({".html", ".htm"})
MEDIA_SUFFIXES = STILL_SUFFIXES | MOTION_SUFFIXES | PAGE_SUFFIXES


# --------------------------------------------------------------------------
# mode
# --------------------------------------------------------------------------

def resolve_mode(explicit: Optional[str] = None) -> str:
  """``"json"`` or ``"text"``; a pipe gets JSON, a terminal gets text."""
  forced = explicit or os.environ.get("RLMCP_OUTPUT")
  if forced in ("json", "text"):
    return forced
  try:
    return "text" if sys.stdout.isatty() else "json"
  except (AttributeError, ValueError):  # stdout replaced or already closed.
    return "json"


def resolve_open(explicit: Optional[str] = None) -> str:
  """``"auto"`` | ``"never"`` | ``"always"`` -- whether to show artifacts."""
  choice = explicit or os.environ.get("RLMCP_OPEN")
  return choice if choice in ("auto", "never", "always") else "auto"


def _use_color() -> bool:
  if os.environ.get("NO_COLOR"):
    return False
  try:
    return sys.stdout.isatty()
  except (AttributeError, ValueError):
    return False


class _Ink:
  """ANSI codes, or empty strings when colour is off."""

  def __init__(self, enabled: bool):
    self.on = enabled

  def _wrap(self, code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if self.on else text

  def dim(self, text: str) -> str:
    return self._wrap("2", text)

  def bold(self, text: str) -> str:
    return self._wrap("1", text)

  def red(self, text: str) -> str:
    return self._wrap("31", text)

  def green(self, text: str) -> str:
    return self._wrap("32", text)

  def cyan(self, text: str) -> str:
    return self._wrap("36", text)


# Tables want every column the terminal has; prose does not. A paragraph set
# 200 characters wide is measurably harder to read than one set at 100, so the
# two are measured separately rather than sharing one number.
PROSE_WIDTH = 100


def _width() -> int:
  return max(40, shutil.get_terminal_size((100, 24)).columns)


def _prose_room(room: int) -> int:
  return max(20, min(room, PROSE_WIDTH))


# --------------------------------------------------------------------------
# scalars
# --------------------------------------------------------------------------

# Keys the session protocol uses for wall-clock seconds. "t" is the event log's
# and metrics.jsonl's; the rest are the record and status fields.
_TIME_KEYS = frozenset({"t", "at", "time", "timestamp"})


def _looks_like_epoch(key: str, value: Any) -> bool:
  """A float in the seconds-since-1970 range, under a key that means a time."""
  return (
      isinstance(value, float)
      and (key in _TIME_KEYS or key.endswith(("_at", "_time")))
      and 1e9 < value < 4e9
  )


def format_scalar(value: Any, key: str = "") -> str:
  """One value as a person reads it.

  Floats get six significant figures, which keeps ``3e-05`` and ``114.232``
  both legible. Timestamps are the exception: ``%g`` would print a wall-clock
  time as ``1.78756e+09``, so ``*_at`` keys are resolved to local time.
  """
  if value is None:
    return "-"
  if isinstance(value, bool):
    return "true" if value else "false"
  if _looks_like_epoch(key, value):
    return _dt.datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
  if isinstance(value, float):
    return f"{value:.6g}"
  return str(value)


def _is_scalar(value: Any) -> bool:
  return value is None or isinstance(value, (str, int, float, bool))


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

def _columns(rows: Sequence[Dict[str, Any]]) -> List[str]:
  """Union of keys, in the order they first appear."""
  seen: Dict[str, None] = {}
  for row in rows:
    for key in row:
      seen.setdefault(key, None)
  return list(seen)


def render_table(rows: Sequence[Dict[str, Any]], indent: str = "",
                 max_width: Optional[int] = None) -> Optional[List[str]]:
  """A list of flat dicts as aligned columns, numbers right-aligned.

  Returns None when the columns cannot fit ``max_width``. A record's
  hypothesis is a paragraph, and a column sized to hold it pushes every other
  field off the screen -- the caller falls back to one block per row, where
  the paragraph can wrap under its own key.
  """
  columns = _columns(rows)
  ink = _Ink(_use_color())
  cells = [
      [format_scalar(row.get(c), c) if _is_scalar(row.get(c)) else _inline(row.get(c))
       for c in columns]
      for row in rows
  ]
  widths = [
      max(len(columns[i]), max((len(r[i]) for r in cells), default=0))
      for i in range(len(columns))
  ]
  numeric = [
      all(_is_scalar(row.get(c)) and isinstance(row.get(c), (int, float))
          and not isinstance(row.get(c), bool)
          for row in rows if row.get(c) is not None)
      for c in columns
  ]

  def line(values: List[str], header: bool = False) -> str:
    parts = []
    for i, text in enumerate(values):
      parts.append(text.rjust(widths[i]) if numeric[i] and not header
                   else text.ljust(widths[i]))
    return indent + "  ".join(parts).rstrip()

  total = len(indent) + sum(widths) + 2 * max(len(widths) - 1, 0)
  if max_width is not None and total > max_width:
    return None

  out = [ink.dim(line(columns, header=True))]
  out.extend(line(row) for row in cells)
  return out


def render_rows(rows: Sequence[Sequence[Any]], indent: str = "") -> List[str]:
  """Equal-shaped rows of scalars as aligned columns, with no header."""
  cells = [[format_scalar(v) for v in row] for row in rows]
  span = max(len(r) for r in cells)
  widths = [max((len(r[i]) for r in cells if i < len(r)), default=0)
            for i in range(span)]
  numeric = [
      all(isinstance(row[i], (int, float)) and not isinstance(row[i], bool)
          for row in rows if i < len(row))
      for i in range(span)
  ]
  out = []
  for row in cells:
    parts = [row[i].rjust(widths[i]) if numeric[i] else row[i].ljust(widths[i])
             for i in range(len(row))]
    out.append(indent + "  ".join(parts).rstrip())
  return out


def _inline(value: Any) -> str:
  """A short one-line stand-in for a nested value inside a table cell."""
  if isinstance(value, dict):
    return "{" + ", ".join(f"{k}={format_scalar(v, k)}"
                           for k, v in list(value.items())[:3]) + "}"
  if isinstance(value, (list, tuple)):
    if all(_is_scalar(v) for v in value):
      return ", ".join(format_scalar(v) for v in value)
    return f"[{len(value)} items]"
  return format_scalar(value)


def _table_able(value: Any) -> bool:
  """A list of flat-enough dicts, worth aligning into columns."""
  if not isinstance(value, (list, tuple)) or len(value) < 1:
    return False
  if not all(isinstance(v, dict) for v in value):
    return False
  return all(
      _is_scalar(cell) or _is_scalar_list(cell)
      for row in value for cell in row.values()
  )


def _is_scalar_list(value: Any) -> bool:
  return isinstance(value, (list, tuple)) and all(_is_scalar(v) for v in value)


# --------------------------------------------------------------------------
# the generic renderer
# --------------------------------------------------------------------------

def _wrap(text: str, room: int, indent: str = "") -> List[str]:
  """Wrap on whitespace only.

  textwrap's defaults split on hyphens and chop over-long words, which turns a
  session path into ``2026-08-\n24_walk003``. A token with nowhere to break
  is better left overflowing than made unreadable -- and unselectable.
  """
  lines = textwrap.wrap(
      text, width=max(room, 20), break_on_hyphens=False, break_long_words=False,
      initial_indent=indent, subsequent_indent=indent,
  )
  return lines or [indent + text]


def render_block(value: Any, indent: str = "", key: str = "",
                 width: Optional[int] = None) -> List[str]:
  """Any JSON value as indented lines. Wraps long text; never drops it."""
  width = width or _width()
  ink = _Ink(_use_color())

  if _is_scalar(value):
    return [indent + format_scalar(value, key)]

  if isinstance(value, (list, tuple)):
    if not value:
      return [indent + ink.dim("(none)")]
    if _table_able(value):
      table = render_table(value, indent, max_width=width)
      if table is not None:
        return table
    if all(_is_scalar(v) for v in value):
      return _wrap(", ".join(format_scalar(v) for v in value),
                   _prose_room(width - len(indent)), indent
                   ) or [indent + ink.dim("(none)")]
    if all(_is_scalar_list(v) and v for v in value):
      # Rows of the same shape: metric series are [iteration, value] pairs, and
      # a record's metrics are [name, value]. One line each, columns aligned.
      return render_rows(value, indent)
    out: List[str] = []
    for i, item in enumerate(value):
      out.append(indent + ink.dim(f"[{i}]"))
      out.extend(render_block(item, indent + "  ", width=width))
    return out

  if isinstance(value, dict):
    if not value:
      return [indent + ink.dim("(empty)")]
    out = []
    # Scalars first as an aligned key column; nested values get their own
    # block below, so a long hypothesis cannot push the numbers out of line.
    scalars = {k: v for k, v in value.items() if _is_scalar(v)}
    nested = {k: v for k, v in value.items() if not _is_scalar(v)}
    pad = max((len(k) for k in scalars), default=0)
    for key_name, item in scalars.items():
      text = format_scalar(item, key_name)
      label = indent + ink.dim(key_name.ljust(pad)) + "  "
      room = width - len(indent) - pad - 2
      if len(text) <= room or room < 20:
        out.append(label + text)
      else:
        # Long prose (a hypothesis, a falsifier) wraps under its own key
        # rather than being cut -- text mode must not lose content.
        wrapped = _wrap(text, _prose_room(room))
        out.append(label + wrapped[0])
        out.extend(indent + " " * (pad + 2) + line for line in wrapped[1:])
    for key_name, item in nested.items():
      out.append(indent + ink.dim(key_name))
      out.extend(render_block(item, indent + "  ", key=key_name, width=width))
    return out

  return [indent + str(value)]


def render_generic(payload: Any, width: Optional[int] = None) -> str:
  """The fallback: unwrap the ok/result/error envelope, then print the value.

  Every ``_call`` result wears that envelope, including the ones from
  extension verbs this module has never heard of, so unwrapping belongs here
  rather than in any single command's renderer.
  """
  width = width or _width()
  ink = _Ink(_use_color())

  if isinstance(payload, dict) and "ok" in payload:
    rest = {k: v for k, v in payload.items()
            if k not in ("ok", "error", "result") and v is not None}
    if not payload.get("ok"):
      lines = [ink.red("error: ") + str(payload.get("error") or "failed")]
      if rest:
        lines.extend(render_block(rest, "  ", width=width))
      return "\n".join(lines)
    result = payload.get("result")
    if result is None and not rest:
      return ink.green("ok")
    lines = []
    if result is not None:
      lines.extend(render_block(result, "", width=width))
    if rest:
      lines.extend(render_block(rest, "", width=width))
    return "\n".join(lines) or ink.green("ok")

  return "\n".join(render_block(payload, "", width=width))


_RENDERERS: Dict[str, Callable[[Any], str]] = {}


def register(command: str) -> Callable[[Callable[[Any], str]], Callable[[Any], str]]:
  """Attach a hand-written renderer to one trainer command name."""

  def decorate(fn: Callable[[Any], str]) -> Callable[[Any], str]:
    _RENDERERS[command] = fn
    return fn

  return decorate


def render(payload: Any, command: Optional[str] = None) -> str:
  fn = _RENDERERS.get(command or "")
  return fn(payload) if fn else render_generic(payload)


# --------------------------------------------------------------------------
# showing artifacts
# --------------------------------------------------------------------------

def find_artifacts(payload: Any) -> List[Path]:
  """Media files named anywhere in a result, in the order they appear.

  Keyed on the suffix rather than the key name: the controller already writes
  ``image_path`` / ``video_path`` / ``path``, but an extension is free to
  invent its own key and should still get its picture shown.
  """
  found: List[Path] = []

  def walk(value: Any) -> None:
    if isinstance(value, dict):
      for item in value.values():
        walk(item)
    elif isinstance(value, (list, tuple)):
      for item in value:
        walk(item)
    elif isinstance(value, str) and len(value) < 4096:
      if not value or "\n" in value:
        return
      path = Path(value)
      if path.suffix.lower() in MEDIA_SUFFIXES and path not in found:
        try:
          if path.is_file():
            found.append(path)
        except OSError:
          pass

  walk(payload)
  return found


def _kitty_capable() -> bool:
  term = os.environ.get("TERM", "")
  program = os.environ.get("TERM_PROGRAM", "")
  return bool(
      os.environ.get("KITTY_WINDOW_ID")
      or "kitty" in term
      or "ghostty" in term
      or os.environ.get("GHOSTTY_RESOURCES_DIR")
      or program in ("ghostty", "WezTerm")
      or os.environ.get("WEZTERM_PANE")
  )


def _kitty_show(path: Path) -> bool:
  """Draw a PNG inline with the kitty graphics protocol.

  ``f=100`` hands the terminal the file's own PNG bytes, so no decoder is
  needed on this side -- which is the whole reason this path exists in a
  standard-library-only module.
  """
  try:
    data = base64.standard_b64encode(path.read_bytes())
  except OSError:
    return False
  chunk = 4096
  try:
    for offset in range(0, len(data), chunk):
      piece = data[offset:offset + chunk]
      more = 1 if offset + chunk < len(data) else 0
      control = f"a=T,f=100,m={more}" if offset == 0 else f"m={more}"
      sys.stdout.write(f"\033_G{control};{piece.decode('ascii')}\033\\")
    sys.stdout.write("\n")
    sys.stdout.flush()
  except (OSError, ValueError):
    return False
  return True


def _spawn(argv: List[str]) -> bool:
  """Launch a viewer without waiting on it, and without it holding the shell."""
  try:
    subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # Survives this process; closing it is the user's.
    )
  except (OSError, ValueError):
    return False
  return True


def show(path: Path) -> Optional[str]:
  """Put one artifact in front of a person; returns how, or None if it didn't.

  A ladder, best first: inline in the terminal if it can draw, then a
  converter if one is installed, then the desktop's own viewer. Callers must
  not reach here in JSON mode -- an agent wants the path, which it can read
  back itself, not a window opening on someone's screen.
  """
  suffix = path.suffix.lower()
  still = suffix in STILL_SUFFIXES and suffix not in PAGE_SUFFIXES
  motion = suffix in MOTION_SUFFIXES

  if still and not motion:
    if suffix == ".png" and _kitty_capable() and _kitty_show(path):
      return "inline"
    for tool, argv in (
        ("imgcat", ["imgcat", str(path)]),
        ("chafa", ["chafa", "--animate=off", str(path)]),
        ("viu", ["viu", "-s", str(path)]),
    ):
      if shutil.which(tool):
        try:
          subprocess.run(argv, check=False, timeout=10)
        except (OSError, subprocess.SubprocessError):
          continue
        return tool

  if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
    if shutil.which("xdg-open") and _spawn(["xdg-open", str(path)]):
      return "xdg-open"
    if sys.platform == "darwin" and _spawn(["open", str(path)]):
      return "open"
  return None


# A command that made a picture names one. A result naming a handful is a
# listing -- a record's whole asset shelf -- and opening a window per file is
# not what asking to see a run's history means. --open raises the bar without
# removing it.
AUTO_LIMIT = 2
OPEN_LIMIT = 8


def show_artifacts(payload: Any, policy: str = "auto"
                   ) -> Tuple[List[Tuple[Path, str]], List[Path]]:
  """Show the artifacts a result names.

  Returns ``(shown, held_back)`` -- the second list is what the cap or a
  missing viewer kept off the screen, so the caller can say so. Paths are
  printed either way; this only decides what opens.
  """
  found = find_artifacts(payload)
  if policy == "never" or not found:
    return [], []
  limit = AUTO_LIMIT if policy == "auto" else OPEN_LIMIT
  if len(found) > limit:
    return [], found

  shown: List[Tuple[Path, str]] = []
  held: List[Path] = []
  for path in found:
    how = show(path)
    (shown.append((path, how)) if how else held.append(path))
  return shown, held


def note(text: str) -> str:
  """A dimmed aside -- 'opened X', 'nothing to show'. Never carries content."""
  return _Ink(_use_color()).dim(text)
