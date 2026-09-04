"""A run on another machine: :class:`WireSession`, the second transport.

:class:`~rlmcp.session.Session` reaches a run by reading its directory. This
reaches one through ``rlmcp hostd`` (:mod:`rlmcp.hostd`) on the machine the
run is on, over HTTP, and implements the same seventeen names
(:data:`~rlmcp.session.WIRE_SURFACE`) -- so everything written against
:class:`~rlmcp.session.SessionClient` works unchanged with a run it cannot
see. ``address`` is a URL here and a path there, and nothing may parse it.

Plain HTTP with JSON bodies, from the standard library: a lab box behind a
VPN, a container on a rented GPU, or the same machine with
``RLMCP_TRANSPORT=wire`` for honesty's sake (design.md). Dialling *out* from
the host through a relay is a later layer on top of this one; the surface
here does not change for it.

Authentication is one bearer token per host, presented on every request.
It is the host's token, issued by whoever runs the host, never a person's.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import Any

from rlmcp.session import Request, Response

DEFAULT_TIMEOUT_S = 30.0
"""Per request. A ``wait`` asks the host to hold the connection for its own
timeout, so that one is the exception and adds the wait on top."""


class WireError(RuntimeError):
  """The host answered with an error, or did not answer."""


def is_url(address: str) -> bool:
  return str(address).startswith(("http://", "https://"))


class _Http:
  """One host, one token. The only thing here that knows how to make a request."""

  def __init__(self, base: str, token: str = "", timeout: float = DEFAULT_TIMEOUT_S):
    self.base = base.rstrip("/")
    self.token = token or os.environ.get("RLMCP_HOST_TOKEN", "")
    self.timeout = timeout

  def json(self, method: str, path: str, body: Any = None,
           params: dict[str, Any] | None = None, timeout: float | None = None,
           raw: bytes | None = None) -> Any:
    answer = self.bytes(method, path, body, params, timeout, raw=raw)
    return json.loads(answer.decode()) if answer else None

  def bytes(self, method: str, path: str, body: Any = None,
            params: dict[str, Any] | None = None, timeout: float | None = None,
            raw: bytes | None = None) -> bytes:
    url = self.base + path
    if params:
      clean = {k: v for k, v in params.items() if v is not None}
      if clean:
        url += "?" + urllib.parse.urlencode(clean)
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    headers = {"accept": "application/json"}
    if raw is not None:
      headers["content-type"] = "application/octet-stream"
    elif data is not None:
      headers["content-type"] = "application/json"
    if self.token:
      headers["authorization"] = f"Bearer {self.token}"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
      with urllib.request.urlopen(req, timeout=timeout or self.timeout) as answer:
        return answer.read()
    except urllib.error.HTTPError as exc:
      try:
        detail = json.loads(exc.read().decode()).get("error")
      except Exception:
        detail = exc.reason
      raise WireError(f"{method} {path}: {exc.code} {detail}") from None
    except urllib.error.URLError as exc:
      raise WireError(f"{self.base} did not answer: {exc.reason}") from None


class WireSession:
  """A run reached through ``rlmcp hostd``. Satisfies :class:`SessionClient`.

  ``address`` is ``http://host:port/v1/sessions/<key>``, the string
  ``--session`` takes for a remote run and the one payloads carry. The key is
  what :attr:`Session.key` is on the host: ``<run>/<session>``.
  """

  def __init__(self, address: str, token: str = "", timeout: float = DEFAULT_TIMEOUT_S):
    if not is_url(address):
      raise ValueError(f"not a wire address: {address!r} (expected http://…/v1/sessions/<key>)")
    base, marker, key = address.partition("/v1/sessions/")
    if not marker or not key:
      raise ValueError(f"a wire address names a session: {address!r}")
    self._address = address.rstrip("/")
    self._key = urllib.parse.unquote(key.strip("/"))
    self._http = _Http(base, token, timeout)
    self._path = "/v1/sessions/" + urllib.parse.quote(self._key, safe="")

  # -- identity ------------------------------------------------------------

  @property
  def address(self) -> str:
    return self._address

  @property
  def key(self) -> str:
    return self._key

  @property
  def name(self) -> str:
    return self._key.split("/")[0] if "/" in self._key else self._key

  def __repr__(self) -> str:
    return f"WireSession({self._address!r})"

  # -- reads ---------------------------------------------------------------

  def info(self) -> dict[str, Any]:
    return self._http.json("GET", self._path + "/info") or {}

  def status(self) -> dict[str, Any]:
    return self._http.json("GET", self._path + "/status") or {}

  def params(self) -> dict[str, Any]:
    return self._http.json("GET", self._path + "/params") or {}

  def metrics(self, last_n: int | None = None,
              since_seq: int | None = None) -> list[dict[str, Any]]:
    return self._http.json("GET", self._path + "/metrics",
                           params={"last_n": last_n, "since_seq": since_seq}) or []

  def events(self, last_n: int | None = None,
             since_seq: int | None = None) -> list[dict[str, Any]]:
    return self._http.json("GET", self._path + "/events",
                           params={"last_n": last_n, "since_seq": since_seq}) or []

  def metrics_count(self) -> int:
    return int((self._http.json("GET", self._path + "/metrics_count") or {}).get("count", 0))

  def list_artifacts(self) -> list[dict[str, Any]]:
    # No ``path`` in these rows: there is none a caller here could open.
    return self._http.json("GET", self._path + "/artifacts") or []

  def read_artifact(self, name: str) -> bytes:
    if not name or "/" in name or "\\" in name or name in (".", ".."):
      raise ValueError(f"'{name}' is not an artifact name. Pass the `name` from "
                       "list_artifacts(), not a path.")
    return self._http.bytes("GET", self._path + "/artifacts/" + urllib.parse.quote(name, safe=""))

  # -- commands ------------------------------------------------------------

  def submit(self, cmd: str, **args: Any) -> Request:
    answer = self._http.json("POST", self._path + "/submit", {"cmd": cmd, "args": args})
    return Request.from_dict(answer)

  def poll(self, req_id: str, consume: bool = False) -> Response | None:
    answer = self._http.json("GET", self._path + "/poll",
                             params={"req_id": req_id, "consume": int(bool(consume))})
    return Response.from_dict(answer) if answer else None

  def wait(self, req_id: str, timeout: float = 120.0, interval: float = 0.1) -> Response:
    # The host holds the request for `timeout`; this side allows that plus a
    # margin, so a slow trainer is reported by the host's message, not ours.
    answer = self._http.json("GET", self._path + "/wait",
                             params={"req_id": req_id, "timeout": timeout},
                             timeout=timeout + 15.0)
    return Response.from_dict(answer)

  def call(self, cmd: str, timeout: float = 120.0, **args: Any) -> Response:
    req = self.submit(cmd, **args)
    return self.wait(req.req_id, timeout=timeout)

  # -- liveness ------------------------------------------------------------

  def liveness(self) -> str:
    return self.liveness_info()["state"]

  def liveness_info(self) -> dict[str, Any]:
    return self._http.json("GET", self._path + "/liveness_info") or {"state": "dead"}


class WireHost:
  """One ``hostd``: what it is, what runs it has, what it can execute."""

  def __init__(self, url: str, token: str = "", timeout: float = DEFAULT_TIMEOUT_S):
    self.url = url.rstrip("/")
    self._http = _Http(self.url, token, timeout)

  def host(self) -> dict[str, Any]:
    """Identity and state: ``host_id``, GPUs, disk, the sessions root."""
    return self._http.json("GET", "/v1/host") or {}

  def sessions(self, include_play: bool = False) -> Iterator[WireSession]:
    """Every session on the host, newest first, as :class:`WireSession`."""
    rows = self._http.json("GET", "/v1/sessions",
                           params={"include_play": int(include_play)}) or []
    for row in rows:
      yield WireSession(self.url + "/v1/sessions/" + urllib.parse.quote(row["key"], safe=""),
                        token=self._http.token, timeout=self._http.timeout)

  def session(self, key: str) -> WireSession:
    return WireSession(self.url + "/v1/sessions/" + urllib.parse.quote(key, safe=""),
                       token=self._http.token, timeout=self._http.timeout)

  # -- jobs: the host executes ---------------------------------------------

  def submit_job(self, argv: list[str], cwd: str = "", env: dict[str, str] | None = None,
                 label: str = "") -> dict[str, Any]:
    """Start a process on the host. Returns the job as the host describes it."""
    return self._http.json("POST", "/v1/jobs",
                           {"argv": list(argv), "cwd": cwd, "env": dict(env or {}),
                            "label": label}) or {}

  def jobs(self) -> list[dict[str, Any]]:
    return self._http.json("GET", "/v1/jobs") or []

  def job(self, job_id: str, tail: int = 40) -> dict[str, Any]:
    return self._http.json("GET", "/v1/jobs/" + urllib.parse.quote(job_id, safe=""),
                           params={"tail": tail}) or {}

  # -- trees: the code a job runs from ---------------------------------------

  def has_tree(self, tree: str) -> str:
    """The host's path for a tree it already holds, or ""."""
    try:
      return str(self._http.json("GET", "/v1/trees/" + urllib.parse.quote(tree, safe=""))
                 .get("path") or "")
    except WireError as exc:
      if " 404 " in str(exc):
        return ""
      raise

  def push_tree(self, tree: str, archive: bytes) -> str:
    """Send ``git archive`` bytes of ``tree``; returns the host's path for it.

    Idempotent by tree id: a host that has it answers with the path and
    unpacks nothing. A job whose ``cwd`` is that path runs exactly the code
    the launcher stamped, on a machine with no repository at all.
    """
    answer = self._http.json("PUT", "/v1/trees/" + urllib.parse.quote(tree, safe=""),
                             body=None, timeout=self._http.timeout * 10, raw=archive)
    return str(answer.get("path") or "")

  def cancel_job(self, job_id: str) -> dict[str, Any]:
    path = "/v1/jobs/" + urllib.parse.quote(job_id, safe="") + "/cancel"
    return self._http.json("POST", path) or {}


def connect(address: str, token: str = "", timeout: float = DEFAULT_TIMEOUT_S):
  """The right client for an address: a :class:`Session` for a directory, a
  :class:`WireSession` for a URL.

  The one place the two transports meet. Everything that takes ``--session``
  should resolve it here, so a remote run is a different string and nothing
  else.
  """
  if is_url(address):
    return WireSession(address, token=token, timeout=timeout)
  from rlmcp.session import Session
  return Session.open(address)
