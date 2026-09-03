"""A run on another machine: ``rlmcp hostd`` and :class:`WireSession`.

The daemon runs on a thread in this process, bound to a free localhost port,
over a real session directory; the client reaches it over real HTTP. Every
test asks the same question of both transports -- the directory and the
wire -- and requires the same answer, which is the whole claim of
``WIRE_SURFACE``: a reader that stays inside it does not know where the run
is.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from rlmcp import hostd, wire
from rlmcp.session import WIRE_SURFACE, Response, Session, SessionClient


@pytest.fixture
def root(tmp_path) -> Path:
  return tmp_path / "logs"


@pytest.fixture
def local(root) -> Session:
  """A session the way a trainer leaves it: created, with a little history."""
  session = Session(root / "run-a" / "rlmcp").create({"task": "Demo-Task", "num_envs": 8})
  session.publish_status({"iteration": 3, "state": "training"})
  session.append_metrics(1, {"reward": 0.5})
  session.append_metrics(2, {"reward": 0.7})
  session.append_metrics(3, {"reward": 0.9})
  session.append_event("note", {"text": "hello"})
  session.artifacts.mkdir(parents=True, exist_ok=True)
  (session.artifacts / "frame.png").write_bytes(b"\x89PNG not really")
  return session


@pytest.fixture
def server(root, local):
  host = hostd.Host(root, token="s3cret")
  srv = hostd.Server(host, bind="127.0.0.1", port=0).start()
  yield srv
  srv.stop()


@pytest.fixture
def remote(server, local) -> wire.WireSession:
  return wire.WireHost(server.url, token="s3cret").session(local.key)


# The surface.


def test_a_wire_session_is_a_session_client_and_nothing_more(remote):
  assert isinstance(remote, SessionClient)
  for name in WIRE_SURFACE:
    assert hasattr(remote, name), name
  assert not hasattr(remote, "dir"), "no directory to reach for"


def test_identity_is_the_same_run_seen_from_afar(local, remote, server):
  assert remote.key == local.key == "run-a/rlmcp"
  assert remote.name == local.name == "run-a"
  assert remote.address.startswith(server.url + "/v1/sessions/")
  assert wire.is_url(remote.address) and not wire.is_url(local.address)


def test_every_read_answers_the_same_over_the_wire(local, remote):
  assert remote.info() == local.info()
  assert remote.status() == local.status()
  assert remote.params() == local.params()
  assert remote.metrics() == local.metrics()
  assert remote.metrics(last_n=1) == local.metrics(last_n=1)
  assert remote.metrics(since_seq=1) == local.metrics(since_seq=1)
  assert remote.events() == local.events()
  assert remote.metrics_count() == local.metrics_count() == 3
  assert remote.liveness_info() == local.liveness_info()
  assert remote.liveness() == local.liveness()


def test_artifacts_cross_as_bytes_and_carry_no_path(local, remote):
  theirs = remote.list_artifacts()
  mine = local.list_artifacts()
  assert [r["name"] for r in theirs] == [r["name"] for r in mine] == ["frame.png"]
  assert "path" not in theirs[0] and "path" in mine[0]
  assert remote.read_artifact("frame.png") == local.read_artifact("frame.png")
  with pytest.raises(ValueError):
    remote.read_artifact("../session.json")
  with pytest.raises(wire.WireError):
    remote.read_artifact("missing.png")


def test_a_command_crosses_the_wire_and_the_answer_comes_back(local, remote):
  request = remote.submit("set_parameter", name="lr", value=1e-4)
  pending = local.pop_requests()
  assert [r.cmd for r in pending] == ["set_parameter"]
  assert pending[0].args == {"name": "lr", "value": 1e-4}
  assert pending[0].req_id == request.req_id

  assert remote.poll(request.req_id) is None
  local.respond(Response(req_id=request.req_id, ok=True, result={"lr": 1e-4}))
  answer = remote.wait(request.req_id, timeout=5.0)
  assert answer.ok and answer.result == {"lr": 1e-4}
  assert remote.poll(request.req_id) is None, "the caller consumed it, as locally"


def test_wait_times_out_with_the_hosts_own_words(local, remote):
  request = remote.submit("status")
  answer = remote.wait(request.req_id, timeout=0.3)
  assert not answer.ok
  assert "not running" in (answer.error or "") or "Timed out" in (answer.error or "")


def test_call_is_submit_then_wait(local, remote):
  import threading

  def trainer():
    for _ in range(50):
      for req in local.pop_requests():
        local.respond(Response(req_id=req.req_id, ok=True, result=req.args))
        return
      time.sleep(0.02)

  threading.Thread(target=trainer, daemon=True).start()
  answer = remote.call("echo", timeout=5.0, x=1)
  assert answer.ok and answer.result == {"x": 1}


# The host.


def test_the_host_names_itself_and_lists_its_sessions(server, local, root):
  host = wire.WireHost(server.url, token="s3cret")
  described = host.host()
  assert described["host_id"] == hostd.host_id(root)
  assert described["root"] == str(root.resolve()) and described["sessions"] == 1
  assert isinstance(described["gpus"], list)
  listed = list(host.sessions())
  assert [s.key for s in listed] == [local.key]
  assert listed[0].status() == local.status()


def test_the_host_id_is_minted_once_and_kept_beside_the_sessions(root):
  first = hostd.host_id(root)
  assert first == hostd.host_id(root) and len(first) == 12
  assert (root / ".rlmcp-host-id").read_text().strip() == first


def test_a_key_is_a_name_never_a_path(server):
  host = wire.WireHost(server.url, token="s3cret")
  with pytest.raises(wire.WireError) as err:
    host.session("../../etc/passwd").status()
  assert "404" in str(err.value)
  with pytest.raises(wire.WireError):
    host.session("run-a/nope").status()


def test_the_token_is_checked_on_every_request(server, local):
  with pytest.raises(wire.WireError) as err:
    wire.WireHost(server.url, token="wrong").host()
  assert "401" in str(err.value)
  with pytest.raises(wire.WireError):
    wire.WireHost(server.url, token="wrong").session(local.key).status()


def test_binding_beyond_localhost_needs_a_token(root):
  with pytest.raises(ValueError):
    hostd.Server(hostd.Host(root), bind="0.0.0.0", port=0)


# Jobs: the host executes.


def test_a_job_runs_here_and_its_log_and_exit_come_back(server, tmp_path):
  host = wire.WireHost(server.url, token="s3cret")
  probe = "import os; print('cwd', os.getcwd()); print(os.environ['X'])"
  job = host.submit_job([sys.executable, "-c", probe],
                        cwd=str(tmp_path), env={"X": "marked"}, label="probe")
  assert job["state"] in ("running", "succeeded") and job["label"] == "probe"
  for _ in range(100):
    job = host.job(job["id"])
    if job["state"] in ("succeeded", "failed", "cancelled"):
      break
    time.sleep(0.05)
  assert job["state"] == "succeeded" and job["exit_code"] == 0
  assert job["log"][-1] == "marked" and job["log"][-2] == f"cwd {tmp_path.resolve()}"
  assert [j["id"] for j in host.jobs()] == [job["id"]]


def test_a_job_can_be_cancelled_and_says_so(server):
  host = wire.WireHost(server.url, token="s3cret")
  job = host.submit_job([sys.executable, "-c", "import time; time.sleep(30)"])
  cancelled = host.cancel_job(job["id"])
  for _ in range(100):
    if cancelled["state"] != "running":
      break
    time.sleep(0.05)
    cancelled = host.job(job["id"])
  assert cancelled["state"] == "cancelled"


def test_a_job_that_cannot_start_is_a_failed_job_not_an_exception(server):
  host = wire.WireHost(server.url, token="s3cret")
  with pytest.raises(wire.WireError) as err:
    host.submit_job(["/nonexistent/binary"])
  assert "500" in str(err.value) and "could not start" in str(err.value)


# The meeting point.


def test_connect_picks_the_transport_from_the_address(local, remote, server):
  assert isinstance(wire.connect(local.address), Session)
  again = wire.connect(remote.address, token="s3cret")
  assert isinstance(again, wire.WireSession) and again.status() == local.status()
  with pytest.raises(ValueError):
    wire.WireSession(server.url + "/v1/host")


# Trees: the code a job runs from.


def _archive_of(tmp_path: Path) -> bytes:
  import subprocess as sp
  src = tmp_path / "pkg-src"
  (src / "pkg").mkdir(parents=True)
  (src / "pkg" / "task.py").write_text("weight = 3.0\n")
  return sp.run(["tar", "-c", "-C", str(src), "."], capture_output=True, check=True).stdout


def test_a_tree_is_pushed_once_and_a_job_runs_from_it(server, root, tmp_path):
  host = wire.WireHost(server.url, token="s3cret")
  tree = "0123456789abcdef0123456789abcdef01234567"
  assert host.has_tree(tree) == ""
  path = host.push_tree(tree, _archive_of(tmp_path))
  assert path and Path(path).is_dir()
  assert (Path(path) / "pkg" / "task.py").read_text() == "weight = 3.0\n"
  assert Path(path).parent == root / ".rlmcp-hostd" / "trees"
  assert host.has_tree(tree) == path
  assert host.push_tree(tree, b"garbage, not a tar") == path, "already held: nothing unpacked"

  job = host.submit_job([sys.executable, "-c", "print(open('pkg/task.py').read())"], cwd=path)
  for _ in range(100):
    job = host.job(job["id"])
    if job["state"] in ("succeeded", "failed"):
      break
    time.sleep(0.05)
  assert job["state"] == "succeeded"
  assert [line for line in job["log"] if line][-1] == "weight = 3.0"


def test_a_tree_id_is_a_git_object_name_and_an_archive_is_a_tar(server, tmp_path):
  host = wire.WireHost(server.url, token="s3cret")
  with pytest.raises(wire.WireError) as err:
    host.push_tree("../etc", _archive_of(tmp_path))
  assert "400" in str(err.value)
  with pytest.raises(wire.WireError) as err:
    host.push_tree("abcdef0123456", b"not a tar at all")
  assert "400" in str(err.value) and "tar" in str(err.value)
