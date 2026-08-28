"""Telemetry: the metric buffer, the plotters, and trace persistence."""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rlmcp.cli import main as cli_main
from rlmcp.core.telemetry import plotter
from rlmcp.core.telemetry.buffer import TelemetryBuffer
from rlmcp.core.telemetry.trace import TraceRecorder, load_npz

# TelemetryBuffer: rows are maintained, not rebuilt.


def test_as_rows_is_maintained_incrementally_not_rebuilt_from_series():
  buf = TelemetryBuffer(maxlen=100)
  for i in range(60):
    buf.push(i, {"a": float(i), "b": float(2 * i)})

  # Mechanism assertion: sever the per-series storage. An implementation that
  # rebuilds rows by walking every stored point of every series would blow up
  # here; incremental row maintenance never touches it on the read path.
  buf._series = None
  rows = buf.as_rows(last_n=5)

  assert [r["iteration"] for r in rows] == [55, 56, 57, 58, 59]
  assert rows[-1] == {"iteration": 59, "a": 59.0, "b": 118.0}


def test_as_rows_returns_copies_not_internal_state():
  buf = TelemetryBuffer(maxlen=10)
  buf.push(1, {"a": 1.0})

  rows = buf.as_rows()
  rows[0]["a"] = 999.0

  assert buf.as_rows()[0]["a"] == 1.0


def test_last_n_zero_returns_nothing():
  buf = TelemetryBuffer(maxlen=10)
  buf.push(1, {"a": 1.0})

  assert buf.as_rows(last_n=0) == []
  assert buf.get_series("a", last_n=0) == []
  # None still means everything.
  assert len(buf.as_rows(last_n=None)) == 1
  assert len(buf.get_series("a", last_n=None)) == 1


def test_same_iteration_repush_is_last_wins():
  buf = TelemetryBuffer(maxlen=10)
  buf.push(3, {"a": 1.0})
  buf.push(3, {"a": 2.0})  # The service loop re-runs at one iteration.

  assert buf.get_series("a") == [(3, 2.0)]  # One point, not a doubled x.
  rows = buf.as_rows()
  assert len(rows) == 1 and rows[0]["a"] == 2.0


def test_rows_are_bounded_by_maxlen():
  buf = TelemetryBuffer(maxlen=10)
  for i in range(25):
    buf.push(i, {"a": float(i)})

  rows = buf.as_rows()
  assert len(rows) == 10
  assert rows[0]["iteration"] == 15 and rows[-1]["iteration"] == 24


def test_out_of_order_iterations_keep_rows_sorted():
  buf = TelemetryBuffer(maxlen=10)
  buf.push(5, {"a": 5.0})
  buf.push(3, {"a": 3.0})  # e.g. iteration moved backward after a rollback.
  buf.push(7, {"a": 7.0})

  assert [r["iteration"] for r in buf.as_rows()] == [3, 5, 7]


def test_non_floatable_values_are_reported_once_per_key():
  drops: list[tuple[str, Any]] = []
  buf = TelemetryBuffer(maxlen=10, on_drop=lambda k, v: drops.append((k, v)))

  buf.push(1, {"good": 1.0, "bad": "verbal note"})
  buf.push(2, {"bad": "another note"})  # Same key: not re-reported.

  assert drops == [("bad", "verbal note")]
  assert buf.get_series("bad") == []
  assert buf.get_series("good") == [(1, 1.0)]
  assert "bad" not in buf.as_rows()[-1]


# Plotters: no figure leaks on exception paths.


def test_plot_metric_series_closes_its_figure_when_data_is_bad():
  before = set(plt.get_fignums())
  with pytest.raises(Exception):
    plotter.plot_metric_series({"a": [(0, object())]})  # Not plottable.
  assert set(plt.get_fignums()) == before


def test_plot_trace_closes_its_figure_when_finishing_fails(monkeypatch):
  def exploding_finish(fig, dpi=110):
    raise RuntimeError("no memory for the png")

  monkeypatch.setattr(plotter, "_finish", exploding_finish)
  before = set(plt.get_fignums())
  data = {
      "joint_pos": np.zeros((5, 2), dtype=np.float32),
      "time": np.arange(5, dtype=np.float32) * 0.02,
  }
  with pytest.raises(RuntimeError):
    plotter.plot_trace(data)
  assert set(plt.get_fignums()) == before


def test_plot_terrain_status_closes_its_figure_when_finishing_fails(monkeypatch):
  def exploding_finish(fig, dpi=110):
    raise RuntimeError("boom")

  monkeypatch.setattr(plotter, "_finish", exploding_finish)
  before = set(plt.get_fignums())
  status = {"per_terrain": [{"terrain": "flat", "num_envs": 4, "level_mean": 1.0}],
            "level_ceiling": 3}
  with pytest.raises(RuntimeError):
    plotter.plot_terrain_status(status)
  assert set(plt.get_fignums()) == before


# Trace persistence: typed metadata, no pickle required (or allowed by default).


def _saved_trace(tmp_path, meta):
  recorder = TraceRecorder(capacity=8, dt=0.02)
  recorder.arm(env_id=1, num_steps=4,
               labels={"joint_pos": ["hip", "knee"]}, meta=meta)
  for i in range(4):
    recorder.record({"joint_pos": np.array([i, -i], dtype=np.float32)})
  return recorder.save_npz(tmp_path / "trace.npz")


def test_npz_meta_round_trips_types(tmp_path):
  path = _saved_trace(tmp_path, meta={"iteration": 7, "where": "stairs",
                                      "frac": 0.5})
  loaded = load_npz(path)

  assert loaded["meta"]["iteration"] == 7  # An int, not the string "7".
  assert loaded["meta"]["frac"] == 0.5
  assert loaded["meta"]["where"] == "stairs"
  assert loaded["labels"]["joint_pos"] == ["hip", "knee"]
  np.testing.assert_allclose(loaded["data"]["joint_pos"][:, 0], [0, 1, 2, 3])


def test_npz_needs_no_pickle_to_load(tmp_path):
  path = _saved_trace(tmp_path, meta={"iteration": 7})

  raw = np.load(path, allow_pickle=False)  # The hardened setting must suffice.
  try:
    for key in raw.files:
      raw[key]  # Accessing an object array would raise here.
  finally:
    raw.close()


def test_legacy_pickled_trace_is_refused_by_default(tmp_path):
  """Old traces stored labels/meta as pickled object arrays -- an arbitrary
  code execution vector when analysing an untrusted file, so loading them now
  takes an explicit opt-in."""
  path = tmp_path / "legacy.npz"
  np.savez(
      path,
      joint_pos=np.zeros((4, 2), dtype=np.float32),
      time=np.arange(4, dtype=np.float32) * 0.02,
      __labels__joint_pos=np.array(["hip", "knee"], dtype=object),
      __meta__=np.array(["iteration=7", "where=stairs"], dtype=object),
  )

  with pytest.raises(ValueError, match="allow_legacy"):
    load_npz(path)

  loaded = load_npz(path, allow_legacy=True)
  assert loaded["labels"]["joint_pos"] == ["hip", "knee"]
  # The legacy format stringified values; reading it back keeps that truth.
  assert loaded["meta"] == {"iteration": "7", "where": "stairs"}
  assert loaded["data"]["joint_pos"].shape == (4, 2)


def _legacy_trace(tmp_path) -> Any:
  """A pre-JSON trace: labels/meta pickled, data long enough to analyse."""
  path = tmp_path / "legacy.npz"
  np.savez(
      path,
      joint_pos=np.zeros((16, 2), dtype=np.float32),
      time=np.arange(16, dtype=np.float32) * 0.02,
      __labels__joint_pos=np.array(["hip", "knee"], dtype=object),
      __meta__=np.array(["iteration=7"], dtype=object),
  )
  return path


def test_cli_analyze_surfaces_the_pickle_refusal_as_a_result(tmp_path, capsys):
  """`rlmcp analyze` on a legacy trace: a clean error payload and rc 1,
  never a traceback -- and the hint names the CLI opt-in."""
  rc = cli_main(["analyze", str(_legacy_trace(tmp_path))])

  payload = json.loads(capsys.readouterr().out)
  assert rc == 1
  assert payload["ok"] is False
  assert "pickled" in payload["error"]
  assert "--allow-legacy" in payload["hint"]


def test_cli_analyze_allow_legacy_opts_in_with_a_warning(tmp_path, capsys):
  rc = cli_main(["analyze", str(_legacy_trace(tmp_path)), "--allow-legacy"])

  captured = capsys.readouterr()
  payload = json.loads(captured.out)
  assert rc == 0
  assert payload["ok"] is True
  assert payload["result"]["report"]["num_steps"] == 16
  assert payload["result"]["meta"] == {"iteration": "7"}
  assert "--allow-legacy" in captured.err  # The one-line warning, off stdout.
