"""Trace analysis: does it name the problem an agent should fix?"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rlmcp.core.diagnostics import (
  analyze_trace,
  dominant_frequency,
  high_frequency_content,
  summarize_metric_history,
)

DT = 0.02
STEPS = 500
T = np.arange(STEPS) * DT


def _gait(amplitude: float = 1.0, hz: float = 1.5) -> np.ndarray:
  return amplitude * np.sin(2 * np.pi * hz * T)


def _trace(joint_pos: np.ndarray, **extra) -> dict:
  data = {
      "joint_pos": joint_pos.astype(np.float32),
      "joint_vel": np.gradient(joint_pos, DT, axis=0).astype(np.float32),
  }
  data.update({k: v.astype(np.float32) for k, v in extra.items()})
  return data


def test_dominant_frequency_finds_the_gait_band():
  freq, share = dominant_frequency(_gait(hz=2.0), DT)
  assert 1.8 < freq < 2.2
  assert share > 0.5


def test_high_frequency_content_sees_chatter_under_a_big_gait():
  signal = _gait(amplitude=1.0, hz=1.5) + 0.4 * np.sin(2 * np.pi * 16 * T)
  share, peak = high_frequency_content(signal, DT, cutoff_hz=8.0)
  assert share > 0.1
  assert 15 < peak < 17


def test_a_clean_gait_is_not_flagged_as_buzzing():
  clean = np.stack([_gait(), _gait(0.5)], axis=1)
  report = analyze_trace(_trace(clean), {"joint_pos": ["hip", "knee"]}, dt=DT)

  assert report["smoothness"]["num_buzzing_joints"] == 0
  assert "No high-frequency joint chatter" in report["verdict"][0]


def test_a_chattering_joint_is_named():
  noisy = np.stack(
      [_gait(), _gait(0.5) + 0.15 * np.sin(2 * np.pi * 15 * T)], axis=1
  )
  report = analyze_trace(_trace(noisy), {"joint_pos": ["hip", "knee"]}, dt=DT)
  buzzing = report["smoothness"]["buzzing_joints"]

  assert [b["name"] for b in buzzing] == ["knee"]
  assert buzzing[0]["peak_hf_hz"] > 8.0
  assert "knee" in report["verdict"][0]


def _many_joints(n: int, hf_amplitude) -> np.ndarray:
  """n joints on a shared gait, each with its own high-frequency component."""
  columns = []
  for j in range(n):
    amp = hf_amplitude(j)
    columns.append(_gait(0.5 + 0.02 * j) + amp * np.sin(2 * np.pi * (12 + j) * T))
  return np.stack(columns, axis=1)


def test_one_joint_standing_out_is_named():
  joints = _many_joints(20, lambda j: 0.2 if j == 7 else 0.004)
  report = analyze_trace(_trace(joints), dt=DT)["smoothness"]

  assert report["num_buzzing_joints"] == 1
  assert report["buzzing_joints"][0]["name"] == "joint[7]"


def test_a_uniformly_jittery_robot_is_not_reported_as_20_bad_joints():
  """Flagging almost every joint is the same as flagging none."""
  joints = _many_joints(20, lambda j: 0.2)
  report = analyze_trace(_trace(joints), dt=DT)

  smooth = report["smoothness"]
  assert smooth["hf_share_median"] > 0.35
  assert smooth["num_buzzing_joints"] <= 2  # Nobody stands out from the crowd.
  assert len(smooth["worst_hf_joints"]) == 5  # The ranking is still there.
  assert "whole body" in report["verdict"][0]
  assert "global smoothness problem" in report["verdict"][0]


def test_jerk_grows_with_chatter_amplitude():
  jerks = []
  for amplitude in (0.0, 0.05, 0.15):
    signal = np.stack(
        [_gait(), _gait(0.5) + amplitude * np.sin(2 * np.pi * 15 * T)], axis=1
    )
    jerks.append(analyze_trace(_trace(signal), dt=DT)["smoothness"]["joint_jerk_rms"])

  assert jerks[0] < jerks[1] < jerks[2]


def test_tracking_error_is_reported_against_the_command():
  data = _trace(
      np.stack([_gait(), _gait()], axis=1),
      command=np.tile([1.0, 0.0, 0.0], (STEPS, 1)),
      base_lin_vel=np.tile([0.4, 0.0, 0.0], (STEPS, 1)),
      base_ang_vel=np.zeros((STEPS, 3)),
  )
  report = analyze_trace(data, dt=DT)

  assert report["tracking"]["commanded_speed_mean"] == 1.0
  assert abs(report["tracking"]["lin_vel_error_x_mean"] + 0.6) < 1e-3
  assert any("tracking is poor" in line for line in report["verdict"])


def test_shuffling_feet_are_called_out():
  data = _trace(
      np.stack([_gait(), _gait()], axis=1),
      foot_contact=np.ones((STEPS, 2)),
  )
  report = analyze_trace(data, dt=DT)

  assert report["gait"]["contact_fraction"] == 1.0
  assert any("shuffling" in line for line in report["verdict"])


def test_stepping_gait_reports_air_time():
  contact = np.stack([_gait() > 0, _gait() < 0], axis=1).astype(np.float32)
  report = analyze_trace(_trace(np.stack([_gait(), _gait()], 1), foot_contact=contact), dt=DT)

  assert 0.2 < report["gait"]["mean_air_time_s"] < 0.5
  assert 1.0 < report["gait"]["step_frequency_hz"] < 2.5


def test_tilt_is_measured_from_projected_gravity():
  upright = np.tile([0.0, 0.0, -1.0], (STEPS, 1))
  leaning = np.tile([0.6, 0.0, -0.8], (STEPS, 1))

  a = analyze_trace(_trace(np.stack([_gait()] * 2, 1), projected_gravity=upright), dt=DT)
  b = analyze_trace(_trace(np.stack([_gait()] * 2, 1), projected_gravity=leaning), dt=DT)

  assert a["posture"]["tilt_deg_mean"] < 1.0
  assert b["posture"]["tilt_deg_mean"] > 30.0
  assert any("leans" in line for line in b["verdict"])


def test_short_traces_degrade_gracefully():
  report = analyze_trace({"joint_pos": np.zeros((2, 3), dtype=np.float32)}, dt=DT)
  assert "too short" in report["note"].lower()


def test_a_1d_channel_is_read_as_one_component_not_a_crash():
  """A ``(n,)`` channel is one signal sampled per step. The smoothness and
  gait sections must treat it as width one -- the reading the divergence scan
  already takes -- rather than raising IndexError on ``shape[1]``."""
  rng = np.random.default_rng(3)
  pos = _gait() + 0.05 * rng.normal(size=STEPS)  # 1-D on purpose.
  vel = np.gradient(pos, DT)
  contact = (_gait() > 0).astype(np.float32)  # 1-D on purpose.

  report = analyze_trace(
      {"joint_pos": pos, "joint_vel": vel, "foot_contact": contact}, dt=DT
  )

  smooth = report["smoothness"]
  assert smooth["chatter_measured"] is True
  assert smooth["num_joints"] == 1
  assert "joint_jerk_rms" in smooth
  assert 0.0 < report["gait"]["contact_fraction"] < 1.0


def test_metric_history_summary_reports_trend():
  rows = [{"iteration": i, "reward": float(i)} for i in range(60)]
  summary = summarize_metric_history(rows, ["reward", "missing"], window=10)

  assert summary["reward"]["trend"] == "up"
  assert summary["reward"]["latest"] == 59.0
  assert "missing" not in summary


def test_nan_trace_leads_with_divergence_not_clean_metrics():
  pos = np.stack([_gait(), _gait(0.5)], axis=1).astype(np.float32)
  vel = np.gradient(pos, DT, axis=0).astype(np.float32)
  pos[400:] = np.nan  # The policy blows up 80% of the way in.
  report = analyze_trace({"joint_pos": pos, "joint_vel": vel}, dt=DT)

  div = report["divergence"]
  assert div["nonfinite_channels"] == [
      {"channel": "joint_pos", "first_bad_step": 400, "bad_step_fraction": 0.2}
  ]
  assert div["finite_prefix_steps"] == 400
  first = report["verdict"][0]
  assert "NaN" in first and "joint_pos" in first
  # Surviving metrics describe the finite prefix, and the verdict says so.
  assert any("400 finite steps" in line for line in report["verdict"])
  json.dumps(report, allow_nan=False)  # Strict-JSON safe: no bare NaN tokens.


def test_fully_nan_trace_reports_divergence_only():
  bad = np.full((STEPS, 2), np.nan, dtype=np.float32)
  report = analyze_trace({"joint_pos": bad, "joint_vel": bad.copy()}, dt=DT)

  assert report["divergence"]["finite_prefix_steps"] == 0
  assert "smoothness" not in report
  assert "tracking" not in report
  assert "NaN" in report["verdict"][0]
  assert not any(
      "No high-frequency joint chatter" in line for line in report["verdict"]
  )
  json.dumps(report, allow_nan=False)


def test_sub_nyquist_rate_reports_chatter_unmeasurable():
  dt = 0.1  # Nyquist 5 Hz sits below the 8 Hz chatter cutoff.
  t = np.arange(STEPS) * dt
  buzz = np.sin(2 * np.pi * 1.5 * t) + 2.0 * np.sin(2 * np.pi * 4.5 * t)
  pos = np.stack([buzz, buzz], axis=1).astype(np.float32)
  vel = np.gradient(pos, dt, axis=0).astype(np.float32)
  report = analyze_trace({"joint_pos": pos, "joint_vel": vel}, dt=dt)

  smooth = report["smoothness"]
  assert smooth["chatter_measured"] is False
  assert "Nyquist" in smooth["chatter_unmeasurable_reason"]
  assert any(
      "Chatter unmeasurable: Nyquist 5.0 Hz <= cutoff 8.0 Hz" in line
      for line in report["verdict"]
  )
  assert not any(
      "No high-frequency joint chatter" in line for line in report["verdict"]
  )


def test_four_step_trace_reports_chatter_unmeasurable():
  pos = np.array([[0.0, 0.0], [1.0, -1.0], [0.0, 0.0], [-1.0, 1.0]], dtype=np.float32)
  vel = np.gradient(pos, DT, axis=0).astype(np.float32)
  report = analyze_trace({"joint_pos": pos, "joint_vel": vel}, dt=DT)

  smooth = report["smoothness"]
  assert smooth["chatter_measured"] is False
  assert "4 steps" in smooth["chatter_unmeasurable_reason"]
  assert "8" in smooth["chatter_unmeasurable_reason"]
  assert any("Chatter unmeasurable" in line for line in report["verdict"])
  assert not any(
      "No high-frequency joint chatter" in line for line in report["verdict"]
  )


def test_missing_joint_vel_channel_is_named_not_read_as_clean():
  pos = np.stack([_gait(), _gait(0.5)], axis=1).astype(np.float32)
  report = analyze_trace({"joint_pos": pos}, dt=DT)

  smooth = report["smoothness"]
  assert smooth["chatter_measured"] is False
  assert smooth["chatter_unmeasurable_reason"] == "no joint_vel channel in the trace"
  assert any("no joint_vel channel" in line for line in report["verdict"])
  assert not any(
      "No high-frequency joint chatter" in line for line in report["verdict"]
  )


def test_measured_clean_trace_still_gets_the_affirmative():
  clean = np.stack([_gait(), _gait(0.5)], axis=1)
  report = analyze_trace(_trace(clean), dt=DT)

  assert report["smoothness"]["chatter_measured"] is True
  assert "No high-frequency joint chatter" in report["verdict"][0]


def test_zero_width_channel_is_skipped_with_a_note():
  data = _trace(
      np.stack([_gait(), _gait(0.5)], axis=1),
      action=np.zeros((STEPS, 0)),
  )
  report = analyze_trace(data, dt=DT)

  assert report["skipped_empty_channels"] == ["action"]
  assert "action_range" not in report["smoothness"]
  assert report["smoothness"]["chatter_measured"] is True


def test_zero_width_joint_vel_reads_as_unmeasurable_not_clean():
  pos = np.stack([_gait(), _gait(0.5)], axis=1).astype(np.float32)
  report = analyze_trace(
      {"joint_pos": pos, "joint_vel": np.zeros((STEPS, 0), dtype=np.float32)}, dt=DT
  )

  assert report["skipped_empty_channels"] == ["joint_vel"]
  assert report["smoothness"]["chatter_measured"] is False
  assert "empty" in report["smoothness"]["chatter_unmeasurable_reason"]


def test_metric_history_short_series_is_not_read_against_one_sample():
  # 16 alternating samples have no trend, but the old first-sample baseline
  # would have called this "up".
  rows = [{"iteration": i, "reward": 4.0 if i % 2 == 0 else 6.0} for i in range(16)]
  summary = summarize_metric_history(rows, ["reward"], window=20)

  assert summary["reward"]["trend"] == "flat"
  assert "short history" in summary["reward"]["trend_note"]
  assert summary["reward"]["num_points"] == 16


def test_metric_history_single_point_reports_insufficient_history():
  summary = summarize_metric_history(
      [{"iteration": 0, "reward": 1.0}], ["reward"], window=20
  )

  assert summary["reward"]["trend"] == "insufficient_history"
  assert "delta_vs_previous_window" not in summary["reward"]


def test_object_dtype_numeric_channel_scans_and_analyzes():
  # np.savez with allow_pickle round-trips hand-built traces as object arrays;
  # the divergence scan must coerce them, not crash on the isfinite ufunc.
  pos = np.stack([_gait(), _gait(0.5)], axis=1).astype(np.float32)
  vel = np.gradient(pos, DT, axis=0)
  data = {"joint_pos": pos.astype(object), "joint_vel": vel.astype(object)}
  data["joint_pos"][400:] = np.nan
  report = analyze_trace(data, dt=DT)

  assert report["divergence"]["first_bad_step"] == 400
  assert report["divergence"]["nonfinite_channels"][0]["channel"] == "joint_pos"
  assert report["smoothness"]["chatter_measured"] is True
  assert "NaN" in report["verdict"][0]
  json.dumps(report, allow_nan=False)


def test_non_numeric_channel_raises_a_named_error():
  pos = np.stack([_gait(), _gait(0.5)], axis=1).astype(np.float32)
  ragged = np.empty((STEPS,), dtype=object)
  for i in range(STEPS):
    ragged[i] = [0.0] * (1 + i % 3)  # Ragged rows: np.savez pickles these too.

  with pytest.raises(ValueError, match="ragged_channel"):
    analyze_trace({"joint_pos": pos, "ragged_channel": ragged}, dt=DT)


def test_advice_does_not_name_keys_from_one_backends_vocabulary():
  """`diagnose` runs on every backend, so its advice has to describe the lever
  rather than name a key that may not exist.

  Found on a real Genesis run: the smoothness verdict told the operator to
  "raise action_rate_l2 or lower action.joint_pos.scale_gain", which are mjlab
  term names. On that run the keys were `reward.action_rate.weight` and
  `action.scale`, so the one actionable sentence in the report pointed at
  nothing.
  """
  source = (Path(__file__).resolve().parent.parent
            / "rlmcp" / "core" / "diagnostics.py").read_text()
  for mjlab_key in ("action_rate_l2", "action.joint_pos.scale_gain"):
    assert mjlab_key not in source, (
        f"diagnostics advice names '{mjlab_key}', which only exists on mjlab"
    )
