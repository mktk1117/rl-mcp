"""Turn raw traces into numbers a tuning agent can act on.

Everything here is pure numpy so it runs equally well inside the trainer or in a
separate process reading a saved ``.npz``.

The vocabulary is deliberately blunt -- ``jerk_rms``, ``buzzing_joints``,
``tracking_error`` -- because the consumer is usually an LLM deciding which
reward weight to nudge, and it needs a name it can map onto a term.

Trace channels are looked up by the ``CHANNEL_*`` names declared (with shapes
and units) in :mod:`rlmcp.adapters.base` -- the contract every
``sample_state`` implementation writes to.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from rlmcp.adapters.base import (
    CHANNEL_ACTION,
    CHANNEL_BASE_ANG_VEL,
    CHANNEL_BASE_LIN_VEL,
    CHANNEL_BASE_POS,
    CHANNEL_COMMAND,
    CHANNEL_FOOT_CONTACT,
    CHANNEL_JOINT_POS,
    CHANNEL_JOINT_TORQUE,
    CHANNEL_JOINT_VEL,
    CHANNEL_PROJECTED_GRAVITY,
    CHANNEL_REWARD,
    CHANNEL_TIME,
)


def _rms(x: np.ndarray) -> float:
  if x.size == 0:
    return 0.0
  return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))


def _safe(x: float) -> float:
  return float(x) if np.isfinite(x) else 0.0


def _diff(x: np.ndarray, dt: float, order: int) -> np.ndarray:
  out = x.astype(np.float64)
  for _ in range(order):
    out = np.diff(out, axis=0) / dt
  return out


def _top_offenders(
    per_component: np.ndarray, labels: Sequence[str], k: int = 5
) -> List[Dict[str, Any]]:
  if per_component.size == 0:
    return []
  order = np.argsort(-np.abs(per_component))[:k]
  out = []
  for i in order:
    name = labels[i] if i < len(labels) else f"[{i}]"
    out.append({"name": name, "value": round(float(per_component[i]), 5)})
  return out


def dominant_frequency(signal: np.ndarray, dt: float) -> tuple[float, float]:
  """Peak non-DC frequency of a 1-D signal and its share of total AC power."""
  freqs, power = _ac_spectrum(signal, dt)
  if power.size == 0:
    return 0.0, 0.0
  total = float(np.sum(power))
  if total <= 0:
    return 0.0, 0.0
  idx = int(np.argmax(power))
  return float(freqs[idx]), float(power[idx] / total)


def high_frequency_content(
    signal: np.ndarray, dt: float, cutoff_hz: float
) -> tuple[float, float]:
  """Share of AC power above ``cutoff_hz``, and the peak frequency up there.

  Chatter usually rides *on top of* a healthy 1-3 Hz gait rather than replacing
  it, so the dominant frequency stays low even when a joint is buzzing. The
  fraction of power above the gait band is the signal that actually separates
  the two.
  """
  freqs, power = _ac_spectrum(signal, dt)
  if power.size == 0:
    return 0.0, 0.0
  total = float(np.sum(power))
  if total <= 0:
    return 0.0, 0.0
  mask = freqs >= cutoff_hz
  if not np.any(mask):
    return 0.0, 0.0
  hf = power[mask]
  peak = float(freqs[mask][int(np.argmax(hf))])
  return float(np.sum(hf) / total), peak


def _ac_spectrum(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
  """Power spectrum of a 1-D signal with the DC bin removed."""
  n = signal.shape[0]
  if n < 8 or dt <= 0:
    return np.zeros(0), np.zeros(0)
  x = signal.astype(np.float64) - np.mean(signal)
  power = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
  freqs = np.fft.rfftfreq(n, d=dt)
  if power.size < 2:
    return np.zeros(0), np.zeros(0)
  return freqs[1:], power[1:]


def analyze_trace(
    data: Dict[str, np.ndarray],
    labels: Optional[Dict[str, List[str]]] = None,
    dt: float = 0.02,
    buzz_freq_hz: float = 8.0,
    buzz_power_frac: float = 0.15,
    buzz_outlier_margin: float = 0.12,
    whole_body_hf_median: float = 0.35,
) -> Dict[str, Any]:
  """Summarise a per-step trace into smoothness / tracking / effort metrics.

  Args:
    data: channel name -> (num_steps, width) array, as produced by
      :meth:`rlmcp.core.telemetry.trace.TraceRecorder.snapshot`.
    labels: optional channel -> component names (joint names, etc).
    dt: environment step duration in seconds.
    buzz_freq_hz: boundary between the gait band and chatter.
    buzz_power_frac: absolute floor on the share of joint-velocity power above
      ``buzz_freq_hz`` before a joint can be flagged.
    buzz_outlier_margin: how far above the robot's median share a joint must sit
      to be called out individually.
    whole_body_hf_median: median share above which the finding is reported as a
      whole-body smoothness problem rather than a list of bad joints.
  """
  labels = labels or {}
  report: Dict[str, Any] = {"num_steps": 0, "dt": dt}

  # Channels with no data (zero width or zero rows) would raise from the
  # reductions below; drop them up front and record which ones were dropped.
  skipped_empty = sorted(
      k for k, v in data.items() if k != CHANNEL_TIME and v.size == 0
  )
  if skipped_empty:
    data = {k: v for k, v in data.items() if k not in skipped_empty}
    report["skipped_empty_channels"] = skipped_empty

  # A 1-D channel is one signal sampled per step; make its width of one
  # explicit so the ``shape[1]`` consumers below (chatter, tracking, gait)
  # read it truthfully instead of raising. The same reading the divergence
  # scan already takes with its ``reshape(n, -1)``. TraceRecorder always
  # writes 2-D, so this fires only for hand-built dicts and external files.
  widened = {
      k: v.reshape(-1, 1)
      for k, v in data.items()
      if k != CHANNEL_TIME and v.ndim == 1
  }
  if widened:
    data = {**data, **widened}

  any_channel = next((v for k, v in data.items() if k != CHANNEL_TIME), None)
  if any_channel is None and skipped_empty:
    report["note"] = "Every channel in the trace is empty; nothing to analyse."
    return report
  if any_channel is None or any_channel.shape[0] < 3:
    report["note"] = "Trace too short to analyse (need at least 3 steps)."
    return report
  report["num_steps"] = int(any_channel.shape[0])
  report["duration_s"] = round(float(any_channel.shape[0] * dt), 3)

  # Non-finite values mean the policy or the simulation diverged, and every
  # statistic computed downstream of them is a lie: NaN fails every threshold
  # comparison, and _safe() would launder it into a plausible-looking 0.0.
  # Detect this first, report it loudly, and analyse only the finite prefix.
  nonfinite: List[Dict[str, Any]] = []
  coerced: Dict[str, np.ndarray] = {}
  finite_prefix = int(any_channel.shape[0])
  for name in sorted(k for k in data if k != CHANNEL_TIME):
    arr = data[name]
    flat = arr.reshape(arr.shape[0], -1)
    try:
      finite = np.isfinite(flat)
    except TypeError:
      # Object-dtype channels (np.savez silently pickles these) still scan if
      # they are numeric; anything else must fail naming the channel rather
      # than leak a bare ufunc TypeError from deep inside the scan.
      try:
        flat = flat.astype(np.float64)
      except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Channel {name!r} is not numeric and cannot be checked for "
            "non-finite values; drop it from the trace or fix its dtype."
        ) from exc
      finite = np.isfinite(flat)
      coerced[name] = flat.reshape(arr.shape)
    finite_rows = np.all(finite, axis=1)
    if bool(np.all(finite_rows)):
      continue
    bad_rows = ~finite_rows
    first_bad = int(np.argmax(bad_rows))
    nonfinite.append({
        "channel": name,
        "first_bad_step": first_bad,
        "bad_step_fraction": round(float(np.mean(bad_rows)), 4),
    })
    finite_prefix = min(finite_prefix, first_bad)
  if coerced:
    # Analyse the float64 view of coerced channels so the metric sections see
    # plain numeric arrays, exactly as if the npz had been saved unpickled.
    data = {**data, **coerced}
  if nonfinite:
    report["divergence"] = {
        "nonfinite_channels": nonfinite,
        "first_bad_step": finite_prefix,
        "finite_prefix_steps": finite_prefix,
    }
    if finite_prefix < 3:
      report["note"] = (
          f"Non-finite values corrupt the trace from step {finite_prefix}; too "
          "few finite steps to compute metrics."
      )
      report["verdict"] = _verdict(report)
      return report
    report["note"] = (
        f"Non-finite values from step {finite_prefix}; metrics cover only the "
        f"{finite_prefix} finite steps before divergence."
    )
    data = {k: v[:finite_prefix] for k, v in data.items()}

  joint_labels = labels.get(CHANNEL_JOINT_POS) or labels.get(CHANNEL_JOINT_VEL) or []

  # Smoothness: how violently the commanded and realised motion changes.
  smooth: Dict[str, Any] = {}
  if CHANNEL_ACTION in data:
    action = data[CHANNEL_ACTION]
    d1 = _diff(action, dt, 1)
    d2 = _diff(action, dt, 2)
    smooth["action_rate_rms"] = round(_safe(_rms(d1)), 5)
    smooth["action_accel_rms"] = round(_safe(_rms(d2)), 5)
    smooth["action_range"] = round(_safe(float(np.max(np.abs(action)))), 4)
  if CHANNEL_JOINT_POS in data:
    jerk = _diff(data[CHANNEL_JOINT_POS], dt, 3)
    smooth["joint_jerk_rms"] = round(_safe(_rms(jerk)), 3)
    per_joint = np.sqrt(np.mean(np.square(jerk), axis=0)) if jerk.size else np.zeros(0)
    smooth["jerkiest_joints"] = _top_offenders(per_joint, joint_labels)
  if CHANNEL_JOINT_VEL in data:
    jv = data[CHANNEL_JOINT_VEL]
    smooth["joint_vel_rms"] = round(_safe(_rms(jv)), 4)
    smooth["joint_vel_max"] = round(_safe(float(np.max(np.abs(jv)))), 4)
    accel = _diff(jv, dt, 1)
    smooth["joint_accel_rms"] = round(_safe(_rms(accel)), 3)

    # A healthy gait lives at 1-3 Hz. Chatter rides on top of it, so measure the
    # share of power above the gait band rather than the single dominant peak.
    #
    # An absolute threshold alone is not enough: a real 50 Hz humanoid policy
    # puts high-frequency content in *every* joint (contact impacts, PD
    # tracking), and flagging 27 of 29 joints tells the reader nothing. So a
    # joint is called out only when it also stands out against the robot's own
    # median -- and when the whole body is above the cutoff, that is reported as
    # its own, different, finding.
    #
    # The measurement only exists when the spectrum can see above the cutoff.
    # A slow trace (Nyquist at or below the cutoff) or a short one (< 8 steps)
    # says nothing about chatter, and must not read as "clean".
    nyquist_hz = (0.5 / dt) if dt > 0 else 0.0
    if nyquist_hz <= buzz_freq_hz:
      smooth["chatter_measured"] = False
      smooth["chatter_unmeasurable_reason"] = (
          f"Nyquist {nyquist_hz:.1f} Hz <= cutoff {buzz_freq_hz:.1f} Hz"
      )
    elif jv.shape[0] < 8:
      smooth["chatter_measured"] = False
      window_name = "finite prefix" if "divergence" in report else "trace"
      smooth["chatter_unmeasurable_reason"] = (
          f"{window_name} too short for a spectrum ({jv.shape[0]} steps, need >= 8)"
      )
    else:
      per_joint = []
      for j in range(jv.shape[1]):
        share, peak = high_frequency_content(jv[:, j], dt, buzz_freq_hz)
        name = joint_labels[j] if j < len(joint_labels) else f"joint[{j}]"
        per_joint.append(
            {"name": name, "hf_power_share": round(share, 3), "peak_hf_hz": round(peak, 2)}
        )
      per_joint.sort(key=lambda d: -d["hf_power_share"])
      shares = np.array([d["hf_power_share"] for d in per_joint], dtype=np.float64)
      median_share = float(np.median(shares)) if shares.size else 0.0

      whole_body = median_share >= whole_body_hf_median
      if whole_body:
        # Naming the "worst" eight joints of a robot that shakes everywhere points
        # the reader at a joint problem that does not exist. Report the condition
        # instead, and keep the ranking under a name that does not imply a verdict.
        buzzing: List[Dict[str, Any]] = []
      else:
        outlier_floor = max(buzz_power_frac, median_share + buzz_outlier_margin)
        buzzing = [d for d in per_joint if d["hf_power_share"] >= outlier_floor]

      smooth["chatter_measured"] = True
      smooth["worst_hf_joints"] = per_joint[:5]
      smooth["buzzing_joints"] = buzzing[:8]
      smooth["num_buzzing_joints"] = len(buzzing)
      smooth["whole_body_hf"] = bool(whole_body)
      smooth["hf_share_median"] = round(median_share, 3)
      smooth["hf_cutoff_hz"] = buzz_freq_hz
      smooth["num_joints"] = int(jv.shape[1])
  else:
    # Without joint velocities the chatter measurement never ran; say so rather
    # than letting the verdict's "no chatter" read as a measurement result.
    smooth["chatter_measured"] = False
    smooth["chatter_unmeasurable_reason"] = (
        f"{CHANNEL_JOINT_VEL} channel is empty"
        if CHANNEL_JOINT_VEL in (report.get("skipped_empty_channels") or [])
        else f"no {CHANNEL_JOINT_VEL} channel in the trace"
    )
  if smooth:
    report["smoothness"] = smooth

  # Tracking: is the robot doing what the command asked?
  track: Dict[str, Any] = {}
  cmd = data.get(CHANNEL_COMMAND)
  lin = data.get(CHANNEL_BASE_LIN_VEL)
  ang = data.get(CHANNEL_BASE_ANG_VEL)
  if cmd is not None and lin is not None and cmd.shape[1] >= 2 and lin.shape[1] >= 2:
    err_xy = lin[:, :2] - cmd[:, :2]
    # RMS of the per-sample error magnitude. Averaging the x and y components
    # separately would halve a pure-x error and hide it.
    track["lin_vel_error_rms"] = round(
        _safe(_rms(np.linalg.norm(err_xy, axis=1))), 4
    )
    track["lin_vel_error_x_mean"] = round(_safe(float(np.mean(err_xy[:, 0]))), 4)
    track["lin_vel_error_y_mean"] = round(_safe(float(np.mean(err_xy[:, 1]))), 4)
    track["commanded_speed_mean"] = round(
        _safe(float(np.mean(np.linalg.norm(cmd[:, :2], axis=1)))), 4
    )
    track["achieved_speed_mean"] = round(
        _safe(float(np.mean(np.linalg.norm(lin[:, :2], axis=1)))), 4
    )
  if cmd is not None and ang is not None and cmd.shape[1] >= 3 and ang.shape[1] >= 3:
    err_yaw = ang[:, 2] - cmd[:, 2]
    track["ang_vel_error_rms"] = round(_safe(_rms(err_yaw)), 4)
  if track:
    report["tracking"] = track

  # Effort: torque cost and saturation.
  if CHANNEL_JOINT_TORQUE in data:
    tau = data[CHANNEL_JOINT_TORQUE]
    effort = {
        "torque_rms": round(_safe(_rms(tau)), 3),
        "torque_abs_p95": round(_safe(float(np.percentile(np.abs(tau), 95))), 3),
        "torque_abs_max": round(_safe(float(np.max(np.abs(tau)))), 3),
    }
    per_joint = np.sqrt(np.mean(np.square(tau.astype(np.float64)), axis=0))
    effort["hardest_working_joints"] = _top_offenders(per_joint, joint_labels)
    report["effort"] = effort

  # Posture and base motion.
  posture: Dict[str, Any] = {}
  if CHANNEL_PROJECTED_GRAVITY in data and data[CHANNEL_PROJECTED_GRAVITY].shape[1] >= 3:
    pg = data[CHANNEL_PROJECTED_GRAVITY]
    tilt = np.degrees(np.arccos(np.clip(-pg[:, 2], -1.0, 1.0)))
    posture["tilt_deg_mean"] = round(_safe(float(np.mean(tilt))), 2)
    posture["tilt_deg_max"] = round(_safe(float(np.max(tilt))), 2)
  if CHANNEL_BASE_POS in data and data[CHANNEL_BASE_POS].shape[1] >= 3:
    height = data[CHANNEL_BASE_POS][:, 2]
    posture["base_height_mean"] = round(_safe(float(np.mean(height))), 3)
    posture["base_height_std"] = round(_safe(float(np.std(height))), 4)
  if CHANNEL_BASE_ANG_VEL in data:
    posture["base_ang_vel_rms"] = round(_safe(_rms(data[CHANNEL_BASE_ANG_VEL])), 4)
  if posture:
    report["posture"] = posture

  # Gait, when foot contact was captured.
  if CHANNEL_FOOT_CONTACT in data:
    contact = data[CHANNEL_FOOT_CONTACT] > 0.5
    gait: Dict[str, Any] = {
        "contact_fraction": round(float(np.mean(contact)), 3),
        "flight_fraction": round(float(np.mean(~np.any(contact, axis=1))), 3),
    }
    air_times = []
    for f in range(contact.shape[1]):
      col = contact[:, f]
      runs, current = [], 0
      for value in col:
        if value:
          if current:
            runs.append(current)
          current = 0
        else:
          current += 1
      if current:
        runs.append(current)
      if runs:
        air_times.append(float(np.mean(runs)) * dt)
    if air_times:
      gait["mean_air_time_s"] = round(float(np.mean(air_times)), 3)
      gait["step_frequency_hz"] = (
          round(1.0 / (2.0 * float(np.mean(air_times))), 2)
          if float(np.mean(air_times)) > 0
          else 0.0
      )
    report["gait"] = gait

  if CHANNEL_REWARD in data:
    report["reward"] = {
        "mean_step_reward": round(_safe(float(np.mean(data[CHANNEL_REWARD]))), 4),
        "min_step_reward": round(_safe(float(np.min(data[CHANNEL_REWARD]))), 4),
    }

  report["verdict"] = _verdict(report)
  return report


def _verdict(report: Dict[str, Any]) -> List[str]:
  """Short plain-language read of the numbers, for an agent or a human."""
  lines: List[str] = []
  smooth = report.get("smoothness", {})
  track = report.get("tracking", {})
  posture = report.get("posture", {})
  gait = report.get("gait", {})

  # Divergence outranks everything: an agent must never read smoothness or
  # tracking numbers off a trace whose channels went non-finite.
  divergence = report.get("divergence")
  if divergence:
    channels = divergence.get("nonfinite_channels") or []
    named = ", ".join(
        f"{c['channel']} (from step {c['first_bad_step']}, "
        f"{100.0 * c['bad_step_fraction']:.0f}% of steps)"
        for c in channels[:6]
    )
    if len(channels) > 6:
      named += f", and {len(channels) - 6} more"
    lines.append(
        f"DIVERGENCE: NaN/Inf values in {named}. The policy or simulation has "
        "blown up -- fix that before trusting any other number in this report."
    )
    prefix = int(divergence.get("finite_prefix_steps", 0))
    if prefix < 3:
      lines.append(
          f"Only {prefix} finite steps precede the first non-finite sample -- "
          "too few to compute any metrics."
      )
      return lines
    lines.append(
        f"Metrics below cover only the {prefix} finite steps before the first "
        "non-finite sample."
    )

  if smooth.get("chatter_measured"):
    buzzing = smooth.get("buzzing_joints") or []
    median_share = smooth.get("hf_share_median")
    num_joints = smooth.get("num_joints")
    if smooth.get("whole_body_hf"):
      worst = ", ".join(b["name"] for b in (smooth.get("worst_hf_joints") or [])[:3])
      lines.append(
          f"The whole body carries high-frequency motion (median {median_share:.2f} of "
          f"joint-velocity power above {smooth.get('hf_cutoff_hz', 8)} Hz across "
          f"{num_joints} joints; worst: {worst}). This is a global smoothness problem "
          "-- raise action_rate_l2 or lower action.joint_pos.scale_gain rather than "
          "chasing one joint."
      )
    elif buzzing:
      names = ", ".join(b["name"] for b in buzzing[:4])
      lines.append(
          f"{len(buzzing)} of {num_joints} joints stand out above the gait band "
          f"({names}), against a median share of {median_share:.2f} -- localised "
          "chatter, usually cured by a larger action_rate_l2 penalty or a lower "
          "action scale."
      )
    else:
      # Only a measurement that actually ran may declare the trace clean.
      lines.append("No high-frequency joint chatter stands out from the rest.")
  else:
    reason = smooth.get(
        "chatter_unmeasurable_reason", f"no {CHANNEL_JOINT_VEL} channel in the trace"
    )
    lines.append(
        f"Chatter unmeasurable: {reason} -- do not read the absence of a "
        "chatter finding as smooth motion."
    )

  if smooth.get("joint_jerk_rms", 0) > 5000:
    lines.append(
        f"Joint jerk RMS is very high ({smooth['joint_jerk_rms']:.0f} rad/s^3); "
        "motion is not smooth."
    )

  if track:
    err = track.get("lin_vel_error_rms")
    cmd_speed = track.get("commanded_speed_mean", 0.0)
    if err is not None and cmd_speed > 0.05:
      ratio = err / max(cmd_speed, 1e-6)
      if ratio > 0.5:
        lines.append(
            f"Linear velocity tracking is poor (RMS error {err:.2f} m/s against a "
            f"{cmd_speed:.2f} m/s command)."
        )
      else:
        lines.append(
            f"Linear velocity tracking is reasonable (RMS error {err:.2f} m/s, "
            f"command {cmd_speed:.2f} m/s)."
        )
    elif cmd_speed <= 0.05:
      lines.append("Command was near zero for this window; tracking numbers are weak evidence.")

  tilt = posture.get("tilt_deg_mean")
  if tilt is not None and tilt > 20:
    lines.append(f"Torso leans {tilt:.0f} deg from vertical on average -- posture is off.")

  if gait.get("contact_fraction") is not None:
    cf = gait["contact_fraction"]
    if cf > 0.95:
      lines.append("Feet are almost always in contact: the robot is shuffling, not stepping.")
    elif cf < 0.2:
      lines.append("Feet are rarely in contact: the robot is likely falling or hopping.")

  return lines


def summarize_metric_history(
    rows: Sequence[Dict[str, Any]], keys: Sequence[str], window: int = 20
) -> Dict[str, Any]:
  """Latest value, recent mean, and trend for selected iteration metrics."""
  out: Dict[str, Any] = {}
  for key in keys:
    series = [
        (row.get("iteration"), float(row[key]))
        for row in rows
        if isinstance(row.get(key), (int, float))
    ]
    if not series:
      continue
    values = np.array([v for _, v in series], dtype=np.float64)
    recent = values[-window:]
    entry: Dict[str, Any] = {
        "latest": round(float(values[-1]), 5),
        "mean_recent": round(float(np.mean(recent)), 5),
        "num_points": int(values.size),
    }
    if values.size >= 2 * window:
      delta = float(np.mean(recent) - np.mean(values[-2 * window : -window]))
    elif values.size >= 2:
      # Not enough history for two full windows. Compare the halves of what
      # exists (and say so below) instead of pitting a window mean against the
      # single first sample, which manufactures a trend out of noise.
      split = values.size // 2
      delta = float(np.mean(values[split:]) - np.mean(values[:split]))
      entry["trend_note"] = (
          f"short history ({values.size} points < {2 * window}); trend compares "
          "the halves of what exists"
      )
    else:
      entry["trend"] = "insufficient_history"
      out[key] = entry
      continue
    entry["delta_vs_previous_window"] = round(delta, 5)
    entry["trend"] = "up" if delta > 1e-6 else ("down" if delta < -1e-6 else "flat")
    out[key] = entry
  return out
