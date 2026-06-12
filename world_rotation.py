#!/usr/bin/env python3
"""
world_rotation.py
-----------------
True 3D rotation metrics for GolfVision, computed from MediaPipe's
pose_world_landmarks instead of the 2D image landmarks.

WHY THIS EXISTS
===============
The original rotation_metric() in analyze_swing.py uses atan2(dy, dx) on the
2D IMAGE landmarks. That measures the on-screen TILT of the shoulder/hip line,
not rotation around the spine. From a down-the-line camera, a real backswing
turn happens mostly in DEPTH (toward/away from the lens), which barely changes
on-screen x/y -- so the metric massively under-reports (e.g. 10.8 deg for a
near-full turn). This module fixes that by measuring rotation in the
TRANSVERSE (horizontal) plane using the world landmarks' real depth axis.

COORDINATE NOTE (MediaPipe world landmarks)
==========================================
  x = left/right (metres), origin at mid-hip
  y = vertical   (metres), DOWN is positive in image convention
  z = depth      (metres), toward/away from camera
Rotation around the vertical spine axis lives in the x-z plane -> atan2(dz, dx).

CAVEATS (be honest with yourself)
=================================
  * Single-camera depth (z) is ESTIMATED by MediaPipe, not stereo-measured.
    Expect jitter -> we smooth (see RotationTracker).
  * Angles are reported RELATIVE TO ADDRESS, which is what coaches mean by
    "shoulder turn". You must tell the tracker which frame is address.
  * Still no club/ball. This module only covers body rotation.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Sequence, Deque


# --- landmark indices (same mapping as analyze_swing.py) -------------------
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24


def _xz_heading(p_left, p_right) -> Optional[float]:
    """Compass heading (deg) of the left->right segment in the x-z ground
    plane. p_left / p_right are world landmarks with .x, .y, .z in metres."""
    dx = p_right.x - p_left.x
    dz = p_right.z - p_left.z
    if math.hypot(dx, dz) < 1e-6:
        return None
    return math.degrees(math.atan2(dz, dx))


def _wrap_180(deg: float) -> float:
    """Normalize an angle difference into (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


def _confident(landmark, min_conf: float = 0.4) -> bool:
    vis = float(getattr(landmark, "visibility", 1.0))
    pres = float(getattr(landmark, "presence", 1.0))
    return min(vis, pres) >= min_conf


class RotationTracker:
    """Tracks address-relative hip and shoulder rotation across a swing.

    Usage per frame:
        tracker.update(world_landmarks)            # every frame
        ...once you know address frame...
        tracker.set_address(world_landmarks_at_address)
        hip, sho, xf = tracker.metrics(world_landmarks)

    In practice you don't know the address frame until you've seen the swing.
    Two options:
      A) Two-pass: collect all frames, detect address (first stable frame),
         set it, then re-read metrics. Cleanest.
      B) One-pass live: assume frame 0 is address (golfer set up before
         recording). Simpler, slightly less robust. Good enough to start.
    """

    def __init__(self, smoothing_window: int = 5) -> None:
        self._addr_shoulder: Optional[float] = None
        self._addr_hip: Optional[float] = None
        self._sho_buf: Deque[float] = deque(maxlen=smoothing_window)
        self._hip_buf: Deque[float] = deque(maxlen=smoothing_window)

    def set_address(self, world_landmarks: Sequence) -> None:
        """Capture the address-frame headings as the zero reference."""
        self._addr_shoulder = self._raw_shoulder(world_landmarks)
        self._addr_hip = self._raw_hip(world_landmarks)

    def _raw_shoulder(self, wl: Sequence) -> Optional[float]:
        ls, rs = wl[L_SHOULDER], wl[R_SHOULDER]
        if not (_confident(ls) and _confident(rs)):
            return None
        return _xz_heading(ls, rs)

    def _raw_hip(self, wl: Sequence) -> Optional[float]:
        lh, rh = wl[L_HIP], wl[R_HIP]
        if not (_confident(lh) and _confident(rh)):
            return None
        return _xz_heading(lh, rh)

    def _smooth(self, buf: Deque[float], value: Optional[float]) -> Optional[float]:
        if value is None:
            return buf[-1] if buf else None
        buf.append(value)
        return sum(buf) / len(buf)

    def metrics(self, world_landmarks: Sequence):
        """Return (hip_rotation, shoulder_rotation, x_factor) in degrees,
        all relative to address. Any may be None if landmarks are unreliable
        or address hasn't been set yet."""
        raw_sho = self._raw_shoulder(world_landmarks)
        raw_hip = self._raw_hip(world_landmarks)

        sho_abs = self._smooth(self._sho_buf, raw_sho)
        hip_abs = self._smooth(self._hip_buf, raw_hip)

        if self._addr_shoulder is None or sho_abs is None:
            shoulder_rot = None
        else:
            shoulder_rot = abs(_wrap_180(sho_abs - self._addr_shoulder))

        if self._addr_hip is None or hip_abs is None:
            hip_rot = None
        else:
            hip_rot = abs(_wrap_180(hip_abs - self._addr_hip))

        if shoulder_rot is None or hip_rot is None:
            x_factor = None
        else:
            x_factor = shoulder_rot - hip_rot

        return hip_rot, shoulder_rot, x_factor
