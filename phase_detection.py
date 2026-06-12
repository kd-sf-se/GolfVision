#!/usr/bin/env python3
"""Swing phase detection helpers for GolfVision."""

from __future__ import annotations

import math
from typing import Dict, List, Optional


def _first_valid(values: List[Optional[float]], fallback: float = 0.0) -> float:
    for value in values:
        if value is not None and math.isfinite(value):
            return value
    return fallback


def _argmin_valid(values: List[Optional[float]], start_idx: int = 0) -> Optional[int]:
    best_idx = None
    best_value = None
    for idx in range(start_idx, len(values)):
        value = values[idx]
        if value is None or not math.isfinite(value):
            continue
        if best_value is None or value < best_value:
            best_value = value
            best_idx = idx
    return best_idx


def _argmax_valid(values: List[Optional[float]], start_idx: int = 0) -> Optional[int]:
    best_idx = None
    best_value = None
    for idx in range(start_idx, len(values)):
        value = values[idx]
        if value is None or not math.isfinite(value):
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_idx = idx
    return best_idx


def detect_phases(frames) -> Dict[str, int]:
    """Detect key swing checkpoints from per-frame metrics."""
    if not frames:
        return {"address": 0, "top": 0, "impact": 0, "finish": 0}

    shoulder_raw = [frame.get("shoulder_rot_raw") for frame in frames]
    wrist_y = [frame.get("wrist_y") for frame in frames]
    hand_speed = [frame.get("hand_speed") for frame in frames]

    baseline = _first_valid(shoulder_raw, fallback=0.0)
    motion_onset = None
    for idx in range(len(shoulder_raw) - 2):
        sustained = True
        for check in (idx, idx + 1, idx + 2):
            value = shoulder_raw[check]
            if value is None or abs(value - baseline) <= 5.0:
                sustained = False
                break
        if sustained:
            motion_onset = idx
            break

    if motion_onset is None:
        address = 0
    else:
        address = max(0, motion_onset - 3)

    address_baseline = shoulder_raw[address]
    if address_baseline is None:
        address_baseline = baseline

    shoulder_rel = []
    for value in shoulder_raw:
        if value is None:
            shoulder_rel.append(None)
        else:
            shoulder_rel.append(abs(value - address_baseline))

    top_by_rotation = _argmax_valid(shoulder_rel, start_idx=address) or address
    top_by_wrist = _argmin_valid(wrist_y, start_idx=address) or top_by_rotation
    if abs(top_by_rotation - top_by_wrist) > 5:
        top = top_by_wrist
    else:
        top = top_by_rotation

    impact = _argmax_valid(hand_speed, start_idx=top + 1)
    if impact is None:
        impact = min(len(frames) - 1, top + 1)

    finish = len(frames) - 1
    for idx in range(len(frames) - 1, -1, -1):
        if frames[idx].get("pose_reliable", False):
            finish = idx
            break

    return {"address": address, "top": top, "impact": impact, "finish": finish}


def kinematic_sequence_ok(frames, top_idx: int) -> bool:
    """Return True if hip peak angular velocity occurs before shoulder peak."""
    if not frames or top_idx >= len(frames) - 1:
        return False

    hand_speed = [frame.get("hand_speed") for frame in frames]
    impact_idx = _argmax_valid(hand_speed, start_idx=top_idx + 1)
    if impact_idx is None or impact_idx <= top_idx:
        impact_idx = len(frames) - 1

    hip_peak_frame = None
    hip_peak_vel = -1.0
    sho_peak_frame = None
    sho_peak_vel = -1.0

    for idx in range(top_idx + 1, impact_idx + 1):
        prev = frames[idx - 1]
        current = frames[idx]

        hip_prev = prev.get("hip_rotation")
        hip_now = current.get("hip_rotation")
        if hip_prev is not None and hip_now is not None:
            hip_vel = abs(hip_now - hip_prev)
            if hip_vel > hip_peak_vel:
                hip_peak_vel = hip_vel
                hip_peak_frame = idx

        sho_prev = prev.get("shoulder_rotation")
        sho_now = current.get("shoulder_rotation")
        if sho_prev is not None and sho_now is not None:
            sho_vel = abs(sho_now - sho_prev)
            if sho_vel > sho_peak_vel:
                sho_peak_vel = sho_vel
                sho_peak_frame = idx

    if hip_peak_frame is None or sho_peak_frame is None:
        return False
    return hip_peak_frame < sho_peak_frame
