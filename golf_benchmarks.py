"""
golf_benchmarks.py
------------------
Pro-golfer swing benchmarks + comparison helpers for a single-camera
2D pose-estimation overlay tool.

HOW TO USE
==========
1. Your script already produces per-frame metrics (shoulder rotation, hip
   rotation, elbow angles, etc.).
2. First detect swing PHASES (see detect_phases() stub at bottom) so each
   metric can be read at the right checkpoint.
3. At each checkpoint, call compare(metric_name, value, phase) -> verdict.

SOURCE CONFIDENCE LEGEND  (be honest with yourself about these)
==============================================================
  SOURCED      = from 3D motion-capture research (Meister/Stanford et al.).
                 NOTE: these were measured in 3D. Your single 2D camera will
                 read projected angles, so treat them as DIRECTIONAL targets,
                 not exact truth. Your own swing-to-swing CONSISTENCY is the
                 more reliable signal.
  HEURISTIC    = widely-taught coaching number, NOT verified against the
                 academic biomechanics sources. Reasonable, but soft.
  UNCONFIRMED  = no clean published single value; track for CONSISTENCY only.

CSI = Clubhead Speed at Impact.  CV = Coefficient of Variation (lower = pros
are more consistent on this metric = the benchmark is more trustworthy).
"""

# ---------------------------------------------------------------------------
# BENCHMARKS
# Each entry: target value, +/- tolerance band, the phase it's measured at,
# a source-confidence tag, and a note. Edit freely as you learn.
# ---------------------------------------------------------------------------

BENCHMARKS = {
    # ----- POWER / WIND-UP -------------------------------------------------
    "x_factor": {
        "target": 56.0, "tolerance": 8.0, "unit": "deg", "phase": "top",
        "source": "SOURCED",
        "note": "Hip-to-shoulder separation. Pros ~56 deg, very consistent "
                "(CV 7.4%). Compute as shoulder_rotation - hip_rotation. "
                "Your single highest-value add.",
    },
    "shoulder_rotation_top": {
        "target": 90.0, "tolerance": 10.0, "unit": "deg", "phase": "top",
        "source": "HEURISTIC",
        "note": "Classic '90 deg shoulders at the top' coaching checkpoint. "
                "Academic studies confirm pros turn MORE than amateurs, but "
                "the exact 90 is a teaching heuristic, not a measured mean.",
    },
    "hip_rotation_top": {
        "target": 45.0, "tolerance": 10.0, "unit": "deg", "phase": "top",
        "source": "HEURISTIC",
        "note": "Classic '45 deg hips at the top'. Same caveat as above.",
    },

    # ----- TILT / OBLIQUITY ------------------------------------------------
    "s_factor": {
        "target": 48.0, "tolerance": 8.0, "unit": "deg", "phase": "impact",
        "source": "SOURCED",
        "note": "Shoulder obliquity (tilt of lead shoulder). Pros ~48 deg, "
                "peaks just AFTER impact (CV 8.4%).",
    },
    "o_factor": {
        "target": None, "tolerance": None, "unit": "deg", "phase": "impact",
        "source": "UNCONFIRMED",
        "note": "Pelvic obliquity. Correlates with CSI but less tightly than "
                "X/S, and the study did not give a clean single benchmark "
                "value. Track it for CONSISTENCY across your own swings.",
    },

    # ----- POSTURE / SAFETY ------------------------------------------------
    "spine_angle_change": {
        "target": 0.0, "tolerance": 5.0, "unit": "deg",
        "phase": "address_to_impact",
        "source": "SOURCED",
        "note": "Change in spine angle from address to impact. Good golfers "
                "hold it within ~+/-5 deg (diagnostic-system spec). 'Early "
                "extension' = standing up out of posture = a common fault.",
    },

    # ----- TIMING ----------------------------------------------------------
    "tempo_ratio": {
        "target": 3.0, "tolerance": 0.5, "unit": "ratio",
        "phase": "address_to_impact",
        "source": "HEURISTIC",
        "note": "Backswing time : downswing time, classically ~3:1. Compute "
                "from your frame counts (frames_backswing / frames_downswing) "
                "since you know the fps. Widely taught, not verified vs the "
                "academic sources here.",
    },
}

# Kinematic sequence is special: it is an ORDER, not a single number.
# Pros initiate the downswing with the PELVIS, THEN the upper torso (SOURCED).
# i.e. peak pelvis angular velocity should occur BEFORE peak torso angular
# velocity. You verify this by finding the frame of each peak velocity and
# checking pelvis_peak_frame < torso_peak_frame. This is arguably the #2
# most important thing to add after X-factor.
KINEMATIC_SEQUENCE_RULE = ("pelvis", "torso")  # expected firing order


# ---------------------------------------------------------------------------
# COMPARISON HELPER
# ---------------------------------------------------------------------------

def compare(metric_name, value, phase=None):
    """Return a verdict dict for a measured value vs its benchmark.

    verdict: 'in_band' | 'low' | 'high' | 'no_benchmark' | 'wrong_phase'
    """
    b = BENCHMARKS.get(metric_name)
    if b is None:
        return {"verdict": "no_benchmark", "metric": metric_name}

    if phase is not None and b["phase"] not in (phase, "address_to_impact"):
        return {"verdict": "wrong_phase", "metric": metric_name,
                "expected_phase": b["phase"], "got_phase": phase}

    if b["target"] is None:
        return {"verdict": "no_benchmark", "metric": metric_name,
                "note": "Track for consistency only - no published target."}

    low = b["target"] - b["tolerance"]
    high = b["target"] + b["tolerance"]
    if value < low:
        verdict = "low"
    elif value > high:
        verdict = "high"
    else:
        verdict = "in_band"

    return {
        "verdict": verdict, "metric": metric_name, "value": round(value, 1),
        "target": b["target"], "band": (low, high), "source": b["source"],
    }


def check_kinematic_sequence(pelvis_peak_frame, torso_peak_frame):
    """Pros fire pelvis BEFORE torso on the downswing. Returns True if good."""
    return pelvis_peak_frame < torso_peak_frame

