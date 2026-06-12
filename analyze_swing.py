#!/usr/bin/env python3
"""Analyze a golf swing video with MediaPipe Tasks API and OpenCV."""

from __future__ import annotations

import argparse
import math
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
from world_rotation import RotationTracker

DEFAULT_MODEL_PATH = Path(".models/pose_landmarker_full.task")
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

# 33-landmark BlazePose index mapping.
LANDMARK_INDEX = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

POSE_CONNECTIONS = tuple(
    (connection.start, connection.end)
    for connection in mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
)
ROTATIONS = (0, 90, 180, 270)
HUD_BG = (20, 20, 20)
HUD_TEXT = (255, 255, 255)
SKELETON_COLOR = (0, 255, 255)
SKELETON_OUTLINE = (0, 0, 0)
POINT_COLOR = (255, 80, 0)


@dataclass(frozen=True)
class SideMapping:
    lead: str
    trail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze golf swing video with MediaPipe Tasks API + OpenCV."
    )
    parser.add_argument("--input", required=True, help="Path to input iPhone video.")
    parser.add_argument(
        "--output",
        default="swing_analyze.mp4",
        help="Output annotated video path (default: swing_analyze.mp4).",
    )
    parser.add_argument(
        "--handedness",
        default="right",
        choices=("right", "left"),
        help="Golfer handedness for lead/trail labeling (default: right).",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Local path for Pose Landmarker .task model.",
    )
    return parser.parse_args()


def ensure_model_exists(model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return model_path

    print(f"Downloading MediaPipe pose model to {model_path} ...")
    urllib.request.urlretrieve(DEFAULT_MODEL_URL, model_path)
    return model_path


def create_pose_landmarker(
    model_path: Path, running_mode: mp.tasks.vision.RunningMode
) -> mp.tasks.vision.PoseLandmarker:
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def try_get_metadata_rotation(capture: cv2.VideoCapture) -> Optional[int]:
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    if not hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
        return None

    meta = capture.get(cv2.CAP_PROP_ORIENTATION_META)
    if math.isnan(meta):
        return None

    return normalize_rotation(meta)


def normalize_rotation(value: float) -> Optional[int]:
    rounded = int(round(value))
    normalized = rounded % 360
    snapped = min(ROTATIONS, key=lambda x: abs(x - normalized))
    if abs(snapped - normalized) > 5:
        return None
    return snapped


def rotate_frame(frame: cv2.typing.MatLike, rotation: int) -> cv2.typing.MatLike:
    if rotation == 0:
        return frame
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation: {rotation}")


def get_landmark_confidence(landmark) -> float:
    visibility = float(getattr(landmark, "visibility", 1.0))
    presence = float(getattr(landmark, "presence", 1.0))
    return min(visibility, presence)


def is_landmark_reliable(landmark, min_confidence: float = 0.4) -> bool:
    return get_landmark_confidence(landmark) >= min_confidence


def score_pose_upright(landmarks: Sequence) -> float:
    # Use simple body geometry checks to choose upright orientation.
    nose = landmarks[LANDMARK_INDEX["nose"]]
    left_shoulder = landmarks[LANDMARK_INDEX["left_shoulder"]]
    right_shoulder = landmarks[LANDMARK_INDEX["right_shoulder"]]
    left_hip = landmarks[LANDMARK_INDEX["left_hip"]]
    right_hip = landmarks[LANDMARK_INDEX["right_hip"]]

    core_points = (nose, left_shoulder, right_shoulder, left_hip, right_hip)
    conf_score = sum(get_landmark_confidence(lm) for lm in core_points)

    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2.0
    hip_center_y = (left_hip.y + right_hip.y) / 2.0

    score = conf_score
    if nose.y < shoulder_center_y:
        score += 2.0
    else:
        score -= 2.0

    if shoulder_center_y < hip_center_y:
        score += 3.0
    else:
        score -= 3.0

    shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
    hip_slope = abs(left_hip.y - right_hip.y)
    score += max(0.0, 1.0 - shoulder_slope * 8.0)
    score += max(0.0, 1.0 - hip_slope * 8.0)
    return score


def score_rotation_image_mode(
    landmarker: mp.tasks.vision.PoseLandmarker,
    frame_bgr: cv2.typing.MatLike,
    rotation: int,
) -> float:
    rotated = rotate_frame(frame_bgr, rotation)
    rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(image)
    if not result.pose_landmarks:
        return -1e9
    return score_pose_upright(result.pose_landmarks[0])


def choose_rotation(
    first_frame: cv2.typing.MatLike, model_path: Path, metadata_rotation: Optional[int]
) -> int:
    landmarker = create_pose_landmarker(
        model_path=model_path, running_mode=mp.tasks.vision.RunningMode.IMAGE
    )
    try:
        best_rotation = 0
        best_score = -1e9
        scores: Dict[int, float] = {}
        for rotation in ROTATIONS:
            score = score_rotation_image_mode(landmarker, first_frame, rotation)
            scores[rotation] = score
            if score > best_score:
                best_score = score
                best_rotation = rotation

        if metadata_rotation is None:
            return best_rotation

        metadata_score = scores.get(metadata_rotation, -1e9)
        # Prefer metadata unless pose checks indicate a clearly better alternative.
        if best_score - metadata_score > 1.5:
            return best_rotation
        return metadata_rotation
    finally:
        landmarker.close()


def clamp_point(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    px = int(min(max(x, 0.0), 1.0) * (width - 1))
    py = int(min(max(y, 0.0), 1.0) * (height - 1))
    return (px, py)


def draw_line_with_outline(
    frame: cv2.typing.MatLike,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.line(frame, start, end, SKELETON_OUTLINE, thickness + 4, cv2.LINE_AA)
    cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)


def draw_point_with_outline(
    frame: cv2.typing.MatLike, point: Tuple[int, int], radius: int
) -> None:
    cv2.circle(frame, point, radius + 2, SKELETON_OUTLINE, -1, cv2.LINE_AA)
    cv2.circle(frame, point, radius, POINT_COLOR, -1, cv2.LINE_AA)


def draw_skeleton_overlay(frame: cv2.typing.MatLike, landmarks: Sequence) -> None:
    h, w = frame.shape[:2]
    line_thickness = max(4, int(min(h, w) * 0.005))
    point_radius = max(5, int(min(h, w) * 0.006))

    for start_idx, end_idx in POSE_CONNECTIONS:
        start_lm = landmarks[start_idx]
        end_lm = landmarks[end_idx]
        if not (is_landmark_reliable(start_lm, 0.35) and is_landmark_reliable(end_lm, 0.35)):
            continue
        start_pt = clamp_point(start_lm.x, start_lm.y, w, h)
        end_pt = clamp_point(end_lm.x, end_lm.y, w, h)
        draw_line_with_outline(frame, start_pt, end_pt, SKELETON_COLOR, line_thickness)

    for idx in range(33):
        landmark = landmarks[idx]
        if not is_landmark_reliable(landmark, 0.25):
            continue
        point = clamp_point(landmark.x, landmark.y, w, h)
        draw_point_with_outline(frame, point, point_radius)


def point_tuple(landmarks: Sequence, idx: int) -> Tuple[float, float]:
    landmark = landmarks[idx]
    return (float(landmark.x), float(landmark.y))


def angle_from_three_points(
    a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
) -> Optional[float]:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    ba_norm = math.hypot(ba[0], ba[1])
    bc_norm = math.hypot(bc[0], bc[1])
    if ba_norm < 1e-8 or bc_norm < 1e-8:
        return None

    dot = (ba[0] * bc[0]) + (ba[1] * bc[1])
    cosine = max(-1.0, min(1.0, dot / (ba_norm * bc_norm)))
    return math.degrees(math.acos(cosine))


def line_rotation_angle(left: Tuple[float, float], right: Tuple[float, float]) -> float:
    delta_x = right[0] - left[0]
    delta_y = right[1] - left[1]
    return math.degrees(math.atan2(delta_y, delta_x))


def side_mapping_for_handedness(handedness: str) -> SideMapping:
    if handedness == "right":
        return SideMapping(lead="left", trail="right")
    return SideMapping(lead="right", trail="left")


def elbow_angle(landmarks: Sequence, side: str) -> Optional[float]:
    shoulder = landmarks[LANDMARK_INDEX[f"{side}_shoulder"]]
    elbow = landmarks[LANDMARK_INDEX[f"{side}_elbow"]]
    wrist = landmarks[LANDMARK_INDEX[f"{side}_wrist"]]
    if not (
        is_landmark_reliable(shoulder, 0.4)
        and is_landmark_reliable(elbow, 0.4)
        and is_landmark_reliable(wrist, 0.4)
    ):
        return None
    return angle_from_three_points(
        point_tuple(landmarks, LANDMARK_INDEX[f"{side}_shoulder"]),
        point_tuple(landmarks, LANDMARK_INDEX[f"{side}_elbow"]),
        point_tuple(landmarks, LANDMARK_INDEX[f"{side}_wrist"]),
    )


def rotation_metric(landmarks: Sequence, left_name: str, right_name: str) -> Optional[float]:
    left_landmark = landmarks[LANDMARK_INDEX[left_name]]
    right_landmark = landmarks[LANDMARK_INDEX[right_name]]
    if not (is_landmark_reliable(left_landmark, 0.4) and is_landmark_reliable(right_landmark, 0.4)):
        return None
    return line_rotation_angle(
        point_tuple(landmarks, LANDMARK_INDEX[left_name]),
        point_tuple(landmarks, LANDMARK_INDEX[right_name]),
    )


def compute_metrics(landmarks: Sequence, handedness: str) -> Dict[str, Optional[float]]:
    side_map = side_mapping_for_handedness(handedness)
    return {
        "Lead Elbow": elbow_angle(landmarks, side_map.lead),
        "Trail Elbow": elbow_angle(landmarks, side_map.trail),
        "Hip Rotation": rotation_metric(landmarks, "left_hip", "right_hip"),
        "Shoulder Rotation": rotation_metric(landmarks, "left_shoulder", "right_shoulder"),
    }


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f} deg"


def draw_metrics_hud(
    frame: cv2.typing.MatLike, metrics: Dict[str, Optional[float]], frame_idx: int
) -> None:
    h, _ = frame.shape[:2]
    panel_h = max(150, int(h * 0.22))
    cv2.rectangle(frame, (0, 0), (560, panel_h), HUD_BG, -1)

    cv2.putText(
        frame,
        f"Frame: {frame_idx}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        HUD_TEXT,
        2,
        cv2.LINE_AA,
    )

    y = 68
    for label, value in metrics.items():
        cv2.putText(
            frame,
            f"{label}: {format_metric(value)}",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            HUD_TEXT,
            2,
            cv2.LINE_AA,
        )
        y += 32


def safe_fps(capture: cv2.VideoCapture) -> float:
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        return 30.0
    return fps


def get_first_frame(capture: cv2.VideoCapture):
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError("Failed to read first frame from input video.")
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frame


def open_writer(output_path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create output writer at {output_path}.")
    return writer


def analyze_video(
    input_path: Path,
    output_path: Path,
    handedness: str,
    model_path: Path,
) -> None:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open input video: {input_path}")

    landmarker = None
    writer = None
    try:
        first_frame = get_first_frame(capture)
        metadata_rotation = try_get_metadata_rotation(capture)
        selected_rotation = choose_rotation(first_frame, model_path, metadata_rotation)
        print(f"Using clockwise rotation: {selected_rotation} deg")

        fps = safe_fps(capture)
        first_upright = rotate_frame(first_frame, selected_rotation)
        out_h, out_w = first_upright.shape[:2]
        writer = open_writer(output_path, fps, (out_w, out_h))

        landmarker = create_pose_landmarker(
            model_path=model_path, running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        tracker = RotationTracker(smoothing_window=5)
        address_set = False

        frame_idx = 0
        timestamp_ms = 0
        step_ms = max(1, int(round(1000.0 / fps)))

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            upright = rotate_frame(frame, selected_rotation)
            rgb = cv2.cvtColor(upright, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            metrics: Dict[str, Optional[float]] = {
                "Lead Elbow": None,
                "Trail Elbow": None,
                "Hip Rotation": None,
                "Shoulder Rotation": None,
            }
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                draw_skeleton_overlay(upright, landmarks)
                metrics = compute_metrics(landmarks, handedness)
                if not address_set and result.pose_world_landmarks:
                    tracker.set_address(result.pose_world_landmarks[0])
                    address_set = True
                if result.pose_world_landmarks:
                    world_landmarks = result.pose_world_landmarks[0]
                    hip_rot, sho_rot, x_factor = tracker.metrics(world_landmarks)
                    metrics["Hip Rotation"] = hip_rot
                    metrics["Shoulder Rotation"] = sho_rot
                    metrics["X-Factor"] = x_factor

            draw_metrics_hud(upright, metrics, frame_idx)
            writer.write(upright)

            frame_idx += 1
            timestamp_ms += step_ms

            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames...")

        print(f"Completed. Processed {frame_idx} frames.")
    finally:
        if landmarker is not None:
            landmarker.close()
        if writer is not None:
            writer.release()
        capture.release()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()

    if not input_path.exists():
        print(f"Input video not found: {input_path}", file=sys.stderr)
        return 1

    try:
        ensured_model = ensure_model_exists(model_path)
        analyze_video(
            input_path=input_path,
            output_path=output_path,
            handedness=args.handedness,
            model_path=ensured_model,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Annotated video written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
