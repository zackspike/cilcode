"""Video processor module.

Analyses every frame of a video file using MediaPipe Face Landmarker and
Pose Landmarker (Tasks API) simultaneously to produce attention-triangulation
metadata.

Model files are downloaded automatically to ``~/.cache/cilcode/models/`` on
first use, or you can supply explicit paths via the ``face_model_path`` and
``pose_model_path`` parameters of :func:`analyse`.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe Tasks aliases
# ---------------------------------------------------------------------------

_BaseOptions = mp.tasks.BaseOptions
_FaceLandmarker = mp.tasks.vision.FaceLandmarker
_FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
_PoseLandmarker = mp.tasks.vision.PoseLandmarker
_PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
_PoseLandmark = mp.tasks.vision.PoseLandmark
_RunningMode = mp.tasks.vision.RunningMode

# ---------------------------------------------------------------------------
# Model download helpers
# ---------------------------------------------------------------------------

_MODELS_CACHE_DIR = Path.home() / ".cache" / "video_grader_ai" / "models"

_MODEL_URLS = {
    "face": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "pose": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}

_MODEL_FILENAMES = {
    "face": "face_landmarker.task",
    "pose": "pose_landmarker_lite.task",
}


def _ensure_model(key: str) -> Path:
    """Return the path to a cached model, downloading it if necessary."""
    _MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = _MODELS_CACHE_DIR / _MODEL_FILENAMES[key]
    if not dest.exists():
        url = _MODEL_URLS[key]
        print(f"      Downloading {key} model from {url} …")
        urllib.request.urlretrieve(url, dest)
    return dest


# ---------------------------------------------------------------------------
# Landmark index constants
# ---------------------------------------------------------------------------

# Face-Mesh landmark indices used for gaze / attention estimation.
# Nose tip: 1; Chin: 152; eye corners, mouth corners.
_FACE_ATTENTION_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}

# Pose landmark indices for body-language cues.
_POSE_LANDMARK_KEYS = {
    "left_shoulder": _PoseLandmark.LEFT_SHOULDER.value,
    "right_shoulder": _PoseLandmark.RIGHT_SHOULDER.value,
    "left_wrist": _PoseLandmark.LEFT_WRIST.value,
    "right_wrist": _PoseLandmark.RIGHT_WRIST.value,
    "left_elbow": _PoseLandmark.LEFT_ELBOW.value,
    "right_elbow": _PoseLandmark.RIGHT_ELBOW.value,
}

# ---------------------------------------------------------------------------
# Per-frame extraction helpers
# ---------------------------------------------------------------------------


def _extract_face_metadata(face_landmarks: list) -> dict:
    """Return normalised landmark coordinates and a simple attention score.

    Parameters
    ----------
    face_landmarks:
        List of :class:`mediapipe.tasks.components.containers.NormalizedLandmark`
        objects for a single detected face.
    """
    points = {}
    for name, idx in _FACE_ATTENTION_LANDMARKS.items():
        lm_pt = face_landmarks[idx]
        points[name] = {
            "x": round(lm_pt.x, 4),
            "y": round(lm_pt.y, 4),
            "z": round(lm_pt.z, 4),
        }

    # Rough head-pose deviation from centre as an attention proxy.
    # Horizontal offset of the nose tip from 0.5 (normalised frame width).
    nose_x = face_landmarks[_FACE_ATTENTION_LANDMARKS["nose_tip"]].x
    attention_score = max(0.0, 1.0 - abs(nose_x - 0.5) * 4)

    return {"landmarks": points, "attention_score": round(attention_score, 4)}


def _extract_pose_metadata(pose_landmarks: list) -> dict:
    """Return normalised shoulder / wrist / elbow positions and gesture cues.

    Parameters
    ----------
    pose_landmarks:
        List of :class:`mediapipe.tasks.components.containers.NormalizedLandmark`
        objects for a single detected person.
    """
    points = {}
    for name, idx in _POSE_LANDMARK_KEYS.items():
        pt = pose_landmarks[idx]
        points[name] = {
            "x": round(pt.x, 4),
            "y": round(pt.y, 4),
            "z": round(pt.z, 4),
            "visibility": round(pt.visibility if pt.visibility is not None else 0.0, 4),
        }

    # Simple gesture heuristic: are wrists raised above the shoulders?
    # In normalised image coordinates, a smaller y value means higher up.
    left_wrist_raised = (
        points["left_wrist"]["y"] < points["left_shoulder"]["y"]
        and points["left_wrist"]["visibility"] > 0.5
    )
    right_wrist_raised = (
        points["right_wrist"]["y"] < points["right_shoulder"]["y"]
        and points["right_wrist"]["visibility"] > 0.5
    )

    return {
        "landmarks": points,
        "left_wrist_raised": left_wrist_raised,
        "right_wrist_raised": right_wrist_raised,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyse(
    video_path: str | os.PathLike,
    *,
    sample_every_n_frames: int = 15,
    max_frames: int = 600,
    face_model_path: str | os.PathLike | None = None,
    pose_model_path: str | os.PathLike | None = None,
) -> dict:
    """Run Face Landmarker + Pose Landmarker analysis on *video_path*.

    Parameters
    ----------
    video_path:
        Path to the video file.
    sample_every_n_frames:
        Process one frame every *n* frames to keep processing fast while
        still capturing the overall presentation dynamics.
    max_frames:
        Hard cap on the total number of frames sampled.
    face_model_path:
        Path to the Face Landmarker ``.task`` model file.  If *None*, the
        model is downloaded automatically to ``~/.cache/cilcode/models/``.
    pose_model_path:
        Path to the Pose Landmarker ``.task`` model file.  If *None*, the
        model is downloaded automatically.

    Returns
    -------
    dict
        Keys:

        * ``total_frames_sampled`` – int
        * ``face_detected_frames`` – int
        * ``pose_detected_frames`` – int
        * ``avg_attention_score`` – float in [0, 1]
        * ``gesture_frames`` – dict with ``left_wrist_raised`` and
          ``right_wrist_raised`` counts.
        * ``frame_metadata`` – list of per-frame dicts (may be large).
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    face_model = Path(face_model_path) if face_model_path else _ensure_model("face")
    pose_model = Path(pose_model_path) if pose_model_path else _ensure_model("pose")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    face_options = _FaceLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=str(face_model)),
        running_mode=_RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=str(pose_model)),
        running_mode=_RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_metadata: list[dict] = []
    attention_scores: list[float] = []
    gesture_counts = {"left_wrist_raised": 0, "right_wrist_raised": 0}
    face_detected = 0
    pose_detected = 0
    frame_idx = 0
    sampled = 0

    with (
        _FaceLandmarker.create_from_options(face_options) as face_landmarker,
        _PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
    ):
        while sampled < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % sample_every_n_frames != 0:
                continue

            sampled += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(frame_idx * 1000 / fps)

            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            face_meta: dict | None = None
            pose_meta: dict | None = None

            if face_result.face_landmarks:
                face_detected += 1
                face_meta = _extract_face_metadata(face_result.face_landmarks[0])
                attention_scores.append(face_meta["attention_score"])

            if pose_result.pose_landmarks:
                pose_detected += 1
                pose_meta = _extract_pose_metadata(pose_result.pose_landmarks[0])
                if pose_meta["left_wrist_raised"]:
                    gesture_counts["left_wrist_raised"] += 1
                if pose_meta["right_wrist_raised"]:
                    gesture_counts["right_wrist_raised"] += 1

            frame_metadata.append(
                {
                    "frame_index": frame_idx,
                    "face": face_meta,
                    "pose": pose_meta,
                }
            )

    cap.release()

    avg_attention = float(np.mean(attention_scores)) if attention_scores else 0.0

    return {
        "total_frames_sampled": sampled,
        "face_detected_frames": face_detected,
        "pose_detected_frames": pose_detected,
        "avg_attention_score": round(avg_attention, 4),
        "gesture_frames": gesture_counts,
        "frame_metadata": frame_metadata,
    }
