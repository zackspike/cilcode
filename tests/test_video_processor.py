"""Tests for video_processor.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import video_processor


def _make_landmark(x=0.5, y=0.4, z=0.0, visibility=1.0):
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    lm.visibility = visibility
    return lm


class TestExtractFaceMetadata:
    def _make_face_landmarks(self, nose_x=0.5):
        """Build a list of 478 fake landmarks for FaceLandmarker output."""
        lms = [_make_landmark() for _ in range(478)]
        # Override nose tip (index 1)
        lms[1] = _make_landmark(x=nose_x)
        return lms

    def test_returns_expected_keys(self):
        result = video_processor._extract_face_metadata(self._make_face_landmarks())
        assert "landmarks" in result
        assert "attention_score" in result

    def test_attention_score_range(self):
        result = video_processor._extract_face_metadata(self._make_face_landmarks(nose_x=0.5))
        assert 0.0 <= result["attention_score"] <= 1.0

    def test_attention_score_drops_when_off_centre(self):
        score_centre = video_processor._extract_face_metadata(
            self._make_face_landmarks(nose_x=0.5)
        )["attention_score"]
        score_side = video_processor._extract_face_metadata(
            self._make_face_landmarks(nose_x=0.9)
        )["attention_score"]
        assert score_centre > score_side


class TestExtractPoseMetadata:
    def _make_pose_landmarks(self, left_wrist_above=False, right_wrist_above=False):
        """Build a list of 33 fake pose landmarks."""
        import mediapipe as mp

        lms = [_make_landmark() for _ in range(33)]

        left_shoulder_idx = mp.tasks.vision.PoseLandmark.LEFT_SHOULDER.value
        right_shoulder_idx = mp.tasks.vision.PoseLandmark.RIGHT_SHOULDER.value
        left_wrist_idx = mp.tasks.vision.PoseLandmark.LEFT_WRIST.value
        right_wrist_idx = mp.tasks.vision.PoseLandmark.RIGHT_WRIST.value

        lms[left_shoulder_idx] = _make_landmark(x=0.3, y=0.5)
        lms[right_shoulder_idx] = _make_landmark(x=0.7, y=0.5)
        lms[left_wrist_idx] = _make_landmark(x=0.3, y=0.35 if left_wrist_above else 0.65)
        lms[right_wrist_idx] = _make_landmark(x=0.7, y=0.35 if right_wrist_above else 0.65)

        return lms

    def test_returns_expected_keys(self):
        result = video_processor._extract_pose_metadata(self._make_pose_landmarks())
        assert "landmarks" in result
        assert "left_wrist_raised" in result
        assert "right_wrist_raised" in result

    def test_wrist_not_raised(self):
        result = video_processor._extract_pose_metadata(
            self._make_pose_landmarks(left_wrist_above=False, right_wrist_above=False)
        )
        assert result["left_wrist_raised"] is False
        assert result["right_wrist_raised"] is False

    def test_wrist_raised(self):
        result = video_processor._extract_pose_metadata(
            self._make_pose_landmarks(left_wrist_above=True, right_wrist_above=True)
        )
        assert result["left_wrist_raised"] is True
        assert result["right_wrist_raised"] is True


class TestAnalyse:
    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            video_processor.analyse("/nonexistent/video.mp4")

    def test_returns_expected_keys(self, tmp_path):
        """Analyse returns the expected summary keys when mocked end-to-end."""
        dummy_video = tmp_path / "video.mp4"
        dummy_video.write_bytes(b"\x00")

        # Fake model files so _ensure_model is not called
        face_model = tmp_path / "face.task"
        pose_model = tmp_path / "pose.task"
        face_model.write_bytes(b"\x00")
        pose_model.write_bytes(b"\x00")

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, dummy_frame), (False, None)]

        # FaceLandmarker mock
        mock_face_result = MagicMock()
        mock_face_result.face_landmarks = None
        mock_face_landmarker = MagicMock()
        mock_face_landmarker.__enter__ = lambda s: s
        mock_face_landmarker.__exit__ = MagicMock(return_value=False)
        mock_face_landmarker.detect_for_video.return_value = mock_face_result

        # PoseLandmarker mock
        mock_pose_result = MagicMock()
        mock_pose_result.pose_landmarks = None
        mock_pose_landmarker = MagicMock()
        mock_pose_landmarker.__enter__ = lambda s: s
        mock_pose_landmarker.__exit__ = MagicMock(return_value=False)
        mock_pose_landmarker.detect_for_video.return_value = mock_pose_result

        with (
            patch("video_processor.cv2.VideoCapture", return_value=mock_cap),
            patch.object(
                video_processor._FaceLandmarker,
                "create_from_options",
                return_value=mock_face_landmarker,
            ),
            patch.object(
                video_processor._PoseLandmarker,
                "create_from_options",
                return_value=mock_pose_landmarker,
            ),
        ):
            result = video_processor.analyse(
                dummy_video,
                sample_every_n_frames=1,
                face_model_path=face_model,
                pose_model_path=pose_model,
            )

        expected_keys = {
            "total_frames_sampled",
            "face_detected_frames",
            "pose_detected_frames",
            "avg_attention_score",
            "gesture_frames",
            "frame_metadata",
        }
        assert expected_keys.issubset(result.keys())
