"""Tests for main.py orchestration."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import main


_TRANSCRIPT_DATA = {
    "text": "Buenas tardes, voy a presentar mi proyecto.",
    "segments": [],
    "language": "es",
}
_MP_METADATA = {
    "total_frames_sampled": 50,
    "face_detected_frames": 45,
    "pose_detected_frames": 44,
    "avg_attention_score": 0.9,
    "gesture_frames": {"left_wrist_raised": 2, "right_wrist_raised": 1},
    "frame_metadata": [],
}
_EVALUATION = {
    "puntuacion_global": 9,
    "claridad_verbal": 9,
    "coherencia_contenido": 9,
    "lenguaje_corporal": 8,
    "nivel_atencion": 9,
    "fortalezas": ["Excelente claridad"],
    "areas_mejora": [],
    "resumen": "Presentación sobresaliente.",
}


class TestRun:
    def test_run_calls_all_stages(self, tmp_path):
        dummy_video = tmp_path / "pres.mp4"
        dummy_video.write_bytes(b"\x00")

        with (
            patch("main.audio_processor.transcribe", return_value=_TRANSCRIPT_DATA) as mock_transcribe,
            patch("main.video_processor.analyse", return_value=_MP_METADATA) as mock_analyse,
            patch("main.evaluator.evaluate", return_value=_EVALUATION) as mock_evaluate,
        ):
            result = main.run(str(dummy_video))

        mock_transcribe.assert_called_once()
        mock_analyse.assert_called_once()
        mock_evaluate.assert_called_once()
        assert result["puntuacion_global"] == 9

    def test_run_saves_json_when_flag_set(self, tmp_path):
        dummy_video = tmp_path / "pres.mp4"
        dummy_video.write_bytes(b"\x00")

        with (
            patch("main.audio_processor.transcribe", return_value=_TRANSCRIPT_DATA),
            patch("main.video_processor.analyse", return_value=_MP_METADATA),
            patch("main.evaluator.evaluate", return_value=_EVALUATION),
        ):
            main.run(str(dummy_video), save=True)

        out_path = tmp_path / "pres_evaluation.json"
        assert out_path.exists()
        saved = json.loads(out_path.read_text())
        assert saved["puntuacion_global"] == 9

    def test_run_propagates_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main.run(str(tmp_path / "nonexistent.mp4"))


class TestParseArgs:
    def test_defaults(self):
        args = main._parse_args(["video.mp4"])
        assert args.video == "video.mp4"
        assert args.whisper_model == "base"
        assert args.sample_every == 15
        assert args.save is False

    def test_custom_args(self):
        args = main._parse_args(["v.mp4", "--whisper-model", "small", "--sample-every", "30", "--save"])
        assert args.whisper_model == "small"
        assert args.sample_every == 30
        assert args.save is True
