"""Tests for evaluator.py."""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

import evaluator


_SAMPLE_TRANSCRIPT = "Esta es una presentación sobre el sistema solar."
_SAMPLE_METADATA = {
    "total_frames_sampled": 100,
    "face_detected_frames": 90,
    "pose_detected_frames": 88,
    "avg_attention_score": 0.85,
    "gesture_frames": {"left_wrist_raised": 5, "right_wrist_raised": 3},
}
_VALID_EVALUATION = {
    "puntuacion_global": 8,
    "claridad_verbal": 7,
    "coherencia_contenido": 9,
    "lenguaje_corporal": 8,
    "nivel_atencion": 9,
    "fortalezas": ["Buen contacto visual", "Vocabulario preciso"],
    "areas_mejora": ["Hablar más despacio"],
    "resumen": "El alumno demostró un buen dominio del tema.",
}


def _make_mock_genai(response_text: str):
    """Return a mock google.genai module with a stubbed Client."""
    mock_response = MagicMock()
    mock_response.text = response_text

    mock_models = MagicMock()
    mock_models.generate_content.return_value = mock_response

    mock_client = MagicMock()
    mock_client.models = mock_models

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client
    return mock_genai


class TestBuildUserMessage:
    def test_contains_transcript(self):
        msg = evaluator._build_user_message(_SAMPLE_TRANSCRIPT, _SAMPLE_METADATA)
        assert _SAMPLE_TRANSCRIPT in msg

    def test_contains_attention_score(self):
        msg = evaluator._build_user_message(_SAMPLE_TRANSCRIPT, _SAMPLE_METADATA)
        assert "avg_attention_score" in msg

    def test_contains_gesture_frames(self):
        msg = evaluator._build_user_message(_SAMPLE_TRANSCRIPT, _SAMPLE_METADATA)
        assert "gesture_frames" in msg


class TestEvaluate:
    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GOOGLE_API_KEY", None)
            with pytest.raises(EnvironmentError, match="GOOGLE_API_KEY"):
                evaluator.evaluate(_SAMPLE_TRANSCRIPT, _SAMPLE_METADATA)

    def test_returns_parsed_dict(self):
        mock_genai = _make_mock_genai(json.dumps(_VALID_EVALUATION))
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key"}),
            patch("evaluator.genai", mock_genai),
        ):
            result = evaluator.evaluate(_SAMPLE_TRANSCRIPT, _SAMPLE_METADATA)

        assert result["puntuacion_global"] == 8
        assert isinstance(result["fortalezas"], list)

    def test_strips_markdown_fences(self):
        fenced = f"```json\n{json.dumps(_VALID_EVALUATION)}\n```"
        mock_genai = _make_mock_genai(fenced)
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key"}),
            patch("evaluator.genai", mock_genai),
        ):
            result = evaluator.evaluate(_SAMPLE_TRANSCRIPT, _SAMPLE_METADATA)

        assert result["puntuacion_global"] == 8

    def test_invalid_json_raises_value_error(self):
        mock_genai = _make_mock_genai("This is not JSON at all.")
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key"}),
            patch("evaluator.genai", mock_genai),
        ):
            with pytest.raises(ValueError, match="non-JSON"):
                evaluator.evaluate(_SAMPLE_TRANSCRIPT, _SAMPLE_METADATA)
