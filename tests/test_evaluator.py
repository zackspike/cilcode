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
    "vocabulario": 4,
    "precision_gramatical": 3,
    "pronunciacion_fluidez": 4,
    "contenido_organizacion": 5,
    "lenguaje_corporal_tiempo_preparacion": 4,
    "puntuacion_total": 20,
    "aprobado": True,
    "fortalezas": ["Buen contacto visual", "Vocabulario preciso"],
    "areas_mejora": ["Hablar más despacio"],
    "resumen": "El alumno demostró un buen dominio del tema.",
}

_MINIMAL_RUBRIC = {
    "nivel": "Test Level",
    "idioma": "en",
    "escala_min": 1,
    "escala_max": 5,
    "umbral_aprobatorio": 6,
    "descripcion_escala": {
        "1": "Weak",
        "2": "Fair",
        "3": "Moderate",
        "4": "Good",
        "5": "Excellent",
    },
    "criterios": [
        {"clave": "grammar", "nombre": "Grammar", "descripcion": "Grammar control."},
        {"clave": "fluency", "nombre": "Fluency", "descripcion": "Speech fluency."},
    ],
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


class TestValidateRubric:
    def test_valid_rubric_passes(self):
        evaluator.validate_rubric(_MINIMAL_RUBRIC)  # should not raise

    def test_missing_key_raises(self):
        rubric = {k: v for k, v in _MINIMAL_RUBRIC.items() if k != "nivel"}
        with pytest.raises(ValueError, match="nivel"):
            evaluator.validate_rubric(rubric)

    def test_empty_criterios_raises(self):
        rubric = {**_MINIMAL_RUBRIC, "criterios": []}
        with pytest.raises(ValueError, match="criterios"):
            evaluator.validate_rubric(rubric)

    def test_duplicate_clave_raises(self):
        rubric = {
            **_MINIMAL_RUBRIC,
            "criterios": [
                {"clave": "grammar", "nombre": "G", "descripcion": "d"},
                {"clave": "grammar", "nombre": "G2", "descripcion": "d2"},
            ],
        }
        with pytest.raises(ValueError, match="duplicada"):
            evaluator.validate_rubric(rubric)

    def test_clave_with_space_raises(self):
        rubric = {
            **_MINIMAL_RUBRIC,
            "criterios": [
                {"clave": "bad clave", "nombre": "B", "descripcion": "d"},
            ],
        }
        with pytest.raises(ValueError, match="formato"):
            evaluator.validate_rubric(rubric)

    def test_umbral_out_of_range_raises(self):
        rubric = {**_MINIMAL_RUBRIC, "umbral_aprobatorio": 100}
        with pytest.raises(ValueError, match="umbral_aprobatorio"):
            evaluator.validate_rubric(rubric)

    def test_min_ge_max_raises(self):
        rubric = {**_MINIMAL_RUBRIC, "escala_min": 5, "escala_max": 5}
        with pytest.raises(ValueError, match="escala_min"):
            evaluator.validate_rubric(rubric)

    def test_incomplete_descripcion_escala_raises(self):
        rubric = {
            **_MINIMAL_RUBRIC,
            "descripcion_escala": {"1": "Weak", "2": "Fair"},  # missing 3,4,5
        }
        with pytest.raises(ValueError, match="descripcion_escala"):
            evaluator.validate_rubric(rubric)


class TestLoadRubric:
    def test_loads_default(self):
        rubric = evaluator.load_rubric(None)
        assert rubric["nivel"] == "Elementary 1"
        assert len(rubric["criterios"]) == 5

    def test_loads_from_path(self, tmp_path):
        rubric_file = tmp_path / "rubric.json"
        rubric_file.write_text(json.dumps(_MINIMAL_RUBRIC), encoding="utf-8")
        rubric = evaluator.load_rubric(rubric_file)
        assert rubric["nivel"] == "Test Level"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            evaluator.load_rubric(tmp_path / "nonexistent.json")

    def test_invalid_json_raises_value_error(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json at all", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON válido"):
            evaluator.load_rubric(bad_file)

    def test_invalid_schema_raises_value_error(self, tmp_path):
        bad_rubric = {"nivel": "X"}  # missing required fields
        bad_file = tmp_path / "bad_schema.json"
        bad_file.write_text(json.dumps(bad_rubric), encoding="utf-8")
        with pytest.raises(ValueError, match="faltan campos"):
            evaluator.load_rubric(bad_file)


class TestBuildSystemPrompt:
    def test_contains_all_claves(self):
        prompt = evaluator._build_system_prompt(_MINIMAL_RUBRIC)
        for c in _MINIMAL_RUBRIC["criterios"]:
            assert c["clave"] in prompt

    def test_contains_nivel(self):
        prompt = evaluator._build_system_prompt(_MINIMAL_RUBRIC)
        assert _MINIMAL_RUBRIC["nivel"] in prompt

    def test_contains_umbral_aprobatorio(self):
        prompt = evaluator._build_system_prompt(_MINIMAL_RUBRIC)
        assert str(_MINIMAL_RUBRIC["umbral_aprobatorio"]) in prompt

    def test_contains_descripcion_escala(self):
        prompt = evaluator._build_system_prompt(_MINIMAL_RUBRIC)
        for desc in _MINIMAL_RUBRIC["descripcion_escala"].values():
            assert desc in prompt

    def test_returns_string(self):
        prompt = evaluator._build_system_prompt(_MINIMAL_RUBRIC)
        assert isinstance(prompt, str)


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

        assert result["puntuacion_total"] == 20
        assert isinstance(result["fortalezas"], list)

    def test_strips_markdown_fences(self):
        fenced = f"```json\n{json.dumps(_VALID_EVALUATION)}\n```"
        mock_genai = _make_mock_genai(fenced)
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key"}),
            patch("evaluator.genai", mock_genai),
        ):
            result = evaluator.evaluate(_SAMPLE_TRANSCRIPT, _SAMPLE_METADATA)

        assert result["puntuacion_total"] == 20

    def test_invalid_json_raises_value_error(self):
        mock_genai = _make_mock_genai("This is not JSON at all.")
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key"}),
            patch("evaluator.genai", mock_genai),
        ):
            with pytest.raises(ValueError, match="non-JSON"):
                evaluator.evaluate(_SAMPLE_TRANSCRIPT, _SAMPLE_METADATA)

    def test_evaluate_with_custom_rubric(self):
        custom_eval = {"grammar": 4, "fluency": 3, "puntuacion_total": 7,
                       "aprobado": True, "fortalezas": [], "areas_mejora": [], "resumen": "ok"}
        mock_genai = _make_mock_genai(json.dumps(custom_eval))
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key"}),
            patch("evaluator.genai", mock_genai),
        ):
            result = evaluator.evaluate(_SAMPLE_TRANSCRIPT, _SAMPLE_METADATA, rubric=_MINIMAL_RUBRIC)

        assert result["grammar"] == 4

    def test_evaluate_without_rubric_calls_load_rubric(self):
        mock_genai = _make_mock_genai(json.dumps(_VALID_EVALUATION))
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key"}),
            patch("evaluator.genai", mock_genai),
            patch("evaluator.load_rubric", return_value=_MINIMAL_RUBRIC) as mock_load,
        ):
            evaluator.evaluate(_SAMPLE_TRANSCRIPT, _SAMPLE_METADATA)

        mock_load.assert_called_once_with(None)
