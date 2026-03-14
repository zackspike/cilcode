"""Tests for audio_processor.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import audio_processor


class TestTranscribe:
    def _make_whisper_result(self):
        return {
            "text": " Hello, this is a test.",
            "segments": [
                {"start": 0.0, "end": 2.0, "text": " Hello, this is a test."}
            ],
            "language": "en",
        }

    def test_returns_expected_keys(self, tmp_path):
        dummy_video = tmp_path / "video.mp4"
        dummy_video.write_bytes(b"\x00")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = self._make_whisper_result()

        with patch("audio_processor.whisper.load_model", return_value=mock_model):
            result = audio_processor.transcribe(dummy_video)

        assert set(result.keys()) == {"text", "segments", "language"}

    def test_text_is_stripped(self, tmp_path):
        dummy_video = tmp_path / "video.mp4"
        dummy_video.write_bytes(b"\x00")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = self._make_whisper_result()

        with patch("audio_processor.whisper.load_model", return_value=mock_model):
            result = audio_processor.transcribe(dummy_video)

        assert not result["text"].startswith(" ")
        assert result["text"] == "Hello, this is a test."

    def test_language_returned(self, tmp_path):
        dummy_video = tmp_path / "video.mp4"
        dummy_video.write_bytes(b"\x00")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = self._make_whisper_result()

        with patch("audio_processor.whisper.load_model", return_value=mock_model):
            result = audio_processor.transcribe(dummy_video)

        assert result["language"] == "en"

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            audio_processor.transcribe("/nonexistent/path/video.mp4")

    def test_get_transcript_text_returns_string(self, tmp_path):
        dummy_video = tmp_path / "video.mp4"
        dummy_video.write_bytes(b"\x00")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = self._make_whisper_result()

        with patch("audio_processor.whisper.load_model", return_value=mock_model):
            text = audio_processor.get_transcript_text(dummy_video)

        assert isinstance(text, str)
        assert text == "Hello, this is a test."
