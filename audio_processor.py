"""Audio processor module.

Transcribes a video/audio file to text using OpenAI Whisper.
"""

from __future__ import annotations

import os
from pathlib import Path

import whisper


def transcribe(video_path: str | os.PathLike, model_name: str = "base") -> dict:
    """Transcribe the audio track of *video_path* using Whisper.

    Parameters
    ----------
    video_path:
        Path to the video (or audio) file.  Whisper accepts most common
        container formats (mp4, avi, mov, mp3, wav …).
    model_name:
        Whisper model size.  One of ``tiny``, ``base``, ``small``,
        ``medium``, ``large``.  Defaults to ``base`` as a reasonable
        balance between speed and quality.

    Returns
    -------
    dict
        A mapping with at least the keys:

        * ``text`` – full transcript as a single string.
        * ``segments`` – list of segment dicts produced by Whisper
          (each segment includes ``start``, ``end``, ``text``).
        * ``language`` – detected language code.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    model = whisper.load_model(model_name)
    result = model.transcribe(str(path))

    return {
        "text": result.get("text", "").strip(),
        "segments": result.get("segments", []),
        "language": result.get("language", "unknown"),
    }


def get_transcript_text(video_path: str | os.PathLike, model_name: str = "base") -> str:
    """Convenience wrapper that returns only the transcript string."""
    return transcribe(video_path, model_name)["text"]
