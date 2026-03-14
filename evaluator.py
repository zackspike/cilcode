"""Evaluator module.

Sends the Whisper transcript and the MediaPipe metadata to the Gemini
1.5-pro model and returns a structured evaluation of the student's
presentation.
"""

from __future__ import annotations

import json
import os

import google.genai as genai
from dotenv import load_dotenv

load_dotenv()

_MODEL_NAME = "gemini-1.5-pro"

_SYSTEM_PROMPT = """\
Eres un evaluador experto en presentaciones académicas.
Tu tarea es analizar la presentación de un alumno a partir de:
1. La transcripción de audio de la presentación.
2. Metadatos de análisis corporal (atención, gestos) obtenidos con MediaPipe.

Devuelve **únicamente** un objeto JSON válido con la siguiente estructura
(sin texto adicional fuera del JSON):

{
  "puntuacion_global": <entero 0-10>,
  "claridad_verbal": <entero 0-10>,
  "coherencia_contenido": <entero 0-10>,
  "lenguaje_corporal": <entero 0-10>,
  "nivel_atencion": <entero 0-10>,
  "fortalezas": [<string>, ...],
  "areas_mejora": [<string>, ...],
  "resumen": "<string de 2-4 oraciones>"
}
"""


def _build_user_message(transcript: str, mediapipe_metadata: dict) -> str:
    """Compose the user-turn message sent to Gemini."""
    meta_summary = {
        "total_frames_sampled": mediapipe_metadata.get("total_frames_sampled"),
        "face_detected_frames": mediapipe_metadata.get("face_detected_frames"),
        "pose_detected_frames": mediapipe_metadata.get("pose_detected_frames"),
        "avg_attention_score": mediapipe_metadata.get("avg_attention_score"),
        "gesture_frames": mediapipe_metadata.get("gesture_frames"),
    }

    return (
        "## Transcripción de la presentación\n\n"
        f"{transcript}\n\n"
        "## Metadatos de MediaPipe\n\n"
        f"```json\n{json.dumps(meta_summary, ensure_ascii=False, indent=2)}\n```"
    )


def evaluate(transcript: str, mediapipe_metadata: dict) -> dict:
    """Evaluate a student presentation using Gemini 1.5-pro.

    Parameters
    ----------
    transcript:
        Full text of the presentation produced by Whisper.
    mediapipe_metadata:
        Dictionary returned by :func:`video_processor.analyse`.

    Returns
    -------
    dict
        Structured evaluation with scores and feedback.  Keys match the
        JSON schema defined in :data:`_SYSTEM_PROMPT`.

    Raises
    ------
    EnvironmentError
        If ``GOOGLE_API_KEY`` is not set in the environment / `.env` file.
    ValueError
        If Gemini returns a response that cannot be parsed as valid JSON.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set.  "
            "Copy .env.example to .env and add your key."
        )

    client = genai.Client(api_key=api_key)

    user_message = _build_user_message(transcript, mediapipe_metadata)
    full_prompt = f"{_SYSTEM_PROMPT}\n\n{user_message}"

    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=full_prompt,
    )
    raw_text = response.text.strip()

    # Strip optional markdown code fences (e.g. ```json ... ``` or ``` ... ```).
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        # Remove opening fence line; remove closing fence if it is exactly ```
        inner_lines = lines[1:]
        if inner_lines and inner_lines[-1].strip() == "```":
            inner_lines = inner_lines[:-1]
        raw_text = "\n".join(inner_lines).strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini returned non-JSON response:\n{response.text}"
        ) from exc

    return result
