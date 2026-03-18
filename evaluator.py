"""Evaluator module.

Sends the Whisper transcript and the MediaPipe metadata to the Gemini
1.5-pro model and returns a structured evaluation of the student's
presentation.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import google.genai as genai
from dotenv import load_dotenv

load_dotenv()

_MODEL_NAME = "gemini-2.5-flash"
_DEFAULT_RUBRIC_PATH = Path(__file__).parent / "rubrics" / "elementary1_cil.json"

_CLAVE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_rubric(rubric: dict) -> None:
    """Validate a rubric dictionary.

    Parameters
    ----------
    rubric:
        Dictionary loaded from a rubric JSON file.

    Raises
    ------
    ValueError
        If any required field is missing or any constraint is violated.
    """
    required_top = {
        "nivel", "idioma", "escala_min", "escala_max",
        "umbral_aprobatorio", "descripcion_escala", "criterios",
    }
    missing = required_top - rubric.keys()
    if missing:
        raise ValueError(f"Rúbrica inválida: faltan campos requeridos: {sorted(missing)}")

    escala_min = rubric["escala_min"]
    escala_max = rubric["escala_max"]
    if escala_min >= escala_max:
        raise ValueError(
            f"Rúbrica inválida: escala_min ({escala_min}) debe ser menor que escala_max ({escala_max})"
        )

    desc_escala = rubric["descripcion_escala"]
    for nivel in range(escala_min, escala_max + 1):
        if str(nivel) not in desc_escala:
            raise ValueError(
                f"Rúbrica inválida: descripcion_escala no cubre el nivel {nivel}"
            )

    criterios = rubric["criterios"]
    if not criterios:
        raise ValueError("Rúbrica inválida: criterios no puede estar vacío")

    claves_vistas: set[str] = set()
    for c in criterios:
        for campo in ("clave", "nombre", "descripcion"):
            if campo not in c:
                raise ValueError(f"Rúbrica inválida: criterio sin campo '{campo}': {c}")
        clave = c["clave"]
        if not _CLAVE_RE.match(clave):
            raise ValueError(
                f"Rúbrica inválida: clave '{clave}' no cumple el formato ^[a-z][a-z0-9_]*$"
            )
        if clave in claves_vistas:
            raise ValueError(f"Rúbrica inválida: clave duplicada '{clave}'")
        claves_vistas.add(clave)

    n_criterios = len(criterios)
    puntuacion_min_posible = escala_min * n_criterios
    puntuacion_max_posible = escala_max * n_criterios
    umbral = rubric["umbral_aprobatorio"]
    if not (puntuacion_min_posible <= umbral <= puntuacion_max_posible):
        raise ValueError(
            f"Rúbrica inválida: umbral_aprobatorio ({umbral}) fuera del rango posible "
            f"[{puntuacion_min_posible}, {puntuacion_max_posible}]"
        )


def load_rubric(path: str | Path | None = None) -> dict:
    """Load and validate a rubric from a JSON file.

    Parameters
    ----------
    path:
        Path to the rubric JSON file.  ``None`` loads the default
        Elementary 1 CIL rubric bundled with the package.

    Returns
    -------
    dict
        Validated rubric dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not valid JSON or fails schema validation.
    """
    rubric_path = Path(path) if path is not None else _DEFAULT_RUBRIC_PATH

    if not rubric_path.exists():
        raise FileNotFoundError(f"Archivo de rúbrica no encontrado: {rubric_path}")

    try:
        rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"El archivo de rúbrica no es JSON válido: {rubric_path}") from exc

    validate_rubric(rubric)
    return rubric


def _build_system_prompt(rubric: dict) -> str:
    """Build the Gemini system prompt from a validated rubric.

    Parameters
    ----------
    rubric:
        Validated rubric dictionary (from :func:`load_rubric`).

    Returns
    -------
    str
        System prompt string to pass to Gemini.
    """
    nivel = rubric["nivel"]
    idioma = rubric["idioma"]
    tema = rubric["tema"]
    escala_min = rubric["escala_min"]
    escala_max = rubric["escala_max"]
    umbral = rubric["umbral_aprobatorio"]
    desc_escala = rubric["descripcion_escala"]
    criterios = rubric["criterios"]
    n = len(criterios)

    # Tabla de escala (de mayor a menor)
    escala_lines = []
    for nivel_val in range(escala_max, escala_min - 1, -1):
        escala_lines.append(f"- {nivel_val}: {desc_escala[str(nivel_val)]}")
    escala_block = "\n".join(escala_lines)

    # Lista de criterios
    criterios_lines = []
    for i, c in enumerate(criterios, 1):
        criterios_lines.append(f"{i}. {c['clave']}: {c['descripcion']}")
    criterios_block = "\n".join(criterios_lines)

    # Claves para el JSON esperado
    claves_json_lines = []
    for c in criterios:
        claves_json_lines.append(f'  "{c["clave"]}": <entero {escala_min}-{escala_max}>,')
    claves_json_block = "\n".join(claves_json_lines)

    puntuacion_min = escala_min * n
    puntuacion_max = escala_max * n
    not_passing_threshold = escala_min + 1

    return (
        f"Eres un evaluador experto del Centro Institucional de Lenguas (CIL) de la UADY.\n"
        f"Tu tarea es analizar la presentación de un alumno de nivel {nivel} "
        f"considerando que el tema de la presentación es {tema} "
        f"(idioma de presentación: {idioma}) a partir de:\n"
        f"1. La transcripción de audio de la presentación.\n"
        f"2. Metadatos de análisis corporal (atención, gestos) obtenidos con MediaPipe.\n\n"
        f"Evalúa con base en la rúbrica oficial del CIL. Cada criterio se puntúa del {escala_min} al {escala_max}:\n"
        f"{escala_block}\n\n"
        f"NOTA: puntajes ≤ {not_passing_threshold} NO son aprobatorios.\n\n"
        f"Criterios de la rúbrica:\n"
        f"{criterios_block}\n\n"
        f"Devuelve **únicamente** un objeto JSON válido con la siguiente estructura\n"
        f"(sin texto adicional fuera del JSON):\n\n"
        f"{{\n"
        f"{claves_json_block}\n"
        f'  "puntuacion_total": <entero {puntuacion_min}-{puntuacion_max}>,\n'
        f'  "aprobado": <booleano, true si puntuacion_total > {umbral}>,\n'
        f'  "fortalezas": [<string>, ...],\n'
        f'  "areas_mejora": [<string>, ...],\n'
        f'  "resumen": "<string de 2-4 oraciones>"\n'
        f"}}"
    )


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


def evaluate(
    transcript: str,
    mediapipe_metadata: dict,
    rubric: dict | None = None,
) -> dict:
    """Evaluate a student presentation using Gemini 1.5-pro.

    Parameters
    ----------
    transcript:
        Full text of the presentation produced by Whisper.
    mediapipe_metadata:
        Dictionary returned by :func:`video_processor.analyse`.
    rubric:
        Validated rubric dictionary.  ``None`` loads the default
        Elementary 1 CIL rubric, preserving backwards compatibility.

    Returns
    -------
    dict
        Structured evaluation with scores and feedback.

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

    if rubric is None:
        rubric = load_rubric(None)

    client = genai.Client(api_key=api_key)

    user_message = _build_user_message(transcript, mediapipe_metadata)
    system_prompt = _build_system_prompt(rubric)
    full_prompt = f"{system_prompt}\n\n{user_message}"

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
