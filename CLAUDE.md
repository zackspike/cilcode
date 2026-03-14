# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is **RevFast**, a cloud-based Decision Support System (DSS) being built for the **Centro Institucional de Lenguas (CIL)** at the Universidad Autónoma de Yucatán (UADY). It assists teachers in pre-evaluating student video presentations using multimodal AI. The AI analysis is a preliminary suggestion only — teachers retain final grading authority.

- **Team**: Isaac Herrera, Sofia Reyes, Karina Puch, Jesus Tec, Lexus Parra
- **Supervisor**: Luis Basto
- **Timeline**: March–May 2026

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py <video_path> [--whisper-model {tiny,base,small,medium,large}] [--sample-every N] [--rubric RUBRIC_PATH] [--save]

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_audio_processor.py

# Run a single test class or function
pytest tests/test_audio_processor.py::TestTranscribe::test_returns_expected_keys

# Lint
ruff check .
```

Set `GOOGLE_API_KEY` in a `.env` file (see `.env.example`) before running the pipeline.

## Architecture

This project grades student video presentations (1–5 min) through a 3-stage pipeline:

```
VIDEO FILE
  ├── [audio] → audio_processor.py  (Whisper)  → transcript {text, segments, language}
  └── [frames] → video_processor.py (MediaPipe) → frame metadata {attention, gestures}
                                    ↓
                              evaluator.py (Gemini 1.5-pro)
                              ← rubric (load_rubric / validate_rubric / _build_system_prompt)
                                    ↓
                         JSON grade {<criterios dinámicos>,
                                     puntuacion_total, aprobado,
                                     fortalezas, areas_mejora, resumen}
```

**`main.py`** — CLI entry point and orchestrator. Calls the three modules in sequence and optionally saves the result as `<video>_evaluation.json`.

**`audio_processor.py`** — Wraps Whisper. `transcribe()` returns a dict; `get_transcript_text()` is a convenience wrapper returning only the string.

**`video_processor.py`** — Wraps MediaPipe Face Mesh (478 landmarks) + Pose (33 landmarks). Processes every Nth frame (default 15), capped at 600 sampled frames. MediaPipe `.task` model files are auto-downloaded to `~/.cache/video_grader_ai/models/` on first use. Attention score is computed from nose-tip X offset from center; gesture detection checks wrist Y < shoulder Y.

**`evaluator.py`** — Sends transcript + summarized MediaPipe metadata to `gemini-1.5-pro`. Builds the system prompt dynamically from a rubric via `_build_system_prompt()`. Public functions: `validate_rubric()`, `load_rubric()`, `_build_system_prompt()`, `evaluate(rubric=None)`. Raises `EnvironmentError` when `GOOGLE_API_KEY` is missing and `ValueError` on non-JSON responses.

**`rubrics/elementary1_cil.json`** — Default rubric (CIL Elementary 1). Loaded automatically when no `--rubric` flag is passed.

## Error Contract

| Exception | Raised by | Cause |
|---|---|---|
| `FileNotFoundError` | `audio_processor`, `video_processor` | Video file not found |
| `RuntimeError` | `video_processor` | OpenCV cannot open the video |
| `EnvironmentError` | `evaluator` | `GOOGLE_API_KEY` not set |
| `ValueError` | `evaluator` | Gemini response is not valid JSON |

## Rubric JSON Schema

Rubric files live in `rubrics/`. Required fields:

| Field | Type | Description |
|---|---|---|
| `nivel` | string | Level name (e.g. `"Elementary 1"`) |
| `idioma` | string | Presentation language code (`"en"`, `"es"`) |
| `escala_min` | int | Minimum score per criterion |
| `escala_max` | int | Maximum score per criterion; must be > `escala_min` |
| `umbral_aprobatorio` | int | Minimum `puntuacion_total` to pass (exclusive: `> umbral`) |
| `descripcion_escala` | object | String keys for each integer in `[escala_min, escala_max]` |
| `criterios` | array | Non-empty list of `{clave, nombre, descripcion}`; `clave` must match `^[a-z][a-z0-9_]*$` and be unique |

## CIL Rubric (rubricaejemplo.pdf)

The evaluator maps to the official CIL Elementary 1 rubric. Each criterion is scored **1–5**; scores ≤ 3 are **not passing** (grey area). Total is 5–25 points; passing threshold is > 15.

| Key | Rubric Criterion |
|---|---|
| `vocabulario` | Control of basic vocabulary |
| `precision_gramatical` | Control of basic grammar structures |
| `pronunciacion_fluidez` | Pronunciation/intonation accuracy + speech fluency |
| `contenido_organizacion` | Content relevance, alignment with assignment, organization |
| `lenguaje_corporal_tiempo_preparacion` | Body language (eye contact, posture, gestures), timing, preparation (doesn't read notes) |

## Evaluation Output Schema

The output keys are dynamic — they match the `clave` values in the active rubric. For the default Elementary 1 rubric:

```json
{
  "vocabulario": 1-5,
  "precision_gramatical": 1-5,
  "pronunciacion_fluidez": 1-5,
  "contenido_organizacion": 1-5,
  "lenguaje_corporal_tiempo_preparacion": 1-5,
  "puntuacion_total": 5-25,
  "aprobado": true|false,
  "fortalezas": ["..."],
  "areas_mejora": ["..."],
  "resumen": "..."
}
```

## Key Files

| File | Purpose |
|---|---|
| `rubrics/elementary1_cil.json` | Default rubric bundled with the project |
| `MANUAL_USUARIO.md` | Full user manual in Spanish |
