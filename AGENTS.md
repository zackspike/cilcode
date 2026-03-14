# Video Grader AI - AGENTS.md

## Project Context
Aplicación Python para la evaluación automatizada de presentaciones de alumnos (1-5 min) usando Whisper (Audio), MediaPipe (Gestos) y Gemini API (Análisis Semántico).

## Tech Stack & Commands
- **Core:** Python 3.11+, MediaPipe, OpenAI-Whisper, Google-GenerativeAI.
- **Setup:** `pip install -r requirements.txt`
- **Lint:** `ruff check .`
- **Test:** `pytest tests/`

## Critical Rules
- **No API Keys in Code:** Usar siempre `.env` para `GOOGLE_API_KEY`.
- **Modularidad:** Separar `audio_processor.py`, `video_processor.py` y `evaluator.py`.
- **MediaPipe:** Usar el modelo `Face Mesh` y `Pose` simultáneamente para triangulación de atención.
- **Gemini Usage:** Utilizar `gemini-1.5-pro` para la inferencia final enviando la transcripción y los metadatos de MediaPipe.