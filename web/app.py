"""RevFast — servidor web Flask para el evaluador de presentaciones CIL."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

# Cargar .env desde la raíz del proyecto
load_dotenv(Path(__file__).parent.parent / ".env")

# Agregar raíz del proyecto al path para importar los módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

import audio_processor  # noqa: E402
import evaluator as ev  # noqa: E402
import video_processor  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

_DEFAULT_RUBRIC_PATH = Path(__file__).parent.parent / "rubrics" / "elementary1_cil.json"

# Almacén de trabajos en memoria: {job_id: {status, step, message, progress, ...}}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Ejecutor del pipeline en background
# ---------------------------------------------------------------------------

def _update_job(job_id: str, **kwargs: object) -> None:
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


def _run_pipeline(
    job_id: str,
    video_path: str,
    whisper_model: str,
    sample_every: int,
    rubric: dict,
) -> None:
    try:
        _update_job(
            job_id,
            step=1,
            message=f"Transcribiendo audio con Whisper ({whisper_model})…",
            progress=10,
        )
        transcript_data = audio_processor.transcribe(video_path, model_name=whisper_model)
        transcript = transcript_data["text"]
        lang = transcript_data.get("language", "?")

        _update_job(
            job_id,
            step=2,
            message=(
                f"Audio transcrito ({len(transcript)} car., idioma: {lang}). "
                "Analizando video con MediaPipe…"
            ),
            progress=45,
        )
        mp_meta = video_processor.analyse(video_path, sample_every_n_frames=sample_every)

        _update_job(
            job_id,
            step=3,
            message="Solicitando evaluación al modelo Gemini…",
            progress=75,
        )
        evaluation = ev.evaluate(transcript, mp_meta, rubric=rubric)

        mp_summary = {
            k: mp_meta.get(k)
            for k in (
                "total_frames_sampled",
                "face_detected_frames",
                "pose_detected_frames",
                "avg_attention_score",
                "gesture_frames",
            )
        }

        with _jobs_lock:
            _jobs[job_id].update(
                status="done",
                step=4,
                message="Evaluación completada.",
                progress=100,
                result=evaluation,
                transcript=transcript,
                language=lang,
                mp_summary=mp_summary,
                rubric=rubric,
            )

    except Exception as exc:  # noqa: BLE001
        with _jobs_lock:
            _jobs[job_id].update(
                status="error",
                message=str(exc),
                progress=0,
            )
    finally:
        try:
            Path(video_path).unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/rubric", methods=["GET"])
def get_rubric():
    try:
        return jsonify(ev.load_rubric(None))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.route("/api/rubric", methods=["POST"])
def save_rubric():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Se requiere un objeto JSON en el cuerpo de la petición."}), 400
    try:
        ev.validate_rubric(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    _DEFAULT_RUBRIC_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return jsonify({"ok": True})


@app.route("/api/evaluate", methods=["POST"])
def start_evaluation():
    if "video" not in request.files:
        return jsonify({"error": "No se proporcionó archivo de video."}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "No se seleccionó ningún archivo."}), 400

    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        return jsonify({"error": f"Formato de video no soportado: {ext}. Usa mp4, avi, mov, mkv o webm."}), 400

    whisper_model = request.form.get("whisper_model", "base")
    if whisper_model not in {"tiny", "base", "small", "medium", "large"}:
        whisper_model = "base"

    try:
        sample_every = max(1, int(request.form.get("sample_every", 15)))
    except (TypeError, ValueError):
        sample_every = 15

    tema = (request.form.get("tema") or "libre").strip()

    # Rúbrica: usar JSON del formulario o cargar la predeterminada
    rubric_json = request.form.get("rubric")
    if rubric_json:
        try:
            rubric = json.loads(rubric_json)
            ev.validate_rubric(rubric)
        except (json.JSONDecodeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
    else:
        try:
            rubric = ev.load_rubric(None)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500

    # El tema del formulario tiene precedencia
    rubric["tema"] = tema

    job_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{job_id}{ext}"
    file.save(str(video_path))

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "step": 0,
            "message": "Iniciando análisis…",
            "progress": 2,
            "created_at": time.time(),
        }

    threading.Thread(
        target=_run_pipeline,
        args=(job_id, str(video_path), whisper_model, sample_every, rubric),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Trabajo no encontrado."}), 404
    return jsonify(job)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5005)
