"""Main entry point for the Video Grader AI application.

Usage
-----
    python main.py <path_to_video>

The script orchestrates the three processing stages:

1. **Audio** – Whisper transcribes the speech in the video.
2. **Video** – MediaPipe extracts Face Mesh + Pose attention metadata.
3. **Evaluation** – Gemini 1.5-pro produces a structured grade with feedback.

The final evaluation is printed as formatted JSON to stdout and optionally
saved to a ``<video_name>_evaluation.json`` file next to the source video.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import audio_processor
import evaluator
import video_processor


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a student video presentation automatically."
    )
    parser.add_argument(
        "video",
        metavar="VIDEO_PATH",
        help="Path to the video file to evaluate (mp4, avi, mov …).",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base).",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=15,
        metavar="N",
        help="Sample MediaPipe every N frames (default: 15).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the evaluation JSON next to the video file.",
    )
    return parser.parse_args(argv)


def run(
    video_path: str,
    whisper_model: str = "base",
    sample_every: int = 15,
    save: bool = False,
) -> dict:
    """Run the full evaluation pipeline.

    Parameters
    ----------
    video_path:
        Path to the video file.
    whisper_model:
        Whisper model size string.
    sample_every:
        Frame sampling interval for MediaPipe.
    save:
        If *True*, write the evaluation JSON next to the source video.

    Returns
    -------
    dict
        Evaluation result from Gemini.
    """
    path = Path(video_path)

    print(f"[1/3] Transcribing audio with Whisper ({whisper_model}) …")
    transcript_data = audio_processor.transcribe(path, model_name=whisper_model)
    transcript = transcript_data["text"]
    print(f"      Language detected: {transcript_data['language']}")
    print(f"      Transcript ({len(transcript)} chars): {transcript[:120]}{'...' if len(transcript) > 120 else ''}")

    print(f"\n[2/3] Analysing video with MediaPipe (every {sample_every} frames) …")
    mp_metadata = video_processor.analyse(path, sample_every_n_frames=sample_every)
    print(f"      Frames sampled: {mp_metadata['total_frames_sampled']}")
    print(f"      Face detected: {mp_metadata['face_detected_frames']} frames")
    print(f"      Pose detected: {mp_metadata['pose_detected_frames']} frames")
    print(f"      Avg attention score: {mp_metadata['avg_attention_score']:.2f}")

    print("\n[3/3] Requesting evaluation from Gemini 1.5-pro …")
    evaluation = evaluator.evaluate(transcript, mp_metadata)

    print("\n=== Evaluation Result ===")
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))

    if save:
        out_path = path.with_name(path.stem + "_evaluation.json")
        out_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nEvaluation saved to: {out_path}")

    return evaluation


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        run(
            args.video,
            whisper_model=args.whisper_model,
            sample_every=args.sample_every,
            save=args.save,
        )
    except (FileNotFoundError, EnvironmentError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
