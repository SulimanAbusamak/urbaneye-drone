"""
speech_recognition_hf.py

Hugging Face speech-to-text layer for UrbanEye.

Normal website flow:
- The dashboard records audio in the browser.
- The backend saves that exact upload as voice_command.webm.
- The backend calls transcribe_audio_file(...).
- The returned text is sent into the SAME command path as the website text box.

Important:
- This file does NOT control the drone.
- It only converts audio -> text.
- The nav code still handles text parsing, OSM resolving, flight, hover-control, and landing.
- Do not run watch mode during the website demo unless you intentionally want a standalone file watcher.

Required environment variable:
- HF_TOKEN

Optional environment variables:
- HF_STT_PROVIDER: default "hf-inference"
- HF_STT_MODEL: default "openai/whisper-large-v3-turbo"
- URBANEYE_VOICE_AUDIO_FILE: default "voice_command.webm"
- URBANEYE_VOICE_QUEUE_FILE: default "voice_commands.jsonl" (standalone/debug only)
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from huggingface_hub import InferenceClient


# ---------------------------------------------------------------------------
# Hugging Face hosted speech-to-text config
# ---------------------------------------------------------------------------
HF_STT_PROVIDER = os.getenv("HF_STT_PROVIDER", "hf-inference")
HF_STT_MODEL = os.getenv("HF_STT_MODEL", "openai/whisper-large-v3-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")

# Website/backend audio file
VOICE_AUDIO_FILE = os.getenv("URBANEYE_VOICE_AUDIO_FILE", "voice_command.webm")

# Optional standalone/debug queue file. The website backend does NOT need this.
VOICE_COMMAND_QUEUE_FILE = os.getenv("URBANEYE_VOICE_QUEUE_FILE", "voice_commands.jsonl")

WATCH_POLL_SEC = 0.5
FILE_STABLE_WAIT_SEC = 0.35


def _get_client() -> InferenceClient:
    """Create the HF client only when transcription is actually requested."""
    token = os.getenv("HF_TOKEN") or HF_TOKEN
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Set it before using voice recognition.")
    return InferenceClient(provider=HF_STT_PROVIDER, api_key=token, timeout=90)


def _extract_text(output: Any) -> str:
    """Handle the common Hugging Face output shapes and return clean text."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, dict):
        return str(output.get("text", "")).strip()
    if hasattr(output, "text"):
        return str(output.text).strip()
    return str(output).strip()


def transcribe_audio_file(audio_file: str | os.PathLike[str] = VOICE_AUDIO_FILE) -> str:
    """Send one audio file to Hugging Face Whisper and return recognized text."""
    audio_path = Path(audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.stat().st_size <= 0:
        raise RuntimeError(f"Audio file is empty: {audio_path}")

    client = _get_client()
    output = client.automatic_speech_recognition(str(audio_path), model=HF_STT_MODEL)
    return _extract_text(output)


def transcribe_audio_bytes(audio_bytes: bytes, suffix: str = ".webm") -> str:
    """Transcribe raw bytes. Useful for API uploads and tests."""
    if not audio_bytes:
        raise RuntimeError("Audio data is empty.")

    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        return transcribe_audio_file(tmp_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def send_text_to_nav_queue(text: str, queue_file: str = VOICE_COMMAND_QUEUE_FILE) -> bool:
    """Standalone/debug helper: append recognized text to a JSONL queue file.

    The website backend does not use this queue. It sends the transcript directly
    into controller.submit_command(), so old voice_command.webm files cannot be
    accidentally reused when the user later sends a normal text command.
    """
    text = (text or "").strip()
    if not text:
        return False

    item = {
        "source": "website_voice",
        "text": text,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(queue_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return True


# Backward-compatible alias in case any old script still imports this name.
send_text_to_nav = send_text_to_nav_queue


def process_audio_once(audio_file: str = VOICE_AUDIO_FILE, send_to_queue: bool = False) -> str:
    """Transcribe one audio file. Optionally write it to the debug JSONL queue."""
    text = transcribe_audio_file(audio_file)
    print(f"[VOICE] Heard: {text}")
    if send_to_queue:
        if send_text_to_nav_queue(text):
            print(f"[VOICE] Wrote to debug queue: {text}")
        else:
            print("[VOICE] Empty transcription. Nothing written to queue.")
    return text


def _file_signature(path: Path):
    try:
        stat = path.stat()
        return (stat.st_mtime, stat.st_size)
    except FileNotFoundError:
        return None


def watch_audio_file(audio_file: str = VOICE_AUDIO_FILE, send_to_queue: bool = True):
    """Standalone watcher for debugging only.

    Website flow should use the backend endpoint instead of watch mode. The
    backend transcribes only the newly uploaded blob and then immediately submits
    the resulting text command.
    """
    audio_path = Path(audio_file)
    last_done_signature = _file_signature(audio_path)  # avoids reusing an old file on startup

    print("=" * 70)
    print("UrbanEye Hugging Face Speech Recognition - standalone watcher")
    print("=" * 70)
    print(f"Watching audio file: {audio_path.resolve()}")
    print(f"Queue file: {Path(VOICE_COMMAND_QUEUE_FILE).resolve()}")
    print(f"Model: {HF_STT_MODEL}")
    print("Existing audio at startup is ignored; only future file updates are transcribed.")
    print("Press Ctrl+C to stop.")
    print("=" * 70)

    while True:
        signature = _file_signature(audio_path)
        if signature is not None and signature != last_done_signature:
            # Wait briefly so the browser/backend finishes writing the file.
            time.sleep(FILE_STABLE_WAIT_SEC)
            stable_signature = _file_signature(audio_path)
            if stable_signature == signature:
                try:
                    print("\n[VOICE] New website audio detected. Transcribing...")
                    process_audio_once(str(audio_path), send_to_queue=send_to_queue)
                    last_done_signature = stable_signature
                except Exception as e:
                    print(f"[VOICE ERROR] {e}")
                    last_done_signature = stable_signature

        time.sleep(WATCH_POLL_SEC)


def main():
    parser = argparse.ArgumentParser(description="UrbanEye Hugging Face speech-to-text")
    parser.add_argument("--file", default=VOICE_AUDIO_FILE, help="Audio file path, default voice_command.webm")
    parser.add_argument("--once", action="store_true", help="Transcribe once instead of watching the file")
    parser.add_argument("--send-to-queue", action="store_true", help="Write transcript to voice_commands.jsonl for standalone/debug use")
    args = parser.parse_args()

    if args.once:
        process_audio_once(args.file, send_to_queue=args.send_to_queue)
    else:
        watch_audio_file(args.file, send_to_queue=args.send_to_queue)


if __name__ == "__main__":
    main()
