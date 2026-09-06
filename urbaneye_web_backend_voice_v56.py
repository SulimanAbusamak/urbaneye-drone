"""
UrbanEye web backend bridge.

This keeps your existing flight files as the source of truth and adds a local
HTTP API so the dashboard text box can control the drone.

Run from the same folder as:
  drone_nav_final.py
  llm_resolver.py
  object_avoidance_final.py
  unreal_camera_reader.py

Install once:
  pip install fastapi uvicorn pydantic pymavlink airsim opencv-python numpy requests huggingface_hub

Start the backend:
  python -m uvicorn urbaneye_web_backend_voice_v56:app --host 127.0.0.1 --port 8000

Then open the integrated HTML dashboard in your browser.
"""

from __future__ import annotations

import asyncio
import base64
import io
import hashlib
import json
import os
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import requests

# IMPORTANT: all real flight behavior is reused from your current stable file.
# This backend should not rewrite the drone logic; it only replaces terminal input
# with API commands coming from the website.
import drone_nav_final as nav
import vl_resolver

VOICE_COMMAND_FILE = Path(__file__).resolve().parent / "voice_command.wav"
BACKEND_FIX_VERSION = "v56_demo_clean_speed_place_feedback"
SAVED_LOCATIONS_FILE = getattr(nav, "SAVED_LOCATIONS_FILE", "saved_locations.json")

# ---------------------------------------------------------------------------
# Computer Vision API bridge
# ---------------------------------------------------------------------------
# The CV Flask API stays separate in D:\scripts\UrbanEye_API and runs on port 5000.
# The website talks to this FastAPI backend on port 8000, then this backend forwards
# the frame/image to the CV API. This keeps navigation and CV separate.
CV_API_BASE = os.getenv("URBANEYE_CV_API_BASE", "http://127.0.0.1:5000").rstrip("/")
CV_API_TIMEOUT_SEC = float(os.getenv("URBANEYE_CV_API_TIMEOUT_SEC", "120"))

# ---------------------------------------------------------------------------
# v30 Hugging Face Vision-Language detector config
# ---------------------------------------------------------------------------
# Main detector: Qwen2.5-VL through Hugging Face Router OpenAI-compatible API.
# YOLO/EasyOCR backup is disabled for the final demo by default.
VL_ENABLE = os.getenv("URBANEYE_VL_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off"}
VL_MODEL = getattr(vl_resolver, "VL_MODEL", os.getenv("URBANEYE_VL_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud"))
VL_BASE_URL = os.getenv("URBANEYE_VL_BASE_URL", "https://router.huggingface.co/v1")
VL_TIMEOUT_SEC = float(os.getenv("URBANEYE_VL_TIMEOUT_SEC", "120"))
VL_PLATE_MIN_CONFIDENCE = float(os.getenv("URBANEYE_VL_PLATE_MIN_CONFIDENCE", "70"))
VL_MAX_IMAGE_WIDTH = int(os.getenv("URBANEYE_VL_MAX_IMAGE_WIDTH", "1600"))
VL_JPEG_QUALITY = int(os.getenv("URBANEYE_VL_JPEG_QUALITY", "85"))
# smart = call YOLO if VL fails, says no violation, has unreadable plate, or lacks boxes.
# always = always call YOLO and merge plate/box backup where useful.
# never = VL only.
YOLO_BACKUP_MODE = os.getenv("URBANEYE_YOLO_BACKUP_MODE", "never").strip().lower()

KNOWN_PLATES = [
    p.strip() for p in os.getenv(
        "URBANEYE_KNOWN_PLATES",
        "11-11111,13-57911,22-22222,18-69854,12-34567,12-12121,23-23244,34-35362,"
        "67-45678,98-76543,54-23456,78-45372,45-38923,67-67676,74-74378,12-64296,"
        "44-87523,29-38472,33-27151,22-38452,12-18351"
    ).split(",") if p.strip()
]


# ---------------------------------------------------------------------------
# v20 direct Hugging Face ASR helper
# ---------------------------------------------------------------------------
HF_STT_MODEL = os.getenv("HF_STT_MODEL", "openai/whisper-large-v3")
HF_STT_API_URL = os.getenv("HF_STT_API_URL", "").strip()
HF_STT_TIMEOUT_SEC = float(os.getenv("HF_STT_TIMEOUT_SEC", "90"))


def _get_hf_token() -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set.")
    return token


def _hf_asr_urls():
    if HF_STT_API_URL:
        yield HF_STT_API_URL
        return
    model_path = quote(HF_STT_MODEL.strip(), safe="/")
    yield f"https://router.huggingface.co/hf-inference/models/{model_path}"
    yield f"https://api-inference.huggingface.co/models/{model_path}"


def _extract_transcript(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("text", "")).strip()
    if hasattr(value, "text"):
        return str(value.text).strip()
    return str(value).strip()


def _direct_hf_transcribe_audio(audio_bytes: bytes, content_type: str) -> str:
    """Post audio to HF ourselves so no library/path can send video/webm."""
    if not audio_bytes:
        raise RuntimeError("Audio data is empty.")

    headers = {
        "Authorization": f"Bearer {_get_hf_token()}",
        "Content-Type": content_type,
        "Accept": "application/json",
    }

    last_error = None
    for url in _hf_asr_urls():
        try:
            response = requests.post(url, headers=headers, data=audio_bytes, timeout=HF_STT_TIMEOUT_SEC)
        except Exception as e:
            last_error = e
            continue

        if 200 <= response.status_code < 300:
            try:
                parsed = response.json()
            except Exception:
                parsed = response.text
            return _extract_transcript(parsed)

        body = response.text.strip()
        last_error = RuntimeError(f"HF ASR request failed ({response.status_code}): {body}")
        if response.status_code not in {404, 405, 410}:
            break

    raise last_error or RuntimeError("HF ASR request failed.")


def _parse_terminal_command_safe(raw: str) -> Dict[str, Any]:
    """Backend-safe wrapper around the current nav parser.

    Some drone_nav_final versions expose parse_terminal_command only, while
    newer ones may expose parse_terminal_command_safe. This wrapper supports both
    without changing the flight code.
    """
    try:
        parser = getattr(nav, "parse_terminal_command_safe", None) or getattr(nav, "parse_terminal_command")
        return parser(raw)
    except Exception as e:
        return {"type": "unknown", "error": str(e), "parser": "backend_safe"}


def _resolve_place_with_osm_safe(place_text: str, command: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Backend-safe wrapper around the current OSM resolver."""
    try:
        resolver = getattr(nav, "resolve_place_with_osm_safe", None) or getattr(nav, "resolve_place_with_osm")
        return resolver(place_text, command)
    except TypeError:
        # Older resolver signature may accept only place_text.
        try:
            resolver = getattr(nav, "resolve_place_with_osm_safe", None) or getattr(nav, "resolve_place_with_osm")
            return resolver(place_text)
        except Exception as e:
            return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _resolve_saved_location_safe(place_text: str, command: Optional[Dict[str, Any]] = None, filepath: str = SAVED_LOCATIONS_FILE) -> Dict[str, Any]:
    """Use saved-location support only when the loaded nav file actually has it."""
    resolver = getattr(nav, "resolve_saved_location", None)
    if resolver is None:
        return {"ok": False, "error": "saved location support is not available in this drone_nav_final.py"}
    try:
        return resolver(place_text, command, filepath=filepath)
    except TypeError:
        try:
            return resolver(place_text, command)
        except Exception as e:
            return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _save_current_location_safe(name: str, pos: Dict[str, Any], aliases=None, filepath: str = SAVED_LOCATIONS_FILE) -> Dict[str, Any]:
    """Use save-location support only when the loaded nav file actually has it."""
    saver = getattr(nav, "save_current_location", None)
    if saver is None:
        return {"ok": False, "error": "save location support is not available in this drone_nav_final.py"}
    try:
        return saver(name, pos, aliases=aliases or [], filepath=filepath)
    except TypeError:
        try:
            return saver(name, pos, aliases=aliases or [])
        except Exception as e:
            return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


class CommandRequest(BaseModel):
    command: str


class UrbanEyeController:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.command_queue: "queue.Queue[str]" = queue.Queue()
        self.logs = []

        self.boot_thread: Optional[threading.Thread] = None
        self.telemetry_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self.started = False
        self.ready = False
        self.phase = "Offline"
        self.last_error: Optional[str] = None
        self.last_command: str = ""
        self.last_voice_transcript: str = ""
        self.last_command_message: str = ""
        self.active_target_name: str = ""
        self.active_target_gps: Optional[Dict[str, float]] = None

        self.distance_left_m: Optional[float] = None
        self.agl_m: Optional[float] = None
        self.speed_mps: float = 0.0
        self.raw_gps_speed_mps: float = 0.0
        self.position: Optional[Dict[str, Any]] = None

        self.mav = None
        self.home_pos = None
        self.pos_reader = None
        self.ground_reader = None
        self.streamer = None
        self.forward_reader = None
        self.forward_active = False
        self.cruise_rel_alt = None

        self.hover_mode = False
        self.hover_agl_current = nav.HOVER_AGL
        self.last_hover_control_command = None

        # Optional future camera bridge. It is lazy so the backend still works
        # even before you decide to show the Unreal PNG feed on the website.
        self.camera_reader = None

        # Latest CV result from the separate Flask CV API.
        self.last_cv_result: Optional[Dict[str, Any]] = None
        self.last_cv_time: Optional[float] = None
        self.last_cv_frame_signature: str = ""

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with self.lock:
            self.logs.append(line)
            self.logs = self.logs[-250:]

    def set_phase(self, phase: str) -> None:
        with self.lock:
            self.phase = phase
        self.log(phase)

    def set_command_message(self, message: str) -> None:
        """Message meant for the website command box."""
        with self.lock:
            self.last_command_message = str(message or "")

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "started": self.started,
                "ready": self.ready,
                "phase": self.phase,
                "status": self.phase,
                "last_error": self.last_error,
                "last_command": self.last_command,
                "last_voice_transcript": self.last_voice_transcript,
                "last_command_message": self.last_command_message,
                "target_name": self.active_target_name,
                "distance_left_m": self.distance_left_m,
                "agl_m": self.agl_m,
                "speed_mps": self.speed_mps,
                "speed_kmh": self.speed_mps * 3.6,
                "raw_gps_speed_mps": getattr(self, "raw_gps_speed_mps", 0.0),
                "position": self.position,
                "hover_mode": self.hover_mode,
                "hover_agl_target": self.hover_agl_current,
                "forward_active": self.forward_active,
                "last_cv_result": self.last_cv_result,
                "last_cv_time": self.last_cv_time,
                "last_cv_frame_signature": self.last_cv_frame_signature,
                "logs": list(self.logs[-30:]),
            }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start_async(self) -> Dict[str, Any]:
        with self.lock:
            if self.boot_thread and self.boot_thread.is_alive():
                return {"ok": True, "message": "UrbanEye backend is already starting/running."}
            if self.started and self.ready:
                return {"ok": True, "message": "UrbanEye drone system is already ready."}

            self.stop_event.clear()
            self.boot_thread = threading.Thread(target=self._boot_and_command_loop, daemon=True)
            self.boot_thread.start()
            return {"ok": True, "message": "UrbanEye drone system is starting."}

    def _boot_and_command_loop(self) -> None:
        try:
            with self.lock:
                self.started = True
                self.ready = False
                self.last_error = None
            self.set_phase("Connecting to PX4")

            self.mav = nav.mavutil.mavlink_connection(nav.MAVLINK_CONNECTION)
            self.log("Waiting for PX4 heartbeat...")
            self.mav.wait_heartbeat()
            self.mav.target_system = 1
            self.mav.target_component = 1

            self.set_phase("Reading home position")
            self.home_pos = nav.wait_for_home_position(self.mav)
            if not self.home_pos:
                raise RuntimeError("Could not read PX4 home position.")
            self.log(
                f"Home position lat={self.home_pos['lat']:.10f}, "
                f"lon={self.home_pos['lon']:.10f}"
            )

            self.set_phase("Connecting to AirSim AGL sensor")
            self.ground_reader = nav.AirSimGroundReader(
                vehicle_name=nav.VEHICLE_NAME,
                sensor_name=nav.DISTANCE_SENSOR_NAME,
            )
            self.ground_reader.start()
            time.sleep(1.0)

            self.pos_reader = nav.PositionReader(self.mav)
            self.streamer = nav.SetpointStreamer(
                self.mav,
                self.pos_reader,
                self.ground_reader,
                self.home_pos["lat"],
                self.home_pos["lon"],
            )
            self.pos_reader.start()
            self.streamer.start()

            self.telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
            self.telemetry_thread.start()

            self.set_phase("Taking off to cruise AGL")
            takeoff_ok, self.cruise_rel_alt = nav.perform_takeoff_sequence(
                self.mav,
                self.pos_reader,
                self.streamer,
                self.ground_reader,
            )
            if self.cruise_rel_alt is None and self.home_pos is not None:
                self.cruise_rel_alt = nav.DESIRED_AGL

            if nav.FORWARD_ENABLE_AFTER_TAKEOFF and nav.FORWARD_OBSTACLE_ENABLE and takeoff_ok:
                self.forward_reader, self.forward_active = nav.ensure_forward_reader_started(self.forward_reader)

            with self.lock:
                self.ready = True
            self.set_phase("Ready for command")

            while not self.stop_event.is_set():
                try:
                    raw = self.command_queue.get(timeout=0.25)
                except queue.Empty:
                    continue
                self._execute_command(raw)

        except Exception as e:
            with self.lock:
                self.last_error = str(e)
                self.ready = False
                self.phase = "Error"
            self.log(f"ERROR: {e}")
            traceback.print_exc()

    def _display_speed_for_phase(self, raw_gps_speed: float) -> float:
        """Return a clean website speed value.

        Modes:
        - raw/gps/estimated: show the raw smoothed GPS estimate.
        - commanded: show the fixed configured PX4 speed.
        - smooth_clamped/default: show a moving/smoothed value, but clamp spikes so
          the dashboard does not jump to unrealistic values like 11 m/s when
          CRUISE_SPEED is 5 m/s.
        """
        mode = os.getenv("URBANEYE_SPEED_DISPLAY_MODE", "smooth_clamped").strip().lower()
        cruise = float(getattr(nav, "CRUISE_SPEED", 5.0))
        hover_move = float(getattr(nav, "HOVER_CONTROL_MICRO_SPEED", getattr(nav, "HOVER_CONTROL_SPEED", 2.0)))
        raw = max(0.0, float(raw_gps_speed or 0.0))
        phase = str(self.phase or "").lower()

        if mode in {"raw", "gps", "estimated"}:
            return min(60.0, raw)

        if mode in {"commanded", "target"}:
            if "flying" in phase:
                return cruise
            if "hover-control" in phase or "executing hover" in phase:
                return hover_move
            return 0.0

        # Default demo mode: value moves naturally, but cannot spike far above
        # the configured speed.
        if "flying" in phase:
            return min(raw, cruise + 0.8)

        if "hover-control" in phase or "executing hover" in phase:
            return min(raw, hover_move + 0.8)

        if (
            "hover" in phase
            or "ready" in phase
            or "arrived" in phase
            or "connecting" in phase
            or "reading home" in phase
            or "taking off" in phase
            or "takeoff" in phase
            or "landing" in phase
            or "landed" in phase
            or "offline" in phase
            or "error" in phase
        ):
            return 0.0

        return min(raw, cruise + 0.8)

    def _telemetry_loop(self) -> None:
        last_pos = None
        last_t = None
        last_raw_speed = 0.0

        while not self.stop_event.is_set():
            try:
                if self.pos_reader is not None:
                    pos = self.pos_reader.get_position()
                else:
                    pos = None

                agl = None
                if self.pos_reader is not None and self.ground_reader is not None:
                    agl = nav.get_believable_agl(self.pos_reader, self.ground_reader)

                now = time.time()
                raw_speed = last_raw_speed

                if pos and last_pos and last_t:
                    dt = max(0.05, now - last_t)
                    meters = nav.gps_to_distance(last_pos["lat"], last_pos["lon"], pos["lat"], pos["lon"])
                    instant_speed = min(60.0, max(0.0, meters / dt))

                    # Ignore huge one-sample jumps. They are usually GPS/update timing spikes,
                    # not real commanded movement.
                    if instant_speed > 18.0 and meters > 6.0:
                        instant_speed = raw_speed

                    raw_speed = (0.80 * raw_speed) + (0.20 * instant_speed)
                    last_raw_speed = raw_speed

                display_speed = self._display_speed_for_phase(raw_speed)

                dist_left = None
                if pos and self.active_target_gps:
                    dist_left = nav.gps_to_distance(
                        pos["lat"],
                        pos["lon"],
                        self.active_target_gps["lat"],
                        self.active_target_gps["lon"],
                    )

                with self.lock:
                    self.position = pos
                    self.agl_m = agl
                    self.speed_mps = float(display_speed)
                    self.raw_gps_speed_mps = float(raw_speed)
                    self.distance_left_m = dist_left

                if pos:
                    last_pos = pos
                    last_t = now

            except Exception as e:
                with self.lock:
                    self.last_error = str(e)
            time.sleep(0.5)

    # ------------------------------------------------------------------
    # Website command handling
    # ------------------------------------------------------------------
    def submit_command(self, raw: str) -> Dict[str, Any]:
        raw = (raw or "").strip()
        if not raw:
            raise HTTPException(status_code=400, detail="Command is empty.")
        if not self.started:
            # Nice behavior for demos: first command can start the drone system.
            self.start_async()
            return {
                "ok": True,
                "accepted": True,
                "message": "Drone system was offline, so it is starting first. Send the command again when status is Ready.",
            }
        if not self.ready:
            return {
                "ok": False,
                "accepted": False,
                "message": "Drone system is still starting. Wait until status is Ready for command.",
            }

        with self.lock:
            self.last_command = raw
            self.last_command_message = "Command received. Resolving now..."
        self.command_queue.put(raw)
        return {"ok": True, "accepted": True, "message": "Command received. Resolving now..."}

    def _execute_command(self, raw: str) -> None:
        self.log(f"Command received: {raw}")
        self.set_phase("Parsing command")
        command = _parse_terminal_command_safe(raw)

        ctype = command.get("type")

        if ctype == "empty":
            self.set_phase("Ready for command")
            return

        if ctype == "exit":
            self.stop_event.set()
            self._land_and_cleanup(reason="Exit command")
            return

        if ctype == "land":
            self._land_and_cleanup(reason="Landing command")
            return

        if ctype == "save_location":
            self._handle_save_location(command)
            return

        if ctype == "repeat_control":
            self._handle_repeat_control()
            return

        if ctype == "unknown":
            msg = "Unsupported command: " + str(command.get("error", "unknown command"))
            self.log(msg)
            self.set_command_message(msg)
            self.set_phase("Ready for command")
            return

        if ctype == "control":
            self._handle_hover_control(command)
            return

        if ctype == "navigate":
            self._handle_navigation(command)
            return

        self.log(f"Unsupported command type: {ctype}")
        self.set_phase("Ready for command")

    def _handle_save_location(self, command: Dict[str, Any]) -> None:
        self.set_phase("Saving current location")
        pos_to_save = nav.wait_for_fresh_position(self.pos_reader, timeout=2.0)
        if not pos_to_save:
            self.log("SAVE failed: no fresh drone GPS position.")
            self.set_phase("Ready for command")
            return

        result = _save_current_location_safe(
            command.get("name", ""),
            pos_to_save,
            aliases=command.get("aliases") or [],
            filepath=SAVED_LOCATIONS_FILE,
        )
        if not result.get("ok"):
            self.log("SAVE failed: " + str(result.get("error", "unknown error")))
        else:
            saved_loc = result["location"]
            self.log(f"Saved local location: {saved_loc.get('name')}")
        self.set_phase("Ready for command")

    def _handle_repeat_control(self) -> None:
        if not self.hover_mode:
            self.log("Repeat ignored: hover-control is not active yet.")
            self.set_phase("Ready for command")
            return
        if self.last_hover_control_command is None:
            self.log("Repeat ignored: no previous hover-control command.")
            self.set_phase("Ready for command")
            return
        self.set_phase("Repeating hover-control command")
        self.hover_agl_current = nav.execute_hover_control_command(
            self.last_hover_control_command,
            self.pos_reader,
            self.streamer,
            self.ground_reader,
            self.mav,
            self.hover_agl_current,
        )
        self.set_phase("Hovering")

    def _handle_hover_control(self, command: Dict[str, Any]) -> None:
        if not self.hover_mode:
            self.log("Hover-control ignored: reach a destination first.")
            self.set_phase("Ready for command")
            return
        if self.streamer is None or not self.streamer.is_alive():
            self.log("Hover-control ignored: streamer is not active.")
            self.set_phase("Ready for command")
            return

        self.set_phase("Executing hover-control")
        self.hover_agl_current = nav.execute_hover_control_command(
            command,
            self.pos_reader,
            self.streamer,
            self.ground_reader,
            self.mav,
            self.hover_agl_current,
        )
        repeatable = nav.make_repeatable_hover_command(command)
        if repeatable is not None:
            self.last_hover_control_command = repeatable
            self.log("Saved for repeat: " + nav.describe_hover_command(repeatable))
        self.set_phase("Hovering")

    def _handle_navigation(self, command: Dict[str, Any]) -> None:
        self.hover_mode = False
        self.last_hover_control_command = None
        if self.streamer is not None:
            self.streamer.hover_yaw_deg = None

        place_text = command["place_text"]
        self.set_command_message(f"Resolving destination: {place_text}")
        self.set_phase("Resolving saved location")
        try:
            resolved = _resolve_saved_location_safe(place_text, command, filepath=SAVED_LOCATIONS_FILE)
        except Exception as e:
            resolved = {"ok": False, "error": str(e)}

        if resolved.get("ok"):
            self.log(
                f"Saved location match: {resolved['display_name']} "
                f"score={float(resolved.get('match_score', 0.0)):.2f}"
            )
        else:
            self.set_phase("Resolving destination with OpenStreetMap")
            resolved = _resolve_place_with_osm_safe(place_text, command)
            if not resolved.get("ok"):
                detail = str(resolved.get("error", "unknown error"))
                self.log("Resolver error: " + detail)
                self.set_command_message("Could not find this place in the allowed Amman map area. Try a saved POI or a more specific Amman location.")
                self.set_phase("Ready for command")
                return

        gps = {"lat": float(resolved["lat"]), "lon": float(resolved["lon"]), "alt": 0.0}
        name = str(resolved.get("display_name", place_text))
        with self.lock:
            self.active_target_name = name
            self.active_target_gps = {"lat": gps["lat"], "lon": gps["lon"]}
            self.distance_left_m = None
        self.log(f"Resolved target: {name}")
        self.log(f"lat={gps['lat']:.10f}, lon={gps['lon']:.10f}")
        self.set_command_message(f"Destination found: {name}")

        if self.streamer is None or not self.streamer.is_alive():
            self.set_phase("Relaunching")
            self.streamer = nav.SetpointStreamer(
                self.mav,
                self.pos_reader,
                self.ground_reader,
                self.home_pos["lat"],
                self.home_pos["lon"],
            )
            self.streamer.start()
            takeoff_ok, self.cruise_rel_alt = nav.perform_takeoff_sequence(
                self.mav,
                self.pos_reader,
                self.streamer,
                self.ground_reader,
            )
            if self.cruise_rel_alt is None:
                pos_now = self.pos_reader.get_position()
                self.cruise_rel_alt = pos_now["alt_rel"] if pos_now else nav.DESIRED_AGL
            if nav.FORWARD_ENABLE_AFTER_TAKEOFF and nav.FORWARD_OBSTACLE_ENABLE and takeoff_ok:
                self.forward_reader, self.forward_active = nav.ensure_forward_reader_started(self.forward_reader)
            else:
                self.forward_active = False

        self.set_phase("Flying")
        success = nav.fly_to_target(
            name,
            gps,
            self.pos_reader,
            self.streamer,
            self.ground_reader,
            self.mav,
            self.cruise_rel_alt,
            self.forward_reader,
            forward_active=self.forward_active,
        )

        if success:
            self.set_phase("Arrived - descending to hover")
            nav.hold_at_target(
                name,
                gps,
                self.pos_reader,
                self.streamer,
                self.ground_reader,
                self.cruise_rel_alt,
                hold_agl=nav.HOVER_AGL,
                hold_seconds=nav.HOVER_DURATION,
            )
            pos_after_hold = nav.wait_for_fresh_position(self.pos_reader, timeout=2.0)
            if pos_after_hold:
                self.streamer.set_target(
                    pos_after_hold["lat"],
                    pos_after_hold["lon"],
                    max(nav.MIN_REL_ALT, self.cruise_rel_alt - (nav.DESIRED_AGL - nav.HOVER_AGL)),
                    target_agl=nav.HOVER_AGL,
                )
                self.streamer.hover_yaw_deg = nav.get_heading_from_position(pos_after_hold)
            self.hover_mode = True
            self.hover_agl_current = nav.HOVER_AGL
            self.set_command_message(f"Arrived at destination: {name}")
            self.set_phase("Hovering")
        else:
            self.hover_mode = False
            pos = self.pos_reader.get_position()
            if pos and self.streamer is not None:
                self.streamer.set_target(
                    pos["lat"],
                    pos["lon"],
                    max(nav.MIN_REL_ALT, self.cruise_rel_alt - (nav.DESIRED_AGL - nav.HOVER_AGL)),
                    target_agl=nav.HOVER_AGL,
                )
            self.set_command_message("Target not reached. Try a closer or clearer destination.")
            self.set_phase("Target not reached")

    def _land_and_cleanup(self, reason: str = "Landing") -> None:
        self.set_phase(reason)
        self.hover_mode = False
        self.last_hover_control_command = None
        try:
            self.streamer, self.forward_reader, self.forward_active = nav.perform_landing_sequence(
                self.mav,
                self.pos_reader,
                self.streamer,
                self.ground_reader,
                self.forward_reader,
            )
        except Exception as e:
            self.log("Landing error: " + str(e))
        with self.lock:
            self.ready = True
            self.active_target_name = ""
            self.active_target_gps = None
            self.distance_left_m = None
        self.set_phase("Landed - ready to relaunch")

    # ------------------------------------------------------------------
    # Future Unreal PNG camera support
    # ------------------------------------------------------------------
    def get_camera_status(self) -> Dict[str, Any]:
        try:
            if self.camera_reader is None:
                from unreal_camera_reader import UnrealCameraReader
                self.camera_reader = UnrealCameraReader()
                self.camera_reader.start()
            return self.camera_reader.get_status()
        except Exception as e:
            return {"has_frame": False, "fresh": False, "last_error": str(e)}

    def get_camera_png_bytes(self) -> Optional[bytes]:
        try:
            if self.camera_reader is None:
                from unreal_camera_reader import UnrealCameraReader
                self.camera_reader = UnrealCameraReader()
                self.camera_reader.start()
                time.sleep(0.2)
            frame = self.camera_reader.get_latest_frame()
            if frame is None:
                return None
            import cv2
            ok, encoded = cv2.imencode(".png", frame)
            if not ok:
                return None
            return encoded.tobytes()
        except Exception as e:
            with self.lock:
                self.last_error = str(e)
            return None


# ---------------------------------------------------------------------------
# CV API bridge helpers
# ---------------------------------------------------------------------------
def _post_image_bytes_to_cv_api(
    image_bytes: bytes,
    filename: str = "urbaneye_frame.png",
    content_type: str = "image/png",
    endpoint: str = "analyze_file",
) -> Dict[str, Any]:
    """Forward one image to the separate Flask CV API and return its JSON result.

    v26 uses the raw /analyze_file endpoint for auto-scan. That means the
    image goes through the same CV path as manual screenshot testing, not the
    optimized fast/compact endpoint.
    """
    if not image_bytes:
        raise RuntimeError("Image data is empty.")

    endpoint = endpoint.strip("/") or "analyze_file"
    files = {"image": (filename, image_bytes, content_type or "application/octet-stream")}

    url = f"{CV_API_BASE}/{endpoint}"
    response = requests.post(url, files=files, timeout=CV_API_TIMEOUT_SEC)

    if response.status_code == 404 and endpoint != "analyze_file":
        url = f"{CV_API_BASE}/analyze_file"
        response = requests.post(url, files=files, timeout=CV_API_TIMEOUT_SEC)

    response.raise_for_status()
    try:
        return response.json()
    except Exception as e:
        raise RuntimeError(f"CV API returned non-JSON response: {e}")


def _image_data_url_for_history(image_bytes: bytes, max_width: int = 960, jpeg_quality: int = 78) -> str:
    """Return a small browser-displayable image for Firestore history.

    The CV model receives the original PNG bytes. This history copy is only a
    compressed display snapshot so Firestore rows stay small and load fast.
    """
    try:
        import cv2
        import numpy as np

        arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("OpenCV could not decode image")

        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            frame = cv2.resize(frame, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if ok:
            return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    except Exception:
        pass

    return "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")


def _image_size_from_bytes(image_bytes: bytes) -> Dict[str, int]:
    """Return original frame width/height so browser can scale CV boxes correctly."""
    try:
        import cv2
        import numpy as np
        arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            h, w = frame.shape[:2]
            return {"width": int(w), "height": int(h)}
    except Exception:
        pass
    return {"width": 0, "height": 0}


def _image_signature(image_bytes: bytes) -> str:
    return hashlib.sha1(image_bytes).hexdigest()


def _store_latest_cv_result(result: Dict[str, Any], frame_signature: str = "") -> None:
    with controller.lock:
        controller.last_cv_result = result
        controller.last_cv_time = time.time()
        if frame_signature:
            controller.last_cv_frame_signature = frame_signature


def _cv_api_health_payload() -> Dict[str, Any]:
    """Check the separate Flask CV API without crashing the dashboard backend."""
    try:
        response = requests.get(f"{CV_API_BASE}/health", timeout=5)
        payload = response.json() if response.headers.get("content-type", "").lower().startswith("application/json") else {"raw": response.text}
        return {
            "ok": response.ok,
            "cv_api_base": CV_API_BASE,
            "status_code": response.status_code,
            "cv_api": payload,
            "vl_enabled": VL_ENABLE,
            "vl_model": VL_MODEL,
            "yolo_backup_mode": YOLO_BACKUP_MODE,
        }
    except Exception as e:
        return {
            "ok": False,
            "cv_api_base": CV_API_BASE,
            "error": str(e),
            "vl_enabled": VL_ENABLE,
            "vl_model": VL_MODEL,
            "yolo_backup_mode": YOLO_BACKUP_MODE,
        }


# ---------------------------------------------------------------------------
# v30 Qwen VL detector helpers
# ---------------------------------------------------------------------------
def _extract_json_object(text: Any) -> Optional[Dict[str, Any]]:
    """Extract one JSON object from a VL model response."""
    if text is None:
        return None
    if isinstance(text, dict):
        return text
    if isinstance(text, list):
        text = "\n".join(str(x) for x in text)
    text = str(text).strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        if isinstance(value, dict):
            return value
    except Exception:
        return None
    return None


def _image_data_url_for_vl(image_bytes: bytes) -> str:
    """Return a smaller image payload for the Hugging Face VL request.

    The local history image still keeps the full saved frame. This only reduces
    upload/token processing time for Qwen VL. Use max width 1600 by default so
    plates are not crushed too much.
    """
    try:
        import cv2
        import numpy as np
        arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            h, w = frame.shape[:2]
            max_width = max(640, int(VL_MAX_IMAGE_WIDTH))
            if w > max_width:
                scale = max_width / float(w)
                frame = cv2.resize(frame, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            quality = max(60, min(95, int(VL_JPEG_QUALITY)))
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    except Exception:
        pass
    return "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")


def _safe_box(box: Any, width: int, height: int) -> list:
    """Normalize a model box to [x1,y1,x2,y2] pixels. Accepts pixel or 0-1000 coordinates."""
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return []
    try:
        nums = [float(box[i]) for i in range(4)]
    except Exception:
        return []

    # If the model returns normalized 0-1000 coords, convert to pixels.
    if width > 0 and height > 0 and max(nums) <= 1000 and (width > 1100 or height > 1100):
        # Only do this if values look like Qwen normalized coords rather than tiny real pixel boxes.
        # A real pixel box in this project normally has x2/y2 much larger than 100 for the car.
        # The prompt asks for pixel coords, but this fallback handles normalized output.
        if max(nums) > max(width, height) * 0.9 or max(nums) > 700:
            nums = [nums[0] * width / 1000.0, nums[1] * height / 1000.0, nums[2] * width / 1000.0, nums[3] * height / 1000.0]

    x1, y1, x2, y2 = nums
    x1 = max(0, min(width - 1 if width else x1, x1))
    x2 = max(0, min(width - 1 if width else x2, x2))
    y1 = max(0, min(height - 1 if height else y1, y1))
    y2 = max(0, min(height - 1 if height else y2, y2))
    if x2 <= x1 or y2 <= y1:
        return []
    return [round(x1), round(y1), round(x2), round(y2)]


def _normalize_incident_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if "lean" in text or "window" in text or "out_window" in text:
        return "person_leaning_out_window"
    if "trash" in text or "garbage" in text or "object" in text:
        return "trash_object"
    if "top" in text or "roof" in text or "on_car" in text or "person" in text:
        return "person_on_top_car"
    return "person_on_top_car"


def _clean_known_plate(value: Any) -> str:
    plate = str(value or "").strip().upper()
    if not plate or plate.lower() in {"unreadable", "unknown", "none", "null", "n/a"}:
        return "unreadable"
    plate = re.sub(r"[^0-9-]", "", plate)
    digits = re.sub(r"\D", "", plate)
    if len(digits) == 7:
        plate = f"{digits[:2]}-{digits[2:]}"
    return plate if plate in KNOWN_PLATES else "unreadable"


def _vl_prompt(width: int, height: int) -> str:
    plates_block = "\n".join(KNOWN_PLATES)
    return f"""
You are the vision detector for the UrbanEye drone dashboard.
Analyze this drone image and detect ONLY these violation classes:
- person_on_top_car: a person is on top of a car, sitting/standing/lying on the car, or clearly outside on the car body.
- person_leaning_out_window: a person is leaning out of a car window or hanging out from the side/window.
- trash_object: visible trash/garbage object on the road/scene.

Known license plates. If a plate is readable, choose ONLY from this list:
{plates_block}

Rules:
- Return JSON only. No markdown and no explanation outside JSON.
- If there are multiple violating cars, return multiple vehicle objects.
- If there is a violation but the plate is not clearly readable, still report the violation and set plate_number to "unreadable".
- Only choose a plate number if it is clearly the closest match from the known list. Do not invent a plate.
- If unsure between two plates, use "unreadable".
- Use original image pixel coordinates for boxes. Image size is width={width}, height={height}.
- Boxes must be [x1, y1, x2, y2]. If you cannot localize a box, use [].

Return exactly this JSON shape:
{{
  "violation_detected": true,
  "vehicles": [
    {{
      "violation_detected": true,
      "incident_type": "person_on_top_car",
      "confidence": 0-100,
      "description": "short description",
      "car_description": "short car description like red sedan",
      "car_box": [x1, y1, x2, y2],
      "incident_box": [x1, y1, x2, y2],
      "plate_visible": true,
      "plate_number": "one known plate or unreadable",
      "plate_confidence": 0-100,
      "plate_box": [x1, y1, x2, y2]
    }}
  ],
  "manual_review": true,
  "reason": "short reason"
}}
""".strip()


def _call_qwen_vl(image_bytes: bytes, width: int, height: int) -> Dict[str, Any]:
    if not VL_ENABLE:
        raise RuntimeError("AI detection is disabled.")
    if not image_bytes:
        raise RuntimeError("Image data is empty.")

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Python package 'openai' is not installed. Run: pip install openai") from e

    client = OpenAI(
        base_url=VL_BASE_URL,
        api_key=_get_hf_token(),
        timeout=VL_TIMEOUT_SEC,
    )

    completion = client.chat.completions.create(
        model=VL_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _vl_prompt(width, height)},
                    {"type": "image_url", "image_url": {"url": _image_data_url_for_vl(image_bytes)}},
                ],
            }
        ],
        temperature=0,
        max_tokens=900,
    )

    content = completion.choices[0].message.content
    parsed = _extract_json_object(content)
    if not parsed:
        raise RuntimeError(f"VL returned non-JSON output: {content}")
    return parsed


def _vl_to_cv_result(vl: Dict[str, Any], image_size: Dict[str, int], elapsed_sec: float) -> Dict[str, Any]:
    width = int(image_size.get("width") or 0)
    height = int(image_size.get("height") or 0)
    vehicles = vl.get("vehicles")
    if not isinstance(vehicles, list):
        vehicles = []

    incidents = []
    plates = []
    incident_types = []
    readable_plates = []

    for idx, vehicle in enumerate(vehicles, start=1):
        if not isinstance(vehicle, dict):
            continue
        v_violation = bool(vehicle.get("violation_detected", vl.get("violation_detected", False)))
        if not v_violation:
            continue

        incident_type = _normalize_incident_type(vehicle.get("incident_type"))
        confidence = float(vehicle.get("confidence") or vehicle.get("violation_confidence") or 75.0)
        incident_box = _safe_box(vehicle.get("incident_box") or vehicle.get("car_box"), width, height)
        incidents.append({
            "type": incident_type,
            "confidence": round(max(0.0, min(100.0, confidence)), 2),
            "box": incident_box,
            "source": "qwen_vl",
            "description": str(vehicle.get("description") or vehicle.get("car_description") or "").strip(),
        })
        incident_types.append(incident_type)

        plate_number = _clean_known_plate(vehicle.get("plate_number"))
        plate_conf = float(vehicle.get("plate_confidence") or 0.0)
        plate_visible = bool(vehicle.get("plate_visible", plate_number != "unreadable"))
        plate_box = _safe_box(vehicle.get("plate_box"), width, height)
        if plate_number != "unreadable" and plate_conf >= VL_PLATE_MIN_CONFIDENCE:
            readable_plates.append(plate_number)
            ocr_status = "read"
        else:
            plate_number = "unreadable"
            ocr_status = "manual_review" if plate_visible else "not_visible"

        # Always include a plate item when a vehicle violation exists so the dashboard
        # can show manual review if the plate is not readable.
        plates.append({
            "plate_number": plate_number,
            "detection_confidence": round(max(0.0, min(100.0, plate_conf if plate_visible else 0.0)), 2),
            "ocr_confidence": round(max(0.0, min(100.0, plate_conf)), 2),
            "ocr_status": ocr_status,
            "ocr_method": "qwen_vl_known_plate_match",
            "support_count": 1,
            "box": plate_box,
            "crop_size": {},
            "source": "qwen_vl",
        })

    violation = bool(incidents) or bool(vl.get("violation_detected", False))
    if violation and not incidents:
        incidents.append({
            "type": _normalize_incident_type(vl.get("incident_type")),
            "confidence": float(vl.get("confidence") or 70.0),
            "box": [],
            "source": "qwen_vl",
            "description": str(vl.get("reason") or "VL detected a violation.").strip(),
        })
        incident_types.append(incidents[-1]["type"])

    return {
        "incidents": incidents,
        "license_plates": plates,
        "summary": {
            "api_version": "qwen_vl_v30_main_yolo_backup",
            "detector": "qwen_vl",
            "vl_model": VL_MODEL,
            "violation_detected": bool(violation),
            "incidents_detected": len(incidents),
            "plates_detected": len(plates),
            "readable_plates_detected": len(readable_plates),
            "full_plates_detected": len(readable_plates),
            "partial_plates_detected": 0,
            "incident_types": sorted(set(incident_types)),
            "plate_numbers": readable_plates,
            "full_plate_numbers": readable_plates,
            "partial_plate_numbers": [],
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": round(elapsed_sec, 3),
            "device": "hf_router",
            "manual_review": bool(vl.get("manual_review", False)) or len(readable_plates) == 0,
            "reason": str(vl.get("reason") or "").strip(),
            "plate_min_confidence": VL_PLATE_MIN_CONFIDENCE,
            "yolo_backup_mode": YOLO_BACKUP_MODE,
        },
    }


def _readable_plates_from_result(result: Dict[str, Any]) -> list:
    summary = result.get("summary", {}) if isinstance(result, dict) else {}
    plates = summary.get("plate_numbers") or summary.get("full_plate_numbers") or []
    if plates:
        return [p for p in plates if p and p != "unreadable"]
    out = []
    for p in result.get("license_plates", []) if isinstance(result, dict) else []:
        text = p.get("plate_number")
        if text and text != "unreadable" and p.get("ocr_status") in {"read", "partial_read"}:
            out.append(text)
    return out


def _should_call_yolo_backup(vl_result: Optional[Dict[str, Any]], vl_error: Optional[Exception] = None) -> bool:
    mode = YOLO_BACKUP_MODE
    if mode in {"0", "false", "no", "off", "never"}:
        return False
    if mode == "always":
        return True
    # Fast demo mode: keep YOLO only as a real fallback if the model call fails.
    # This avoids adding 2-3 seconds after every successful VL result just because
    # the plate is unreadable or a box is missing.
    if mode in {"error_only", "vl_error_only", "fail_only"}:
        return vl_error is not None or not vl_result
    if vl_error is not None or not vl_result:
        return True
    summary = vl_result.get("summary", {})
    violation = bool(summary.get("violation_detected")) or bool(vl_result.get("incidents"))
    has_readable_plate = bool(_readable_plates_from_result(vl_result))
    has_incident_box = any(bool(i.get("box")) for i in vl_result.get("incidents", []))
    return (not violation) or (not has_readable_plate) or (not has_incident_box)


def _merge_vl_and_yolo(vl_result: Dict[str, Any], yolo_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not yolo_result:
        return vl_result

    vl_summary = vl_result.get("summary", {})
    y_summary = yolo_result.get("summary", {})
    vl_violation = bool(vl_summary.get("violation_detected")) or bool(vl_result.get("incidents"))
    y_violation = bool(y_summary.get("violation_detected")) or bool(yolo_result.get("incidents"))

    # If model says no but YOLO says yes, use YOLO as backup.
    if not vl_violation and y_violation:
        yolo_result.setdefault("summary", {})["detector"] = "yolo_backup"
        yolo_result["summary"]["vl_model"] = VL_MODEL
        yolo_result["summary"]["api_version"] = "yolo_backup_from_v30"
        return yolo_result

    # Main case: keep model violation, but use YOLO plate if model could not read one.
    merged = json.loads(json.dumps(vl_result))
    if not _readable_plates_from_result(merged):
        yolo_plates = []
        for p in yolo_result.get("license_plates", []):
            text = p.get("plate_number")
            if text and text != "unreadable" and p.get("ocr_status") in {"read", "partial_read"}:
                yolo_plates.append(p)
        if yolo_plates:
            merged["license_plates"] = yolo_plates + [p for p in merged.get("license_plates", []) if p.get("plate_number") == "unreadable"]
            plate_nums = []
            for p in yolo_plates:
                text = p.get("plate_number")
                if text and text not in plate_nums:
                    plate_nums.append(text)
            merged.setdefault("summary", {})["plate_numbers"] = plate_nums
            merged["summary"]["full_plate_numbers"] = plate_nums
            merged["summary"]["readable_plates_detected"] = len(plate_nums)
            merged["summary"]["plates_detected"] = max(len(merged.get("license_plates", [])), len(plate_nums))
            merged["summary"]["plate_source"] = "yolo_backup"

    # If model boxes are empty but YOLO has boxes, keep YOLO incidents as visual backup only.
    if not any(bool(i.get("box")) for i in merged.get("incidents", [])) and yolo_result.get("incidents"):
        merged["incidents"] = yolo_result["incidents"]
        merged.setdefault("summary", {})["box_source"] = "yolo_backup"

    merged.setdefault("summary", {})["detector"] = "qwen_vl_main_yolo_backup"
    merged["summary"]["yolo_backup_used"] = True
    return merged


def _analyze_image_vl_main_yolo_backup(image_bytes: bytes, filename: str = "urbaneye_latest.png", content_type: str = "image/png") -> Dict[str, Any]:
    image_size = _image_size_from_bytes(image_bytes)
    start = time.time()
    vl_result = None
    vl_error = None

    if VL_ENABLE:
        try:
            raw_vl = _call_qwen_vl(image_bytes, image_size.get("width", 0), image_size.get("height", 0))
            vl_result = _vl_to_cv_result(raw_vl, image_size, time.time() - start)
        except Exception as e:
            vl_error = e
            controller.log(f"[AI DETECTION WARNING] detection failed: {e}")

    yolo_result = None
    if _should_call_yolo_backup(vl_result, vl_error):
        try:
            yolo_result = _post_image_bytes_to_cv_api(
                image_bytes,
                filename=filename,
                content_type=content_type,
                endpoint="analyze_file",
            )
        except Exception as e:
            controller.log(f"[YOLO BACKUP WARNING] Local CV API failed: {e}")
            if vl_result is None:
                raise

    if vl_result is None:
        if yolo_result is not None:
            yolo_result.setdefault("summary", {})["detector"] = "yolo_only_vl_failed"
            return yolo_result
        raise RuntimeError(vl_error or "Vision model and backup detection both failed.")

    merged = _merge_vl_and_yolo(vl_result, yolo_result)
    merged.setdefault("summary", {})["processing_time_seconds"] = round(time.time() - start, 3)
    return merged



# ---------------------------------------------------------------------------
# v32 separate VL resolver override
# ---------------------------------------------------------------------------
# The old inline VL helper functions are kept above for compatibility, but the
# active analyzer below delegates Qwen2.5-VL work to vl_resolver.py so the vision
# model is separated from the website/backend controller, like llm_resolver.py is
# separated from the drone navigation parser.
def _analyze_image_vl_main_yolo_backup(image_bytes: bytes, filename: str = "urbaneye_latest.png", content_type: str = "image/png") -> Dict[str, Any]:
    start = time.time()
    vl_result = None
    vl_error = None

    if getattr(vl_resolver, "VL_ENABLE", True):
        try:
            vl_result = vl_resolver.analyze_image_bytes(image_bytes, known_plates=KNOWN_PLATES)
        except Exception as e:
            vl_error = e
            controller.log(f"[AI DETECTION WARNING] detection failed: {e}")

    yolo_result = None
    if _should_call_yolo_backup(vl_result, vl_error):
        try:
            yolo_result = _post_image_bytes_to_cv_api(
                image_bytes,
                filename=filename,
                content_type=content_type,
                endpoint="analyze_file",
            )
        except Exception as e:
            controller.log(f"[YOLO BACKUP WARNING] Local CV API failed: {e}")

    if vl_result:
        merged = _merge_vl_and_yolo(vl_result, yolo_result)
        merged.setdefault("summary", {})["resolver_file"] = "vl_resolver.py"
        merged["summary"]["backend_version"] = BACKEND_FIX_VERSION
        # Keep total pipeline time visible even when YOLO backup was also called.
        merged["summary"]["processing_time_seconds"] = round(time.time() - start, 3)
        return merged

    if yolo_result:
        yolo_result.setdefault("summary", {})["detector"] = "yolo_backup_only"
        yolo_result["summary"]["vl_error"] = str(vl_error) if vl_error else "Vision model disabled or returned no result."
        yolo_result["summary"]["backend_version"] = BACKEND_FIX_VERSION
        return yolo_result

    if vl_error:
        raise vl_error
    raise RuntimeError("Both vl_resolver and YOLO backup returned no result.")


def _sanitize_detector_labels_for_demo(result: Dict[str, Any]) -> Dict[str, Any]:
    """Hide provider/model-specific detector labels from logs/status payloads."""
    if not isinstance(result, dict):
        return result

    summary = result.get("summary")
    if isinstance(summary, dict):
        # Keep the dashboard generic: no qwen_vl / model provider labels.
        summary["detector"] = "ai_detection"
        summary.pop("vl_model", None)
        summary.pop("resolver_file", None)

    for incident in result.get("incidents", []) if isinstance(result.get("incidents"), list) else []:
        if isinstance(incident, dict):
            incident["source"] = "ai_detection"

    for plate in result.get("license_plates", []) if isinstance(result.get("license_plates"), list) else []:
        if isinstance(plate, dict):
            plate["source"] = "ai_detection"
            if str(plate.get("ocr_method", "")).lower().startswith("qwen"):
                plate["ocr_method"] = "ai_known_plate_match"
            if str(plate.get("ocr_method", "")).lower().startswith("vision_model"):
                plate["ocr_method"] = "ai_known_plate_match"

    return result


controller = UrbanEyeController()

app = FastAPI(title="UrbanEye Local Drone Bridge", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "UrbanEye web backend", "voice_fix_version": BACKEND_FIX_VERSION}


@app.post("/api/start")
def start() -> Dict[str, Any]:
    return controller.start_async()


@app.get("/api/status")
def status() -> Dict[str, Any]:
    return controller.snapshot()


@app.post("/api/command")
def command(req: CommandRequest) -> Dict[str, Any]:
    return controller.submit_command(req.command)


@app.post("/api/land")
def land() -> Dict[str, Any]:
    return controller.submit_command("land")


@app.get("/api/logs")
def logs() -> Dict[str, Any]:
    return {"logs": controller.snapshot()["logs"]}


@app.post("/api/voice/upload")
async def upload_voice_command(request: Request) -> Dict[str, Any]:
    """Receive one browser recording, transcribe it, then submit text to nav.

    This endpoint intentionally processes only the audio bytes from THIS request.
    Normal text commands still go through /api/command and never read any old
    voice_command.wav file.
    """
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="No audio data received.")

    VOICE_COMMAND_FILE.write_bytes(data)

    try:
        # v20 real fix:
        # Do NOT import/call speech_recognition_hf for website voice.
        # The backend posts directly to Hugging Face with a forced audio MIME,
        # so no old function signature/cache can break this path.
        header_hex = data[:16].hex(" ")
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            forced_audio_type = "audio/wav"
            detected_container = "wav"
        elif len(data) >= 4 and data[:4] == b"\x1A\x45\xDF\xA3":
            forced_audio_type = "audio/webm;codecs=opus"
            detected_container = "webm"
        elif len(data) >= 4 and data[:4] == b"OggS":
            forced_audio_type = "audio/ogg"
            detected_container = "ogg"
        else:
            forced_audio_type = "audio/wav"
            detected_container = "unknown-forced-wav"

        self_reported_type = request.headers.get("content-type", "")
        controller.log(
            f"Voice upload saved as voice_command.wav | stt_model={HF_STT_MODEL} | browser_type={self_reported_type or 'none'} "
            f"| detected={detected_container} | sending_to_hf={forced_audio_type} | header={header_hex}"
        )

        def _transcribe_with_forced_audio_type():
            return _direct_hf_transcribe_audio(data, forced_audio_type)

        loop = asyncio.get_running_loop()
        transcript = await loop.run_in_executor(None, _transcribe_with_forced_audio_type)
        transcript = (transcript or "").strip()
        if not transcript:
            raise HTTPException(status_code=422, detail="Speech recognition returned empty text.")

        with controller.lock:
            controller.last_voice_transcript = transcript

        nav_result = controller.submit_command(transcript)
        accepted = bool(nav_result.get("accepted", False))
        return {
            "ok": bool(nav_result.get("ok", False)),
            "accepted": accepted,
            "filename": "voice_command.wav",
            "path": str(VOICE_COMMAND_FILE),
            "size_bytes": len(data),
            "detected_container": detected_container,
            "sent_content_type_to_hf": forced_audio_type,
            "browser_content_type": self_reported_type,
            "voice_fix_version": BACKEND_FIX_VERSION,
            "transcript": transcript,
            "nav_result": nav_result,
            "message": f"Voice transcribed and sent as text: {transcript}",
        }
    except HTTPException:
        raise
    except Exception as e:
        with controller.lock:
            controller.last_error = str(e)
        raise HTTPException(status_code=500, detail=f"Voice saved, but transcription failed: {e}")


@app.get("/api/cv/health")
def cv_health() -> Dict[str, Any]:
    """Dashboard check for the separate Flask CV API on port 5000."""
    return _cv_api_health_payload()


@app.post("/api/cv/reset-session")
def cv_reset_session() -> Dict[str, Any]:
    """Reset duplicate/signature memory when the operator manually starts a new VL scan session."""
    with controller.lock:
        controller.last_cv_frame_signature = ""
        controller.last_cv_result = None
        controller.last_cv_time = None
    controller.log("Model detection session reset by dashboard.")
    return {"ok": True, "message": "CV detection session reset."}


@app.post("/api/cv/analyze-current-frame")
async def cv_analyze_current_frame(request: Request) -> Dict[str, Any]:
    """Analyze the latest Unreal PNG frame.

    v49 behavior:
    - Start Detection button can run continuous one-at-a-time detection
    - Capture + Analyze Now button can force one immediate snapshot analysis
    - reads the current Unreal PNG frame into memory BEFORE sending to VL
    - skips duplicate frames unless {"force": true} is sent
    - uses the AI detection pipeline only for the final demo
    - returns the image + position metadata so the dashboard can auto-save
      violation rows directly into Capture History
    """
    force = False
    try:
        payload = await request.json()
        if isinstance(payload, dict):
            force = bool(payload.get("force") or payload.get("single") or payload.get("manual"))
    except Exception:
        force = False

    png = controller.get_camera_png_bytes()
    if png is None:
        raise HTTPException(status_code=404, detail="No Unreal PNG frame is ready yet.")

    frame_signature = _image_signature(png)
    with controller.lock:
        if (not force) and controller.last_cv_frame_signature == frame_signature:
            return {
                "ok": True,
                "source": "current_frame",
                "skipped": True,
                "reason": "same_frame_already_analyzed",
                "frame_signature": frame_signature,
                "last_cv_time": controller.last_cv_time,
            }

    try:
        result = _analyze_image_vl_main_yolo_backup(
            png,
            filename="urbaneye_latest.png",
            content_type="image/png",
        )
        result = _sanitize_detector_labels_for_demo(result)
        _store_latest_cv_result(result, frame_signature=frame_signature)
        summary = result.get("summary", {}) if isinstance(result, dict) else {}
        violation = bool(summary.get("violation_detected")) or bool(result.get("incidents"))
        controller.log(
            "AI scan complete | "
            f"violation={violation} | "
            f"incidents={summary.get('incidents_detected', 0)} | "
            f"plates={summary.get('plates_detected', 0)} | "
            f"time={summary.get('processing_time_seconds', 'N/A')}s"
        )
        return {
            "ok": True,
            "source": "current_frame",
            "skipped": False,
            "cv_api_base": CV_API_BASE,
            "frame_signature": frame_signature,
            "image_data_url": _image_data_url_for_history(png),
            "image_size": _image_size_from_bytes(png),
            "position": controller.position,
            "target_name": controller.active_target_name,
            "result": result,
        }
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=f"CV API is not running at {CV_API_BASE}. Start D:\\scripts\\UrbanEye_API\\app.py first.",
        )
    except Exception as e:
        with controller.lock:
            controller.last_error = str(e)
        raise HTTPException(status_code=500, detail=f"CV analysis failed: {e}")


@app.post("/api/cv/analyze-upload")
async def cv_analyze_upload(request: Request) -> Dict[str, Any]:
    """Analyze an uploaded dashboard image using the separate Flask CV API.

    The dashboard sends the raw image bytes here. This avoids requiring
    python-multipart and keeps the browser talking only to the main backend.
    """
    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image data received.")

    content_type = request.headers.get("content-type", "image/png")
    ext = "jpg" if "jpeg" in content_type.lower() else "png"

    try:
        result = _analyze_image_vl_main_yolo_backup(
            image_bytes,
            filename=f"dashboard_upload.{ext}",
            content_type=content_type,
        )
        _store_latest_cv_result(result)
        summary = result.get("summary", {}) if isinstance(result, dict) else {}
        controller.log(
            "CV uploaded-image analysis complete | "
            f"incidents={summary.get('incidents_detected', 0)} | "
            f"plates={summary.get('plates_detected', 0)} | "
            f"time={summary.get('processing_time_seconds', 'N/A')}s"
        )
        return {
            "ok": True,
            "source": "uploaded_image",
            "cv_api_base": CV_API_BASE,
            "result": result,
        }
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=f"CV API is not running at {CV_API_BASE}. Start D:\\scripts\\UrbanEye_API\\app.py first.",
        )
    except Exception as e:
        with controller.lock:
            controller.last_error = str(e)
        raise HTTPException(status_code=500, detail=f"CV analysis failed: {e}")



@app.get("/api/voice/status")
def voice_status() -> Dict[str, Any]:
    exists = VOICE_COMMAND_FILE.exists()
    return {
        "ok": True,
        "exists": exists,
        "filename": "voice_command.wav",
        "path": str(VOICE_COMMAND_FILE),
        "size_bytes": VOICE_COMMAND_FILE.stat().st_size if exists else 0,
    }


@app.get("/api/camera/status")
def camera_status() -> Dict[str, Any]:
    return controller.get_camera_status()


@app.get("/api/camera/frame.png")
def camera_frame() -> Response:
    png = controller.get_camera_png_bytes()
    if png is None:
        raise HTTPException(status_code=404, detail="No Unreal PNG frame is ready yet.")
    return Response(content=png, media_type="image/png")
