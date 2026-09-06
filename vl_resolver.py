"""
vl_resolver.py

UrbanEye Hugging Face Qwen2.5-VL resolver for vision-based violation detection.

Purpose:
- Keep the vision-language model logic separate from the website backend.
- Analyze one drone camera image and return a dashboard-friendly CV result.
- Use known plate numbers only, so the model does not invent random plates.

Required environment variable:
- HF_TOKEN

Optional environment variables:
- URBANEYE_VL_MODEL: default "Qwen/Qwen2.5-VL-7B-Instruct"
- URBANEYE_VL_BASE_URL: default "https://router.huggingface.co/v1"
- URBANEYE_VL_TIMEOUT_SEC: default 120
- URBANEYE_VL_PLATE_MIN_CONFIDENCE: default 70
- URBANEYE_VL_MAX_IMAGE_WIDTH: default 1600
- URBANEYE_VL_JPEG_QUALITY: default 85
- URBANEYE_KNOWN_PLATES: comma-separated known plate list

Public functions:
- analyze_image_bytes(image_bytes, known_plates=None)
- analyze_image_file(image_path, known_plates=None)
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


VL_ENABLE = os.getenv("URBANEYE_VL_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off"}
VL_MODEL = os.getenv("URBANEYE_VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
VL_BASE_URL = os.getenv("URBANEYE_VL_BASE_URL", "https://router.huggingface.co/v1")
VL_TIMEOUT_SEC = float(os.getenv("URBANEYE_VL_TIMEOUT_SEC", "120"))
VL_PLATE_MIN_CONFIDENCE = float(os.getenv("URBANEYE_VL_PLATE_MIN_CONFIDENCE", "70"))
VL_MAX_IMAGE_WIDTH = int(os.getenv("URBANEYE_VL_MAX_IMAGE_WIDTH", "1600"))
VL_JPEG_QUALITY = int(os.getenv("URBANEYE_VL_JPEG_QUALITY", "85"))

DEFAULT_KNOWN_PLATES = [
    p.strip() for p in os.getenv(
        "URBANEYE_KNOWN_PLATES",
        "11-11111,13-57911,22-22222,18-69854,12-34567,12-12121,23-23244,34-35362,"
        "67-45678,98-76543,54-23456,78-45372,45-38923,67-67676,74-74378,12-64296,"
        "44-87523,29-38472,33-27151,22-38452,12-18351"
    ).split(",") if p.strip()
]

ALLOWED_INCIDENT_TYPES = {
    "person_on_top_car",
    "person_leaning_out_window",
    "trash_object",
}


def _get_hf_token() -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Run: setx HF_TOKEN \"your_token_here\" then reopen CMD.")
    return token


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    cleaned = str(text).strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None

    try:
        value = json.loads(match.group(0))
        if isinstance(value, dict):
            return value
    except Exception:
        return None

    return None


def _image_size_from_bytes(image_bytes: bytes) -> Dict[str, int]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as im:
            return {"width": int(im.width), "height": int(im.height)}
    except Exception:
        return {"width": 0, "height": 0}


def _image_data_url_for_vl(image_bytes: bytes) -> str:
    """Compress the image before sending it to the VL model.

    This keeps the request smaller and faster while preserving enough detail for
    the model. If Pillow is unavailable or compression fails, the original image
    is sent as a PNG data URL.
    """
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as im:
            im = im.convert("RGB")
            max_width = max(640, int(VL_MAX_IMAGE_WIDTH))
            if im.width > max_width:
                scale = max_width / float(im.width)
                new_h = max(1, int(im.height * scale))
                im = im.resize((max_width, new_h))

            quality = max(60, min(95, int(VL_JPEG_QUALITY)))
            out = io.BytesIO()
            im.save(out, format="JPEG", quality=quality, optimize=True)
            encoded = base64.b64encode(out.getvalue()).decode("ascii")
            return "data:image/jpeg;base64," + encoded
    except Exception:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return "data:image/png;base64," + encoded


def _safe_box(value: Any, width: int = 0, height: int = 0) -> List[int]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []

    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return []

    if width > 0:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
    if height > 0:
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

    if x2 <= x1 or y2 <= y1:
        return []
    return [x1, y1, x2, y2]


def _normalize_incident_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace(" ", "_").replace("-", "_")
    if text in ALLOWED_INCIDENT_TYPES:
        return text

    if "lean" in text or "window" in text:
        return "person_leaning_out_window"
    if "trash" in text or "garbage" in text or "object" in text:
        return "trash_object"
    return "person_on_top_car"


def _clean_known_plate(value: Any, known_plates: List[str]) -> str:
    plate = str(value or "").strip().upper()
    plate = re.sub(r"[^0-9\-]", "", plate)
    plate = re.sub(r"-+", "-", plate).strip("-")

    digits = re.sub(r"\D", "", plate)
    if len(digits) == 7:
        plate = f"{digits[:2]}-{digits[2:]}"

    return plate if plate in known_plates else "unreadable"


def _build_prompt(known_plates: List[str], width: int, height: int) -> str:
    plates_block = "\n".join(known_plates)
    return f"""
You are the UrbanEye vision-language detector.
Analyze this drone image for traffic/public-safety violations.

Detect ONLY these violation classes:
1. person_on_top_car
2. person_leaning_out_window
3. trash_object

Known license plates. If a plate is visible, choose ONLY from this exact list:
{plates_block}

Important rules:
- Return JSON only. No markdown. No explanation outside JSON.
- If people are sitting, standing, riding, or clearly positioned on top of a car, use person_on_top_car.
- If a person is leaning out of a car window/door, use person_leaning_out_window.
- If a trash object is visible as the violation, use trash_object.
- If no violation exists, return violation_detected false and vehicles [].
- For plate_number, choose from the known list only. Never invent a plate.
- If the plate is blurry or uncertain, return "unreadable".
- Give plate_confidence from 0 to 100.
- If plate_confidence is below {VL_PLATE_MIN_CONFIDENCE:.0f}, the system will treat it as manual review.
- Bounding boxes should be pixel coordinates in the original image size: width={width}, height={height}.
- If you are not sure about a box, return an empty list [].

Return this exact JSON shape:
{{
  "violation_detected": true,
  "vehicles": [
    {{
      "violation_detected": true,
      "incident_type": "person_on_top_car",
      "confidence": 0,
      "description": "short description",
      "car_description": "short car color/type description",
      "car_box": [x1, y1, x2, y2],
      "incident_box": [x1, y1, x2, y2],
      "plate_visible": true,
      "plate_number": "unreadable",
      "plate_confidence": 0,
      "plate_box": [x1, y1, x2, y2]
    }}
  ],
  "manual_review": true,
  "reason": "short reason"
}}
""".strip()


def call_qwen_vl(image_bytes: bytes, known_plates: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return the raw JSON object from Qwen VL."""
    if not VL_ENABLE:
        raise RuntimeError("VL detector is disabled by URBANEYE_VL_ENABLE=0.")
    if not image_bytes:
        raise RuntimeError("Image data is empty.")

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Python package 'openai' is not installed. Run: pip install openai") from e

    known_plates = list(known_plates or DEFAULT_KNOWN_PLATES)
    image_size = _image_size_from_bytes(image_bytes)
    width = int(image_size.get("width") or 0)
    height = int(image_size.get("height") or 0)

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
                    {"type": "text", "text": _build_prompt(known_plates, width, height)},
                    {"type": "image_url", "image_url": {"url": _image_data_url_for_vl(image_bytes)}},
                ],
            }
        ],
        temperature=0,
        max_tokens=800,
    )

    content = completion.choices[0].message.content
    parsed = _extract_json_object(content)
    if not parsed:
        raise RuntimeError(f"VL returned non-JSON output: {content}")
    return parsed


def vl_to_cv_result(raw_vl: Dict[str, Any], image_size: Dict[str, int], elapsed_sec: float, known_plates: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convert raw VL JSON into the same shape the dashboard expects from CV."""
    known_plates = list(known_plates or DEFAULT_KNOWN_PLATES)
    width = int(image_size.get("width") or 0)
    height = int(image_size.get("height") or 0)

    vehicles = raw_vl.get("vehicles")
    if not isinstance(vehicles, list):
        vehicles = []

    incidents = []
    plates = []
    incident_types = []
    readable_plates = []

    for vehicle in vehicles:
        if not isinstance(vehicle, dict):
            continue

        v_violation = bool(vehicle.get("violation_detected", raw_vl.get("violation_detected", False)))
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

        plate_number = _clean_known_plate(vehicle.get("plate_number"), known_plates)
        plate_conf = float(vehicle.get("plate_confidence") or 0.0)
        plate_visible = bool(vehicle.get("plate_visible", plate_number != "unreadable"))
        plate_box = _safe_box(vehicle.get("plate_box"), width, height)

        if plate_number != "unreadable" and plate_conf >= VL_PLATE_MIN_CONFIDENCE:
            readable_plates.append(plate_number)
            ocr_status = "read"
        else:
            plate_number = "unreadable"
            ocr_status = "manual_review" if plate_visible else "not_visible"

        # Always include a plate item for a violating vehicle so History can show
        # manual review if the plate is not readable.
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

    violation = bool(incidents) or bool(raw_vl.get("violation_detected", False))
    if violation and not incidents:
        incidents.append({
            "type": _normalize_incident_type(raw_vl.get("incident_type")),
            "confidence": float(raw_vl.get("confidence") or 70.0),
            "box": [],
            "source": "qwen_vl",
            "description": str(raw_vl.get("reason") or "VL detected a violation.").strip(),
        })
        incident_types.append(incidents[-1]["type"])

    return {
        "incidents": incidents,
        "license_plates": plates,
        "summary": {
            "api_version": "qwen_vl_resolver_v32",
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
            "manual_review": bool(raw_vl.get("manual_review", False)) or len(readable_plates) == 0,
            "reason": str(raw_vl.get("reason") or "").strip(),
            "plate_min_confidence": VL_PLATE_MIN_CONFIDENCE,
        },
    }


def analyze_image_bytes(image_bytes: bytes, known_plates: Optional[List[str]] = None) -> Dict[str, Any]:
    """Analyze image bytes and return dashboard-friendly CV result."""
    start = time.time()
    image_size = _image_size_from_bytes(image_bytes)
    raw = call_qwen_vl(image_bytes, known_plates=known_plates)
    return vl_to_cv_result(raw, image_size, time.time() - start, known_plates=known_plates)


def analyze_image_file(image_path: str | Path, known_plates: Optional[List[str]] = None) -> Dict[str, Any]:
    """Analyze an image file and return dashboard-friendly CV result."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path.resolve()}")
    return analyze_image_bytes(path.read_bytes(), known_plates=known_plates)


if __name__ == "__main__":
    image = Path(os.getenv("URBANEYE_TEST_IMAGE", "test_image.png"))
    result = analyze_image_file(image)
    print(json.dumps(result, indent=2, ensure_ascii=False))
