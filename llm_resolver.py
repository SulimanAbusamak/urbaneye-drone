"""
llm_resolver.py

Hugging Face Qwen resolver for UrbanEye with:
- navigation parsing
- Arabic-priority OSM search candidates
- hover-control parsing
- local repeat-command handling
- deterministic "turn around" = 180 degree parsing
- local rejection of greetings/small talk so the LLM does not chat/respond

Used by the nav code through:
    parse_terminal_command(raw)
    resolve_place_with_osm(place_text, command=None)

Required environment variable:
- HF_TOKEN: Hugging Face access token

Optional environment variables:
- HF_PROVIDER: default "featherless-ai"
- HF_MODEL: default "Qwen/Qwen2.5-7B-Instruct"
- HF_MAX_TOKENS: default 220
- HF_TEMPERATURE: default 0.0
- NOMINATIM_URL: default "https://nominatim.openstreetmap.org/search"
- NOMINATIM_USER_AGENT: default "UrbanEye-Resolver/1.0"
- NOMINATIM_COUNTRY_CODES: default "jo"
- NOMINATIM_VIEWBOX: default Amman/Jordan-focused bounding box
"""

import json
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, Optional

import requests
from huggingface_hub import InferenceClient


# ---------------------------------------------------------------------------
# Hugging Face hosted LLM config
# ---------------------------------------------------------------------------
HF_PROVIDER = os.getenv("HF_PROVIDER", "featherless-ai")
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Keep output small. The resolver should only return JSON.
MAX_TOKENS = int(os.getenv("HF_MAX_TOKENS", "220"))
TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.0"))


# ---------------------------------------------------------------------------
# OSM / Nominatim config
# ---------------------------------------------------------------------------
NOMINATIM_URL = os.getenv("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
NOMINATIM_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "UrbanEye-Resolver/1.0")
DEFAULT_COUNTRY_CODES = os.getenv("NOMINATIM_COUNTRY_CODES", "jo")

# Format: left,top,right,bottom = west,north,east,south
# This loosely focuses OSM search around Amman/Jordan.
DEFAULT_VIEWBOX = os.getenv("NOMINATIM_VIEWBOX", "35.68,32.09,36.10,31.55")


# ---------------------------------------------------------------------------
# Local saved location database config
# ---------------------------------------------------------------------------
# This file lets UrbanEye build its own private/custom map.
# The drone/nav code can save the current GPS point here, and future navigation
# commands check this file before falling back to OpenStreetMap.
SAVED_LOCATIONS_FILE = os.getenv("URBANEYE_SAVED_LOCATIONS_FILE", "saved_locations.json")
SAVED_LOCATION_MIN_MATCH_SCORE = float(os.getenv("URBANEYE_SAVED_LOCATION_MIN_SCORE", "0.74"))


# ---------------------------------------------------------------------------
# Local parser dictionaries
# ---------------------------------------------------------------------------
_PREFIX_PATTERNS = [
    r"^go\s+to\s+",
    r"^fly\s+to\s+",
    r"^navigate\s+to\s+",
    r"^take\s+me\s+to\s+",
    r"^روح\s+على\s+",
    r"^اذهب\s+الى\s+",
    r"^اذهب\s+إلى\s+",
    r"^وديني\s+على\s+",
    r"^روح\s+ل\s*",
]

_LAND_WORDS = {"land", "هبوط", "landing"}
_EXIT_WORDS = {"exit", "quit", "close", "stop program", "خروج", "سكر"}

# Local parsing for commands like:
#   save this location as checkpoint one
#   save new location with name sports entrance
#   احفظ موقعي باسم نقطة التفتيش
# These are handled locally because the nav code must provide the current GPS.
_SAVE_LOCATION_PATTERNS = [
    r"^\s*(?:save|store|remember)\s+(?:this|current|my|new\s+)?(?:location|place|point|spot|position)\s*(?:as|called|named|with\s+name)\s+(.+?)\s*$",
    r"^\s*(?:save|store|remember)\s+(?:this|current|my)\s+(?:location|place|point|spot|position)\s+(.+?)\s*$",
    r"^\s*(?:save|store|remember)\s+(?:here|this\s+spot)\s*(?:as|called|named|with\s+name)\s+(.+?)\s*$",
    r"^\s*(?:احفظ|خزن|سجل)\s+(?:هذا|هاي|هاذ|الموقع|المكان|النقطة|مكاني|موقعي|الموقع\s+الحالي|المكان\s+الحالي|النقطة\s+الحالية)?\s*(?:باسم|اسمها|اسمه|ك|كـ)\s+(.+?)\s*$",
    r"^\s*(?:احفظ|خزن|سجل)\s+(?:مكاني|موقعي|الموقع\s+الحالي|المكان\s+الحالي|النقطة\s+الحالية)\s+(.+?)\s*$",
]

# Repeat is handled locally so we do not waste an API call and so the drone code
# can safely repeat the last hover-control command from its own memory.
_REPEAT_PHRASES = {
    "again",
    "do it again",
    "same again",
    "repeat",
    "repeat that",
    "one more time",
    "مرة ثانية",
    "مره ثانيه",
    "كرر",
    "كررها",
    "عيد",
    "عيدها",
    "نفس الشي",
    "نفسه",
    "كمان مرة",
    "كمان مره",
}

# Small talk / greetings are rejected locally so the model does not answer them.
_CHITCHAT_PHRASES = {
    "hi",
    "hello",
    "hey",
    "hi how are you",
    "hello how are you",
    "how are you",
    "how r u",
    "how are u",
    "what's up",
    "whats up",
    "good morning",
    "good evening",
    "مرحبا",
    "اهلا",
    "أهلا",
    "اهلين",
    "السلام عليكم",
    "كيفك",
    "كيف الحال",
    "كيف حالك",
    "شو اخبارك",
}

_ARABIC_RIGHT = {"يمين", "اليمين", "يمينا", "يميناً"}
_ARABIC_LEFT = {"يسار", "اليسار", "شمال", "الشمال", "يسارا", "يساراً"}
_ARABIC_FORWARD = {"قدام", "للأمام", "امام", "أمام", "forward"}
_ARABIC_BACK = {"ورا", "للخلف", "خلف", "back", "backward"}
_ARABIC_UP = {"فوق", "لفوق", "اعلى", "أعلى", "اطلع", "ارتفع", "ارتفاع", "اصعد", "صعود"}
_ARABIC_DOWN = {"تحت", "لتحت", "انزل", "نزل", "اهبط", "هبوط", "اخفض"}

_TURN_AROUND_PHRASES = [
    "turn around",
    "turn back",
    "u turn",
    "u-turn",
    "make a u turn",
    "make a u-turn",
    "rotate around",
    "face backward",
    "face backwards",
    "look behind",
    "turn 180",
    "turn 180 degrees",
    "rotate 180",
    "rotate 180 degrees",
    "لف ورا",
    "لف للخلف",
    "لف 180",
    "لف مية وثمانين",
    "لف ميه وثمانين",
    "استدر",
    "استدير",
    "استدر للخلف",
    "دور للخلف",
]


def _normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\u0600-\u06FF\s\-']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _has_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def _dedupe(items):
    seen = set()
    out = []
    for item in items:
        if item is None:
            continue
        item = str(item).strip()
        if not item:
            continue
        key = re.sub(r"\s+", " ", item.lower())
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _clean_prefixes(text: str) -> str:
    place_text = (text or "").strip()
    for pattern in _PREFIX_PATTERNS:
        place_text = re.sub(pattern, "", place_text, flags=re.IGNORECASE).strip()
    return place_text or (text or "").strip()


def _clean_saved_location_name(name: str) -> str:
    """Normalize a user-provided saved-location name without changing Arabic/English wording."""
    name = (name or "").strip()
    # If a loose fallback pattern captured the connector word too, remove it.
    name = re.sub(r"^(?:as|called|named|with\s+name)\s+", "", name, flags=re.IGNORECASE).strip()
    name = re.sub(r"^(?:باسم|اسمها|اسمه|ك|كـ)\s+", "", name).strip()
    name = re.sub(r"^[\s'\"`]+|[\s'\"`]+$", "", name)
    name = re.sub(r"[.،,;:]+$", "", name).strip()
    name = re.sub(r"\s+", " ", name)
    return name


def _extract_save_location_name(text: str) -> Optional[str]:
    """Return the requested saved-place name, or None if this is not a save command."""
    raw = (text or "").strip()
    if not raw:
        return None

    for pattern in _SAVE_LOCATION_PATTERNS:
        match = re.match(pattern, raw, flags=re.IGNORECASE)
        if not match:
            continue
        name = _clean_saved_location_name(match.group(1))
        compact = _normalize_text(name)
        if not compact or compact in {
            "location", "place", "point", "spot", "position", "here",
            "current location", "current place", "الموقع", "المكان", "النقطة"
        }:
            return None
        return name

    return None


def _strip_common_location_suffixes(text: str) -> str:
    """Remove suffixes added for OSM so saved-location matching sees the real name."""
    value = (text or "").strip()
    suffixes = [
        r",?\s*amman\s*,?\s*jordan\s*$",
        r",?\s*jordan\s*$",
        r",?\s*عمان\s*,?\s*الأردن\s*$",
        r",?\s*عمان\s*,?\s*الاردن\s*$",
        r",?\s*الأردن\s*$",
        r",?\s*الاردن\s*$",
    ]
    for suffix in suffixes:
        value = re.sub(suffix, "", value, flags=re.IGNORECASE).strip()
    return value or (text or "").strip()


def _saved_location_key(text: str) -> str:
    text = _strip_common_location_suffixes(text)
    text = _clean_prefixes(text)
    text = re.sub(r"\b(?:saved|local|location|place|point)\b", " ", text, flags=re.IGNORECASE)
    return _normalize_text(text)


def _saved_locations_path(filepath: Optional[str] = None) -> str:
    return filepath or SAVED_LOCATIONS_FILE


def _default_saved_locations_data() -> Dict[str, Any]:
    return {"version": 1, "locations": []}


def load_saved_locations(filepath: Optional[str] = None) -> Dict[str, Any]:
    """Load saved local GPS locations. Missing/empty files return an empty database."""
    path = _saved_locations_path(filepath)
    if not os.path.exists(path):
        return _default_saved_locations_data()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return _default_saved_locations_data()

    if isinstance(data, list):
        data = {"version": 1, "locations": data}
    if not isinstance(data, dict):
        return _default_saved_locations_data()
    if "locations" not in data or not isinstance(data["locations"], list):
        data["locations"] = []
    data.setdefault("version", 1)
    return data


def _write_saved_locations(data: Dict[str, Any], filepath: Optional[str] = None) -> None:
    path = _saved_locations_path(filepath)
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def save_current_location(
    name: str,
    current_position: Dict[str, Any],
    aliases=None,
    filepath: Optional[str] = None,
) -> Dict[str, Any]:
    """Save or update the drone's current GPS position in saved_locations.json."""
    clean_name = _clean_saved_location_name(name)
    if not clean_name:
        return {"ok": False, "error": "Saved location name is empty."}

    if not isinstance(current_position, dict):
        return {"ok": False, "error": "No current drone position is available."}

    try:
        lat = float(current_position["lat"])
        lon = float(current_position["lon"])
    except Exception:
        return {"ok": False, "error": "Current drone position does not contain valid lat/lon."}

    data = load_saved_locations(filepath)
    locations = data.setdefault("locations", [])
    now = datetime.now(timezone.utc).isoformat()

    alias_list = _dedupe(aliases or [])
    gps = {"lat": lat, "lon": lon}
    for key in ("alt_rel", "alt_msl", "yaw", "hdg"):
        if key in current_position and current_position.get(key) is not None:
            try:
                gps[key] = float(current_position[key])
            except Exception:
                pass

    new_key = _saved_location_key(clean_name)
    updated = False
    saved_entry = None

    for loc in locations:
        existing_name = str(loc.get("name", ""))
        if _saved_location_key(existing_name) == new_key:
            existing_aliases = _dedupe((loc.get("aliases") or []) + alias_list)
            loc.update({
                "name": clean_name,
                "aliases": existing_aliases,
                "gps": gps,
                "created_from": loc.get("created_from", "current_drone_position"),
                "created_at": loc.get("created_at", now),
                "updated_at": now,
            })
            saved_entry = loc
            updated = True
            break

    if saved_entry is None:
        saved_entry = {
            "name": clean_name,
            "aliases": alias_list,
            "gps": gps,
            "created_from": "current_drone_position",
            "created_at": now,
            "updated_at": now,
        }
        locations.append(saved_entry)

    # Keep the file stable/readable. Sort by normalized name.
    locations.sort(key=lambda loc: _saved_location_key(str(loc.get("name", ""))))
    _write_saved_locations(data, filepath)

    return {
        "ok": True,
        "updated": updated,
        "location": saved_entry,
        "filepath": _saved_locations_path(filepath),
    }


def _saved_match_score(query_key: str, candidate_key: str) -> float:
    if not query_key or not candidate_key:
        return 0.0
    if query_key == candidate_key:
        return 1.0
    if candidate_key in query_key or query_key in candidate_key:
        return 0.96
    return SequenceMatcher(None, query_key, candidate_key).ratio()


def resolve_saved_location(
    place_text: str,
    command: Optional[Dict[str, Any]] = None,
    filepath: Optional[str] = None,
    min_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Resolve a navigation phrase against saved_locations.json before using OSM."""
    data = load_saved_locations(filepath)
    locations = data.get("locations") or []
    if not locations:
        return {"ok": False, "error": "No saved locations yet.", "source": "saved_location"}

    raw_candidates = [place_text]
    if isinstance(command, dict):
        raw_candidates.extend([
            command.get("place_text"),
            command.get("extracted_location"),
            command.get("original_text"),
        ])
        raw_candidates.extend(command.get("osm_candidates") or [])

    expanded_queries = []
    for item in _dedupe(raw_candidates):
        item = str(item).strip()
        if not item:
            continue
        expanded_queries.append(item)
        expanded_queries.append(_strip_common_location_suffixes(item))
        expanded_queries.append(_clean_prefixes(_strip_common_location_suffixes(item)))

    query_keys = _dedupe([_saved_location_key(q) for q in expanded_queries])
    threshold = SAVED_LOCATION_MIN_MATCH_SCORE if min_score is None else float(min_score)

    best = None
    for loc in locations:
        name = str(loc.get("name", "")).strip()
        aliases = loc.get("aliases") or []
        candidate_names = _dedupe([name] + aliases)
        for candidate in candidate_names:
            candidate_key = _saved_location_key(candidate)
            for query_key in query_keys:
                score = _saved_match_score(query_key, candidate_key)
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "location": loc,
                        "matched_name": candidate,
                        "query_key": query_key,
                    }

    if not best or best["score"] < threshold:
        return {
            "ok": False,
            "error": "No saved location match.",
            "source": "saved_location",
            "best_score": 0.0 if not best else best["score"],
        }

    loc = best["location"]
    gps = loc.get("gps") or loc
    try:
        lat = float(gps["lat"])
        lon = float(gps["lon"])
    except Exception:
        return {"ok": False, "error": "Saved location has invalid GPS.", "source": "saved_location"}

    return {
        "ok": True,
        "lat": lat,
        "lon": lon,
        "display_name": loc.get("name", best["matched_name"]),
        "source": "saved_location",
        "matched_name": best["matched_name"],
        "match_score": best["score"],
        "saved_location": loc,
    }


def _is_chitchat(text: str) -> bool:
    compact = _normalize_text(text)
    return compact in {_normalize_text(p) for p in _CHITCHAT_PHRASES}


def _is_repeat_command(text: str) -> bool:
    compact = _normalize_text(text)
    return compact in {_normalize_text(p) for p in _REPEAT_PHRASES}


def _is_turn_around_command(text: str) -> bool:
    compact = _normalize_text(text)
    return any(_normalize_text(p) in compact for p in _TURN_AROUND_PHRASES)


def _first_number(text: str) -> Optional[float]:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _arabic_place_variants(place: str):
    """Return conservative Arabic spelling variants without losing the user's original text."""
    place = (place or "").strip()
    variants = [place]
    if not place:
        return variants

    # Common Arabic spelling difference: ابو / أبو
    if "ابو" in place:
        variants.append(place.replace("ابو", "أبو"))
    if "أبو" in place:
        variants.append(place.replace("أبو", "ابو"))

    # Light normalization for hamza forms; keep original first.
    variants.append(place.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا"))
    return _dedupe(variants)


def _local_amman_alias_candidates(place: str):
    """Small hand-written aliases for Amman names where LLM transliteration often fails."""
    text = (place or "").strip()
    compact = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    candidates = []

    # Abu Nuseir / Abu Nseir is commonly misspelled by LLMs and OSM providers.
    if "ابو نصير" in compact:
        if "مسجد" in compact:
            candidates.extend([
                "مسجد أبو نصير, عمان, الأردن",
                "مسجد ابو نصير, عمان, الأردن",
                "Abu Nuseir Mosque, Amman, Jordan",
                "Abu Nseir Mosque, Amman, Jordan",
            ])
        candidates.extend([
            "أبو نصير, عمان, الأردن",
            "ابو نصير, عمان, الأردن",
            "Abu Nuseir, Amman, Jordan",
            "Abu Nseir, Amman, Jordan",
        ])

    return _dedupe(candidates)


def _build_osm_candidates(original_text: str, extracted_location: str = "", osm_query: str = ""):
    """Build a prioritized OSM search list.

    Arabic-first behavior prevents bad LLM transliterations from being the first
    query sent to OSM.
    """
    original_text = (original_text or "").strip()
    cleaned_original = _clean_prefixes(original_text) if original_text else ""
    extracted_location = (extracted_location or "").strip()
    osm_query = (osm_query or "").strip()

    candidates = []

    # 1) If the user typed Arabic, trust that first.
    for place in _dedupe([cleaned_original, extracted_location]):
        if _has_arabic(place):
            for variant in _arabic_place_variants(place):
                candidates.extend(_local_amman_alias_candidates(variant))
                candidates.append(f"{variant}, عمان, الأردن")
                candidates.append(f"{variant}, Amman, Jordan")
                candidates.append(variant)

    # 2) Try exact extracted English/Arabic if any.
    for place in _dedupe([extracted_location, cleaned_original]):
        if place and not _has_arabic(place):
            if "amman" not in place.lower() and "jordan" not in place.lower():
                candidates.append(f"{place}, Amman, Jordan")
            candidates.append(place)

    # 3) LLM OSM-friendly query last.
    if osm_query:
        candidates.append(osm_query)

    return _dedupe(candidates)


def _fallback_control_parse(text: str) -> Optional[Dict[str, Any]]:
    """Small local parser for obvious hover-control commands.

    This runs before the LLM for clear hover commands so typos like
    "go write 10" become move-right, not turn/yaw.
    """
    lowered = (text or "").lower().strip()
    lowered = re.sub(r"\bwrite\b", "right", lowered)  # common voice/typing typo
    lowered = lowered.replace("infront", "in front")
    words = set(re.findall(r"[\w\u0600-\u06FF]+", lowered))
    value = _first_number(lowered)

    # Deterministic local handling: turn around = 180 degrees.
    if _is_turn_around_command(lowered):
        return {
            "type": "control",
            "control_action": "turn",
            "direction": "right",
            "degrees": 180.0,
            "parser": "fallback",
        }

    # turn / yaw commands
    if any(w in lowered for w in ["turn", "rotate", "yaw", "raw", "لف", "دور", "استدر"]):
        direction = None
        if "right" in lowered or words.intersection(_ARABIC_RIGHT):
            direction = "right"
        elif "left" in lowered or words.intersection(_ARABIC_LEFT):
            direction = "left"

        if direction:
            return {
                "type": "control",
                "control_action": "turn",
                "direction": direction,
                "degrees": value if value is not None else 15.0,
                "parser": "fallback",
            }

    # movement commands. "go right/left" means lateral movement, not yaw.
    movement_words = ["move", "go", "in front", "forward", "back", "backward", "behind", "تحرك", "امشي", "قدم", "رجع", "ورا"]
    has_direction_word = (
        "forward" in lowered or "in front" in lowered or
        "back" in lowered or "backward" in lowered or "behind" in lowered or
        "right" in lowered or "left" in lowered or
        bool(words.intersection(_ARABIC_FORWARD | _ARABIC_BACK | _ARABIC_RIGHT | _ARABIC_LEFT))
    )
    if any(w in lowered for w in movement_words) and has_direction_word:
        direction = None
        if "forward" in lowered or "in front" in lowered or words.intersection(_ARABIC_FORWARD):
            direction = "forward"
        elif "back" in lowered or "backward" in lowered or "behind" in lowered or words.intersection(_ARABIC_BACK):
            direction = "backward"
        elif "right" in lowered or words.intersection(_ARABIC_RIGHT):
            direction = "right"
        elif "left" in lowered or words.intersection(_ARABIC_LEFT):
            direction = "left"

        if direction:
            return {
                "type": "control",
                "control_action": "move",
                "direction": direction,
                "meters": value if value is not None else 5.0,
                "parser": "fallback",
            }

    # altitude commands
    # Examples: go up 5, ascend 5, climb 10, go down, descend, اطلع فوق 5, انزل 5
    altitude_words = [
        "up", "down", "higher", "lower", "ascend", "descend", "climb",
        "rise", "drop", "increase altitude", "decrease altitude", "altitude"
    ]
    has_altitude_word = any(w in lowered for w in altitude_words) or bool(words.intersection(_ARABIC_UP | _ARABIC_DOWN))
    if has_altitude_word:
        if (
            "up" in lowered or "higher" in lowered or "ascend" in lowered or
            "climb" in lowered or "rise" in lowered or "increase altitude" in lowered or
            bool(words.intersection(_ARABIC_UP))
        ):
            return {
                "type": "control",
                "control_action": "altitude",
                "mode": "up",
                "meters": value if value is not None else 5.0,
                "parser": "fallback",
            }
        if (
            "down" in lowered or "lower" in lowered or "descend" in lowered or
            "drop" in lowered or "decrease altitude" in lowered or
            bool(words.intersection(_ARABIC_DOWN))
        ):
            return {
                "type": "control",
                "control_action": "altitude",
                "mode": "down",
                "meters": value if value is not None else 5.0,
                "parser": "fallback",
            }

    if lowered in {"hover", "hold", "اثبت", "خليك", "وقف مكانك"}:
        return {"type": "control", "control_action": "hover", "parser": "fallback"}

    return None


def _fallback_parse(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    lowered = text.lower()

    if not text:
        return {"type": "empty"}
    save_name = _extract_save_location_name(text)
    if save_name:
        return {"type": "save_location", "name": save_name, "parser": "fallback"}
    if _is_chitchat(text):
        return {"type": "unknown", "error": "Not a supported drone command.", "parser": "local"}
    if lowered in _LAND_WORDS:
        return {"type": "land", "parser": "fallback"}
    if lowered in _EXIT_WORDS:
        return {"type": "exit", "parser": "fallback"}

    control = _fallback_control_parse(text)
    if control:
        return control

    place = _clean_prefixes(text)

    if "amman" not in place.lower() and "jordan" not in place.lower() and "عمان" not in place:
        osm_query = f"{place}, Amman, Jordan"
    else:
        osm_query = place

    osm_candidates = _build_osm_candidates(text, place, osm_query)
    return {
        "type": "navigate",
        "place_text": osm_candidates[0] if osm_candidates else osm_query,
        "extracted_location": place,
        "original_text": text,
        "osm_candidates": osm_candidates,
        "parser": "fallback",
    }


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a model response."""
    if not text:
        return None

    cleaned = text.strip()

    # Remove common markdown fences if the model ignores instructions.
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


def _call_hf_qwen(raw: str) -> Optional[Dict[str, Any]]:
    """Call hosted Qwen through Hugging Face Inference Providers."""
    if not HF_TOKEN:
        return None

    system_prompt = """
You are the NLP command parser for UrbanEye, a simulated autonomous drone project in Amman, Jordan.
Your job is NOT to fly the drone, NOT to invent GPS coordinates, and NOT to chat with the user.
Your only job is to convert the user's Arabic/English command into strict JSON.

Return JSON only. No markdown. No explanation. No friendly replies.

Allowed actions:
1. navigate
2. save_location
3. land
4. exit
5. control
6. unknown

If the user greets you, asks how you are, asks a general question, or writes anything that is not a supported drone command, return:
{"action":"unknown","reason":"not a supported drone command"}
Do not answer the greeting or question.

Save-location rules:
- If the user wants to save/store/remember the current location, action = "save_location".
- Extract only the requested saved name into location_name. Do not invent coordinates.
- Examples: "save this location as checkpoint one", "save new location with name sports entrance", "احفظ موقعي باسم نقطة التفتيش".

Navigation rules:
- If the user wants to go/fly/navigate/روح/وديني/اذهب to a place, action = "navigate".
- Extract only the place name, not the command words.
- Convert the place into an OpenStreetMap-friendly search query.
- If the command is Arabic, keep the Arabic place name in extracted_location exactly as the user wrote it. Do not force bad transliterations.
- If the place is likely in Amman/Jordan, add "Amman, Jordan" to the OSM query.
- For Arabic neighborhood/place names in Amman, prefer known spellings such as Abu Nuseir / Abu Nseir, not random spellings.
- Do not output latitude or longitude.

Hover-control rules:
- Use action = "control" for small manual adjustments after the drone has arrived and is hovering.
- The parser does not decide whether the drone is hovering. The navigation code will enforce that.
- For turning/yaw: use control_action = "turn", direction = "left" or "right", degrees = number. If the user writes "raw" near a turn command, treat it as a typo for "yaw".
- IMPORTANT: "turn around", "turn back", "u-turn", "لف ورا", "استدر", and similar phrases mean a 180-degree turn.
- IMPORTANT: "go right", "move right", "go left", "move left", "go forward", and "go back" are movement commands, not yaw commands.
- For small position movement: use control_action = "move", direction = "forward", "backward", "left", or "right", meters = number.
- For altitude changes such as go up, go down, ascend, descend, climb, lower, اطلع, اصعد, انزل, or اهبط: use control_action = "altitude", mode = "up", "down", or "set", meters or agl = number.
- For hold/hover in place: use control_action = "hover".
- If the user says "a little" or "شوي" and gives no number, use 5 meters for movement or 15 degrees for turning.
- Never use large defaults.

Required JSON examples:
{"action":"save_location","location_name":"checkpoint one"}
{"action":"save_location","location_name":"نقطة التفتيش"}
{"action":"navigate","extracted_location":"المدينة الرياضية","osm_query":"Sports City, Amman, Jordan"}
{"action":"navigate","extracted_location":"دوار المدينة الرياضية","osm_query":"Sports City Circle, Amman, Jordan"}
{"action":"navigate","extracted_location":"الجامعة الأردنية","osm_query":"University of Jordan, Amman, Jordan"}
{"action":"navigate","extracted_location":"مسجد ابو نصير","osm_query":"Abu Nuseir Mosque, Amman, Jordan"}
{"action":"control","control_action":"turn","direction":"right","degrees":15}
{"action":"control","control_action":"turn","direction":"left","degrees":20}
{"action":"control","control_action":"turn","direction":"right","degrees":180}
{"action":"control","control_action":"move","direction":"forward","meters":5}
{"action":"control","control_action":"move","direction":"right","meters":3}
{"action":"control","control_action":"move","direction":"right","meters":10}
{"action":"control","control_action":"altitude","mode":"up","meters":5}
{"action":"control","control_action":"altitude","mode":"down","meters":5}
{"action":"control","control_action":"altitude","mode":"set","agl":30}
{"action":"control","control_action":"hover"}
{"action":"land"}
{"action":"exit"}
{"action":"unknown","reason":"not a supported drone command"}
""".strip()

    user_prompt = f"Parse this drone command into JSON only:\n{raw}"

    try:
        client = InferenceClient(provider=HF_PROVIDER, api_key=HF_TOKEN)
        completion = client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        content = completion.choices[0].message.content
        return _extract_json_object(content)

    except Exception as e:
        print(f"  [LLM WARNING] Hugging Face Qwen call failed: {e}")
        return None


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _normalize_control(llm_json: Dict[str, Any]) -> Dict[str, Any]:
    action = str(llm_json.get("control_action", "")).strip().lower()
    direction = str(llm_json.get("direction", "")).strip().lower()
    mode = str(llm_json.get("mode", "")).strip().lower()

    if action == "turn":
        if direction not in {"left", "right"}:
            return {"type": "unknown", "error": "Turn command missing left/right direction.", "parser": "qwen", "raw_llm": llm_json}
        return {
            "type": "control",
            "control_action": "turn",
            "direction": direction,
            "degrees": _as_float(llm_json.get("degrees"), 15.0),
            "parser": "qwen",
            "raw_llm": llm_json,
        }

    if action == "move":
        if direction not in {"forward", "backward", "left", "right"}:
            return {"type": "unknown", "error": "Move command missing direction.", "parser": "qwen", "raw_llm": llm_json}
        return {
            "type": "control",
            "control_action": "move",
            "direction": direction,
            "meters": _as_float(llm_json.get("meters"), 5.0),
            "parser": "qwen",
            "raw_llm": llm_json,
        }

    if action == "altitude":
        if mode not in {"up", "down", "set"}:
            return {"type": "unknown", "error": "Altitude command missing mode.", "parser": "qwen", "raw_llm": llm_json}
        result = {
            "type": "control",
            "control_action": "altitude",
            "mode": mode,
            "parser": "qwen",
            "raw_llm": llm_json,
        }
        if mode == "set":
            result["agl"] = _as_float(llm_json.get("agl"), 30.0)
        else:
            result["meters"] = _as_float(llm_json.get("meters"), 5.0)
        return result

    if action == "hover":
        return {"type": "control", "control_action": "hover", "parser": "qwen", "raw_llm": llm_json}

    return {"type": "unknown", "error": "Unsupported control action.", "parser": "qwen", "raw_llm": llm_json}


# ---------------------------------------------------------------------------
# Public API used by nav code
# ---------------------------------------------------------------------------
def parse_terminal_command(raw: str) -> Dict[str, Any]:
    """
    Main parser used by the nav code.

    Returns one of:
    - {"type": "empty"}
    - {"type": "land"}
    - {"type": "exit"}
    - {"type": "save_location", "name": "local name"}
    - {"type": "navigate", "place_text": "OSM query here"}
    - {"type": "control", "control_action": ...}
    - {"type": "repeat_control"}
    - {"type": "unknown", "error": ...}
    """
    text = (raw or "").strip()
    lowered = text.lower()

    if not text:
        return {"type": "empty"}

    save_name = _extract_save_location_name(text)
    if save_name:
        return {"type": "save_location", "name": save_name, "parser": "local"}

    # Do not let the model chat. Greetings/questions are not drone commands.
    if _is_chitchat(text):
        return {"type": "unknown", "error": "Not a supported drone command.", "parser": "local"}

    # Repeat is intentionally local/stateless for the LLM. The nav code decides
    # whether there is a safe previous hover-control command to repeat.
    if _is_repeat_command(text):
        return {"type": "repeat_control", "parser": "local"}

    # Fast local handling for obvious safety/menu commands.
    if lowered in _LAND_WORDS:
        return {"type": "land", "parser": "local"}
    if lowered in _EXIT_WORDS:
        return {"type": "exit", "parser": "local"}

    # Fast local hover-control parsing for obvious commands. This avoids LLM
    # mistakes like treating "go right 10" as a yaw/turn command.
    local_control = _fallback_control_parse(text)
    if local_control:
        local_control["parser"] = "local"
        return local_control

    llm_json = _call_hf_qwen(text)
    if not llm_json:
        return _fallback_parse(text)

    action = str(llm_json.get("action", "")).strip().lower()

    if action == "save_location":
        save_name = _clean_saved_location_name(
            str(
                llm_json.get("location_name")
                or llm_json.get("name")
                or llm_json.get("saved_name")
                or ""
            )
        )
        if not save_name:
            return {"type": "unknown", "error": "Save-location command missing a location name.", "parser": "qwen", "raw_llm": llm_json}
        return {"type": "save_location", "name": save_name, "parser": "qwen", "raw_llm": llm_json}

    if action == "land":
        return {"type": "land", "parser": "qwen", "raw_llm": llm_json}

    if action == "exit":
        return {"type": "exit", "parser": "qwen", "raw_llm": llm_json}

    if action == "control":
        return _normalize_control(llm_json)

    if action == "navigate":
        extracted_location = str(llm_json.get("extracted_location", "")).strip()
        osm_query = str(llm_json.get("osm_query", "")).strip()

        # Safety fallback if model forgot osm_query.
        if not osm_query:
            osm_query = extracted_location or _clean_prefixes(text)
            if "amman" not in osm_query.lower() and "jordan" not in osm_query.lower() and "عمان" not in osm_query:
                osm_query = f"{osm_query}, Amman, Jordan"

        osm_candidates = _build_osm_candidates(text, extracted_location, osm_query)

        return {
            "type": "navigate",
            "place_text": osm_candidates[0] if osm_candidates else osm_query,
            "extracted_location": extracted_location,
            "original_text": text,
            "osm_candidates": osm_candidates,
            "parser": "qwen",
            "raw_llm": llm_json,
        }

    reason = str(llm_json.get("reason", "Command not supported.")).strip()
    return {
        "type": "unknown",
        "error": reason,
        "parser": "qwen",
        "raw_llm": llm_json,
    }


def _query_nominatim_once(query: str) -> Dict[str, Any]:
    headers = {"User-Agent": NOMINATIM_USER_AGENT}
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1,
        "countrycodes": DEFAULT_COUNTRY_CODES,
        "bounded": 1,
        "viewbox": DEFAULT_VIEWBOX,
    }

    response = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=12)
    response.raise_for_status()
    results = response.json()
    return results[0] if results else None


def resolve_place_with_osm(place_text: str, command: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Send candidate OSM queries to OpenStreetMap Nominatim.

    Behavior:
    - If the user typed Arabic, search the Arabic wording first.
    - Then try Arabic + Amman/Jordan variants.
    - Then try local Amman aliases for known tricky names.
    - Then try the LLM English query.

    Returns:
    - {"ok": True, "lat": float, "lon": float, "display_name": str}
    or
    - {"ok": False, "error": str}
    """
    base_query = (place_text or "").strip()
    if not base_query:
        return {"ok": False, "error": "Empty OSM query."}

    candidates = []
    if isinstance(command, dict):
        candidates.extend(command.get("osm_candidates") or [])
        candidates.extend(_build_osm_candidates(
            command.get("original_text", ""),
            command.get("extracted_location", ""),
            base_query,
        ))
    candidates.append(base_query)
    candidates = _dedupe(candidates)

    tried = []
    last_error = None

    for query in candidates:
        tried.append(query)
        try:
            best = _query_nominatim_once(query)
        except Exception as e:
            last_error = str(e)
            continue

        if not best:
            continue

        try:
            return {
                "ok": True,
                "lat": float(best["lat"]),
                "lon": float(best["lon"]),
                "display_name": best.get("display_name", query),
                "osm_query": query,
                "tried_queries": tried,
            }
        except Exception:
            last_error = "OpenStreetMap returned an invalid result."
            continue

    if last_error:
        return {
            "ok": False,
            "error": f"OSM lookup failed after trying {len(tried)} queries. Last error: {last_error}",
            "tried_queries": tried,
        }

    return {
        "ok": False,
        "error": f"Place not found on OpenStreetMap after trying {len(tried)} queries.",
        "tried_queries": tried,
    }
