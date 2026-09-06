# UrbanEye

**An autonomous drone for urban traffic monitoring, flown by talking to it — in Arabic or English.**

Tell the drone where to go in plain language. It works out what you meant, finds the real
coordinates of the place you named, flies there autonomously through a 3D reconstruction of
Amman, and analyses what its camera sees.

<p align="center">
  <a href="https://youtu.be/X64fSH44iSk">
    <img src="https://img.youtube.com/vi/X64fSH44iSk/maxresdefault.jpg" alt="Watch the UrbanEye demo" width="800">
  </a>
</p>

<p align="center"><b><a href="https://youtu.be/X64fSH44iSk">▶ Watch the full demo</a></b></p>

---

## What it does

Say **"روح على مول مكة"** or **"fly to the Roman Theatre"** — typed or spoken — and the drone:

1. **Transcribes** the command if it was spoken (Whisper large-v3)
2. **Understands** it — what action, and which destination (Qwen2.5)
3. **Grounds** the destination into real GPS coordinates (saved locations, then OpenStreetMap)
4. **Flies** there autonomously via PX4, avoiding obstacles on the way
5. **Watches** the scene with a vision-language model that reads the camera feed
6. **Reports** everything live on a web dashboard

It also accepts follow-up commands while hovering — *"turn right 90 degrees"*, *"go up 10 metres"*,
*"land"* — and can remember a spot for later: *"save this location as checkpoint one"*.

---

## Features

**Bilingual natural-language control**
Arabic, English, and mixed commands. Colloquial Jordanian phrasing is supported
(*روح على*, *وديني*, *اطلع على*), not just formal Arabic.

**Voice or text**
Speak into the dashboard or type. Both paths reach the same parser.

**Real-world destination grounding**
Place names are resolved to actual coordinates via a saved-location database first, then
OpenStreetMap Nominatim, restricted to the Amman area.

**Vision-language scene analysis**
A drone camera frame is sent to Qwen2.5-VL, which describes what it sees and flags traffic
violations, restricted to a known set of plate numbers so it cannot invent one.

**Autonomous flight with obstacle avoidance**
PX4 handles stabilisation and flight; a forward distance sensor triggers a brake-climb-resume
manoeuvre when something is in the way.

**Live web dashboard**
Telemetry, camera feed, map, and a command console in one page.

---

## How it works

```
        Operator command  (Arabic / English, typed or spoken)
                            |
                            v
        +---------------------------------------+
        |  Whisper large-v3   speech -> text     |
        +---------------------------------------+
                            |
                            v
        +---------------------------------------+
        |  Qwen2.5-7B         what action?       |
        |                     which place?       |
        +---------------------------------------+
                            |
                            v
        +---------------------------------------+
        |  Saved locations  ->  OpenStreetMap    |
        |  place name  ->  latitude, longitude   |
        +---------------------------------------+
                            |
                            v
        +---------------------------------------+
        |  PX4 + MAVLink      autonomous flight  |
        |  distance sensor    obstacle avoidance |
        +---------------------------------------+
                            |
                            v
        +---------------------------------------+
        |  Qwen2.5-VL         scene analysis     |
        |  from the drone camera feed            |
        +---------------------------------------+
                            |
                            v
                    Live web dashboard
```

---

## Built with

| Layer | Technology |
|---|---|
| Speech recognition | Whisper large-v3 (Hugging Face) |
| Language understanding | Qwen2.5-7B-Instruct |
| Vision-language analysis | Qwen2.5-VL |
| Geocoding | OpenStreetMap Nominatim |
| Flight control | PX4 Autopilot (SITL), MAVLink, pymavlink |
| Simulation | Unreal Engine + Cesium (3D Amman), AirSim / Colosseum |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |

---

## Repository structure

| File | Purpose |
|---|---|
| `urbaneye_web_backend_voice_v56.py` | FastAPI backend — API, command routing, telemetry |
| `drone_nav_final.py` | Flight control: takeoff, navigation, hover control, landing |
| `llm_resolver.py` | Command parsing and destination resolution |
| `vl_resolver.py` | Vision-language scene and violation analysis |
| `speech_recognition_hf.py` | Speech-to-text |
| `object_avoidance_final.py` | Reactive obstacle avoidance |
| `unreal_camera_reader.py` | Camera frames from the simulator |
| `UrbanEye_dashboard_voice_v56.html` | Web dashboard |
| `saved_locations.json` | Operator-saved points of interest |
| `Run_UrbanEye_Backend_Website_v56.bat` | Launcher — starts PX4, backend, and dashboard |

---

## Getting started

### Prerequisites

- Unreal Engine with the Cesium plugin and AirSim/Colosseum
- PX4 Autopilot running in SITL (via WSL on Windows)
- Python 3.10+
- A Hugging Face access token

### Install

```bash
git clone https://github.com/SulimanAbusamak/urbaneye-drone.git
cd urbaneye-drone
pip install -r requirements.txt
```

### Configure

```bash
set HF_TOKEN=your_token_here
```

The launcher sets the rest:

| Variable | Default in the launcher |
|---|---|
| `HF_STT_MODEL` | `openai/whisper-large-v3` |
| `URBANEYE_VL_MODEL` | `Qwen/Qwen2.5-VL-72B-Instruct` |
| `URBANEYE_VL_ENABLE` | `1` |
| `URBANEYE_VL_PLATE_MIN_CONFIDENCE` | `70` |

### Run

Start Unreal with the Amman map, then:

```
Run_UrbanEye_Backend_Website_v56.bat
```

This starts PX4, launches the backend on port 8000, and opens the dashboard.

---

## Notes

This project runs entirely in simulation. The drone, the city, and the camera feed are all
inside Unreal Engine — no physical aircraft is involved.

An earlier version of the detection pipeline used YOLO with EasyOCR for licence-plate reading.
It was superseded by the vision-language approach and disabled in the final configuration, so
it is not included here.

---

## Author

**Suliman Abusamak** — Applied Science Private University, Jordan
Graduation project, 2026.
