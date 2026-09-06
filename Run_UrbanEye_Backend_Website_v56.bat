@echo off
title UrbanEye Backend + Website v56
cd /d "%~dp0"

echo ==================================================
echo              UrbanEye v56 Launcher
echo ==================================================
echo Final demo setup: AI detection pipeline only.
echo YOLO / EasyOCR backup is disabled and not started.
echo This keeps drone_nav_final.py unchanged.
echo.
echo Website/backend path: %~dp0
echo.

if exist __pycache__ rmdir /s /q __pycache__
set PYTHONDONTWRITEBYTECODE=1

rem Speech-to-text
set URBANEYE_STT_DEBUG=1
set HF_STT_MODEL=openai/whisper-large-v3

rem AI detection settings
set URBANEYE_VL_ENABLE=1
set URBANEYE_VL_MODEL=Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud
set URBANEYE_YOLO_BACKUP_MODE=never
set URBANEYE_VL_PLATE_MIN_CONFIDENCE=70
set URBANEYE_VL_MAX_IMAGE_WIDTH=960
set URBANEYE_VL_JPEG_QUALITY=65
set URBANEYE_VL_MAX_TOKENS=300

rem Website speed display: smoothed GPS estimate clamped around the configured PX4 cruise speed.
set URBANEYE_SPEED_DISPLAY_MODE=smooth_clamped

echo Starting UrbanEye PX4 launcher...
if exist "Start_UrbanEye_PX4_v8.bat" (
    start "UrbanEye PX4" cmd /k call "%~dp0Start_UrbanEye_PX4_v8.bat"
) else (
    echo WARNING: Start_UrbanEye_PX4_v8.bat not found in this folder.
)

timeout /t 2 /nobreak >nul

echo Starting UrbanEye backend v56...
start "UrbanEye Backend v56" cmd /k python -m uvicorn urbaneye_web_backend_voice_v56:app --host 127.0.0.1 --port 8000

timeout /t 3 /nobreak >nul

echo Opening UrbanEye dashboard v56...
start "" "%~dp0UrbanEye_dashboard_voice_v56.html"
