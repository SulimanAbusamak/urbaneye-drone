@echo off
title UrbanEye PX4 SITL
cd /d C:\Users\User

echo ==========================================================
echo                UrbanEye PX4 SITL Launcher
echo ==========================================================
echo Windows folder:
cd
echo.
echo This follows your manual steps:
echo   1. Open CMD at C:\Users\User
echo   2. Type wsl
echo   3. cd ~/PX4-Autopilot/
echo   4. make px4_sitl none
echo.
echo NOTE: No 10-second wait after starting WSL.
echo ==========================================================
echo.

wsl bash -ic "echo '[UrbanEye] WSL started.'; cd ~/PX4-Autopilot/ || { echo '[UrbanEye ERROR] Could not find ~/PX4-Autopilot/'; echo '[UrbanEye] Type pwd and ls to debug.'; exec bash; }; echo '[UrbanEye] Current WSL folder:'; pwd; echo '[UrbanEye] Running: make px4_sitl none'; make px4_sitl none; echo; echo '[UrbanEye] PX4 stopped or failed. Keep this window open for logs.'; exec bash"

echo.
echo WSL/PX4 command ended. If this was unexpected, read the messages above.
pause
