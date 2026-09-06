"""
unreal_camera_reader.py

UrbanEye Unreal RGB camera bridge.

Reads the PNG frame that Unreal exports from:
    SceneCaptureComponent2D -> RT_UrbanEye_RGB -> D:\\UrbanEyeCamera\\urbaneye_latest.png

This file does NOT run YOLO and does NOT control the drone.
It only keeps the latest Unreal RGB frame ready for the navigation/model code.

Requirements:
    pip install opencv-python numpy
"""

import shutil
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np


class UnrealCameraReader(threading.Thread):
    def __init__(
        self,
        frame_path: str = r"D:\UrbanEyeCamera\urbaneye_latest.png",
        poll_interval: float = 0.05,
        stale_after: float = 3.0,
    ):
        super().__init__(daemon=True)
        self.frame_path = Path(frame_path)
        self.temp_copy_path = self.frame_path.with_name("_urbaneye_python_read_copy.png")
        self.poll_interval = float(poll_interval)
        self.stale_after = float(stale_after)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._frame: Optional[np.ndarray] = None
        self._last_mtime: Optional[float] = None
        self._last_update_time: float = 0.0
        self._frames_read: int = 0
        self._last_error: Optional[str] = None

    def stop(self) -> None:
        self._stop_event.set()

    def get_latest_frame(self, copy: bool = True) -> Optional[np.ndarray]:
        """Return the newest BGR OpenCV frame, or None if no frame is ready."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy() if copy else self._frame

    def has_fresh_frame(self) -> bool:
        with self._lock:
            if self._frame is None:
                return False
            return (time.time() - self._last_update_time) <= self.stale_after

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            has_frame = self._frame is not None
            age = (time.time() - self._last_update_time) if has_frame else None
            h = int(self._frame.shape[0]) if has_frame else 0
            w = int(self._frame.shape[1]) if has_frame else 0
            return {
                "has_frame": has_frame,
                "fresh": bool(has_frame and age is not None and age <= self.stale_after),
                "age_sec": float(age) if age is not None else 9999.0,
                "frames_read": int(self._frames_read),
                "width": w,
                "height": h,
                "frame_path": str(self.frame_path),
                "last_error": self._last_error,
            }

    def _safe_read_png(self) -> Optional[np.ndarray]:
        """
        Unreal can be writing the PNG while Python is reading.
        Copy first, then read the copy. This avoids partial/locked reads.
        """
        try:
            if not self.frame_path.exists():
                return None

            shutil.copyfile(self.frame_path, self.temp_copy_path)
            img = cv2.imread(str(self.temp_copy_path), cv2.IMREAD_COLOR)

            if img is None or img.size == 0:
                return None

            return img

        except Exception as e:
            with self._lock:
                self._last_error = str(e)
            return None

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                if not self.frame_path.exists():
                    time.sleep(self.poll_interval)
                    continue

                mtime = self.frame_path.stat().st_mtime

                # Only reload when Unreal writes a new PNG.
                if self._last_mtime is None or mtime != self._last_mtime:
                    img = self._safe_read_png()

                    if img is not None:
                        with self._lock:
                            self._frame = img
                            self._last_mtime = mtime
                            self._last_update_time = time.time()
                            self._frames_read += 1
                            self._last_error = None

            except Exception as e:
                with self._lock:
                    self._last_error = str(e)

            time.sleep(self.poll_interval)


if __name__ == "__main__":
    # Standalone test:
    # 1. Press Play in Unreal.
    # 2. Make sure Unreal is exporting D:\\UrbanEyeCamera\\urbaneye_latest.png.
    # 3. Run: python unreal_camera_reader.py
    reader = UnrealCameraReader()
    reader.start()

    print("[UrbanEye Camera] Reader started. Press q to quit.")

    while True:
        frame = reader.get_latest_frame()

        if frame is not None:
            status = reader.get_status()
            display = frame.copy()
            cv2.putText(
                display,
                f"frames={status['frames_read']} age={status['age_sec']:.1f}s",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("UrbanEye Unreal RGB Camera", display)

        key = cv2.waitKey(100) & 0xFF
        if key == ord("q"):
            break

    reader.stop()
    cv2.destroyAllWindows()
    print("[UrbanEye Camera] Closed.")
