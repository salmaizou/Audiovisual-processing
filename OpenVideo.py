"""Tâche 0.1 : ouvrir et lire la vidéo. Quitter avec la touche q."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def play_video(video_path: str | Path) -> None:
    """Lit la vidéo à sa cadence nominale dans une fenêtre OpenCV."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    delay_ms = max(1, round(1000 / fps)) if fps > 0 else 25
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            cv2.imshow("Lecture de la vidéo - q pour quitter", frame)
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", default="Pub_C+_352_288_1.mp4")
    play_video(parser.parse_args().video)
