"""Tâche 0.2 : afficher les métadonnées d'une vidéo."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def video_info(video_path: str | Path) -> dict[str, int | float]:
    """Retourne les informations nécessaires avant la segmentation."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        return {
            "nombre_d_images": frame_count,
            "largeur_pixels": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "hauteur_pixels": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": fps,
            "duree_secondes": frame_count / fps if fps else 0.0,
        }
    finally:
        capture.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", default="Pub_C+_352_288_1.mp4")
    info = video_info(parser.parse_args().video)
    print("===== Informations vidéo =====")
    for key, value in info.items():
        print(f"{key:20}: {value}")
