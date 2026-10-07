"""Partie edges 2 : détection de contours Canny sur une image de la vidéo.

Les trois couples de seuils demandés sont calculés : (20, 60), (50, 150)
et (100, 200).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_video_frame(video_path: str | Path, frame_index: int) -> np.ndarray:
    """Lit une seule image BGR de la vidéo (indice commençant à 0)."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success or frame is None:
            raise IndexError(f"Image {frame_index} introuvable.")
        return frame
    finally:
        capture.release()


def canny_edges(frame: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
    """Retourne la carte Canny avec fond blanc et contours noirs."""
    if low_threshold >= high_threshold:
        raise ValueError("Le seuil bas doit être inférieur au seuil haut.")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges_white = cv2.Canny(gray, low_threshold, high_threshold)
    return cv2.bitwise_not(edges_white)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", default="Pub_C+_352_288_1.mp4")
    parser.add_argument("--index", type=int, default=100, help="Indice de l'image à traiter.")
    parser.add_argument("--output-dir", default="results/edges")
    parser.add_argument("--show", action="store_true", help="Affiche les résultats dans des fenêtres.")
    args = parser.parse_args()

    frame = read_video_frame(args.video, args.index)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for low, high in ((20, 60), (50, 150), (100, 200)):
        edges = canny_edges(frame, low, high)
        output_path = output_dir / f"frame_{args.index:04d}_canny_{low}_{high}.png"
        cv2.imwrite(str(output_path), edges)
        print(f"Canny ({low}, {high}) : {output_path}")
        if args.show:
            cv2.imshow(f"Canny ({low}, {high})", edges)

    if args.show:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
