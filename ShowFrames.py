"""Tâche 0.4 : récupérer et afficher la matrice d'une image choisie."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_frame(video_path: str | Path, frame_index: int) -> np.ndarray:
    """Lit une image d'indice 0-based et retourne sa matrice BGR uint8."""
    if frame_index < 0:
        raise ValueError("L'indice doit être positif ou nul.")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success or frame is None:
            raise IndexError(f"Image {frame_index} introuvable dans la vidéo.")
        return frame
    finally:
        capture.release()


def show_frame(frame: np.ndarray, title: str) -> None:
    """Affiche une matrice BGR ; fermer la fenêtre ou appuyer sur une touche."""
    cv2.imshow(title, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", default="Pub_C+_352_288_1.mp4")
    parser.add_argument("--index", type=int, default=0, help="Indice de l'image (base 0).")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="N'ouvre pas la fenêtre (affiche seulement les informations de la matrice).",
    )
    args = parser.parse_args()

    image_matrix = read_frame(args.video, args.index)
    print(f"Image {args.index} : shape={image_matrix.shape}, dtype={image_matrix.dtype}, format=BGR")
    print("Un pixel s'écrit [bleu, vert, rouge]. Exemple, pixel (0, 0) :", image_matrix[0, 0])
    if not args.no_show:
        show_frame(image_matrix, f"Image {args.index}")
