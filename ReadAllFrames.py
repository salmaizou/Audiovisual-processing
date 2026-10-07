"""Tâche 0.3 : lire toutes les images dans une matrice NumPy.

La matrice retournée a la forme (nombre_images, hauteur, largeur, 3), en BGR.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_all_frames(video_path: str | Path) -> np.ndarray:
    """Retourne toutes les images de la vidéo dans un seul tableau NumPy."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {video_path}")

    expected_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if expected_count <= 0 or height <= 0 or width <= 0:
        capture.release()
        raise RuntimeError("Métadonnées vidéo invalides.")

    # Pré-allocation : évite de dupliquer toutes les images lors de np.stack().
    video_matrix = np.empty((expected_count, height, width, 3), dtype=np.uint8)
    frame_count = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_count == expected_count:
                # Rare, mais protège si le conteneur annonce un nombre erroné.
                raise RuntimeError("La vidéo contient plus d'images que prévu.")
            video_matrix[frame_count] = frame
            frame_count += 1
    finally:
        capture.release()

    if frame_count == 0:
        raise RuntimeError("Aucune image n'a pu être lue.")
    return video_matrix[:frame_count]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", default="Pub_C+_352_288_1.mp4")
    parser.add_argument(
        "--save",
        metavar="FICHIER.npy",
        default="video_matrix.npy",
        help="Fichier de sortie de la matrice (défaut : video_matrix.npy).",
    )
    parser.add_argument(
        "--print-frame-matrix",
        type=int,
        metavar="INDEX",
        help="Affiche dans le terminal la matrice BGR complète d'une image donnée.",
    )
    args = parser.parse_args()

    video_matrix = read_all_frames(args.video)
    print(f"Matrice vidéo : shape={video_matrix.shape}, dtype={video_matrix.dtype}, format=BGR")
    np.save(args.save, video_matrix)
    print(f"Matrice complète enregistrée dans : {args.save}")
    if args.print_frame_matrix is not None:
        if not 0 <= args.print_frame_matrix < len(video_matrix):
            raise IndexError(f"Indice invalide : choisir entre 0 et {len(video_matrix) - 1}.")
        print(f"\nMatrice BGR complète de l'image {args.print_frame_matrix} :")
        print(video_matrix[args.print_frame_matrix])
