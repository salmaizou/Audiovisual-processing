"""Partie edges 1 : gradient de Sobel d'une image extraite de la vidéo.

Exemple : .venv/bin/python SobelGradient.py --index 100 --show
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


def sobel_gradient(frame: np.ndarray, kernel_size: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Retourne l'image grise et la magnitude Sobel normalisée."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    magnitude_u8 = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Convention demandée : fond blanc, contours noirs.
    edges_black = cv2.bitwise_not(magnitude_u8)
    return gray, edges_black


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", default="Pub_C+_352_288_1.mp4")
    parser.add_argument("--index", type=int, default=100, help="Indice de l'image à traiter.")
    parser.add_argument("--kernel-size", type=int, choices=(3, 5, 7), default=3)
    parser.add_argument("--output-dir", default="results/edges")
    parser.add_argument("--show", action="store_true", help="Affiche les résultats dans des fenêtres.")
    args = parser.parse_args()

    frame = read_video_frame(args.video, args.index)
    gray, magnitude = sobel_gradient(frame, args.kernel_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gray_path = output_dir / f"frame_{args.index:04d}_gray.png"
    gradient_path = output_dir / f"frame_{args.index:04d}_sobel_k{args.kernel_size}.png"
    cv2.imwrite(str(gray_path), gray)
    cv2.imwrite(str(gradient_path), magnitude)
    print(f"Image grise : {gray_path}")
    print(f"Contours Sobel (fond blanc, contours noirs) : {gradient_path}")

    if args.show:
        cv2.imshow("Image grise", gray)
        cv2.imshow("Gradient de Sobel", magnitude)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
