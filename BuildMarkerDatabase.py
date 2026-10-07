"""Construire une base de marqueurs ORB pour une séquence vidéo de référence.

Exemple :
    .venv/bin/python BuildMarkerDatabase.py Pub_C+_352_288_1.mp4 \
        --name pub_cplus --start 0 --end 250 --step 25 --output marker_database
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    """Lit une image de la vidéo déjà ouverte."""
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = capture.read()
    if not success or frame is None:
        raise IndexError(f"Image {frame_index} introuvable.")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reference_video",
        nargs="?",
        default="SpaceCowboys.mp4",
        help="Vidéo ou extrait correspondant à la séquence à reconnaître.",
    )
    parser.add_argument("--name", default="pub_cplus", help="Nom de la séquence, par exemple pub_cplus.")
    parser.add_argument("--start", type=int, default=0, help="Première image de référence (incluse).")
    parser.add_argument("--end", type=int, help="Dernière image de référence (exclue, défaut : fin).")
    parser.add_argument("--step", type=int, default=25, help="Une image marqueur toutes les N images.")
    parser.add_argument("--max-features", type=int, default=1000, help="Nombre maximal de points ORB par image.")
    parser.add_argument("--output", default="marker_database", help="Nouveau dossier de la base de marqueurs.")
    args = parser.parse_args()

    if args.start < 0 or args.step <= 0 or args.max_features <= 0:
        raise ValueError("start doit être >= 0 ; step et max-features doivent être > 0.")

    output_dir = Path(args.output)
    if output_dir.exists():
        raise FileExistsError(f"Le dossier existe déjà : {output_dir}. Choisissez un autre --output.")

    capture = cv2.VideoCapture(args.reference_video)
    if not capture.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir : {args.reference_video}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    end = frame_count if args.end is None else args.end
    if not args.start < end <= frame_count:
        capture.release()
        raise ValueError(f"Intervalle invalide : 0 <= start < end <= {frame_count}.")

    output_dir.mkdir(parents=True)
    descriptors_dir = output_dir / "descriptors"
    descriptors_dir.mkdir()
    orb = cv2.ORB_create(nfeatures=args.max_features)
    markers: list[dict[str, int | float | str]] = []

    try:
        for frame_index in range(args.start, end, args.step):
            frame = read_frame(capture, frame_index)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, descriptors = orb.detectAndCompute(gray, None)
            if descriptors is None or len(descriptors) < 2:
                continue
            filename = f"marker_{frame_index:06d}.npy"
            np.save(descriptors_dir / filename, descriptors)
            markers.append(
                {
                    "frame_index": frame_index,
                    "time_seconds": frame_index / fps if fps else 0.0,
                    "descriptor_file": f"descriptors/{filename}",
                    "descriptor_count": int(len(descriptors)),
                }
            )
    finally:
        capture.release()

    if not markers:
        raise RuntimeError("Aucun descripteur ORB n'a été trouvé dans cet intervalle.")

    manifest = {
        "format_version": 1,
        "sequence_name": args.name,
        "reference_video": str(Path(args.reference_video).resolve()),
        "fps": fps,
        "reference_start_frame": args.start,
        "reference_end_frame": end,
        "sample_step": args.step,
        "feature": "ORB",
        "markers": markers,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Base créée : {output_dir}")
    print(f"Séquence : {args.name} ; marqueurs ORB : {len(markers)}")
    print(f"Manifest : {manifest_path}")


if __name__ == "__main__":
    main()
