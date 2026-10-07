"""Rechercher les marqueurs d'une base ORB dans une vidéo cible.

Exemple :
    .venv/bin/python RecognizeSequence.py marker_database Pub_C+_352_288_1.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def good_match_count(
    marker_descriptors: np.ndarray,
    target_descriptors: np.ndarray | None,
    ratio: float,
) -> int:
    """Compte les correspondances ORB conservées après le test de Lowe."""
    if target_descriptors is None or len(marker_descriptors) < 2 or len(target_descriptors) < 2:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(marker_descriptors, target_descriptors, k=2)
    return sum(1 for pair in pairs if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance)


def merge_detections(detections: list[dict[str, Any]], max_gap_frames: int) -> list[dict[str, Any]]:
    """Regroupe les images détectées successives en occurrences temporelles."""
    if not detections:
        return []
    groups: list[list[dict[str, Any]]] = [[detections[0]]]
    for detection in detections[1:]:
        if detection["target_frame"] - groups[-1][-1]["target_frame"] <= max_gap_frames:
            groups[-1].append(detection)
        else:
            groups.append([detection])

    events = []
    for group in groups:
        best = max(group, key=lambda item: item["good_matches"])
        events.append(
            {
                "start_frame": group[0]["target_frame"],
                "end_frame": group[-1]["target_frame"],
                "best_frame": best["target_frame"],
                "best_good_matches": best["good_matches"],
                "matched_reference_frame": best["reference_frame"],
            }
        )
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "database",
        nargs="?",
        default="marker_database",
        help="Dossier créé par BuildMarkerDatabase.py.",
    )
    parser.add_argument(
        "target_video",
        nargs="?",
        default="SpaceCowboys.mp4",
        help="Vidéo dans laquelle rechercher la séquence.",
    )
    parser.add_argument("--scan-step", type=int, default=5, help="Analyse une image sur N dans la vidéo cible.")
    parser.add_argument("--ratio", type=float, default=0.75, help="Seuil du test de Lowe (0 < ratio < 1).")
    parser.add_argument("--min-good-matches", type=int, default=18, help="Nombre minimal de correspondances ORB.")
    parser.add_argument("--max-features", type=int, default=1000)
    parser.add_argument("--output", default="sequence_detections.json")
    args = parser.parse_args()

    if args.scan_step <= 0 or not 0 < args.ratio < 1 or args.min_good_matches <= 0:
        raise ValueError("Paramètres de détection invalides.")

    database_dir = Path(args.database)
    manifest = json.loads((database_dir / "manifest.json").read_text(encoding="utf-8"))
    marker_data = [
        (marker, np.load(database_dir / str(marker["descriptor_file"])))
        for marker in manifest["markers"]
    ]

    capture = cv2.VideoCapture(args.target_video)
    if not capture.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir : {args.target_video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    orb = cv2.ORB_create(nfeatures=args.max_features)
    detections: list[dict[str, Any]] = []

    try:
        for frame_index in range(0, frame_count, args.scan_step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, target_descriptors = orb.detectAndCompute(gray, None)
            scores = [
                (good_match_count(descriptors, target_descriptors, args.ratio), marker)
                for marker, descriptors in marker_data
            ]
            best_score, best_marker = max(scores, key=lambda item: item[0])
            if best_score >= args.min_good_matches:
                detections.append(
                    {
                        "target_frame": frame_index,
                        "target_time_seconds": frame_index / fps if fps else 0.0,
                        "reference_frame": best_marker["frame_index"],
                        "good_matches": best_score,
                    }
                )
    finally:
        capture.release()

    events = merge_detections(detections, max_gap_frames=args.scan_step * 2)
    for event in events:
        event["start_time_seconds"] = event["start_frame"] / fps if fps else 0.0
        event["end_time_seconds"] = event["end_frame"] / fps if fps else 0.0

    result = {
        "sequence_name": manifest["sequence_name"],
        "target_video": str(Path(args.target_video).resolve()),
        "scan_step": args.scan_step,
        "min_good_matches": args.min_good_matches,
        "raw_detections": detections,
        "events": events,
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Détections brutes : {len(detections)}")
    print(f"Occurrences regroupées : {len(events)}")
    print(f"Résultat : {output_path}")


if __name__ == "__main__":
    main()
