from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, List, Any


def cluster_faces(
    videos: Iterable[Path],
    frame_rate: float = 1.0,
    eps: float = 0.5,
    min_samples: int = 2,
) -> Dict[int, List[Dict[str, Any]]]:
    """Detect faces in ``videos`` and cluster them by similarity.

    Parameters
    ----------
    videos:
        Iterable of video file paths.
    frame_rate:
        Frames per second to sample for detection.
    eps, min_samples:
        Parameters passed to :class:`sklearn.cluster.DBSCAN`.

    Returns
    -------
    dict
        Mapping of cluster id to list of occurrences. Each occurrence contains
        ``video`` name, ``timestamp`` in seconds and ``bbox`` dictionary with
        ``top``, ``right``, ``bottom`` and ``left`` coordinates.
    """
    try:
        import cv2
        import numpy as np
        from sklearn.cluster import DBSCAN
        import face_recognition
    except Exception as exc:  # pragma: no cover - heavy optional deps
        raise RuntimeError(
            "face clustering requires cv2, face_recognition and scikit-learn"
        ) from exc

    embeddings: List[np.ndarray] = []
    occurrences: List[Dict[str, Any]] = []

    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            cap.release()
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(int(round(fps / frame_rate)), 1)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                rgb = frame[:, :, ::-1]
                boxes = face_recognition.face_locations(rgb)
                if boxes:
                    encs = face_recognition.face_encodings(rgb, boxes)
                    ts = frame_idx / fps
                    for (top, right, bottom, left), enc in zip(boxes, encs):
                        embeddings.append(enc)
                        occurrences.append(
                            {
                                "video": video.name,
                                "timestamp": ts,
                                "bbox": {
                                    "top": int(top),
                                    "right": int(right),
                                    "bottom": int(bottom),
                                    "left": int(left),
                                },
                            }
                        )
            frame_idx += 1
        cap.release()

    if not embeddings:
        return {}

    data = np.vstack(embeddings)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = data / norms

    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(
        normalized
    )

    clusters: Dict[int, List[Dict[str, Any]]] = {}
    for label, occ in zip(labels, occurrences):
        clusters.setdefault(int(label), []).append(occ)
    return clusters
