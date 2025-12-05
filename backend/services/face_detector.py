"""Face detection helper built on top of OpenCV YuNet."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import cv2
import httpx
import numpy as np

from backend import config

logger = logging.getLogger(__name__)

YUNET_MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/master/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_MODEL_NAME = "face_detection_yunet_2023mar.onnx"


class FaceDetector:
    """
    Thin wrapper around cv2.FaceDetectorYN (YuNet) to estimate horizontal face focus.

    We keep a single detector instance per process and lazily resize it for every frame
    sequence to avoid re-loading weights for each crop request.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        download_timeout: float = 60.0,
        score_threshold: float = 0.85,
        nms_threshold: float = 0.3,
    ):
        self.model_path = model_path or self._ensure_model(download_timeout)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self._detector = None
        self._input_size: Optional[tuple[int, int]] = None

    @staticmethod
    def _ensure_model(timeout: float) -> Path:
        models_dir = Path(config.TEMP_DIR) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / YUNET_MODEL_NAME

        if model_path.exists():
            return model_path

        logger.info("Downloading YuNet face detector weights to %s", model_path)
        with httpx.stream("GET", YUNET_MODEL_URL, timeout=timeout) as response:
            response.raise_for_status()
            with open(model_path, "wb") as model_file:
                for chunk in response.iter_bytes():
                    model_file.write(chunk)

        logger.info("YuNet model downloaded successfully.")
        return model_path

    def _get_detector(self, width: int, height: int):
        size = (width, height)
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                str(self.model_path),
                "",
                size,
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold,
                top_k=5000,
            )
            self._input_size = size
        elif self._input_size != size:
            self._detector.setInputSize(size)
            self._input_size = size
        return self._detector

    def _detect_faces(self, frame: np.ndarray) -> Sequence[dict]:
        height, width = frame.shape[:2]
        detector = self._get_detector(width, height)
        detector.setInputSize((width, height))
        _, faces = detector.detect(frame)

        if faces is None:
            return []

        results = []
        for face in faces:
            # YuNet output layout:
            # [x, y, w, h, score, landmarks...]
            x, y, w, h, score = face[:5]
            if score < self.score_threshold:
                continue
            results.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "w": float(w),
                    "h": float(h),
                    "score": float(score),
                    "area": float(w * h),
                    "center_x": float(x + w / 2.0),
                    "width": width,
                }
            )
        return results

    def estimate_horizontal_focus(
        self,
        video_path: str,
        max_samples: int = 6,
    ) -> Optional[float]:
        """
        Return averaged face center ratio (0..1) for the supplied video clip.

        Args:
            video_path: path to the already cut clip (few dozen seconds max)
            max_samples: number of frames to sample uniformly across the clip
        """
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            logger.warning("YuNet: unable to open video %s for face detection", video_path)
            return None

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step = max(frame_count // max_samples, 1)
        sample_indices = list(range(0, frame_count, step))[:max_samples]

        weighted_centers: list[tuple[float, float]] = []

        for index in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            faces = self._detect_faces(frame)
            if not faces:
                continue

            best_face = max(faces, key=lambda f: f["score"] * f["area"])
            center_ratio = best_face["center_x"] / best_face["width"]
            weight = best_face["score"] * best_face["area"]
            weighted_centers.append((center_ratio, weight))

        capture.release()

        if not weighted_centers:
            logger.info("YuNet: no faces found in %s", video_path)
            return None

        numerator = sum(center * weight for center, weight in weighted_centers)
        denominator = sum(weight for _, weight in weighted_centers)
        if denominator <= 0:
            return None

        focus = numerator / denominator
        focus = float(max(0.0, min(1.0, focus)))
        logger.info("YuNet: estimated horizontal focus %.3f for %s", focus, video_path)
        return focus

