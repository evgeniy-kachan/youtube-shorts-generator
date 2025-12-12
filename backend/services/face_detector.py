"""Face detection helper powered by InsightFace."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face detector using InsightFace for horizontal face focus estimation.
    """

    def __init__(
        self,
        model_name: str = "buffalo_s",
        det_thresh: float = 0.5,
        ctx_id: int = 0,
    ):
        """
        Args:
            model_name: InsightFace model pack name (buffalo_s, buffalo_l, etc.)
            det_thresh: Detection threshold (0.0-1.0)
            ctx_id: Context ID: 0 for GPU, -1 for CPU
        """
        self.model_name = model_name
        self.det_thresh = det_thresh
        self.ctx_id = ctx_id
        self._detector = None
        self._init_detector()

    def _init_detector(self):
        """Initialize InsightFace detector (lazy load)."""
        try:
            from insightface.app import FaceAnalysis
            
            logger.info("Initializing InsightFace with model=%s, ctx_id=%d", self.model_name, self.ctx_id)
            self._detector = FaceAnalysis(name=self.model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self._detector.prepare(ctx_id=self.ctx_id, det_thresh=self.det_thresh, det_size=(640, 640))
            logger.info("InsightFace initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize InsightFace: %s", e, exc_info=True)
            raise RuntimeError(f"InsightFace initialization failed: {e}") from e

    def _detect_faces(self, frame: np.ndarray) -> Sequence[dict]:
        """Detect faces in frame using InsightFace."""
        if self._detector is None:
            return []
        
        try:
            faces = self._detector.get(frame)
            detections = []
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                detections.append({
                    "x": float(x1),
                    "y": float(y1),
                    "w": float(w),
                    "h": float(h),
                    "score": float(face.det_score),
                    "area": float(w * h),
                    "center_x": float(x1 + w / 2.0),
                    "center_y": float(y1 + h / 2.0),
                    "width": float(frame.shape[1]),
                    "height": float(frame.shape[0]),
                })
            return detections
        except Exception as e:
            logger.warning("Face detection failed: %s", e)
            return []

    def estimate_horizontal_focus(
        self,
        video_path: str,
        max_samples: int = 6,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Return averaged face center ratio (0..1) and multi-face span ratio for the clip.

        Args:
            video_path: path to the already cut clip (few dozen seconds max)
            max_samples: number of frames to sample uniformly across the clip
        """
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            logger.warning("InsightFace: unable to open video %s for face detection", video_path)
            return None

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step = max(frame_count // max_samples, 1)
        sample_indices = list(range(0, frame_count, step))[:max_samples]

        weighted_centers: list[tuple[float, float]] = []
        weighted_spans: list[tuple[float, float]] = []

        for index in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            faces = self._detect_faces(frame)
            # Drop tiny detections (<1% площади кадра), чтобы не тянуть шум
            # Keep even небольшие лица: порог 0.2% площади кадра, чтобы не отбрасывать средние планы
            faces = [
                f for f in faces
                if f.get("area", 0.0) >= 0.002 * f.get("width", 1.0) * f.get("height", 1.0)
            ]
            if not faces:
                continue

            # Debug log
            logger.info("InsightFace frame %d: found %d faces", index, len(faces))
            for i, f in enumerate(faces):
                logger.info(
                    "  Face %d: x=%.1f, y=%.1f, w=%.1f, h=%.1f, score=%.2f, center_x=%.1f (ratio=%.3f)",
                    i,
                    f["x"],
                    f["y"],
                    f["w"],
                    f["h"],
                    f["score"],
                    f["center_x"],
                    f["center_x"] / max(f["width"], 1.0),
                )

            # If 2+ faces: center between leftmost and rightmost faces to keep both in frame.
            # Also capture span ratio to decide auto zoom-out upstream.
            # Otherwise: pick the strongest single face.
            if len(faces) >= 2:
                left_face = min(faces, key=lambda f: f["center_x"])
                right_face = max(faces, key=lambda f: f["center_x"])
                frame_width = max(left_face.get("width", 1.0), right_face.get("width", 1.0))
                span_ratio = (right_face["center_x"] - left_face["center_x"]) / frame_width
                center_ratio = (left_face["center_x"] + right_face["center_x"]) / (2.0 * frame_width)
                weight = sum(f["score"] * f["area"] for f in faces)
                logger.info(
                    "  Multi-face span=%.3f center=%.3f weight=%.0f",
                    span_ratio,
                    center_ratio,
                    weight,
                )
                weighted_centers.append((center_ratio, weight))
                weighted_spans.append((span_ratio, weight))
            else:
                best_face = max(faces, key=lambda f: f["score"] * f["area"])
                center_ratio = best_face["center_x"] / best_face["width"]
                weight = best_face["score"] * best_face["area"]
                weighted_centers.append((center_ratio, weight))

        capture.release()

        if not weighted_centers:
            logger.info("InsightFace: no faces found in %s", video_path)
            return None, None

        numerator = sum(center * weight for center, weight in weighted_centers)
        denominator = sum(weight for _, weight in weighted_centers)
        if denominator <= 0:
            return None, None

        focus_raw = numerator / denominator
        # If all detections cluster too far to a border, treat as unreliable and fall back to center.
        if focus_raw < 0.15 or focus_raw > 0.85:
            logger.warning(
                "InsightFace: focus_raw=%.3f looks unreliable (all faces near edge), "
                "fallback to center crop 0.5",
                focus_raw,
            )
            focus_raw = 0.5

        focus = float(max(0.0, min(1.0, focus_raw)))
        # Clamp extreme values to avoid hard-left/right crops when detections are uncertain
        focus = float(max(0.2, min(0.8, focus)))

        span = None
        if weighted_spans:
            span_num = sum(span * weight for span, weight in weighted_spans)
            span_den = sum(weight for _, weight in weighted_spans)
            if span_den > 0:
                span = float(span_num / span_den)

        logger.info(
            "InsightFace: focus raw=%.3f clamped=%.3f span=%s samples=%d",
            focus_raw,
            focus,
            f"{span:.3f}" if span is not None else "n/a",
            len(weighted_centers),
        )
        logger.info("InsightFace: estimated horizontal focus %.3f span %s for %s", focus, span, video_path)
        return focus, span
