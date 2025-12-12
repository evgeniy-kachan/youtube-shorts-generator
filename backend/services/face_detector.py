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
        dialogue: list[dict] | None = None,
        segment_start: float = 0.0,
    ) -> Optional[float]:
        """
        Return averaged face center ratio (0..1) for the clip.

        Args:
            video_path: path to the already cut clip (few dozen seconds max)
            max_samples: number of frames to sample
            dialogue: list of dialogue turns with speaker, start, end times
            segment_start: absolute start time of the segment in the original video
        
        Strategy:
        - If dialogue is provided: sample frames from primary speaker's speech moments
        - If multiple faces detected: pick the one closest to center
        - Fallback to uniform sampling if no dialogue or no faces found
        """
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            logger.warning("InsightFace: unable to open video %s for face detection", video_path)
            return None

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        
        # Determine sample timestamps
        sample_indices = self._get_sample_indices(
            dialogue=dialogue,
            segment_start=segment_start,
            frame_count=frame_count,
            fps=fps,
            max_samples=max_samples,
        )

        weighted_centers: list[tuple[float, float]] = []

        for index in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            faces = self._detect_faces(frame)
            # Drop tiny detections (<0.2% area)
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

            # If multiple faces: pick the one closest to center (0.5)
            # This focuses on the main speaker rather than trying to fit everyone
            if len(faces) >= 2:
                frame_width = faces[0].get("width", 1.0)
                best_face = min(
                    faces,
                    key=lambda f: abs(f["center_x"] / frame_width - 0.5)
                )
                center_ratio = best_face["center_x"] / frame_width
                weight = best_face["score"] * best_face["area"]
                logger.info(
                    "  Multi-face: picked center-most face (ratio=%.3f, weight=%.0f)",
                    center_ratio,
                    weight,
                )
                weighted_centers.append((center_ratio, weight))
            else:
                best_face = max(faces, key=lambda f: f["score"] * f["area"])
                center_ratio = best_face["center_x"] / best_face["width"]
                weight = best_face["score"] * best_face["area"]
                weighted_centers.append((center_ratio, weight))

        capture.release()

        if not weighted_centers:
            logger.info("InsightFace: no faces found in %s", video_path)
            return None

        numerator = sum(center * weight for center, weight in weighted_centers)
        denominator = sum(weight for _, weight in weighted_centers)
        if denominator <= 0:
            return None

        focus_raw = numerator / denominator
        # If all detections cluster too far to a border, treat as unreliable and fall back to center.
        if focus_raw < 0.15 or focus_raw > 0.85:
            logger.warning(
                "InsightFace: focus_raw=%.3f looks unreliable (all faces near edge), "
                "fallback to center crop 0.5",
                focus_raw,
            )
            return 0.5

        focus = float(max(0.0, min(1.0, focus_raw)))
        # Clamp extreme values to avoid hard-left/right crops when detections are uncertain
        focus = float(max(0.2, min(0.8, focus)))

        logger.info(
            "InsightFace: focus raw=%.3f clamped=%.3f samples=%d",
            focus_raw,
            focus,
            len(weighted_centers),
        )
        logger.info("InsightFace: estimated horizontal focus %.3f for %s", focus, video_path)
        return focus
    
    def _get_sample_indices(
        self,
        dialogue: list[dict] | None,
        segment_start: float,
        frame_count: int,
        fps: float,
        max_samples: int,
    ) -> list[int]:
        """
        Return frame indices to sample for face detection.
        
        If dialogue is provided, sample from primary speaker's speech moments.
        Otherwise, sample uniformly across the clip.
        """
        if not dialogue:
            # Uniform sampling fallback
            step = max(frame_count // max_samples, 1)
            return list(range(0, frame_count, step))[:max_samples]
        
        # Find primary speaker (longest total speech duration)
        speaker_durations: dict[str, float] = {}
        for turn in dialogue:
            speaker = turn.get("speaker")
            if not speaker:
                continue
            start = turn.get("start", 0.0)
            end = turn.get("end", 0.0)
            duration = max(0.0, end - start)
            speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration
        
        if not speaker_durations:
            # No valid speakers, fallback to uniform
            step = max(frame_count // max_samples, 1)
            return list(range(0, frame_count, step))[:max_samples]
        
        primary_speaker = max(speaker_durations, key=speaker_durations.get)
        logger.info(
            "Primary speaker: %s (%.1fs total), segment_start=%.2fs",
            primary_speaker,
            speaker_durations[primary_speaker],
            segment_start,
        )
        
        # Collect primary speaker's speech intervals relative to segment
        speech_intervals: list[tuple[float, float]] = []
        for turn in dialogue:
            if turn.get("speaker") != primary_speaker:
                continue
            # dialogue times are absolute in original video
            abs_start = turn.get("start", 0.0)
            abs_end = turn.get("end", 0.0)
            # Convert to relative to segment
            rel_start = max(0.0, abs_start - segment_start)
            rel_end = max(0.0, abs_end - segment_start)
            if rel_end > rel_start:
                speech_intervals.append((rel_start, rel_end))
        
        if not speech_intervals:
            # Primary speaker not in this segment, fallback to uniform
            logger.info("Primary speaker not in this segment, using uniform sampling")
            step = max(frame_count // max_samples, 1)
            return list(range(0, frame_count, step))[:max_samples]
        
        # Sample frames from speech intervals
        total_speech_duration = sum(end - start for start, end in speech_intervals)
        samples_per_interval = max(1, max_samples // len(speech_intervals))
        
        sample_indices: list[int] = []
        for start_time, end_time in speech_intervals:
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            interval_frames = max(1, end_frame - start_frame)
            step = max(1, interval_frames // samples_per_interval)
            for frame_idx in range(start_frame, end_frame, step):
                if frame_idx < frame_count:
                    sample_indices.append(frame_idx)
                if len(sample_indices) >= max_samples:
                    break
            if len(sample_indices) >= max_samples:
                break
        
        if not sample_indices:
            # Fallback to uniform
            step = max(frame_count // max_samples, 1)
            sample_indices = list(range(0, frame_count, step))[:max_samples]
        
        logger.info(
            "Sampled %d frames from %d speech intervals (%.1fs total)",
            len(sample_indices),
            len(speech_intervals),
            total_speech_duration,
        )
        return sample_indices[:max_samples]
