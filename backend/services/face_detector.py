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
        segment_end: float | None = None,
    ) -> Optional[float]:
        """
        Return averaged face center ratio (0..1) for the clip.

        Args:
            video_path: path to the already cut clip (few dozen seconds max)
            max_samples: number of frames to sample
            dialogue: list of dialogue turns with speaker, start, end times
            segment_start: absolute start time of the segment in the original video
            segment_end: absolute end time of the segment (for primary speaker detection)
        
        Strategy:
        - Sample frames from all speakers proportionally
        - Single face: use its center
        - Multiple faces, span < 0.40: center between extremes (both fit)
        - Multiple faces, span >= 0.40: focus on primary speaker (who talks more)
        - If speakers are on different positions: focus on primary speaker cluster
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
        single_face_positions: list[float] = []  # Track single-face positions to identify primary speaker location

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

            # Logic:
            # - Single face: use its center, track for primary speaker identification
            # - Multiple faces, span < 0.40: center between extremes (show both)
            # - Multiple faces, span >= 0.40: pick face closest to primary speaker position
            if len(faces) >= 2:
                frame_width = faces[0].get("width", 1.0)
                centers = [f["center_x"] / frame_width for f in faces]
                span = max(centers) - min(centers)

                if span < 0.40:
                    # Faces are close enough to show both
                    center_ratio = (min(centers) + max(centers)) / 2.0
                    weight = sum(f["score"] * f["area"] for f in faces)
                    logger.info(
                        "  Multi-face (span=%.3f < 0.40): show both, center=%.3f, weight=%.0f",
                        span,
                        center_ratio,
                        weight,
                    )
                    weighted_centers.append((center_ratio, weight))
                else:
                    # Faces too far apart - will decide later based on primary speaker position
                    logger.info(
                        "  Multi-face (span=%.3f >= 0.40): defer to primary speaker position",
                        span,
                    )
            else:
                # Single face - track for identifying primary speaker position
                best_face = max(faces, key=lambda f: f["score"] * f["area"])
                center_ratio = best_face["center_x"] / best_face["width"]
                weight = best_face["score"] * best_face["area"]
                single_face_positions.append(center_ratio)
                weighted_centers.append((center_ratio, weight))

        capture.release()

        if not weighted_centers:
            logger.info("InsightFace: no faces found in %s", video_path)
            return None

        # Check if we have two speakers on different positions (left vs right)
        if len(single_face_positions) >= 3:
            min_pos = min(single_face_positions)
            max_pos = max(single_face_positions)
            position_span = max_pos - min_pos
            
            if position_span > 0.35:
                logger.info(
                    "Detected speakers on different positions (span=%.3f): left=%.3f, right=%.3f",
                    position_span,
                    min_pos,
                    max_pos,
                )
                
                # Determine primary speaker (who talks more in this segment)
                primary_speaker = self._get_primary_speaker(dialogue, segment_start, segment_end)
                
                if primary_speaker:
                    # Split single-face positions into left and right clusters
                    midpoint = (min_pos + max_pos) / 2.0
                    left_positions = [p for p in single_face_positions if p < midpoint]
                    right_positions = [p for p in single_face_positions if p >= midpoint]
                    
                    logger.info(
                        "Position clusters: left=%d faces (avg=%.3f), right=%d faces (avg=%.3f)",
                        len(left_positions),
                        sum(left_positions) / len(left_positions) if left_positions else 0.0,
                        len(right_positions),
                        sum(right_positions) / len(right_positions) if right_positions else 0.0,
                    )
                    
                    # Heuristic: primary speaker is in the cluster with MORE single-face frames
                    # (because they talk more, so more frames show only them)
                    primary_cluster_positions = None
                    if len(left_positions) > len(right_positions):
                        primary_cluster_positions = left_positions
                        logger.info("Primary speaker '%s' identified on LEFT", primary_speaker)
                    elif len(right_positions) > len(left_positions):
                        primary_cluster_positions = right_positions
                        logger.info("Primary speaker '%s' identified on RIGHT", primary_speaker)
                    else:
                        # Equal counts - use center as fallback
                        logger.info("Equal cluster sizes, using center")
                        primary_cluster_positions = None
                    
                    if primary_cluster_positions:
                        focus = sum(primary_cluster_positions) / len(primary_cluster_positions)
                        focus = float(max(0.2, min(0.8, focus)))
                        logger.info("Focusing on primary speaker cluster: %.3f", focus)
                        return focus

        # Normal weighted average calculation
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
    
    def diagnose_final_crop(self, video_path: str, max_samples: int = 3) -> None:
        """
        Diagnostic: check where faces ended up in the final cropped video.
        Logs face positions in the 1080px wide frame to verify crop quality.
        """
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            logger.warning("Diagnose: unable to open video %s", video_path)
            return
        
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step = max(frame_count // max_samples, 1)
        sample_indices = list(range(0, frame_count, step))[:max_samples]
        
        logger.info("=" * 60)
        logger.info("POST-CROP DIAGNOSTIC for %s", video_path)
        logger.info("=" * 60)
        
        for index in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            
            faces = self._detect_faces(frame)
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]
            
            logger.info("Frame %d (resolution %dx%d): found %d faces", index, frame_width, frame_height, len(faces))
            
            if not faces:
                logger.info("  No faces detected")
                continue
            
            for i, f in enumerate(faces):
                x = f["x"]
                w = f["w"]
                center_x = f["center_x"]
                
                # Check if face is cut off
                left_cut = x < 0
                right_cut = (x + w) > frame_width
                
                status = "OK"
                if left_cut and right_cut:
                    status = "CUT BOTH SIDES"
                elif left_cut:
                    status = "CUT LEFT"
                elif right_cut:
                    status = "CUT RIGHT"
                
                logger.info(
                    "  Face %d: x=%.1f center_x=%.1f w=%.1f [%.1f - %.1f] score=%.2f | %s",
                    i,
                    x,
                    center_x,
                    w,
                    x,
                    x + w,
                    f["score"],
                    status,
                )
                logger.info(
                    "    Position: %.1f%% from left (0%%=left edge, 50%%=center, 100%%=right edge)",
                    (center_x / frame_width) * 100,
                )
        
        logger.info("=" * 60)
        capture.release()
    
    def _get_primary_speaker(
        self,
        dialogue: list[dict] | None,
        segment_start: float,
        segment_end: float | None,
    ) -> str | None:
        """
        Determine which speaker talks the most in the given segment.
        
        Args:
            dialogue: list of dialogue turns with speaker, start, end times
            segment_start: absolute start time of the segment
            segment_end: absolute end time of the segment (optional, inferred from dialogue if None)
        
        Returns:
            Speaker identifier (str) or None if cannot determine
        """
        if not dialogue:
            return None
        
        # If segment_end not provided, use max end time from dialogue
        if segment_end is None:
            segment_end = max((turn.get("end", segment_start) for turn in dialogue), default=segment_start + 30.0)
        
        speaker_durations: dict[str, float] = {}
        
        for turn in dialogue:
            speaker = turn.get("speaker")
            if not speaker:
                continue
            
            # Calculate overlap with segment [segment_start, segment_end]
            turn_start = turn.get("start", 0.0)
            turn_end = turn.get("end", 0.0)
            
            overlap_start = max(turn_start, segment_start)
            overlap_end = min(turn_end, segment_end)
            overlap_duration = max(0.0, overlap_end - overlap_start)
            
            if overlap_duration > 0:
                speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + overlap_duration
        
        if not speaker_durations:
            return None
        
        # Find speaker with most time
        primary_speaker = max(speaker_durations.items(), key=lambda x: x[1])[0]
        total_speech = sum(speaker_durations.values())
        
        logger.info(
            "Primary speaker in segment [%.1f-%.1f]: '%s' (%.1fs / %.1fs = %.0f%%)",
            segment_start,
            segment_end,
            primary_speaker,
            speaker_durations[primary_speaker],
            total_speech,
            (speaker_durations[primary_speaker] / total_speech * 100) if total_speech > 0 else 0,
        )
        
        return primary_speaker
    
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
        
        If dialogue is provided, sample from ALL speakers proportionally to their speech time.
        Otherwise, sample uniformly across the clip.
        """
        if not dialogue:
            # Uniform sampling fallback
            step = max(frame_count // max_samples, 1)
            return list(range(0, frame_count, step))[:max_samples]
        
        # Calculate speech duration for each speaker
        speaker_durations: dict[str, float] = {}
        speaker_intervals: dict[str, list[tuple[float, float]]] = {}
        
        for turn in dialogue:
            speaker = turn.get("speaker")
            if not speaker:
                continue
            abs_start = turn.get("start", 0.0)
            abs_end = turn.get("end", 0.0)
            # Convert to relative to segment
            rel_start = max(0.0, abs_start - segment_start)
            rel_end = max(0.0, abs_end - segment_start)
            duration = max(0.0, rel_end - rel_start)
            
            if duration > 0:
                speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration
                if speaker not in speaker_intervals:
                    speaker_intervals[speaker] = []
                speaker_intervals[speaker].append((rel_start, rel_end))
        
        if not speaker_durations:
            # No valid speakers, fallback to uniform
            step = max(frame_count // max_samples, 1)
            return list(range(0, frame_count, step))[:max_samples]
        
        total_speech_time = sum(speaker_durations.values())
        
        # Allocate samples proportionally to each speaker's speech time
        # But ensure each speaker gets at least 1 sample
        sample_allocation: dict[str, int] = {}
        remaining_samples = max_samples
        
        # First pass: give everyone at least 1 sample
        for speaker in speaker_durations:
            sample_allocation[speaker] = 1
            remaining_samples -= 1
        
        # Second pass: distribute remaining samples proportionally
        if remaining_samples > 0:
            for speaker, duration in speaker_durations.items():
                proportion = duration / total_speech_time
                extra_samples = int(proportion * remaining_samples)
                sample_allocation[speaker] += extra_samples
        
        logger.info(
            "Speaker durations: %s, sample allocation: %s",
            {k: f"{v:.1f}s" for k, v in speaker_durations.items()},
            sample_allocation,
        )
        
        # Sample frames from each speaker's intervals
        sample_indices: list[int] = []
        
        for speaker, num_samples in sample_allocation.items():
            if num_samples <= 0:
                continue
            intervals = speaker_intervals.get(speaker, [])
            if not intervals:
                continue
            
            samples_per_interval = max(1, num_samples // len(intervals))
            speaker_samples = []
            
            for start_time, end_time in intervals:
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                interval_frames = max(1, end_frame - start_frame)
                step = max(1, interval_frames // samples_per_interval)
                
                for frame_idx in range(start_frame, end_frame, step):
                    if frame_idx < frame_count:
                        speaker_samples.append(frame_idx)
                    if len(speaker_samples) >= num_samples:
                        break
                if len(speaker_samples) >= num_samples:
                    break
            
            sample_indices.extend(speaker_samples[:num_samples])
        
        if not sample_indices:
            # Fallback to uniform
            step = max(frame_count // max_samples, 1)
            sample_indices = list(range(0, frame_count, step))[:max_samples]
        
        # Sort and limit
        sample_indices = sorted(set(sample_indices))[:max_samples]
        
        logger.info(
            "Sampled %d frames from %d speakers (%.1fs total speech)",
            len(sample_indices),
            len(speaker_durations),
            total_speech_time,
        )
        return sample_indices
