"""Face detection helper powered by InsightFace."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SimpleTracker:
    """
    IoU-based tracker with prediction support for profile faces.
    Key feature: tracks survive when SCRFD misses detection (profile view).
    After MIN_HITS frames, track is considered "confirmed" and will be used
    even when detection fails.
    """

    MIN_HITS = 3  # Track must be detected N times to be confirmed
    MAX_AGE = 15  # Keep track alive for N frames without detection

    def __init__(self, max_age: int = 15, iou_thresh: float = 0.25):
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.tracks: list[dict] = []
        self.next_id = 1

    @staticmethod
    def _iou(b1, b2) -> float:
        x1, y1, x2, y2 = b1
        a1, b1_, a2, b2_ = b2
        inter_x1 = max(x1, a1)
        inter_y1 = max(y1, b1_)
        inter_x2 = min(x2, a2)
        inter_y2 = min(y2, b2_)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (a2 - a1) * (b2_ - b1_)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def reset(self):
        self.tracks.clear()
        self.next_id = 1

    def get_confirmed_tracks(self) -> list[dict]:
        """Return tracks that have been detected at least MIN_HITS times."""
        return [t for t in self.tracks if t.get("hits", 0) >= self.MIN_HITS]

    def update(self, faces: Sequence[dict]) -> list[dict]:
        """
        Update tracks with new detections.
        Returns: list of faces with smoothed positions (includes predicted tracks).
        """
        # Age existing tracks
        for t in self.tracks:
            t["age"] += 1

        # Prepare detections
        det_boxes = []
        det_faces = []
        for f in faces:
            x1 = f["x"]
            y1 = f["y"]
            x2 = f["x"] + f["w"]
            y2 = f["y"] + f["h"]
            det_boxes.append((x1, y1, x2, y2))
            det_faces.append(f)

        assigned_tracks = set()
        assigned_dets = set()

        # Greedy matching by IoU
        for det_idx, box in enumerate(det_boxes):
            best_iou = self.iou_thresh
            best_track = None
            for ti, t in enumerate(self.tracks):
                if ti in assigned_tracks:
                    continue
                iou = self._iou(box, t["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track = ti
            if best_track is not None:
                # Update track with smoothing
                t = self.tracks[best_track]
                t["bbox"] = box
                t["age"] = 0
                t["hits"] = t.get("hits", 0) + 1
                t["last_face"] = det_faces[det_idx]  # Store last detected face
                # Smooth center_x
                det_cx = (box[0] + box[2]) / 2.0
                smoothed_cx = 0.7 * det_cx + 0.3 * t["center_x"]
                t["center_x"] = smoothed_cx
                assigned_tracks.add(best_track)
                assigned_dets.add(det_idx)

        # New tracks for unassigned detections
        for det_idx, box in enumerate(det_boxes):
            if det_idx in assigned_dets:
                continue
            det_cx = (box[0] + box[2]) / 2.0
            self.tracks.append(
                {
                    "id": self.next_id,
                    "bbox": box,
                    "center_x": det_cx,
                    "age": 0,
                    "hits": 1,
                    "last_face": det_faces[det_idx],
                }
            )
            self.next_id += 1

        # Drop stale tracks
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]

        # Build output: detected faces + confirmed tracks without detection
        out = []
        used_track_ids = set()

        # First, add all detected faces with smoothed positions
        for det_idx, f in enumerate(det_faces):
            box = det_boxes[det_idx]
            best_track = None
            best_iou = self.iou_thresh
            for t in self.tracks:
                iou = self._iou(box, t["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track = t
            if best_track:
                f = dict(f)
                f["center_x"] = best_track["center_x"]
                f["track_id"] = best_track["id"]
                used_track_ids.add(best_track["id"])
            out.append(f)

        # Add confirmed tracks that weren't detected this frame (profile prediction)
        for t in self.tracks:
            if t["id"] in used_track_ids:
                continue
            # Only use confirmed tracks (seen MIN_HITS times) that are still fresh
            if t.get("hits", 0) >= self.MIN_HITS and t["age"] <= 5:
                # Use last known face with predicted position
                if "last_face" in t:
                    predicted_face = dict(t["last_face"])
                    predicted_face["center_x"] = t["center_x"]
                    predicted_face["track_id"] = t["id"]
                    predicted_face["predicted"] = True  # Mark as predicted
                    out.append(predicted_face)
                    logger.debug(
                        "Track %d: using predicted position (age=%d, hits=%d)",
                        t["id"], t["age"], t["hits"]
                    )

        return out


class FaceDetector:
    """
    Face detector using InsightFace for horizontal face focus estimation.
    Uses tracking to maintain face positions when SCRFD misses profile views.
    """

    def __init__(
        self,
        model_name: str = "antelopev2",  # default to SCRFD (better on profiles)
        det_thresh: float = 0.15,  # Lower threshold to catch profile faces
        ctx_id: int = 0,
    ):
        """
        Args:
            model_name: InsightFace model pack name (scrfd_*, buffalo_s, etc.)
            det_thresh: Detection threshold (0.0-1.0)
            ctx_id: Context ID: 0 for GPU, -1 for CPU
        """
        self.model_name = model_name
        self.det_thresh = det_thresh
        self.ctx_id = ctx_id
        self._detector = None
        self.enable_tracking = os.getenv("FACE_TRACKING", "0") == "1"
        self._tracker: Optional["SimpleTracker"] = None
        self._init_detector()

    def _init_detector(self):
        """Initialize InsightFace detector (lazy load)."""
        try:
            from insightface.app import FaceAnalysis
            
            logger.info("Initializing InsightFace with model=%s, ctx_id=%d", self.model_name, self.ctx_id)
            self._detector = FaceAnalysis(
                name=self.model_name,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            # Larger det_size to catch small faces on wide shots
            # 1920x1920 allows detection of faces ~20px wide on 4K video
            self._detector.prepare(ctx_id=self.ctx_id, det_thresh=self.det_thresh, det_size=(1280, 1280))
            logger.info("InsightFace initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize InsightFace: %s", e, exc_info=True)
            raise RuntimeError(f"InsightFace initialization failed: {e}") from e

    # Minimum face width in pixels to filter out noise detections
    # Set to 16 to catch small faces on wide shots / profile views
    MIN_FACE_WIDTH_PX = 16

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
                # Filter out tiny detections (noise)
                if w < self.MIN_FACE_WIDTH_PX:
                    logger.debug("Skipping tiny face detection: w=%d < %d", w, self.MIN_FACE_WIDTH_PX)
                    continue
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
    ) -> tuple[Optional[float], Optional[tuple[float, float]]]:
        """
        Return averaged face center ratio (0..1) for the clip and two-speaker positions if detected.

        Args:
            video_path: path to the already cut clip (few dozen seconds max)
            max_samples: number of frames to sample
            dialogue: list of dialogue turns with speaker, start, end times
            segment_start: absolute start time of the segment in the original video
            segment_end: absolute end time of the segment (for primary speaker detection)
        
        Returns:
            Tuple of (focus_ratio, two_speaker_positions)
            - focus_ratio: 0..1 horizontal focus position
            - two_speaker_positions: (left_pos, right_pos) if two speakers detected, else None
        
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
            return None, None

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
        single_face_data: list[tuple[int, float]] = []  # Track (frame_index, position) for single faces
        multi_face_data: list[tuple[float, float]] = []  # Track (left_pos, right_pos) from multi-face frames

        for index in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            faces = self._detect_faces(frame)
            # Drop tiny detections (<0.2% area)
            faces = [
                f for f in faces
                if f.get("area", 0.0) >= 0.0005 * f.get("width", 1.0) * f.get("height", 1.0)
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
            # - Multiple faces: check size ratio first
            #   - If one face is 3x+ larger: treat as single face (the larger one is main subject)
            #   - If similar sizes and span < 0.40: center between extremes (show both)
            #   - If similar sizes and span >= 0.40: track positions for primary speaker detection
            if len(faces) >= 2:
                frame_width = faces[0].get("width", 1.0)
                
                # Sort faces by area (largest first)
                faces_sorted = sorted(faces, key=lambda f: f["area"], reverse=True)
                largest_face = faces_sorted[0]
                second_face = faces_sorted[1]
                
                # Check if largest face is significantly bigger (3x+ area)
                size_ratio = largest_face["area"] / max(second_face["area"], 1.0)
                
                if size_ratio >= 3.0:
                    # Largest face is dominant - focus on it, ignore smaller ones
                    center_ratio = largest_face["center_x"] / frame_width
                    weight = largest_face["score"] * largest_face["area"]
                    single_face_data.append((index, center_ratio))
                    weighted_centers.append((center_ratio, weight))
                    logger.info(
                        "  Multi-face but one dominant (size_ratio=%.1fx): focus on largest at %.3f",
                        size_ratio,
                        center_ratio,
                    )
                else:
                    # Similar sized faces - use position-based logic
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
                        # Faces too far apart - track both positions for primary speaker detection
                        left_pos = min(centers)
                        right_pos = max(centers)
                        multi_face_data.append((left_pos, right_pos))
                        logger.info(
                            "  Multi-face (span=%.3f >= 0.40): track positions left=%.3f right=%.3f",
                            span,
                            left_pos,
                            right_pos,
                        )
            else:
                # Single face - track for identifying primary speaker position
                best_face = max(faces, key=lambda f: f["score"] * f["area"])
                center_ratio = best_face["center_x"] / best_face["width"]
                weight = best_face["score"] * best_face["area"]
                single_face_data.append((index, center_ratio))
                weighted_centers.append((center_ratio, weight))

        capture.release()

        if not weighted_centers:
            logger.info("InsightFace: no faces found in %s", video_path)
            return None, None

        # Check if we have two speakers on different positions
        # Use multi-face data (wide span frames) to identify left/right speaker positions
        two_speaker_mode = False
        left_speaker_pos: float | None = None
        right_speaker_pos: float | None = None
        
        if multi_face_data:
            # We have frames showing both speakers far apart
            # But only trust this if we have enough multi-face samples
            # compared to single-face samples (at least 30% of total)
            total_samples = len(multi_face_data) + len(single_face_data)
            multi_face_ratio = len(multi_face_data) / max(total_samples, 1)
            
            if multi_face_ratio >= 0.3 or len(multi_face_data) >= 2:
                # Enough evidence of two speakers
                left_positions = [left for left, right in multi_face_data]
                right_positions = [right for left, right in multi_face_data]
                left_speaker_pos = sum(left_positions) / len(left_positions)
                right_speaker_pos = sum(right_positions) / len(right_positions)
                two_speaker_mode = True
                
                logger.info(
                    "Detected two-speaker setup from %d multi-face frames (%.0f%% of samples): left=%.3f, right=%.3f",
                    len(multi_face_data),
                    multi_face_ratio * 100,
                    left_speaker_pos,
                    right_speaker_pos,
                )
            else:
                # Not enough multi-face frames - single-face data is more reliable
                # Use weighted centers from single-face frames instead
                logger.info(
                    "Only %d multi-face frame(s) (%.0f%% of samples) - not enough for two-speaker mode, using single-face data",
                    len(multi_face_data),
                    multi_face_ratio * 100,
                )
        elif len(single_face_data) >= 3:
            # Fallback: check if single-face positions span across frame
            single_positions = [pos for idx, pos in single_face_data]
            min_pos = min(single_positions)
            max_pos = max(single_positions)
            position_span = max_pos - min_pos
            
            if position_span > 0.35:
                two_speaker_mode = True
                midpoint = (min_pos + max_pos) / 2.0
                left_positions = [p for p in single_positions if p < midpoint]
                right_positions = [p for p in single_positions if p >= midpoint]
                
                if left_positions and right_positions:
                    left_speaker_pos = sum(left_positions) / len(left_positions)
                    right_speaker_pos = sum(right_positions) / len(right_positions)
                    logger.info(
                        "Detected two-speaker setup from single-face span (%.3f): left=%.3f, right=%.3f",
                        position_span,
                        left_speaker_pos,
                        right_speaker_pos,
                    )
        
        # If two-speaker mode, determine primary speaker and focus on them
        # But also consider single-face weighted centers to avoid extreme positions
        if two_speaker_mode and left_speaker_pos is not None and right_speaker_pos is not None:
            primary_speaker = self._get_primary_speaker(dialogue, segment_start, segment_end)
            
            # Calculate average position from single-face frames (if any)
            single_face_avg = None
            if weighted_centers:
                num = sum(c * w for c, w in weighted_centers)
                den = sum(w for _, w in weighted_centers)
                if den > 0:
                    single_face_avg = num / den
            
            if primary_speaker and dialogue:
                # Calculate which position (left or right) corresponds to primary speaker
                primary_interval_frames = self._get_speaker_frame_indices(
                    dialogue, primary_speaker, segment_start, frame_count, fps
                )
                
                # Determine primary speaker position by comparing to left/right
                primary_single_faces = []
                for frame_idx, pos in single_face_data:
                    if frame_idx in primary_interval_frames:
                        primary_single_faces.append(pos)
                
                if primary_single_faces:
                    avg_primary_pos = sum(primary_single_faces) / len(primary_single_faces)
                    
                    # Is primary speaker closer to left or right position?
                    if abs(avg_primary_pos - left_speaker_pos) < abs(avg_primary_pos - right_speaker_pos):
                        focus = left_speaker_pos
                        side = "LEFT"
                    else:
                        focus = right_speaker_pos
                        side = "RIGHT"
                    
                    # Blend with single-face average to avoid extreme positions
                    # If we have many single-face samples, trust them more
                    if single_face_avg is not None and len(weighted_centers) >= 2:
                        blend_weight = min(0.4, len(weighted_centers) / 10.0)  # Up to 40% blend
                        focus = (1 - blend_weight) * focus + blend_weight * single_face_avg
                        logger.info(
                            "Primary speaker '%s' on %s (%.3f), blended %.0f%% with single-face avg (%.3f) -> focus=%.3f",
                            primary_speaker,
                            side,
                            left_speaker_pos if side == "LEFT" else right_speaker_pos,
                            blend_weight * 100,
                            single_face_avg,
                            focus,
                        )
                    else:
                        logger.info(
                            "Primary speaker '%s' (%.0f%% speech) identified on %s (%.3f), focusing there",
                            primary_speaker,
                            self._get_speaker_percentage(dialogue, primary_speaker, segment_start, segment_end),
                            side,
                            focus,
                        )
                    
                    focus = float(max(0.2, min(0.8, focus)))
                    return focus, (left_speaker_pos, right_speaker_pos)
            
            # Fallback: use center between both speakers, blended with single-face avg
            focus = (left_speaker_pos + right_speaker_pos) / 2.0
            if single_face_avg is not None:
                focus = 0.5 * focus + 0.5 * single_face_avg
                logger.info(
                    "No primary speaker, blending center (%.3f) with single-face avg (%.3f) -> focus=%.3f",
                    (left_speaker_pos + right_speaker_pos) / 2.0,
                    single_face_avg,
                    focus,
                )
            else:
                logger.info("Unable to identify primary speaker, using center=%.3f", focus)
            focus = float(max(0.2, min(0.8, focus)))
            return focus, (left_speaker_pos, right_speaker_pos)

        # Normal weighted average calculation
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
            return 0.5, None

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
        return focus, None

    # ------------------------------------------------------------------ #
    # Scene detection helper for dynamic timeline splitting
    # ------------------------------------------------------------------ #
    def _detect_scene_changes(
        self,
        video_path: str,
        segment_start: float = 0.0,
        segment_end: float | None = None,
    ) -> list[float]:
        """
        Detect abrupt scene changes (camera cuts) using TransNetV2 neural network.
        
        TransNetV2 is a deep learning model specifically designed for shot boundary
        detection with F1=96.2% on BBC Planet Earth benchmark.
        https://github.com/soCzech/TransNetV2
        
        Args:
            video_path: Path to video file
            segment_start: Start time in seconds
            segment_end: End time in seconds (None = full video)
        
        Returns:
            List of timestamps (in seconds, relative to segment_start) where scene changes occur
        """
        try:
            from backend.services.transnet_detector import get_transnet_detector
            
            detector = get_transnet_detector()
            scene_changes = detector.detect_scenes(
                video_path,
                segment_start=segment_start,
                segment_end=segment_end,
            )
            
            logger.info(
                "TransNetV2: detected %d scene changes in [%.1f, %.1f]",
                len(scene_changes), segment_start, segment_end or 0
            )
            return scene_changes
            
        except ImportError as e:
            logger.warning("TransNetV2 not available: %s, falling back to face-jump detection", e)
            return []
        except Exception as exc:
            logger.warning("TransNetV2 scene detection failed for %s: %s", video_path, exc)
            return []

    # ------------------------------------------------------------------ #
    # Face-jump detection (camera switch detection via face position)
    # ------------------------------------------------------------------ #
    def _detect_face_jumps(
        self,
        video_path: str,
        segment_start: float = 0.0,
        segment_end: float | None = None,
        sample_interval: float = 0.15,  # Sample every 150ms
        jump_threshold: float = 0.25,   # 25% position change = camera switch
    ) -> list[float]:
        """
        Detect camera switches by looking for sudden jumps in face position.
        
        If face.center_x changes by more than jump_threshold between consecutive
        samples, it's likely a camera switch.
        
        Returns:
            List of timestamps where face jumps were detected
        """
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            return []
        
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        if (not math.isfinite(fps)) or fps <= 1e-3:
            fps = 25.0
        duration = frame_count / fps
        
        if segment_end is None:
            segment_end = duration
        
        # Sample frames at regular intervals
        sample_times = []
        t = 0.0
        while t < duration:
            sample_times.append(t)
            t += sample_interval
        
        # Detect faces at each sample point
        face_positions: list[tuple[float, float | None]] = []  # (time, position)
        
        for sample_time in sample_times:
            frame_idx = int(sample_time * fps)
            if frame_idx >= frame_count:
                break
            
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok or frame is None:
                face_positions.append((sample_time, None))
                continue
            
            faces = self._detect_faces(frame)
            if not faces:
                face_positions.append((sample_time, None))
                continue
            
            # Use largest face
            best_face = max(faces, key=lambda f: f.get("area", 0))
            frame_width = best_face.get("width", 1.0)
            position = best_face["center_x"] / max(frame_width, 1.0)
            face_positions.append((sample_time, position))
        
        capture.release()
        
        # Find jumps
        jump_times: list[float] = []
        prev_pos: float | None = None
        prev_time: float = 0.0
        
        for sample_time, position in face_positions:
            if position is None:
                continue
            
            if prev_pos is not None:
                diff = abs(position - prev_pos)
                if diff >= jump_threshold:
                    # Jump detected!
                    # Use midpoint between samples as the jump time
                    jump_time = (prev_time + sample_time) / 2.0
                    jump_times.append(jump_time)
                    logger.info(
                        "Face-jump detected at %.2fs: position %.3f -> %.3f (diff=%.3f)",
                        jump_time, prev_pos, position, diff
                    )
            
            prev_pos = position
            prev_time = sample_time
        
        logger.info(
            "Face-jump detection: found %d jumps in [%.1f, %.1f]s",
            len(jump_times), segment_start, segment_end or duration
        )
        
        return jump_times

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Interview-optimized focus timeline (TransNetV2 first, then SCRFD)
    # ------------------------------------------------------------------ #
    
    # Constants for focus detection
    NUM_DETECTIONS_PER_SCENE = 15  # 15 SCRFD detections spread EVENLY across entire scene
    CROP_WIDTH_PX = 1080  # Target crop width for 9:16 vertical video
    
    def build_focus_timeline(
        self,
        video_path: str,
        dialogue: list[dict] | None = None,
        segment_start: float = 0.0,
        segment_end: float | None = None,
        sample_period: float = 0.33,  # ignored
    ) -> list[dict]:
        """
        Build focus timeline optimized for interview videos (3-camera setup).
        
        ALGORITHM:
        1. TransNetV2 detects scene changes (wide shot, close-up A, close-up B)
        2. For each scene: 12 SCRFD detections every 2 frames (~1 sec coverage)
        3. Decision logic:
           - 1 face → center on it
           - 2 faces, fit in 1080px → center between them  
           - 2 faces, DON'T fit → SPEAKER-AWARE (focus on who's speaking)
        """
        logger.info(
            "build_focus_timeline: video=%s, dialogue_len=%d, segment=[%.2f, %s]",
            video_path, len(dialogue) if dialogue else 0, segment_start, segment_end
        )
        
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            logger.warning("Cannot open video %s", video_path)
            return []

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        if not math.isfinite(fps) or fps <= 1e-3:
            fps = 25.0
        duration = frame_count / fps
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        
        # Calculate scale factor (video is scaled to fit 1920px height)
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        scale_factor = 1920.0 / frame_height if frame_height > 0 else 1.0
        scaled_width = int(frame_width * scale_factor)
        
        # When face IS detected, allow wide range but keep margin for face width
        # (no longer using tight 0.30-0.70 bounds that pushed faces to edges)
        safe_min = 0.15  # Leave room for face width on left
        safe_max = 0.85  # Leave room for face width on right
        
        # ============================================================
        # STEP 1: TransNetV2 scene detection
        # NOTE: video_path is already the CUT segment (starts at 0), so use local time
        # ============================================================
        scene_changes = self._detect_scene_changes(video_path, 0.0, None)
        scene_boundaries = [0.0] + scene_changes + [duration]
        
        logger.info(
            "TransNetV2: %d scenes detected, boundaries: %s",
            len(scene_boundaries) - 1,
            [f"{t:.2f}s" for t in scene_boundaries]
        )
        
        # ============================================================
        # STEP 2: For each scene, do 7 consecutive detections at START
        # ============================================================
        segments: list[dict] = []
        last_valid_focus = 0.5
        
        # Build speaker timeline for SPEAKER-AWARE mode
        speaker_at_time: dict[float, str] = {}
        if dialogue:
            for turn in dialogue:
                speaker = turn.get("speaker")
                t_start = turn.get("start", 0.0) - segment_start
                t_end = turn.get("end", 0.0) - segment_start
                if speaker and t_end > t_start:
                    for t in range(int(t_start * 10), int(t_end * 10)):
                        speaker_at_time[t / 10.0] = speaker
        
        for scene_idx in range(len(scene_boundaries) - 1):
            scene_start_t = scene_boundaries[scene_idx]
            scene_end_t = scene_boundaries[scene_idx + 1]
            scene_duration = scene_end_t - scene_start_t
            
            # Spread samples EVENLY across the ENTIRE scene duration
            all_faces: list[list[dict]] = []
            
            for det_idx in range(self.NUM_DETECTIONS_PER_SCENE):
                # Calculate sample position: evenly spread across scene
                if self.NUM_DETECTIONS_PER_SCENE > 1:
                    progress = det_idx / (self.NUM_DETECTIONS_PER_SCENE - 1)
                else:
                    progress = 0.5
                sample_time = scene_start_t + progress * scene_duration
                frame_idx = int(sample_time * fps)
                if frame_idx >= frame_count:
                    break
                    
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                    
                faces = self._detect_faces(frame)
                # Filter tiny faces
                faces = [
                    f for f in faces
                    if f.get("w", 0) >= self.MIN_FACE_WIDTH_PX
                    and f.get("area", 0) >= 0.0005 * f.get("width", 1) * f.get("height", 1)
                ]
                if faces:
                    all_faces.append(faces)
            
            # ============================================================
            # STEP 3: Analyze detections and decide focus
            # ============================================================
            if not all_faces:
                # No faces detected → keep previous focus
                scene_focus = last_valid_focus
                logger.info("Scene %d [%.2f-%.2f]: NO faces → fallback %.3f", 
                           scene_idx, scene_start_t, scene_end_t, scene_focus)
            else:
                # Count faces per detection
                face_counts = [len(faces) for faces in all_faces]
                most_common_count = max(set(face_counts), key=face_counts.count)
                
                # Get all face positions (normalized 0-1)
                all_positions: list[float] = []
                for faces in all_faces:
                    for f in faces:
                        pos = f["center_x"] / frame_width  # FIX: divide by ORIGINAL frame width!
                        all_positions.append(pos)
                
                if most_common_count == 1:
                    # === SINGLE FACE (close-up) → center on it ===
                    # Use safe bounds (wider range) since we KNOW where the face is
                    avg_pos = sum(all_positions) / len(all_positions)
                    scene_focus = max(safe_min, min(safe_max, avg_pos))
                    logger.info(
                        "Scene %d [%.2f-%.2f]: 1 FACE (close-up) → focus=%.3f",
                        scene_idx, scene_start_t, scene_end_t, scene_focus
                    )
                    
                elif most_common_count >= 2:
                    # === TWO FACES (wide shot) → check if they fit ===
                    # Get leftmost and rightmost positions
                    left_pos = min(all_positions)
                    right_pos = max(all_positions)
                    
                    # Calculate if both faces fit in 1080px crop
                    # Positions are normalized (0-1), convert to pixels in scaled video
                    left_px = left_pos * scaled_width
                    right_px = right_pos * scaled_width
                    span_px = right_px - left_px
                    
                    # Add margin for face width (~100px each side)
                    required_width = span_px + 200
                    
                    if required_width <= self.CROP_WIDTH_PX:
                        # === BOTH FIT → center between them ===
                        # Use safe bounds since we know where faces are
                        center_pos = (left_pos + right_pos) / 2.0
                        scene_focus = max(safe_min, min(safe_max, center_pos))
                        logger.info(
                            "Scene %d [%.2f-%.2f]: 2 FACES FIT (span=%dpx < %dpx) → center=%.3f",
                            scene_idx, scene_start_t, scene_end_t, 
                            int(required_width), self.CROP_WIDTH_PX, scene_focus
                        )
                    else:
                        # === DON'T FIT → SPEAKER-AWARE mode ===
                        # Find who's speaking at scene start
                        speaking_speaker = speaker_at_time.get(round(scene_start_t, 1))
                        
                        if speaking_speaker and dialogue:
                            # Find speaker position from detection
                            # Assume left speaker is SPEAKER_00, right is SPEAKER_01
                            # (common interview convention)
                            # Use safe bounds since we know face positions
                            if "00" in speaking_speaker or "0" == speaking_speaker[-1]:
                                scene_focus = max(safe_min, min(safe_max, left_pos))
                                logger.info(
                                    "Scene %d [%.2f-%.2f]: 2 FACES DON'T FIT → SPEAKER-AWARE: %s (left) → focus=%.3f",
                                    scene_idx, scene_start_t, scene_end_t, speaking_speaker, scene_focus
                                )
                            else:
                                scene_focus = max(safe_min, min(safe_max, right_pos))
                                logger.info(
                                    "Scene %d [%.2f-%.2f]: 2 FACES DON'T FIT → SPEAKER-AWARE: %s (right) → focus=%.3f",
                                    scene_idx, scene_start_t, scene_end_t, speaking_speaker, scene_focus
                                )
                        else:
                            # No diarization → pick side with LARGER face (or right if equal)
                            # Collect face sizes for left and right clusters
                            left_sizes: list[float] = []
                            right_sizes: list[float] = []
                            mid_pos = (left_pos + right_pos) / 2.0
                            
                            for faces in all_faces:
                                for f in faces:
                                    pos = f["center_x"] / frame_width  # FIX: divide by ORIGINAL frame width!
                                    face_w = f.get("face_w", 50)  # face width in pixels
                                    if pos < mid_pos:
                                        left_sizes.append(face_w)
                                    else:
                                        right_sizes.append(face_w)
                            
                            avg_left = sum(left_sizes) / len(left_sizes) if left_sizes else 0
                            avg_right = sum(right_sizes) / len(right_sizes) if right_sizes else 0
                            
                            # Use safe bounds since we know face positions
                            if avg_left > avg_right * 1.2:  # Left is significantly larger
                                scene_focus = max(safe_min, min(safe_max, left_pos))
                                logger.info(
                                    "Scene %d [%.2f-%.2f]: 2 FACES DON'T FIT, NO DIARIZATION → LARGER LEFT (%.0f vs %.0f) → focus=%.3f",
                                    scene_idx, scene_start_t, scene_end_t, avg_left, avg_right, scene_focus
                                )
                            else:
                                # Right is larger or equal → pick right (default)
                                scene_focus = max(safe_min, min(safe_max, right_pos))
                                logger.info(
                                    "Scene %d [%.2f-%.2f]: 2 FACES DON'T FIT, NO DIARIZATION → RIGHT (%.0f vs %.0f) → focus=%.3f",
                                    scene_idx, scene_start_t, scene_end_t, avg_left, avg_right, scene_focus
                                )
                else:
                    scene_focus = last_valid_focus
                    
                last_valid_focus = scene_focus
            
            segments.append({
                "start": scene_start_t,
                "end": scene_end_t,
                "focus": scene_focus,
            })
        
        capture.release()
        
        logger.info("Built focus timeline: %d segments", len(segments))
        return segments

    # ------------------------------------------------------------------ #
    # Vertical focus timeline (y-axis) for better framing of heads       #
    # ------------------------------------------------------------------ #
    def build_vertical_focus_timeline(
        self,
        video_path: str,
        segment_start: float = 0.0,
        segment_end: float | None = None,
        sample_period: float = 0.15,
    ) -> list[dict]:
        """
        Build a coarse per-time vertical focus timeline (0..1 from top to bottom).
        - 1 face: use its vertical center
        - 2+ faces: use min/max centers averaged (keep both)
        - 0 faces: keep last
        """
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            logger.warning("InsightFace: unable to open video %s for vertical timeline", video_path)
            return []

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        if frame_count <= 0:
            frame_count = 1
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        if (not math.isfinite(fps)) or fps <= 1e-3:
            fps = 25.0
        duration = frame_count / fps

        sample_step = max(1, int(sample_period * fps))
        timeline_raw: list[tuple[float, float | None]] = []

        for frame_idx in range(0, frame_count, sample_step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok or frame is None:
                timeline_raw.append((frame_idx / fps, None))
                continue
            faces = self._detect_faces(frame)
            # Filter tiny detections (already done in _detect_faces but double-check)
            faces = [
                f for f in faces
                if f.get("area", 0.0) >= 0.0005 * f.get("width", 1.0) * f.get("height", 1.0)
                and f.get("w", 0.0) >= self.MIN_FACE_WIDTH_PX
            ]
            if not faces:
                timeline_raw.append((frame_idx / fps, None))
                continue
            
            # If one face is much larger, use only that one for vertical focus
            if len(faces) >= 2:
                faces_sorted = sorted(faces, key=lambda f: f["area"], reverse=True)
                size_ratio = faces_sorted[0]["area"] / max(faces_sorted[1]["area"], 1.0)
                if size_ratio >= 3.0:
                    faces = [faces_sorted[0]]  # Use only the dominant face
            
            frame_h = faces[0].get("height", frame.shape[0])
            centers_y = [f["center_y"] / frame_h for f in faces]
            focus_y = (min(centers_y) + max(centers_y)) / 2.0
            focus_y = float(max(0.10, min(0.90, focus_y)))
            timeline_raw.append((frame_idx / fps, focus_y))

        capture.release()

        if not timeline_raw:
            return []

        # Fill gaps: keep last
        filled: list[tuple[float, float]] = []
        last = 0.5
        for ts, val in timeline_raw:
            if val is None:
                filled.append((ts, last))
            else:
                last = val
                filled.append((ts, val))

        # Smooth window 3
        smoothed: list[tuple[float, float]] = []
        window: list[float] = []
        for ts, v in filled:
            window.append(v)
            if len(window) > 3:
                window.pop(0)
            smoothed.append((ts, sum(window) / len(window)))

        # Merge if small change
        merged: list[dict] = []
        seg_start = smoothed[0][0]
        seg_focus = smoothed[0][1]
        threshold = 0.08

        for i in range(1, len(smoothed)):
            ts, v = smoothed[i]
            if abs(v - seg_focus) <= threshold:
                continue
            merged.append({"start": seg_start, "end": ts, "focus_y": seg_focus})
            seg_start = ts
            seg_focus = v
        merged.append({"start": seg_start, "end": duration, "focus_y": seg_focus})

        # Detect scene changes and split segments at those boundaries
        # NOTE: video_path is already the CUT segment (starts at 0), so use local time
        scene_changes = self._detect_scene_changes(video_path, 0.0, None)
        
        if scene_changes:
            split_segments: list[dict] = []
            for seg in merged:
                seg_start_time = seg["start"]
                seg_end_time = seg["end"]
                seg_focus_val = seg["focus_y"]
                
                # Find scene changes within this segment
                cuts_in_seg = [sc for sc in scene_changes if seg_start_time < sc < seg_end_time]
                
                if not cuts_in_seg:
                    split_segments.append(seg)
                else:
                    # Split segment at each scene change
                    cuts_sorted = sorted(cuts_in_seg)
                    prev_cut = seg_start_time
                    for cut_ts in cuts_sorted:
                        split_segments.append({"start": prev_cut, "end": cut_ts, "focus_y": seg_focus_val})
                        prev_cut = cut_ts
                    # Add final sub-segment
                    split_segments.append({"start": prev_cut, "end": seg_end_time, "focus_y": seg_focus_val})
            
            merged = split_segments

        # Filter very short
        cleaned: list[dict] = []
        for seg in merged:
            if cleaned and (seg["end"] - seg["start"]) < 1.2:
                prev = cleaned[-1]
                prev["end"] = seg["end"]
            else:
                cleaned.append(seg)

        return cleaned
    
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
    
    def _get_speaker_percentage(
        self,
        dialogue: list[dict] | None,
        speaker: str,
        segment_start: float,
        segment_end: float | None,
    ) -> float:
        """Calculate what percentage of segment this speaker talks."""
        if not dialogue:
            return 0.0
        
        if segment_end is None:
            segment_end = max((turn.get("end", segment_start) for turn in dialogue), default=segment_start + 30.0)
        
        speaker_duration = 0.0
        total_duration = 0.0
        
        for turn in dialogue:
            turn_start = turn.get("start", 0.0)
            turn_end = turn.get("end", 0.0)
            overlap_start = max(turn_start, segment_start)
            overlap_end = min(turn_end, segment_end)
            overlap_duration = max(0.0, overlap_end - overlap_start)
            
            if overlap_duration > 0:
                total_duration += overlap_duration
                if turn.get("speaker") == speaker:
                    speaker_duration += overlap_duration
        
        if total_duration <= 0:
            return 0.0
        
        return (speaker_duration / total_duration) * 100.0
    
    def _get_speaker_frame_indices(
        self,
        dialogue: list[dict] | None,
        speaker: str,
        segment_start: float,
        frame_count: int,
        fps: float,
    ) -> set[int]:
        """Get frame indices that fall within this speaker's speech intervals."""
        if not dialogue:
            return set()
        
        speaker_frames = set()
        
        for turn in dialogue:
            if turn.get("speaker") != speaker:
                continue
            
            # Convert absolute time to relative frame indices
            turn_start = turn.get("start", 0.0)
            turn_end = turn.get("end", 0.0)
            rel_start = max(0.0, turn_start - segment_start)
            rel_end = max(0.0, turn_end - segment_start)
            
            start_frame = int(rel_start * fps)
            end_frame = int(rel_end * fps)
            
            for frame_idx in range(start_frame, min(end_frame, frame_count)):
                speaker_frames.add(frame_idx)
        
        return speaker_frames
    
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
