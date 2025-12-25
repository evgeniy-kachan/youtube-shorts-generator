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
        det_thresh: float = 0.30,  # Balance: catch profiles but reduce noise
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
            self._detector.prepare(ctx_id=self.ctx_id, det_thresh=self.det_thresh, det_size=(1280, 1280), nms_thresh=0.45)
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
    # Speaker-aware focus timeline (follows the speaking person)
    # ------------------------------------------------------------------ #
    def build_focus_timeline(
        self,
        video_path: str,
        dialogue: list[dict] | None = None,
        segment_start: float = 0.0,
        segment_end: float | None = None,
        sample_period: float = 0.33,  # ignored in new approach
    ) -> list[dict]:
        """
        Build focus timeline that FOLLOWS THE SPEAKING PERSON.
        
        SPEAKER-AWARE APPROACH:
        1. Use diarization data (dialogue) to find speaker segments
        2. Group consecutive segments by speaker
        3. For each speaker group, sample 2 frames to find their position
        4. Focus follows whoever is speaking
        
        Falls back to scene-based sampling if no dialogue data.
        """
        # === DIAGNOSTIC LOGGING ===
        dialogue_len = len(dialogue) if dialogue else 0
        dialogue_preview = []
        if dialogue and len(dialogue) > 0:
            for i, turn in enumerate(dialogue[:3]):  # first 3 turns
                dialogue_preview.append({
                    "speaker": turn.get("speaker"),
                    "start": turn.get("start"),
                    "end": turn.get("end"),
                })
        logger.info(
            "build_focus_timeline called: dialogue_len=%d, segment_start=%.2f, segment_end=%s, preview=%s",
            dialogue_len,
            segment_start,
            segment_end,
            dialogue_preview,
        )
        # === END DIAGNOSTIC ===
        
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            logger.warning("InsightFace: unable to open video %s for timeline", video_path)
            return []

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        if frame_count <= 0:
            frame_count = 1
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        if (not math.isfinite(fps)) or fps <= 1e-3:
            fps = 25.0
        duration = frame_count / fps

        # Helper to compute focus from faces
        min_bound = 0.20
        max_bound = 0.80

        def _focus_for_faces(faces: list) -> Optional[float]:
            if not faces:
                return None
            faces_f = [
                f for f in faces
                if f.get("area", 0.0) >= 0.0005 * f.get("width", 1.0) * f.get("height", 1.0)
                and f.get("w", 0.0) >= self.MIN_FACE_WIDTH_PX
            ]
            if not faces_f:
                return None

            frame_width = max(1e-3, faces_f[0].get("width", 1.0))

            if len(faces_f) == 1:
                f = faces_f[0]
                return float(max(min_bound, min(max_bound, f["center_x"] / frame_width)))

            # Multiple faces: check if one is much larger
            faces_sorted = sorted(faces_f, key=lambda f: f["area"], reverse=True)
            largest = faces_sorted[0]
            second = faces_sorted[1]
            size_ratio = largest["area"] / max(second["area"], 1.0)
            
            if size_ratio >= 3.0:
                return float(max(min_bound, min(max_bound, largest["center_x"] / frame_width)))

            # Multiple similar-sized faces: center between them
            centers = [f["center_x"] / frame_width for f in faces_f]
            center_avg = (min(centers) + max(centers)) / 2.0
            return float(max(min_bound, min(max_bound, center_avg)))

        def _sample_focus_at_time(t: float) -> Optional[float]:
            """Sample face focus at a specific time."""
            frame_idx = int(t * fps)
            if frame_idx >= frame_count:
                return None
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok or frame is None:
                return None
            faces = self._detect_faces(frame)
            return _focus_for_faces(faces)

        # ============================================================
        # SPEAKER-AWARE MODE: Use diarization to follow the speaker
        # ============================================================
        if dialogue and len(dialogue) >= 2:
            logger.info("Using SPEAKER-AWARE focus (diarization data available)")
            
            # Pre-calculate speech duration for each speaker (for overlap handling)
            def _get_speaker_duration_after(time_point: float) -> dict[str, float]:
                """Calculate total speech duration for each speaker AFTER a given time."""
                durations: dict[str, float] = {}
                for seg in dialogue:
                    seg_start = seg.get("start", 0.0) - segment_start
                    seg_end = seg.get("end", 0.0) - segment_start
                    if seg_end <= time_point:
                        continue  # This segment is before our time point
                    speaker = seg.get("speaker", "unknown")
                    # Only count the part after time_point
                    effective_start = max(seg_start, time_point)
                    effective_duration = max(0.0, seg_end - effective_start)
                    durations[speaker] = durations.get(speaker, 0.0) + effective_duration
                return durations
            
            # Group consecutive dialogue segments by speaker
            speaker_groups: list[dict] = []
            current_speaker = None
            group_start = 0.0
            
            for seg in dialogue:
                speaker = seg.get("speaker", "unknown")
                seg_start = seg.get("start", 0.0) - segment_start  # relative to clip
                seg_end = seg.get("end", 0.0) - segment_start
                
                # Clamp to clip boundaries
                seg_start = max(0.0, seg_start)
                seg_end = min(duration, seg_end)
                
                if seg_start >= seg_end:
                    continue
                
                if speaker != current_speaker:
                    # New speaker - close previous group
                    if current_speaker is not None and group_start < seg_start:
                        speaker_groups.append({
                            "speaker": current_speaker,
                            "start": group_start,
                            "end": seg_start,
                        })
                    current_speaker = speaker
                    group_start = seg_start
            
            # Close last group
            if current_speaker is not None:
                speaker_groups.append({
                    "speaker": current_speaker,
                    "start": group_start,
                    "end": duration,
                })
            
            logger.info(
                "Found %d speaker groups: %s",
                len(speaker_groups),
                [(g["speaker"], f"{g['start']:.1f}-{g['end']:.1f}s") for g in speaker_groups]
            )
            
            # Build speaker position cache: try multiple samples to find each speaker's position
            speaker_positions: dict[str, float] = {}
            
            # Collect ALL segments for each speaker (to sample from multiple places)
            speaker_all_segments: dict[str, list[tuple[float, float]]] = {}
            for group in speaker_groups:
                speaker = group["speaker"]
                if speaker not in speaker_all_segments:
                    speaker_all_segments[speaker] = []
                speaker_all_segments[speaker].append((group["start"], group["end"]))
            
            # For each speaker, try up to 10 samples across all their segments
            MAX_SAMPLES_PER_SPEAKER = 10
            SAMPLE_OFFSETS = [0.1, 0.3, 0.5, 0.7, 0.9]  # Sample at different points
            
            for speaker, segs in speaker_all_segments.items():
                samples_found: list[float] = []
                
                for seg_start, seg_end in segs:
                    if len(samples_found) >= MAX_SAMPLES_PER_SPEAKER:
                        break
                    seg_dur = seg_end - seg_start
                    
                    for offset in SAMPLE_OFFSETS:
                        if len(samples_found) >= MAX_SAMPLES_PER_SPEAKER:
                            break
                        sample_time = seg_start + offset * seg_dur
                        focus_val = _sample_focus_at_time(sample_time)
                        if focus_val is not None:
                            samples_found.append(focus_val)
                            logger.debug(
                                "Speaker '%s': sample at t=%.2fs -> focus=%.3f",
                                speaker, sample_time, focus_val
                            )
                
                if samples_found:
                    speaker_positions[speaker] = sum(samples_found) / len(samples_found)
                    logger.info(
                        "Speaker '%s' position: %.3f (from %d samples)",
                        speaker, speaker_positions[speaker], len(samples_found)
                    )
                else:
                    logger.warning("Speaker '%s': NO face detected in any sample!", speaker)
            
            # FALLBACK: If a speaker has no position, infer from OPPOSITE side of known speakers
            known_positions = list(speaker_positions.values())
            if known_positions:
                known_avg = sum(known_positions) / len(known_positions)
                
                for speaker in speaker_all_segments.keys():
                    if speaker not in speaker_positions:
                        # Infer: if known speakers are on left, this one is probably on right
                        if known_avg < 0.50:
                            inferred_pos = 0.75  # Right side
                        else:
                            inferred_pos = 0.30  # Left side
                        
                        speaker_positions[speaker] = inferred_pos
                        logger.info(
                            "Speaker '%s': INFERRED position %.3f (opposite of known avg %.3f)",
                            speaker, inferred_pos, known_avg
                        )
            
            # Handle short segments (likely overlaps): keep focus on dominant speaker
            MIN_SEGMENT_DURATION = 3.0  # Segments shorter than 3s → merge to reduce jitter
            
            # Minimum position difference to switch focus (avoid jerky transitions)
            MIN_FOCUS_DIFF = 0.12  # Don't switch if positions differ by less than 12%
            
            # Find the dominant speaker (most total speech)
            total_durations = _get_speaker_duration_after(0.0)
            dominant_speaker = max(total_durations, key=total_durations.get) if total_durations else None
            
            logger.info(
                "Speaker durations: %s, dominant: %s",
                {k: f"{v:.1f}s" for k, v in total_durations.items()},
                dominant_speaker
            )
            
            # Build focus timeline from speaker groups
            # Short segments (overlaps) → keep focus on speaker with MORE text after
            segments: list[dict] = []
            last_focus = 0.5
            last_speaker = None
            
            for i, group in enumerate(speaker_groups):
                speaker = group["speaker"]
                seg_duration = group["end"] - group["start"]
                
                # Check if this is a short segment (likely overlap/interruption)
                if seg_duration < MIN_SEGMENT_DURATION:
                    # Look ahead: who speaks more after this segment?
                    future_durations = _get_speaker_duration_after(group["end"])
                    
                    # If the previous speaker has more upcoming speech, keep focus on them
                    if last_speaker and future_durations.get(last_speaker, 0) >= future_durations.get(speaker, 0):
                        logger.info(
                            "Short segment [%.2f-%.2f] (%.1fs): '%s' interrupted, keeping focus on '%s'",
                            group["start"], group["end"], seg_duration, speaker, last_speaker
                        )
                        # Extend previous segment instead of creating new one
                        if segments:
                            segments[-1]["end"] = group["end"]
                        continue
                    else:
                        logger.info(
                            "Short segment [%.2f-%.2f] (%.1fs): '%s' becomes dominant after",
                            group["start"], group["end"], seg_duration, speaker
                        )
                
                # Normal segment - use speaker's position
                new_focus = speaker_positions.get(speaker, last_focus)
                
                # Check if position difference is too small to bother switching
                focus_diff = abs(new_focus - last_focus)
                if focus_diff < MIN_FOCUS_DIFF and segments:
                    # Positions are too close - don't switch, extend previous segment
                    logger.info(
                        "Segment [%.2f-%.2f]: '%s' focus=%.3f too close to previous %.3f (diff=%.3f < %.2f), keeping previous",
                        group["start"], group["end"], speaker, new_focus, last_focus, focus_diff, MIN_FOCUS_DIFF
                    )
                    segments[-1]["end"] = group["end"]
                    continue
                
                focus = new_focus
                
                segments.append({
                    "start": group["start"],
                    "end": group["end"],
                    "focus": focus,
                })
                last_focus = focus
                last_speaker = speaker
                
                logger.info(
                    "Segment [%.2f-%.2f]: speaker '%s' -> focus=%.3f",
                    group["start"], group["end"], speaker, focus
                )
            
            capture.release()
            
            if segments:
                # ============================================================
                # DETECT CAMERA SWITCHES via TransNetV2 (neural network)
                # ============================================================
                scene_changes = self._detect_scene_changes(video_path, segment_start, segment_end)
                
                if scene_changes:
                    # Split segments at camera switch points (TransNetV2 detected cuts)
                    split_segments: list[dict] = []
                    
                    for seg in segments:
                        seg_start = seg["start"]
                        seg_end = seg["end"]
                        seg_focus = seg["focus"]
                        
                        # Find scene changes within this segment
                        cuts_in_seg = [t for t in scene_changes if seg_start < t < seg_end]
                        
                        if not cuts_in_seg:
                            split_segments.append(seg)
                        else:
                            # Split segment at each cut
                            prev_time = seg_start
                            for cut_time in sorted(cuts_in_seg):
                                if cut_time > prev_time + 0.3:  # Min 300ms segment
                                    split_segments.append({
                                        "start": prev_time,
                                        "end": cut_time,
                                        "focus": seg_focus,  # Will be re-sampled below
                                    })
                                prev_time = cut_time
                            # Add final part
                            if seg_end > prev_time + 0.3:
                                split_segments.append({
                                    "start": prev_time,
                                    "end": seg_end,
                                    "focus": seg_focus,
                                })
                    
                    logger.info(
                        "TransNetV2: Split %d segments at %d camera cuts -> %d segments",
                        len(segments), len(scene_changes), len(split_segments)
                    )
                    
                    # Re-sample focus for each new segment (to get correct position for each camera)
                    resample_cap = cv2.VideoCapture(video_path)
                    if resample_cap.isOpened():
                        resample_fps = resample_cap.get(cv2.CAP_PROP_FPS) or 25.0
                        resample_frame_count = int(resample_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                        
                        for seg in split_segments:
                            # Sample 3 frames in this segment
                            samples: list[float] = []
                            seg_dur = seg["end"] - seg["start"]
                            for offset in [0.2, 0.5, 0.8]:
                                sample_time = seg["start"] + offset * seg_dur
                                frame_idx = int(sample_time * resample_fps)
                                if frame_idx >= resample_frame_count:
                                    continue
                                resample_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                                ok, frame = resample_cap.read()
                                if not ok or frame is None:
                                    continue
                                faces = self._detect_faces(frame)
                                focus_val = _focus_for_faces(faces)
                                if focus_val is not None:
                                    samples.append(focus_val)
                            
                            if samples:
                                seg["focus"] = sum(samples) / len(samples)
                                logger.info(
                                    "Re-sampled segment [%.2f-%.2f]: %d samples -> focus=%.3f",
                                    seg["start"], seg["end"], len(samples), seg["focus"]
                                )
                        resample_cap.release()
                    
                    segments = split_segments
                
                # ============================================================
                # ADD SMOOTH TRANSITIONS (cross-fade) between segments
                # ============================================================
                TRANSITION_DURATION = 0.4  # 400ms transition between focus positions
                
                smoothed_segments: list[dict] = []
                
                for i, seg in enumerate(segments):
                    if i == 0:
                        # First segment - no transition needed
                        smoothed_segments.append(seg)
                        continue
                    
                    prev_seg = smoothed_segments[-1]
                    prev_focus = prev_seg["focus"]
                    curr_focus = seg["focus"]
                    focus_diff = abs(curr_focus - prev_focus)
                    
                    if focus_diff >= MIN_FOCUS_DIFF:
                        # Significant focus change - add transition
                        transition_start = seg["start"]
                        transition_end = min(seg["start"] + TRANSITION_DURATION, seg["end"])
                        
                        # Shorten previous segment slightly to make room for transition
                        if prev_seg["end"] > transition_start:
                            prev_seg["end"] = transition_start
                        
                        # Create transition segment (midpoint focus)
                        mid_focus = (prev_focus + curr_focus) / 2.0
                        
                        # Only add transition if there's room
                        if transition_end > transition_start + 0.1:
                            smoothed_segments.append({
                                "start": transition_start,
                                "end": transition_end,
                                "focus": mid_focus,
                                "is_transition": True,  # Mark as transition for logging
                            })
                            logger.info(
                                "Added transition [%.2f-%.2f]: focus %.3f -> %.3f (mid=%.3f)",
                                transition_start, transition_end, prev_focus, curr_focus, mid_focus
                            )
                            
                            # Adjust current segment to start after transition
                            seg = dict(seg)  # Copy to avoid modifying original
                            seg["start"] = transition_end
                    
                    # Add current segment (possibly shortened)
                    if seg["end"] > seg["start"]:
                        smoothed_segments.append(seg)
                
                segments = smoothed_segments
                logger.info("Built SPEAKER-AWARE focus timeline: %d segments (with transitions)", len(segments))
                return segments
        
        # ============================================================
        # FALLBACK: Scene-based sampling (no diarization data)
        # Uses TransNetV2 neural network for accurate scene detection
        # ============================================================
        logger.info("Using SCENE-BASED focus (no diarization data)")
        
        # Use TransNetV2 for scene detection
        scene_changes = self._detect_scene_changes(video_path, segment_start, segment_end)
        
        scene_boundaries = [0.0] + scene_changes + [duration]
        
        logger.info(
            "TransNetV2 scene-based focus: %d scenes, boundaries: %s",
            len(scene_boundaries) - 1,
            [f"{t:.2f}s" for t in scene_boundaries],
        )
        
        segments: list[dict] = []
        last_valid_focus = 0.5

        for i in range(len(scene_boundaries) - 1):
            scene_start_t = scene_boundaries[i]
            scene_end_t = scene_boundaries[i + 1]
            scene_dur = scene_end_t - scene_start_t
            
            # Sample 5 frames spread across scene
            focus_values: list[float] = []
            for sample_idx in range(5):
                offset_ratio = 0.1 + (sample_idx / 4.0) * 0.8
                sample_time = scene_start_t + offset_ratio * scene_dur
                focus_val = _sample_focus_at_time(sample_time)
                if focus_val is not None:
                    focus_values.append(focus_val)
            
            if focus_values:
                # SMART AVERAGING: Don't average to "nobody" in the center!
                min_pos = min(focus_values)
                max_pos = max(focus_values)
                spread = max_pos - min_pos
                
                if spread > 0.25:
                    # Two-speaker setup detected (positions spread > 25%)
                    # DON'T average to center where nobody is!
                    # Instead: use the MOST COMMON position (mode-like)
                    
                    # Split into left (<0.55) and right (>=0.55) clusters
                    left_cluster = [v for v in focus_values if v < 0.55]
                    right_cluster = [v for v in focus_values if v >= 0.55]
                    
                    if left_cluster and right_cluster:
                        # Both speakers present - center between their AVERAGE positions
                        left_avg = sum(left_cluster) / len(left_cluster)
                        right_avg = sum(right_cluster) / len(right_cluster)
                        # Find center that shows both (weighted by frequency)
                        left_weight = len(left_cluster)
                        right_weight = len(right_cluster)
                        scene_focus = (left_avg * left_weight + right_avg * right_weight) / (left_weight + right_weight)
                        logger.info(
                            "Scene %d: TWO-SPEAKER setup (spread=%.2f), left=%.2f (%d), right=%.2f (%d) -> focus=%.3f",
                            i, spread, left_avg, left_weight, right_avg, right_weight, scene_focus
                        )
                    elif left_cluster:
                        # Only left speaker
                        scene_focus = sum(left_cluster) / len(left_cluster)
                        logger.info("Scene %d: LEFT speaker only -> focus=%.3f", i, scene_focus)
                    else:
                        # Only right speaker
                        scene_focus = sum(right_cluster) / len(right_cluster)
                        logger.info("Scene %d: RIGHT speaker only -> focus=%.3f", i, scene_focus)
                else:
                    # Single speaker or tight cluster - simple average is OK
                    scene_focus = sum(focus_values) / len(focus_values)
                    logger.info("Scene %d: single cluster (spread=%.2f) -> focus=%.3f", i, spread, scene_focus)
                
                # CLAMP to safe bounds (0.20-0.80) to ensure crop stays within 1080x1920
                scene_focus = max(min_bound, min(max_bound, scene_focus))
                last_valid_focus = scene_focus
            else:
                scene_focus = last_valid_focus
                logger.info("Scene %d: no faces -> fallback focus=%.3f", i, scene_focus)
            
            segments.append({
                "start": scene_start_t,
                "end": scene_end_t,
                "focus": scene_focus,
            })
            
            logger.info(
                "Scene %d [%.2f-%.2f]: %d samples -> focus=%.3f",
                i, scene_start_t, scene_end_t, len(focus_values), scene_focus
            )

        capture.release()
        
        logger.info("Built scene-based focus timeline: %d segments", len(segments))
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
        scene_changes = self._detect_scene_changes(video_path, segment_start, segment_end)
        
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
