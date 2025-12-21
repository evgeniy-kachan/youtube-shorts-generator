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
        model_name: str = "antelopev2",  # default to SCRFD (better on profiles)
        det_thresh: float = 0.20,
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
            # Larger det_size + lower threshold to catch harder poses
            self._detector.prepare(ctx_id=self.ctx_id, det_thresh=self.det_thresh, det_size=(1280, 1280))
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
            # - Multiple faces, span < 0.40: center between extremes (show both)
            # - Multiple faces, span >= 0.40: track positions for later primary speaker detection
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
            return None

        # Check if we have two speakers on different positions
        # Use multi-face data (wide span frames) to identify left/right speaker positions
        two_speaker_mode = False
        left_speaker_pos: float | None = None
        right_speaker_pos: float | None = None
        
        if multi_face_data:
            # We have frames showing both speakers far apart
            # Average their positions across all multi-face frames
            left_positions = [left for left, right in multi_face_data]
            right_positions = [right for left, right in multi_face_data]
            left_speaker_pos = sum(left_positions) / len(left_positions)
            right_speaker_pos = sum(right_positions) / len(right_positions)
            two_speaker_mode = True
            
            logger.info(
                "Detected two-speaker setup from %d multi-face frames: left=%.3f, right=%.3f",
                len(multi_face_data),
                left_speaker_pos,
                right_speaker_pos,
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
        if two_speaker_mode and left_speaker_pos is not None and right_speaker_pos is not None:
            primary_speaker = self._get_primary_speaker(dialogue, segment_start, segment_end)
            
            if primary_speaker and dialogue:
                # Calculate which position (left or right) corresponds to primary speaker
                # Heuristic: match single-face frames to primary speaker's speech intervals
                primary_interval_frames = self._get_speaker_frame_indices(
                    dialogue, primary_speaker, segment_start, frame_count, fps
                )
                
                # Count which cluster (left/right) has more overlap with primary speaker frames
                midpoint = (left_speaker_pos + right_speaker_pos) / 2.0
                left_cluster_indices = [
                    idx for idx in sample_indices 
                    if idx in primary_interval_frames
                ]
                
                # Determine primary speaker position by comparing to left/right
                # Use single-face positions that occurred during primary speaker's speech
                primary_single_faces = []
                for frame_idx, pos in single_face_data:
                    if frame_idx in primary_interval_frames:
                        primary_single_faces.append(pos)
                
                if primary_single_faces:
                    avg_primary_pos = sum(primary_single_faces) / len(primary_single_faces)
                    
                    # Is primary speaker closer to left or right position?
                    if abs(avg_primary_pos - left_speaker_pos) < abs(avg_primary_pos - right_speaker_pos):
                        focus = left_speaker_pos
                        logger.info(
                            "Primary speaker '%s' (%.0f%% speech) identified on LEFT (%.3f), focusing there",
                            primary_speaker,
                            self._get_speaker_percentage(dialogue, primary_speaker, segment_start, segment_end),
                            focus,
                        )
                    else:
                        focus = right_speaker_pos
                        logger.info(
                            "Primary speaker '%s' (%.0f%% speech) identified on RIGHT (%.3f), focusing there",
                            primary_speaker,
                            self._get_speaker_percentage(dialogue, primary_speaker, segment_start, segment_end),
                            focus,
                        )
                    
                    focus = float(max(0.2, min(0.8, focus)))
                    return focus
            
            # Fallback: use center between both speakers
            focus = (left_speaker_pos + right_speaker_pos) / 2.0
            logger.info("Unable to identify primary speaker, using center=%.3f", focus)
            focus = float(max(0.2, min(0.8, focus)))
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
        Detect abrupt scene changes (camera cuts) using PySceneDetect.
        
        Args:
            video_path: Path to video file
            segment_start: Start time in seconds
            segment_end: End time in seconds (None = full video)
        
        Returns:
            List of timestamps (in seconds) where scene changes occur
        """
        try:
            from scenedetect import detect, ContentDetector
            from scenedetect.frame_timecode import FrameTimecode
        except ImportError:
            logger.warning("PySceneDetect not installed, scene detection disabled")
            return []
        
        try:
            # Use OpenCV to get fps/duration (PySceneDetect high-level API no longer exposes VideoStream)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise OSError("unable to open video for scene detection")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            duration = frame_count / fps
            cap.release()

            if segment_end is None:
                segment_end = duration
            
            # Convert time boundaries to frame numbers
            start_frame = int(segment_start * fps)
            end_frame = int(segment_end * fps)
            
            # Create start/end timecodes
            start_tc = FrameTimecode(start_frame, fps)
            end_tc = FrameTimecode(end_frame, fps)
            
            # Detect scenes using ContentDetector (adaptive threshold)
            # threshold: lower = more sensitive (default 27.0, мы используем 24.0 для интервью)
            scene_list = detect(
                video_path,
                ContentDetector(threshold=24.0, min_scene_len=18),  # чуть менее чувствительно, минимум ~0.7s при 25fps
                start_time=start_tc,
                end_time=end_tc,
            )
            
            # Extract timestamps of scene changes (start of each new scene)
            scene_changes = []
            for i, (start_time, end_time) in enumerate(scene_list):
                if i > 0:  # Skip first scene start (it's segment_start)
                    timestamp = start_time.get_seconds()
                    # Adjust to segment-relative time
                    relative_ts = timestamp - segment_start
                    if 0 < relative_ts < (segment_end - segment_start):
                        scene_changes.append(relative_ts)
                        logger.info(
                            "Scene change detected at %.2fs (absolute: %.2fs)",
                            relative_ts, timestamp
                        )
            
            logger.info(
                "Detected %d scene changes in [%.1f, %.1f]",
                len(scene_changes), segment_start, segment_end or 0
            )
            return scene_changes
            
        except Exception as exc:
            logger.warning("Scene detection failed for %s: %s", video_path, exc)
            return []

    # ------------------------------------------------------------------ #
    # Dynamic focus timeline (per-frame sampling, no smoothing)
    # ------------------------------------------------------------------ #
    def build_focus_timeline(
        self,
        video_path: str,
        dialogue: list[dict] | None = None,
        segment_start: float = 0.0,
        segment_end: float | None = None,
        sample_period: float = 0.33,
    ) -> list[dict]:
        """
        Build a coarse per-time focus timeline for dynamic cropping.

        - Samples frames every `sample_period` seconds (no smoothing).
        - For each sample, picks focus using the same rules:
          * 1 face  -> center on that face
          * 2 faces, span < 0.35 -> center between faces (show both)
          * 2+ faces, span >= 0.35 -> focus on primary speaker if known,
            otherwise closest to center
          * 0 faces -> fallback to last focus (or center 0.5)
        - Returns a list of segments {start, end, focus} where focus stays
          approximately stable (merged if delta < 0.05).
        """
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            logger.warning("InsightFace: unable to open video %s for timeline", video_path)
            return []

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        duration = frame_count / fps

        # Primary speaker for disambiguation
        primary_speaker = self._get_primary_speaker(dialogue, segment_start, segment_end)

        # Precompute speaker time windows in frames (for mapping primary to position)
        primary_frames: set[int] = set()
        if primary_speaker:
            primary_frames = self._get_speaker_frame_indices(dialogue, primary_speaker, segment_start, frame_count, fps)

        # Helper to choose focus per detected faces in one frame
        min_bound = 0.15
        max_bound = 0.85
        # Store last known positions to keep focus near a speaker when faces vanish
        priors = {"primary": None, "pair": None}

        def _focus_for_faces(faces, frame_index: int) -> Optional[float]:
            if not faces:
                return None
            # Drop tiny detections
            faces_f = [
                f for f in faces
                if f.get("area", 0.0) >= 0.0005 * f.get("width", 1.0) * f.get("height", 1.0)
            ]
            if not faces_f:
                return None

            if len(faces_f) == 1:
                f = faces_f[0]
                center_clamped = float(max(min_bound, min(max_bound, f["center_x"] / f["width"])))
                priors["primary"] = center_clamped
                return center_clamped

            frame_width = faces_f[0].get("width", 1.0)
            centers = [f["center_x"] / frame_width for f in faces_f]
            span = max(centers) - min(centers)

            if span < 0.45:
                center_pair = float(max(min_bound, min(max_bound, (min(centers) + max(centers)) / 2.0)))
                priors["pair"] = center_pair
                return center_pair

            # span wide: pick primary speaker position if we can infer left/right
            left_pos = min(centers)
            right_pos = max(centers)

            if primary_speaker:
                # Heuristic: if current frame is inside primary speaker speech, decide closest cluster to center
                if frame_index in primary_frames:
                    # Assume primary speaker is nearer to center (0.5) or pick closer cluster
                    if abs(0.5 - left_pos) < abs(0.5 - right_pos):
                        focus_primary = float(max(min_bound, min(max_bound, left_pos)))
                    else:
                        focus_primary = float(max(min_bound, min(max_bound, right_pos)))
                    priors["primary"] = focus_primary
                    return focus_primary

            # Fallback: closest to center
            best = min(centers, key=lambda c: abs(c - 0.5))
            return float(max(min_bound, min(max_bound, best)))

        sample_step = max(1, int(sample_period * fps))
        timeline_raw: list[tuple[float, Optional[float]]] = []

        last_detect_ts = 0.0
        for frame_idx in range(0, frame_count, sample_step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok or frame is None:
                timeline_raw.append((frame_idx / fps, None))
                continue
            faces = self._detect_faces(frame)
            focus_val = _focus_for_faces(faces, frame_idx)
            timeline_raw.append((frame_idx / fps, focus_val))
            if focus_val is not None:
                last_detect_ts = frame_idx / fps

        capture.release()

        # Fill gaps: carry last known focus; если лиц долго нет — опираемся на priors (primary/pair), и только затем центр
        filled: list[tuple[float, float]] = []
        last_focus = 0.5
        last_seen_ts = 0.0
        no_face_reset_sec = 1.5  # быстрее сброс при пропаже лиц (было 3.0)
        for ts, focus in timeline_raw:
            if focus is None:
                if (ts - last_seen_ts) > no_face_reset_sec:
                    fallback = priors.get("primary") or priors.get("pair") or last_focus or 0.5
                    last_focus = fallback
                focus = last_focus
            else:
                last_focus = focus
                last_seen_ts = ts
            filled.append((ts, focus))

        if not filled:
            return []

        # Лёгкое сглаживание (окно 3) внутри стабильного плана, БЕЗ гистерезиса.
        # Instant switch при резких изменениях (scene changes), но усредняем мелкие колебания.
        smoothed: list[tuple[float, float]] = []
        window: list[float] = []
        jump_threshold = 0.10  # что считаем "резким скачком" (новый план)

        for ts, fval in filled:
            # Если резкий скачок → сбрасываем окно и переключаемся МГНОВЕННО
            if window and abs(fval - window[-1]) > jump_threshold:
                window = [fval]
                smoothed.append((ts, fval))
                continue
            # Иначе добавляем в окно и усредняем (убирает шум детектора)
                window.append(fval)
                if len(window) > 5:
                    window.pop(0)
            smoothed.append((ts, sum(window) / len(window)))

        # Merge into segments if focus change is small (< 0.10)
        merged: list[dict] = []
        seg_start = smoothed[0][0]
        seg_focus = smoothed[0][1]
        threshold = 0.10

        for i in range(1, len(smoothed)):
            ts, fval = smoothed[i]
            if abs(fval - seg_focus) <= threshold:
                continue
            # close current segment at this timestamp
            merged.append({"start": seg_start, "end": ts, "focus": seg_focus})
            seg_start = ts
            seg_focus = fval

        # tail
        merged.append({"start": seg_start, "end": duration, "focus": seg_focus})

        # Detect scene changes (camera cuts) and split segments at those boundaries
        scene_changes = self._detect_scene_changes(video_path, segment_start, segment_end)
        
        # Split merged segments at scene change boundaries
        if scene_changes:
            split_segments: list[dict] = []
            for seg in merged:
                seg_start_time = seg["start"]
                seg_end_time = seg["end"]
                seg_focus_val = seg["focus"]
                
                # Find scene changes within this segment
                cuts_in_seg = [sc for sc in scene_changes if seg_start_time < sc < seg_end_time]
                
                if not cuts_in_seg:
                    split_segments.append(seg)
                else:
                    # Split segment at each scene change
                    cuts_sorted = sorted(cuts_in_seg)
                    prev_cut = seg_start_time
                    for cut_ts in cuts_sorted:
                        split_segments.append({"start": prev_cut, "end": cut_ts, "focus": seg_focus_val})
                        prev_cut = cut_ts
                    # Add final sub-segment
                    split_segments.append({"start": prev_cut, "end": seg_end_time, "focus": seg_focus_val})
            
            merged = split_segments

        # Filter extremely short segments (<1.2s) by merging with previous
        cleaned: list[dict] = []
        for seg in merged:
            if cleaned and (seg["end"] - seg["start"]) < 1.2:
                # merge into previous
                prev = cleaned[-1]
                new_end = seg["end"]
                # keep focus of previous; extend duration
                prev["end"] = new_end
            else:
                cleaned.append(seg)

        logger.info("Built focus timeline: %s", cleaned)
        return cleaned

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
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
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
            faces = [
                f for f in faces
                if f.get("area", 0.0) >= 0.0005 * f.get("width", 1.0) * f.get("height", 1.0)
            ]
            if not faces:
                timeline_raw.append((frame_idx / fps, None))
                continue
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
