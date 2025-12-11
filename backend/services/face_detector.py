"""Face detection helper powered by UltraFace (ONNXRuntime)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from backend import config

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = BACKEND_DIR / "assets" / "models" / "ultraface"
ULTRAFACE_MODEL_NAME = "version-RFB-320.onnx"


class FaceDetector:
    """
    Thin wrapper around the UltraFace ONNX model to estimate horizontal face focus.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ):
        self.model_path = model_path or self._resolve_default_model()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self._detector = UltraFace(
            self.model_path,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
        )

    @staticmethod
    def _resolve_default_model() -> Path:
        # Allow overriding via config but fallback to shipped asset to avoid network calls.
        candidate = getattr(config, "ULTRAFACE_MODEL_PATH", None)
        if candidate:
            candidate_path = Path(candidate)
            if candidate_path.exists():
                return candidate_path

        bundled = ASSETS_DIR / ULTRAFACE_MODEL_NAME
        if not bundled.exists():
            raise FileNotFoundError(
                f"UltraFace model not found at {bundled}. "
                "Ensure assets are included or provide ULTRAFACE_MODEL_PATH."
            )
        return bundled

    def _detect_faces(self, frame: np.ndarray) -> Sequence[dict]:
        return self._detector.detect(frame)

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
            logger.warning("UltraFace: unable to open video %s for face detection", video_path)
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
            # Drop tiny detections (<1% площади кадра), чтобы не тянуть шум
            # Keep even небольшие лица: порог 0.2% площади кадра, чтобы не отбрасывать средние планы
            faces = [
                f for f in faces
                if f.get("area", 0.0) >= 0.002 * f.get("width", 1.0) * f.get("height", 1.0)
            ]
            if not faces:
                continue

            # Debug log
            logger.info("UltraFace frame %d: found %d faces", index, len(faces))
            for i, f in enumerate(faces):
                logger.info(
                    (
                        "  Face %d: x=%.1f, y=%.1f, w=%.1f, h=%.1f, score=%.2f, "
                        "center_x=%.1f, center_y=%.1f, "
                        "center_x_input=%.1f, center_y_input=%.1f, "
                        "ratio_x=%.3f, ratio_x_in=%.3f, ratio_y=%.3f, ratio_y_in=%.3f, "
                        "width=%.1f, height=%.1f, input_w=%.1f, input_h=%.1f"
                    ),
                    i,
                    f["x"],
                    f.get("y", -1.0),
                    f["w"],
                    f.get("h", -1.0),
                    f["score"],
                    f["center_x"],
                    f.get("center_y", -1.0),
                    f.get("center_x_input", -1.0),
                    f.get("center_y_input", -1.0),
                    f["center_x"] / max(f["width"], 1.0),
                    f.get("center_x_input", 0.0) / max(f.get("input_width", 1.0), 1.0),
                    f.get("center_y", 0.0) / max(f.get("height", 1.0), 1.0),
                    f.get("center_y_input", 0.0) / max(f.get("input_height", 1.0), 1.0),
                    f["width"],
                    f["height"],
                    f.get("input_width", -1.0),
                    f.get("input_height", -1.0),
                )

            best_face = max(faces, key=lambda f: f["score"] * f["area"])
            center_ratio = best_face["center_x"] / best_face["width"]
            weight = best_face["score"] * best_face["area"]
            weighted_centers.append((center_ratio, weight))

        capture.release()

        if not weighted_centers:
            logger.info("UltraFace: no faces found in %s", video_path)
            return None

        numerator = sum(center * weight for center, weight in weighted_centers)
        denominator = sum(weight for _, weight in weighted_centers)
        if denominator <= 0:
            return None

        focus_raw = numerator / denominator
        # If all detections cluster too far to a border, treat as unreliable and fall back to center.
        if focus_raw < 0.15 or focus_raw > 0.85:
            logger.warning(
                "UltraFace: focus_raw=%.3f looks unreliable (all faces near edge), "
                "fallback to center crop 0.5",
                focus_raw,
            )
            return 0.5

        focus = float(max(0.0, min(1.0, focus_raw)))
        # Clamp extreme values to avoid hard-left/right crops when detections are uncertain
        focus = float(max(0.2, min(0.8, focus)))
        logger.info(
            "UltraFace: focus raw=%.3f clamped=%.3f samples=%d",
            focus_raw,
            focus,
            len(weighted_centers),
        )
        logger.info("UltraFace: estimated horizontal focus %.3f for %s", focus, video_path)
        return focus


class UltraFace:
    """UltraFace ONNX runtime wrapper."""

    def __init__(
        self,
        model_path: Path,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ):
        self.model_path = Path(model_path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.input_size = (320, 240)
        self.mean = np.array([127.0, 127.0, 127.0], dtype=np.float32)
        self.std = 128.0
        self.center_variance = 0.1
        self.size_variance = 0.2
        # Feature maps for input 320x240 with strides 8/16/32/64:
        # h = ceil(240/stride), w = ceil(320/stride)
        self.feature_maps = [[30, 40], [15, 20], [8, 10], [4, 5]]
        self.min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        self.shrinkage = [[8, 8], [16, 16], [32, 32], [64, 64]]
        self.priors = self._generate_priors()
        self.session = self._create_session()
        self.input_name = self.session.get_inputs()[0].name

    def _create_session(self) -> ort.InferenceSession:
        providers: list[str] = []
        try:
            available = ort.get_available_providers()
        except Exception:  # pragma: no cover - defensive
            available = ["CPUExecutionProvider"]

        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        logger.info("UltraFace providers: %s", providers)
        return ort.InferenceSession(str(self.model_path), providers=providers)

    def _generate_priors(self) -> np.ndarray:
        priors: list[list[float]] = []
        image_h, image_w = self.input_size[1], self.input_size[0]  # note: (w, h)
        for idx, feature_map in enumerate(self.feature_maps):
            scale_w = image_w / self.shrinkage[idx][1]
            scale_h = image_h / self.shrinkage[idx][0]
            for y in range(feature_map[0]):
                for x in range(feature_map[1]):
                    cx = (x + 0.5) / scale_w
                    cy = (y + 0.5) / scale_h
                    for box_size in self.min_boxes[idx]:
                        s_kx = box_size / image_w
                        s_ky = box_size / image_h
                        priors.append([cx, cy, s_kx, s_ky])
        return np.array(priors, dtype=np.float32)

    def detect(self, frame: np.ndarray) -> Sequence[dict]:
        orig_h, orig_w = frame.shape[:2]
        resized = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_LINEAR)
        image = resized.astype(np.float32)
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
        input_tensor = np.expand_dims(image, axis=0)

        scores, boxes = self.session.run(None, {self.input_name: input_tensor})
        scores = scores[0][:, 1]
        mask = scores > self.score_threshold
        if not np.any(mask):
            return []

        filtered_scores = scores[mask]
        filtered_boxes = boxes[0][mask]
        decoded_boxes_input = self._decode_boxes(filtered_boxes)
        decoded_boxes = decoded_boxes_input.copy()

        # IMPORTANT: _decode_boxes returns coordinates relative to input_size (320x240).
        # We MUST scale them to the original image size. Do it explicitly per column.
        scale_w = float(orig_w) / float(self.input_size[0])
        scale_h = float(orig_h) / float(self.input_size[1])
        decoded_boxes[:, 0] = decoded_boxes[:, 0] * scale_w  # xmin
        decoded_boxes[:, 2] = decoded_boxes[:, 2] * scale_w  # xmax
        decoded_boxes[:, 1] = decoded_boxes[:, 1] * scale_h  # ymin
        decoded_boxes[:, 3] = decoded_boxes[:, 3] * scale_h  # ymax
        if decoded_boxes.size > 0:
            logger.info(
                "UltraFace scale_w=%.2f scale_h=%.2f sample_box=[%.1f %.1f %.1f %.1f] orig_w=%d orig_h=%d",
                scale_w,
                scale_h,
                decoded_boxes[0, 0],
                decoded_boxes[0, 1],
                decoded_boxes[0, 2],
                decoded_boxes[0, 3],
                orig_w,
                orig_h,
            )

        decoded_boxes = self._clip_boxes(decoded_boxes, orig_w, orig_h)

        keep = self._nms(decoded_boxes, filtered_scores)
        detections = []
        for idx in keep:
            xmin, ymin, xmax, ymax = decoded_boxes[idx]
            score = float(filtered_scores[idx])
            w = xmax - xmin
            h = ymax - ymin
            xmin_in, ymin_in, xmax_in, ymax_in = decoded_boxes_input[idx]
            center_x_in = float((xmin_in + xmax_in) / 2.0)
            center_y_in = float((ymin_in + ymax_in) / 2.0)
            detections.append(
                {
                    "x": float(xmin),
                    "y": float(ymin),
                    "w": float(w),
                    "h": float(h),
                    "score": score,
                    "area": float(w * h),
                    "center_x": float(xmin + w / 2.0),
                    "center_y": float(ymin + h / 2.0),
                    "width": float(orig_w),
                    "height": float(orig_h),
                    "center_x_input": center_x_in,
                    "center_y_input": center_y_in,
                    "input_width": float(self.input_size[0]),
                    "input_height": float(self.input_size[1]),
                }
            )
        return detections

    def _decode_boxes(self, raw_boxes: np.ndarray) -> np.ndarray:
        priors = self.priors[: raw_boxes.shape[0]]
        boxes = np.concatenate(
            [
                priors[:, :2] + raw_boxes[:, :2] * self.center_variance * priors[:, 2:],
                priors[:, 2:] * np.exp(raw_boxes[:, 2:] * self.size_variance),
            ],
            axis=1,
        )
        boxes[:, :2] -= boxes[:, 2:] / 2  # cx, cy -> xmin, ymin
        boxes[:, 2:] += boxes[:, :2]      # width/height -> xmax, ymax
        boxes[:, [0, 2]] *= self.input_size[0]
        boxes[:, [1, 3]] *= self.input_size[1]
        return boxes

    def _clip_boxes(self, boxes: np.ndarray, width: int, height: int) -> np.ndarray:
        boxes[:, 0] = np.clip(boxes[:, 0], 0, width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, height)
        return boxes

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> Sequence[int]:
        if boxes.size == 0:
            return []
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            iou = self._iou(boxes[i], boxes[order[1:]])
            inds = np.where(iou <= self.nms_threshold)[0]
            order = order[inds + 1]
        return keep

    @staticmethod
    def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if boxes.size == 0:
            return np.array([])
        x_min = np.maximum(box[0], boxes[:, 0])
        y_min = np.maximum(box[1], boxes[:, 1])
        x_max = np.minimum(box[2], boxes[:, 2])
        y_max = np.minimum(box[3], boxes[:, 3])
        inter = np.clip(x_max - x_min, a_min=0, a_max=None) * np.clip(
            y_max - y_min, a_min=0, a_max=None
        )
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - inter + 1e-5
        return inter / union

