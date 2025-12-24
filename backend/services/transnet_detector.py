"""
TransNetV2 Shot Boundary Detection

Based on: https://github.com/soCzech/TransNetV2
Paper: TransNet V2: An effective deep network architecture for fast shot transition detection
https://arxiv.org/abs/2008.04838
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Model weights URL
WEIGHTS_URL = "https://github.com/soCzech/TransNetV2/raw/master/inference-pytorch/transnetv2-pytorch-weights.pth"


class TransNetV2(nn.Module):
    """
    TransNetV2 PyTorch implementation for shot boundary detection.
    
    This is a re-implementation of the official TensorFlow model in PyTorch.
    The model processes video frames in batches and predicts shot boundaries.
    """
    
    def __init__(
        self,
        F: int = 16,
        L: int = 3,
        S: int = 2,
        D: int = 1024,
    ):
        super().__init__()
        
        self.SDDCNN = nn.ModuleList([
            StackedDDCNNV2(in_filters=3, n_blocks=S, filters=F, stochastic_depth_drop_prob=0.) 
            for _ in range(L)
        ])
        self.frame_sim_layer = FrameSimilarity(
            sum([F * 2 ** (i + 1) for i in range(L)]),
            lookup_window=101,
            output_dim=128,
        )
        self.color_hist_layer = ColorHistograms(
            lookup_window=101,
            output_dim=128,
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(
            sum([F * 2 ** (i + 1) for i in range(L)]) + 128 + 128, D
        )
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1)
        
        # Parameters for inference
        self._input_size = (27, 48)  # H, W for resizing frames
        
    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            frames: (B, T, H, W, C) tensor of video frames, normalized to [0, 1]
        
        Returns:
            Tuple of (single_frame_pred, all_frames_pred) tensors
        """
        # Transpose to (B, T, C, H, W) for conv layers
        x = frames.permute(0, 1, 4, 2, 3)
        
        # Extract features from each SDDCNN block
        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)
        
        # Concatenate all block features
        x = torch.cat(block_features, dim=2)
        
        # Flatten spatial dimensions
        x = x.flatten(start_dim=2)
        
        # Get frame similarity and color histogram features
        x_fs = self.frame_sim_layer(x)
        x_ch = self.color_hist_layer(frames)
        
        # Concatenate all features
        x = torch.cat([x, x_fs, x_ch], dim=2)
        
        # Classification
        x = self.dropout(F.relu(self.fc1(x)))
        single = torch.sigmoid(self.cls_layer1(x))
        all_frames = torch.sigmoid(self.cls_layer2(x))
        
        return single.squeeze(-1), all_frames.squeeze(-1)


class StackedDDCNNV2(nn.Module):
    """Stacked Dilated Dense CNN V2 block."""
    
    def __init__(
        self,
        in_filters: int,
        n_blocks: int,
        filters: int,
        stochastic_depth_drop_prob: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            DilatedDCNNV2(
                in_filters if i == 0 else filters * 4, 
                filters,
                activation=nn.ReLU() if i < n_blocks - 1 else None,
            )
            for i in range(n_blocks)
        ])
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.drop_prob = stochastic_depth_drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.pool(x)


class DilatedDCNNV2(nn.Module):
    """Dilated Dense CNN V2 block."""
    
    def __init__(
        self,
        in_filters: int,
        filters: int,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_filters, filters, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(in_filters + filters, filters, kernel_size=(1, 3, 3), 
                               padding=(0, 2, 2), dilation=(1, 2, 2))
        self.conv3 = nn.Conv3d(in_filters + 2 * filters, filters, kernel_size=(1, 3, 3),
                               padding=(0, 4, 4), dilation=(1, 4, 4))
        self.conv4 = nn.Conv3d(in_filters + 3 * filters, filters, kernel_size=(1, 3, 3),
                               padding=(0, 8, 8), dilation=(1, 8, 8))
        self.bn = nn.BatchNorm3d(filters * 4)
        self.activation = activation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = F.relu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = F.relu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class FrameSimilarity(nn.Module):
    """Frame similarity computation layer."""
    
    def __init__(
        self,
        in_filters: int,
        lookup_window: int = 101,
        output_dim: int = 128,
    ):
        super().__init__()
        self.projection = nn.Linear(in_filters, output_dim)
        self.lookup_window = lookup_window
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=-1)
        
        # Compute similarity with neighboring frames
        B, T, D = x.shape
        similarities = []
        
        half_window = self.lookup_window // 2
        for t in range(T):
            start = max(0, t - half_window)
            end = min(T, t + half_window + 1)
            neighbors = x[:, start:end, :]  # (B, W, D)
            center = x[:, t:t+1, :]  # (B, 1, D)
            sim = (center * neighbors).sum(dim=-1)  # (B, W)
            # Pad to fixed size
            pad_left = t - start
            pad_right = half_window - (end - t - 1)
            sim = F.pad(sim, (half_window - pad_left, half_window - pad_right + 1 - sim.shape[1] + pad_left), value=0)
            similarities.append(sim[:, :self.lookup_window])
        
        return torch.stack(similarities, dim=1)  # (B, T, lookup_window)


class ColorHistograms(nn.Module):
    """Color histogram comparison layer."""
    
    def __init__(
        self,
        lookup_window: int = 101,
        output_dim: int = 128,
    ):
        super().__init__()
        self.lookup_window = lookup_window
        self.output_dim = output_dim
        
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, H, W, C)
        B, T, H, W, C = frames.shape
        
        # Compute simple color histograms (mean color per frame)
        mean_colors = frames.mean(dim=(2, 3))  # (B, T, C)
        
        # Compute similarity with neighboring frames
        similarities = []
        half_window = self.lookup_window // 2
        
        for t in range(T):
            start = max(0, t - half_window)
            end = min(T, t + half_window + 1)
            neighbors = mean_colors[:, start:end, :]  # (B, W, C)
            center = mean_colors[:, t:t+1, :]  # (B, 1, C)
            # Color difference
            diff = ((center - neighbors) ** 2).sum(dim=-1)  # (B, W)
            sim = 1.0 / (1.0 + diff)  # Convert to similarity
            # Pad to fixed size
            pad_size = self.lookup_window - sim.shape[1]
            sim = F.pad(sim, (0, pad_size), value=0)
            similarities.append(sim[:, :self.lookup_window])
        
        return torch.stack(similarities, dim=1)  # (B, T, lookup_window)


class TransNetV2Detector:
    """
    High-level interface for TransNetV2 shot boundary detection.
    
    Usage:
        detector = TransNetV2Detector()
        scene_changes = detector.detect_scenes("video.mp4")
    """
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        threshold: float = 0.5,
    ):
        """
        Initialize TransNetV2 detector.
        
        Args:
            device: Device to run inference on ("cuda" or "cpu")
            threshold: Detection threshold (0.0-1.0)
        """
        self.device = device
        self.threshold = threshold
        self._model: Optional[TransNetV2] = None
        self._weights_path = Path(__file__).parent / "transnetv2_weights.pth"
        
    def _ensure_model(self):
        """Load model weights if not already loaded."""
        if self._model is not None:
            return
            
        logger.info("Loading TransNetV2 model...")
        self._model = TransNetV2()
        
        # Try to load weights
        if self._weights_path.exists():
            try:
                state_dict = torch.load(self._weights_path, map_location=self.device)
                self._model.load_state_dict(state_dict)
                logger.info("TransNetV2 weights loaded from %s", self._weights_path)
            except Exception as e:
                logger.warning("Failed to load TransNetV2 weights: %s, using random init", e)
        else:
            logger.warning(
                "TransNetV2 weights not found at %s. "
                "Download from: %s",
                self._weights_path, WEIGHTS_URL
            )
            # Try to download
            self._download_weights()
        
        self._model.to(self.device)
        self._model.eval()
    
    def _download_weights(self):
        """Download model weights from GitHub."""
        try:
            import urllib.request
            logger.info("Downloading TransNetV2 weights from %s", WEIGHTS_URL)
            urllib.request.urlretrieve(WEIGHTS_URL, self._weights_path)
            
            state_dict = torch.load(self._weights_path, map_location=self.device)
            self._model.load_state_dict(state_dict)
            logger.info("TransNetV2 weights downloaded and loaded successfully")
        except Exception as e:
            logger.warning("Failed to download TransNetV2 weights: %s", e)
    
    def _preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        Preprocess frames for TransNetV2.
        
        Args:
            frames: (N, H, W, C) array of BGR frames
        
        Returns:
            (1, N, 27, 48, 3) tensor of RGB frames normalized to [0, 1]
        """
        # Resize to 48x27 (W, H)
        processed = []
        for frame in frames:
            resized = cv2.resize(frame, (48, 27), interpolation=cv2.INTER_AREA)
            # BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            processed.append(rgb)
        
        # Stack and normalize
        frames_arr = np.stack(processed, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(frames_arr).unsqueeze(0)  # (1, N, H, W, C)
    
    def detect_scenes(
        self,
        video_path: str,
        segment_start: float = 0.0,
        segment_end: Optional[float] = None,
    ) -> list[float]:
        """
        Detect scene changes in video.
        
        Args:
            video_path: Path to video file
            segment_start: Start time in seconds
            segment_end: End time in seconds (None = full video)
        
        Returns:
            List of timestamps (relative to segment_start) where scene changes occur
        """
        self._ensure_model()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("TransNetV2: unable to open video %s", video_path)
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        duration = frame_count / fps if fps > 0 else 0
        
        if segment_end is None:
            segment_end = duration
        
        # Calculate frame range
        start_frame = int(segment_start * fps)
        end_frame = int(segment_end * fps)
        
        # Read frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if len(frames) < 2:
            logger.warning("TransNetV2: not enough frames for detection")
            return []
        
        logger.info("TransNetV2: processing %d frames from [%.1f, %.1f]s", 
                   len(frames), segment_start, segment_end)
        
        # Process in batches of 100 frames with overlap
        batch_size = 100
        all_predictions = []
        
        for start_idx in range(0, len(frames), batch_size - 10):  # Overlap of 10 frames
            end_idx = min(start_idx + batch_size, len(frames))
            batch_frames = frames[start_idx:end_idx]
            
            if len(batch_frames) < 10:
                break
            
            # Preprocess
            input_tensor = self._preprocess_frames(np.array(batch_frames))
            input_tensor = input_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                single_pred, _ = self._model(input_tensor)
            
            predictions = single_pred.squeeze(0).cpu().numpy()
            
            # Handle overlap
            if start_idx == 0:
                all_predictions.extend(predictions.tolist())
            else:
                # Skip overlapping frames
                all_predictions.extend(predictions[10:].tolist())
        
        # Find scene changes (peaks above threshold)
        scene_changes = []
        predictions_arr = np.array(all_predictions)
        
        # Find local maxima above threshold
        for i in range(1, len(predictions_arr) - 1):
            if (predictions_arr[i] > self.threshold and 
                predictions_arr[i] > predictions_arr[i-1] and 
                predictions_arr[i] > predictions_arr[i+1]):
                # Convert frame index to relative timestamp
                timestamp = i / fps
                scene_changes.append(timestamp)
                logger.info(
                    "TransNetV2: scene change at %.2fs (frame %d, confidence %.3f)",
                    timestamp, i, predictions_arr[i]
                )
        
        logger.info(
            "TransNetV2: detected %d scene changes in [%.1f, %.1f]s",
            len(scene_changes), segment_start, segment_end
        )
        
        return scene_changes


# Singleton instance for efficiency
_detector: Optional[TransNetV2Detector] = None


def get_transnet_detector() -> TransNetV2Detector:
    """Get or create TransNetV2 detector singleton."""
    global _detector
    if _detector is None:
        _detector = TransNetV2Detector()
    return _detector

