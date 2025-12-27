"""
TransNetV2 Shot Boundary Detection

Official PyTorch implementation from: https://github.com/soCzech/TransNetV2
Paper: TransNet V2: An effective deep network architecture for fast shot transition detection
https://arxiv.org/abs/2008.04838

Weights from: https://huggingface.co/MiaoshouAI/transnetv2-pytorch-weights
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Hugging Face weights URL (pre-converted PyTorch weights)
WEIGHTS_URL = "https://huggingface.co/MiaoshouAI/transnetv2-pytorch-weights/resolve/main/transnetv2-pytorch-weights.pth"
WEIGHTS_FILENAME = "transnetv2-pytorch-weights.pth"


# ==============================================================================
# Official TransNetV2 PyTorch Implementation
# Source: https://github.com/soCzech/TransNetV2/blob/master/inference-pytorch/transnetv2_pytorch.py
# ==============================================================================

class TransNetV2(nn.Module):
    def __init__(self, F=16, L=3, S=2, D=1024,
                 use_many_hot_targets=True,
                 use_frame_similarity=True,
                 use_color_histograms=True,
                 use_mean_pooling=False,
                 dropout_rate=0.5,
                 use_convex_comb_reg=False,
                 use_resnet_features=False,
                 use_resnet_like_top=False,
                 frame_similarity_on_last_layer=False):
        super(TransNetV2, self).__init__()

        if use_resnet_features or use_resnet_like_top or use_convex_comb_reg or frame_similarity_on_last_layer:
            raise NotImplementedError("Some options not implemented in Pytorch version of Transnet!")

        self.SDDCNN = nn.ModuleList(
            [StackedDDCNNV2(in_filters=3, n_blocks=S, filters=F, stochastic_depth_drop_prob=0.)] +
            [StackedDDCNNV2(in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2 ** i)
             for i in range(1, L)]
        )

        self.frame_sim_layer = FrameSimilarity(
            sum([(F * 2 ** i) * 4 for i in range(L)]),
            lookup_window=101,
            output_dim=128,
            similarity_dim=128,
            use_bias=True
        ) if use_frame_similarity else None

        self.color_hist_layer = ColorHistograms(
            lookup_window=101,
            output_dim=128
        ) if use_color_histograms else None

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6  # 3x6 for spatial dimensions
        if use_frame_similarity:
            output_dim += 128
        if use_color_histograms:
            output_dim += 128

        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1) if use_many_hot_targets else None

        self.use_mean_pooling = use_mean_pooling
        self.eval()

    def forward(self, inputs):
        assert isinstance(inputs, torch.Tensor) and list(inputs.shape[2:]) == [27, 48, 3] and inputs.dtype == torch.uint8, \
            "incorrect input type and/or shape"

        # uint8 of shape [B, T, H, W, 3] to float of shape [B, 3, T, H, W]
        x = inputs.permute([0, 4, 1, 2, 3]).float()
        x = x.div_(255.)

        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)

        if self.use_mean_pooling:
            x = torch.mean(x, dim=[3, 4])
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = x.reshape(x.shape[0], x.shape[1], -1)

        if self.frame_sim_layer is not None:
            x = torch.cat([self.frame_sim_layer(block_features), x], 2)

        if self.color_hist_layer is not None:
            x = torch.cat([self.color_hist_layer(inputs), x], 2)

        x = self.fc1(x)
        x = F.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        one_hot = self.cls_layer1(x)

        if self.cls_layer2 is not None:
            return one_hot, {"many_hot": self.cls_layer2(x)}

        return one_hot


class StackedDDCNNV2(nn.Module):
    def __init__(self, in_filters, n_blocks, filters,
                 shortcut=True,
                 use_octave_conv=False,
                 pool_type="avg",
                 stochastic_depth_drop_prob=0.0):
        super(StackedDDCNNV2, self).__init__()

        if use_octave_conv:
            raise NotImplementedError("Octave convolution not implemented in Pytorch version of Transnet!")

        assert pool_type == "max" or pool_type == "avg"

        self.shortcut = shortcut
        self.DDCNN = nn.ModuleList([
            DilatedDCNNV2(in_filters if i == 1 else filters * 4, filters,
                          octave_conv=use_octave_conv,
                          activation=F.relu if i != n_blocks else None)
            for i in range(1, n_blocks + 1)
        ])

        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) if pool_type == "max" else nn.AvgPool3d(kernel_size=(1, 2, 2))
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    def forward(self, inputs):
        x = inputs
        shortcut = None

        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                shortcut = x

        x = F.relu(x)

        if self.shortcut is not None:
            if self.stochastic_depth_drop_prob != 0.:
                if self.training:
                    if random.random() < self.stochastic_depth_drop_prob:
                        x = shortcut
                    else:
                        x = x + shortcut
                else:
                    x = (1 - self.stochastic_depth_drop_prob) * x + shortcut
            else:
                x += shortcut

        x = self.pool(x)
        return x


class DilatedDCNNV2(nn.Module):
    def __init__(self, in_filters, filters, batch_norm=True, activation=None, octave_conv=False):
        super(DilatedDCNNV2, self).__init__()

        if octave_conv:
            raise NotImplementedError("Octave convolution not implemented in Pytorch version of Transnet!")

        assert not (octave_conv and batch_norm)

        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1, use_bias=not batch_norm)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2, use_bias=not batch_norm)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4, use_bias=not batch_norm)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8, use_bias=not batch_norm)

        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None
        self.activation = activation

    def forward(self, inputs):
        conv1 = self.Conv3D_1(inputs)
        conv2 = self.Conv3D_2(inputs)
        conv3 = self.Conv3D_4(inputs)
        conv4 = self.Conv3D_8(inputs)

        x = torch.cat([conv1, conv2, conv3, conv4], dim=1)

        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class Conv3DConfigurable(nn.Module):
    def __init__(self, in_filters, filters, dilation_rate, separable=True, octave=False, use_bias=True,
                 kernel_initializer=None):
        super(Conv3DConfigurable, self).__init__()

        if octave:
            raise NotImplementedError("Octave convolution not implemented in Pytorch version of Transnet!")
        if kernel_initializer is not None:
            raise NotImplementedError("Kernel initializers are not implemented in Pytorch version of Transnet!")

        assert not (separable and octave)

        if separable:
            # (2+1)D convolution https://arxiv.org/pdf/1711.11248.pdf
            conv1 = nn.Conv3d(in_filters, 2 * filters, kernel_size=(1, 3, 3),
                              dilation=(1, 1, 1), padding=(0, 1, 1), bias=False)
            conv2 = nn.Conv3d(2 * filters, filters, kernel_size=(3, 1, 1),
                              dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 0, 0), bias=use_bias)
            self.layers = nn.ModuleList([conv1, conv2])
        else:
            conv = nn.Conv3d(in_filters, filters, kernel_size=3,
                             dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 1, 1), bias=use_bias)
            self.layers = nn.ModuleList([conv])

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class FrameSimilarity(nn.Module):
    def __init__(self, in_filters, similarity_dim=128, lookup_window=101, output_dim=128,
                 stop_gradient=False, use_bias=False):
        super(FrameSimilarity, self).__init__()

        if stop_gradient:
            raise NotImplementedError("Stop gradient not implemented in Pytorch version of Transnet!")

        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)

        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    def forward(self, inputs):
        x = torch.cat([torch.mean(x, dim=[3, 4]) for x in inputs], dim=1)
        x = torch.transpose(x, 1, 2)

        x = self.projection(x)
        x = F.normalize(x, p=2, dim=2)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]

        similarities_padded = F.pad(similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2])

        batch_indices = torch.arange(0, batch_size, device=x.device).view([batch_size, 1, 1]).repeat(
            [1, time_window, self.lookup_window])
        time_indices = torch.arange(0, time_window, device=x.device).view([1, time_window, 1]).repeat(
            [batch_size, 1, self.lookup_window])
        lookup_indices = torch.arange(0, self.lookup_window, device=x.device).view([1, 1, self.lookup_window]).repeat(
            [batch_size, time_window, 1]) + time_indices

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]
        return F.relu(self.fc(similarities))


class ColorHistograms(nn.Module):
    def __init__(self, lookup_window=101, output_dim=None):
        super(ColorHistograms, self).__init__()

        self.fc = nn.Linear(lookup_window, output_dim) if output_dim is not None else None
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    @staticmethod
    def compute_color_histograms(frames):
        frames = frames.int()

        def get_bin(frames):
            # returns 0 .. 511
            R, G, B = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]
            R, G, B = R >> 5, G >> 5, B >> 5
            return (R << 6) + (G << 3) + B

        batch_size, time_window, height, width, no_channels = frames.shape
        assert no_channels == 3

        frames_flatten = frames.view(batch_size * time_window, height * width, 3)
        binned_values = get_bin(frames_flatten)

        frame_bin_prefix = (torch.arange(0, batch_size * time_window, device=frames.device) << 9).view(-1, 1)
        binned_values = (binned_values + frame_bin_prefix).view(-1)

        histograms = torch.zeros(batch_size * time_window * 512, dtype=torch.int32, device=frames.device)
        histograms.scatter_add_(0, binned_values,
                                torch.ones(len(binned_values), dtype=torch.int32, device=frames.device))

        histograms = histograms.view(batch_size, time_window, 512).float()
        histograms_normalized = F.normalize(histograms, p=2, dim=2)

        return histograms_normalized

    def forward(self, inputs):
        x = self.compute_color_histograms(inputs)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]

        similarities_padded = F.pad(similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2])

        batch_indices = torch.arange(0, batch_size, device=x.device).view([batch_size, 1, 1]).repeat(
            [1, time_window, self.lookup_window])
        time_indices = torch.arange(0, time_window, device=x.device).view([1, time_window, 1]).repeat(
            [batch_size, 1, self.lookup_window])
        lookup_indices = torch.arange(0, self.lookup_window, device=x.device).view([1, 1, self.lookup_window]).repeat(
            [batch_size, time_window, 1]) + time_indices

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]

        if self.fc is not None:
            return F.relu(self.fc(similarities))
        return similarities


# ==============================================================================
# High-level Detector Interface
# ==============================================================================

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
        threshold: float = 0.3,  # Lower to catch more scene changes
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
        self._weights_path = Path(__file__).parent / WEIGHTS_FILENAME
        
    def _ensure_model(self):
        """Load model weights if not already loaded."""
        if self._model is not None:
            return
            
        logger.info("Loading TransNetV2 model...")
        self._model = TransNetV2()
        
        # Try to load weights
        if self._weights_path.exists():
            try:
                state_dict = torch.load(self._weights_path, map_location=self.device, weights_only=True)
                self._model.load_state_dict(state_dict)
                logger.info("TransNetV2 weights loaded from %s", self._weights_path)
            except Exception as e:
                logger.warning("Failed to load TransNetV2 weights: %s", e)
                self._download_weights()
        else:
            logger.warning(
                "TransNetV2 weights not found at %s. Downloading...",
                self._weights_path
            )
            self._download_weights()
        
        self._model.to(self.device)
        self._model.eval()
    
    def _download_weights(self):
        """Download model weights from Hugging Face."""
        try:
            import urllib.request
            logger.info("Downloading TransNetV2 weights from %s", WEIGHTS_URL)
            urllib.request.urlretrieve(WEIGHTS_URL, self._weights_path)
            
            state_dict = torch.load(self._weights_path, map_location=self.device, weights_only=True)
            self._model.load_state_dict(state_dict)
            logger.info("TransNetV2 weights downloaded and loaded successfully (%.1f MB)",
                       self._weights_path.stat().st_size / 1024 / 1024)
        except Exception as e:
            logger.error("Failed to download TransNetV2 weights: %s", e)
            raise RuntimeError(f"Could not download TransNetV2 weights: {e}")
    
    def _preprocess_frames(self, frames: list[np.ndarray]) -> torch.Tensor:
        """
        Preprocess frames for TransNetV2.
        
        Args:
            frames: List of BGR frames (H, W, 3)
        
        Returns:
            (1, N, 27, 48, 3) uint8 tensor of RGB frames
        """
        processed = []
        for frame in frames:
            # Resize to 48x27 (W, H)
            resized = cv2.resize(frame, (48, 27), interpolation=cv2.INTER_AREA)
            # BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            processed.append(rgb)
        
        # Stack to numpy array
        frames_arr = np.stack(processed, axis=0).astype(np.uint8)
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
        
        # Process in batches of 100 frames (model expects batches of ~100)
        batch_size = 100
        all_predictions = []
        
        for start_idx in range(0, len(frames), batch_size - 10):  # Overlap of 10 frames
            end_idx = min(start_idx + batch_size, len(frames))
            batch_frames = frames[start_idx:end_idx]
            
            if len(batch_frames) < 10:
                break
            
            # Pad to at least 100 frames if needed (model expects 100)
            while len(batch_frames) < 100:
                batch_frames.append(batch_frames[-1])  # Repeat last frame
            
            # Preprocess
            input_tensor = self._preprocess_frames(batch_frames)
            input_tensor = input_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                single_pred, _ = self._model(input_tensor)
                single_pred = torch.sigmoid(single_pred)
            
            predictions = single_pred.squeeze(0).cpu().numpy()
            
            # Handle overlap - only take predictions for actual frames
            actual_count = min(end_idx - start_idx, len(predictions))
            if start_idx == 0:
                all_predictions.extend(predictions[:actual_count].tolist())
            else:
                # Skip overlapping 10 frames
                skip = 10
                if actual_count > skip:
                    all_predictions.extend(predictions[skip:actual_count].tolist())
        
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
