#!/usr/bin/env python3
"""Test NeMo diarization directly (without RQ worker)"""
import torch
torch.cuda.init()
print('CUDA initialized')

from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf
import os

os.makedirs('/tmp/nemo_out', exist_ok=True)

config = OmegaConf.create({
    'device': 'cuda',
    'verbose': True,
    'num_workers': 0,
    'sample_rate': 16000,
    'batch_size': 64,
    'diarizer': {
        'manifest_filepath': '/tmp/test_manifest.json',
        'out_dir': '/tmp/nemo_out',
        'oracle_vad': False,
        'collar': 0.25,
        'ignore_overlap': True,
        'speaker_embeddings': {
            'model_path': 'titanet_large',
            'parameters': {
                'window_length_in_sec': [1.5, 1.25, 1.0, 0.75, 0.5],
                'shift_length_in_sec': [0.75, 0.625, 0.5, 0.375, 0.25],
                'multiscale_weights': [1, 1, 1, 1, 1],
                'save_embeddings': False,
            }
        },
        'clustering': {
            'parameters': {
                'oracle_num_speakers': False,
                'max_num_speakers': 8,
                'enhanced_count_thres': 80,
                'max_rp_threshold': 0.25,
                'sparse_search_volume': 30,
            }
        },
        'vad': {
            'model_path': 'vad_multilingual_marblenet',
            'parameters': {
                'window_length_in_sec': 0.15,
                'shift_length_in_sec': 0.01,
                'smoothing': 'median',
                'overlap': 0.5,
                'onset': 0.4,
                'offset': 0.3,
                'pad_onset': 0.05,
                'pad_offset': -0.1,
                'min_duration_on': 0.2,
                'min_duration_off': 0.2,
                'filter_speech_first': True,
            }
        },
    }
})

print('Config created')
diarizer = ClusteringDiarizer(cfg=config)
print('Diarizer initialized successfully!')
