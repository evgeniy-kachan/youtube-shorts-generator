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
        'speaker_embeddings': {
            'model_path': 'titanet_large',
        },
        'clustering': {
            'parameters': {
                'oracle_num_speakers': False,
                'max_num_speakers': 8,
            }
        },
        'vad': {
            'model_path': 'vad_multilingual_marblenet',
        },
    }
})

print('Config created')
diarizer = ClusteringDiarizer(cfg=config)
print('Diarizer initialized successfully!')
