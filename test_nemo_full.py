#!/usr/bin/env python3
"""Test NeMo diarization directly (without RQ worker) with real audio"""
import torch
torch.cuda.init()
print('CUDA init OK')

# cuBLAS test
a = torch.randn(64, 64, device='cuda')
b = torch.randn(64, 64, device='cuda')
c = torch.mm(a, b)
del a, b, c
torch.cuda.synchronize()
print('cuBLAS OK')

# Load NeMo
from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf
import json
import tempfile
import os

# Create manifest
with tempfile.TemporaryDirectory() as tmpdir:
    # Extract audio first
    print('Extracting audio...')
    os.system('ffmpeg -y -i /opt/youtube-shorts-generator/temp/test3.MOV -ac 1 -ar 16000 /tmp/test_audio.wav 2>/dev/null')
    
    manifest_path = os.path.join(tmpdir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump({'audio_filepath': '/tmp/test_audio.wav', 'offset': 0, 'duration': 60, 'label': 'infer', 'text': '-'}, f)
        f.write('\n')
    
    config = OmegaConf.create({
        'device': 'cuda',
        'verbose': True,
        'num_workers': 0,
        'sample_rate': 16000,
        'batch_size': 64,
        'diarizer': {
            'manifest_filepath': manifest_path,
            'out_dir': tmpdir,
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
    
    print('Creating diarizer...')
    diarizer = ClusteringDiarizer(cfg=config)
    print('Running diarization...')
    diarizer.diarize()
    print('DONE!')
