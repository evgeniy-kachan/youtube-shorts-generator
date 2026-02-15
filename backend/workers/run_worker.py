#!/usr/bin/env python3
"""
GPU Worker for processing transcription and diarization tasks.

This worker runs in a separate process and has exclusive access to the GPU.
It processes tasks from the Redis queue sequentially.

Usage:
    # From project root:
    python -m backend.workers.run_worker
    
    # Or directly:
    /opt/youtube-shorts-generator/venv-asr/bin/python backend/workers/run_worker.py

Environment variables:
    REDIS_HOST: Redis server host (default: localhost)
    REDIS_PORT: Redis server port (default: 6379)
    HUGGINGFACE_TOKEN: Required for Pyannote diarization
"""
from __future__ import annotations

import logging
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger("gpu_worker")


def main():
    """Start the GPU worker."""
    import torch
    from redis import Redis
    from rq import Worker, Queue
    
    # Log GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("=" * 60)
        logger.info("GPU Worker starting")
        logger.info("GPU: %s (%.1f GB)", gpu_name, gpu_memory)
        logger.info("CUDA version: %s", torch.version.cuda)
        logger.info("=" * 60)
    else:
        logger.warning("=" * 60)
        logger.warning("GPU Worker starting WITHOUT GPU!")
        logger.warning("CUDA is not available, tasks will run on CPU")
        logger.warning("=" * 60)
    
    # Redis connection
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_DB", "0"))
    
    logger.info("Connecting to Redis: %s:%d", redis_host, redis_port)
    
    try:
        redis_conn = Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
        )
        redis_conn.ping()
        logger.info("Redis connection successful")
    except Exception as e:
        logger.error("Failed to connect to Redis: %s", e)
        sys.exit(1)
    
    # Create queue
    queue = Queue("gpu_tasks", connection=redis_conn)
    logger.info("Listening on queue: %s", queue.name)
    
    # Start worker
    worker = Worker(
        queues=[queue],
        connection=redis_conn,
        name=f"gpu_worker_{os.getpid()}",
    )
    
    logger.info("Worker ready, waiting for tasks...")
    logger.info("Press Ctrl+C to stop")
    
    try:
        worker.work(with_scheduler=False)
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error("Worker error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
