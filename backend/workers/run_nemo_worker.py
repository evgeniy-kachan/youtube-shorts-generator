#!/usr/bin/env python3
"""
NeMo Worker for speaker diarization tasks.

This worker runs in a separate process with its own CUDA context.
It uses venv-nemo environment and processes NeMo diarization tasks.

Usage:
    # From project root (using venv-nemo):
    /opt/youtube-shorts-generator/venv-nemo/bin/python -m backend.workers.run_nemo_worker
    
Environment variables:
    REDIS_HOST: Redis server host (default: localhost)
    REDIS_PORT: Redis server port (default: 6379)
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys

# CRITICAL: Set multiprocessing start method to 'spawn' BEFORE importing torch
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

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

logger = logging.getLogger("nemo_worker")


def main():
    """Start the NeMo worker."""
    # NOTE: Do NOT import torch here!
    # NeMo tasks use lazy imports - torch/NeMo are imported INSIDE task functions.
    # If we import torch here, CUDA context would be initialized too early.
    from redis import Redis
    from rq import Queue, SimpleWorker
    
    # Log startup info (without touching CUDA)
    logger.info("=" * 60)
    logger.info("NeMo Worker starting")
    logger.info("NOTE: CUDA will be initialized INSIDE task (lazy import)")
    logger.info("NOTE: torch/NeMo NOT imported at worker startup")
    logger.info("Multiprocessing start method: %s", mp.get_start_method())
    logger.info("=" * 60)
    
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
    
    # Create queue - separate queue for NeMo tasks
    queue = Queue("nemo_tasks", connection=redis_conn)
    logger.info("Listening on queue: %s", queue.name)
    
    # Use SimpleWorker - executes jobs in same process (no fork)
    worker = SimpleWorker(
        queues=[queue],
        connection=redis_conn,
        name=f"nemo_worker_{os.getpid()}",
    )
    
    logger.info("Worker ready, waiting for NeMo tasks...")
    logger.info("Press Ctrl+C to stop")
    
    try:
        worker.work(with_scheduler=False, burst=False)
    except KeyboardInterrupt:
        logger.info("NeMo worker stopped by user")
    except Exception as e:
        logger.error("NeMo worker error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
