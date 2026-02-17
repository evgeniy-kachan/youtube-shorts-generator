#!/usr/bin/env python3
"""
NeMo Worker for Selectel server.
Connects to Redis on main server and processes NeMo diarization tasks.
"""
import os
import sys
import time
import logging
import multiprocessing as mp
import signal
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nemo_worker")

# IMPORTANT: Set spawn method for CUDA compatibility
mp.set_start_method("spawn", force=True)

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "178.72.132.235")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "NeMo2026SecurePass!")
STATUS_KEY = "nemo_server:status"
HEARTBEAT_KEY = "nemo_server:heartbeat"
STATUS_TTL = 120  # seconds


def set_status(redis_conn, status: str):
    """Set NeMo server status in Redis with TTL."""
    redis_conn.setex(STATUS_KEY, STATUS_TTL, status)
    redis_conn.setex(HEARTBEAT_KEY, STATUS_TTL, str(time.time()))
    logger.info(f"Status set to: {status}")


def heartbeat_loop(redis_conn, stop_event):
    """Background thread to update status and heartbeat TTL."""
    while not stop_event.is_set():
        try:
            current = redis_conn.get(STATUS_KEY)
            if current and current.decode() != "offline":
                redis_conn.expire(STATUS_KEY, STATUS_TTL)
                redis_conn.setex(HEARTBEAT_KEY, STATUS_TTL, str(time.time()))
        except Exception as e:
            logger.warning(f"Heartbeat error: {e}")
        time.sleep(30)  # Update every 30 seconds


def main():
    # Lazy imports after set_start_method
    from redis import Redis
    from rq import SimpleWorker, Queue
    import threading
    
    logger.info("=" * 60)
    logger.info("NeMo Worker (Selectel) starting")
    logger.info(f"Connecting to Redis: {REDIS_HOST}:{REDIS_PORT}")
    logger.info("=" * 60)
    
    try:
        redis_conn = Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            socket_connect_timeout=10
        )
        redis_conn.ping()
        logger.info("Redis connection successful!")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        sys.exit(1)
    
    # Set initial status
    set_status(redis_conn, "ready")
    
    # Start heartbeat thread
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=heartbeat_loop, 
        args=(redis_conn, stop_event),
        daemon=True
    )
    heartbeat_thread.start()
    
    # Cleanup on exit
    def cleanup():
        logger.info("Shutting down, setting status to offline...")
        stop_event.set()
        set_status(redis_conn, "offline")
    
    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *args: sys.exit(0))
    
    # Create queue and worker
    queue = Queue("nemo_tasks", connection=redis_conn)
    
    worker = SimpleWorker(
        queues=[queue],
        connection=redis_conn,
        name=f"nemo_selectel_{os.getpid()}"
    )
    
    logger.info("Listening on queue: nemo_tasks")
    logger.info("Worker ready, waiting for tasks...")
    
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
