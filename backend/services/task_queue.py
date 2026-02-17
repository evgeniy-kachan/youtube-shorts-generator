"""
Task Queue Service - wrapper over Redis Queue (RQ).

Provides simple interface for enqueueing GPU tasks and waiting for results.
All GPU-intensive tasks (WhisperX, Pyannote, NeMo) run in a separate worker process.

Usage:
    from backend.services.task_queue import get_task_queue
    
    queue = get_task_queue()
    
    # Enqueue a task
    job = queue.enqueue_transcription(audio_path, diarizer="pyannote")
    
    # Wait for result (blocking)
    result = job.wait_result(timeout=600)
    
    # Or check status
    if job.is_finished:
        result = job.result
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

from redis import Redis
from rq import Queue
from rq.job import Job

logger = logging.getLogger(__name__)

# Redis connection settings
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)  # For remote NeMo worker

# Queue settings
GPU_QUEUE_NAME = "gpu_tasks"
NEMO_QUEUE_NAME = "nemo_tasks"  # Separate queue for NeMo - runs in isolated worker
DEFAULT_TIMEOUT = 1800  # 30 minutes max per task


class TaskQueue:
    """
    Wrapper over RQ for GPU task management.
    """
    
    def __init__(
        self,
        redis_host: str = REDIS_HOST,
        redis_port: int = REDIS_PORT,
        redis_db: int = REDIS_DB,
        redis_password: str = REDIS_PASSWORD,
    ):
        self.redis_conn = Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
        )
        self.queue = Queue(
            name=GPU_QUEUE_NAME,
            connection=self.redis_conn,
            default_timeout=DEFAULT_TIMEOUT,
        )
        # Separate queue for NeMo - processed by isolated nemo-worker
        # This prevents CUDA context conflicts between Pyannote and NeMo
        self.nemo_queue = Queue(
            name=NEMO_QUEUE_NAME,
            connection=self.redis_conn,
            default_timeout=3600,  # 1 hour for NeMo
        )
        logger.info(
            "TaskQueue initialized: redis=%s:%d, queues=[%s, %s]",
            redis_host, redis_port, GPU_QUEUE_NAME, NEMO_QUEUE_NAME
        )
    
    def is_redis_available(self) -> bool:
        """Check if Redis is available."""
        try:
            self.redis_conn.ping()
            return True
        except Exception as e:
            logger.error("Redis not available: %s", e)
            return False
    
    def enqueue_transcription(
        self,
        audio_path: str,
        diarizer: str = "pyannote",
        model: str = "large-v3",
        language: str = "en",
        num_speakers: int = 0,
        device: str = "cuda",
        hf_token: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> JobWrapper:
        """
        Enqueue transcription + diarization task.
        
        Args:
            audio_path: Path to audio/video file
            diarizer: "pyannote" or "nemo"
            model: WhisperX model name
            language: Expected language
            num_speakers: Number of speakers (0 = auto-detect)
            device: "cuda" or "cpu"
            hf_token: HuggingFace token for Pyannote
            timeout: Max execution time in seconds
            
        Returns:
            JobWrapper with methods to check status and get result
        """
        hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        
        job = self.queue.enqueue(
            "backend.workers.gpu_tasks.transcribe_and_diarize",
            audio_path=audio_path,
            diarizer=diarizer,
            model=model,
            language=language,
            num_speakers=num_speakers,
            device=device,
            hf_token=hf_token,
            job_timeout=timeout,
        )
        
        logger.info(
            "Enqueued transcription task: job_id=%s, diarizer=%s, file=%s",
            job.id, diarizer, audio_path
        )
        
        return JobWrapper(job)
    
    def enqueue_diarization_only(
        self,
        audio_path: str,
        diarizer: str = "pyannote",
        num_speakers: int = 0,
        device: str = "cuda",
        hf_token: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> JobWrapper:
        """
        Enqueue diarization-only task (no transcription).
        
        Useful for re-diarizing with different settings.
        """
        hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        
        if diarizer == "nemo":
            func = "backend.workers.gpu_tasks.diarize_nemo"
            kwargs = {
                "audio_path": audio_path,
                "num_speakers": num_speakers,
                "device": device,
            }
        else:
            func = "backend.workers.gpu_tasks.diarize_pyannote"
            kwargs = {
                "audio_path": audio_path,
                "num_speakers": num_speakers,
                "device": device,
                "hf_token": hf_token,
            }
        
        job = self.queue.enqueue(func, **kwargs, job_timeout=timeout)
        
        logger.info(
            "Enqueued diarization task: job_id=%s, diarizer=%s",
            job.id, diarizer
        )
        
        return JobWrapper(job)
    
    def enqueue_nemo_diarization(
        self,
        audio_path: str,
        num_speakers: int = 0,
        max_speakers: int = 8,
        timeout: int = 3600,  # 1 hour for long videos
    ) -> JobWrapper:
        """
        Enqueue NeMo MSDD diarization task.
        
        Runs in SEPARATE nemo_tasks queue, processed by isolated nemo-worker.
        This prevents CUDA context conflicts - nemo-worker has its own CUDA context.
        
        Returns full diarization result with speaker stats.
        """
        # Use separate nemo_tasks queue - processed by nemo-worker
        job = self.nemo_queue.enqueue(
            "backend.workers.nemo_tasks.nemo_diarize_task",
            audio_path=audio_path,
            num_speakers=num_speakers,
            max_speakers=max_speakers,
            job_timeout=timeout,
        )
        
        logger.info(
            "Enqueued NeMo diarization task to nemo_tasks: job_id=%s, file=%s",
            job.id, audio_path
        )
        
        return JobWrapper(job)
    
    def get_job(self, job_id: str) -> Optional[JobWrapper]:
        """Get job by ID."""
        try:
            job = Job.fetch(job_id, connection=self.redis_conn)
            return JobWrapper(job)
        except Exception:
            return None
    
    def get_queue_length(self) -> int:
        """Get number of pending jobs in queue."""
        return len(self.queue)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "queue_name": self.queue.name,
            "pending": len(self.queue),
            "failed": len(self.queue.failed_job_registry),
            "finished": len(self.queue.finished_job_registry),
        }
    
    def get_nemo_server_status(self) -> Dict[str, Any]:
        """
        Check if remote NeMo server (Selectel) is online.
        
        The NeMo worker sets 'nemo_server:status' key in Redis when it starts.
        """
        try:
            status = self.redis_conn.get("nemo_server:status")
            last_heartbeat = self.redis_conn.get("nemo_server:heartbeat")
            
            if status:
                status = status.decode("utf-8")
                heartbeat_ts = float(last_heartbeat.decode("utf-8")) if last_heartbeat else 0
                
                # Check if heartbeat is recent (within 60 seconds)
                import time
                is_alive = (time.time() - heartbeat_ts) < 60 if heartbeat_ts else False
                
                return {
                    "available": status == "ready" and is_alive,
                    "status": status,
                    "last_heartbeat": heartbeat_ts,
                    "message": "NeMo server (Selectel) is online" if is_alive else "NeMo server heartbeat stale"
                }
            else:
                return {
                    "available": False,
                    "status": "offline",
                    "message": "NeMo server not connected"
                }
        except Exception as e:
            logger.error("Error checking NeMo server status: %s", e)
            return {
                "available": False,
                "status": "error",
                "message": f"Error: {str(e)}"
            }


class JobWrapper:
    """
    Wrapper over RQ Job with convenient methods.
    """
    
    def __init__(self, job: Job):
        self._job = job
    
    @property
    def id(self) -> str:
        return self._job.id
    
    @property
    def status(self) -> str:
        return self._job.get_status()
    
    @property
    def is_queued(self) -> bool:
        return self._job.get_status() == "queued"
    
    @property
    def is_started(self) -> bool:
        return self._job.get_status() == "started"
    
    @property
    def is_finished(self) -> bool:
        return self._job.get_status() == "finished"
    
    @property
    def is_failed(self) -> bool:
        return self._job.get_status() == "failed"
    
    @property
    def result(self) -> Any:
        """Get job result (None if not finished)."""
        return self._job.result
    
    @property
    def error(self) -> Optional[str]:
        """Get error message if job failed."""
        if self.is_failed:
            return str(self._job.exc_info)
        return None
    
    @property
    def progress(self) -> float:
        """Get job progress (0.0 to 1.0)."""
        status = self.status
        if status == "queued":
            return 0.0
        elif status == "started":
            return 0.5  # We don't have granular progress
        elif status == "finished":
            return 1.0
        elif status == "failed":
            return 1.0
        return 0.0
    
    def wait_result(self, timeout: int = 600, poll_interval: float = 1.0) -> Any:
        """
        Wait for job to complete and return result.
        
        Args:
            timeout: Max seconds to wait
            poll_interval: Seconds between status checks
            
        Returns:
            Job result
            
        Raises:
            TimeoutError: If job doesn't complete in time
            RuntimeError: If job fails
        """
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > timeout:
                raise TimeoutError(f"Job {self.id} timed out after {timeout}s")
            
            status = self.status
            
            if status == "finished":
                return self.result
            
            if status == "failed":
                raise RuntimeError(f"Job {self.id} failed: {self.error}")
            
            time.sleep(poll_interval)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "job_id": self.id,
            "status": self.status,
            "progress": self.progress,
            "is_finished": self.is_finished,
            "is_failed": self.is_failed,
            "error": self.error,
        }


# Singleton instance
_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """Get or create TaskQueue singleton."""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
    return _task_queue


def is_queue_available() -> bool:
    """Check if task queue (Redis) is available."""
    try:
        queue = get_task_queue()
        return queue.is_redis_available()
    except Exception:
        return False
