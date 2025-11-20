"""Utility helpers for working with temp/output directories."""
from pathlib import Path
import shutil

from backend import config


def get_temp_dir(create: bool = True) -> Path:
    """
    Return path to the temporary working directory.
    Creates the directory if it does not exist (by default).
    """
    temp_dir = Path(config.TEMP_DIR)
    if create:
        temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def get_output_dir(video_id: str | None = None, create: bool = True) -> Path:
    """
    Return path to the output directory (optionally namespaced by video_id).
    """
    output_dir = Path(config.OUTPUT_DIR)
    if video_id:
        output_dir = output_dir / video_id
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def clear_temp_dir():
    """Remove all contents of the temporary directory."""
    temp_dir = Path(config.TEMP_DIR)
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

