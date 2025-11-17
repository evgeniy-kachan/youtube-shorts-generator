"""Main FastAPI application."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.routers import video
from backend.config import OUTPUT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="YouTube Shorts Generator API",
    description="AI-powered service to extract and process viral moments from YouTube videos",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(video.router)

# Serve output files as static files
output_dir = Path(OUTPUT_DIR)
if output_dir.exists():
    app.mount("/output", StaticFiles(directory=str(output_dir)), name="output")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "YouTube Shorts Generator API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    from backend.config import HOST, PORT
    
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(
        "backend.main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )

