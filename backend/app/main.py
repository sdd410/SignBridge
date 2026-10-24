"""FastAPI Application Entrypoint for SignBridge."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.app.config.settings import settings
from backend.app.inference.model_loader import ModelLoader
from backend.app.api.routes import router as api_router
from backend.app.api.websocket import ws_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown procedures."""
    logger.info("Initializing SignBridge Inference Server...")
    # Preload model at server launch
    loader = ModelLoader.get_instance()
    if loader.is_ready():
        logger.info(f"Model ready on {loader.device} with {loader.num_classes} classes")
    else:
        logger.warning(f"Model not ready: {loader.error_message}")
    yield
    logger.info("Shutting down SignBridge Server...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Real-Time Sign Language Gesture Recognition & Translation API",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(api_router, prefix="/api", tags=["REST API"])
app.include_router(ws_router, tags=["WebSocket"])

# Mount static files for generated plots (curves, confusion matrix)
weights_dir = settings.weights_dir
if weights_dir.exists():
    app.mount("/static/plots", StaticFiles(directory=str(weights_dir)), name="plots")


@app.get("/")
def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "online",
        "docs_url": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.host, port=settings.port, reload=True)
