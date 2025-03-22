import os
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import torch
import librosa
from io import BytesIO
from transformers import AutoProcessor, AutoModelForCTC


class Settings(BaseSettings):
    app_name: str = "Simple Speech2Text API"
    model_name: str = "facebook/wav2vec2-base-960h"
    max_file_size: int = 1024 * 1024 * 5  # 5MB
    allowed_content_types: list = ["audio/wav", "audio/mpeg", "audio/webm", "audio/flac", "audio/x-flac"]
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=settings.log_level
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML model...")
    start = time.time()

    try:
        app.state.processor = AutoProcessor.from_pretrained(settings.model_name)
        app.state.model = AutoModelForCTC.from_pretrained(settings.model_name).to("cpu")
        load_time = time.time() - start
        logger.info(f"Model loaded in {load_time:.2f}s")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise RuntimeError("Failed to initialize model") from e

    yield


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscriptionResponse(BaseModel):
    status: str
    transcription: str
    processing_time: float


class ErrorResponse(BaseModel):
    status: str
    error_type: str
    message: str


async def validate_audio_file(file: UploadFile):
    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: {file.content_type}"
        )

    if file.size > settings.max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds limit: {settings.max_file_size} bytes"
        )


@app.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def transcribe_audio(
        file: Annotated[
            UploadFile,
            File(description="Audio file (WAV, MP3, WEBM, FLAC) up to 5MB")
        ]
):
    start_time = time.time()

    try:
        await validate_audio_file(file)

        contents = await file.read()
        audio_buffer = BytesIO(contents)
        waveform, _ = librosa.load(audio_buffer, sr=16000)

        inputs = app.state.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        with torch.no_grad():
            logits = app.state.model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = app.state.processor.batch_decode(predicted_ids)[0]

        processing_time = time.time() - start_time

        return {
            "status": "success",
            "transcription": transcription,
            "processing_time": processing_time
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error on the server during transcription"
        )


@app.get("/health", include_in_schema=False)
async def health_check():
    return {"status": "healthy"}


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Simple Speech2Text API - See /docs for documentation"}


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_type": exc.__class__.__name__,
            "message": exc.detail
        }
    )