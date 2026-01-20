import io
import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from PIL import Image
from transformers import AutoModel, AutoProcessor
from contextlib import asynccontextmanager


MODEL_ID = os.getenv("MODEL_ID", "google/siglip2-so400m-patch14-384")
DEVICE = os.getenv("DEVICE", "cpu")

model = None
processor = None


class TextIn(BaseModel):
    text: str = Field(..., min_length=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()
    model.to(DEVICE)
    print(model.device)
    yield
    model = None
    processor = None

app = FastAPI(title="SigLIP2 Inference", lifespan=lifespan)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}



def _embed_text(text: str) -> list[float]:
    if model is None or processor is None:
        raise RuntimeError("model not loaded")

    inputs = processor(text=[text], return_tensors="pt", padding="max_length")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = F.normalize(features, p=2, dim=-1)

    embedding = features[0].float().cpu().numpy()
    return embedding.tolist()


def _embed_image(images: list[Image.Image]) -> list[list[float]]:
    if model is None or processor is None:
        raise RuntimeError("model not loaded")

    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = F.normalize(features, p=2, dim=-1)

    embeddings = features.float().cpu().numpy().tolist()
    return embeddings


@app.post("/embed/text")
async def embed_text(payload: TextIn) -> Dict[str, object]:
    embedding = _embed_text(payload.text)
    return {"embedding": embedding, "dim": len(embedding)}


@app.post("/embed/image")
async def embed_image(files: list[UploadFile] = File(...)) -> Dict[str, object]:
    if not files:
        raise HTTPException(status_code=400, detail="no files provided")

    images = []
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"only image uploads are supported: {file.filename}",
            )

        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"empty file: {file.filename}")

        try:
            image = Image.open(io.BytesIO(data))
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"invalid image data: {file.filename}",
            )

        images.append(image.convert("RGB"))

    embeddings = _embed_image(images)
    dim = len(embeddings[0]) if embeddings else 0
    return {"embeddings": embeddings, "dim": dim}
