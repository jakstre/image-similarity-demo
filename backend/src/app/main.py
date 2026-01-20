import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .db import Base, Image


INFERENCE_URL = os.getenv("INFERENCE_URL")
DATABASE_URL = os.getenv(
    "DATABASE_URL"
)

class TextItemIn(BaseModel):
    text: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None


class SearchTextIn(BaseModel):
    text: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=50)


client: Optional[httpx.AsyncClient] = None
Session: Optional[async_sessionmaker[AsyncSession]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, Session
    client = httpx.AsyncClient(timeout=30)
    engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    yield 
    if client is not None:
        await client.aclose()
        client = None
    if engine is not None:
        await engine.dispose()
    Session = None

app = FastAPI(title="Similarity Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


async def embed_text(text: str) -> np.ndarray:
    if client is None:
        raise HTTPException(status_code=500, detail="inference client not ready")

    response = await client.post(f"{INFERENCE_URL}/embed/text", json={"text": text})
    response.raise_for_status()
    payload = response.json()
    return np.array(payload["embedding"], dtype="float32").reshape(1, -1)


async def embed_image_bytes(files: list[tuple[str, Optional[str], bytes]]) -> np.ndarray:
    if client is None:
        raise HTTPException(status_code=500, detail="inference client not ready")

    payload = [
        ("files", (filename or "upload", data, content_type))
        for filename, content_type, data in files
    ]
    response = await client.post(f"{INFERENCE_URL}/embed/image", files=payload)
    response.raise_for_status()
    response_payload = response.json()
    return np.array(response_payload["embeddings"], dtype="float32")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _image_to_item(image: Image, score: Optional[float] = None) -> Dict[str, Any]:
    item = {
        "metadata": dict(image.metadata_ or {}),
        "id": image.id,
        "image_url": image.image_url,
        "url": image.url,
        "featured": image.featured,
        "blur_hash": image.blur_hash,
        "description": image.description,
        "width": image.width,
        "height": image.height,
        "aspect_ratio": float(image.aspect_ratio),
        "views": image.views,
        "downloads": image.downloads,
        "submitted_at": image.submitted_at.isoformat() if image.submitted_at else None,
        "created_at": image.created_at.isoformat(),
    }
    if score is not None:
        item["score"] = float(score)
    return item


@app.post("/search/text")
async def search_text(query: SearchTextIn) -> Dict[str, Any]:
    embedding = await embed_text(query.text)
    if Session is None:
        raise HTTPException(status_code=500, detail="database not ready")
    distance = Image.embedding.cosine_distance(embedding[0].tolist())
    stmt = (
        select(Image, distance.label("score"))
        .order_by(distance)
        .limit(query.k)
    )
    async with Session() as session:
        rows = (await session.execute(stmt)).all()
    results = [_image_to_item(image, score) for image, score in rows]
    return {"query": query.text, "results": results}


@app.post("/search/image")
async def search_image(file: UploadFile = File(...), k: int = Query(10, ge=1, le=50)) -> Dict[str, Any]:

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="only image uploads are supported")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    if Session is None:
        raise HTTPException(status_code=500, detail="database not ready")
    embedding = await embed_image_bytes([(file.filename or "upload", file.content_type, data)])
    distance = Image.embedding.cosine_distance(embedding[0].tolist())
    stmt = select(Image, distance.label("score")).order_by(distance).limit(k)
    async with Session() as session:
        rows = (await session.execute(stmt)).all()
    results = [_image_to_item(image, score) for image, score in rows]
    return {"results": results}


@app.get("/images")
async def list_images(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
) -> Dict[str, Any]:
    if Session is None:
        raise HTTPException(status_code=500, detail="database not ready")
    offset = (page - 1) * per_page
    async with Session() as session:
        base_stmt = select(Image)
        total = (
            await session.execute(
                select(func.count()).select_from(Image)
            )
        ).scalar_one()
        rows = (
            await session.execute(
                base_stmt.order_by(Image.created_at.desc())
                .offset(offset)
                .limit(per_page)
            )
        ).scalars().all()
    return {
        "items": [_image_to_item(image) for image in rows],
        "page": page,
        "per_page": per_page,
        "total": total,
        "has_more": offset + per_page < total,
    }


@app.get("/items/{item_id}")
async def get_item(item_id: str) -> Dict[str, Any]:
    if Session is None:
        raise HTTPException(status_code=500, detail="database not ready")
    async with Session() as session:
        image = (
            await session.execute(select(Image).where(Image.id == item_id))
        ).scalar_one_or_none()
    if image is None:
        raise HTTPException(status_code=404, detail="item not found")
    return _image_to_item(image)
