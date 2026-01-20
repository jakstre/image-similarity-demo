import os
import asyncio
import math
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from datasets import load_dataset
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from tqdm import tqdm

from app.db import Base, Image
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference:8001")
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@postgres:5432/unsplash"
)
DATASET_NAME = os.getenv("DATASET_NAME", "1aurent/unsplash-lite")
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "train")
MAX_ITEMS = int(os.getenv("MAX_ITEMS", "1000"))
BULK_SIZE = int(os.getenv("BULK_SIZE", "30"))
INFERENCE_BATCH_SIZE = int(os.getenv("INFERENCE_BATCH_SIZE", "30"))

async def wait_for_service(
    client: httpx.AsyncClient,
    base_url: str,
    retries: int = 10,
    delay_s: float = 1.0,
) -> None:
    for attempt in range(retries):
        try:
            response = await client.get(f"{base_url}/health")
            response.raise_for_status()
            return
        except httpx.HTTPError:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(delay_s)


async def wait_for_db(engine, retries: int = 10, delay_s: float = 1.0) -> None:
    for attempt in range(retries):
        try:
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            return
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(delay_s)


async def embed_text(client: httpx.AsyncClient, text: str) -> list[float]:
    response = await client.post(f"{INFERENCE_URL}/embed/text", json={"text": text})
    response.raise_for_status()
    return response.json()["embedding"]


async def embed_image_batch(
    client: httpx.AsyncClient, files: list[tuple[str, str, bytes]]
) -> list[list[float]]:
    payload_files = [
        ("files", (filename, data, content_type))
        for filename, content_type, data in files
    ]
    response = await client.post(f"{INFERENCE_URL}/embed/image", files=payload_files)
    response.raise_for_status()
    payload = response.json()
    return payload["embeddings"]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.lower() == "nan":
        return True
    return False


def _require_int(value: Any, field_name: str) -> int:
    if _is_missing(value):
        raise ValueError(f"missing required field: {field_name}")
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid int for {field_name}: {value!r}") from exc


def _require_float(value: Any, field_name: str) -> float:
    if _is_missing(value):
        raise ValueError(f"missing required field: {field_name}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid float for {field_name}: {value!r}") from exc


def _clean_metadata(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, dict):
        return {key: _clean_metadata(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean_metadata(item) for item in value]
    return value


async def fetch_image_bytes(client: httpx.AsyncClient, url: str) -> bytes:
    response = await client.get(url)
    response.raise_for_status()
    return response.content


async def fetch_image_batch(
    client: httpx.AsyncClient, batch: list[tuple[int, dict]]
) -> list[tuple[int, dict, bytes]]:
    tasks = [fetch_image_bytes(client, row["photo"]["image_url"]) for _, row in batch]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    fetched = []
    for (idx, row), result in zip(batch, results):
        if isinstance(result, Exception):
            continue
        fetched.append((idx, row, result))
    return fetched


def _prepare_row(row: dict) -> dict:
    keywords = row.get("keywords") or []
    cleaned = []
    for keyword in keywords:
        if not isinstance(keyword, dict):
            continue
        if not keyword.get("suggested_by_user"):
            continue
        value = keyword.get("keyword")
        if _is_missing(value):
            continue
        cleaned.append(str(value))
    row["keywords"] = cleaned
    return row


def prepare_entry(idx: int, row: dict, embedding: list[float]) -> dict:
    photo = row.get("photo") or {}
    image_url = photo.get("image_url")
    item_id = photo.get("id") or f"unsplash-{idx + 1}"
    stats = row.get("stats") or {}
    metadata = _clean_metadata(row)
    metadata["source"] = "unsplash-lite"
    metadata["image_url"] = image_url
    created_at = _extract_created_at(metadata)
    metadata.pop("photo", None)
    metadata.pop("stats", None)
    return {
        "id": item_id,
        "embedding": embedding,
        "image_url": image_url,
        "url": _optional_str(photo.get("url")),
        "featured": _optional_bool(photo.get("featured")),
        "blur_hash": _optional_str(photo.get("blur_hash")),
        "description": _optional_str(photo.get("description")),
        "width": _require_int(photo.get("width"), "photo.width"),
        "height": _require_int(photo.get("height"), "photo.height"),
        "aspect_ratio": _require_float(photo.get("aspect_ratio"), "photo.aspect_ratio"),
        "views": _require_int(stats.get("views"), "stats.views"),
        "downloads": _require_int(stats.get("downloads"), "stats.downloads"),
        "submitted_at": _extract_submitted_at(photo),
        "created_at": created_at,
        "metadata_": metadata,
    }


def _parse_datetime(value: str) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _extract_created_at(metadata: dict) -> Optional[datetime]:
    candidates = [metadata.get("created_at")]
    photo = metadata.get("photo")
    if isinstance(photo, dict):
        candidates.append(photo.get("submitted_at"))
    for value in candidates:
        if isinstance(value, str):
            parsed = _parse_datetime(value)
            if parsed is not None:
                return parsed
    return None


def _extract_submitted_at(photo: dict) -> Optional[datetime]:
    value = photo.get("submitted_at")
    if isinstance(value, str):
        return _parse_datetime(value)
    return None


def _optional_str(value: Any) -> Optional[str]:
    if _is_missing(value):
        return None
    return str(value)


def _optional_bool(value: Any) -> Optional[bool]:
    if _is_missing(value):
        return None
    return bool(value)


async def upsert_images(session: AsyncSession, entries: list[dict]) -> None:
    if not entries:
        return
    stmt = insert(Image).values(entries)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Image.id],
        set_={
            "embedding": stmt.excluded.embedding,
            "image_url": stmt.excluded.image_url,
            "url": stmt.excluded.url,
            "featured": stmt.excluded.featured,
            "blur_hash": stmt.excluded.blur_hash,
            "description": stmt.excluded.description,
            "width": stmt.excluded.width,
            "height": stmt.excluded.height,
            "aspect_ratio": stmt.excluded.aspect_ratio,
            "views": stmt.excluded.views,
            "downloads": stmt.excluded.downloads,
            "submitted_at": stmt.excluded.submitted_at,
            "created_at": stmt.excluded.created_at,
            Image.metadata_: stmt.excluded.metadata,
        },
    )
    await session.execute(stmt)


async def main() -> None:
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    cols = ["photo", "photographer", "exif", "location", "stats", "keywords"]
    ds = ds.select_columns(cols)
    if MAX_ITEMS > 0:
        limit = min(MAX_ITEMS, len(ds))
        ds = ds.select(range(limit))
    engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
    await wait_for_db(engine)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with httpx.AsyncClient(timeout=10.0) as client:
        await wait_for_service(client, INFERENCE_URL)
        items: list[dict] = []
        batch: list[tuple[int, dict]] = []
        async with Session() as session:
            for idx, row in enumerate(tqdm(ds, total=len(ds), desc="Indexing images")):
                photo = row.get("photo") or {}
                image_url = photo.get("image_url")
                if _is_missing(image_url):
                    continue
                row = _prepare_row(row)
                batch.append((idx, row))
                if len(batch) < INFERENCE_BATCH_SIZE:
                    continue

                fetched = await fetch_image_batch(client, batch)
                batch = []
                if not fetched:
                    continue

                files = [
                    (f"{row['photo'].get('id', idx)}.jpg", "image/jpeg", image_bytes)
                    for idx, row, image_bytes in fetched
                ]
                embeddings = await embed_image_batch(client, files)
                for (idx, row, _), embedding in zip(fetched, embeddings):
                    items.append(prepare_entry(idx, row, embedding))
                while len(items) >= BULK_SIZE:
                    await upsert_images(session, items[:BULK_SIZE])
                    await session.commit()
                    items = items[BULK_SIZE:]

            if batch:
                fetched = await fetch_image_batch(client, batch)
                if fetched:
                    files = [
                        (f"{row['photo'].get('id', idx)}.jpg", "image/jpeg", image_bytes)
                        for idx, row, image_bytes in fetched
                    ]
                    embeddings = await embed_image_batch(client, files)
                    for (idx, row, _), embedding in zip(fetched, embeddings):
                        items.append(prepare_entry(idx, row, embedding))
                while len(items) >= BULK_SIZE:
                    await upsert_images(session, items[:BULK_SIZE])
                    await session.commit()
                    items = items[BULK_SIZE:]
            if items:
                await upsert_images(session, items)
                await session.commit()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS images_embedding_hnsw "
                "ON images USING hnsw (embedding vector_cosine_ops)"
            )
        )
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
