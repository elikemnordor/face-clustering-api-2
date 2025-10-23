from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import io, requests, numpy as np
from sqlalchemy.orm import Session
from models import FaceEmbedding  # your SQLAlchemy model
from database import SessionLocal  # your DB session
from face_utils import get_embedding  # your face embedding function

app = FastAPI()

# Pydantic schema for URL-based input
class ImageInput(BaseModel):
    image_url: Optional[str] = None


@app.post("/process-face")
async def process_face(
    file: Optional[UploadFile] = File(None),
    data: Optional[ImageInput] = None
):
    """
    Accepts either:
    1. Multipart upload with a file
    2. JSON body with { "image_url": "https://..." }
    """

    image_bytes = None

    # Case 1: Direct file upload
    if file is not None:
        image_bytes = await file.read()

    # Case 2: Remote image URL
    elif data and data.image_url:
        try:
            resp = requests.get(data.image_url, timeout=10)
            resp.raise_for_status()
            image_bytes = resp.content
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")

    else:
        raise HTTPException(status_code=422, detail="Field required: file or image_url")

    # ---- Process Image ----
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Extract embedding
    embedding = get_embedding(image)
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)

    embedding = embedding.astype(float).tolist()

    # ---- Store in Database ----
    session: Session = SessionLocal()
    try:
        face_entry = FaceEmbedding(embedding=embedding)
        session.add(face_entry)
        session.commit()
        session.refresh(face_entry)
        face_id = face_entry.id
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

    # ---- Return Response ----
    return {
        "message": "Face processed successfully",
        "face_id": face_id,
    }

# ---------------------------------------
# ✅ Entry point
# ---------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)

