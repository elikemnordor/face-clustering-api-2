import io
import os
import base64
import logging
from typing import List

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, LargeBinary, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import insightface
from insightface.app import FaceAnalysis

# ----------------------------- Logging -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face_api")

# ----------------------------- Database Setup -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./faces.db")

Base = declarative_base()

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    face_id = Column(String, primary_key=True)
    embedding = Column(LargeBinary)
    similarity = Column(Float)

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# ----------------------------- Face Model Setup -----------------------------
face_app = FaceAnalysis(providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0)

# ----------------------------- FastAPI Setup -----------------------------
app = FastAPI(title="Face Clustering API", version="2.0")

class ImageData(BaseModel):
    image_base64: str

# ----------------------------- Helper Functions -----------------------------
def get_embedding(image: Image.Image) -> List[np.ndarray]:
    """Detects multiple faces and returns list of embeddings."""
    arr = np.array(image)
    faces = face_app.get(arr)
    if not faces:
        raise ValueError("No faces detected.")
    embeddings = [f.embedding for f in faces]
    logger.info(f"✅ Detected {len(embeddings)} faces.")
    return embeddings

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_best_match(session, new_embedding: np.ndarray, threshold: float = 0.5):
    """Compare new embedding to existing ones, return (id, similarity)."""
    all_faces = session.query(FaceEmbedding).all()
    best_id, best_score = None, 0.0
    for face in all_faces:
        existing = np.frombuffer(face.embedding, dtype=np.float32)
        score = cosine_similarity(new_embedding, existing)
        if score > best_score:
            best_score, best_id = score, face.face_id
    return (best_id, best_score) if best_score >= threshold else (None, best_score)

# ----------------------------- API Endpoint -----------------------------
@app.post("/process-face")
async def process_face(data: ImageData):
    logger.info("📸 Received Base64 image")
    try:
        image_data = base64.b64decode(data.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        logger.error(f"❌ Invalid image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format")

    try:
        embeddings = get_embedding(image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session = SessionLocal()
    results = []
    try:
        for idx, emb in enumerate(embeddings):
            face_id, sim = find_best_match(session, emb)
            if face_id is None:
                face_id = f"face_{len(session.query(FaceEmbedding).all()) + 1}"
                session.add(FaceEmbedding(
                    face_id=face_id,
                    embedding=emb.astype(np.float32).tobytes(),
                    similarity=0.0
                ))
                session.commit()
                logger.info(f"🆕 New face stored as {face_id}")
            else:
                logger.info(f"👯 Existing face matched: {face_id} (sim={sim:.2f})")

            results.append({
                "status": "success",
                "face_id": face_id,
                "similarity": float(sim),
                "message": "Processed successfully"
            })

        return results
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        raise HTTPException(status_code=500, detail="Processing error")
    finally:
        session.close()

# ----------------------------- Run Server -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
