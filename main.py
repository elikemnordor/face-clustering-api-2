import os
import io
import base64
import numpy as np
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from insightface.app import FaceAnalysis
from sqlalchemy import create_engine, Column, Integer, LargeBinary, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face_api")

# -------------------------------------------------------------------
# Database setup
# -------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./faces.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    embedding = Column(LargeBinary)
Base.metadata.create_all(bind=engine)

# -------------------------------------------------------------------
# InsightFace setup
# -------------------------------------------------------------------
app_insight = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app_insight.prepare(ctx_id=0)

# -------------------------------------------------------------------
# FastAPI setup
# -------------------------------------------------------------------
app = FastAPI(title="Multi-Face Clustering API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_best_match(embedding, db_embeddings):
    """Find the most similar face embedding in DB"""
    best_match_id = None
    best_similarity = 0.0

    for face in db_embeddings:
        emb_db = np.frombuffer(face.embedding, dtype=np.float32)
        sim = cosine_similarity(embedding, emb_db)
        if sim > best_similarity:
            best_similarity = sim
            best_match_id = face.id
    return best_match_id, best_similarity

def save_new_embedding(session, embedding):
    new_face = FaceEmbedding(embedding=embedding.tobytes())
    session.add(new_face)
    session.commit()
    session.refresh(new_face)
    return new_face.id

# -------------------------------------------------------------------
# Request model
# -------------------------------------------------------------------
class FaceRequest(BaseModel):
    image_base64: str

# -------------------------------------------------------------------
# Main endpoint
# -------------------------------------------------------------------
@app.post("/process-face")
async def process_face(image_base64: str = Form(...)):
    logger.info("📸 Received Base64 image")

    try:
        # Decode the base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        logger.error(f"❌ Failed to decode image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Convert image to numpy array
    img_np = np.array(image)
    faces = app_insight.get(img_np)

    if not faces:
        logger.warning("🚫 No faces detected in image")
        return [{"status": "error", "message": "No faces detected"}]

    logger.info(f"🧠 Detected {len(faces)} face(s)")

    # Load all embeddings from DB
    db = SessionLocal()
    db_faces = db.query(FaceEmbedding).all()

    THRESHOLD = 0.5  # Adjust similarity threshold
    results = []

    # Process all detected faces
    for idx, face in enumerate(faces):
        embedding = face.normed_embedding
        match_id, similarity = find_best_match(embedding, db_faces)

        if match_id is not None and similarity >= THRESHOLD:
            result = {
                "status": "success",
                "face_id": f"face_{match_id}",
                "similarity": float(similarity),
                "message": f"Matched existing face (face_{match_id})",
            }
            logger.info(f"✅ Matched existing face ID {match_id} (sim={similarity:.3f})")
        else:
            new_id = save_new_embedding(db, embedding)
            result = {
                "status": "success",
                "face_id": f"face_{new_id}",
                "similarity": 0.0,
                "message": "New face added",
            }
            logger.info(f"🆕 New face added with ID {new_id}")

        results.append(result)

    db.close()
    return results
