import io
import base64
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import face_recognition
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, Float
from sqlalchemy.orm import sessionmaker, declarative_base
import uuid
import os

# ----------------------------
# DATABASE SETUP
# ----------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    face_id = Column(String, index=True)
    embedding = Column(LargeBinary)  # Store embedding bytes
    similarity = Column(Float, default=1.0)
    image_url = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)

# ----------------------------
# FASTAPI APP
# ----------------------------
app = FastAPI(title="Face Clustering API (Base64-ready)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# REQUEST MODEL
# ----------------------------
class FaceRequest(BaseModel):
    image_base64: str  # base64 string of an image

# ----------------------------
# HELPERS
# ----------------------------
def get_face_embedding(image: Image.Image):
    """Extract 128-d face embedding from a PIL Image."""
    rgb_image = np.array(image.convert("RGB"))
    face_locations = face_recognition.face_locations(rgb_image)
    if not face_locations:
        raise ValueError("No face detected in the image.")
    encodings = face_recognition.face_encodings(rgb_image, face_locations)
    if not encodings:
        raise ValueError("Failed to encode face.")
    return encodings[0]

def calculate_similarity(embedding1, embedding2):
    """Cosine similarity between two embeddings."""
    dot = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return float(dot / (norm1 * norm2))

# ----------------------------
# MAIN ENDPOINT
# ----------------------------
@app.post("/process-face")
def process_face(request: FaceRequest):
    try:
        # Decode Base64 image
        image_data = base64.b64decode(request.image_base64.split(",")[-1])
        image = Image.open(io.BytesIO(image_data))

        # Extract embedding
        embedding = get_face_embedding(image)
        embedding_bytes = embedding.astype(np.float32).tobytes()

        db = SessionLocal()
        faces = db.query(FaceEmbedding).all()

        # Compare with existing embeddings
        match_face_id = None
        highest_similarity = 0.0
        for face in faces:
            existing = np.frombuffer(face.embedding, dtype=np.float32)
            sim = calculate_similarity(existing, embedding)
            if sim > 0.6 and sim > highest_similarity:
                highest_similarity = sim
                match_face_id = face.face_id

        # Create or reuse face_id
        if match_face_id:
            face_id = match_face_id
        else:
            face_id = str(uuid.uuid4())
            db.add(FaceEmbedding(face_id=face_id, embedding=embedding_bytes))

        db.commit()
        db.close()

        return {
            "status": "success",
            "face_id": face_id,
            "match": bool(match_face_id),
            "similarity": highest_similarity
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
