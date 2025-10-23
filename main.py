import os
import io
import traceback
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from insightface.app import FaceAnalysis
from PIL import Image
import uvicorn
import logging

# ---------------------------------------
# ✅ Logging setup
# ---------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face_api")

# ---------------------------------------
# ✅ App setup
# ---------------------------------------
app = FastAPI(title="Face Clustering API", version="1.1")

# Enable CORS for testing with Colab or n8n
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------
# ✅ Database setup
# ---------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    Base = declarative_base()
except Exception as e:
    logger.error(f"❌ Database connection failed: {e}")
    raise

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    face_id = Column(String, index=True)
    embedding = Column(LargeBinary)
    image_url = Column(String)
    similarity = Column(Float)

Base.metadata.create_all(bind=engine)

# ---------------------------------------
# ✅ Face model init
# ---------------------------------------
try:
    logger.info("Initializing FaceAnalysis model (buffalo_l)...")
    app_insight = FaceAnalysis(name="buffalo_l")
    app_insight.prepare(ctx_id=-1, det_size=(640, 640))  # use CPU
    logger.info("✅ Face model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Face model failed to load: {e}")
    raise

# ---------------------------------------
# ✅ Utility functions
# ---------------------------------------
def get_embedding(image: Image.Image):
    try:
        arr = np.array(image)
        faces = app_insight.get(arr)
        if not faces:
            return None
        return faces[0].embedding.astype(np.float32)
    except Exception as e:
        logger.error(f"❌ Embedding extraction error: {e}")
        raise

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------------------------------
# ✅ Global error handler
# ---------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error(f"Unhandled exception at {request.url}:\n{tb}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_type": exc.__class__.__name__,
            "message": str(exc),
            "trace": tb.splitlines()[-5:],  # last few lines for context
            "hint": "Check Railway logs for full trace or verify inputs.",
        },
    )

# ---------------------------------------
# ✅ Routes
# ---------------------------------------
@app.get("/")
def home():
    return {"message": "Face Clustering API is running"}

@app.post("/process-face")
async def process_face(file: UploadFile = File(...)):
    logger.info(f"📸 Received file: {file.filename}")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        embedding = get_embedding(image)

        if embedding is None:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "No face detected in image",
                    "hint": "Ensure the image contains a clear, front-facing face."
                },
                status_code=400,
            )

        session = SessionLocal()
        embeddings = session.query(FaceEmbedding).all()

        best_match = None
        best_score = 0.0
        for e in embeddings:
            existing_emb = np.frombuffer(e.embedding, dtype=np.float32)
            sim = cosine_similarity(embedding, existing_emb)
            if sim > best_score:
                best_score = sim
                best_match = e

        THRESHOLD = 0.6
        face_id = best_match.face_id if best_match and best_score >= THRESHOLD else f"face_{len(embeddings)+1}"

        new_entry = FaceEmbedding(
            face_id=face_id,
            embedding=embedding.tobytes(),
            image_url=file.filename,
            similarity=float(best_score),
        )
        session.add(new_entry)
        session.commit()
        session.close()

        logger.info(f"✅ Processed {file.filename} | face_id={face_id} | similarity={best_score:.3f}")

        return {
            "status": "success",
            "face_id": face_id,
            "similarity": float(best_score),
            "message": "Processed successfully",
        }

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ Error processing {file.filename}:\n{tb}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error_type": e.__class__.__name__,
                "message": str(e),
                "trace": tb.splitlines()[-5:],
                "hint": "Check if the uploaded file is a valid image.",
            },
        )

# ---------------------------------------
# ✅ Entry point
# ---------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)

