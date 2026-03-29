# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers.chat import router as chat_router
from backend.database import engine, Base
import os
from huggingface_hub import HfApi

# Download embeddings from HF Dataset on startup
api = HfApi()
api.download_folder(
    repo_id="Parv09/medassist-embeddings",
    repo_type="dataset",
    local_dir="data/embeddings",
    token=os.environ.get("HF_TOKEN")
)

# Create tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title       = "MedAssist API",
    description = "AI-powered medical symptom checker",
    version     = "1.0.0"
)

# CORS — allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],  # restrict to frontend URL in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Include routers
app.include_router(chat_router)

@app.get("/")
def root():
    return {"status": "MedAssist API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}