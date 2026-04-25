"""
config.py
---------
Centralized configuration management using Pydantic Settings.
Toggles between 'chroma' and 'pinecone' providers via vector_db_type.
"""
import os
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # Toggle between 'chroma' (local) and 'pinecone' (cloud)
    vector_db_type: Literal["chroma", "pinecone"] = "chroma"
    
    # Chroma Config (used for Vector DB if type=chroma, AND for local BM25 cache)
    chroma_dir: str = "./chroma_data"
    
    # Pinecone Config
    pinecone_api_key: str = os.environ.get("PINECONE_API_KEY", "")
    pinecone_index_name: str = "insurance-policies"
    
    # LLM and Device Config
    admin_password: str = "admin123"
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")
    hf_device: str = "cpu" # Set to 'cuda' for GPU acceleration
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
