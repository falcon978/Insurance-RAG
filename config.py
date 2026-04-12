import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    chroma_dir: str = "./chroma_data"
    admin_password: str = "admin123"
    gemini_api_key: str = "" # from ENV
    hf_device: str = "cpu"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()