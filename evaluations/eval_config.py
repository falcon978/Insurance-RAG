"""
eval_config.py
--------------
Centralized configuration for the DeepEval test suite.
Manages judge models, API keys, metric thresholds, and rate limits.
By keeping this modular, you can easily swap models or strictness for different environments.
"""
import os
from pydantic_settings import BaseSettings

class EvalSettings(BaseSettings):
    # API Keys
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    confident_api_key: str = os.getenv("CONFIDENT_API_KEY", "")

    # Judge Model Configuration
    # Using Llama 3.3 70B as the default strong judge
    judge_model_name: str = os.getenv("JUDGE_MODEL_NAME", "llama-3.3-70b-versatile")

    # Top K Retrieval and Reranking Configurations
    retrieve_top_k = int(os.getenv("retrieve_top_k", "15"))
    rerank_top_k = int(os.getenv("rerank_top_k", "5"))
    
    # Passing Thresholds for DeepEval Standard Metrics
    recall_threshold: float = 0.8
    precision_threshold: float = 0.8
    faithfulness_threshold: float = 0.9
    relevancy_threshold: float = 0.8
    
    # Passing Threshold for Custom Rubrics (GEval)
    reasoning_threshold: float = 0.7

    # Rate Limiting (Crucial for free tier APIs like Groq)
    # Pauses the pytest loop to avoid HTTP 429 Too Many Requests errors
    rate_limit_delay_seconds: int = 30

    class Config:
        env_file = ".env"
        extra = "ignore"

# Instantiate the settings so they can be imported globally
eval_settings = EvalSettings()