import os
from typing import Literal
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

EmbeddingProvider = Literal["local", "openai"]

class EmbeddingsConfig:
    """Configuration management for different embedding providers"""
    
    @staticmethod
    def get_provider() -> EmbeddingProvider:
        """Get the configured embedding provider"""
        provider = os.getenv("EMBEDDINGS_PROVIDER", "local").lower()
        if provider not in ("local", "openai"):
            raise ValueError(f"Invalid EMBEDDINGS_PROVIDER: {provider}. Must be 'local' or 'openai'")
        return provider
    
    @staticmethod
    def create_embeddings() -> OpenAIEmbeddings:
        """Create OpenAIEmbeddings instance pointing to the configured endpoint. The OpenAIEmbeddings class supports both local embedding models from llama-cpp and openai's models."""
        provider = EmbeddingsConfig.get_provider()
        
        if provider == 'openai':
            return OpenAIEmbeddings(
                model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large"),
                api_key=os.getenv('OPENAI_API_KEY')
            )
        else:
            return OpenAIEmbeddings(
                model=os.getenv("LOCAL_EMBEDDINGS_MODEL", "Qwen3-Embedding-4B"),
                base_url=os.getenv("LOCAL_EMBEDDINGS_BASE_URL", "http://localhost:8081/v1"),
                api_key=os.getenv("LOCAL_EMBEDDINGS_API_KEY", "dummy") # For a llama-cpp server, this value is usually ignored unless exlicitly configured in the server
            )