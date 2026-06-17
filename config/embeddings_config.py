import os
from typing import Literal, List
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EmbeddingProvider = Literal["local", "openai", "sentence_transformers"]

class SentenceTransformersEmbedding(Embeddings):
    """Custom embedding class wrapping SentenceTransformers models"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("SENTENCE_TRANSFORMERS_MODEL", "BAAI/bge-m3")
        self.model = SentenceTransformer(self.model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search documents using SentenceTransformers"""
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed search query using SentenceTransformers"""
        return self.model.encode(text, convert_to_numpy=True).tolist()
        

class EmbeddingsConfig:
    """Configuration management for different embedding providers"""
    
    @staticmethod
    def get_provider() -> EmbeddingProvider:
        """Get the configured embedding provider"""
        provider = os.getenv("EMBEDDINGS_PROVIDER", "local").lower()
        if provider not in ("local", "openai", "sentence_transformers"):
            raise ValueError(f"Invalid EMBEDDINGS_PROVIDER: {provider}. Must be 'local', 'openai', or 'sentence_transformers'")
        return provider
    
    @staticmethod
    def create_embeddings():
        """Create embeddings instance based on the configured provider"""
        provider = EmbeddingsConfig.get_provider()
        
        if provider == 'openai':
            return OpenAIEmbeddings(
                model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large"),
                api_key=os.getenv('OPENAI_API_KEY')
            )
        elif provider == 'sentence_transformers':
            return SentenceTransformersEmbedding(
                model_name=os.getenv("SENTENCE_TRANSFORMERS_MODEL", "BAAI/bge-m3")
            )
        else:
            return OpenAIEmbeddings(
                model=os.getenv("LOCAL_EMBEDDINGS_MODEL", "Qwen3-Embedding-4B"),
                base_url=os.getenv("LOCAL_EMBEDDINGS_BASE_URL", "http://localhost:8081/v1"),
                api_key=os.getenv("LOCAL_EMBEDDINGS_API_KEY", "dummy") # For a llama-cpp server, this value is usually ignored unless exlicitly configured in the server
            )