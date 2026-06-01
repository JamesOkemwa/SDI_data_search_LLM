import os
from typing import Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

LLMProvider = Literal["local", "openai"]

class LLMConfig:
    """Configuration managament for different LLM providers"""
    
    @staticmethod
    def get_provider() -> LLMProvider:
        """Get the configured LLM Provider"""
        provider = os.getenv("LLM_PROVIDER", "local").lower()
        if provider not in ("local", "openai"):
            raise ValueError(f"Invalid LLM_PROVIDER: {provider}. Must be 'local' or 'openai'")
        return provider
    
    @staticmethod
    def create_llm(temperature: float = 0.0) -> ChatOpenAI:
        """Function to create an LLM instance for Langchain based on the configuration"""
        provider = LLMConfig.get_provider()
        temp = temperature or float(os.getenv("LLM_TEMPERATURE", "0.3"))
        
        if provider == "openai":
            return ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=temp
            )
        else:
            return ChatOpenAI(
                model=os.getenv("LOCAL_LLM_MODEL", "gemma-4"),
                base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8080/v1"),
                api_key=os.getenv("LOCAL_LLM_API_KEY", "dummy"),
                temperature=temp
            )