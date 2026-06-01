import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
from config.llm_config import LLMConfig

load_dotenv()

# configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryIntent(BaseModel):
    """Structured representation of a geospatial query intent."""

    raw_theme: str = Field(description="Core search topic/phrase verbatim from the query")
    locations: List[str] = Field(default_factory=list, description="List of place names or locations for geocoding")
    themes: List[str] = Field(default_factory=list, description="Main themes from the query that have not explicitly been stated")
    publishers: List[str] = Field(default_factory=list, description="List of publishers or data sources mentioned")
    language: str = Field(description="Language used in the query")

    @validator('raw_theme')
    def validate_raw_theme(cls, v):
        """Ensure raw theme is not empty."""
        if not v or not v.strip():
            raise ValueError("raw_theme cannot be empty")
        return v.strip()

    @property
    def has_location(self) -> bool:
        """Check if the query contains location mentions"""
        return len(self.locations) > 0
    
    @property
    def core_search_terms(self) -> List[str]:
        """Get core search terms combining raw_theme and themes."""
        terms = [self.raw_theme]
        terms.extend(self.themes)
        return [term.strip() for term in terms if term.strip()]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "raw_theme": self.raw_theme,
            "locations": self.locations,
            "themes": self.themes,
            "publishers": self.publishers,
            "language": self.language,
            "has_location": self.has_location,
            "core_search_terms": self.core_search_terms
        }
    

class QueryParser:
    """ Parser for extracting structured query intent from natural language queries."""

    SYSTEM_PROMPT = """You are a geospatial query specialist that helps users find spatial datasets. 
    You excel at understanding what users are looking for in terms of geographic locations, data themes, and publishers."""

    USER_PROMPT = """
    Your task is to extract from this dataset search query:
        1. raw_theme: Raw theme or core search phrase (exact user wording for the main topic)
        2. locations: Place names that will be used for geocoding. Include cities, countries, regions - anything that can be geocoded.
        3. themes: Main themes, keywords or topics that have not explicitly been stated. They should be in the same language as the raw_theme. Includes themes/keywords/topics such as (traffic, weather, population, transportation, environment).
        4. publishers: Organizations, agencies, or data publishers mentioned (e.g., "city of Berlin", "European Space Agency")
        5. language: Language used in the user query (e.g English)

    Format your response as a JSON that matches this pydantic schema:
    {format_instructions}

    Query: {query}
    """

    def __init__(self):
        """Initialize the parser."""
        self.model = LLMConfig.create_llm()
        self.parser = PydanticOutputParser(pydantic_object=QueryIntent)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                ("user", self.USER_PROMPT)
            ]
        )

        self.chain = self.prompt | self.model | self.parser

    def parse(self, query:str) -> QueryIntent:
        """Parse a natural language query into a structured intent."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            logger.info(f"Parsing query: {query}")
            result = self.chain.invoke({"query": query, "format_instructions": self.parser.get_format_instructions()})
            logger.info(f"Successfully parsed: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to parse query: {e}")
            raise