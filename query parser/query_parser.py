import logging
from typing import List, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration for the query parser."""
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.0


class QueryIntent(BaseModel):
    """Structured representation of a geospatial query intent."""

    raw_theme: str = Field(description="Core search topic/phrase verbatim from the query")
    locations: List[str] = Field(default_factory=list, description="List of place names or locations for geocoding")
    themes: List[str] = Field(default_factory=list, description="Main themes from the query")
    publishers: List[str] = Field(default_factory=list, description="List of publishers or data sources mentioned")

    @validator('raw_theme')
    def validate_raw_theme(cls, v):
        """Ensure raw theme is not empty."""
        if not v or not v.strip():
            raise ValueError("raw_theme cannot be empty")
        return v.strip()
    

class QueryParser:
    """ Parser for extracting structured query intent from natural language queries."""

    SYSTEM_PROMPT = "You are a geospatial query specialist"
    USER_PROMPT = """
    Your task is to extract from this dataset search query:
        1. raw_theme: Raw theme or core search theme (exact user wording)
        2. locations: Place names for that will be used later for geocoding. A location can be a town, city, country or region which can be geocoded using a geocoding API.
        3. themes: Main themes, keywords and topics that are relevant to the query. Where possible, add two themes that match the user query
        4. publishers: Organizations or data publishers mentioned
    Format your response as a JSON that matches this pydantic schema:
    {format_instructions}

    Query: {query}
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize the parser."""
        self.config = config or Config()
        self.model = ChatOpenAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature
        )
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
            logger.info("Parsing query: {query}")
            result = self.chain.invoke({"query": query, "format_instructions": self.parser.get_format_instructions()})
            logger.info("Successfully parsed: {result.raw_theme}")
            return result
        except Exception as e:
            logger.error(f"Failed to parse query: {e}")
            raise

def parse_query(query: str, config: Optional[Config] = None) -> QueryIntent:
    """Convenience function to parse a query using the default configuration."""
    parser = QueryParser(config)
    return parser.parse(query)


if __name__ == "__main__":
    load_dotenv()

    test_queries = [
        "I am looking for datasets about cycling paths in Muenster, Germany.",
        "What data is available on air quality from the European Environmental Agency?",
        "Where can I find information about agricultural areas for 2025 in Bavaria?"
    ]

    # parser = QueryParser()

    for query in test_queries:
        try:
            result = parse_query(query)
            print(f"\nQuery: {query}")
            print(f"Theme: {result.raw_theme}")
            print(f"Locations: {result.locations}")
            print(f"Themes: {result.themes}")
            print(f"Publishers: {result.publishers}")
        except Exception as e:
            pass