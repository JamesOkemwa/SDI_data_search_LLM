import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))  # Ensure port is an integer
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "dcat_collection")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RDF_FILE_PATH = DATA_DIR / "gdi_de_catalog.rdf"