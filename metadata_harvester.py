## Script for harvesting metadata from the local DCAT metadata XML file and the data.europa.eu platform and populating the vector and spatial index
## Run it only once when initializing the application

import logging
import requests
from typing import List
from rdflib import Graph, Namespace, URIRef, Literal, RDF

from models.dataset import Dataset
from parsers.rdf_parser import RDFParser
from vector_stores.qdrant_store import QdrantVectorStoreManager
from pg_database.postgis_db import PostGISService

# Configuration
CATALOGUE_ID = "nipp"
LANGUAGE = "hr"
LIMIT = 100
START_INDEX = 0
QDRANT_BATCH_SIZE = 100

# API endpoints
EDP_SEARCH_API = "https://data.europa.eu/api/hub/search/datasets"
EDP_METADATA_API = "https://data.europa.eu/api/hub/repo/datasets"

# RDF Namespaces
DCT = Namespace("http://purl.org/dc/terms/")
DCAT = Namespace("http://www.w3.org/ns/dcat#")
LOCN = Namespace("http://www.w3.org/ns/locn#")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")
PROV = Namespace("http://www.w3.org/ns/prov#")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def list_catalogue_datasets(catalogue_id: str, limit: int = 100) -> List:
    """
    Fetch dataset IDs from a European Data Portal catalogue.
    
    Args:
        catalogue_id: The catalogue identifier (e.g., 'nipp')
        limit: Maximum number of dataset IDs to return
        
    Returns:
        List of dataset IDs, or empty list if request fails
    """
    url = f"{EDP_SEARCH_API}?catalogue={catalogue_id}&limit={limit}"
    
    try:
        logger.info(f"Fetching datasets from catalogue '{catalogue_id}'...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        dataset_ids = response.json()
        logger.info(f"Found {len(dataset_ids)} datasets")
        return dataset_ids
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching datasets: {e}")
        return []
    
def get_localized_value(graph: Graph, subject: URIRef, 
                       predicate: URIRef, language: str) -> str:
    """
    Extract a localized value from RDF graph.
    
    Attempts to find a value in the specified language. Falls back to
    any available literal if the target language is not found.
    
    Args:
        graph: RDF graph
        subject: Subject URI
        predicate: Predicate URI
        language: Language code (e.g., 'hr', 'en', 'de')
        
    Returns:
        Localized string value or 'N/A' if not found
    """
    # Try to find value in target language
    for obj in graph.objects(subject, predicate):
        if isinstance(obj, Literal) and obj.language == language:
            return str(obj)
    
    # Fall back to first available literal
    for obj in graph.objects(subject, predicate):
        if isinstance(obj, Literal):
            return str(obj)
    
    return "N/A"

def process_dataset(dataset_id: str, language: str) -> Dataset:
    """
    Process a single dataset: fetch metadata, parse it, and create Dataset object.
    
    Args:
        dataset_id: The dataset identifier
        language: Language code for extracting localized content
        
    Returns:
        Dataset object, or None if processing fails
    """
    url = f"{EDP_METADATA_API}/{dataset_id}.jsonld"
    
    try:
        # Fetch JSON-LD metadata
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        json_ld_data = response.text
        
        # Parse JSON-LD to RDF graph
        graph = Graph()
        graph.parse(data=json_ld_data, format="json-ld")
        
        # Find dataset URI
        dataset_uri = None
        for subject in graph.subjects(RDF.type, DCAT.Dataset):
            dataset_uri = subject
            break
        
        if not dataset_uri:
            logger.warning(f"No DCAT Dataset found for {dataset_id}")
            return None
        
        # Extract title and description
        title = get_localized_value(graph, dataset_uri, DCT.title, language)
        description = get_localized_value(graph, dataset_uri, DCT.description, language)
        
        # Extract keywords
        keywords = [
            str(kw) for kw in graph.objects(dataset_uri, DCAT.keyword)
            if isinstance(kw, Literal) and kw.language == language
        ]
        keywords = keywords or ["N/A"]
        
        # Extract spatial extent
        spatial_extent = []
        for spatial in graph.objects(dataset_uri, DCT.spatial):
            bbox = graph.value(spatial, DCAT.bbox)
            geom = graph.value(spatial, LOCN.geometry)
            if bbox:
                spatial_extent.append(str(bbox))
            elif geom:
                spatial_extent.append(str(geom))
            else:
                spatial_extent.append(str(spatial))
        spatial_extent = spatial_extent or ["N/A"]
        
        # Extract access and download URLs
        access_urls = []
        download_urls = []
        for dist in graph.objects(dataset_uri, DCAT.distribution):
            access_url = graph.value(dist, DCAT.accessURL)
            download_url = graph.value(dist, DCAT.downloadURL)
            if access_url:
                access_urls.append(str(access_url))
            if download_url:
                download_urls.append(str(download_url))
        
        access_urls = access_urls or ["N/A"]
        download_urls = download_urls or ["N/A"]
        
        # Create and return Dataset object
        dataset = Dataset(
            dataset_id=dataset_id,
            titles=[title],
            descriptions=[description],
            keywords=keywords,
            access_urls=access_urls,
            download_urls=download_urls,
            spatial_extent=spatial_extent[0]
        )
        
        logger.info(f"Processed: {dataset_id}")
        return dataset
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching metadata for {dataset_id}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error processing {dataset_id}: {e}")
        return None

def index_datasets_in_postgis(datasets: List[Dataset]) -> bool:
    """Index datasets in PostGIS database"""
    logger.info("Indexing datasets in PostGIS")
    try:
        postgis_service = PostGISService()
        postgis_service.connect()
        postgis_service.initialize_schema()
        inserted_count = postgis_service.insert_datasets(datasets)
        postgis_service.disconnect()
        logger.info(f"Inserted {inserted_count} datasets into PostGIS")
        return True
    except Exception as e:
        logger.error(f"Error indexing in PostGIS: {e}")
        return False
    
def index_datasets_in_qdrant(datasets: List[Dataset], batch_size: int = 10) -> bool:
    """Index datasets in Qdrant vector store."""
    logger.info(f"Indexing {len(datasets)} datasets in Qdrant (batch size: {batch_size})...")
    try:
        vector_store = QdrantVectorStoreManager()
        vector_store.initialize()
        
        total_datasets = len(datasets)
        for i in range(0, total_datasets, batch_size):
            batch = datasets[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_datasets + batch_size - 1) // batch_size
            
            vector_store.add_datasets(batch)
            logger.info(f"Batch {batch_num}/{total_batches}: Added {len(batch)} datasets ({i + len(batch)}/{total_datasets} total)")
            
        logger.info(f"Added {len(datasets)} datasets to Qdrant")
        return True
    except Exception as e:
        logger.error(f"Error indexing in Qdrant: {e}")
        return False

def index_datasets(datasets: List[Dataset], batch_size: int = QDRANT_BATCH_SIZE) -> bool:
    """Index the datasets in both the vector database and spatial index"""
    return (
        index_datasets_in_postgis(datasets) and 
        index_datasets_in_qdrant(datasets, batch_size=batch_size)
    )
    
def harvest_and_index_datasets(catalogue_id: str = CATALOGUE_ID,
                               language: str = LANGUAGE,
                               limit: int = LIMIT,
                               start_index: int = START_INDEX) -> bool:
    """
    Main harvest function: fetch datasets, process them, and index in databases.
    
    Args:
        catalogue_id: The catalogue to harvest from
        language: Language code for metadata extraction
        limit: Maximum number of datasets to harvest
        start_index: Starting index in the dataset list
        
    Returns:
        True if harvest succeeded, False otherwise
    """
    logger.info(f"Starting harvest from catalogue '{catalogue_id}' in language '{language}'")
    
    # Fetch dataset IDs
    dataset_ids = list_catalogue_datasets(catalogue_id, limit=limit)
    if not dataset_ids:
        logger.error("Failed to fetch dataset IDs")
        return False
    
    # Apply start index
    dataset_ids = dataset_ids[start_index:]
    logger.info(f"Processing {len(dataset_ids)} datasets starting from index {start_index}")
    
    # Process each dataset
    processed_datasets = []
    for dataset_id in dataset_ids:
        dataset = process_dataset(dataset_id, language=language)
        if dataset:
            processed_datasets.append(dataset)
        else:
            logger.warning(f"Failed to process dataset {dataset_id}")
    
    if not processed_datasets:
        logger.error("No datasets were successfully processed")
        return False
    
    logger.info(f"Successfully processed {len(processed_datasets)} datasets")
    
    return index_datasets(processed_datasets)

def harvest_from_local_file(file_path: str='data/gdi_de_catalog.rdf') -> bool:
    """Harvest datasets from a local RDF file and index them"""

    try:
        parser = RDFParser()
        datasets = parser.parse_file(file_path)

        logger.info(f"Parsed {len(datasets)} datasets from local file")

        return index_datasets(datasets)
    except Exception as e:
        logger.error(f"Error harvesting from local file: {e}")
        return False

if __name__ == "__main__":
    local_file_success = harvest_from_local_file()
    if local_file_success:
        logger.info("Local harvest succeeded")

    harvest_success = harvest_and_index_datasets(
        catalogue_id=CATALOGUE_ID,
        language=LANGUAGE,
        limit=LIMIT,
        start_index=START_INDEX
    )

    if not (local_file_success or harvest_success):
        logger.error("Harvest failed!")
        exit(1)