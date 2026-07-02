## Harvests metadata from configured sources and populates the vector and spatial index.
## Run once when initializing the application.

import logging
from typing import List

from connectors.base_connector import BaseMetadataConnector
from connectors.csw_connector import CSWConnector
from connectors.edp_connector import EDPConnector
from connectors.rdf_file_connector import RDFFileConnector
from models.dataset import Dataset
from pg_database.postgis_db import PostGISService
from vector_stores.qdrant_store import QdrantVectorStoreManager

# ── Source configuration ───────────────────────────────────────────────────────

LOCAL_RDF_FILE = "data/gdi_de_catalog.rdf"

EDP_CATALOGUE_ID = "nipp"
EDP_LANGUAGE = "hr"
EDP_LIMIT = 100

# CSW_URL = "https://metawal.wallonie.be/geonetwork/srv/eng/csw"
CSW_URL = "https://atlas.thuenen.de/catalogue/csw"

QDRANT_BATCH_SIZE = 500

# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_datasets_in_postgis(datasets: List[Dataset]) -> bool:
    logger.info(f"Indexing {len(datasets)} datasets in PostGIS")
    try:
        service = PostGISService()
        service.connect()
        service.initialize_schema()
        inserted = service.insert_datasets(datasets)
        service.disconnect()
        logger.info(f"Inserted {inserted} datasets into PostGIS")
        return True
    except Exception as e:
        logger.error(f"Error indexing in PostGIS: {e}")
        return False


def index_datasets_in_qdrant(datasets: List[Dataset],
                              batch_size: int = QDRANT_BATCH_SIZE) -> bool:
    logger.info(f"Indexing {len(datasets)} datasets in Qdrant (batch size: {batch_size})")
    try:
        store = QdrantVectorStoreManager()
        store.initialize()

        total = len(datasets)
        for i in range(0, total, batch_size):
            batch = datasets[i:i + batch_size]
            store.add_datasets(batch)
            batch_num = (i // batch_size) + 1
            total_batches = (total + batch_size - 1) // batch_size
            logger.info(
                f"Batch {batch_num}/{total_batches}: "
                f"added {len(batch)} datasets ({i + len(batch)}/{total} total)"
            )

        logger.info(f"Finished adding {total} datasets to Qdrant")
        return True
    except Exception as e:
        logger.error(f"Error indexing in Qdrant: {e}")
        return False


def index_datasets(datasets: List[Dataset],
                   batch_size: int = QDRANT_BATCH_SIZE) -> bool:
    """Index datasets in both PostGIS and Qdrant."""
    return (
        index_datasets_in_postgis(datasets)
        and index_datasets_in_qdrant(datasets, batch_size=batch_size)
    )


# ── Harvesting ────────────────────────────────────────────────────────────────

def run_connector(connector: BaseMetadataConnector) -> List[Dataset]:
    """Run a single connector and return whatever datasets it produces."""
    name = type(connector).__name__
    logger.info(f"Running {name}...")
    try:
        datasets = connector.fetch_datasets()
        logger.info(f"{name}: fetched {len(datasets)} datasets")
        return datasets
    except Exception as e:
        logger.error(f"{name} failed: {e}")
        return []


def harvest_and_index(connectors: List[BaseMetadataConnector],
                      batch_size: int = QDRANT_BATCH_SIZE) -> bool:
    """
    Run every connector, merge the results, and index them.

    Returns True if at least one dataset was successfully indexed.
    """
    all_datasets: List[Dataset] = []
    for connector in connectors:
        all_datasets.extend(run_connector(connector))

    if not all_datasets:
        logger.error("No datasets were collected from any connector")
        return False

    logger.info(f"Total datasets collected: {len(all_datasets)}")
    return index_datasets(all_datasets, batch_size=batch_size)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    connectors: List[BaseMetadataConnector] = [
        RDFFileConnector(file_path=LOCAL_RDF_FILE),
        EDPConnector(
            catalogue_id=EDP_CATALOGUE_ID,
            language=EDP_LANGUAGE,
            limit=EDP_LIMIT,
        ),
        CSWConnector(
            url=CSW_URL
        ),
    ]

    success = harvest_and_index(connectors)
    if not success:
        logger.error("Harvest failed!")
        exit(1)

    logger.info("Harvest completed successfully")
