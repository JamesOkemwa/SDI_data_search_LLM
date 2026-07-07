import logging
from typing import List

from connectors.base_connector import BaseMetadataConnector
from models.dataset import Dataset
from parsers.rdf_parser import RDFParser

logger = logging.getLogger(__name__)


class RDFFileConnector(BaseMetadataConnector):
    """Harvests datasets from a local RDF/XML file using the existing RDFParser"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def fetch_datasets(self) -> List[Dataset]:
        parser = RDFParser()
        datasets = parser.parse_file(self.file_path)
        logger.info(f"RDFFileConnector: parsed {len(datasets)} datasets from {self.file_path}")
        return datasets