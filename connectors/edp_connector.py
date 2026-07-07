import logging
from typing import List, Optional

import requests
from rdflib import Graph, Literal, Namespace, RDF, URIRef

from connectors.base_connector import BaseMetadataConnector
from models.dataset import Dataset

DCT = Namespace("http://purl.org/dc/terms/")
DCAT = Namespace("http://www.w3.org/ns/dcat#")
LOCN = Namespace("http://www.w3.org/ns/locn#")

EDP_SEARCH_API = "https://data.europa.eu/api/hub/search/datasets"
EDP_METADATA_API = "https://data.europa.eu/api/hub/repo/datasets"

logger = logging.getLogger(__name__)


class EDPConnector(BaseMetadataConnector):
    """Fetches datasets from the European Data Portal (data.europa.eu) API."""
    
    def __init__(self, catalogue_id: str, language: str = "en", limit: int = 100, start_index: int = 0, timeout: int = 10):
        self.catalogue_id = catalogue_id
        self.language = language
        self.limit = limit
        self.start_index = start_index
        self.timeout = timeout
        
    def fetch_datasets(self) -> List[Dataset]:
        dataset_ids = self._list_catalogue_datasets()
        if not dataset_ids:
            logger.error("Failed to fetch dataset IDs from EDP")
            return []
        
        dataset_ids = dataset_ids[self.start_index:]
        logger.info(f"Processing {len(dataset_ids)} EDP datasets from index {self.start_index}")
        
        datasets = []
        for dataset_id in dataset_ids:
            dataset = self._process_dataset(dataset_id)
            if dataset:
                datasets.append(dataset)
            else:
                logger.warning(f"Failed to process EDP dataset {dataset_id}")
                
        return datasets
    
    def _list_catalogue_datasets(self) -> List[str]:
        """Fetch datasets IDs for a particular catalogue from the EDP"""
        url = f"{EDP_SEARCH_API}?catalogue={self.catalogue_id}&limit={self.limit}"
        try:
            logger.info(f"Fetching datasets from EDP catalogue: '{self.catalogue_id}'...")
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            dataset_ids = response.json()
            logger.info(f"Found {len(dataset_ids)} EDP datasets")
            return dataset_ids
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching EDP dataset list: {e}")
            return []
        
    def _process_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """Process a single dataset: fetch its metadata, parse it, and create Dataset object."""
        url = f"{EDP_METADATA_API}/{dataset_id}.jsonld"
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()

            graph = Graph()
            graph.parse(data=response.text, format="json-ld")

            dataset_uri = None
            for subject in graph.subjects(RDF.type, DCAT.Dataset):
                dataset_uri = subject
                break

            if not dataset_uri:
                logger.warning(f"No DCAT Dataset found for {dataset_id}")
                return None

            title = self._get_localized_value(graph, dataset_uri, DCT.title)
            description = self._get_localized_value(graph, dataset_uri, DCT.description)

            keywords = [
                str(kw) for kw in graph.objects(dataset_uri, DCAT.keyword)
                if isinstance(kw, Literal) and kw.language == self.language
            ] or ["N/A"]

            spatial_extent = self._extract_spatial_extent(graph, dataset_uri)

            access_urls, download_urls = [], []
            for dist in graph.objects(dataset_uri, DCAT.distribution):
                if url_ := graph.value(dist, DCAT.accessURL):
                    access_urls.append(str(url_))
                if url_ := graph.value(dist, DCAT.downloadURL):
                    download_urls.append(str(url_))

            dataset = Dataset(
                dataset_id=dataset_id,
                titles=[title],
                descriptions=[description],
                keywords=keywords,
                access_urls=access_urls or ["N/A"],
                download_urls=download_urls or ["N/A"],
                spatial_extent=spatial_extent,
            )
            
            logger.info(f"Processed: {dataset_id}")
            return dataset
        
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching EDP metadata for {dataset_id}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error processing EDP dataset {dataset_id}: {e}")
            return None

    def _get_localized_value(self, graph: Graph, subject: URIRef,
                             predicate: URIRef) -> str:
        """
        Extract a localized value from RDF graph.
    
        Attempts to find a value in the specified language. Falls back to
        any available literal if the target language is not found.
        """
        for obj in graph.objects(subject, predicate):
            if isinstance(obj, Literal) and obj.language == self.language:
                return str(obj)
        for obj in graph.objects(subject, predicate):
            if isinstance(obj, Literal):
                return str(obj)
        return "N/A"

    def _extract_spatial_extent(self, graph: Graph, dataset_uri: URIRef) -> Optional[str]:
        """Extract the spatial extents of a dataset"""
        for spatial in graph.objects(dataset_uri, DCT.spatial):
            bbox = graph.value(spatial, DCAT.bbox)
            geom = graph.value(spatial, LOCN.geometry)
            if bbox:
                return str(bbox)
            if geom:
                return str(geom)
        return None