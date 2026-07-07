from abc import ABC, abstractmethod
from typing import List

from models.dataset import Dataset


class BaseMetadataConnector(ABC):
    """Abstract base class for all metadata source connectors"""
    
    @abstractmethod
    def fetch_datasets(self) -> List[Dataset]:
        """Fetch datasets from the source and return them as Dataset objects"""
