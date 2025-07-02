from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

@dataclass
class Dataset:
    """Represents a DCAT dataset with its metadata."""
    titles: List[str]
    descriptions: List[str]
    keywords: List[str]
    access_urls: List[str]
    download_urls: List[str]
    dataset_id: Optional[str] = None

    @property
    def primary_title(self) -> Optional[str]:
        """
        DCAT Datasets may contain multiple titles in different languages. 
        Returns the first title or None if no titles exist.
        """
        return self.titles[0] if self.titles else None
    
    def to_content(self) -> str:
        """
        Combines the dataset's metadata into a single text chunk. This is useful for embedding and searching.
        """

        content_parts = []
        if self.titles:
            content_parts.append(f"Title: {'; '.join(self.titles)}")
        if self.descriptions:
            content_parts.append(f"Description: {'; '.join(self.descriptions)}")
        if self.keywords:
            content_parts.append(f"Keywords: {', '.join(self.keywords)}")

        return "\n".join(content_parts)

    def to_metadata(self) -> dict:
        """
        Returns the dataset's metadata as a dictionary. This is useful for filtering and searching.
        """
        return {
            "title": self.primary_title,
            "dataset_id": self.dataset_id,
            "keywords": self.keywords,
            "access_urls": self.access_urls,
            "download_urls": self.download_urls,
        }

        