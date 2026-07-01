from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Dataset:
    """Represents a DCAT dataset with its metadata."""
    dataset_id: str
    titles: List[str]
    descriptions: List[str]
    keywords: List[str]
    access_urls: List[str]
    download_urls: List[str]
    spatial_extent: Optional[str] = None # WKT representation of the spatial extent

    @property
    def primary_title(self) -> Optional[str]:
        """
        DCAT Datasets may contain multiple titles in different languages. 
        Returns the first title or None if no titles exist.
        """
        return self.titles[0] if self.titles else None
    
    @property
    def primary_description(self) -> Optional[str]:
        """
        Returns the first title is the dataset contains multiple titles in different languages
        """
        return self.descriptions[0] if self.descriptions else None
    
    @property
    def spatial_extent_wkt(self) -> Optional[str]:
        """
        Returns the spatial extent in WKT format if available or None if not set.
        """
        return self.spatial_extent if self.spatial_extent else None
    
    def to_content(self) -> str:
        """
        Combines the dataset's metadata into a single text chunk. This is useful for embedding and searching.
        """

        content_parts = []
        titles = [t for t in self.titles if t]
        descriptions = [d for d in self.descriptions if d]
        keywords = [k for k in self.keywords if k]
        if titles:
            content_parts.append(f"Title: {'; '.join(titles)}")
        if descriptions:
            content_parts.append(f"Description: {'; '.join(descriptions)}")
        if keywords:
            content_parts.append(f"Keywords: {', '.join(keywords)}")

        return "\n".join(content_parts)

    def to_metadata(self) -> dict:
        """
        Returns the dataset's metadata as a dictionary. This is useful for filtering and searching.
        """
        return {
            "dataset_id": self.dataset_id,
            "title": self.primary_title,
            "description": self.primary_description,
            "keywords": self.keywords,
            "spatial_extent": self.spatial_extent_wkt,
            "access_urls": self.access_urls,
            "download_urls": self.download_urls
        }