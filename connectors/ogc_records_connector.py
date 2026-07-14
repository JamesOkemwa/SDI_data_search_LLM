import logging
from typing import Any, Dict, List, Optional, Tuple

from owslib.ogcapi.records import Records
from owslib.util import Authentication
from shapely.geometry import box, shape

from connectors.base_connector import BaseMetadataConnector
from models.dataset import Dataset

logger = logging.getLogger(__name__)

# Link relations that point to an accessible representation of the resource
# (landing pages, service descriptions) rather than a raw data download.
ACCESS_RELS = {"about", "canonical", "service", "service-desc", "describedby", "self", "collection", "alternate"}
# Link relations that point directly at downloadable data.
DOWNLOAD_RELS = {"data", "download", "enclosure"}


class OGCRecordsConnector(BaseMetadataConnector):
    """
    Fetches datasets from an OGC API - Records endpoint
    (https://docs.ogc.org/is/20-004r1/20-004r1.html) using owslib's Records client.

    Paginates through the `items` resource of one or more record collections via
    offset/limit, converting each GeoJSON record Feature into a Dataset.
    """

    def __init__(self, base_url: str, collection_id: Optional[str] = None,
                 page_size: int = 50, timeout: int = 30, verify_ssl: bool = True):
        self.base_url = base_url.rstrip("/")
        self.collection_id = collection_id
        self.page_size = page_size
        self.timeout = timeout
        self.verify_ssl = verify_ssl

    def fetch_datasets(self) -> List[Dataset]:
        try:
            client = Records(
                self.base_url,
                timeout=self.timeout,
                auth=Authentication(verify=self.verify_ssl),
            )
        except Exception as e:
            logger.error(f"OGCRecordsConnector: failed to connect to {self.base_url}: {e}")
            return []

        collection_ids = [self.collection_id] if self.collection_id else self._list_collections(client)
        if not collection_ids:
            logger.error(f"OGCRecordsConnector: no record collections found at {self.base_url}")
            return []

        datasets: List[Dataset] = []
        for collection_id in collection_ids:
            datasets.extend(self._fetch_collection(client, collection_id))

        logger.info(f"OGCRecordsConnector: finished with {len(datasets)} datasets from {self.base_url}")
        return datasets

    def _list_collections(self, client: Records) -> List[str]:
        """Discover every record collection id exposed by the server."""
        try:
            collections = client.collections()
            collection_items = collections.get("collections", []) if isinstance(collections, dict) else collections

            ids: List[str] = []
            for collection in collection_items:
                if isinstance(collection, dict):
                    collection_id = collection.get("id")
                else:
                    collection_id = collection

                if collection_id:
                    ids.append(str(collection_id))

            logger.info(f"OGCRecordsConnector: discovered {len(ids)} record collection(s) at {self.base_url}")
            return ids
        except Exception as e:
            logger.error(f"OGCRecordsConnector: failed to list collections at {self.base_url}: {e}")
            return []

    def _fetch_collection(self, client: Records, collection_id: str) -> List[Dataset]:
        """Paginate through a single collection's items and convert each to a Dataset."""
        datasets: List[Dataset] = []
        offset = 0

        while True:
            try:
                payload = client.collection_items(
                    collection_id, limit=self.page_size, offset=offset  # type: ignore[arg-type]
                )
            except Exception as e:
                logger.error(
                    f"OGCRecordsConnector: failed to fetch items for '{collection_id}' "
                    f"at offset {offset}: {e}"
                )
                break

            for feature in payload.get("features", []):
                dataset = self._record_to_dataset(feature)
                if dataset:
                    datasets.append(dataset)

            number_matched = payload.get("numberMatched")
            number_returned = payload.get("numberReturned", len(payload.get("features", [])))

            logger.info(
                f"OGCRecordsConnector: fetched {len(datasets)}/{number_matched if number_matched is not None else '?'} "
                f"datasets from collection '{collection_id}'"
            )

            if not number_returned:
                break
            offset += number_returned
            if number_matched is not None and offset >= number_matched:
                break

        return datasets

    def _record_to_dataset(self, feature: Dict[str, Any]) -> Optional[Dataset]:
        """Convert a single GeoJSON record Feature into a Dataset object."""
        try:
            properties = feature.get("properties") or {}
            dataset_id = feature.get("id") or properties.get("identifier")
            if not dataset_id:
                logger.warning("OGCRecordsConnector: skipping record without an id")
                return None

            title = properties.get("title") or ""
            description = properties.get("description") or properties.get("abstract") or ""
            keywords = self._normalize_values(properties.get("keywords"))

            access_urls, download_urls = self._extract_urls(feature.get("links") or [])
            spatial_extent = self._extract_spatial_extent(feature.get("geometry"), feature.get("bbox"))

            return Dataset(
                dataset_id=str(dataset_id),
                titles=[title],
                descriptions=[description],
                keywords=keywords,
                access_urls=access_urls,
                download_urls=download_urls,
                spatial_extent=spatial_extent,
            )
        except Exception as e:
            logger.warning(f"OGCRecordsConnector: error converting record to Dataset: {e}")
            return None

    def _extract_urls(self, links: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """Splits record links into access URLs and download URLs based on their `rel`."""
        access_urls: List[str] = []
        download_urls: List[str] = []
        seen_access = set()
        seen_download = set()

        for link in links:
            href = link.get("href", "")
            rel = (link.get("rel") or "").lower()
            if not href.startswith(("http://", "https://")):
                continue  # skip non-web links (e.g. mqtt broker channels)

            if rel in DOWNLOAD_RELS:
                if href not in seen_download:
                    download_urls.append(href)
                    seen_download.add(href)
            elif rel in ACCESS_RELS:
                if href not in seen_access:
                    access_urls.append(href)
                    seen_access.add(href)

        return access_urls, download_urls

    def _normalize_values(self, value: Any) -> List[str]:
        """Normalizes scalar or sequence metadata into a list of strings."""
        if not value:
            return []

        if isinstance(value, str):
            return [value]

        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if item]

        return [str(value)]

    def _extract_spatial_extent(
        self,
        geometry: Optional[Dict[str, Any]],
        bbox: Optional[Any],
    ) -> Optional[str]:
        """Converts a record geometry or bbox to a WKT string."""
        if geometry:
            try:
                return shape(geometry).wkt
            except Exception as e:
                logger.warning(f"OGCRecordsConnector: could not convert geometry to WKT: {e}")

        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                minx, miny, maxx, maxy = map(float, bbox[:4])
                return box(minx, miny, maxx, maxy).wkt
            except (TypeError, ValueError) as e:
                logger.warning(f"OGCRecordsConnector: could not convert bbox to WKT: {e}")

        return None
