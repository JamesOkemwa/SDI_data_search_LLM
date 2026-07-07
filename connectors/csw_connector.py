import logging
from typing import List, Optional, Tuple

from owslib.csw import CatalogueServiceWeb
from owslib.ows import ExceptionReport

from connectors.base_connector import BaseMetadataConnector
from models.dataset import Dataset

logger = logging.getLogger(__name__)


class CSWConnector(BaseMetadataConnector):
    """
    Fetches datasets from an OGC CSW endpoint using Dublin Core output schema.

    Paginates through all available records up to max_records
    """

    def __init__(self, url: str, page_size: int = 50, timeout: int = 30):
        self.url = url
        self.page_size = page_size
        self.timeout = timeout

    def fetch_datasets(self) -> List[Dataset]:
        try:
            csw = CatalogueServiceWeb(self.url, timeout=self.timeout)  # type: ignore[operator]
        except Exception as e:
            logger.error(f"CSWConnector: failed to connect to {self.url}: {e}")
            return []

        datasets: List[Dataset] = []
        start_position = 1
        matched_total: Optional[int] = None

        while True:
            results, records, skipped = self._fetch_with_fallback(
                csw, start_position, self.page_size, matched_total
            )

            for _, record in records.items():
                dataset = self._record_to_dataset(record)
                if dataset:
                    datasets.append(dataset)

            if skipped:
                logger.warning(
                    f"CSWConnector: skipped {len(skipped)} problematic record(s) "
                    f"at position(s) {skipped}"
                )

            matched = results.get("matches", 0)
            nextrecord = results.get("nextrecord", 0)

            if matched and matched_total is None:
                matched_total = matched

            logger.info(
                f"CSWConnector: fetched {len(datasets)} datasets "
                f"(position {start_position}, matched={matched})"
            )

            if not nextrecord or nextrecord == 0:
                break
            start_position = nextrecord

        logger.info(f"CSWConnector: finished with {len(datasets)} datasets from {self.url}")
        return datasets

    def _fetch_page(self, csw, startposition: int, maxrecords: int, esn: str = "full") -> Tuple[dict, dict]:
        csw.getrecords2(
            startposition=startposition,
            maxrecords=maxrecords,
            esn=esn,
        )
        return csw.results, csw.records

    def _fetch_with_fallback(
        self,
        csw,
        startposition: int,
        maxrecords: int,
        matched_total: Optional[int],
    ) -> Tuple[dict, dict, List[int]]:
        try:
            results, records = self._fetch_page(csw, startposition, maxrecords, esn="full")
            return results, records, []
        except ExceptionReport as err:
            logger.warning(
                f"CSWConnector: full-page fetch failed at position {startposition}: {err}. "
                "Falling back to single-record fetch for this window."
            )

        recovered: dict = {}
        skipped: List[int] = []

        for pos in range(startposition, startposition + maxrecords):
            try:
                _, single = self._fetch_page(csw, pos, 1, esn="full")
                recovered.update(single)
            except ExceptionReport:
                skipped.append(pos)

        try:
            summary_results, _ = self._fetch_page(csw, startposition, maxrecords, esn="summary")
            return summary_results, recovered, skipped
        except ExceptionReport:
            fallback_results = {
                "matches": matched_total or 0,
                "returned": len(recovered),
                "nextrecord": startposition + maxrecords,
            }
            return fallback_results, recovered, skipped
    
    def _record_to_dataset(self, record) -> Optional[Dataset]:
        """Convert a CSW record into a Dataset object"""
        
        try:
            dataset_id = getattr(record, "identifier", None) or str(id(record))
            
            title = getattr(record, "title", None) or ""
            abstract = getattr(record, "abstract", None) or ""
            subjects = list(getattr(record, "subjects", None) or [])
            
            access_urls, download_urls = self._extract_urls(record)
            spatial_extent = self._extract_spatial_extent(record)
            
            return Dataset(
                dataset_id=dataset_id,
                titles=[title],
                descriptions=[abstract],
                keywords=subjects,
                access_urls=access_urls,
                download_urls=download_urls,
                spatial_extent=spatial_extent
            )
        except Exception as e:
            logger.warning(f"CSWConnector: error converting record to Dataset: {e}")
            return None
        
    def _extract_urls(self, record) -> tuple[List[str], List[str]]:
        """
        Splits record references into access URLs and download URLs
        """
        access_urls: List[str] = []
        download_urls: List[str] = []
        
        for ref in getattr(record, "references", None) or []:
            url = ref.get("url", "")
            scheme = ref.get("scheme", "")
            if not url:
                continue
            if "download" in (scheme or "").lower() or "WWW:DOWNLOAD" in (scheme or ""):
                download_urls.append(url)
            else:
                access_urls.append(url)

        # Also check record.uris (ISO records)
        for uri in getattr(record, "uris", None) or []:
            url = uri.get("url", "")
            protocol = uri.get("protocol", "")
            if not url:
                continue
            if "download" in (protocol or "").lower():
                download_urls.append(url)
            else:
                access_urls.append(url)

        return access_urls, download_urls
    
    def _extract_spatial_extent(self, record) -> Optional[str]:
        """Converts the record bbox (if present) to a WKT polygon string"""
        bbox = getattr(record, "bbox", None)
        if bbox is None:
            return None
        try:
            minx = float(bbox.minx)
            miny = float(bbox.miny)
            maxx = float(bbox.maxx)
            maxy = float(bbox.maxy)
            return (
                f"POLYGON(({minx} {miny}, {maxx} {miny}, "
                f"{maxx} {maxy}, {minx} {maxy}, {minx} {miny}))"
            )
        except (AttributeError, ValueError, TypeError) as e:
            logger.warning(f"CSWConnector: could not convert bbox to WKT: {e}")
            return None