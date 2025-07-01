from typing import List
from rdflib import Graph, Namespace, RDF
from ..models.dataset import Dataset


class RDFParser:
    """Parses RDF files to extract DCAT datasets and their metadata."""

    def __init__(self):
        self.dcat = Namespace("http://www.w3.org/ns/dcat#")
        self.dct = Namespace("http://purl.org/dc/terms/")

    def parse_file(self, file_path: str) -> List[Dataset]:
        """
        Parses RDF files and return a list of Dataset objects
        """

        graph = self._load_graph(file_path)
        return self._extract_datasets(graph)
    
    def _load_graph(self, file_path: str) -> Graph:
        """
        Loads an RDF graph from a file.
        """

        graph = Graph()
        with open(file_path, "r", encoding="utf-8") as f:
            graph.parse(f, format="application/rdf+xml")
        return graph

    def _extract_datasets(self, graph: Graph) -> List[Dataset]:
        """
        Extracts all datasets from the RDF grapg and returns them as a list of Dataset objects.
        """

        datasets = []
        dataset_subjects = list(graph.subjects(RDF.type, self.dcat.Dataset))

        for dataset_uri in dataset_subjects:
            dataset = self._extract_single_dataset(graph, dataset_uri)
            datasets.append(dataset)

        return datasets
    
    def _extract_single_dataset(self, graph: Graph, dataset_uri) -> Dataset:
        """
        Extracts a single dataset from the RDF graph and returns it as a Dataset object.
        """

        titles = [str(title) for title in graph.objects(dataset_uri, self.dct.title)]
        descriptions = [str(description) for description in graph.objects(dataset_uri, self.dct.description)]
        keywords = [str(keyword) for keyword in graph.objects(dataset_uri, self.dcat.keyword)]
        access_urls, download_urls = self._extract_distribution_urls(graph, dataset_uri)

        return Dataset(
            titles=titles,
            descriptions=descriptions,
            keywords=keywords,
            access_urls=access_urls,
            download_urls=download_urls
        )
    
    def _extract_distribution_urls(self, graph: Graph, dataset_uri) -> tuple[List[str], List[str]]:
        """
        Extracts access and download URLs from the dataset's distributions.
        """

        distributions = list(graph.objects(dataset_uri, self.dcat.distribution))
        access_urls = []
        download_urls = []

        for distribution in distributions:
            access_urls.extend([str(url) for url in graph.objects(distribution, self.dcat.accessURL)])
            download_urls.extend([str(url) for url in graph.objects(distribution, self.dcat.downloadURL)])

        return access_urls, download_urls