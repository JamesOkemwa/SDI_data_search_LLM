from typing import List
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from models.dataset import Dataset


class QdrantVectorStoreManager:
    """Manages Qdrant vector store operations."""

    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "dcat_colllection"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = None
        self.vector_store = None
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    def initialize(self) -> None:
        """
        Initialiaze Qdrant client and vector store.
        """
        self.client = QdrantClient(host=self.host, port=self.port)
        self._ensure_collection_exists()

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def add_datasets(self, datasets: List[Dataset]) -> None:
        """
        Adds datasets to the vector store.
        """
        if not self.vector_store:
            raise ValueError("Vector store is not initialized. Call initialize() first.")
        
        documents = self._datasets_to_documents(datasets)
        self.vector_store.add_documents(documents)

    def _ensure_collection_exists(self) -> None:
        """
        Create a Qdrant collection if it does not exist.
        """
        if not self.client.collection_exists(collection_name=self.collection_name):
            test_embedding = self.embeddings.embed_query("test") # example embedding to determine size
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "size": len(test_embedding),
                    "distance": "Cosine",
                },
            )

    def _datasets_to_documents(self, datasets: List[Dataset]) -> List[Document]:
        """
        Converts a list of Dataset objects to a list of langchain Document objects.
        """
        documents = []
        for dataset in datasets:
            document = Document(
                page_content=dataset.to_content(),
                metadata=dataset.to_metadata()
            )
            documents.append(document)
        return documents