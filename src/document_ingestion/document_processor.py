"""Document processing module for loading and splitting documents."""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
)


class DocumentProcessor:
    """Loads documents from URLs and local files, then splits them into chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_from_url(self, url: str) -> List[Document]:
        """Load documents from a web page."""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_file(self, file_path: Path) -> List[Document]:
        """Load a document based on its file type."""

        if file_path.suffix.lower() == ".pdf":
            return PyPDFLoader(str(file_path)).load()

        elif file_path.suffix.lower() == ".txt":
            return TextLoader(str(file_path), encoding="utf-8").load()

        return []

    def load_documents( self,urls: List[str],data_folder: str,) -> List[Document]:
        """
        Load documents from URLs and all supported files in the data folder.
        """
        documents: List[Document] = []

        # Load documents from URLs
        for url in urls:
            documents.extend(self.load_from_url(url))

        # Load documents from data folder
        folder = Path(data_folder)

        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {data_folder}")

        for file in folder.iterdir():
            if file.is_file():
                documents.extend(self.load_from_file(file))

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        return self.splitter.split_documents(documents)

    def process_documents(
        self,
        urls: List[str],
        data_folder: str,
    ) -> List[Document]:
        """
        Complete document processing pipeline.
        """
        documents = self.load_documents(urls, data_folder)
        return self.split_documents(documents)