from pathlib import Path
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader # LlamaIndex for easy PDF loading
from llama_index.core.schema import Document

class FileProcessor:
    """Uploads and splits files/PDFs into text fragments (Nodes)."""
    
    def __init__(self, chunk_size=1024, chunk_overlap=20):
        self.parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_document(self, file_path: str) -> list[Document]:
        """Loads a PDF or other supported file type into LlamaIndex Documents."""
        file = Path(file_path)
        print(f"Loading document: {file.name}")
        
        # Simple file type detection, can be extended
        if file.suffix.lower() == ".pdf":
            loader = PDFReader()
        else:
            # Add other loaders or default to SimpleDirectoryReader (requires install)
            raise ValueError(f"Unsupported file type: {file.suffix}")
            
        return loader.load_data(file)

    def split_document(self, documents: list[Document]):
        """Splits the loaded documents into Nodes (text fragments)."""
        print(f"Splitting {len(documents)} document(s) into nodes.")
        return self.parser.get_nodes_from_documents(documents, show_progress=True)