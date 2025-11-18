from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.embeddings import BaseEmbedding
from config.settings import EMBEDDING_MODEL_NAME

class EmbeddingSelector:
    """Selects and loads the specified embedding model (OLLAMA or Sentence Transformer)."""
    
    @staticmethod
    def get_embedding_model(source: str = "ollama") -> BaseEmbedding:
        """
        Loads the embedding model.
        :param source: 'ollama' or 'sentence_transformer'
        :return: A LlamaIndex BaseEmbedding instance.
        """
        if source.lower() == "ollama":
            # Assumes ollama server is running locally
            print(f"Loading Ollama embedding model: {EMBEDDING_MODEL_NAME}")
            return OllamaEmbedding(
                model_name=EMBEDDING_MODEL_NAME,
                base_url="http://localhost:11434"
            )
        elif source.lower() == "sentence_transformer":
            print(f"Loading Sentence Transformer model: {EMBEDDING_MODEL_NAME}")
            # Uses HuggingFaceEmbedding for Sentence Transformers (e.g., 'all-MiniLM-L6-v2')
            return HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        else:
            raise ValueError(f"Unknown embedding source: {source}")