# src/rag_core/query_engine.py

from llama_index.core import VectorStoreIndex, QueryEngine
from llama_index.llms.ollama import Ollama
from db.atlas_connector import AtlasConnector
from embeddings.embedding_selector import EmbeddingSelector
from config.settings import OLLAMA_MODEL_NAME

class RAGQueryEngine:
    """The main RAG query class using LlamaIndex components."""
    
    def __init__(self, index_name: str):
        # 1. Get components
        embed_model = EmbeddingSelector.get_embedding_model(source="ollama")
        db_connector = AtlasConnector()
        
        # 2. Get Vector Store
        vector_store = db_connector.get_vector_store(index_name, embed_model)
        
        # 3. Create Index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )
        
        # 4. Create Query Engine
        # OLLAMA as the LLM
        ollama_llm = Ollama(model=OLLAMA_MODEL_NAME, request_timeout=120.0)
        self.query_engine: QueryEngine = self.index.as_query_engine(llm=ollama_llm)

    def query(self, prompt: str) -> str:
        """Executes a RAG query."""
        print(f"Processing query: {prompt[:30]}...")
        response = self.query_engine.query(prompt)
        return str(response)