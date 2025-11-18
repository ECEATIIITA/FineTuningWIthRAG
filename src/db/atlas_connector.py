from pymongo import MongoClient
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from config.settings import MONGO_URI, DB_NAME, COLLECTION_NAME

class AtlasConnector:
    """Handles connection to MongoDB Atlas and provides the Vector Store."""
    
    def __init__(self):
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            print("Successfully connected to MongoDB Atlas.")
        except Exception as e:
            print(f"Error connecting to MongoDB Atlas: {e}")
            raise

    def get_vector_store(self, index_name: str, embed_model: BaseEmbedding) -> MongoDBAtlasVectorSearch:
        """Returns the LlamaIndex MongoDBAtlasVectorSearch instance."""
        return MongoDBAtlasVectorSearch(
            mongodb_client=self.client,
            db_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            index_name=index_name  # The name of the Atlas Vector Search index
        )

    def close_connection(self):
        """Closes the MongoDB connection."""
        self.client.close()