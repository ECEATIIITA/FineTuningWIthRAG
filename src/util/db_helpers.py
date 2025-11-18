from pymongo import MongoClient
from config.settings import MONGO_URI, DB_NAME, COLLECTION_NAME

class AtlasIndexHelper:
    """Helper to programmatically create the Atlas Vector Search index."""
    
    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]

    def create_vector_index(self, index_name: str, embedding_dim: int, field_name: str = "embedding"):
        """
        Creates a MongoDB Atlas Vector Search index configuration.
        Requires the user to run the aggregation stage to apply the change.
        """
        index_definition = {
            "createSearchIndex": COLLECTION_NAME,
            "name": index_name,
            "definition": {
                "mappings": {
                    "dynamic": True,
                    "fields": [
                        {
                            "type": "vector",
                            "path": field_name,
                            "numDimensions": embedding_dim,
                            "similarity": "cosine" # or "euclidean", "dotProduct"
                        }
                    ]
                }
            }
        }
        
        try:
            # The command is run against the database
            self.db.command(index_definition)
            print(f"Successfully sent command to create index '{index_name}' on collection '{COLLECTION_NAME}'.")
            print("Verify creation in the Atlas UI.")
        except Exception as e:
            print(f"Error creating index: {e}")
            raise
            
    def __del__(self):
        self.client.close()