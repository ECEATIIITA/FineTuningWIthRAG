import pytest
from src.db.atlas_connector import AtlasConnector
from src.config.settings import DB_NAME, COLLECTION_NAME, MONGO_URI # Assume minimal test settings

@pytest.fixture(scope="module")
def db_connector():
    # Requires a running/accessible Atlas DB for this test to pass
    try:
        connector = AtlasConnector()
        yield connector
    finally:
        connector.close_connection()

def test_db_connection_success(db_connector):
    """Test if the connector successfully initializes the client."""
    assert db_connector.client is not None

def test_db_name_access(db_connector):
    """Test if the correct database is accessed."""
    assert db_connector.db.name == DB_NAME

# Add more tests for:
# - Embedding selection (e.g., check model type)
# - File processing (e.g., check node count after splitting)
# - Queue functionality (e.g., check put/get)