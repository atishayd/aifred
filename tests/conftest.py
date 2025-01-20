import os
import shutil
import pytest
from datetime import datetime

@pytest.fixture(autouse=True)
def setup_test_environment():
    # Create test directories
    os.makedirs('tests/resources', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Create test .env file with Atlas test database
    with open('.env.test', 'w') as f:
        f.write(f"""
MONGODB_USERNAME=test_user
MONGODB_PASSWORD=test_password
MONGODB_CLUSTER=test-cluster
OPENAI_API_KEY=test_key
""")
    
    yield
    
    # Cleanup
    if os.path.exists('.env.test'):
        os.remove('.env.test')
    if os.path.exists('logs'):
        shutil.rmtree('logs')

@pytest.fixture
def mock_mongodb(mocker):
    """Mock MongoDB connection for testing"""
    mock_client = mocker.patch('pymongo.MongoClient')
    mock_db = mock_client.return_value['aifred']
    return mock_db 