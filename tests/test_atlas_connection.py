import unittest
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi

class TestAtlasConnection(unittest.TestCase):
    def setUp(self):
        load_dotenv()
        self.username = os.getenv('MONGODB_USERNAME')
        self.password = os.getenv('MONGODB_PASSWORD')
        self.cluster = os.getenv('MONGODB_CLUSTER')
        self.uri = f"mongodb+srv://{self.username}:{self.password}@{self.cluster}/?retryWrites=true&w=majority&appName=aifred"
        
    def test_connection(self):
        client = None
        try:
            client = MongoClient(
                self.uri,
                tlsCAFile=certifi.where()
            )
            db = client['aifred']
            result = db.command('ping')
            self.assertEqual(result.get('ok'), 1.0)
        except Exception as e:
            self.fail(f"Connection test failed: {str(e)}")
        finally:
            if client:
                client.close()

    def tearDown(self):
        if hasattr(self, 'client'):
            self.client.close() 