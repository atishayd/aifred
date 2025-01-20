from pymongo import MongoClient
from dotenv import load_dotenv
import os
import certifi

def test_mongodb_connection():
    try:
        # Load environment variables
        load_dotenv()
        
        # Get credentials
        username = os.getenv('MONGODB_USERNAME')
        password = os.getenv('MONGODB_PASSWORD')
        cluster = os.getenv('MONGODB_CLUSTER')
        
        # Construct connection string
        uri = f"mongodb+srv://{username}:{password}@{cluster}/?retryWrites=true&w=majority&appName=aifred"
        
        print("Attempting to connect to MongoDB Atlas...")
        print(f"Username: {username}")
        print(f"Cluster: {cluster}")
        
        # Try to connect with SSL certificate verification
        client = MongoClient(
            uri,
            tlsCAFile=certifi.where()
        )
        
        # Test connection by executing a command
        result = client.admin.command('ping')
        
        if result.get('ok') == 1.0:
            print("✅ Successfully connected to MongoDB Atlas!")
            
            # List available databases
            print("\nAvailable databases:")
            for db_name in client.list_database_names():
                print(f"- {db_name}")
                
            # Create and test aifred database
            db = client['aifred']
            print("\nTesting aifred database...")
            db.test_collection.insert_one({"test": "connection"})
            print("✅ Successfully wrote to aifred database!")
            
            # Cleanup test data
            db.test_collection.delete_one({"test": "connection"})
            
        else:
            print("❌ Connection test failed!")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    test_mongodb_connection() 