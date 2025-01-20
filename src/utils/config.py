import os
from dotenv import load_dotenv
import json

def load_config():
    """Load configuration from .env and config.json"""
    load_dotenv()
    
    # Load default config
    config = {
        "mongodb_uri": os.getenv("MONGODB_URI", "mongodb://localhost:27017/"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "app_settings": {
            "hand_raise_threshold": 0.3,
            "face_recognition_tolerance": 0.6,
            "question_relevance_threshold": 0.7
        }
    }
    
    # Override with local config if exists
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            local_config = json.load(f)
            config.update(local_config)
            
    return config 