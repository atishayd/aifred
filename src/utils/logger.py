import logging
import os
from datetime import datetime

class Logger:
    def __init__(self):
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Set up logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/aifred_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('AiFred')
        
    def info(self, message):
        self.logger.info(message)
        
    def error(self, message, exc_info=True):
        self.logger.error(message, exc_info=exc_info)
        
    def warning(self, message):
        self.logger.warning(message)
        
    def debug(self, message):
        self.logger.debug(message)

# Create global logger instance
logger = Logger() 