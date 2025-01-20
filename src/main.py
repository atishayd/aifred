import sys
from PyQt5.QtWidgets import QApplication, QMessageBox
from ui.main_window import MainWindow
from database.db_manager import DatabaseManager
from utils.config import load_config
from utils.logger import logger

def main():
    try:
        # Initialize application
        app = QApplication(sys.argv)
        
        # Load configuration
        config = load_config()
        
        # Initialize database
        db_manager = DatabaseManager()
        db_manager.initialize()
        
        # Create and show main window
        window = MainWindow(db_manager, config)
        window.show()
        
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        QMessageBox.critical(None, "Error",
                           f"Application failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 