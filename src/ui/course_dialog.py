from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QListWidget, QLabel, QInputDialog, QMessageBox)
from PyQt5.QtCore import Qt

class CourseSelectionDialog(QDialog):
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.selected_course = None
        
        self.setWindowTitle("Select Course")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Course list
        self.course_list = QListWidget()
        self.course_list.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.course_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        create_button = QPushButton("Create New Course")
        create_button.clicked.connect(self.create_course)
        
        select_button = QPushButton("Select Course")
        select_button.clicked.connect(self.accept)
        
        button_layout.addWidget(create_button)
        button_layout.addWidget(select_button)
        layout.addLayout(button_layout)
        
        self.load_courses()
        
    def load_courses(self):
        """Load courses from database"""
        self.course_list.clear()
        courses = self.db_manager.get_all_courses()
        for course in courses:
            item = f"{course['course_code']} - {course['course_name']}"
            self.course_list.addItem(item)
            
    def create_course(self):
        """Create a new course"""
        course_name, ok = QInputDialog.getText(self, "New Course", "Course Name:")
        if ok and course_name:
            course_code, ok = QInputDialog.getText(self, "New Course", "Course Code:")
            if ok and course_code:
                try:
                    self.db_manager.create_course(course_name, course_code)
                    self.load_courses()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to create course: {str(e)}")
                    
    def get_selected_course(self):
        """Get the selected course"""
        current_item = self.course_list.currentItem()
        if current_item:
            course_code = current_item.text().split(' - ')[0]
            return self.db_manager.get_course_by_code(course_code)
        return None 