from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QGridLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

class StudentDetailsDialog(QDialog):
    def __init__(self, student, parent=None):
        super().__init__(parent)
        self.student = student
        
        self.setWindowTitle(f"Student Details - {student['name']}")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # Student photo
        photo = QImage.fromData(student['photo'])
        photo_label = QLabel()
        photo_label.setPixmap(QPixmap.fromImage(photo).scaled(
            240, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        photo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(photo_label)
        
        # Student details
        details_grid = QGridLayout()
        details_grid.addWidget(QLabel("Name:"), 0, 0)
        details_grid.addWidget(QLabel(student['name']), 0, 1)
        details_grid.addWidget(QLabel("Student ID:"), 1, 0)
        details_grid.addWidget(QLabel(str(student['student_id'])), 1, 1)
        details_grid.addWidget(QLabel("Registered:"), 2, 0)
        details_grid.addWidget(QLabel(student['created_at'].strftime("%Y-%m-%d")), 2, 1)
        
        layout.addLayout(details_grid)
        
        # Attendance summary
        attendance_label = QLabel("Recent Attendance")
        attendance_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(attendance_label)
        
        # Add attendance history here
        
        # Engagement summary
        engagement_label = QLabel("Engagement Summary")
        engagement_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(engagement_label)
        
        # Add engagement metrics here
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button) 