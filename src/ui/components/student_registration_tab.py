from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QLineEdit, QTableWidget, QTableWidgetItem, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import face_recognition
import numpy as np
from datetime import datetime

class StudentRegistrationTab(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.camera = None
        self.timer = None
        self.current_student_name = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Student info input
        info_layout = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter student name")
        info_layout.addWidget(self.name_input)
        
        self.register_button = QPushButton("Start Registration")
        self.register_button.clicked.connect(self.start_registration)
        info_layout.addWidget(self.register_button)
        
        layout.addLayout(info_layout)
        
        # Camera preview
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        layout.addWidget(self.camera_label)
        
        # Registered students table
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Name", "Registration Date"])
        layout.addWidget(self.table)
        
        self.update_students_table()
        
    def start_registration(self):
        if not self.name_input.text():
            return
            
        if self.camera is None:
            self.current_student_name = self.name_input.text()
            self.camera = cv2.VideoCapture(0)
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)
            self.register_button.setText("Capture")
        else:
            self.capture_and_register()
            
    def capture_and_register(self):
        ret, frame = self.camera.read()
        if ret:
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if len(face_locations) == 1:
                face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                self.db_manager.add_student(self.current_student_name, face_encoding)
                self.update_students_table()
                
            self.cleanup_camera()
            
    def cleanup_camera(self):
        if self.timer:
            self.timer.stop()
        if self.camera:
            self.camera.release()
        self.camera = None
        self.current_student_name = None
        self.register_button.setText("Start Registration")
        self.name_input.clear() 

    def update_frame(self):
        if self.camera is None:
            return
        
        ret, frame = self.camera.read()
        if ret:
            # Convert frame to RGB for face detection
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Draw rectangles around detected faces
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Convert to Qt format for display
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image))

    def update_students_table(self):
        """Update the table of registered students"""
        self.table.setRowCount(0)
        
        # Get all students from database
        students = self.db_manager.db.students.find().sort('created_at', -1)
        
        for student in students:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(student['name']))
            self.table.setItem(row, 1, QTableWidgetItem(
                student.get('created_at', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
            )) 

    def registerStudent(self, student_data):
        # ... existing code around registration ...
        if not was_registration_successful:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Registration Failed")
            msg_box.setText("Registration failed. Please try again.")
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setStandardButtons(QMessageBox.Ok)

            # Connect the QMessageBox button to our new method
            msg_box.buttonClicked.connect(self._onRegistrationFailureAcknowledged)
            msg_box.exec_()
        else:
            # ... existing code for successful registration ...
            pass
        # ... possibly more existing code ...

    def _onRegistrationFailureAcknowledged(self, clicked_button):
        """
        Reset the UI back to the countdown stage so the user can try again.
        """
        # Call the appropriate method to restart the countdown stage
        self.startCountdown()  # Or whatever method sets the tab into 'countdown' mode.

    def startCountdown(self):
        # ... existing code that shows the countdown / face detection stage ...
        pass
    # ... existing code ... 