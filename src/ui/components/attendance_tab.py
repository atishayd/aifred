from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt, QTimer
import cv2
import face_recognition
import numpy as np

class AttendanceTab(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.camera = None
        self.timer = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Camera preview
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        layout.addWidget(self.camera_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.toggle_camera)
        controls_layout.addWidget(self.start_button)
        
        self.capture_button = QPushButton("Mark Attendance")
        self.capture_button.clicked.connect(self.mark_attendance)
        self.capture_button.setEnabled(False)
        controls_layout.addWidget(self.capture_button)
        
        layout.addLayout(controls_layout)
        
        # Attendance table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Student", "Time", "Status"])
        layout.addWidget(self.table)
        
    def toggle_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)
            self.start_button.setText("Stop Camera")
            self.capture_button.setEnabled(True)
        else:
            self.timer.stop()
            self.camera.release()
            self.camera = None
            self.start_button.setText("Start Camera")
            self.capture_button.setEnabled(False)
            
    def update_frame(self):
        if self.camera is None:
            return
            
        ret, frame = self.camera.read()
        if ret:
            # Convert frame to RGB for face_recognition
            rgb_frame = frame[:, :, ::-1]
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Draw rectangles around faces
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Convert to Qt format for display
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image))
            
    def mark_attendance(self):
        if self.camera is None:
            return
            
        ret, frame = self.camera.read()
        if ret:
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for encoding in face_encodings:
                # Find matching student in database
                student = self.find_matching_student(encoding)
                if student:
                    # Mark attendance
                    self.db_manager.mark_attendance(student['_id'])
                    self.update_attendance_table()
                    
    def find_matching_student(self, encoding):
        # Get all students from database
        students = self.db_manager.db.students.find()
        for student in students:
            stored_encoding = np.frombuffer(student['face_embedding'])
            if face_recognition.compare_faces([stored_encoding], encoding)[0]:
                return student
        return None
        
    def update_attendance_table(self):
        # Clear existing rows
        self.table.setRowCount(0)
        
        # Get today's attendance
        today = datetime.now().date()
        attendance = self.db_manager.db.attendance.find({
            'date': today.isoformat()
        }).sort('timestamp', -1)
        
        for record in attendance:
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            student = self.db_manager.get_student(record['student_id'])
            self.table.setItem(row, 0, QTableWidgetItem(student['name']))
            self.table.setItem(row, 1, QTableWidgetItem(record['timestamp'].strftime('%H:%M:%S')))
            self.table.setItem(row, 2, QTableWidgetItem(record['status'])) 