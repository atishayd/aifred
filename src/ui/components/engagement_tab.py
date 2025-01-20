from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QTableWidget, QTableWidgetItem, QProgressBar)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import openai
from utils.audio_processor import AudioProcessor

class EngagementTab(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.camera = None
        self.timer = None
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.hand_raise_threshold = 0.3  # Threshold for hand raise detection
        self.is_recording_audio = False
        self.audio_processor = AudioProcessor()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Camera preview for hand raise detection
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        layout.addWidget(self.camera_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.camera_button = QPushButton("Start Camera")
        self.camera_button.clicked.connect(self.toggle_camera)
        controls_layout.addWidget(self.camera_button)
        
        self.audio_button = QPushButton("Start Recording")
        self.audio_button.clicked.connect(self.toggle_audio_recording)
        controls_layout.addWidget(self.audio_button)
        
        layout.addLayout(controls_layout)
        
        # Engagement metrics
        metrics_layout = QHBoxLayout()
        
        # Hand raises counter
        hand_raises_layout = QVBoxLayout()
        self.hand_raises_label = QLabel("Hand Raises: 0")
        hand_raises_layout.addWidget(self.hand_raises_label)
        self.hand_raises_progress = QProgressBar()
        hand_raises_layout.addWidget(self.hand_raises_progress)
        metrics_layout.addLayout(hand_raises_layout)
        
        # Questions counter
        questions_layout = QVBoxLayout()
        self.questions_label = QLabel("Relevant Questions: 0")
        questions_layout.addWidget(self.questions_label)
        self.questions_progress = QProgressBar()
        questions_layout.addWidget(self.questions_progress)
        metrics_layout.addLayout(questions_layout)
        
        layout.addLayout(metrics_layout)
        
        # Engagement history table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Student", "Time", "Hand Raises", "Questions"])
        layout.addWidget(self.table)
        
    def toggle_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)
            self.camera_button.setText("Stop Camera")
        else:
            self.timer.stop()
            self.camera.release()
            self.camera = None
            self.camera_button.setText("Start Camera")
            
    def update_frame(self):
        if self.camera is None:
            return
            
        ret, frame = self.camera.read()
        if ret:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Check for hand raise
                self.detect_hand_raise(results.pose_landmarks)
                
                # Draw pose landmarks
                self.draw_pose_landmarks(frame, results.pose_landmarks)
            
            # Convert to Qt format for display
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image))
            
    def detect_hand_raise(self, landmarks):
        # Get relevant landmarks
        left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Check if either hand is raised above shoulder
        if (left_wrist.y < left_shoulder.y - self.hand_raise_threshold or 
            right_wrist.y < right_shoulder.y - self.hand_raise_threshold):
            self.log_hand_raise()
            
    def draw_pose_landmarks(self, frame, landmarks):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )
        
    def toggle_audio_recording(self):
        if not self.is_recording_audio:
            self.audio_processor.start_recording()
            self.audio_button.setText("Stop Recording")
        else:
            audio_file = self.audio_processor.stop_recording()
            if audio_file:
                transcript, is_relevant = self.audio_processor.analyze_question(audio_file)
                # Update database with question analysis
                self.db_manager.log_engagement(
                    student_id="default_student",  # TODO: Identify student
                    relevant_questions=1 if is_relevant else 0
                )
                self.update_engagement_table()
            self.audio_button.setText("Start Recording")
        self.is_recording_audio = not self.is_recording_audio
        
    def log_hand_raise(self):
        # TODO: Implement face recognition to identify student
        # For now, log for a default student
        self.db_manager.log_engagement(
            student_id="default_student",
            hand_raises=1
        )
        self.update_engagement_table()
        
    def update_engagement_table(self):
        # Clear existing rows
        self.table.setRowCount(0)
        
        # Get today's engagement records
        today = datetime.now().date()
        engagement = self.db_manager.db.engagement.find({
            'timestamp': {
                '$gte': datetime.combine(today, datetime.min.time()),
                '$lte': datetime.combine(today, datetime.max.time())
            }
        }).sort('timestamp', -1)
        
        for record in engagement:
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            student = self.db_manager.get_student(record['student_id'])
            self.table.setItem(row, 0, QTableWidgetItem(student['name']))
            self.table.setItem(row, 1, QTableWidgetItem(record['timestamp'].strftime('%H:%M:%S')))
            self.table.setItem(row, 2, QTableWidgetItem(str(record['hand_raises'])))
            self.table.setItem(row, 3, QTableWidgetItem(str(record['relevant_questions']))) 