import unittest
import cv2
import numpy as np
from src.database.db_manager import DatabaseManager
from src.utils.audio_processor import AudioProcessor
from datetime import datetime

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.db_manager = DatabaseManager()
        self.db_manager.initialize()
        self.audio_processor = AudioProcessor()
        
    def test_full_student_workflow(self):
        # 1. Register student
        student_name = "Integration Test Student"
        face_encoding = np.random.rand(128)  # Simulate face encoding
        student_result = self.db_manager.add_student(student_name, face_encoding)
        student_id = student_result.inserted_id
        
        # 2. Mark attendance
        attendance_result = self.db_manager.mark_attendance(student_id, datetime.now().date())
        self.assertIsNotNone(attendance_result.inserted_id)
        
        # 3. Log hand raise
        engagement_result = self.db_manager.log_engagement(student_id, hand_raises=1)
        self.assertIsNotNone(engagement_result.inserted_id)
        
        # 4. Log question
        question_result = self.db_manager.log_engagement(student_id, relevant_questions=1)
        self.assertIsNotNone(question_result.inserted_id)
        
        # Verify all data is correctly stored
        student = self.db_manager.get_student(student_id)
        self.assertEqual(student['name'], student_name)
        
        # Check engagement metrics
        engagement = list(self.db_manager.db.engagement.find({'student_id': student_id}))
        total_hand_raises = sum(e.get('hand_raises', 0) for e in engagement)
        total_questions = sum(e.get('relevant_questions', 0) for e in engagement)
        self.assertEqual(total_hand_raises, 1)
        self.assertEqual(total_questions, 1) 