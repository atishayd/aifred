import unittest
import numpy as np
from datetime import datetime
from src.database.db_manager import DatabaseManager

class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        self.db_manager = DatabaseManager()
        self.db_manager.initialize()
        # Clean up any existing test data
        self.db_manager.db.students.delete_many({})
        self.db_manager.db.attendance.delete_many({})
        self.db_manager.db.engagement.delete_many({})

    def tearDown(self):
        # Clean up after tests
        if hasattr(self, 'db_manager'):
            self.db_manager.db.students.delete_many({})
            self.db_manager.db.attendance.delete_many({})
            self.db_manager.db.engagement.delete_many({})
            self.db_manager.client.close()
        
    def test_add_student(self):
        # Create test student
        name = "Test Student"
        face_encoding = np.random.rand(128)  # Simulate face encoding
        
        result = self.db_manager.add_student(name, face_encoding)
        self.assertIsNotNone(result.inserted_id)
        
        # Verify student was added
        student = self.db_manager.get_student(result.inserted_id)
        self.assertEqual(student['name'], name)
        
    def test_mark_attendance(self):
        # Add test student first
        student_id = self.db_manager.add_student("Test Student", np.random.rand(128)).inserted_id
        
        # Mark attendance
        date = datetime.now().date()
        result = self.db_manager.mark_attendance(student_id, date)
        self.assertIsNotNone(result.inserted_id)
        
    def test_log_engagement(self):
        # Add test student first
        student_id = self.db_manager.add_student("Test Student", np.random.rand(128)).inserted_id
        
        # Log engagement
        result = self.db_manager.log_engagement(student_id, hand_raises=1, relevant_questions=1)
        self.assertIsNotNone(result.inserted_id) 