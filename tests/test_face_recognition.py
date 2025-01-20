import unittest
import cv2
import face_recognition
import numpy as np
from src.database.db_manager import DatabaseManager

class TestFaceRecognition(unittest.TestCase):
    def setUp(self):
        self.db_manager = DatabaseManager()
        self.db_manager.initialize()
        
    def test_face_detection(self):
        # Test with sample image
        image = cv2.imread('tests/resources/sample_face.jpg')
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        self.assertEqual(len(face_locations), 1)
        
    def test_face_encoding(self):
        # Test face encoding generation
        image = cv2.imread('tests/resources/sample_face.jpg')
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_image)
        self.assertEqual(len(face_encodings), 1)
        self.assertEqual(len(face_encodings[0]), 128) 