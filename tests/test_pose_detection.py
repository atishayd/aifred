import unittest
import cv2
import mediapipe as mp
import numpy as np

class TestPoseDetection(unittest.TestCase):
    def setUp(self):
        self.pose_detector = PoseDetector()
        # Create a more obvious hand raise pose in the test image
        self.test_image = cv2.imread('tests/resources/hand_raise.jpg')
        if self.test_image is None:
            raise FileNotFoundError("Test image not found")
        
    def test_hand_raise_detection(self):
        # Test with sample image of raised hand
        image = cv2.imread('tests/resources/hand_raise.jpg')
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        self.assertIsNotNone(results.pose_landmarks)
        left_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        
        self.assertLess(left_wrist.y, left_shoulder.y) 