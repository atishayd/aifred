import unittest
import os
import numpy as np
from src.utils.audio_processor import AudioProcessor

class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.audio_processor = AudioProcessor()
        self.test_audio_file = "tests/resources/test_question.wav"
        
    def test_audio_recording(self):
        # Start recording
        self.audio_processor.start_recording()
        self.assertTrue(self.audio_processor.recording)
        
        # Generate some test audio data
        test_frames = [np.random.rand(1024).astype(np.float32) for _ in range(10)]
        for frame in test_frames:
            self.audio_processor.audio_callback(frame.tobytes(), 1024, None, None)
            
        # Stop recording and verify file creation
        output_file = self.audio_processor.stop_recording()
        self.assertIsNotNone(output_file)
        self.assertTrue(os.path.exists(output_file))
        
        # Cleanup
        os.remove(output_file)
        
    def test_question_analysis(self):
        transcript, is_relevant = self.audio_processor.analyze_question(self.test_audio_file)
        self.assertIsInstance(transcript, str)
        self.assertIsInstance(is_relevant, bool) 