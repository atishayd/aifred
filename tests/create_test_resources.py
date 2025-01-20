import numpy as np
import wave
import struct

def create_test_audio():
    # Create a simple sine wave
    duration = 1  # seconds
    sample_rate = 44100
    frequency = 440  # Hz
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    samples = (32767 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
    
    # Save as WAV file
    with wave.open('tests/resources/test_question.wav', 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack('h' * len(samples), *samples))

def create_test_images():
    import cv2
    import numpy as np
    
    # Create a sample face image
    face_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    cv2.circle(face_img, (150, 150), 100, (0, 0, 0), 2)
    cv2.circle(face_img, (120, 120), 10, (0, 0, 0), -1)  # Left eye
    cv2.circle(face_img, (180, 120), 10, (0, 0, 0), -1)  # Right eye
    cv2.line(face_img, (130, 180), (170, 180), (0, 0, 0), 2)  # Mouth
    cv2.imwrite('tests/resources/sample_face.jpg', face_img)
    
    # Create a sample hand raise image
    hand_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    cv2.line(hand_img, (150, 150), (150, 50), (0, 0, 0), 2)  # Arm
    cv2.circle(hand_img, (150, 50), 10, (0, 0, 0), -1)  # Hand
    cv2.imwrite('tests/resources/hand_raise.jpg', hand_img)

if __name__ == "__main__":
    create_test_audio()
    create_test_images() 