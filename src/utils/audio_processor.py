import pyaudio
import wave
import openai
import numpy as np
from datetime import datetime
import os

class AudioProcessor:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.recording = False
        
    def start_recording(self):
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self.audio_callback
        )
        self.recording = True
        self.frames = []
        self.stream.start_stream()
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.recording:
            self.frames.append(np.frombuffer(in_data, dtype=np.float32))
        return (in_data, pyaudio.paContinue)
        
    def stop_recording(self):
        if self.stream:
            self.recording = False
            self.stream.stop_stream()
            self.stream.close()
            
            # Save audio file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recordings/question_{timestamp}.wav"
            os.makedirs("recordings", exist_ok=True)
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paFloat32))
                wf.setframerate(44100)
                wf.writeframes(b''.join(self.frames))
                
            return filename
        return None
        
    def analyze_question(self, audio_file):
        """Analyze question using OpenAI APIs"""
        try:
            # Use the new OpenAI client for API v1
            client = openai.OpenAI()
            
            # Transcribe with Whisper
            with open(audio_file, "rb") as audio:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio
                )
            
            # Analyze with GPT
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an educational assistant analyzing student questions."},
                    {"role": "user", "content": f"Is this question relevant to the class? Question: {transcript.text}"}
                ]
            )
            
            is_relevant = "yes" in response.choices[0].message.content.lower()
            return transcript.text, is_relevant
            
        except Exception as e:
            logger.error(f"Error analyzing question: {str(e)}")
            raise 