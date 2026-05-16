import cv2
import librosa
from moviepy import VideoFileClip
import os

class MediaProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")
        self.video_clip = VideoFileClip(video_path)
        
        # OPTIMIZATION: Keep VideoCapture open in memory instead of opening it 100+ times
        self.cap = cv2.VideoCapture(self.video_path)
        
    def extract_audio(self, output_wav_path="temp_audio.wav"):
        """Extracts audio from the video and saves it as a WAV file."""
        print(f"Extracting audio to {output_wav_path}...")
        self.video_clip.audio.write_audiofile(output_wav_path, logger=None)
        return output_wav_path
        
    def load_audio(self, audio_path, sr=16000):
        """Loads audio file into a numpy array at exactly 16000 Hz for YAMNet."""
        waveform, sample_rate = librosa.load(audio_path, sr=sr)
        return waveform, sample_rate

    def get_frame_sequence(self, start_time_s, end_time_s, max_frames=10):
        """Extracts a sequence of frames between start and end time for temporal analysis."""
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        
        start_frame = int(max(0, start_time_s) * fps)
        end_frame = int(end_time_s * fps)
        
        total_frames = end_frame - start_frame
        if total_frames <= 0: 
            return []
        
        # Calculate step to get exactly max_frames evenly spaced
        step = max(1, total_frames // max_frames)
        
        frames = []
        for f in range(start_frame, end_frame, step):
            if len(frames) >= max_frames:
                break
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
        return frames

    def close(self):
        self.video_clip.close()
        self.cap.release()
