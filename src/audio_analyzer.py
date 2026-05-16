import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import os
import librosa
try:
    import joblib
except ImportError:
    joblib = None

class AudioAnalyzer:
    def __init__(self):
        print("Loading YAMNet model from TensorFlow Hub...")
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.class_map_path = self.model.class_map_path().numpy()
        self.labels = self.load_class_map(self.class_map_path)
        
        # Filter out speech, ambient noise, and continuous music/singing (reduces overhead drastically)
        self.ignore_keywords = [
            'Speech', 'Narration', 'Silence', 'Inside, small room', 
            'Outside, rural or natural', 'Noise', 'Environmental noise',
            'Music', 'Singing', 'Humming', 'Lullaby', 'Vocal music', 
            'A capella', 'Chant', 'Mantra', 'Bird'
        ]
        
        # Attempt to load custom Indian sounds classifier
        self.custom_clf = None
        if joblib:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'indian_sounds_model.pkl')
            if os.path.exists(model_path):
                print("Loading custom Indian Sounds classifier...")
                self.custom_clf = joblib.load(model_path)
            else:
                print(f"Custom model not found at {model_path}. Using base YAMNet only.")

    def load_class_map(self, csv_path):
        labels = []
        with tf.io.gfile.GFile(csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                labels.append(row['display_name'])
        return labels

    def is_speech_or_music(self, label):
        for keyword in self.ignore_keywords:
            if keyword.lower() in label.lower():
                return True
        return False

    def process_full_audio(self, waveform, sample_rate, use_custom_model=True, use_hpss=False):
        """
        Runs YAMNet over the entire waveform. 
        YAMNet natively processes in fast 0.96s chunks.
        """
        print("Analyzing audio track with YAMNet...")
        
        if use_hpss:
            print("Applying Harmonic-Percussive Source Separation (HPSS) to strip background music...")
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Keep only the percussive elements (hits, noise, speech) and discard harmonic (music)
                _, waveform = librosa.effects.hpss(waveform)
                
        # YAMNet requires exactly 16000 Hz float32 waveform
        waveform = waveform.astype(np.float32)
        
        scores, embeddings, spectrogram = self.model(waveform)
        scores_np = scores.numpy() # Shape: (N, 521)
        
        events_timeline = []
        
        # YAMNet processes audio in 0.96s frames.
        frame_duration = 0.96 
        
        for i in range(len(scores_np)):
            frame_scores = scores_np[i]
            # Get top 5 predictions for this chunk
            top_indices = np.argsort(frame_scores)[::-1][:5]
            
            results = []
            for idx in top_indices:
                prob = float(frame_scores[idx])
                label = self.labels[idx]
                
                # --- Custom Transfer Learning Override ---
                if use_custom_model and self.custom_clf and prob > 0.1:
                    # Pass this 0.96s frame's embedding to our custom model
                    chunk_embedding = embeddings[i].numpy().reshape(1, -1)
                    custom_label = self.custom_clf.predict(chunk_embedding)[0]
                    custom_prob = np.max(self.custom_clf.predict_proba(chunk_embedding))
                    
                    # If the custom model is highly confident, override YAMNet's generic label
                    if custom_prob >= 0.55:
                        label = f"{custom_label} (Local Context)"
                        prob = custom_prob  # Use the custom model's confidence
                # -----------------------------------------
                
                if not self.is_speech_or_music(label):
                    results.append({"label": label, "confidence": prob})
                    if len(results) >= 3:
                        break
            
            timestamp = i * frame_duration
            
            # If we have a significant non-speech event, log it
            if results and results[0]['confidence'] > 0.1:
                events_timeline.append({
                    "timestamp": timestamp,
                    "end_timestamp": timestamp + frame_duration,
                    "events": results
                })
                
        return events_timeline
