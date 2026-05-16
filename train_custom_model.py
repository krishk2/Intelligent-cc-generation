import os
import ast
import csv
import librosa
import joblib
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("Loading YAMNet model from TensorFlow Hub...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Paths
AUTO_CC_DIR = r"c:\Users\z005a42u\Documents\AutoCC"
MENDELEY_DIR = os.path.join(AUTO_CC_DIR, "Multi label Audiotory Dataset from Diverse Indian Urban Environments")
INDIAN_SOUNDS_DIR = os.path.join(AUTO_CC_DIR, "Indian_sounds_dataset")

# Label Mapping for Mendeley
MENDELEY_MAP = {
    'VHCL': 'Indian Urban Vehicle',
    'HUMN': 'Indian Crowd/Human',
    'BIRD': 'Indian Urban Bird',
    'MAML': 'Street Mammal',
    'BELL': 'Bicycle Bell'
}

X = []
y = []

def extract_embedding(file_path):
    try:
        # YAMNet requires exactly 16000 Hz float32 waveform
        waveform, sr = librosa.load(file_path, sr=16000, mono=True)
        waveform = waveform.astype(np.float32)
        
        scores, embeddings, spectrogram = yamnet_model(waveform)
        # Average the embeddings across the frames of the clip to get a single vector per file
        mean_embedding = np.mean(embeddings.numpy(), axis=0)
        return mean_embedding
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

print("\n--- Phase 1: Processing Mendeley Dataset ---")
csv_path = os.path.join(MENDELEY_DIR, 'clip_labels.csv')
clips_dir = os.path.join(MENDELEY_DIR, 'clips')

if os.path.exists(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            if not row: continue
            file_name = row[0]
            # row[1] looks like "['VHCL']" or "['VHCL', 'HUMN']"
            labels_str = row[1]
            try:
                # Safely parse the string to a python list
                raw_labels = ast.literal_eval(labels_str)
                
                # We will pick the first valid label (excluding silence)
                valid_label = None
                for raw_lbl in raw_labels:
                    if raw_lbl in MENDELEY_MAP:
                        valid_label = MENDELEY_MAP[raw_lbl]
                        break
                        
                if valid_label:
                    file_path = os.path.join(clips_dir, file_name)
                    if os.path.exists(file_path):
                        embedding = extract_embedding(file_path)
                        if embedding is not None:
                            X.append(embedding)
                            y.append(valid_label)
            except:
                pass

print(f"Extracted {len(X)} samples so far.")

print("\n--- Phase 2: Processing Indian Sounds Dataset ---")
if os.path.exists(INDIAN_SOUNDS_DIR):
    for file_name in os.listdir(INDIAN_SOUNDS_DIR):
        if file_name.endswith('.wav'):
            # Example: 19_Rickshaw_Horn_1.wav
            parts = file_name.split('_')
            # Extract everything between the class ID and the sample number
            if len(parts) >= 3:
                class_name = " ".join(parts[1:-1])
                
                file_path = os.path.join(INDIAN_SOUNDS_DIR, file_name)
                embedding = extract_embedding(file_path)
                if embedding is not None:
                    X.append(embedding)
                    y.append(class_name)

print(f"Total samples extracted: {len(X)}")

if len(X) == 0:
    print("No samples found! Exiting.")
    exit()

print("\n--- Phase 3: Training Classifier ---")
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\nModel Performance:")
print(classification_report(y_test, y_pred))

# Save the model
model_path = os.path.join(AUTO_CC_DIR, 'indian_sounds_model.pkl')
joblib.dump(clf, model_path)
print(f"\nSuccess! Model saved to: {model_path}")
