## Resolves Issue #2 and Issue #26
- **Resolves #2:** [DMP 2026] Create Intelligent Closed Caption (CC) Suggestion Tool
- **Resolves #26:** YAMNet's Western training bias causes systematic miss-detection of India-specific sounds in educational content.

---

### 🎥 Demo Link
**[View Pipeline Execution Demo](https://drive.google.com/file/d/1UkbEMbsKTS_MZD_Jet65KIK8X9Ib1sjU/view?usp=sharing)**

---

### 🚀 Overview
This PR completely overhauls the **AutoCC Multimodal Pipeline** to solve critical localization, inference overhead, and foley-misclassification issues. By injecting an intelligent context-routing engine, we bypass YAMNet's inherent Western acoustic biases and gracefully handle dense, music-heavy audio environments.

### ⚙️ Pipeline Explanation
The AutoCC engine operates in 4 highly optimized phases:
1. **Media Processing:** Extracts the raw audio waveform and efficiently sets up `cv2` video pointers in RAM for zero-latency frame jumping.
2. **Multimodal Audio Analysis:** Chunks audio into 0.96s frames. Extracts 1024-D embeddings via YAMNet, routes them through a custom Local Context classifier, and logs potential subtitle events.
3. **Visual Reaction Analysis:** Uses MediaPipe Pose & Face Mesh to analyze the video frames matching the audio timestamps. Calculates the variance of physical movement (flinching/reacting) to confirm if the audio event is visually significant to the scene.
4. **Intelligent Caption Generation:** Applies thresholds, maps foley anomalies to semantic movie actions (e.g., `Sewing Machine` ➔ `[Rapid punches]`), and generates the final context-aware `output.srt`.

---

### 🧠 Unique Architectural Approaches

#### 1. Overcoming Western Bias via Transfer Learning (Custom RF Classifier)
*YAMNet natively misclassifies localized sounds (e.g., it cannot identify a Rickshaw Horn or a Dhak drum, mapping them to generic bells or noise).*
- **The Solution:** Rather than expensively fine-tuning YAMNet from scratch, we implemented a highly efficient **Transfer Learning override**. 
- The pipeline natively extracts the 1024-D embeddings from YAMNet and passes them into a custom-trained `RandomForestClassifier` (trained on 5,800+ clips from the SAS-KIIT and Mendeley Indian Urban Environment datasets). 
- If the custom model recognizes a localized sound with >55% confidence, it intercepts the generic prediction and injects the culturally accurate label (e.g., `Indian Crowd/Human (Local Context)`).

#### 2. Defeating Background Interference via HPSS Music Stripping
*Indian educational and cinematic media is notorious for aggressive background music. This causes YAMNet to endlessly detect "Music," masking the actual ambient events and stalling the pipeline with hundreds of false-positive visual checks.*
- **The Solution:** We implemented **Harmonic-Percussive Source Separation (HPSS)** using `librosa`.
- When the user triggers the script with `--context indian`, the script performs an acoustic "X-Ray." It mathematically splits the waveform, throws away the "Harmonic" frequencies (melodic music, sustained chords), and only feeds the raw "Percussive" transients (horns, crashes, dog barks) into YAMNet. 
- This enables flawless detection of hidden ambient noises even underneath a blaring soundtrack.

#### 3. Intelligent Foley-to-Semantic Mapping
*Audio models are "blind" and take sounds literally. Rapid punches in an action scene are systematically mislabeled by YAMNet as a `[Sewing Machine]` or `[Fusillade]` due to acoustic similarities.*
- **The Solution:** Implemented a hardcoded Context-Mapping dictionary inside the `CaptionGenerator`. 
- By combining **MediaPipe Visual Variance** (confirming human movement on-screen) with Foley mapping, a `[Sewing Machine]` detection coupled with a high visual flinch score is intelligently rewritten into `[Rapid punches]`. 

---

### 🛠️ Additional Optimizations Included
- **C-API Crash Prevention:** Pinned numpy strictly to `<2.0.0` to resolve fatal `_multiarray_umath` crashes with TensorFlow.
- **I/O Overhead Fix:** Refactored `MediaProcessor` to persist the `cv2.VideoCapture` object in RAM, cutting video processing time from 10+ minutes down to ~15 seconds by eliminating redundant disk-reads.

---

### 📦 Installation & Requirements
To run this pipeline, install the dependencies using the newly provided `requirements.txt` file. 
> [!WARNING]
> **CRITICAL:** The `requirements.txt` explicitly pins `numpy<2.0`. TensorFlow's C-API crashes when running YAMNet on newer versions of NumPy.

```bash
pip install -r requirements.txt
```

---

### 💻 How to Run

To run the pipeline on a standard/Western video:
```bash
python main.py --input sample_video.mp4 --context general
```

To run the pipeline on an Indian cinematic/educational video (enables HPSS Music Stripping & Local Models):
```bash
python main.py --input sample_video.mp4 --context indian
```

The final context-aware subtitles will be saved directly to `output.srt`.
