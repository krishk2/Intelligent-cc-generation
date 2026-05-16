import argparse
import json
import os
import time

from src.media_processor import MediaProcessor
from src.audio_analyzer import AudioAnalyzer
from src.visual_analyzer import VisualAnalyzer
from src.caption_generator import CaptionGenerator

def print_header(text):
    print(f"\n{'='*50}")
    print(f" {text}")
    print(f"{'='*50}")

def main(video_path, context):
    start_time = time.time()
    
    # Setup routing flags based on user context
    use_hpss = False
    use_custom_model = False
    
    if context == 'indian':
        use_hpss = True
        use_custom_model = True
        print_header("INDIAN CONTEXT DETECTED: Enabling HPSS and Custom Models")
    else:
        print_header("GENERAL CONTEXT DETECTED: Standard YAMNet processing")
    
    # ---------------------------------------------------------
    # PHASE 1: Media Processing
    # ---------------------------------------------------------
    print_header("PHASE 1: MEDIA PROCESSING")
    mp = MediaProcessor(video_path)
    audio_path = mp.extract_audio()
    waveform, sr = mp.load_audio(audio_path)
    print(f"Loaded audio waveform. Sample rate: {sr} Hz")
    
    # ---------------------------------------------------------
    # PHASE 2: Audio Analysis (YAMNet + Custom Indian Sounds ML)
    # ---------------------------------------------------------
    print_header("PHASE 2: MULTIMODAL AUDIO ANALYSIS")
    aa = AudioAnalyzer()
    
    # Process audio with context-aware routing
    audio_events = aa.process_full_audio(
        waveform, 
        sr, 
        use_custom_model=use_custom_model, 
        use_hpss=use_hpss
    )
    
    # Save intermediate JSON
    with open("phase2_audio_events.json", "w") as f:
        json.dump(audio_events, f, indent=4)
    print(f"Found {len(audio_events)} non-speech audio events. Saved to phase2_audio_events.json")
    
    # ---------------------------------------------------------
    # PHASE 3: Visual Analysis (MediaPipe Reaction Tracking)
    # ---------------------------------------------------------
    print_header("PHASE 3: VISUAL REACTION ANALYSIS")
    va = VisualAnalyzer()
    multimodal_events = []
    
    total_events = len(audio_events)
    for idx, event in enumerate(audio_events, 1):
        # OPTIMIZATION: Skip very weak sounds that aren't worth heavy visual analysis
        if event['events'][0]['confidence'] < 0.3:
            continue
            
        # Print progress on the same line
        print(f"Processing Visuals for Event {idx}/{total_events}...", end="\r")
        
        # OPTIMIZATION: Extract 5 frames instead of 15 (cuts ML workload by 66%)
        frames = mp.get_frame_sequence(event['timestamp'], event['end_timestamp'], max_frames=5)
        
        # Calculate visual reaction variance
        visual_score = va.analyze_sequence_for_reaction(frames)
        
        # Extract best label and confidence from YAMNet output
        best_pred = event['events'][0]
        
        multimodal_event = {
            'timestamp': event['timestamp'],
            'end_timestamp': event['end_timestamp'],
            'label': best_pred['label'],
            'audio_confidence': best_pred['confidence'],
            'visual_significance': visual_score
        }
        multimodal_events.append(multimodal_event)
        
    with open("phase3_multimodal_events.json", "w") as f:
        json.dump(multimodal_events, f, indent=4)
    print("Visual analysis complete. Saved to phase3_multimodal_events.json")
    
    # ---------------------------------------------------------
    # PHASE 4: Decision Engine & Subtitle Generation
    # ---------------------------------------------------------
    print_header("PHASE 4: CAPTION GENERATION")
    cg = CaptionGenerator()
    
    # This will apply thresholds and build output.srt
    output_srt = "output.srt"
    cg.filter_and_generate(multimodal_events, output_srt)
    
    # Cleanup
    mp.close()
    if os.path.exists(audio_path):
        os.remove(audio_path)
        
    elapsed = time.time() - start_time
    print_header(f"PIPELINE COMPLETE ({elapsed:.1f}s)")
    print(f"Subtitles generated at: {output_srt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoCC: Multimodal Video Captioning")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--context", type=str, choices=['general', 'indian'], default='general', 
                        help="Select 'indian' to enable HPSS music stripping and localized ML models.")
    args = parser.parse_args()
    
    main(args.input, args.context)
