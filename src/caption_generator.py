import datetime

class CaptionGenerator:
    def __init__(self, audio_threshold=0.1, visual_threshold=0.3):
        self.audio_threshold = audio_threshold
        self.visual_threshold = visual_threshold
        
        # Hardcoded Foley-to-Semantic Action Mapping
        self.foley_map = {
            "Sewing machine": "Rapid punches",
            "Fusillade": "Rapid punches",
            "Gears": "Scuffle / Grappling",
            "Thump, thud": "Body hits floor",
            "Patter": "Footsteps / Scuffle",
            "Smash, crash": "Smashing hit",
            "Boom": "Heavy strike",
            "Explosion": "Heavy strike"
        }

    def format_timestamp(self, seconds):
        """Converts seconds to SRT timestamp format (HH:MM:SS,mmm)"""
        td = datetime.timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds_int = divmod(remainder, 60)
        milliseconds = int(td.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds_int:02d},{milliseconds:03d}"

    def generate_caption_text(self, label):
        """Formats the label into a standard CC format."""
        # Check if the raw label exists in our intelligent foley dictionary
        mapped_label = self.foley_map.get(label, label)
        
        # Simple formatting: lowercase and put in brackets
        formatted = mapped_label.lower().replace("_", " ")
        return f"[{formatted}]"

    def filter_and_generate(self, multimodal_events, output_srt="output.srt"):
        """
        Takes a list of events with both audio and visual scores,
        filters them, and generates an SRT file.
        """
        print(f"Generating captions to {output_srt}...\n")
        print(f"{'Time (s)':<12} | {'Label':<25} | {'Audio':<6} | {'Visual':<6} | {'Status'}")
        print("-" * 75)
        
        significant_captions = []
        
        for event in multimodal_events:
            audio_conf = event['audio_confidence']
            visual_conf = event['visual_significance']
            label = event['label']
            time_str = f"{event['timestamp']:.1f}"
            
            # The Core Decision Logic
            if audio_conf >= self.audio_threshold and visual_conf >= self.visual_threshold:
                caption_text = self.generate_caption_text(label)
                significant_captions.append({
                    "start": event['timestamp'],
                    "end": event['end_timestamp'],
                    "text": caption_text
                })
                status = "[✔] ACCEPTED"
            else:
                status = "[X] REJECTED"
                
            print(f"{time_str:<12} | {label:<25} | {audio_conf:<6.2f} | {visual_conf:<6.2f} | {status}")
                
        # Write to SRT
        with open(output_srt, "w") as f:
            for i, cap in enumerate(significant_captions, 1):
                start_str = self.format_timestamp(cap['start'])
                end_str = self.format_timestamp(cap['end'])
                
                f.write(f"{i}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"{cap['text']}\n\n")
                
        print(f"Successfully generated {len(significant_captions)} captions.")
        return output_srt
