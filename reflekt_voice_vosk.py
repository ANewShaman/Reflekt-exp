"""
Reflekt Voice Emotion Module - VOSK + VADER Edition (v2.0)
-----------------------------------------------------------------
INTELLIGENCE UPGRADE:
- Replaced manual dictionary with VADER Sentiment Analysis (7,500+ words).
- Added "Semantic Overrides" to map words like "tired" -> "sad".
- Calculates Valence/Arousal dynamically from sentence context.
"""

import json
import queue
import threading
import time
import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer

# --- NEW: VADER INTELLIGENCE ---
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("!! ERROR: vaderSentiment not installed.")
    print("!! Run: pip install vaderSentiment")
    raise

class ReflektVoiceVOSK:
    def __init__(self, engine=None, model_path=None, sample_rate=16000):
        if model_path is None:
            model_path = self._find_vosk_model()
            if model_path is None:
                model_path = "vosk-model-small-en-us-0.15"

        self.engine = engine
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.running = False
        
        # Initialize text variable (Crucial for bridge)
        self.voice_last_text = ""
        
        self.audio_q = queue.Queue()
        self.last_audio_level = 0.0

        # --- VADER BRAIN ---
        self.analyzer = SentimentIntensityAnalyzer()

        # --- SEMANTIC OVERRIDES (The "Translator") ---
        # Maps specific concepts to the 7 sketch emotions
        self.concept_map = {
            # TIRED / LOW ENERGY -> SAD (Rain)
            "sad": ["tired", "exhausted", "sleepy", "drained", "burned", "fatigue", 
                    "weak", "heavy", "low", "depressed", "lonely", "hurt", "grief", 
                    "pain", "hopeless", "empty", "broken", "cry", "sob", "tears"],
            
            # HIGH ENERGY / NEGATIVE -> ANGRY (Spikes)
            "angry": ["mad", "furious", "hate", "rage", "pissed", "annoyed", 
                      "irritated", "hostile", "destroy", "fight", "shout", "yell"],
            
            # FEAR / ANXIETY -> FEAR (Glitch)
            "fear": ["scared", "afraid", "anxious", "nervous", "terrified", "panic", 
                     "worry", "stress", "shaking", "horror", "nightmare"],
            
            # DISGUST -> DISGUST (Toxic Glitch)
            "disgust": ["gross", "nasty", "sick", "vomit", "repulsive", "yuck"],
            
            # POSITIVE -> HAPPY (Glow)
            "happy": ["good", "great", "love", "joy", "excited", "wonderful", 
                      "best", "smile", "laugh", "happy", "amazing"]
        }

        self._load_model()

    def get_volume(self):
        """Returns normalized volume 0.0 - 1.0 for visual effects."""
        return min(1.0, self.last_audio_level / 500.0)

    def _find_vosk_model(self):
        from pathlib import Path
        current_dir = Path.cwd()
        potential_paths = [
            *list(current_dir.glob("vosk-model*")),
            *list(current_dir.glob("models/vosk-model*")),
        ]
        for path in potential_paths:
            if path.exists() and path.is_dir():
                return str(path)
        return "model" 

    def _load_model(self):
        try:
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)
            print("✓ VOSK model loaded (Hearing enabled)")
            print("✓ VADER analyzer loaded (Understanding enabled)")
        except Exception as e:
            print(f"✗ Failed to load VOSK model: {e}")
            self.model = None

    def start(self):
        if not self.model: return
        self.running = True
        threading.Thread(target=self._audio_capture_thread, daemon=True).start()
        threading.Thread(target=self._process_audio_thread, daemon=True).start()

    def _audio_capture_thread(self):
        def callback(indata, frames, time_info, status):
            if self.running:
                data = np.frombuffer(indata, dtype=np.int16)
                self.last_audio_level = float(np.abs(data).mean())
                self.audio_q.put(bytes(indata))

        with sd.RawInputStream(samplerate=self.sample_rate, blocksize=4000, 
                               dtype="int16", channels=1, callback=callback):
            while self.running:
                time.sleep(0.1)

    def _process_audio_thread(self):
        while self.running:
            try:
                data = self.audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    self.voice_last_text = text
                    print(f" [VOICE] \"{text}\"")
                    self._process_text_with_ai(text)

    def _process_text_with_ai(self, text):
        """
        Uses VADER to get valence, and Keyword Mapping to find the emotion label.
        This forces 'tired' -> 'sad' regardless of the numbers.
        """
        text_lower = text.lower()
        
        # 1. Ask VADER for the emotional score
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']  # -1.0 to +1.0
        
        # Map VADER compound to Valence (-1 to 1)
        valence = compound
        
        # Estimate Arousal based on intensity of the sentiment
        # Extreme valence usually implies high arousal
        arousal = abs(compound)
        
        # 2. Find the Semantic Label (The "Override")
        detected_label = None
        
        # Check our concept map for specific triggers (e.g., "tired")
        for emotion, keywords in self.concept_map.items():
            if any(word in text_lower for word in keywords):
                detected_label = emotion
                break
        
        # 3. Fallback Logic if no keyword matches
        if not detected_label:
            if compound <= -0.05:
                # It's negative, but we don't know which one.
                # Default to SAD (Rain) as it's the safest 'negative' base.
                detected_label = "sad"
            elif compound >= 0.05:
                detected_label = "happy"
            else:
                detected_label = "neutral"

        # 4. Send to Engine
        if self.engine:
            print(f"   -> Analysis: {detected_label.upper()} (Val:{valence:.2f})")
            self.engine.update_voice(valence, arousal, dominant=detected_label)