"""
Reflekt Emotion Detection Engine - FER Implementation (v2.2 - Voice Priority)
-------------------------------------------------------------------
- FIX: Convert OpenCV BGR -> RGB before FER detection (critical).
- LOGIC UPGRADE: Voice input now overrides Face input for 8 seconds.
- LOGIC UPGRADE: Semantic labels from voice (e.g. "sad") overwrite visual labels.
"""

from __future__ import annotations

import cv2
import numpy as np
import time
import json
import queue
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
from collections import deque

# ---- Import FER with friendly error handling and fallbacks
FER_CLASS = None
try:
    from fer import FER as _FER
    FER_CLASS = _FER
except Exception:
    try:
        from fer.fer import FER as _FER
        FER_CLASS = _FER
    except Exception:
        print("FER not found or failed to import.")
        print("    Install with:  pip install fer")
        raise

# ============================================================================ #
# NUMPY TYPE CONVERSION HELPER
# ============================================================================ #

def convert_to_native(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj

# ============================================================================ #
# 1) DATA STRUCTURES
# ============================================================================ #

@dataclass
class EmotionFrame:
    """Standardized emotion packet (one analyzed moment)."""
    timestamp: float
    frame_number: int
    dominant: str
    confidence: float
    valence: float
    arousal: float
    all_emotions: Dict[str, float]
    quality: str  # 'high' | 'medium' | 'low' | 'uncertain' | 'no_face'
    processing_time_ms: float
    age: Optional[int] = None
    vibrancy: Optional[float] = None
    source_modality: str = "face" # Track if 'face' or 'voice' is in charge

    def to_json(self) -> str:
        data = asdict(self)
        data = convert_to_native(data)
        return json.dumps(data, ensure_ascii=False)

    def to_dict(self) -> dict:
        data = asdict(self)
        return convert_to_native(data)


@dataclass
class SessionMetadata:
    """Session-level metadata."""
    session_id: str
    start_time: float
    end_time: Optional[float]
    total_frames: int
    frames_analyzed: int
    average_fps: float
    config: dict
    emotion_summary: Dict[str, int]


# ============================================================================ #
# 2) ASYNC EMOTION ENGINE (Non-blocking with FER)
# ============================================================================ #

class AsyncReflektEmotionEngine:
    """
    Async emotion engine using FER library.
    Includes Smart Fusion Logic to prioritize Voice commands.
    """

    EMOTION_MAPS = {
        "valence": {
            "angry": -0.7, "disgust": -0.6, "fear": -0.8,
            "happy": 0.9, "sad": -0.8, "surprise": 0.4, "neutral": 0.0
        },
        "arousal": {
            "angry": 0.8, "disgust": 0.3, "fear": 0.9,
            "happy": 0.7, "sad": -0.5, "surprise": 0.9, "neutral": 0.0
        }
    }

    def __init__(self, config: dict | None = None):
        # Config defaults
        self.config = {
            "frame_skip": 8,               
            "smoothing_window": 4,         
            "min_confidence": 0.30,        
            "use_smoothed_output": True,   
            "serve_api": False,            
            "api_port": 5055,              
            # NOTE: These base weights are modified dynamically by the fuser
            "fusion_weights": {"face": 0.2, "voice": 0.8}, 
            "mtcnn": True,                 
        }
        if config:
            self.config.update(config)

        # State
        self.frame_count: int = 0
        self.analyzed_count: int = 0
        self.session_log: List[EmotionFrame] = []
        self.emotion_history: deque[EmotionFrame] = deque(
            maxlen=self.config["smoothing_window"]
        )
        self.last_valid_emotion: Optional[EmotionFrame] = None
        self.latest_frame: Optional[EmotionFrame] = None

        # Voice state - Stores the emotion + Timestamp of when it was said
        self.voice_last: Optional[Dict] = None

        # Performance tracking
        self.performance_metrics = {
            "total_processing_time": 0.0,
            "frames_processed": 0,
            "fps_history": deque(maxlen=30),
            "queue_drops": 0,
        }

        # Session
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time: float = time.time()

        # Async processing queues
        self.frame_queue: queue.Queue = queue.Queue(maxsize=3)
        self.result_queue: queue.Queue = queue.Queue(maxsize=5)
        
        # Initialize FER detector
        self.detector: Optional[FER_CLASS] = None
        self._init_detector()
        
        # Worker thread
        self._worker_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        self._warmup()

    def _init_detector(self):
        """Initialize FER detector with error handling and MTCNN availability check."""
        mtcnn_requested = bool(self.config.get("mtcnn", False))
        mtcnn_ok = False
        if mtcnn_requested:
            try:
                import mtcnn  # noqa: F401
                mtcnn_ok = True
            except Exception:
                print("! MTCNN requested but not installed. Falling back to OpenCV mode.")
                mtcnn_ok = False

        try:
            self.detector = FER_CLASS(mtcnn=mtcnn_requested and mtcnn_ok)
            self.config["mtcnn"] = (mtcnn_requested and mtcnn_ok)
            print(f"✓ FER detector initialized (MTCNN: {self.config['mtcnn']})")
        except Exception as e:
            print(f"✗ Failed to initialize FER: {e}")
            self.detector = FER_CLASS(mtcnn=False)
            self.config["mtcnn"] = False
            print("✓ FER detector initialized (OpenCV mode)")

    def _warmup(self):
        def warmup_task():
            try:
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.circle(dummy, (320, 240), 80, (255, 255, 255), -1)
                if self.detector:
                    rgb = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
                    _ = self.detector.detect_emotions(rgb)
            except Exception:
                pass
        threading.Thread(target=warmup_task, daemon=True).start()

    # --------------- Background Worker Thread ---------------- #

    def _worker_loop(self):
        while self._worker_running:
            try:
                frame_bgr = self.frame_queue.get(timeout=0.15)
                if frame_bgr is None:  # Shutdown signal
                    break
                
                result = self._analyze_frame_blocking(frame_bgr)
                if result:
                    try:
                        self.result_queue.put_nowait(result)
                    except queue.Full:
                        try:
                            _ = self.result_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.result_queue.put_nowait(result)
                        except queue.Full:
                            pass
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
                continue

    def _pick_face(self, results: List[dict]) -> Optional[dict]:
        if not results:
            return None
        best = None
        best_score = -1.0
        for r in results:
            emos = r.get("emotions", {})
            if not emos:
                continue
            dom, score = max(emos.items(), key=lambda x: float(x[1]))
            if score > best_score:
                best = r
                best_score = float(score)
        return best

    def _analyze_frame_blocking(self, frame_bgr: np.ndarray) -> Optional[EmotionFrame]:
        if not self.detector:
            return self._handle_detection_failure()
            
        t0 = time.time()
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.detector.detect_emotions(frame_rgb)
            if not results:
                return self._handle_detection_failure()
            
            face_data = self._pick_face(results)
            if not face_data:
                return self._handle_detection_failure()

            emotions = face_data.get('emotions', {})
            if not emotions:
                return self._handle_detection_failure()

            dominant, conf01 = max(emotions.items(), key=lambda x: float(x[1]))
            confidence = float(conf01) * 100.0 

            all_emotions = {k: round(float(v) * 100.0, 2) for k, v in emotions.items()}
            valence, arousal = self.compute_valence_arousal(all_emotions)
            vibrancy = round(arousal ** 0.8, 3)
            quality = self._assess_quality(confidence)

            ef = EmotionFrame(
                timestamp=time.time(),
                frame_number=self.frame_count,
                dominant=dominant,
                confidence=round(confidence, 2),
                valence=valence,
                arousal=arousal,
                all_emotions=all_emotions,
                quality=quality,
                processing_time_ms=round((time.time() - t0) * 1000.0, 2),
                age=None,
                vibrancy=vibrancy,
            )

            if confidence < (self.config["min_confidence"] * 100.0) and ef.quality == "medium":
                ef.quality = "low"

            self.analyzed_count += 1
            self.emotion_history.append(ef)
            ef = self._maybe_smooth(ef)
            
            # --- CRITICAL: FUSE WITH VOICE ---
            ef = self._fuse_modalities(ef)

            self.last_valid_emotion = ef
            self.latest_frame = ef
            self.performance_metrics["total_processing_time"] += (time.time() - t0) * 1000.0
            self.performance_metrics["frames_processed"] += 1

            return ef

        except Exception:
            return self._handle_detection_failure()

    # --------------- Public API ---------------- #

    def submit_frame(self, frame_bgr: np.ndarray) -> bool:
        try:
            self.frame_queue.put_nowait(frame_bgr.copy())
            return True
        except queue.Full:
            self.performance_metrics["queue_drops"] += 1
            return False

    def get_latest_result(self) -> Optional[EmotionFrame]:
        result = None
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
            except queue.Empty:
                break
        return result

    # --------------- Voice Fusion Logic ---------------- #

    def update_voice(self, valence: float, arousal: float, dominant: str | None = None):
        """
        Update voice emotion state with a TIMESTAMP.
        This allows the system to know if the voice command is fresh.
        """
        v = max(-1.0, min(1.0, float(valence)))
        a = max(0.0, min(1.0, float(arousal)))
        
        self.voice_last = {
            "valence": v, 
            "arousal": a, 
            "dominant": dominant,
            "timestamp": time.time()  # Track WHEN this was said
        }

    def _fuse_modalities(self, frame: EmotionFrame) -> EmotionFrame:
        """
        Intelligent Fusion:
        If voice data is fresh (< 8 seconds old), it OVERRIDES the face.
        This fixes the 'Tired vs Neutral' problem.
        """
        if not self.voice_last:
            return frame
        
        # Check how old the voice data is
        age = time.time() - self.voice_last.get("timestamp", 0)
        
        # TIMEOUT: Only use voice if it happened in the last 8.0 seconds
        if age > 8.0:
            return frame 
            
        # If voice is fresh, use HEAVY voice weights
        # We practically ignore the face to ensure the 'Tired' command goes through
        wf = 0.1  # Face weight (weak)
        wv = 0.9  # Voice weight (strong)
        
        fused_v = round(wf * frame.valence + wv * self.voice_last["valence"], 3)
        fused_a = round(wf * frame.arousal + wv * self.voice_last["arousal"], 3)
        
        frame.valence, frame.arousal = fused_v, fused_a
        frame.vibrancy = round(fused_a ** 0.8, 3)
        
        # --- LABEL OVERRIDE ---
        # If voice has a specific label (like "sad" for "tired"), FORCE IT.
        voice_dom = self.voice_last.get("dominant")
        if voice_dom and voice_dom != "neutral":
            frame.dominant = voice_dom
            frame.source_modality = "voice" # Debug flag
            
        return frame

    # --------------- Helper Methods ---------------- #

    def compute_valence_arousal(self, emotions: Dict[str, float]) -> Tuple[float, float]:
        v = sum(
            (float(emotions.get(e, 0.0)) / 100.0) * self.EMOTION_MAPS["valence"][e]
            for e in self.EMOTION_MAPS["valence"]
        )
        a = sum(
            (float(emotions.get(e, 0.0)) / 100.0) * self.EMOTION_MAPS["arousal"][e]
            for e in self.EMOTION_MAPS["arousal"]
        )
        v = max(-1.0, min(1.0, v))
        a = max(0.0, min(1.0, a))
        return round(v, 3), round(a, 3)

    def _assess_quality(self, confidence_percent: float) -> str:
        if confidence_percent > 70: return "high"
        elif confidence_percent > 40: return "medium"
        elif confidence_percent > 20: return "low"
        else: return "uncertain"

    def _maybe_smooth(self, frame: EmotionFrame) -> EmotionFrame:
        if not self.config["use_smoothed_output"] or len(self.emotion_history) < 2:
            return frame
        avg_v = float(np.mean([e.valence for e in self.emotion_history]))
        avg_a = float(np.mean([e.arousal for e in self.emotion_history]))
        frame.valence = round(avg_v, 3)
        frame.arousal = round(avg_a, 3)
        frame.vibrancy = round(frame.arousal ** 0.8, 3)
        return frame

    def _handle_detection_failure(self) -> Optional[EmotionFrame]:
        if self.last_valid_emotion:
            prev = self.last_valid_emotion
            return EmotionFrame(
                timestamp=time.time(),
                frame_number=self.frame_count,
                dominant=prev.dominant,
                confidence=0.0,
                valence=prev.valence,
                arousal=prev.arousal,
                all_emotions=prev.all_emotions,
                quality="no_face",
                processing_time_ms=0.0,
                age=None,
                vibrancy=prev.vibrancy,
            )
        return None

    # --------------- Metrics / Export ---------------- #

    def log_frame(self, ef: EmotionFrame):
        self.session_log.append(ef)

    def get_latest_result(self) -> Optional[EmotionFrame]:
        result = None
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
            except queue.Empty:
                break
        return result

    def get_performance_metrics(self) -> Dict:
        fp = self.performance_metrics["frames_processed"]
        if fp == 0: return {"status": "no_data"}
        avg_ms = self.performance_metrics["total_processing_time"] / fp
        return {
            "frames_analyzed": self.analyzed_count,
            "total_frames": self.frame_count,
            "avg_processing_ms": round(avg_ms, 2),
            "queue_drops": self.performance_metrics["queue_drops"],
        }

    def export_session(self, filepath: str | None = None, format: str = "json") -> str:
        out_dir = Path("reflekt_sessions")
        out_dir.mkdir(exist_ok=True)
        if filepath is None:
            filepath = out_dir / f"session_{self.session_id}.{format}"
        else:
            filepath = Path(filepath)

        counts: Dict[str, int] = {}
        for f in self.session_log:
            counts[f.dominant] = counts.get(f.dominant, 0) + 1

        elapsed = max(1e-6, (time.time() - self.start_time))
        metadata = SessionMetadata(
            session_id=self.session_id,
            start_time=self.start_time,
            end_time=time.time(),
            total_frames=self.frame_count,
            frames_analyzed=self.analyzed_count,
            average_fps=self.analyzed_count / elapsed,
            config=self.config,
            emotion_summary=counts,
        )

        if format == "json":
            export_data = {
                "metadata": convert_to_native(asdict(metadata)),
                "frames": [f.to_dict() for f in self.session_log],
            }
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(export_data, fh, indent=2, ensure_ascii=False)
        return str(filepath)

    def shutdown(self):
        self._worker_running = False
        try:
            self.frame_queue.put(None, timeout=0.5)
        except Exception:
            pass
        try:
            self.worker_thread.join(timeout=2.0)
        except Exception:
            pass