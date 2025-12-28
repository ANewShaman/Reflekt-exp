"""
Reflekt Main – INTEGRATED SYSTEM (VOSK + VADER + FER)
-----------------------------------------------------
Launches:
1. Emotion Engine (FER Face Analysis + Fusion Logic)
2. Camera Feeder (Eyes)
3. Voice Engine (VOSK Hearing + VADER Intelligence)
4. WebSocket Bridge (Nervous System)
"""

import threading
import time
import cv2
import sys

from reflekt_emotion_live import AsyncReflektEmotionEngine
from reflekt_voice_vosk import ReflektVoiceVOSK
from bridge_server import start_bridge

# ------------------------------------------------------------
# Camera Feeder (The Eyes)
# ------------------------------------------------------------
def run_camera_feeder(engine, camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"!! ERROR: Could not open camera {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"✓ Camera {camera_index} active (feeding emotion engine)")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        # Feed the frame to the AI engine
        engine.submit_frame(frame)
        time.sleep(0.033) # Cap at ~30 FPS

# ------------------------------------------------------------
# Main Launcher
# ------------------------------------------------------------
def launch(mode="full"):
    print("\n🔮 Starting Reflekt (Hybrid Intelligence System)…\n")

    # 1. Start Emotion Engine (The Brain)
    # Note: fusion_weights are now dynamically handled in the class logic,
    # but we pass a config just in case.
    engine = AsyncReflektEmotionEngine(config={
        "fusion_weights": {"face": 0.2, "voice": 0.8}
    })

    # 2. Start Camera Feeder (The Eyes)
    cam_thread = threading.Thread(
        target=run_camera_feeder, 
        args=(engine,), 
        daemon=True
    )
    cam_thread.start()

    # 3. Start Voice Engine (The Ears & Understanding)
    voice_engine = None
    if mode == "full":
        try:
            voice_engine = ReflektVoiceVOSK(engine=engine)
            # Start voice listening in background
            voice_thread = threading.Thread(target=voice_engine.start, daemon=True)
            voice_thread.start()
            print("✓ Voice engine listening (VOSK + VADER Active)")
        except Exception as e:
            print(f"⚠ Voice engine failed: {e}")
            print("  (Did you install dependencies? 'pip install vosk vaderSentiment')")
    else:
        print("• Demo mode — voice disabled (run 'python reflekt_main.py full' to enable)")

    # 4. Start WebSocket Bridge (The Mouth)
    print("✓ WebSocket bridge starting at ws://localhost:8765")
    start_bridge(engine, voice_engine)

if __name__ == "__main__":
    # Default to 'full' mode if no arg provided
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    launch(mode)