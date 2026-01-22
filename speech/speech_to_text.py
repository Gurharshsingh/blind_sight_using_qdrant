import json
import queue
import os
import sys
import sounddevice as sd
from vosk import Model, KaldiRecognizer


MODEL_PATH = "vosk-model/vosk-model-small-en-us-0.15" 
SAMPLE_RATE = 16000
DURATION = 5  # Seconds to record

def get_model():
    
    
    print(f"✅ Loading model from '{MODEL_PATH}'...")
    return Model(MODEL_PATH)

# Initialize model safely
model = get_model()

def speech_to_text(duration=DURATION):
    q = queue.Queue()

    def callback(indata, frames, time, status):
        
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    try:
        
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            print("🎤 Speak now...")
            rec = KaldiRecognizer(model, SAMPLE_RATE)
            
            
            num_blocks = int(duration * SAMPLE_RATE / 8000)

            for _ in range(num_blocks):
                data = q.get()
                if rec.AcceptWaveform(data):
                    
                    pass

            
            final_json = rec.FinalResult()
            result = json.loads(final_json)
            
            text = result.get("text", "").strip()
            print(f"  Recognized: {text}")
            return text

    except Exception as e:
        print(f" Audio error: {e}")
        return ""

if __name__ == "__main__":
    speech_to_text()