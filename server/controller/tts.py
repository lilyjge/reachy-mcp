"""Generate TTS WAV in a separate process so pyttsx3 runs on the main thread.

pyttsx3 on Windows often deadlocks when runAndWait() is called from a
background thread. This script is meant to be run via subprocess.
"""
import sys
import pyttsx3

def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: tts_generate.py <text> [output.wav]")
    text = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "output.wav"
    engine = pyttsx3.init()
    engine.save_to_file(text, path)
    engine.runAndWait()

if __name__ == "__main__":
    main()
