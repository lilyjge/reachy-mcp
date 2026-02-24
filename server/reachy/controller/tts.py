"""Generate TTS WAV in a separate process so pyttsx3 runs on the main thread.

Tries Groq Orpheus (canopylabs/orpheus-v1-english) first if GROQ_API_KEY is set;
falls back to pyttsx3 if Groq is unavailable or text exceeds 200 characters.
"""
import os
import sys
import pyttsx3
import dotenv
dotenv.load_dotenv()
GROQ_SPEECH_URL = "https://api.groq.com/openai/v1/audio/speech"
GROQ_MODEL = "canopylabs/orpheus-v1-english"
GROQ_VOICE = "autumn"


def _try_groq_tts(text: str, path: str) -> bool:
    """Return True if WAV was written, False to fall back to pyttsx3."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return False
    try:
        import httpx
        r = httpx.post(
            GROQ_SPEECH_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "input": text,
                "voice": GROQ_VOICE,
                "response_format": "wav",
            },
            timeout=30.0,
        )
        if r.is_success and r.content:
            with open(path, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"Groq TTS failed: {e}")
    return False


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: tts.py <text> [output.wav]")
    text = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "output.wav"

    if not _try_groq_tts(text, path):
        print("groq tts failed")
        engine = pyttsx3.init()
        engine.save_to_file(text, path)
        engine.runAndWait()


if __name__ == "__main__":
    main()
