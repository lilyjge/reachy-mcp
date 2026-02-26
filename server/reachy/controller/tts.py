"""Generate TTS WAV in a separate process so pyttsx3 runs on the main thread.

Tries Groq Orpheus (canopylabs/orpheus-v1-english) first if GROQ_API_KEY is set;
optionally uses ElevenLabs if TTS_ENGINE=elevenlabs is configured; falls back to
pyttsx3 if network TTS is unavailable.
"""
import os
import sys
import pyttsx3
import dotenv
dotenv.load_dotenv()
GROQ_SPEECH_URL = "https://api.groq.com/openai/v1/audio/speech"
GROQ_MODEL = "canopylabs/orpheus-v1-english"
GROQ_VOICE = "autumn"
ELEVENLABS_MODEL= "eleven_flash_v2_5" or os.environ.get("ELEVENLABS_MODEL")
ELEVENLABS_VOICE_ID= "M4zkunnpRihDKTNF0D7f"

def _tts_voice() -> str:
    """Return the configured TTS voice (for Groq Orpheus or ElevenLabs)."""
    return os.environ.get("TTS_VOICE", GROQ_VOICE)

def _tts_voice_id() -> str:
    """Return the configured TTS voice ID (for ElevenLabs)."""
    return os.environ.get("ELEVENLABS_VOICE_ID", ELEVENLABS_VOICE_ID)

def _try_groq_tts(text: str, path: str) -> bool:
    """Return True if WAV was written, False to fall back to pyttsx3."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return False
    voice = _tts_voice()
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
                "voice": voice,
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


def _try_elevenlabs_tts(text: str, path: str) -> bool:
    """Return True if WAV was written via ElevenLabs, False to fall back."""
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        return False
    # Voice can be provided as ELEVENLABS_VOICE_ID or generic TTS_VOICE.
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID") or _tts_voice_id()
    if not voice_id:
        return False
    model_id = os.environ.get("ELEVENLABS_MODEL", ELEVENLABS_MODEL)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    try:
        import httpx
        r = httpx.post(
            url,
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "audio/wav",
            },
            json={
                "text": text,
                "model_id": model_id,
            },
            timeout=30.0,
        )
        if r.is_success and r.content:
            with open(path, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"ElevenLabs TTS failed: {e}")
    print(f"ElevenLabs TTS failed: {r.text}")
    return False


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: tts.py <text> [output.wav]")
    text = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "output.wav"

    engine = os.environ.get("TTS_ENGINE", "groq").lower()

    ok = False
    if engine == "elevenlabs":
        ok = _try_elevenlabs_tts(text, path)
        if not ok:
            print("elevenlabs tts failed, falling back to Groq/pyttsx3")

    if not ok:
        ok = _try_groq_tts(text, path)

    if not ok:
        print("network tts failed, falling back to pyttsx3")
        engine_local = pyttsx3.init()
        engine_local.save_to_file(text, path)
        engine_local.runAndWait()


if __name__ == "__main__":
    main()
