import os
import scipy
import soundfile as sf
import numpy as np
import subprocess
import sys
import time
from reachy_mini import ReachyMini
from threading import Thread

_TTS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts.py")

def _play(path: str, mini: ReachyMini) -> None:
    try:
        data, samplerate_in = sf.read(path, dtype="float32")
        if samplerate_in != mini.media.get_output_audio_samplerate():
            data = scipy.signal.resample(
                data,
                int(
                    len(data)
                    * (mini.media.get_output_audio_samplerate() / samplerate_in)
                ),
            )
        if data.ndim > 1:  # convert to mono
            data = np.mean(data, axis=1)
        mini.media.start_playing()
        print("Playing audio...")
        # Push samples in chunks
        chunk_size = 1024
        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            mini.media.push_audio_sample(chunk)
        # Wait for playback to finish: duration = samples / sample_rate
        output_sr = mini.media.get_output_audio_samplerate()
        duration_sec = len(data) / output_sr
        time.sleep(duration_sec)
        mini.media.stop_playing()
        print("Playback finished.")
    except Exception as e:
        # Ensure we stop playing even if there's an error
        try:
            mini.media.stop_playing()
        except:
            pass
        raise

def _speak_worker(mini: ReachyMini, text: str) -> None:
    """Generate TTS and play on mini; runs in a background thread.

    TTS is run in a subprocess so pyttsx3.runAndWait() runs on a main thread,
    avoiding deadlock when this worker runs in a thread (e.g. on Windows).
    """
    try:
        print("Generating audio: " + text)
        project_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(
            [sys.executable, _TTS_SCRIPT, text, "output.wav"],
            check=True,
            cwd=project_dir,
        )
        print("TTS done, playing...")
        _play("server/controller/output.wav", mini)
    except Exception as e:
        print(f"Error in TTS worker: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

def speak(mini: ReachyMini, text: str) -> str:
    """Speak words using text to speech with Reachy Mini's speaker.

    Runs in a background thread to avoid blocking the FastMCP event loop.
    """
    t = Thread(target=_speak_worker, args=(mini, text), daemon=True)
    t.start()