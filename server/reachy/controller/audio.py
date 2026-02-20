import os
import queue
import scipy
import soundfile as sf
import numpy as np
import subprocess
import sys
import threading
import time
from reachy_mini import ReachyMini
from threading import Thread

_TTS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts.py")

# Simple queue for speak requests from multiple agents
_speak_queue: queue.Queue[tuple[ReachyMini, str]] = queue.Queue()
_speak_worker_thread: Thread | None = None
_speak_lock = threading.Lock()


def _play(path: str, mini: ReachyMini) -> None:
    """Play audio file synchronously."""
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


def _speak_worker() -> None:
    """Background thread that processes the speak queue one item at a time."""
    global _speak_worker_thread
    while True:
        try:
            mini, text = _speak_queue.get()
            if mini is None:  # Sentinel to stop
                break
            print("Generating audio: " + text)
            project_dir = os.path.dirname(os.path.abspath(__file__))
            subprocess.run(
                [sys.executable, _TTS_SCRIPT, text, "output.wav"],
                check=True,
                cwd=project_dir,
            )
            print("TTS done, playing...")
            _play("server/reachy/controller/output.wav", mini)
            _speak_queue.task_done()
        except Exception as e:
            print(f"Error in speak worker: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            _speak_queue.task_done()


def speak(mini: ReachyMini, text: str, forcefully_interrupt: bool = False) -> str:
    """Speak words using text to speech with Reachy Mini's speaker.
    
    Multiple calls are queued and processed sequentially by a background thread.

    Args:
        mini: ReachyMini instance
        text: Text to speak
        forcefully_interrupt: If True, clear the queue and stop current playback before queuing this.
    """
    global _speak_worker_thread
    
    with _speak_lock:
        if forcefully_interrupt:
            # Clear queue and stop playback
            try:
                mini.media.stop_playing()
            except Exception:
                pass
            while not _speak_queue.empty():
                try:
                    _speak_queue.get_nowait()
                    _speak_queue.task_done()
                except queue.Empty:
                    break
        
        # Start worker thread if not running
        if _speak_worker_thread is None or not _speak_worker_thread.is_alive():
            _speak_worker_thread = Thread(target=_speak_worker, daemon=True)
            _speak_worker_thread.start()
        
        # Queue the request
        _speak_queue.put((mini, text))
    
    return "Done"