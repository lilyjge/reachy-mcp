import os
import scipy
import soundfile as sf
import numpy as np
import subprocess
import sys
import time
import queue
import threading
from reachy_mini import ReachyMini
from threading import Thread, Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .movement_manager import MovementManager

_TTS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts.py")

# Thread-safe state management for speak queueing
_speak_lock = Lock()
_current_speak_thread: Thread | None = None
_speak_queue: queue.Queue[tuple[ReachyMini, str]] = queue.Queue()
_stop_current_speak = False

# Global reference to movement manager (set by server.py)
_movement_manager: "MovementManager | None" = None

def set_movement_manager(manager: "MovementManager") -> None:
    """Set the global movement manager reference."""
    global _movement_manager
    _movement_manager = manager

def _play(path: str, mini: ReachyMini) -> None:
    """Play audio file, checking for interruption periodically."""
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
        # Push samples in chunks, checking for interruption
        chunk_size = 1024
        for i in range(0, len(data), chunk_size):
            # Check if we should stop
            with _speak_lock:
                if _stop_current_speak:
                    mini.media.stop_playing()
                    print("Playback interrupted.")
                    return
            chunk = data[i : i + chunk_size]
            mini.media.push_audio_sample(chunk)
        # Wait for playback to finish: duration = samples / sample_rate
        output_sr = mini.media.get_output_audio_samplerate()
        duration_sec = len(data) / output_sr
        # Check for interruption during sleep
        elapsed = 0.0
        check_interval = 0.1  # Check every 100ms
        while elapsed < duration_sec:
            time.sleep(min(check_interval, duration_sec - elapsed))
            elapsed += check_interval
            with _speak_lock:
                if _stop_current_speak:
                    mini.media.stop_playing()
                    print("Playback interrupted.")
                    return
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
    After finishing, processes the next item in the queue if any.
    """
    global _current_speak_thread, _stop_current_speak
    
    try:
        # Check for interruption before starting TTS generation
        with _speak_lock:
            if _stop_current_speak:
                print("Speak worker interrupted before TTS generation.")
                return
        
        print("Generating audio: " + text)
        project_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(
            [sys.executable, _TTS_SCRIPT, text, "output.wav"],
            check=True,
            cwd=project_dir,
        )
        
        # Check for interruption after TTS generation
        with _speak_lock:
            if _stop_current_speak:
                print("Speak worker interrupted after TTS generation.")
                return
        
        print("TTS done, playing...")
        _play("server/controller/output.wav", mini)
    except Exception as e:
        print(f"Error in TTS worker: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        # Clear current thread and process queue
        with _speak_lock:
            # Only clear if this thread is still the current one (not interrupted and replaced)
            current_thread = threading.current_thread()
            if _current_speak_thread is current_thread:
                _current_speak_thread = None
            _stop_current_speak = False
            
            # Process next item in queue if available and no thread is running
            if _current_speak_thread is None:
                try:
                    next_mini, next_text = _speak_queue.get_nowait()
                    print(f"Processing queued speak: {next_text}")
                    _current_speak_thread = Thread(
                        target=_speak_worker, 
                        args=(next_mini, next_text), 
                        daemon=True
                    )
                    _current_speak_thread.start()
                except queue.Empty:
                    pass  # No queued items

def speak(mini: ReachyMini, text: str, forcefully_interrupt: bool = False) -> str:
    """Speak words using text to speech with Reachy Mini's speaker.

    Args:
        mini: ReachyMini instance
        text: Text to speak
        forcefully_interrupt: If True and robot is currently speaking, stop current
            speech, clear queue, and speak immediately. If False and robot is speaking,
            queue this request to execute after current speech finishes.

    Runs in a background thread to avoid blocking the FastMCP event loop.
    """
    global _current_speak_thread, _stop_current_speak
    
    # Mark activity to stop breathing and head tracking
    if _movement_manager:
        _movement_manager.mark_activity()
    
    with _speak_lock:
        is_speaking = _current_speak_thread is not None and _current_speak_thread.is_alive()
        
        if forcefully_interrupt and is_speaking:
            # Stop current speech and clear queue
            print("Forcefully interrupting current speech and clearing queue.")
            _stop_current_speak = True
            # Clear the queue
            while not _speak_queue.empty():
                try:
                    _speak_queue.get_nowait()
                except queue.Empty:
                    break
            # Don't reset _stop_current_speak here - let the old thread see it and stop
            # The old thread will reset it in its finally block
            # We'll start a new thread which will have _stop_current_speak = False
            # (reset by the old thread's finally block, or already False if old thread finished)
        
        if forcefully_interrupt or not is_speaking:
            # Start speaking immediately
            # If we interrupted, the old thread will reset _stop_current_speak in its finally block
            # If we didn't interrupt, _stop_current_speak is already False
            _current_speak_thread = Thread(
                target=_speak_worker, 
                args=(mini, text), 
                daemon=True
            )
            _current_speak_thread.start()
        else:
            # Queue the request
            print(f"Robot is currently speaking. Queuing: {text}")
            _speak_queue.put((mini, text))
    
    return "Done"