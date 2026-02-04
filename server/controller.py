import os
import subprocess
import sys
import soundfile as sf
import scipy.signal
import numpy as np
import time
from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves

# TTS is run in a subprocess so pyttsx3 runs on a main thread (avoids Windows deadlock in worker threads).
_TTS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_generate.py")
EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
recorded_emotions = RecordedMoves(EMOTIONS_DATASET)
move_names = recorded_emotions.list_moves()
moves_and_descriptions = {name: recorded_emotions.get(name).description for name in move_names}

def play(path: str, mini: ReachyMini) -> None:
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
    # Wait fosr playback to finish: duration = samples / sample_rate
    output_sr = mini.media.get_output_audio_samplerate()
    duration_sec = len(data) / output_sr
    time.sleep(duration_sec)
    mini.media.stop_playing()
    print("Playback finished.")

def _play_emotion_worker(mini: ReachyMini, emotion: str) -> None:
    """Internal helper to play an emotion in a separate thread."""
    move = recorded_emotions.get(emotion)
    mini.play_move(move, initial_goto_duration=0.7)

def _speak_worker(mini: ReachyMini, text: str) -> None:
    """Generate TTS and play on mini; runs in a background thread.

    TTS is run in a subprocess so pyttsx3.runAndWait() runs on a main thread,
    avoiding deadlock when this worker runs in a thread (e.g. on Windows).
    """
    print("Generating audio: " + text)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(
        [sys.executable, _TTS_SCRIPT, text, "output.wav"],
        check=True,
        cwd=project_dir,
    )
    print("TTS done, playing...")
    play("output.wav", mini)
    