"""Continuous speech-to-text from the robot's mic; posts transcribed text to the client.

Uses Voice Activity Detection (VAD) to detect natural speech boundaries—transcribes when
the user finishes speaking, not on fixed time intervals.
"""

from __future__ import annotations
import os
import threading
import time
import numpy as np
import scipy.signal

import httpx
from reachy_mini import ReachyMini

# Where to POST transcribed text (client endpoint)
STT_URL = os.environ.get("STT_CALLBACK_URL", "http://localhost:8765/stt")

# VAD settings: how long silence before we consider speech finished
SILENCE_THRESHOLD_SEC = 1.5  # seconds of silence before transcribing
# Small chunk duration for VAD (we check VAD on these)
VAD_CHUNK_DURATION = 0.3  # seconds per VAD check
# Minimum speech duration to transcribe (avoid transcribing noise)
MIN_SPEECH_DURATION_SEC = 0.5

# Sleep between get_audio_sample() polls (default backend)
POLL_INTERVAL = 0.1


def _load_whisper_model():
    """Load whisper model once; returns None if whisper not installed."""
    try:
        import whisper
        return whisper.load_model("tiny")
    except ImportError:
        return None


def _load_vad_model():
    """Load VAD model (silero-vad) if available; returns (model, get_speech_timestamps) or (None, None)."""
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad

        model = load_silero_vad()  # returns a single TorchScript model
        return model, get_speech_timestamps
    except Exception:
        # Any import/load problem => no VAD, we’ll use simple RMS fallback
        return None, None


def _has_speech_vad(audio_chunk: np.ndarray, sample_rate: int, vad_model, get_speech_timestamps) -> bool:
    """Check if audio chunk contains speech using VAD. Returns True if speech detected."""
    if vad_model is None or audio_chunk.size == 0:
        return False
    try:
        import torch
        audio_tensor = torch.from_numpy(audio_chunk).float()
        if sample_rate != 16000:
            import torchaudio.transforms as T
            resampler = T.Resample(sample_rate, 16000)
            audio_tensor = resampler(audio_tensor)
        speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=16000)
        return len(speech_timestamps) > 0
    except Exception:
        return False


def _has_speech_simple(audio_chunk: np.ndarray, sample_rate: int) -> bool:
    """Simple energy-based VAD fallback: check if RMS exceeds threshold."""
    if audio_chunk.size == 0:
        return False
    rms = np.sqrt(np.mean(audio_chunk**2))
    threshold = 0.01
    return rms > threshold


def _transcribe(audio_data: np.ndarray, sample_rate: int, model=None) -> str:
    """Convert audio to text. Uses whisper if available; otherwise returns empty string."""
    if audio_data.size == 0:
        return ""
    if model is None:
        model = _load_whisper_model()
    if model is None:
        return ""
    # Whisper's file-path flow shells out to `ffmpeg` on Windows.
    # To avoid that dependency, feed raw audio (numpy float32) directly.
    audio = audio_data.astype(np.float32, copy=False)

    # Resample to 16kHz which Whisper expects.
    target_sr = 16000
    if sample_rate != target_sr and audio.size:
        target_len = int(round(len(audio) * (target_sr / float(sample_rate))))
        if target_len > 0:
            audio = scipy.signal.resample(audio, target_len).astype(np.float32, copy=False)

    result = model.transcribe(audio, fp16=False)
    return (result.get("text") or "").strip()


def _record_until_silence(
    mini: ReachyMini,
    vad_model,
    get_speech_timestamps,
    stop_event: threading.Event,
) -> tuple[np.ndarray, int]:
    """Record from mini.media until speech ends (VAD detects silence for SILENCE_THRESHOLD_SEC).
    Returns (accumulated_audio, sample_rate). Accumulates audio while speech is detected.
    """
    sample_rate = mini.media.get_input_audio_samplerate()
    accumulated_samples = []
    speech_started = False
    last_speech_time = None
    mini.media.start_recording()
    print("started recording")
    try:
        while not stop_event.is_set():
            chunk_samples = []
            t0 = time.time()
            while time.time() - t0 < VAD_CHUNK_DURATION:
                sample = mini.media.get_audio_sample()
                if sample is not None:
                    chunk_samples.append(sample)
                time.sleep(POLL_INTERVAL)
                if stop_event.is_set():
                    break
            if not chunk_samples:
                continue
            chunk_audio = np.concatenate(chunk_samples, axis=0)
            if chunk_audio.ndim > 1:
                chunk_audio = np.mean(chunk_audio, axis=1)
            has_speech = False
            if vad_model is not None:
                has_speech = _has_speech_vad(chunk_audio, sample_rate, vad_model, get_speech_timestamps)
            else:
                has_speech = _has_speech_simple(chunk_audio, sample_rate)
            print("has_speech: " + str(has_speech))
            if has_speech:
                speech_started = True
                last_speech_time = time.time()
                accumulated_samples.append(chunk_audio)
            elif speech_started:
                accumulated_samples.append(chunk_audio)
                if last_speech_time and (time.time() - last_speech_time) >= SILENCE_THRESHOLD_SEC:
                    break
            elif not speech_started:
                continue
    finally:
        mini.media.stop_recording()
    if not accumulated_samples:
        return np.array([]), sample_rate
    audio = np.concatenate(accumulated_samples, axis=0)
    duration = len(audio) / sample_rate
    if duration < MIN_SPEECH_DURATION_SEC:
        return np.array([]), sample_rate
    return audio, sample_rate


def run_stt_loop(mini: ReachyMini, stt_url: str | None = None, stop_event: threading.Event | None = None) -> None:
    """Run in a background thread: continuously listen, detect speech with VAD, transcribe when user finishes speaking, POST to client.
    Stops when stop_event is set (if provided).
    """
    url = (stt_url or STT_URL).rstrip("/")
    stop = stop_event or threading.Event()
    whisper_model = _load_whisper_model()
    vad_model, get_speech_timestamps = _load_vad_model()
    if vad_model is None:
        print("Warning: silero-vad not installed; using simple energy-based VAD (less accurate)", file=__import__("sys").stderr)
    while not stop.is_set():
        try:
            print("recording until silence")
            audio, sr = _record_until_silence(mini, vad_model, get_speech_timestamps, stop)
            if audio.size == 0:
                continue
            print("transcribing")
            text = _transcribe(audio, sr, model=whisper_model)
            if not text:
                continue
            print("posting to client", text)
            try:
                httpx.post(url, json={"text": text}, timeout=5.0)
            except httpx.HTTPError:
                pass
        except Exception:
            if stop.is_set():
                break
            time.sleep(0.5)


def start_stt_loop(mini: ReachyMini, stt_url: str | None = None) -> tuple[threading.Thread, threading.Event]:
    """Start the STT loop in a daemon thread. Returns (thread, stop_event). Call stop_event.set() to stop."""
    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_stt_loop,
        args=(mini, stt_url),
        kwargs={"stop_event": stop_event},
        daemon=True,
    )
    thread.start()
    return thread, stop_event
