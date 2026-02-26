"""Continuous speech-to-text from the robot's mic; posts transcribed text to the client.

Uses Voice Activity Detection (VAD) to detect natural speech boundaries—transcribes when
the user finishes speaking, not on fixed time intervals.
"""

from __future__ import annotations
import io
import os
import queue
import threading
import time
import wave
from collections import deque
import numpy as np
import scipy.signal
import dotenv
import httpx
from reachy_mini import ReachyMini
from .vision import wait_for_eye_contact
from .controls import move_to_audio, center_to_face, listening_worker
dotenv.load_dotenv()

# Groq speech-to-text (OpenAI-compatible)
GROQ_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"

# Where to POST transcribed text (client endpoint)
STT_URL = os.environ.get("STT_CALLBACK_URL") or "http://localhost:%s/stt" % os.environ.get("RAG_AGENT_PORT", "8765")

# VAD settings: how long silence before we consider speech finished
SILENCE_THRESHOLD_SEC = 1.5  # seconds of silence before transcribing
# Small chunk duration for VAD (we check VAD on these)
VAD_CHUNK_DURATION = 0.3  # seconds per VAD check
# Minimum speech duration to transcribe (avoid transcribing noise)
MIN_SPEECH_DURATION_SEC = 0.5

# Sleep between get_audio_sample() polls (default backend)
POLL_INTERVAL = 0.02

# Keep a short pre-speech context so first phonemes are not clipped.
PRE_SPEECH_BUFFER_SEC = 0.25

# Bounded queue between capture and transcription workers.
UTTERANCE_QUEUE_SIZE = 8

# Wake word to trigger recording when eye contact is absent
WAKE_WORD = "hello"


def _float32_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 mono audio to 16-bit PCM WAV bytes."""
    # Clamp and convert to int16
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


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
    threshold = 0.04
    return rms > threshold


def _transcribe(audio_data: np.ndarray, sample_rate: int) -> str:
    """Convert audio to text via Groq API (OpenAI-compatible transcriptions endpoint)."""
    if audio_data.size == 0:
        return ""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return ""
    audio = audio_data.astype(np.float32, copy=False)

    # Resample to 16kHz for optimal speech recognition.
    target_sr = 16000
    if sample_rate != target_sr and audio.size:
        target_len = int(round(len(audio) * (target_sr / float(sample_rate))))
        if target_len > 0:
            audio = scipy.signal.resample(audio, target_len).astype(np.float32, copy=False)

    wav_bytes = _float32_to_wav_bytes(audio, target_sr)
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                GROQ_TRANSCRIPTIONS_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={"model": GROQ_WHISPER_MODEL, "response_format": "json"},
            )
            response.raise_for_status()
            data = response.json()
            return (data.get("text") or "").strip()
    except (httpx.HTTPError, KeyError) as e:
        if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
            print(
                f"Error transcribing audio: {e.response.text}",
                file=__import__("sys").stderr,
            )
        else:
            print(f"Error transcribing audio: {e}", file=__import__("sys").stderr)
    return ""


def _record_until_silence(
    mini: ReachyMini,
    vad_model,
    get_speech_timestamps,
    stop_event: threading.Event,
) -> tuple[np.ndarray, int, tuple[float, bool]]:
    """Record from mini.media until speech ends (VAD detects silence for SILENCE_THRESHOLD_SEC).
    Returns (accumulated_audio, sample_rate, doa). Accumulates audio while speech is detected.
    """
    sample_rate = int(mini.media.get_input_audio_samplerate())
    if sample_rate <= 0:
        return np.array([]), 16000, (0.0, False)
    chunk_target_samples = max(1, int(sample_rate * VAD_CHUNK_DURATION))
    pre_roll_target_samples = max(1, int(sample_rate * PRE_SPEECH_BUFFER_SEC))

    accumulated_samples: list[np.ndarray] = []
    speech_started = False
    last_speech_time = 0.0
    chunk_parts: list[np.ndarray] = []
    chunk_part_samples = 0
    pre_roll_chunks: deque[np.ndarray] = deque()
    pre_roll_samples = 0
    # Shared mutable container so the DoA poller thread can update it.
    # Using a list so the reference is shared; protected by a lock.
    doa_lock = threading.Lock()
    doa_result: list = [(0, False)]  # single-element list holding the best DoA

    def _poll_doa(stop: threading.Event):
        """Continuously poll get_DoA() in the background, keeping the latest valid value."""
        while not stop.is_set():
            try:
                t_doa = mini.media.get_DoA()
                if t_doa is not None and t_doa[1]:
                    with doa_lock:
                        doa_result[0] = t_doa
            except Exception:
                pass
            stop.wait(timeout=0.15)

    doa_stop = threading.Event()
    doa_thread = threading.Thread(target=_poll_doa, args=(doa_stop,), daemon=True)

    try:
        mini.media.start_recording()
        print("started recording")
        doa_thread.start()
    except Exception as e:
        print(f"Error starting recording: {e}", file=__import__("sys").stderr)
        doa_stop.set()
        return np.array([]), sample_rate, doa_result[0]
    try:
        while not stop_event.is_set():
            sample = mini.media.get_audio_sample()
            if sample is None:
                stop_event.wait(timeout=POLL_INTERVAL)
                continue

            sample_chunk = sample.astype(np.float32, copy=False)
            if sample_chunk.ndim > 1:
                sample_chunk = np.mean(sample_chunk, axis=1)
            if sample_chunk.size == 0:
                continue

            chunk_parts.append(sample_chunk)
            chunk_part_samples += sample_chunk.size
            if chunk_part_samples < chunk_target_samples:
                continue

            chunk_audio = np.concatenate(chunk_parts, axis=0)
            chunk_parts.clear()
            chunk_part_samples = 0

            has_speech = False
            if vad_model is not None:
                has_speech = _has_speech_vad(chunk_audio, sample_rate, vad_model, get_speech_timestamps)
            else:
                has_speech = _has_speech_simple(chunk_audio, sample_rate)
            if has_speech:
                now = time.monotonic()
                speech_started = True
                last_speech_time = now
                if pre_roll_chunks:
                    accumulated_samples.extend(pre_roll_chunks)
                    pre_roll_chunks.clear()
                    pre_roll_samples = 0
                accumulated_samples.append(chunk_audio)
            elif speech_started:
                now = time.monotonic()
                accumulated_samples.append(chunk_audio)
                if (now - last_speech_time) >= SILENCE_THRESHOLD_SEC:
                    break

            if not speech_started:
                pre_roll_chunks.append(chunk_audio)
                pre_roll_samples += chunk_audio.size
                while pre_roll_samples > pre_roll_target_samples and pre_roll_chunks:
                    dropped = pre_roll_chunks.popleft()
                    pre_roll_samples -= dropped.size
    finally:
        doa_stop.set()
        try:
            mini.media.stop_recording()
        except Exception as e:
            print(f"Error stopping recording: {e}", file=__import__("sys").stderr)
    with doa_lock:
        doa = doa_result[0]
    if not accumulated_samples:
        return np.array([]), sample_rate, doa
    audio = np.concatenate(accumulated_samples, axis=0)
    duration = len(audio) / sample_rate
    if duration < MIN_SPEECH_DURATION_SEC:
        return np.array([]), sample_rate, doa
    return audio, sample_rate, doa


def _wait_for_trigger(
    mini: ReachyMini,
    utterance_queue: queue.Queue[tuple[np.ndarray, int]],
    stop_event: threading.Event,
) -> bool:
    """Block until eye contact is detected OR the configured wake word is heard.

    Runs the eye-contact watcher in a background thread while this thread
    records audio and checks transcriptions for the wake word.  Returns True
    if either condition fired, False if stop_event was set before either.
    """
    triggered = threading.Event()
    # combined_stop: set when *either* the caller wants us to stop OR we have
    # already been triggered (so both the eye-contact watcher and the recording
    # loop can exit cleanly).
    combined_stop = threading.Event()

    def _sync_combined():
        """Mirror stop_event / triggered → combined_stop."""
        while not stop_event.is_set() and not triggered.is_set():
            time.sleep(0.05)
        combined_stop.set()

    def _eye_contact_watcher():
        if wait_for_eye_contact(mini, combined_stop, poll_interval=0.08):
            print("trigger: eye contact")
            if not combined_stop.is_set():
                triggered.set()
                center_to_face(mini)

    threading.Thread(target=_sync_combined, daemon=True).start()
    threading.Thread(target=_eye_contact_watcher, daemon=True).start()

    # Record audio in a loop and check each utterance for the wake word.
    while not combined_stop.is_set():
        try:
            audio, sr, doa = utterance_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if combined_stop.is_set():
            break
        if audio.size == 0:
            continue
        text = _transcribe(audio, sr)
        print(f"wake-word check: {text!r}")
        if text and WAKE_WORD in text.lower():
            utterance_queue.queue.clear()  # clear any pending utterances, we only care about the trigger
            print("trigger: wake word")
            if not combined_stop.is_set():
                triggered.set()
                move_to_audio(mini, doa)
            break

    return triggered.is_set() and not stop_event.is_set()


def _capture_utterances_worker(
    mini: ReachyMini,
    vad_model,
    get_speech_timestamps,
    stop_event: threading.Event,
    utterance_queue: queue.Queue[tuple[np.ndarray, int]],
) -> None:
    """Capture utterances continuously and hand them to the transcription queue."""
    while not stop_event.is_set():
        audio, sr, doa = _record_until_silence(mini, vad_model, get_speech_timestamps, stop_event)
        if stop_event.is_set():
            break
        if audio.size == 0:
            continue
        try:
            utterance_queue.put((audio, sr, doa), timeout=0.1)
        except queue.Full:
            print("STT queue is full; dropping utterance", file=__import__("sys").stderr)


def run_stt_loop(mini: ReachyMini, stt_url: str | None = None, stop_event: threading.Event | None = None) -> None:
    """Run in a background thread: continuously listen, detect speech with VAD, transcribe when user finishes speaking, POST to client.
    Stops when stop_event is set (if provided).
    """
    url = (stt_url or STT_URL).rstrip("/")
    stop = stop_event or threading.Event()
    vad_model, get_speech_timestamps = _load_vad_model()
    if vad_model is None:
        print("Warning: silero-vad not installed; using simple energy-based VAD (less accurate)", file=__import__("sys").stderr)

    if stop.is_set():
        return

    utterance_queue: queue.Queue[tuple[np.ndarray, int]] = queue.Queue(maxsize=UTTERANCE_QUEUE_SIZE)
    capture_thread = threading.Thread(
        target=_capture_utterances_worker,
        args=(mini, vad_model, get_speech_timestamps, stop, utterance_queue),
        daemon=True,
    )

    # Start the listening pose thread, which will run concurrently and move the robot to a listening pose while we wait for triggers.
    listening = threading.Event()
    listening_thread = threading.Thread(
        target=listening_worker,
        args=(mini, stop, listening),
        daemon=True,
    )

    capture_thread.start()
    listening_thread.start()

    while not stop.is_set():
        print(f"waiting for eye contact or wake word '{WAKE_WORD}'...")
        if not _wait_for_trigger(mini, utterance_queue, stop):
            continue
        listening.set()  # start the listening pose thread while we wait for the user to speak after the trigger
        print("capture triggered, waiting for speech...")
        audio = None
        try:
            audio, sr, _ = utterance_queue.get(timeout=20)
        except queue.Empty:
            print("No speech detected after trigger.")
            listening.clear()  # stop the listening pose thread once we have a transcription to send
            continue
        print("transcribing")
        text = _transcribe(audio, sr)
        if not text:
            continue
        print("posting to client:", text)
        listening.clear()  # stop the listening pose thread once we have a transcription to send
        try:
            httpx.post(url, json={"text": text}, timeout=5.0)
        except httpx.HTTPError as e:
            print(f"Error posting to client in stt loop: {e}")

    listening.clear()  # stop the listening pose thread
    listening_thread.join(timeout=1.0)
    capture_thread.join(timeout=1.0)


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
