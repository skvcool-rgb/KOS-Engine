"""KOS V6.0 — Ears (Speech-to-Text via Whisper)

Converts audio input (files or microphone) into text, then optionally
feeds the transcription into the KOS knowledge graph via the TextDriver.

Whisper and sounddevice are optional heavy dependencies.  If they are
missing, the class loads but methods raise informative RuntimeErrors
instead of crashing the whole system.
"""

from __future__ import annotations

import os
import time
import struct
import tempfile
import wave
from typing import Callable, Dict, Optional

# ---------------------------------------------------------------------------
# Optional dependency: whisper
# ---------------------------------------------------------------------------
_whisper = None
try:
    import whisper as _whisper
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Optional dependency: sounddevice + numpy (microphone capture)
# ---------------------------------------------------------------------------
_sd = None
_np = None
try:
    import sounddevice as _sd
    import numpy as _np
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_whisper():
    if _whisper is None:
        raise RuntimeError(
            "Whisper is not installed.  Run:  pip install openai-whisper"
        )


def _require_microphone():
    if _sd is None or _np is None:
        raise RuntimeError(
            "Microphone capture requires sounddevice and numpy.  "
            "Run:  pip install sounddevice numpy"
        )


def _save_wav(samples, sample_rate: int, path: str):
    """Write a numpy float32 array to a 16-bit WAV file."""
    pcm = (_np.clip(samples, -1.0, 1.0) * 32767).astype(_np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _rms(block):
    """Root-mean-square energy of a numpy array."""
    return float(_np.sqrt(_np.mean(block.astype(_np.float64) ** 2)))


# ---------------------------------------------------------------------------
# Ears
# ---------------------------------------------------------------------------

class Ears:
    """Speech-to-text sense organ for KOS.

    Parameters
    ----------
    model_name : str
        Whisper model size — "tiny", "base", "small", "medium", "large".
        Default is "base" (good accuracy, CPU-friendly).
    device : str or None
        Torch device string.  None = auto (CUDA if available, else CPU).
    sample_rate : int
        Audio sample rate in Hz for microphone capture.
    """

    def __init__(self, model_name: str = "base",
                 device: Optional[str] = None,
                 sample_rate: int = 16_000):
        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate
        self._model = None   # lazy-loaded

    # -- Model management ---------------------------------------------------

    def _get_model(self):
        """Lazy-load the Whisper model on first use."""
        _require_whisper()
        if self._model is None:
            self._model = _whisper.load_model(
                self.model_name, device=self.device
            )
        return self._model

    # -- Core transcription -------------------------------------------------

    def _transcribe(self, audio_path: str) -> Dict:
        """Run Whisper on an audio file and return a standardised result dict."""
        model = self._get_model()
        result = model.transcribe(audio_path)

        text = result.get("text", "").strip()
        language = result.get("language", "unknown")

        # Estimate duration from segments
        segments = result.get("segments", [])
        duration = 0.0
        total_logprob = 0.0
        n_tokens = 0
        for seg in segments:
            end = seg.get("end", 0.0)
            if end > duration:
                duration = end
            # Accumulate token-level log-probabilities for confidence
            avg_lp = seg.get("avg_logprob", 0.0)
            n_tok = max(len(seg.get("tokens", [])), 1)
            total_logprob += avg_lp * n_tok
            n_tokens += n_tok

        # Convert mean log-probability to a 0-1 confidence score
        import math
        confidence = 0.0
        if n_tokens > 0:
            mean_lp = total_logprob / n_tokens
            confidence = round(math.exp(mean_lp), 4)

        return {
            "text": text,
            "language": language,
            "duration_sec": round(duration, 2),
            "confidence": confidence,
        }

    # -- Public API ---------------------------------------------------------

    def listen_from_file(self, audio_path: str) -> Dict:
        """Transcribe an audio file (wav, mp3, flac, etc.) to text.

        Returns
        -------
        dict
            {"text", "language", "duration_sec", "confidence"}
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return self._transcribe(audio_path)

    def listen_from_mic(self, duration_sec: float = 5.0) -> Dict:
        """Record from the default microphone for *duration_sec* seconds,
        then transcribe.

        Returns
        -------
        dict
            {"text", "language", "duration_sec", "confidence"}
        """
        _require_microphone()
        _require_whisper()

        recording = _sd.rec(
            int(duration_sec * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        _sd.wait()

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            _save_wav(recording.flatten(), self.sample_rate, tmp_path)
            return self._transcribe(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def listen_continuous(self, callback: Callable[[Dict], None],
                          silence_threshold: float = 500.0,
                          silence_duration: float = 1.5):
        """Listen from microphone until silence is detected, transcribe,
        and call *callback* with the result dict.  Repeats indefinitely
        until interrupted (Ctrl+C).

        Parameters
        ----------
        callback : callable
            Called with the transcription result dict after each utterance.
        silence_threshold : float
            RMS amplitude below which audio is considered silence.
            Typical range: 100-2000 depending on mic gain.
        silence_duration : float
            Seconds of continuous silence required to trigger transcription.
        """
        _require_microphone()
        _require_whisper()

        chunk_duration = 0.1   # seconds per chunk
        chunk_samples = int(self.sample_rate * chunk_duration)
        silence_chunks_needed = int(silence_duration / chunk_duration)

        print("[Ears] Continuous listening started.  Ctrl+C to stop.")
        try:
            while True:
                frames = []
                silent_chunks = 0
                speaking = False

                while True:
                    chunk = _sd.rec(chunk_samples, samplerate=self.sample_rate,
                                    channels=1, dtype="float32")
                    _sd.wait()
                    energy = _rms(chunk) * 32767  # scale to 16-bit range
                    frames.append(chunk)

                    if energy < silence_threshold:
                        silent_chunks += 1
                        if speaking and silent_chunks >= silence_chunks_needed:
                            break
                    else:
                        silent_chunks = 0
                        speaking = True

                if not speaking:
                    continue

                # Concatenate and transcribe
                audio = _np.concatenate(frames, axis=0).flatten()
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_path = tmp.name
                tmp.close()
                try:
                    _save_wav(audio, self.sample_rate, tmp_path)
                    result = self._transcribe(tmp_path)
                    if result["text"]:
                        callback(result)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        except KeyboardInterrupt:
            print("\n[Ears] Continuous listening stopped.")

    # -- KOS integration ----------------------------------------------------

    def process_for_kos(self, text_driver, text: str) -> Optional[Dict]:
        """Ingest transcribed text into the KOS knowledge graph.

        Parameters
        ----------
        text_driver : kos.drivers.text.TextDriver
            An initialised TextDriver bound to a KOSKernel + KASMLexicon.
        text : str
            The transcribed text to ingest.

        Returns
        -------
        dict or None
            The ingestion stats dict from TextDriver.ingest(), or None if
            the text was empty.
        """
        if not text or not text.strip():
            return None
        return text_driver.ingest(text)


# ---------------------------------------------------------------------------
# Module-level availability check
# ---------------------------------------------------------------------------

if _whisper is None:
    print("[Ears] WARNING: openai-whisper not installed. "
          "Speech-to-text will not work.  pip install openai-whisper")
if _sd is None:
    print("[Ears] WARNING: sounddevice not installed. "
          "Microphone capture will not work.  pip install sounddevice numpy")
