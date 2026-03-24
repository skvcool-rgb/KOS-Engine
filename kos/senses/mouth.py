"""KOS V6.0 — Mouth (Text-to-Speech via OpenAI TTS)

Converts text to speech using the OpenAI TTS API, with an offline
fallback to pyttsx3.  Integrates with the KOS EmotionEngine to adjust
voice characteristics based on emotional state.

openai, pyttsx3, and audio-playback libraries are all optional.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
_openai = None
try:
    import openai as _openai
except ImportError:
    pass

_pyttsx3 = None
try:
    import pyttsx3 as _pyttsx3
except ImportError:
    pass

_playsound_fn = None
try:
    from playsound import playsound as _playsound_fn
except ImportError:
    pass

_simpleaudio = None
try:
    import simpleaudio as _simpleaudio
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOICES = ("alloy", "echo", "fable", "onyx", "nova", "shimmer")
SPEED_MIN = 0.25
SPEED_MAX = 4.0
DEFAULT_VOICE = "nova"
DEFAULT_SPEED = 1.0
DEFAULT_MODEL = "tts-1"


# ---------------------------------------------------------------------------
# Emotion-to-voice parameter mapping
# ---------------------------------------------------------------------------

# Each entry: {"speed_mult": float, "voice_hint": str or None}
# speed_mult is applied multiplicatively on top of the base speed.
_EMOTION_PROFILES: Dict[str, Dict] = {
    "fear":        {"speed_mult": 1.35, "voice_hint": None},
    "anxiety":     {"speed_mult": 1.25, "voice_hint": None},
    "calm":        {"speed_mult": 0.80, "voice_hint": None},
    "joy":         {"speed_mult": 1.15, "voice_hint": None},
    "depression":  {"speed_mult": 0.70, "voice_hint": None},
    "anger":       {"speed_mult": 1.20, "voice_hint": "onyx"},
    "sadness":     {"speed_mult": 0.75, "voice_hint": "fable"},
    "excitement":  {"speed_mult": 1.30, "voice_hint": None},
    "neutral":     {"speed_mult": 1.00, "voice_hint": None},
}


# ---------------------------------------------------------------------------
# Audio playback helpers
# ---------------------------------------------------------------------------

def _play_audio_file(path: str):
    """Best-effort playback of an audio file using whatever is available."""
    # 1. playsound
    if _playsound_fn is not None:
        try:
            _playsound_fn(path)
            return
        except Exception:
            pass

    # 2. simpleaudio (wav only — skip for mp3)
    if _simpleaudio is not None and path.lower().endswith(".wav"):
        try:
            wave_obj = _simpleaudio.WaveObject.from_wave_file(path)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            return
        except Exception:
            pass

    # 3. System command fallback
    try:
        if sys.platform == "win32":
            # Windows Media Player CLI
            subprocess.run(
                ["powershell", "-c",
                 f'(New-Object Media.SoundPlayer "{path}").PlaySync()'],
                check=False, capture_output=True,
            )
        elif sys.platform == "darwin":
            subprocess.run(["afplay", path], check=False, capture_output=True)
        else:
            # Linux — try aplay for wav, mpv/ffplay for mp3
            if path.lower().endswith(".wav"):
                subprocess.run(["aplay", path],
                               check=False, capture_output=True)
            else:
                for player in ("mpv", "ffplay", "cvlc"):
                    try:
                        cmd = [player]
                        if player == "ffplay":
                            cmd += ["-nodisp", "-autoexit"]
                        elif player == "cvlc":
                            cmd += ["--play-and-exit"]
                        cmd.append(path)
                        subprocess.run(cmd, check=False, capture_output=True)
                        return
                    except FileNotFoundError:
                        continue
    except Exception:
        print(f"[Mouth] WARNING: Could not play audio file: {path}")


# ---------------------------------------------------------------------------
# Mouth
# ---------------------------------------------------------------------------

class Mouth:
    """Text-to-speech sense organ for KOS.

    Parameters
    ----------
    voice : str
        One of: alloy, echo, fable, onyx, nova, shimmer.
    speed : float
        Playback speed multiplier (0.25 - 4.0).
    model : str
        OpenAI TTS model name ("tts-1" or "tts-1-hd").
    api_key : str or None
        OpenAI API key.  If None, reads from OPENAI_API_KEY env var.
    """

    def __init__(self, voice: str = DEFAULT_VOICE,
                 speed: float = DEFAULT_SPEED,
                 model: str = DEFAULT_MODEL,
                 api_key: Optional[str] = None):
        if voice not in VOICES:
            raise ValueError(
                f"Invalid voice '{voice}'.  Choose from: {VOICES}"
            )
        self.voice = voice
        self.speed = max(SPEED_MIN, min(SPEED_MAX, speed))
        self.model = model
        self._api_key = api_key
        self._client = None   # lazy OpenAI client
        self._pyttsx_engine = None  # lazy pyttsx3 engine

    # -- Client management --------------------------------------------------

    def _get_openai_client(self):
        if _openai is None:
            return None
        if self._client is None:
            key = self._api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                return None
            self._client = _openai.OpenAI(api_key=key)
        return self._client

    def _get_pyttsx_engine(self):
        if _pyttsx3 is None:
            return None
        if self._pyttsx_engine is None:
            self._pyttsx_engine = _pyttsx3.init()
        return self._pyttsx_engine

    # -- Core TTS -----------------------------------------------------------

    def _tts_openai(self, text: str, voice: str, speed: float,
                    output_path: Optional[str] = None) -> Optional[str]:
        """Generate speech with OpenAI TTS.  Returns path to audio file."""
        client = self._get_openai_client()
        if client is None:
            return None

        response = client.audio.speech.create(
            model=self.model,
            voice=voice,
            input=text,
            speed=speed,
        )

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            )
            output_path = tmp.name
            tmp.close()

        response.stream_to_file(output_path)
        return output_path

    def _tts_pyttsx(self, text: str, speed: float,
                    output_path: Optional[str] = None) -> Optional[str]:
        """Generate speech with pyttsx3 (offline fallback)."""
        engine = self._get_pyttsx_engine()
        if engine is None:
            return None

        # Adjust rate based on speed multiplier
        base_rate = engine.getProperty("rate") or 200
        engine.setProperty("rate", int(base_rate * speed))

        if output_path:
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            return output_path
        else:
            engine.say(text)
            engine.runAndWait()
            return None  # played directly

    # -- Public API ---------------------------------------------------------

    def speak(self, text: str, voice: Optional[str] = None,
              speed: Optional[float] = None):
        """Generate speech from text and play it through speakers.

        Parameters
        ----------
        text : str
            The text to speak.
        voice : str or None
            Override voice for this call.  None uses the default.
        speed : float or None
            Override speed for this call.  None uses the default.
        """
        voice = voice or self.voice
        speed = speed if speed is not None else self.speed
        speed = max(SPEED_MIN, min(SPEED_MAX, speed))

        # Try OpenAI first
        path = self._tts_openai(text, voice, speed)
        if path:
            try:
                _play_audio_file(path)
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            return

        # Fallback to pyttsx3 (plays directly)
        result = self._tts_pyttsx(text, speed)
        if result is None and _pyttsx3 is not None:
            return  # pyttsx3 played it directly via engine.say()

        raise RuntimeError(
            "No TTS backend available.  Install openai (pip install openai) "
            "or pyttsx3 (pip install pyttsx3)."
        )

    def speak_to_file(self, text: str, output_path: str,
                      voice: Optional[str] = None,
                      speed: Optional[float] = None):
        """Generate speech and save to an audio file.

        Parameters
        ----------
        text : str
            The text to speak.
        output_path : str
            Destination file path (mp3 for OpenAI, wav for pyttsx3).
        voice : str or None
            Override voice.
        speed : float or None
            Override speed.
        """
        voice = voice or self.voice
        speed = speed if speed is not None else self.speed
        speed = max(SPEED_MIN, min(SPEED_MAX, speed))

        # Try OpenAI first
        result = self._tts_openai(text, voice, speed, output_path=output_path)
        if result:
            return

        # Fallback to pyttsx3
        result = self._tts_pyttsx(text, speed, output_path=output_path)
        if result:
            return

        raise RuntimeError(
            "No TTS backend available.  Install openai or pyttsx3."
        )

    def speak_with_emotion(self, text: str, emotion_state) -> Dict:
        """Speak text with voice parameters adjusted to match an emotional
        state from the KOS EmotionEngine.

        Parameters
        ----------
        text : str
            The text to speak.
        emotion_state : str or object
            Either a string emotion name (e.g. "fear", "joy", "calm") or
            an object with a .dominant_emotion attribute (as returned by
            EmotionEngine.get_emotion_state()).

        Returns
        -------
        dict
            {"emotion": str, "voice": str, "speed": float, "spoken": bool}
        """
        # Resolve emotion name
        if isinstance(emotion_state, str):
            emotion_name = emotion_state.lower()
        elif hasattr(emotion_state, "dominant_emotion"):
            emotion_name = emotion_state.dominant_emotion.lower()
        elif isinstance(emotion_state, dict):
            emotion_name = emotion_state.get("dominant_emotion", "neutral").lower()
        else:
            emotion_name = "neutral"

        profile = _EMOTION_PROFILES.get(emotion_name,
                                        _EMOTION_PROFILES["neutral"])
        speed = self.speed * profile["speed_mult"]
        speed = max(SPEED_MIN, min(SPEED_MAX, speed))
        voice = profile["voice_hint"] or self.voice

        spoken = False
        try:
            self.speak(text, voice=voice, speed=speed)
            spoken = True
        except RuntimeError:
            pass

        return {
            "emotion": emotion_name,
            "voice": voice,
            "speed": round(speed, 2),
            "spoken": spoken,
        }


# ---------------------------------------------------------------------------
# Module-level availability check
# ---------------------------------------------------------------------------

_has_any_backend = False
if _openai is not None:
    _has_any_backend = True
if _pyttsx3 is not None:
    _has_any_backend = True

if not _has_any_backend:
    print("[Mouth] WARNING: No TTS backend available. "
          "Install openai (pip install openai) or pyttsx3 (pip install pyttsx3).")
elif _openai is None:
    print("[Mouth] INFO: openai not installed.  "
          "Using pyttsx3 offline TTS as fallback.")
