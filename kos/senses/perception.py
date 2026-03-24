"""
KOS V6.0 — Sensory Perception Loop + Multimodal Binding

Connects Eyes, Ears, and Mouth to the core KOS architecture:
1. Sensory Prediction: predict what senses SHOULD detect next
2. Surprise Detection: compare prediction to actual sensory input
3. Attention Filtering: only process surprising/relevant input
4. Multimodal Binding: fuse visual + auditory into KASM vectors
5. Emotion Grounding: sensory input triggers emotional responses

This is the Fristonian active inference loop on real sensory data.
"""

import time
import hashlib
from collections import defaultdict


class SensoryPrediction:
    """
    Predicts what the senses should detect next.
    Surprise = prediction error = attention signal.
    """

    def __init__(self):
        self._visual_history = []   # last N detected objects
        self._audio_history = []    # last N transcriptions
        self._visual_predictions = set()  # expected objects
        self._audio_predictions = set()   # expected sounds/words

    def predict_visual(self) -> set:
        """Predict what objects should be visible based on recent history."""
        if not self._visual_history:
            return set()
        # Simple: predict we'll see the same objects as last frame
        if self._visual_history:
            last = self._visual_history[-1]
            self._visual_predictions = set(last)
        return self._visual_predictions

    def predict_audio(self) -> set:
        """Predict what words/sounds should be heard."""
        if not self._audio_history:
            return set()
        # Predict continuation of recent conversation topic
        if self._audio_history:
            last_words = set()
            for transcript in self._audio_history[-3:]:
                last_words.update(transcript.lower().split())
            self._audio_predictions = last_words
        return self._audio_predictions

    def observe_visual(self, detected_objects: list) -> dict:
        """
        Compare visual prediction to actual detection.
        Returns surprise metrics.
        """
        predicted = self.predict_visual()
        actual = set(obj.get("label", "") for obj in detected_objects)

        self._visual_history.append(list(actual))
        if len(self._visual_history) > 30:
            self._visual_history = self._visual_history[-30:]

        new_objects = actual - predicted
        disappeared = predicted - actual
        surprise = len(new_objects) + len(disappeared)

        return {
            "predicted": list(predicted),
            "actual": list(actual),
            "new_objects": list(new_objects),
            "disappeared": list(disappeared),
            "surprise_level": surprise,
            "is_surprising": surprise > 0,
        }

    def observe_audio(self, transcript: str) -> dict:
        """Compare audio prediction to actual transcription."""
        predicted_words = self.predict_audio()
        actual_words = set(transcript.lower().split())

        self._audio_history.append(transcript)
        if len(self._audio_history) > 20:
            self._audio_history = self._audio_history[-20:]

        new_words = actual_words - predicted_words
        surprise = len(new_words)

        return {
            "transcript": transcript,
            "new_words": list(new_words)[:10],
            "surprise_level": surprise,
            "is_surprising": surprise > 3,
        }


class MultimodalBinder:
    """
    Fuses visual and auditory concepts into unified KASM vectors.

    When KOS sees a cat AND hears "meow":
        BIND cat_multimodal = visual_cat * auditory_meow
    The resulting 10,000-D vector contains BOTH modalities.
    """

    def __init__(self, kasm_engine=None):
        self.kasm = kasm_engine
        self._bindings = {}  # concept -> multimodal KASM vector
        self._modality_log = defaultdict(set)  # concept -> {visual, auditory, text}

    def bind_visual(self, concept: str, visual_features: dict = None):
        """Register a visual observation of a concept."""
        self._modality_log[concept].add("visual")
        if self.kasm:
            if concept not in self._bindings:
                self._bindings[concept] = self.kasm.node(concept + "_multi")
            visual_vec = self.kasm.node(concept + "_visual")
            self._bindings[concept] = self.kasm.bind(
                self._bindings[concept], visual_vec)

    def bind_auditory(self, concept: str, audio_features: dict = None):
        """Register an auditory observation of a concept."""
        self._modality_log[concept].add("auditory")
        if self.kasm:
            if concept not in self._bindings:
                self._bindings[concept] = self.kasm.node(concept + "_multi")
            audio_vec = self.kasm.node(concept + "_audio")
            self._bindings[concept] = self.kasm.bind(
                self._bindings[concept], audio_vec)

    def bind_text(self, concept: str):
        """Register a textual observation of a concept."""
        self._modality_log[concept].add("text")

    def is_grounded(self, concept: str) -> dict:
        """Check how many modalities ground this concept."""
        modalities = self._modality_log.get(concept, set())
        return {
            "concept": concept,
            "modalities": list(modalities),
            "grounding_level": len(modalities),
            "fully_grounded": len(modalities) >= 2,
        }

    def get_all_grounded(self) -> list:
        """Return all concepts with 2+ modalities (grounded)."""
        grounded = []
        for concept, modalities in self._modality_log.items():
            if len(modalities) >= 2:
                grounded.append({
                    "concept": concept,
                    "modalities": list(modalities),
                    "level": len(modalities),
                })
        grounded.sort(key=lambda x: x["level"], reverse=True)
        return grounded


class EmotionGrounding:
    """
    Connects sensory input to the EmotionEngine automatically.

    Loud noise → threat → cortisol spike
    Familiar face → social_bond → oxytocin rise
    New object → novelty → dopamine rise
    Silence after noise → calm → gaba rise
    """

    # Sensory triggers → emotion stimulus mapping
    VISUAL_TRIGGERS = {
        "person": "social_bond",
        "face": "social_bond",
        "dog": "reward",
        "cat": "reward",
        "fire": "threat",
        "knife": "threat",
        "weapon": "threat",
        "food": "reward",
        "baby": "social_bond",
        "smile": "reward",
    }

    AUDIO_TRIGGERS = {
        "help": "threat",
        "danger": "threat",
        "alarm": "threat",
        "scream": "threat",
        "laugh": "reward",
        "music": "reward",
        "silence": "meditation",
        "crying": "grief_event",
        "applause": "reward",
    }

    def __init__(self, emotion_engine=None):
        self.emotion = emotion_engine
        self._trigger_log = []

    def process_visual(self, detections: list) -> list:
        """Process visual detections and trigger emotions."""
        triggered = []
        for det in detections:
            label = det.get("label", "").lower()
            stimulus = self.VISUAL_TRIGGERS.get(label)
            if stimulus and self.emotion:
                self.emotion.apply_stimulus(stimulus)
                triggered.append({
                    "source": "visual",
                    "object": label,
                    "stimulus": stimulus,
                    "emotion_after": self.emotion.current_emotion(),
                })
                self._trigger_log.append(triggered[-1])
        return triggered

    def process_audio(self, transcript: str) -> list:
        """Process audio transcript and trigger emotions."""
        triggered = []
        words = transcript.lower().split()
        for word in words:
            stimulus = self.AUDIO_TRIGGERS.get(word)
            if stimulus and self.emotion:
                self.emotion.apply_stimulus(stimulus)
                triggered.append({
                    "source": "auditory",
                    "word": word,
                    "stimulus": stimulus,
                    "emotion_after": self.emotion.current_emotion(),
                })
                self._trigger_log.append(triggered[-1])
        return triggered

    def process_surprise(self, surprise_level: int) -> list:
        """High surprise triggers novelty/alertness."""
        triggered = []
        if surprise_level > 3 and self.emotion:
            self.emotion.apply_stimulus("novelty")
            triggered.append({
                "source": "surprise",
                "level": surprise_level,
                "stimulus": "novelty",
                "emotion_after": self.emotion.current_emotion(),
            })
        return triggered

    def get_trigger_history(self, last_n: int = 20) -> list:
        return self._trigger_log[-last_n:]


class PerceptionLoop:
    """
    The complete sensory perception loop.

    Integrates: Eyes + Ears + Mouth + Prediction + Binding + Emotion

    Each cycle:
    1. PREDICT what senses should detect
    2. OBSERVE actual sensory input (eyes + ears)
    3. COMPARE prediction to reality (surprise detection)
    4. ATTEND to surprising stimuli only
    5. BIND multimodal observations into KASM vectors
    6. GROUND emotions in sensory triggers
    7. SPEAK response if needed (mouth)
    """

    def __init__(self, eyes=None, ears=None, mouth=None,
                 emotion=None, kasm=None, kernel=None, lexicon=None):
        self.eyes = eyes
        self.ears = ears
        self.mouth = mouth
        self.prediction = SensoryPrediction()
        self.binder = MultimodalBinder(kasm)
        self.grounding = EmotionGrounding(emotion)
        self.kernel = kernel
        self.lexicon = lexicon

        self._cycle_count = 0
        self._events = []
        self._running = False

    def perceive_once(self, verbose: bool = False) -> dict:
        """Run one perception cycle."""
        self._cycle_count += 1
        result = {
            "cycle": self._cycle_count,
            "visual": None,
            "audio": None,
            "surprise": 0,
            "emotions_triggered": [],
            "grounded_concepts": [],
        }

        # EYES
        if self.eyes:
            try:
                detections = self.eyes.see_webcam_frame()
                visual_surprise = self.prediction.observe_visual(detections)
                result["visual"] = visual_surprise

                # Bind visual concepts
                for det in detections:
                    self.binder.bind_visual(det.get("label", "unknown"))

                # Emotion grounding from vision
                emotions = self.grounding.process_visual(detections)
                result["emotions_triggered"].extend(emotions)

                if verbose and visual_surprise["is_surprising"]:
                    print("  [EYES] Surprise! New: %s" % visual_surprise["new_objects"])

            except Exception as e:
                if verbose:
                    print("  [EYES] Error: %s" % str(e)[:50])

        # EARS
        if self.ears:
            try:
                transcript_result = self.ears.listen_from_mic(duration_sec=2)
                if transcript_result and transcript_result.get("text"):
                    text = transcript_result["text"]
                    audio_surprise = self.prediction.observe_audio(text)
                    result["audio"] = audio_surprise

                    # Bind auditory concepts
                    for word in text.split():
                        if len(word) > 3:
                            self.binder.bind_auditory(word.lower())

                    # Emotion grounding from audio
                    emotions = self.grounding.process_audio(text)
                    result["emotions_triggered"].extend(emotions)

                    if verbose and audio_surprise["is_surprising"]:
                        print("  [EARS] Heard: '%s'" % text[:50])

            except Exception as e:
                if verbose:
                    print("  [EARS] Error: %s" % str(e)[:50])

        # Surprise-triggered emotion
        total_surprise = (
            (result["visual"]["surprise_level"] if result["visual"] else 0) +
            (result["audio"]["surprise_level"] if result["audio"] else 0)
        )
        result["surprise"] = total_surprise
        surprise_emotions = self.grounding.process_surprise(total_surprise)
        result["emotions_triggered"].extend(surprise_emotions)

        # Grounded concepts update
        result["grounded_concepts"] = self.binder.get_all_grounded()

        self._events.append(result)
        return result

    def get_status(self) -> dict:
        return {
            "cycles": self._cycle_count,
            "has_eyes": self.eyes is not None,
            "has_ears": self.ears is not None,
            "has_mouth": self.mouth is not None,
            "grounded_concepts": len(self.binder.get_all_grounded()),
            "emotion_triggers": len(self.grounding._trigger_log),
            "visual_history": len(self.prediction._visual_history),
            "audio_history": len(self.prediction._audio_history),
        }
