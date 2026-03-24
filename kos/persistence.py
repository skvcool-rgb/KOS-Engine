"""
KOS Graph Persistence — Save/Load/Auto-save the knowledge graph.
"""

import os
import pickle
import threading
import time


class GraphPersistence:
    """Persist kernel graph and lexicon mappings to disk."""

    def save(self, kernel, lexicon, filepath=".cache/kos_graph.pkl"):
        """Pickle kernel nodes, provenance, contradictions, and lexicon dicts."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            "nodes": kernel.nodes,
            "provenance": kernel.provenance,
            "contradictions": kernel.contradictions,
            "word_to_uuid": lexicon.word_to_uuid,
            "uuid_to_word": lexicon.uuid_to_word,
            # Fix: persist phonetic indexes for typo recovery
            "sound_to_uuids": getattr(lexicon, 'sound_to_uuids', {}),
            "soundex_to_uuids": getattr(lexicon, 'soundex_to_uuids', {}),
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        size_bytes = os.path.getsize(filepath)
        node_count = len(kernel.nodes)
        edge_count = sum(len(n.connections) for n in kernel.nodes.values())
        print("[PERSISTENCE] Saved %d nodes, %d edges (%.1f KB) -> %s" % (
            node_count, edge_count, size_bytes / 1024, filepath))

    def load(self, kernel, lexicon, filepath=".cache/kos_graph.pkl"):
        """Restore kernel and lexicon state from pickle."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        kernel.nodes = data["nodes"]
        kernel.provenance = data["provenance"]
        kernel.contradictions = data["contradictions"]
        lexicon.word_to_uuid = data["word_to_uuid"]
        lexicon.uuid_to_word = data["uuid_to_word"]
        # Fix: restore phonetic indexes for typo recovery
        if "sound_to_uuids" in data:
            lexicon.sound_to_uuids = data["sound_to_uuids"]
        if "soundex_to_uuids" in data:
            lexicon.soundex_to_uuids = data["soundex_to_uuids"]

        node_count = len(kernel.nodes)
        edge_count = sum(len(n.connections) for n in kernel.nodes.values())
        size_bytes = os.path.getsize(filepath)
        print("[PERSISTENCE] Loaded %d nodes, %d edges (%.1f KB) <- %s" % (
            node_count, edge_count, size_bytes / 1024, filepath))

    def auto_save(self, kernel, lexicon, interval_sec=300):
        """Start a daemon thread that saves every interval_sec seconds."""
        def _loop():
            while True:
                time.sleep(interval_sec)
                try:
                    self.save(kernel, lexicon)
                except Exception as e:
                    print("[PERSISTENCE] Auto-save error: %s" % e)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        print("[PERSISTENCE] Auto-save started (every %ds)" % interval_sec)

    def exists(self, filepath=".cache/kos_graph.pkl"):
        """Check if a save file exists."""
        return os.path.exists(filepath)
