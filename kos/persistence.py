"""
KOS V7.0 — Brain Persistence (Rust Binary + Python Metadata)

Two files:
  .kos       — Rust binary (arena nodes, VSA vectors, edges, myelination, provenance)
  .kos.meta  — Python pickle (lexicon mappings, PCE cache, contradictions)

On save: Rust kernel writes binary, Python pickles metadata.
On load: Rust kernel reads binary, Python restores metadata, syncs mirror.
"""

import os
import pickle
import threading
import time


BRAIN_DIR = ".cache"
RUST_FILE = "kos_brain.kos"
META_FILE = "kos_brain.kos.meta"


class GraphPersistence:
    """Persist full KOS state: Rust arena + Python metadata."""

    def __init__(self, brain_dir=BRAIN_DIR):
        self.brain_dir = brain_dir
        self.rust_path = os.path.join(brain_dir, RUST_FILE)
        self.meta_path = os.path.join(brain_dir, META_FILE)

    def save(self, kernel, lexicon, pce=None, learner=None):
        """Save full brain state."""
        os.makedirs(self.brain_dir, exist_ok=True)

        rust_bytes = 0
        # Rust binary save (arena + edges + VSA + myelin + provenance)
        if kernel._rust is not None:
            try:
                rust_bytes = kernel._rust.save(self.rust_path)
            except Exception as e:
                print("[PERSISTENCE] Rust save error: %s" % e)

        # Python metadata (lexicon, PCE, contradictions, learning stats)
        meta = {
            "word_to_uuid": lexicon.word_to_uuid,
            "uuid_to_word": lexicon.uuid_to_word,
            "sound_to_uuids": getattr(lexicon, 'sound_to_uuids', {}),
            "soundex_to_uuids": getattr(lexicon, 'soundex_to_uuids', {}),
            "contradictions": kernel.contradictions,
            "current_tick": kernel.current_tick,
        }

        # PCE prediction cache
        if pce is not None:
            try:
                meta["pce_prediction_cache"] = getattr(pce, 'prediction_cache', {})
                meta["pce_stats"] = pce.get_stats()
            except Exception:
                pass

        # Learning stats
        if learner is not None:
            try:
                meta["learning_stats"] = learner.get_stats()
            except Exception:
                pass

        with open(self.meta_path, "wb") as f:
            pickle.dump(meta, f)

        meta_bytes = os.path.getsize(self.meta_path)
        node_count = kernel._rust.node_count() if kernel._rust else len(kernel.nodes)
        edge_count = kernel._rust.edge_count() if kernel._rust else 0
        print("[BRAIN] Saved %d nodes, %d edges (rust=%d KB, meta=%d KB) -> %s" % (
            node_count, edge_count,
            rust_bytes // 1024, meta_bytes // 1024,
            self.brain_dir))

    def load(self, kernel, lexicon, pce=None):
        """Load full brain state. Returns True if loaded, False if no file."""
        if not self.exists():
            return False

        # Load Rust binary
        if kernel._rust is not None and os.path.exists(self.rust_path):
            try:
                node_count = kernel._rust.load(self.rust_path)
                # Sync Python mirror from Rust
                self._sync_python_mirror(kernel)
                print("[BRAIN] Loaded %d nodes from Rust binary" % node_count)
            except Exception as e:
                print("[BRAIN] Rust load error: %s, trying pickle fallback" % e)
                return self._load_pickle_fallback(kernel, lexicon)

        # Load Python metadata
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "rb") as f:
                    meta = pickle.load(f)

                lexicon.word_to_uuid = meta.get("word_to_uuid", {})
                lexicon.uuid_to_word = meta.get("uuid_to_word", {})
                if "sound_to_uuids" in meta:
                    lexicon.sound_to_uuids = meta["sound_to_uuids"]
                if "soundex_to_uuids" in meta:
                    lexicon.soundex_to_uuids = meta["soundex_to_uuids"]
                kernel.contradictions = meta.get("contradictions", [])
                kernel.current_tick = meta.get("current_tick", 0)

                # Restore PCE cache
                if pce is not None and "pce_prediction_cache" in meta:
                    pce.prediction_cache = meta["pce_prediction_cache"]

            except Exception as e:
                print("[BRAIN] Meta load error: %s" % e)

        return True

    def _sync_python_mirror(self, kernel):
        """Rebuild Python kernel.nodes dict from Rust export."""
        if kernel._rust is None:
            return

        from .node import ConceptNode

        graph_data = kernel._rust.export_graph()
        kernel.nodes.clear()

        for node_id, edges in graph_data:
            node = ConceptNode(node_id)
            for edge_tuple in edges:
                if len(edge_tuple) == 4:
                    tgt_name, weight, myelin, edge_type = edge_tuple
                elif len(edge_tuple) == 3:
                    tgt_name, weight, myelin = edge_tuple
                    edge_type = 0
                else:
                    continue
                node.connections[tgt_name] = {
                    'w': weight,
                    'myelin': myelin,
                    'edge_type': edge_type,
                }
            kernel.nodes[node_id] = node

    def _load_pickle_fallback(self, kernel, lexicon):
        """Fallback: load from old pickle format."""
        old_path = os.path.join(self.brain_dir, "kos_graph.pkl")
        if not os.path.exists(old_path):
            return False
        try:
            with open(old_path, "rb") as f:
                data = pickle.load(f)
            kernel.nodes = data["nodes"]
            kernel.provenance = data.get("provenance", {})
            kernel.contradictions = data.get("contradictions", [])
            lexicon.word_to_uuid = data.get("word_to_uuid", {})
            lexicon.uuid_to_word = data.get("uuid_to_word", {})
            # Rebuild Rust from Python mirror
            if kernel._rust is not None:
                for nid, node in kernel.nodes.items():
                    kernel._rust.add_node(nid)
                    for tgt, data in node.connections.items():
                        w = data['w'] if isinstance(data, dict) else data
                        kernel._rust.add_connection(nid, tgt, float(w), None)
            print("[BRAIN] Loaded from pickle fallback")
            return True
        except Exception as e:
            print("[BRAIN] Pickle fallback error: %s" % e)
            return False

    def exists(self):
        """Check if a brain save exists."""
        return (os.path.exists(self.rust_path) or
                os.path.exists(self.meta_path) or
                os.path.exists(os.path.join(self.brain_dir, "kos_graph.pkl")))

    def auto_save(self, kernel, lexicon, pce=None, learner=None,
                  interval_sec=300):
        """Start daemon thread that saves brain every interval_sec seconds."""
        def _loop():
            while True:
                time.sleep(interval_sec)
                try:
                    self.save(kernel, lexicon, pce=pce, learner=learner)
                except Exception as e:
                    print("[BRAIN] Auto-save error: %s" % e)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        print("[BRAIN] Auto-save started (every %ds)" % interval_sec)

    def brain_size_kb(self):
        """Return total brain size in KB."""
        total = 0
        for path in [self.rust_path, self.meta_path]:
            if os.path.exists(path):
                total += os.path.getsize(path)
        return total / 1024
