import streamlit as st
import sys
import os
import traceback

st.set_page_config(page_title="KOS V4.0", page_icon="brain",
                   layout="centered", initial_sidebar_state="expanded")

# Load secrets for Streamlit Cloud deployment
if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Show boot progress
boot_status = st.empty()
boot_status.info("Step 1/4: Downloading NLTK data...")

try:
    import nltk
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('wordnet', quiet=True)
    boot_status.info("Step 2/4: Importing KOS core...")

    from kos_core_v4 import (KOSKernel, KOSDaemonV4, ASTDriver, VisionDriver,
                              AutonomousForager, KASMCompiler)
    boot_status.info("Step 3/4: Importing lexicon & router...")

    from kos.lexicon import KASMLexicon
    from kos.router import KOSShell
    from kos.drivers.text import TextDriver
    boot_status.info("Step 4/4: Boot complete!")
    boot_status.empty()
    BOOT_OK = True
except Exception as e:
    boot_status.error(f"Boot failed: {e}\n\n```\n{traceback.format_exc()}\n```")
    BOOT_OK = False

import threading
import time

if not BOOT_OK:
    st.stop()

st.markdown("""
<style>
    .stApp { background-color: #212121; color: #ececf1; }
    #MainMenu, footer {visibility: hidden;}

    /* Keep sidebar toggle arrow always visible */
    [data-testid="stSidebarCollapsedControl"] { visibility: visible !important; display: block !important; }
    [data-testid="collapsedControl"] { visibility: visible !important; display: block !important; }

    /* Force ALL text white across the entire app */
    .stApp, .stApp * { color: #ececf1 !important; }

    /* Sidebar dark background + white text */
    [data-testid="stSidebar"] { background-color: #171717; }
    [data-testid="stSidebar"] * { color: #ececf1 !important; }

    /* Metric labels and values */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #ececf1 !important; }

    /* Input fields */
    .stTextInput>div>div>input {
        background-color: #2f2f2f; color: white !important; border-radius: 8px;
    }
    .stTextArea textarea {
        background-color: #2f2f2f; color: white !important; border-radius: 8px;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea { color: #1a1a1a !important; }

    /* Sidebar section headers */
    .stSidebar .stMarkdown p, .stSidebar .stMarkdown h3 { color: #ececf1 !important; }

    /* Buttons */
    .stButton>button {
        background-color: #2f2f2f; color: #ececf1 !important;
        border: 1px solid #444;
    }
    .stButton>button:hover { background-color: #444; }
</style>
""", unsafe_allow_html=True)

if 'kernel' not in st.session_state:
    st.session_state.kernel = KOSKernel()
    st.session_state.lexicon = KASMLexicon()
    st.session_state.shell = KOSShell(st.session_state.kernel,
                                       st.session_state.lexicon)
    st.session_state.daemon = KOSDaemonV4(st.session_state.kernel)
    st.session_state.ast = ASTDriver()
    st.session_state.driver = TextDriver(st.session_state.kernel, st.session_state.lexicon)
    st.session_state.forager = AutonomousForager(st.session_state.kernel, st.session_state.driver)
    st.session_state.kasm = KASMCompiler(st.session_state.kernel)

    # AUTO-LOAD: Restore brain from disk if a saved state exists
    node_count = 0
    if os.path.exists("memory.kos"):
        st.session_state.kernel.load_brain("memory.kos")
        if os.path.exists("lexicon.kos"):
            import pickle
            with open("lexicon.kos", "rb") as f:
                data = pickle.load(f)
                st.session_state.lexicon.uuid_to_word = data["u2w"]
                st.session_state.lexicon.word_to_uuid = data["w2u"]
                st.session_state.lexicon.sound_to_uuids = data["snd"]
        node_count = len(st.session_state.kernel.nodes)

    boot_msg = ("KOS V4.0 Online. Awaiting syntax, math, code, "
                "or semantic queries.")
    if node_count > 0:
        boot_msg += f"\n\n**Auto-loaded {node_count} nodes from disk.**"

    st.session_state.messages = [{
        "role": "assistant",
        "content": boot_msg
    }]

    def background_dream():
        while True:
            time.sleep(300)
            st.session_state.daemon.run_cycle()
    threading.Thread(target=background_dream, daemon=True).start()

with st.sidebar:
    st.markdown("### \U0001f9ec KOS V4.0 Control")
    st.metric("Nodes", len(st.session_state.kernel.nodes))
    st.divider()

    st.markdown("**Autonomous Forager**")
    url = st.text_input("Paste URL to assimilate:")
    if st.button("Forage Web", use_container_width=True):
        if url:
            with st.spinner("Scraping, filtering, and mapping topology..."):
                res = st.session_state.forager.forage(url)

                # Register foraged nodes in the lexicon so Shell can find them
                if "[REJECTED]" not in res and "[ERROR]" not in res:
                    import jellyfish
                    for nid in st.session_state.kernel.nodes:
                        nid_low = nid.lower()
                        # Only register if BOTH word→uuid AND uuid→word are missing
                        # This prevents overwriting good TextDriver entries
                        if nid_low not in st.session_state.lexicon.word_to_uuid \
                                and nid not in st.session_state.lexicon.uuid_to_word:
                            st.session_state.lexicon.word_to_uuid[nid_low] = nid
                            st.session_state.lexicon.uuid_to_word[nid] = nid_low
                            h = jellyfish.metaphone(nid_low)
                            st.session_state.lexicon._add_to_index(h, nid)

                # AUTO-SAVE after successful forage
                st.session_state.kernel.save_brain("memory.kos")
                import pickle
                with open("lexicon.kos", "wb") as f:
                    pickle.dump({
                        "u2w": st.session_state.lexicon.uuid_to_word,
                        "w2u": st.session_state.lexicon.word_to_uuid,
                        "snd": st.session_state.lexicon.sound_to_uuids
                    }, f)

                # INJECT THE FORAGER REPORT DIRECTLY INTO THE PERMANENT CHAT HISTORY!
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**Autonomous Forager Report:**\n`{url}`\n\n> {res}\n\n*Brain auto-saved to disk.*"
                })
            st.rerun()

    st.markdown("**KASM Compiler**")
    kasm_code = st.text_area("Write Machine Topology:")
    if st.button("Compile KASM"):
        st.session_state.kasm.compile(kasm_code)
        st.toast("Topological payload written to core.")

    st.divider()
    st.markdown("#### Graph Memory")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save", use_container_width=True):
            st.session_state.kernel.save_brain("memory.kos")
            import pickle
            with open("lexicon.kos", "wb") as f:
                pickle.dump({
                    "u2w": st.session_state.lexicon.uuid_to_word,
                    "w2u": st.session_state.lexicon.word_to_uuid,
                    "snd": st.session_state.lexicon.sound_to_uuids
                }, f)
            st.toast("Brain & Lexicon saved to disk!")

    with col2:
        if st.button("Load", use_container_width=True):
            if os.path.exists("memory.kos"):
                st.session_state.kernel.load_brain("memory.kos")
                if os.path.exists("lexicon.kos"):
                    import pickle
                    with open("lexicon.kos", "rb") as f:
                        data = pickle.load(f)
                        st.session_state.lexicon.uuid_to_word = data["u2w"]
                        st.session_state.lexicon.word_to_uuid = data["w2u"]
                        st.session_state.lexicon.sound_to_uuids = data["snd"]
                st.toast("Loaded successfully!")
                st.rerun()

st.title("Knowledge Operating System")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input(
    "E.g., 'simulate dropping silicon constraints', or 'prove false'"
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing Logic..."):
            # V4 Triage Router
            if "prove" in prompt.lower():
                from kos_core_v4 import LogicProver
                ans = (f"**Z3 Prover:** "
                       f"{LogicProver().prove(prompt)}")
            elif "simulate" in prompt.lower():
                # Simulated drop in traditional silicon
                ans = (f"**What-If Simulation:** "
                       f"{st.session_state.kernel.simulate({'silicon': -5.0})}")
            else:
                # ==========================================
                # THE FIX: USE THE SHELL & WEAVER!
                # ==========================================
                ans = st.session_state.shell.chat(prompt)

            st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
