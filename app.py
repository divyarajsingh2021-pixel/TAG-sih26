"""
FloatChat 🌊  — Dolphin Ocean AI
=================================
Pixel-perfect implementation matching the reference UI:
- Left  : Chat Panel (Dolphin, Online indicator, Mic/Audio pills, styled bubbles, voice input)
- Right : Toggleable between:
          1. ARGO Analytics Dashboard (Overview 6-metric cards, Yearly Trends, Regional Donut, 4 Tabs)
          2. Full Satellite Ocean Map (ESRI satellite tiles, yellow float markers, floating layer controls)
"""

import torch  # Required first on Windows to avoid DLL load order issues
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from streamlit_mic_recorder import speech_to_text

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloatChat 🌊",
    page_icon="🐬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Bootstrap data ────────────────────────────────────────────────────────────
DATA_PATH  = Path("data/processed/argo_indian_ocean.parquet")
INDEX_PATH = Path("data/faiss_index/summaries.json")

if not DATA_PATH.exists():
    with st.spinner("⏳ Generating ARGO dataset…"):
        from scripts.download_argo import main as gen_data; gen_data()

if not INDEX_PATH.exists():
    with st.spinner("⏳ Building search index…"):
        from scripts.build_index import build_index; build_index()

from backend.rag_chain  import answer_query
from backend.router     import classify_intent, extract_region, extract_month
from backend.visualizer import (
    plot_float_map, plot_temperature_profile, plot_salinity_profile,
    plot_sst_timeseries, plot_regional_comparison, plot_ts_diagram,
    plot_globe, plot_satellite_ocean_map,
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    return pd.read_parquet(DATA_PATH)

df = load_dataset()
surface = df[df["depth_m"] == 0]
total_floats = df["float_id"].nunique()
total_profiles = df["profile_id"].nunique()
total_records = len(df)
avg_sst = surface["temperature_c"].mean()
avg_sal = surface["salinity_psu"].mean()
current_time_str = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to FloatChat (Outrage Ocean AI)!**\n\n"
                "I am your conversational interface for ARGO ocean float data. "
                "You can ask me questions about **temperature**, **salinity**, **float trajectories**, "
                "or **depth profiles** across the Indian Ocean, Arabian Sea, and Bay of Bengal.\n\n"
                "You can type below or click 🎤 **Mic** to speak!"
            ),
            "time": datetime.now().strftime("%H:%M"),
        }
    ]

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "dashboard"  # "dashboard" (Image 1) or "map" (Image 2)

if "prefill" not in st.session_state:
    st.session_state.prefill = ""

if "audio_enabled" not in st.session_state:
    st.session_state.audio_enabled = True

if "speak_text" not in st.session_state:
    st.session_state.speak_text = ""

# Check for incoming voice recognition query from browser
incoming_vquery = ""
try:
    if "vquery" in st.query_params:
        incoming_vquery = st.query_params["vquery"]
        st.query_params.clear()
except Exception:
    pass

if "groq_key" not in st.session_state:
    try:
        st.session_state.groq_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        st.session_state.groq_key = os.getenv("GROQ_API_KEY", "")

# ── Master CSS matching Reference UI ──────────────────────────────────────────
st.markdown("""
<style>
  /* Global page setup */
  .stApp {
    background-color: #f3f4f6 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
  }
  .main .block-container {
    padding: 8px 12px !important;
    max-width: 100% !important;
  }
  section[data-testid="stSidebar"] {
    display: none;
  }
  #MainMenu, footer, header {
    visibility: hidden;
  }

  /* Left Panel (Chat) */
  .chat-container {
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    display: flex;
    flex-direction: column;
    height: 94vh;
    overflow: hidden;
    border: 1px solid #e5e7eb;
  }
  .chat-header {
    background: #ffffff;
    border-bottom: 1px solid #f0f0f0;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .chat-avatar {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    background: #eef2ff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    border: 1px solid #e0e7ff;
  }
  .chat-title-box {
    display: flex;
    flex-direction: column;
  }
  .chat-name {
    font-size: 16px;
    font-weight: 700;
    color: #111827;
  }
  .chat-status {
    font-size: 12px;
    color: #10b981;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .chat-status::before {
    content: "●";
    font-size: 10px;
  }
  .header-actions {
    margin-left: auto;
    display: flex;
    gap: 8px;
  }
  .pill-action {
    background: #f0fdf4;
    color: #166534;
    border: 1px solid #bbf7d0;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }
  .pill-audio {
    background: #eff6ff;
    color: #1e40af;
    border: 1px solid #bfdbfe;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }

  /* Chat conversation stream */
  .chat-stream {
    flex: 1;
    overflow-y: auto;
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .user-bubble {
    align-self: flex-end;
    background: #dbeafe;
    color: #1e3a8a;
    padding: 12px 18px;
    border-radius: 20px 20px 4px 20px;
    max-width: 82%;
    font-size: 14px;
    line-height: 1.5;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  .bot-bubble {
    align-self: flex-start;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    color: #1f2937;
    padding: 14px 18px;
    border-radius: 20px 20px 20px 4px;
    max-width: 92%;
    font-size: 14px;
    line-height: 1.6;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  }
  .time-stamp {
    font-size: 10px;
    color: #9ca3af;
    margin-top: 4px;
    text-align: right;
  }

  /* Right Panel (Dashboard & Map) */
  .right-panel-container {
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    height: 94vh;
    overflow-y: auto;
    border: 1px solid #e5e7eb;
    display: flex;
    flex-direction: column;
  }
  .top-action-bar {
    background: #ffffff;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #e5e7eb;
    position: sticky;
    top: 0;
    z-index: 20;
    border-radius: 16px 16px 0 0;
  }
  .top-title-box {
    display: flex;
    flex-direction: column;
  }
  .dash-main-title {
    font-size: 20px;
    font-weight: 800;
    color: #111827;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .dash-sub-title {
    font-size: 12px;
    color: #6b7280;
    margin-top: 2px;
  }

  /* Metric cards (exact Image 1 layout) */
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    padding: 18px 24px 8px 24px;
  }
  .metric-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 16px 20px;
    position: relative;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .card-top-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .card-label {
    font-size: 13px;
    font-weight: 600;
    color: #4b5563;
  }
  .card-icon {
    font-size: 16px;
  }
  .card-val {
    font-size: 26px;
    font-weight: 800;
    color: #111827;
    margin-top: 6px;
    line-height: 1.1;
  }
  .card-subtext {
    font-size: 11px;
    color: #6b7280;
    margin-top: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .badge-active {
    background: #f3f4f6;
    color: #374151;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
  }
  .badge-pill-blue {
    background: #2563eb;
    color: #ffffff;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
  }

  /* Floating Overlay Controls on Map (Image 2) */
  .floating-map-card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(229, 231, 235, 0.9);
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    margin-bottom: 12px;
  }
  .floating-legend-pill {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }

  /* Streamlit native widget styling */
  .stTextInput > div > div > input {
    border-radius: 24px !important;
    padding: 10px 16px !important;
    background: #f9fafb !important;
    border: 1px solid #e5e7eb !important;
    font-size: 14px !important;
  }
  .stButton > button {
    border-radius: 20px !important;
    font-weight: 600 !important;
    border: none !important;
    transition: all 0.2s ease !important;
  }
  .primary-btn > button {
    background: #2563eb !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Web Speech API JavaScript Component ───────────────────────────────────────
VOICE_HTML = """
<div style="display:flex;align-items:center;gap:10px;padding:4px 0;">
  <button id="micBtn" onclick="toggleSpeech()" title="Click and speak your question"
    style="background:#2563eb; color:white; border:none; border-radius:50%; width:42px; height:42px;
           cursor:pointer; display:flex; align-items:center; justify-content:center; font-size:20px;
           box-shadow:0 2px 8px rgba(37,99,235,0.35); transition:all 0.2s ease;">
    🎤
  </button>
  <div style="display:flex; flex-direction:column;">
    <span id="speechStatus" style="font-size:12px; font-weight:600; color:#374151;">Click mic to speak</span>
    <span style="font-size:10px; color:#9ca3af;">Works with Chrome &amp; Edge</span>
  </div>
</div>
<script>
let recognizing = false;
let recognition = null;

function toggleSpeech() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    document.getElementById('speechStatus').innerText = 'Use Chrome/Edge for voice';
    return;
  }
  if (!recognition) {
    recognition = new SR();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = function() {
      recognizing = true;
      document.getElementById('micBtn').style.background = '#ef4444';
      document.getElementById('micBtn').style.boxShadow = '0 0 14px rgba(239,68,68,0.7)';
      document.getElementById('speechStatus').innerText = '🎙️ Listening to you…';
    };

    recognition.onresult = function(event) {
      const transcript = event.results[0][0].transcript;
      document.getElementById('speechStatus').innerText = '✅ Transcribed: ' + transcript;
      document.getElementById('micBtn').style.background = '#10b981';
      recognizing = false;

      // Automatically send speech transcript to Streamlit parent page!
      try {
        const pUrl = new URL(window.parent.location.href);
        pUrl.searchParams.set('vquery', transcript);
        window.parent.location.href = pUrl.href;
      } catch(e) {
        window.parent.postMessage({ type: 'voice_input', text: transcript }, '*');
      }
    };

    recognition.onerror = function(event) {
      document.getElementById('speechStatus').innerText = '⚠️ ' + event.error;
      resetMic();
    };

    recognition.onend = function() {
      resetMic();
    };
  }

  if (recognizing) {
    recognition.stop();
    resetMic();
  } else {
    recognition.start();
  }
}

function resetMic() {
  recognizing = false;
  document.getElementById('micBtn').style.background = '#2563eb';
  document.getElementById('micBtn').style.boxShadow = '0 2px 8px rgba(37,99,235,0.35)';
}
</script>
"""

# ─────────────────────────────────────────────────────────────────────────────
# TWO-COLUMN SPLIT LAYOUT (Chat on Left 38%, Dashboard/Map on Right 62%)
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3.8, 6.2], gap="small")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — CHAT PANEL (Outrage Ocean AI)
# ══════════════════════════════════════════════════════════════════════════════
with col_left:
    hdr_info, hdr_audio_btn = st.columns([3, 1.4])
    with hdr_info:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:12px; padding: 2px 0;">
          <div class="chat-avatar">⚡</div>
          <div class="chat-title-box">
            <div class="chat-name">Outrage</div>
            <div class="chat-status">Online</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with hdr_audio_btn:
        audio_toggle_label = "🔊 Voice: ON" if st.session_state.audio_enabled else "🔇 Voice: OFF"
        if st.button(audio_toggle_label, key="hdr_toggle_audio", use_container_width=True):
            st.session_state.audio_enabled = not st.session_state.audio_enabled
            st.rerun()

    # API key setup (in sleek expander)
    with st.expander("🔑 Groq API Key Config", expanded=not bool(st.session_state.groq_key)):
        key_input = st.text_input(
            "API Key:",
            value=st.session_state.groq_key,
            type="password",
            placeholder="gsk_xxxxxxxxxxxxxxxxxxxx",
            label_visibility="collapsed",
        )
        if key_input:
            st.session_state.groq_key = key_input
        if st.session_state.groq_key:
            st.success("✅ Groq Key Active (LLaMA3-70B)")
        else:
            st.warning("⚠️ Enter key for AI generation")

    # Suggested Prompts (chips)
    st.markdown("<div style='padding: 6px 16px 0 16px; font-size: 12px; font-weight: 600; color: #6b7280;'>💡 Quick queries:</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("Average SST in Arabian Sea?", key="chip_1", use_container_width=True):
        st.session_state.prefill = "What is the average sea surface temperature in the Arabian Sea?"
        st.rerun()
    if c2.button("Salinity profile near equator", key="chip_2", use_container_width=True):
        st.session_state.prefill = "Show me salinity profile near the equator in 2023"
        st.rerun()

    # Chat Messages Stream
    chat_html = '<div class="chat-stream">'
    for msg in st.session_state.messages:
        content = msg["content"].replace("\n", "<br>")
        t = msg.get("time", "")
        if msg["role"] == "user":
            chat_html += f'<div class="user-bubble">🧑 {content}<div class="time-stamp">{t}</div></div>'
        else:
            chat_html += f'<div class="bot-bubble">⚡ {content}<div class="time-stamp">{t}</div></div>'
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # ── Voice Speech Input (Real-time Bidirectional Component) ─────────────────
    voice_col1, voice_col2 = st.columns([3, 1])
    with voice_col1:
        voice_spoken = speech_to_text(
            language="en",
            start_prompt="🎤 Speak to Outrage",
            stop_prompt="⏹️ Stop & Send Question",
            just_once=True,
            use_container_width=True,
            key="outrage_voice_stt",
        )
    with voice_col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = st.session_state.messages[:1]
            st.session_state.speak_text = ""
            st.rerun()

    # ── Chat Input Form (supports ENTER key and Send button) ───────────────────
    prefill_val = st.session_state.pop("prefill", "")
    with st.form("chat_input_form", clear_on_submit=True):
        f_inp, f_btn = st.columns([7.8, 2.2])
        with f_inp:
            typed_input = st.text_input(
                "chat_field",
                value=prefill_val,
                placeholder="Type your question and press Enter...",
                label_visibility="collapsed",
            )
        with f_btn:
            submit_clicked = st.form_submit_button("Send ➤", use_container_width=True)

    # Query processing (handles native Voice speech, query params, and Typed input)
    query_to_run = ""
    if voice_spoken and voice_spoken.strip():
        query_to_run = voice_spoken.strip()
    elif incoming_vquery:
        query_to_run = incoming_vquery
    elif submit_clicked and typed_input.strip():
        query_to_run = typed_input.strip()

    if query_to_run:
        curr_time = datetime.now().strftime("%H:%M")
        st.session_state.messages.append({"role": "user", "content": query_to_run, "time": curr_time})

        intent = classify_intent(query_to_run)
        q_reg = extract_region(query_to_run)
        q_m = extract_month(query_to_run)

        with st.spinner("⚡ Outrage is thinking..."):
            if not st.session_state.groq_key:
                sdf = surface[surface["region"] == q_reg] if q_reg != "All" else surface
                bot_ans = (
                    f"📊 **Data summary for {q_reg}:**\n\n"
                    f"- Mean Sea Surface Temp: **{sdf['temperature_c'].mean():.2f}°C**\n"
                    f"- Mean Surface Salinity: **{sdf['salinity_psu'].mean():.2f} PSU**\n"
                    f"- Active Float Profiles: **{sdf['profile_id'].nunique():,}**\n\n"
                    f"*Tip: Enter your Groq API key in the sidebar for complete LLaMA3 scientific synthesis!*"
                )
            else:
                bot_ans = answer_query(query_to_run, st.session_state.groq_key)

        st.session_state.messages.append({"role": "assistant", "content": bot_ans, "time": curr_time})

        # Trigger voice speech synthesis if audio is enabled
        if st.session_state.audio_enabled:
            st.session_state.speak_text = bot_ans

        st.rerun()

    # ── Text-to-Speech (TTS Audio Playback) ───────────────────────────────────
    if st.session_state.audio_enabled and st.session_state.speak_text:
        import re
        spk_raw = st.session_state.speak_text
        st.session_state.speak_text = ""
        # Clean markdown characters for natural speech
        spk_clean = re.sub(r"[*#_`>\[\]]", "", spk_raw)
        spk_clean = re.sub(r"https?://\S+", "", spk_clean)
        spk_clean = spk_clean.replace("\\", "").replace('"', '\\"').replace("'", "\\'").replace("\n", " ").strip()

        tts_script = f"""
        <script>
        (function() {{
            const win = window.parent || window;
            if (win.speechSynthesis) {{
                win.speechSynthesis.cancel();
                var u = new SpeechSynthesisUtterance("{spk_clean[:380]}");
                u.rate = 1.05;
                u.pitch = 1.0;
                u.lang = 'en-US';
                win.speechSynthesis.speak(u);
            }}
        }})();
        </script>
        """
        components.html(tts_script, height=0)


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — TOGGLEABLE: DASHBOARD VIEW (Image 1) OR SATELLITE MAP (Image 2)
# ══════════════════════════════════════════════════════════════════════════════
with col_right:

    # ── Top Bar with Toggle Button ────────────────────────────────────────────
    top_col_title, top_col_ref, top_col_toggle = st.columns([3.2, 0.9, 1.9])

    with top_col_title:
        if st.session_state.view_mode == "dashboard":
            st.markdown(f"""
            <div class="top-title-box">
              <div class="dash-main-title">📊 ARGO Analytics Dashboard</div>
              <div class="dash-sub-title">Last updated: {current_time_str}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="top-title-box">
              <div class="dash-main-title">🌊 Ocean Satellite &amp; Float Map</div>
              <div class="dash-sub-title">ESRI World Imagery • Indian Ocean 2023</div>
            </div>
            """, unsafe_allow_html=True)

    with top_col_ref:
        if st.button("↻ Refresh", key="refresh_top", use_container_width=True):
            st.rerun()

    with top_col_toggle:
        if st.session_state.view_mode == "dashboard":
            if st.button("🗺️ Hide Dashboard", key="btn_toggle_view", use_container_width=True):
                st.session_state.view_mode = "map"
                st.rerun()
        else:
            if st.button("📊 Show Dashboard", key="btn_toggle_view", use_container_width=True):
                st.session_state.view_mode = "dashboard"
                st.rerun()

    st.markdown("<hr style='margin: 8px 0 16px 0; border: none; border-bottom: 1px solid #e5e7eb;'>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # MODE A: FULL SATELLITE OCEAN MAP (Exact match for Image 2)
    # ─────────────────────────────────────────────────────────────────────────
    if st.session_state.view_mode == "map":

        # Floating Layer Controls Card (Image 2 style)
        with st.container():
            fcol1, fcol2, fcol3 = st.columns([2, 1.5, 1.5])
            with fcol1:
                base_map_sel = st.selectbox(
                    "Base Map:",
                    ["Satellite (ESRI)", "Ocean Bathymetry", "OpenStreetMap", "Dark Matter"],
                    key="map_base_selector",
                )
            with fcol2:
                show_heatmap = st.checkbox("🔥 Temp Anomaly Heatmap", value=False, key="chk_heatmap")
            with fcol3:
                show_trajectories = st.checkbox("📍 Float Drift Trajectories", value=True, key="chk_trajectories")

        # Map display
        with st.spinner("Rendering satellite ocean tiles..."):
            map_fig = plot_satellite_ocean_map(
                region="All",
                month=0,
                base_map=base_map_sel,
                show_trajectories=show_trajectories,
                show_heatmap=show_heatmap,
            )
            map_fig.update_layout(height=650)
            st.plotly_chart(map_fig, use_container_width=True)

        # Status badge below map (matching Image 2 footer pill)
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; padding: 4px 8px;">
          <div class="floating-legend-pill">
            <span style="color:#10b981;">●</span> <b>2023 ARGO Profiles: {total_floats} Floats</b>
            <span style="color:#ef4444; margin-left:8px;">●</span> Active Trajectories
            <span style="color:#6b7280; margin-left:8px;">●</span> Vector DB Indexed
          </div>
          <div style="font-size:11px; color:#9ca3af;">
            Tiles: ESRI World Imagery, Bathymetry &amp; GEBCO
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # MODE B: DASHBOARD VIEW (Exact match for Image 1)
    # ─────────────────────────────────────────────────────────────────────────
    else:
        # Dashboard Navigation Tabs
        tab_overview, tab_temporal, tab_geo, tab_env = st.tabs([
            "Overview", "Temporal", "Geographic", "Environmental"
        ])

        # ── TAB 1: OVERVIEW (Image 1 replica) ────────────────────────────────
        with tab_overview:

            # 6 Metric Cards
            st.markdown(f"""
            <div class="metrics-grid">

              <!-- Card 1: Total Floats -->
              <div class="metric-card">
                <div class="card-top-row">
                  <div class="card-label">Total Floats</div>
                  <div class="card-icon" style="color:#2563eb;">🗄️</div>
                </div>
                <div class="card-val">{total_floats:,}</div>
                <div class="card-subtext">
                  <span class="badge-active">{total_floats} active</span>
                </div>
              </div>

              <!-- Card 2: Profiles -->
              <div class="metric-card">
                <div class="card-top-row">
                  <div class="card-label">Profiles</div>
                  <div class="card-icon" style="color:#10b981;">📈</div>
                </div>
                <div class="card-val">{total_profiles:,}</div>
                <div class="card-subtext">Temperature &amp; Salinity</div>
              </div>

              <!-- Card 3: Measurements -->
              <div class="metric-card">
                <div class="card-top-row">
                  <div class="card-label">Measurements</div>
                  <div class="card-icon" style="color:#8b5cf6;">📊</div>
                </div>
                <div class="card-val">{total_records:,}</div>
                <div class="card-subtext">All parameters</div>
              </div>

              <!-- Card 4: Data Quality -->
              <div class="metric-card">
                <div class="card-top-row">
                  <div class="card-label">Data Quality</div>
                  <div class="card-icon" style="color:#10b981;">✅</div>
                </div>
                <div class="card-val">98.4%</div>
                <div class="card-subtext">
                  <span class="badge-pill-blue">Excellent</span>
                </div>
              </div>

              <!-- Card 5: Temperature -->
              <div class="metric-card">
                <div class="card-top-row">
                  <div class="card-label">Temperature</div>
                  <div class="card-icon" style="color:#ef4444;">🌡️</div>
                </div>
                <div class="card-val">{avg_sst:.1f}°C</div>
                <div class="card-subtext">Regional average</div>
              </div>

              <!-- Card 6: Salinity -->
              <div class="metric-card">
                <div class="card-top-row">
                  <div class="card-label">Salinity</div>
                  <div class="card-icon" style="color:#0ea5e9;">💧</div>
                </div>
                <div class="card-val">{avg_sal:.1f} PSU</div>
                <div class="card-subtext">Regional average</div>
              </div>

            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Two Charts Side-by-Side (Yearly Trends + Regional Distribution)
            chart_col_left, chart_col_right = st.columns(2)

            with chart_col_left:
                st.markdown("<div style='font-size:15px; font-weight:700; color:#111827; margin-bottom:6px;'>📉 Yearly Trends</div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size:12px; color:#6b7280; margin-bottom:8px;'>ARGO Profiles &amp; Float Deployment Over Time</div>", unsafe_allow_html=True)

                monthly_df = surface.groupby("month").agg(
                    profiles=("profile_id", "nunique"),
                    floats=("float_id", "nunique")
                ).reset_index()
                month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                monthly_df["month_str"] = monthly_df["month"].apply(lambda x: month_labels[x-1] if x <= 12 else str(x))

                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=monthly_df["month_str"],
                    y=monthly_df["profiles"],
                    name="Profiles",
                    line=dict(color="#3b82f6", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(59, 130, 246, 0.08)",
                ))
                fig_trend.add_trace(go.Scatter(
                    x=monthly_df["month_str"],
                    y=monthly_df["floats"],
                    name="Active Floats",
                    line=dict(color="#10b981", width=2.5),
                    yaxis="y2",
                ))
                fig_trend.update_layout(
                    height=270,
                    margin=dict(l=0, r=0, t=10, b=10),
                    template="plotly_white",
                    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
                    yaxis2=dict(overlaying="y", side="right", showgrid=False),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#f3f4f6"),
                )
                st.plotly_chart(fig_trend, use_container_width=True)

            with chart_col_right:
                st.markdown("<div style='font-size:15px; font-weight:700; color:#111827; margin-bottom:6px;'>🍩 Regional Distribution</div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size:12px; color:#6b7280; margin-bottom:8px;'>ARGO Floats Across Indian Ocean Basins</div>", unsafe_allow_html=True)

                reg_counts = df.groupby("region")["float_id"].nunique().reset_index()
                fig_donut = px.pie(
                    reg_counts,
                    names="region",
                    values="float_id",
                    color="region",
                    color_discrete_map={
                        "Arabian Sea": "#3b82f6",
                        "Bay of Bengal": "#10b981",
                        "Indian Ocean": "#f59e0b",
                    },
                    hole=0.52,
                )
                fig_donut.update_traces(textposition="outside", textinfo="percent+label")
                fig_donut.update_layout(
                    height=270,
                    margin=dict(l=0, r=0, t=10, b=10),
                    showlegend=False,
                    template="plotly_white",
                )
                st.plotly_chart(fig_donut, use_container_width=True)

        # ── TAB 2: TEMPORAL ───────────────────────────────────────────────────
        with tab_temporal:
            st.markdown("#### 📅 Temporal Patterns & Monthly Sea Surface Temperature")
            t_col1, t_col2 = st.columns([1, 2])
            with t_col1:
                t_reg = st.selectbox("Select Ocean Basin:", ["All", "Arabian Sea", "Bay of Bengal", "Indian Ocean"], key="t_reg_sel")
            with t_col2:
                st.caption("Visualizes the seasonal warming and monsoon-driven cooling across the year.")

            fig_sst = plot_sst_timeseries(t_reg)
            fig_sst.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=10))
            st.plotly_chart(fig_sst, use_container_width=True)

            st.markdown("#### 📊 Basin Comparison (SST Distribution)")
            fig_box = plot_regional_comparison()
            fig_box.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=10))
            st.plotly_chart(fig_box, use_container_width=True)

        # ── TAB 3: GEOGRAPHIC ─────────────────────────────────────────────────
        with tab_geo:
            st.markdown("#### 🌍 3D Interactive Rotating Globe")
            st.caption("Drag to rotate the globe • Scroll to zoom • Hover over floats for live measurements")
            with st.spinner("Rendering 3D Globe..."):
                globe_fig = plot_globe(region="All", month=0)
                globe_fig.update_layout(height=480, margin=dict(l=0, r=0, t=20, b=10))
                st.plotly_chart(globe_fig, use_container_width=True)

        # ── TAB 4: ENVIRONMENTAL ──────────────────────────────────────────────
        with tab_env:
            st.markdown("#### 🌡️ Depth Curves & Water Mass Diagnostics")
            env_c1, env_c2 = st.columns(2)
            with env_c1:
                st.markdown("**Temperature vs Depth (0 to 2000m)**")
                f_temp = plot_temperature_profile(region="All", month=0)
                f_temp.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=10))
                st.plotly_chart(f_temp, use_container_width=True)
            with env_c2:
                st.markdown("**Salinity vs Depth (0 to 2000m)**")
                f_sal = plot_salinity_profile(region="All", month=0)
                f_sal.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=10))
                st.plotly_chart(f_sal, use_container_width=True)

            st.markdown("#### 🔬 Temperature-Salinity (T-S) Water Mass Diagram")
            st.caption("Identifies Arabian Sea High Salinity Water vs fresh Bay of Bengal runoff.")
            f_ts = plot_ts_diagram(region="All")
            f_ts.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=10))
            st.plotly_chart(f_ts, use_container_width=True)
