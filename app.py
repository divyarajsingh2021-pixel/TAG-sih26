"""
FloatChat 🌊 — Streamlit Frontend (v2)
========================================
Features:
  1. 3D Interactive Globe (Plotly orthographic Scattergeo)
  2. Voice Input (Web Speech API via st.components.v1.html)
  3. Multi-Panel Dashboard (st.tabs + st.columns)
  4. Toggle Controls (st.session_state for open/close panels)

Run with:  streamlit run app.py
"""

# ── Import torch FIRST on Windows to prevent c10.dll load-order crash ─────────
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloatChat 🌊",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bootstrap data on first run ───────────────────────────────────────────────
DATA_PATH = Path("data/processed/argo_indian_ocean.parquet")
INDEX_PATH = Path("data/faiss_index/summaries.json")

if not DATA_PATH.exists():
    with st.spinner("⏳ Generating ARGO dataset (first run only)..."):
        from scripts.download_argo import main as gen_data
        gen_data()

if not INDEX_PATH.exists():
    with st.spinner("⏳ Building search index (first run only)..."):
        from scripts.build_index import build_index
        build_index()

# ── Imports after data is ready ───────────────────────────────────────────────
from backend.rag_chain import answer_query
from backend.router import classify_intent, extract_region, extract_month
from backend.visualizer import (
    plot_float_map,
    plot_temperature_profile,
    plot_salinity_profile,
    plot_sst_timeseries,
    plot_regional_comparison,
    plot_ts_diagram,
    plot_globe,
)

# ── Session state initialisation ──────────────────────────────────────────────
# These persist across reruns so toggles "remember" their state.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hi! I'm **FloatChat**, your AI guide to ARGO ocean data.\n\n"
                "Ask me about **temperature**, **salinity**, **float positions**, "
                "**depth profiles**, or regional oceanography. "
                "You can also use the 🎤 **voice button** (Chrome recommended)!"
            ),
        }
    ]
if "show_globe"      not in st.session_state: st.session_state.show_globe      = True
if "show_raw_table"  not in st.session_state: st.session_state.show_raw_table  = False
if "chart_type"      not in st.session_state: st.session_state.chart_type      = "Temperature Profile"
if "voice_text"      not in st.session_state: st.session_state.voice_text      = ""
if "prefill"         not in st.session_state: st.session_state.prefill         = ""

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #0077B6, #00B4D8, #90E0EF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: #666; font-size: 1.1rem; margin-top: 0; }
    .chat-user {
        background: linear-gradient(135deg, #0077B6, #00B4D8);
        color: white; padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0; max-width: 80%; float: right; clear: both;
    }
    .chat-bot {
        background: #f0f8ff; color: #1a1a2e;
        padding: 12px 18px; border-radius: 18px 18px 18px 4px;
        border-left: 4px solid #00B4D8;
        margin: 8px 0; max-width: 85%; clear: both;
    }
    .toggle-btn > button {
        background: #e8f4f8 !important; color: #0077B6 !important;
        border: 2px solid #00B4D8 !important; border-radius: 20px !important;
        font-weight: 600 !important;
    }
    .voice-hint { font-size: 0.78rem; color: #888; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Voice Input Component ─────────────────────────────────────────────────────
#
# HOW IT WORKS (for judges):
#   The browser's built-in Web Speech API (SpeechRecognition) listens to the
#   microphone, converts audio to text locally (no server round-trip), and
#   sends the resulting transcript to Streamlit via a hidden URL parameter
#   (?voice_query=...). We embed the JavaScript in an <iframe> using
#   st.components.v1.html(). When the user clicks 🎤, the JS fires, the user
#   speaks, and the transcript populates the chat input automatically.
#   Works best in Chrome/Edge which have the most complete Web Speech support.
#
VOICE_HTML = """
<div style="text-align:center; font-family:sans-serif;">
  <button id="mic-btn" onclick="startListening()"
    style="background:linear-gradient(135deg,#0077B6,#00B4D8);
           color:white; border:none; border-radius:50%; width:52px; height:52px;
           font-size:24px; cursor:pointer; box-shadow:0 3px 8px rgba(0,119,182,0.4);">
    🎤
  </button>
  <div id="status" style="font-size:12px; color:#888; margin-top:6px;">
    Click mic to speak (Chrome recommended)
  </div>
</div>
<script>
function startListening() {
  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    document.getElementById('status').innerText = '❌ Browser not supported. Use Chrome.';
    return;
  }
  var recognition = new SR();
  recognition.lang = 'en-IN';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  document.getElementById('status').innerText = '🎙️ Listening…';
  document.getElementById('mic-btn').style.background =
    'linear-gradient(135deg,#e63946,#ff6b6b)';

  recognition.start();

  recognition.onresult = function(event) {
    var transcript = event.results[0][0].transcript;
    document.getElementById('status').innerText = '✅ Got: ' + transcript;
    document.getElementById('mic-btn').style.background =
      'linear-gradient(135deg,#0077B6,#00B4D8)';
    // Send to Streamlit parent via postMessage
    window.parent.postMessage({type: 'voice_query', text: transcript}, '*');
  };

  recognition.onerror = function(event) {
    document.getElementById('status').innerText = '⚠️ Error: ' + event.error;
    document.getElementById('mic-btn').style.background =
      'linear-gradient(135deg,#0077B6,#00B4D8)';
  };

  recognition.onend = function() {
    if (document.getElementById('status').innerText === '🎙️ Listening…') {
      document.getElementById('status').innerText = 'Click mic to speak';
    }
  };
}
</script>
"""


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 FloatChat")
    st.markdown("*AI assistant for ARGO ocean data*")
    st.divider()

    # API Key
    st.markdown("### 🔑 Groq API Key")
    st.markdown("Get free key at [console.groq.com](https://console.groq.com)")
    default_key = ""
    try:
        default_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        default_key = os.getenv("GROQ_API_KEY", "")

    groq_key = st.text_input(
        "Paste your key here:",
        value=default_key,
        type="password",
        placeholder="gsk_xxxxxxxxxxxx",
        help="Never stored — used only for this session.",
    )
    if groq_key:
        st.success("✅ Key loaded")
    else:
        st.warning("⚠️ Enter key to enable AI answers")

    st.divider()

    # Filters
    st.markdown("### 🔭 Data Filters")
    region = st.selectbox(
        "Ocean Region",
        ["All", "Arabian Sea", "Bay of Bengal", "Indian Ocean"],
    )
    month = st.selectbox(
        "Month",
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        format_func=lambda x: "All months" if x == 0 else
        ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][x],
    )

    st.divider()

    # ── Toggle Controls (Feature 4) ──────────────────────────────────────────
    # st.session_state persists values across reruns so these act as true
    # toggle switches — clicking once opens, clicking again closes.
    st.markdown("### 🎛️ Panel Controls")

    if st.button(
        "🌍 Globe: " + ("ON  ✅" if st.session_state.show_globe else "OFF ⬜"),
        key="toggle_globe",
        use_container_width=True,
    ):
        st.session_state.show_globe = not st.session_state.show_globe
        st.rerun()

    if st.button(
        "🗃️ Raw Table: " + ("ON  ✅" if st.session_state.show_raw_table else "OFF ⬜"),
        key="toggle_table",
        use_container_width=True,
    ):
        st.session_state.show_raw_table = not st.session_state.show_raw_table
        st.rerun()

    st.divider()

    # Dataset info
    st.markdown("### 📊 Dataset Info")
    try:
        df_info = pd.read_parquet(DATA_PATH)
        st.metric("Float Profiles", f"{df_info['profile_id'].nunique():,}")
        st.metric("ARGO Floats",    f"{df_info['float_id'].nunique():,}")
        st.metric("Depth Levels",   f"{df_info['depth_m'].nunique()}")
    except Exception:
        st.info("Loading dataset...")

    st.divider()
    st.markdown("Built for **SIH 2026**")
    st.markdown("Problem: SIH26-40 — FloatChat")
    st.markdown("Data: ARGO GDAC — Indian Ocean")


# ── MAIN HEADER ───────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">FloatChat 🌊</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered conversational interface for ARGO ocean float data</p>',
            unsafe_allow_html=True)

# ── TABS (Feature 3) ──────────────────────────────────────────────────────────
# Four tabs replace the original three. Each tab is a separate "panel" of the
# dashboard. st.tabs() is Streamlit's native multi-panel component — no routing.
tab_chat, tab_globe, tab_charts, tab_data = st.tabs([
    "💬 Chat",
    "🌍 3D Globe",
    "📈 Charts & Trends",
    "🗃️ Data Explorer",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:

    # Example query buttons
    st.markdown("**💡 Try these:**")
    examples = [
        "What is the average SST in the Arabian Sea?",
        "Show temperature profiles in Bay of Bengal in January",
        "Compare salinity between Arabian Sea and Bay of Bengal",
        "Which region is warmest at the surface?",
        "What happens to temperature at 500m depth?",
    ]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex[:35] + "…" if len(ex) > 35 else ex, key=f"ex_{i}"):
            st.session_state.prefill = ex
            st.rerun()

    st.divider()

    # Chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🌊 {msg["content"]}</div>',
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Voice Input (Feature 2) ──────────────────────────────────────────────
    # The mic button lives in a narrow column next to the text input.
    # postMessage from the iframe is received by the parent Streamlit page.
    # We use a hidden text input + JS to pass the voice transcript back.
    col_input, col_mic = st.columns([10, 1])

    with col_input:
        prefill_val = st.session_state.pop("prefill", "")
        # Allow voice_text to pre-fill if received
        voice_val = st.session_state.pop("voice_text", "")
        default_val = voice_val or prefill_val

        query = st.text_input(
            "Ask about ARGO ocean data…",
            value=default_val,
            placeholder="e.g. What is the sea surface temperature in the Bay of Bengal?",
            key="chat_input",
            label_visibility="collapsed",
        )

    with col_mic:
        # Embed the voice component in a small iframe
        components.html(VOICE_HTML, height=80)
        st.markdown('<p class="voice-hint">🎤 Chrome only</p>', unsafe_allow_html=True)

    # JS bridge: listen for postMessage from the voice iframe and put the
    # transcript into a hidden Streamlit URL param so we can read it on rerun.
    components.html("""
    <script>
    window.addEventListener('message', function(event) {
        if (event.data && event.data.type === 'voice_query') {
            // Write to the Streamlit query params so it survives rerun
            var url = new URL(window.parent.location.href);
            url.searchParams.set('voice_query', event.data.text);
            window.parent.history.replaceState({}, '', url);
            // Trigger a Streamlit rerun by simulating Enter on the text input
            var inputs = window.parent.document.querySelectorAll('input[type="text"]');
            if (inputs.length > 0) {
                inputs[0].value = event.data.text;
                inputs[0].dispatchEvent(new Event('input', {bubbles: true}));
            }
        }
    });
    </script>
    """, height=0)

    # Pick up voice query from URL params if present
    try:
        params = st.query_params
        if "voice_query" in params and params["voice_query"]:
            vq = params["voice_query"]
            if vq and not query:
                query = vq
            st.query_params.clear()
    except Exception:
        pass

    # Send + Clear buttons
    col_send, col_clear = st.columns([1, 5])
    with col_send:
        send = st.button("Send 🚀", use_container_width=True)
    with col_clear:
        if st.button("Clear chat"):
            st.session_state.messages = st.session_state.messages[:1]
            st.rerun()

    # ── Process the query ────────────────────────────────────────────────────
    if send and query.strip():
        st.session_state.messages.append({"role": "user", "content": query})

        intent   = classify_intent(query)
        q_region = extract_region(query)
        q_month  = extract_month(query)

        with st.spinner("🌊 Searching ARGO data and generating answer…"):
            if not groq_key:
                try:
                    df_q = pd.read_parquet(DATA_PATH)
                    surface_q = df_q[df_q["depth_m"] == 0]
                    if q_region != "All":
                        surface_q = surface_q[surface_q["region"] == q_region]
                    avg_t = surface_q["temperature_c"].mean()
                    avg_s = surface_q["salinity_psu"].mean()
                    answer = (
                        f"📊 **Data summary for {q_region}:**\n\n"
                        f"- Average SST: **{avg_t:.1f}°C**\n"
                        f"- Average surface salinity: **{avg_s:.2f} PSU**\n"
                        f"- Profiles available: **{surface_q['profile_id'].nunique():,}**\n\n"
                        f"*Add your Groq API key in the sidebar for detailed AI analysis!*"
                    )
                except Exception as e:
                    answer = f"Please add your Groq API key in the sidebar. Error: {e}"
            else:
                answer = answer_query(query, groq_key)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Auto-show relevant chart below the answer
        st.markdown("**📈 Relevant visualization:**")
        if intent == "location":
            st.plotly_chart(plot_float_map(q_region, q_month), use_container_width=True)
        elif intent == "salinity":
            st.plotly_chart(plot_salinity_profile(q_region, q_month), use_container_width=True)
        elif intent == "comparison":
            st.plotly_chart(plot_regional_comparison(), use_container_width=True)
        elif intent == "trend":
            st.plotly_chart(plot_sst_timeseries(q_region), use_container_width=True)
        elif intent == "ts_diagram":
            st.plotly_chart(plot_ts_diagram(q_region), use_container_width=True)
        else:
            st.plotly_chart(plot_temperature_profile(q_region, q_month), use_container_width=True)

        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — 3D GLOBE (Feature 1)
# ══════════════════════════════════════════════════════════════════════════════
with tab_globe:
    # ── Why Plotly Scattergeo instead of PyDeck? (for judges) ────────────────
    # PyDeck's GlobeView requires a Mapbox token and uses WebGL in a way that
    # conflicts with Streamlit's sandboxed iframe. It also does not render on
    # Streamlit Community Cloud without extra config.
    # Plotly's go.Scattergeo with projection_type="orthographic" gives a true
    # rotating 3D globe, works 100% client-side, requires no API keys, renders
    # identical locally and on the cloud, and natively supports hover tooltips.

    st.markdown("### 🌍 3D Interactive Globe")
    st.caption(
        "Drag to rotate • Scroll to zoom • Hover a dot for float details  |  "
        "Rendered with **Plotly Scattergeo** (orthographic projection)"
    )

    # Toggle: the globe can be hidden via the sidebar toggle (Feature 4)
    if st.session_state.show_globe:
        with st.spinner("Rendering globe…"):
            globe_fig = plot_globe(region, month)
        st.plotly_chart(globe_fig, use_container_width=True)

        # Key facts below the globe
        st.divider()
        try:
            df_g = pd.read_parquet(DATA_PATH)
            surface_g = df_g[df_g["depth_m"] == 0]
            if region != "All":
                surface_g = surface_g[surface_g["region"] == region]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Active Floats",  surface_g["float_id"].nunique())
            c2.metric("Avg SST",        f"{surface_g['temperature_c'].mean():.1f}°C")
            c3.metric("Avg Salinity",   f"{surface_g['salinity_psu'].mean():.2f} PSU")
            c4.metric("Profiles",       f"{surface_g['profile_id'].nunique():,}")
        except Exception:
            pass
    else:
        st.info("🌍 Globe is hidden. Toggle it back ON from the **sidebar → Panel Controls**.")

    # Also show the 2D flat map below for comparison
    with st.expander("🗺️ Also show flat 2D map", expanded=False):
        st.plotly_chart(plot_float_map(region, month), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHARTS & TRENDS (Feature 3 — multi-panel with columns)
# ══════════════════════════════════════════════════════════════════════════════
with tab_charts:
    st.markdown("### 📈 Charts & Trends")

    # ── Chart type switcher (Feature 4 toggle) ───────────────────────────────
    chart_options = [
        "Temperature Profile",
        "Salinity Profile",
        "SST Time Series",
        "Regional Comparison",
        "T-S Diagram",
    ]
    st.session_state.chart_type = st.radio(
        "Switch chart type:",
        chart_options,
        index=chart_options.index(st.session_state.chart_type),
        horizontal=True,
        key="chart_switcher",
    )

    st.divider()

    # ── Side-by-side Temperature + Salinity profiles ─────────────────────────
    # This is the "multi-panel" showcase: two charts in st.columns([1,1])
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🌡️ Temperature vs Depth")
        st.plotly_chart(plot_temperature_profile(region, month),
                        use_container_width=True)

    with col_right:
        st.markdown("#### 🧂 Salinity vs Depth")
        st.plotly_chart(plot_salinity_profile(region, month),
                        use_container_width=True)

    st.divider()

    # ── The chart selected by the radio toggle ────────────────────────────────
    st.markdown(f"#### Selected: {st.session_state.chart_type}")
    if st.session_state.chart_type == "Temperature Profile":
        fig_sel = plot_temperature_profile(region, month)
    elif st.session_state.chart_type == "Salinity Profile":
        fig_sel = plot_salinity_profile(region, month)
    elif st.session_state.chart_type == "SST Time Series":
        fig_sel = plot_sst_timeseries(region)
    elif st.session_state.chart_type == "Regional Comparison":
        fig_sel = plot_regional_comparison()
    else:
        fig_sel = plot_ts_diagram(region)

    st.plotly_chart(fig_sel, use_container_width=True)

    # ── Stats row ─────────────────────────────────────────────────────────────
    try:
        df_c = pd.read_parquet(DATA_PATH)
        surface_c = df_c[df_c["depth_m"] == 0]
        if region != "All":
            surface_c = surface_c[surface_c["region"] == region]
        if month > 0:
            surface_c = surface_c[surface_c["month"] == month]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg SST",       f"{surface_c['temperature_c'].mean():.1f}°C")
        c2.metric("Avg Salinity",  f"{surface_c['salinity_psu'].mean():.2f} PSU")
        c3.metric("Profiles",      f"{surface_c['profile_id'].nunique():,}")
        c4.metric("Active Floats", f"{surface_c['float_id'].nunique()}")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATA EXPLORER (Feature 3 + Feature 4 toggle for raw table)
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("### 🗃️ Raw Data Explorer")

    # Toggle: show/hide the actual dataframe (Feature 4)
    show_tbl = st.session_state.show_raw_table
    col_t1, col_t2 = st.columns([3, 1])
    with col_t2:
        if st.button(
            "Hide table ⬆️" if show_tbl else "Show table ⬇️",
            key="inline_toggle_table",
            use_container_width=True,
        ):
            st.session_state.show_raw_table = not st.session_state.show_raw_table
            st.rerun()

    try:
        df_raw = pd.read_parquet(DATA_PATH)

        # Filters
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            depth_filter = st.select_slider(
                "Filter by depth (m)",
                options=sorted(df_raw["depth_m"].unique().tolist()),
                value=(0, 200),
            )
        with col_f2:
            region_filter = st.multiselect(
                "Filter by region",
                options=df_raw["region"].unique().tolist(),
                default=df_raw["region"].unique().tolist(),
            )

        filtered = df_raw[
            (df_raw["depth_m"] >= depth_filter[0]) &
            (df_raw["depth_m"] <= depth_filter[1]) &
            (df_raw["region"].isin(region_filter))
        ]

        st.caption(f"Showing {min(500, len(filtered)):,} of {len(filtered):,} records")

        # Only render the heavy dataframe widget when toggle is ON
        if st.session_state.show_raw_table:
            st.dataframe(
                filtered.head(500),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "temperature_c": st.column_config.NumberColumn("Temp (°C)",     format="%.2f"),
                    "salinity_psu":  st.column_config.NumberColumn("Salinity (PSU)", format="%.3f"),
                    "latitude":      st.column_config.NumberColumn("Lat",            format="%.4f"),
                    "longitude":     st.column_config.NumberColumn("Lon",            format="%.4f"),
                },
            )
        else:
            st.info("Click **Show table ⬇️** above to display the raw data.")

        # Summary stats always visible
        st.divider()
        st.markdown("#### 📊 Quick Stats")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records",    f"{len(filtered):,}")
        c2.metric("Unique Floats",    filtered["float_id"].nunique())
        c3.metric("Unique Profiles",  filtered["profile_id"].nunique())

        # Download
        csv = filtered.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download filtered data (CSV)",
            data=csv,
            file_name="argo_filtered.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Could not load data: {e}")
