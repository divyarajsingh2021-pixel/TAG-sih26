"""
FloatChat 🌊 — Streamlit Frontend
===================================
Main entry point. Run with: streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloatChat 🌊",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bootstrap data on first run ──────────────────────────────────────────────
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
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0077B6, #00B4D8, #90E0EF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #666;
        font-size: 1.1rem;
        margin-top: 0;
    }
    .chat-user {
        background: linear-gradient(135deg, #0077B6, #00B4D8);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    .chat-bot {
        background: #f0f8ff;
        color: #1a1a2e;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        border-left: 4px solid #00B4D8;
        margin: 8px 0;
        max-width: 85%;
        clear: both;
    }
    .metric-card {
        background: linear-gradient(135deg, #0077B6, #0096C7);
        color: white;
        padding: 16px;
        border-radius: 12px;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0077B6, #00B4D8);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 8px 24px;
        font-weight: 600;
    }
    .example-query {
        cursor: pointer;
        padding: 6px 12px;
        background: #e8f4f8;
        border-radius: 20px;
        font-size: 0.85rem;
        color: #0077B6;
        display: inline-block;
        margin: 3px;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 FloatChat")
    st.markdown("*AI assistant for ARGO ocean data*")
    st.divider()

    # ── API Key input ──────────────────────────────────────────────────────
    st.markdown("### 🔑 Groq API Key")
    st.markdown(
        "Get your free key at [console.groq.com](https://console.groq.com)",
        unsafe_allow_html=True
    )

    # Try to load from Streamlit secrets first
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
        help="Your key is never stored. Used only for this session.",
    )

    if groq_key:
        st.success("✅ Key loaded")
    else:
        st.warning("⚠️ Enter key to enable AI answers")

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────
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
    viz_type = st.selectbox(
        "Visualization",
        ["Float Map", "Temperature Profile", "Salinity Profile",
         "SST Time Series", "Regional Comparison", "T-S Diagram"],
    )

    st.divider()
    st.markdown("### 📊 Dataset Info")
    try:
        import pandas as pd
        df = pd.read_parquet(DATA_PATH)
        st.metric("Float Profiles", f"{df['profile_id'].nunique():,}")
        st.metric("ARGO Floats", f"{df['float_id'].nunique():,}")
        st.metric("Depth Levels", f"{df['depth_m'].nunique()}")
    except Exception:
        st.info("Loading dataset...")

    st.divider()
    st.markdown("Built for **Neural Nexus Hackathon**")
    st.markdown("Data: ARGO GDAC — Indian Ocean")


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">FloatChat 🌊</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered exploration of ARGO ocean float data</p>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_viz, tab_data = st.tabs(["💬 Chat", "📊 Visualize", "🗃️ Data Explorer"])


# ── CHAT TAB ──────────────────────────────────────────────────────────────────
with tab_chat:
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Hi! I'm FloatChat, your AI guide to ARGO ocean data in the Indian Ocean.\n\n"
                    "I can answer questions about **temperature**, **salinity**, **float positions**, "
                    "**depth profiles**, and regional oceanographic patterns.\n\n"
                    "Try asking me something below!"
                ),
            }
        ]

    # Example queries
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
            st.session_state["prefill"] = ex

    st.divider()

    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🌊 {msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Input
    prefill = st.session_state.pop("prefill", "")
    query = st.text_input(
        "Ask about ARGO ocean data...",
        value=prefill,
        placeholder="e.g. What is the sea surface temperature in the Bay of Bengal in January?",
        key="chat_input",
    )

    col_send, col_clear = st.columns([1, 5])
    with col_send:
        send = st.button("Send 🚀", use_container_width=True)
    with col_clear:
        if st.button("Clear chat", use_container_width=False):
            st.session_state.messages = st.session_state.messages[:1]
            st.rerun()

    if send and query.strip():
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})

        # Detect intent for context
        intent = classify_intent(query)
        q_region = extract_region(query)
        q_month = extract_month(query)

        with st.spinner("🌊 Searching ARGO data..."):
            if not groq_key:
                # Fallback: rule-based answer without LLM
                import pandas as pd
                try:
                    df = pd.read_parquet(DATA_PATH)
                    surface = df[df["depth_m"] == 0]
                    if q_region != "All":
                        surface = surface[surface["region"] == q_region]
                    avg_temp = surface["temperature_c"].mean()
                    avg_salt = surface["salinity_psu"].mean()
                    answer = (
                        f"📊 **Data summary for {q_region}:**\n\n"
                        f"- Average SST: **{avg_temp:.1f}°C**\n"
                        f"- Average surface salinity: **{avg_salt:.2f} PSU**\n"
                        f"- Profiles available: **{surface['profile_id'].nunique():,}**\n\n"
                        f"*Add your Groq API key in the sidebar for detailed AI analysis!*"
                    )
                except Exception as e:
                    answer = f"Please add your Groq API key in the sidebar to get AI-powered answers. Error: {e}"
            else:
                answer = answer_query(query, groq_key)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Show relevant viz automatically
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


# ── VISUALIZE TAB ─────────────────────────────────────────────────────────────
with tab_viz:
    st.markdown(f"### {viz_type}")
    st.caption(f"Region: {region} | Month: {'All' if month == 0 else month}")

    if viz_type == "Float Map":
        fig = plot_float_map(region, month)
    elif viz_type == "Temperature Profile":
        fig = plot_temperature_profile(region, month)
    elif viz_type == "Salinity Profile":
        fig = plot_salinity_profile(region, month)
    elif viz_type == "SST Time Series":
        fig = plot_sst_timeseries(region)
    elif viz_type == "Regional Comparison":
        fig = plot_regional_comparison()
    else:
        fig = plot_ts_diagram(region)

    st.plotly_chart(fig, use_container_width=True)

    # Stats row
    try:
        import pandas as pd
        df = pd.read_parquet(DATA_PATH)
        surface = df[df["depth_m"] == 0]
        if region != "All":
            surface = surface[surface["region"] == region]
        if month > 0:
            surface = surface[surface["month"] == month]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg SST", f"{surface['temperature_c'].mean():.1f}°C")
        c2.metric("Avg Salinity", f"{surface['salinity_psu'].mean():.2f} PSU")
        c3.metric("Profiles", f"{surface['profile_id'].nunique():,}")
        c4.metric("Active Floats", f"{surface['float_id'].nunique()}")
    except Exception:
        pass


# ── DATA EXPLORER TAB ─────────────────────────────────────────────────────────
with tab_data:
    st.markdown("### 🗃️ Raw Data Explorer")

    try:
        import pandas as pd
        df = pd.read_parquet(DATA_PATH)

        col1, col2 = st.columns(2)
        with col1:
            depth_filter = st.select_slider(
                "Filter by depth (m)",
                options=sorted(df["depth_m"].unique().tolist()),
                value=(0, 200),
            )
        with col2:
            region_filter = st.multiselect(
                "Filter by region",
                options=df["region"].unique().tolist(),
                default=df["region"].unique().tolist(),
            )

        filtered = df[
            (df["depth_m"] >= depth_filter[0]) &
            (df["depth_m"] <= depth_filter[1]) &
            (df["region"].isin(region_filter))
        ]

        st.dataframe(
            filtered.head(500),
            use_container_width=True,
            hide_index=True,
            column_config={
                "temperature_c": st.column_config.NumberColumn("Temp (°C)", format="%.2f"),
                "salinity_psu": st.column_config.NumberColumn("Salinity (PSU)", format="%.3f"),
                "latitude": st.column_config.NumberColumn("Lat", format="%.4f"),
                "longitude": st.column_config.NumberColumn("Lon", format="%.4f"),
            }
        )
        st.caption(f"Showing {min(500, len(filtered)):,} of {len(filtered):,} records")

        csv = filtered.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download filtered data (CSV)",
            data=csv,
            file_name="argo_filtered.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Could not load data: {e}")
