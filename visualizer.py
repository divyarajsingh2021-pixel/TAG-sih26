"""
backend/visualizer.py
---------------------
All chart and map generation functions for FloatChat.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Optional

DATA_PATH = Path("data/processed/argo_indian_ocean.parquet")

_df_cache: Optional[pd.DataFrame] = None


def load_data() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        if DATA_PATH.exists():
            _df_cache = pd.read_parquet(DATA_PATH)
        else:
            _df_cache = pd.DataFrame()
    return _df_cache


def filter_data(region: str = "All", month: int = 0, year: int = 2023) -> pd.DataFrame:
    df = load_data()
    if df.empty:
        return df
    df = df[df["year"] == year]
    if region != "All":
        df = df[df["region"] == region]
    if month > 0:
        df = df[df["month"] == month]
    return df


# ── Float Map ────────────────────────────────────────────────────────────────

def plot_float_map(region: str = "All", month: int = 0) -> go.Figure:
    """Interactive map of ARGO float positions."""
    df = filter_data(region=region, month=month)
    surface = df[df["depth_m"] == 0].copy() if not df.empty else df

    if surface.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return fig

    # Sample for performance
    sample = surface.groupby("float_id").first().reset_index()

    fig = px.scatter_mapbox(
        sample,
        lat="latitude",
        lon="longitude",
        color="temperature_c",
        color_continuous_scale="RdYlBu_r",
        size_max=10,
        zoom=2.5,
        center={"lat": 10, "lon": 75},
        mapbox_style="open-street-map",
        hover_data={"float_id": True, "temperature_c": ":.1f", "salinity_psu": ":.2f"},
        labels={"temperature_c": "SST (°C)", "salinity_psu": "Salinity (PSU)"},
        title=f"ARGO Float Positions — {region} {'Month '+str(month) if month else ''}",
    )
    fig.update_layout(
        height=500,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="SST °C"),
    )
    return fig


# ── Temperature Profile ───────────────────────────────────────────────────────

def plot_temperature_profile(region: str = "All", month: int = 0) -> go.Figure:
    """Average temperature vs depth profile."""
    df = filter_data(region=region, month=month)
    if df.empty:
        return go.Figure()

    avg = df.groupby("depth_m")["temperature_c"].agg(["mean", "std"]).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=avg["mean"] + avg["std"], y=avg["depth_m"],
        mode="lines", line=dict(width=0), showlegend=False,
        fillcolor="rgba(68,114,196,0.2)", name="±1 STD",
    ))
    fig.add_trace(go.Scatter(
        x=avg["mean"] - avg["std"], y=avg["depth_m"],
        mode="lines", line=dict(width=0),
        fill="tonextx", fillcolor="rgba(68,114,196,0.2)",
        name="±1 STD",
    ))
    fig.add_trace(go.Scatter(
        x=avg["mean"], y=avg["depth_m"],
        mode="lines+markers",
        line=dict(color="#4472C4", width=3),
        marker=dict(size=6),
        name="Mean Temperature",
    ))

    fig.update_layout(
        title=f"Temperature Profile — {region}",
        xaxis_title="Temperature (°C)",
        yaxis_title="Depth (m)",
        yaxis=dict(autorange="reversed"),
        height=450,
        template="plotly_white",
        legend=dict(x=0.7, y=0.95),
    )
    return fig


# ── Salinity Profile ──────────────────────────────────────────────────────────

def plot_salinity_profile(region: str = "All", month: int = 0) -> go.Figure:
    """Average salinity vs depth profile."""
    df = filter_data(region=region, month=month)
    if df.empty:
        return go.Figure()

    avg = df.groupby("depth_m")["salinity_psu"].agg(["mean", "std"]).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=avg["mean"] + avg["std"], y=avg["depth_m"],
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=avg["mean"] - avg["std"], y=avg["depth_m"],
        mode="lines", line=dict(width=0),
        fill="tonextx", fillcolor="rgba(255,127,14,0.2)",
        name="±1 STD",
    ))
    fig.add_trace(go.Scatter(
        x=avg["mean"], y=avg["depth_m"],
        mode="lines+markers",
        line=dict(color="#FF7F0E", width=3),
        marker=dict(size=6),
        name="Mean Salinity",
    ))

    fig.update_layout(
        title=f"Salinity Profile — {region}",
        xaxis_title="Salinity (PSU)",
        yaxis_title="Depth (m)",
        yaxis=dict(autorange="reversed"),
        height=450,
        template="plotly_white",
    )
    return fig


# ── SST Time Series ───────────────────────────────────────────────────────────

def plot_sst_timeseries(region: str = "All") -> go.Figure:
    """Monthly mean SST over time."""
    df = filter_data(region=region)
    surface = df[df["depth_m"] == 0] if not df.empty else df
    if surface.empty:
        return go.Figure()

    monthly = surface.groupby("month")["temperature_c"].mean().reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["month_name"] = monthly["month"].apply(lambda x: month_names[x-1])

    fig = px.line(
        monthly, x="month_name", y="temperature_c",
        markers=True,
        title=f"Monthly Mean Sea Surface Temperature — {region}",
        labels={"temperature_c": "SST (°C)", "month_name": "Month"},
        color_discrete_sequence=["#E4002B"],
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    fig.update_layout(height=400, template="plotly_white")
    return fig


# ── Regional Comparison ───────────────────────────────────────────────────────

def plot_regional_comparison() -> go.Figure:
    """Box plot comparing SST across regions."""
    df = load_data()
    surface = df[df["depth_m"] == 0] if not df.empty else df
    if surface.empty:
        return go.Figure()

    fig = px.box(
        surface, x="region", y="temperature_c",
        color="region",
        title="SST Distribution by Region",
        labels={"temperature_c": "Temperature (°C)", "region": "Region"},
        color_discrete_map={
            "Arabian Sea": "#1f77b4",
            "Bay of Bengal": "#ff7f0e",
            "Indian Ocean": "#2ca02c",
        },
    )
    fig.update_layout(height=400, template="plotly_white", showlegend=False)
    return fig


# ── TS Diagram ────────────────────────────────────────────────────────────────

def plot_ts_diagram(region: str = "All") -> go.Figure:
    """Temperature-Salinity scatter (water mass identification)."""
    df = filter_data(region=region)
    if df.empty:
        return go.Figure()

    sample = df.sample(min(3000, len(df)), random_state=42)

    fig = px.scatter(
        sample, x="salinity_psu", y="temperature_c",
        color="depth_m",
        color_continuous_scale="Viridis_r",
        opacity=0.5,
        title=f"T-S Diagram — {region}",
        labels={
            "salinity_psu": "Salinity (PSU)",
            "temperature_c": "Temperature (°C)",
            "depth_m": "Depth (m)",
        },
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=450, template="plotly_white")
    return fig
