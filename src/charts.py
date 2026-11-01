"""Shared Plotly styling for the Neon Laboratory theme."""

from __future__ import annotations

import plotly.graph_objects as go

LAB_BG = "#080B0F"
PAPER = "rgba(0,0,0,0)"
PLOT = "#0D1219"
CARD = "#111923"
BORDER = "#202D38"
TEXT = "#EDF4F7"
MUTED = "#91A0AD"
GREEN = "#B6FF4A"
CYAN = "#35E0F2"
AMBER = "#F5C542"
ROSE = "#FF6B8A"

MODEL_COLORS = {
    "knn": CYAN,
    "decision_tree": GREEN,
    "random_forest": AMBER,
}


def lab_layout(fig: go.Figure, *, height: int | None = 380, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PAPER,
        plot_bgcolor=PLOT,
        font=dict(color=TEXT, family="IBM Plex Sans, Segoe UI, sans-serif"),
        title=dict(text=title, font=dict(size=16, color=TEXT)) if title else None,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        margin=dict(l=40, r=24, t=56 if title else 32, b=40),
        height=height,
        colorway=[GREEN, CYAN, AMBER, ROSE, MUTED],
        hoverlabel=dict(bgcolor=CARD, font_color=TEXT, bordercolor=BORDER),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED))
    return fig
