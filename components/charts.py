"""Figuras Plotly de la narrativa. Sistema de color consistente:
  - Serie observada / historico  -> azul marino (NAVY2)
  - Tendencia                    -> azul marino claro (NAVY3)
  - Train                        -> NAVY2   Test (tramo de validacion) -> SUPPORT (ambar)
  - Forecast / prediccion        -> SUPPORT, con banda de IC en SUPPORT_SOFT
  - Baseline naive (referencia)  -> MUTED, discontinua
  - Resultado bueno/malo real    -> POSITIVE / NEGATIVE (solo en resumenes, no en series)
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INK = "#1D2638"
NAVY2 = "#273A5F"
NAVY3 = "#4A628E"
NAVY4 = "#B9C5D6"
MUTED = "#6B7280"
LINE = "#E3DFD5"
POSITIVE = "#6E7F5B"
POSITIVE_SOFT = "rgba(110,127,91,0.14)"
NEGATIVE = "#C2412E"
NEGATIVE_SOFT = "rgba(194,65,46,0.12)"
SUPPORT = "#B8783C"
SUPPORT_SOFT = "rgba(184,120,60,0.16)"
NAVY_SOFT = "rgba(39,58,95,0.10)"
FONT = "Arial, Helvetica, sans-serif"


def _base_layout(fig, height=420, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=12.5),
        hovermode="x unified",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                     font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, linecolor=LINE, tickfont=dict(color=MUTED)),
        yaxis=dict(showgrid=True, gridcolor=LINE, zeroline=False, tickfont=dict(color=MUTED)),
    )
    return fig


def full_series(split_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=split_df["date"], y=split_df["ventas"], name="Ventas semanales",
                              line=dict(color=NAVY2, width=1.6), fill="tozeroy", fillcolor=NAVY_SOFT))
    fig.update_yaxes(title_text="Ventas (€)", rangemode="tozero")
    return _base_layout(fig, height=380, legend=False)


def decomposition(decomp_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                         subplot_titles=("Observado", "Tendencia", "Estacionalidad — 52 semanas", "Residuo"))
    panels = [
        ("observed", NAVY2), ("trend", NAVY3),
        ("seasonal", SUPPORT), ("resid", MUTED),
    ]
    for i, (col, color) in enumerate(panels, start=1):
        fig.add_trace(go.Scatter(x=decomp_df["date"], y=decomp_df[col], line=dict(color=color, width=1.3),
                                  showlegend=False), row=i, col=1)
    fig.update_layout(
        height=620, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=12), showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=LINE, zeroline=False, tickfont=dict(color=MUTED))
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(family=FONT, color=INK, size=12.5)
        ann["x"] = 0
        ann["xanchor"] = "left"
    return fig


def train_test_split_chart(split_df: pd.DataFrame) -> go.Figure:
    train = split_df[split_df["split"] == "train"]
    test = split_df[split_df["split"] == "test"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train["date"], y=train["ventas"], name="Train",
                              line=dict(color=NAVY2, width=1.6)))
    fig.add_trace(go.Scatter(x=test["date"], y=test["ventas"], name="Test",
                              line=dict(color=SUPPORT, width=2.2)))
    fig.add_vline(x=test["date"].iloc[0], line_dash="dash", line_color=MUTED)
    fig.update_yaxes(title_text="Ventas (€)")
    return _base_layout(fig, height=380)


def stationarity_chart(train_split_df: pd.DataFrame, diff_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.12,
                         subplot_titles=("Serie original — train", "Serie diferenciada (d=1)"))
    fig.add_trace(go.Scatter(x=train_split_df["date"], y=train_split_df["ventas"],
                              line=dict(color=NAVY2, width=1.4), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=diff_df["date"], y=diff_df["diff"],
                              line=dict(color=NAVY3, width=1.1), showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=NEGATIVE, row=2, col=1)
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=12), showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=LINE, zeroline=False, tickfont=dict(color=MUTED))
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(family=FONT, color=INK, size=12.5)
        ann["x"] = 0
        ann["xanchor"] = "left"
    return fig


def acf_pacf_chart(acf_pacf: dict) -> go.Figure:
    lags = acf_pacf["lags"]
    band = acf_pacf["conf_band"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF — identificación de q", "PACF — identificación de p"))
    fig.add_trace(go.Bar(x=lags, y=acf_pacf["acf"], marker_color=NAVY2, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=lags, y=acf_pacf["pacf"], marker_color=NAVY3, showlegend=False), row=1, col=2)
    for c in (1, 2):
        fig.add_hline(y=band, line_dash="dot", line_color=MUTED, row=1, col=c)
        fig.add_hline(y=-band, line_dash="dot", line_color=MUTED, row=1, col=c)
        fig.add_hline(y=0, line_color=LINE, row=1, col=c)
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=12), showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=LINE, zeroline=False, tickfont=dict(color=MUTED))
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(family=FONT, color=INK, size=12.5)
    return fig


def residual_diagnostics(resid_df: pd.DataFrame, diag: dict) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Residuos en el tiempo", "Distribución de residuos", "QQ-Plot", "ACF de residuos"),
    )
    fig.add_trace(go.Scatter(x=resid_df["date"], y=resid_df["residual"], line=dict(color=NAVY2, width=1),
                              showlegend=False), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=NEGATIVE, row=1, col=1)

    fig.add_trace(go.Histogram(x=resid_df["residual"], marker_color=NAVY3, nbinsx=25, showlegend=False),
                  row=1, col=2)

    qq_t, qq_s = diag["qq_theoretical"], diag["qq_sample"]
    fig.add_trace(go.Scatter(x=qq_t, y=qq_s, mode="markers",
                              marker=dict(color=NAVY2, size=5), showlegend=False), row=2, col=1)
    lo, hi = min(qq_t), max(qq_t)
    slope = np.polyfit(qq_t, qq_s, 1)
    fig.add_trace(go.Scatter(x=[lo, hi], y=[slope[0] * lo + slope[1], slope[0] * hi + slope[1]],
                              mode="lines", line=dict(color=NEGATIVE, dash="dash"), showlegend=False), row=2, col=1)

    fig.add_trace(go.Bar(x=diag["acf_lags"], y=diag["acf_values"], marker_color=NAVY3, showlegend=False),
                  row=2, col=2)
    band = diag["acf_conf_band"]
    fig.add_hline(y=band, line_dash="dot", line_color=MUTED, row=2, col=2)
    fig.add_hline(y=-band, line_dash="dot", line_color=MUTED, row=2, col=2)

    fig.update_layout(
        height=620, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=12), showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=LINE, zeroline=False, tickfont=dict(color=MUTED))
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(family=FONT, color=INK, size=12.5)
    return fig


def test_forecast_chart(test_forecast_df: pd.DataFrame, split_df: pd.DataFrame, tail_weeks: int = 40) -> go.Figure:
    train_tail = split_df[split_df["split"] == "train"].tail(tail_weeks)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_tail["date"], y=train_tail["ventas"], name="Histórico / Train",
                              line=dict(color=MUTED, width=1.2)))
    fig.add_trace(go.Scatter(x=test_forecast_df["date"], y=test_forecast_df["actual"], name="Real — Test",
                              line=dict(color=NAVY2, width=2)))
    fig.add_trace(go.Scatter(
        x=pd.concat([test_forecast_df["date"], test_forecast_df["date"][::-1]]),
        y=pd.concat([test_forecast_df["upper95"], test_forecast_df["lower95"][::-1]]),
        fill="toself", fillcolor=SUPPORT_SOFT, line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=test_forecast_df["date"], y=test_forecast_df["forecast"], name="Predicción SARIMA",
                              line=dict(color=SUPPORT, width=2.4, dash="dash")))
    fig.add_trace(go.Scatter(x=test_forecast_df["date"], y=test_forecast_df["naive_seasonal"],
                              name="Baseline naive (t-52 sem.)",
                              line=dict(color=NEGATIVE, width=1.4, dash="dot")))
    fig.add_vline(x=test_forecast_df["date"].iloc[0], line_dash="dash", line_color=MUTED)
    fig.update_yaxes(title_text="Ventas (€)")
    return _base_layout(fig, height=440)


def future_forecast_chart(split_df: pd.DataFrame, future_df: pd.DataFrame, tail_weeks: int = 60) -> go.Figure:
    hist_tail = split_df.tail(tail_weeks)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_tail["date"], y=hist_tail["ventas"], name="Histórico",
                              line=dict(color=NAVY2, width=1.6)))
    bridge_x = [hist_tail["date"].iloc[-1]] + list(future_df["date"])
    bridge_y = [hist_tail["ventas"].iloc[-1]] + list(future_df["forecast"])
    fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, name="Forecast — 20 semanas",
                              line=dict(color=SUPPORT, width=2.4, dash="dash"), mode="lines+markers",
                              marker=dict(size=4)))
    band_x = list(future_df["date"]) + list(future_df["date"][::-1])
    band_y = list(future_df["upper95"]) + list(future_df["lower95"][::-1])
    fig.add_trace(go.Scatter(x=band_x, y=band_y, fill="toself", fillcolor=SUPPORT_SOFT,
                              line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
    fig.add_vline(x=hist_tail["date"].iloc[-1], line_dash="dash", line_color=MUTED)
    fig.update_yaxes(title_text="Ventas (€)")
    return _base_layout(fig, height=440)


def playground_horizon_chart(split_df: pd.DataFrame, future_df: pd.DataFrame, horizon: int, tail_weeks: int = 30) -> go.Figure:
    """Igual que future_forecast_chart pero resalta la semana elegida en el
    Playground con un marcador propio, y atenúa el resto del horizonte."""
    hist_tail = split_df.tail(tail_weeks)
    selected = future_df.iloc[horizon - 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_tail["date"], y=hist_tail["ventas"], name="Histórico",
                              line=dict(color=NAVY2, width=1.6)))
    bridge_x = [hist_tail["date"].iloc[-1]] + list(future_df["date"])
    bridge_y = [hist_tail["ventas"].iloc[-1]] + list(future_df["forecast"])
    fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, name="Forecast",
                              line=dict(color=SUPPORT, width=2, dash="dash"), mode="lines+markers",
                              marker=dict(size=4, color=SUPPORT), opacity=0.75))
    band_x = list(future_df["date"]) + list(future_df["date"][::-1])
    band_y = list(future_df["upper95"]) + list(future_df["lower95"][::-1])
    fig.add_trace(go.Scatter(x=band_x, y=band_y, fill="toself", fillcolor=SUPPORT_SOFT,
                              line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[selected["date"]], y=[selected["forecast"]], name="Semana elegida",
                              mode="markers", marker=dict(size=13, color=INK, line=dict(color="white", width=2))))
    fig.add_vline(x=hist_tail["date"].iloc[-1], line_dash="dash", line_color=MUTED)
    fig.update_yaxes(title_text="Ventas (€)")
    return _base_layout(fig, height=420)
