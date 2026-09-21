"""Figuras Plotly de la narrativa. Sistema de color consistente en todo el proyecto:
  - Serie real / histórico        → azul marino (INK / NAVY2)
  - Tendencia, referencias        → azul medio (NAVY3) y gris (MUTED), discontinuas
  - Predicción, test, escenario   → color de apoyo (SUPPORT, ámbar), con banda del intervalo en SUPPORT_SOFT
  - Verde / rojo                  → solo lo semántico; aquí prácticamente no se usan
Legibles en móvil: leyendas arriba, etiquetas cortas y cada diagnóstico en su propia figura (no cuadrículas de
subgráficos que quedarían diminutas en 350 px)."""
from statistics import NormalDist

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
SUPPORT = "#B8783C"
SUPPORT_SOFT = "rgba(184,120,60,0.16)"
NAVY_SOFT = "rgba(39,58,95,0.10)"
FONT = "Arial, Helvetica, sans-serif"
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


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
                    font=dict(size=11.5), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, linecolor=LINE, tickfont=dict(color=MUTED)),
        yaxis=dict(showgrid=True, gridcolor=LINE, zeroline=False, tickfont=dict(color=MUTED)),
    )
    return fig


def _titled(fig, title: str, height: int, legend: bool = False):
    """Figura pequeña con su título dentro (para diagnósticos que se colocan en columnas)."""
    _base_layout(fig, height=height, legend=legend)
    fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0, xanchor="left", font=dict(size=13.5), y=0.97),
                      margin=dict(l=10, r=10, t=46, b=10))
    return fig


def _panel_titles(fig):
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(family=FONT, color=INK, size=12.5)
        ann["x"], ann["xanchor"] = 0, "left"


def full_series(split_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=split_df["date"], y=split_df["ventas"], name="Ventas semanales",
                             line=dict(color=NAVY2, width=1.7), fill="tozeroy", fillcolor=NAVY_SOFT))
    fig.update_yaxes(title_text="Ventas (€)", rangemode="tozero")
    return _base_layout(fig, height=380, legend=False)


def decomposition(decomp_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("Tendencia: hacia dónde van las ventas", "Estacionalidad: lo que se repite cada año",
                                        "Residuo: lo que queda"))
    for i, (col, color) in enumerate((("trend", NAVY3), ("seasonal", SUPPORT), ("resid", MUTED)), start=1):
        fig.add_trace(go.Scatter(x=decomp_df["date"], y=decomp_df[col], line=dict(color=color, width=1.6),
                                 showlegend=False), row=i, col=1)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=INK, size=12), showlegend=False,
                      hovermode="x unified")
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=LINE, zeroline=False, tickfont=dict(color=MUTED))
    _panel_titles(fig)
    return fig


def seasonal_by_year(pivot: pd.DataFrame, full_years: list) -> go.Figure:
    """El patrón de cada año (sin tendencia), alineado por semana: si las líneas se parecen, la estacionalidad existe."""
    colors = {full_years[0]: NAVY4, full_years[1]: NAVY3, full_years[-1]: INK}
    fig = go.Figure()
    for year in full_years:
        fig.add_trace(go.Scatter(x=pivot["week"], y=pivot[str(year)], name=str(year),
                                 line=dict(color=colors.get(year, NAVY3), width=2)))
    fig.add_hline(y=0, line_color=LINE)
    month_ticks = [1, 9, 18, 27, 35, 44]
    fig.update_xaxes(title_text="Semana del año", tickvals=month_ticks)
    fig.update_yaxes(title_text="Ventas sobre la tendencia (€)")
    return _base_layout(fig, height=380)


def train_test_split_chart(split_df: pd.DataFrame) -> go.Figure:
    train, test = split_df[split_df["split"] == "train"], split_df[split_df["split"] == "test"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train["date"], y=train["ventas"], name="Entrenamiento (el modelo aprende)",
                             line=dict(color=NAVY2, width=1.7)))
    fig.add_trace(go.Scatter(x=test["date"], y=test["ventas"], name="Test (el modelo no lo ve)",
                             line=dict(color=SUPPORT, width=2.4)))
    fig.add_vline(x=test["date"].iloc[0], line_dash="dash", line_color=MUTED)
    fig.update_yaxes(title_text="Ventas (€)")
    return _base_layout(fig, height=380)


def stationarity_chart(train_split_df: pd.DataFrame, diff_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.14,
                        subplot_titles=("Serie original (entrenamiento)", "Serie diferenciada (d=1)"))
    fig.add_trace(go.Scatter(x=train_split_df["date"], y=train_split_df["ventas"],
                             line=dict(color=NAVY2, width=1.5), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=diff_df["date"], y=diff_df["diff"],
                             line=dict(color=NAVY3, width=1.2), showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED, row=2, col=1)
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=INK, size=12), showlegend=False,
                      hovermode="x unified")
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=LINE, zeroline=False, tickfont=dict(color=MUTED))
    _panel_titles(fig)
    return fig


def _correlogram(values: list, band: float, title: str, color: str, highlight: dict = None) -> go.Figure:
    lags = list(range(len(values)))
    colors = [color] * len(values)
    for lag, c in (highlight or {}).items():
        colors[lag] = c
    fig = go.Figure(go.Bar(x=lags, y=values, marker_color=colors, hoverinfo="skip"))
    fig.add_hline(y=band, line_dash="dot", line_color=MUTED)
    fig.add_hline(y=-band, line_dash="dot", line_color=MUTED)
    fig.add_hline(y=0, line_color=LINE)
    fig.update_xaxes(title_text="Retardo (semanas)", dtick=13)
    fig.update_yaxes(range=[-0.7, 0.7])
    return _titled(fig, title, height=300)


def acf_chart(acf_pacf: dict) -> go.Figure:
    """ACF de la serie diferenciada; en ámbar, el retardo 52 (una estacionalidad anual)."""
    return _correlogram(acf_pacf["acf"], acf_pacf["conf_band"], "ACF: memoria de la serie diferenciada", NAVY2,
                        highlight={52: SUPPORT} if len(acf_pacf["acf"]) > 52 else None)


def pacf_chart(acf_pacf: dict) -> go.Figure:
    return _correlogram(acf_pacf["pacf"], acf_pacf["conf_band"], "PACF: memoria directa", NAVY3,
                        highlight={52: SUPPORT} if len(acf_pacf["pacf"]) > 52 else None)


def residuals_time(resid: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=resid["date"], y=resid["residual"], line=dict(color=NAVY2, width=1.2), name="Residuo"))
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED)
    fig.update_yaxes(title_text="€")
    return _titled(fig, "Residuos en el tiempo", height=300)


def residuals_hist(resid: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(x=resid["residual"], nbinsx=22, marker_color=NAVY3, hoverinfo="skip"))
    fig.update_xaxes(title_text="Residuo (€)")
    fig.update_yaxes(title_text="Semanas")
    return _titled(fig, "Distribución de los residuos", height=300)


def residuals_qq(resid: pd.DataFrame) -> go.Figure:
    y = np.sort(resid["residual"].values)
    n = len(y)
    x = np.array([NormalDist().inv_cdf((i + 0.5) / n) for i in range(n)])
    slope, intercept = np.polyfit(x, y, 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(color=NAVY2, size=5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[x.min(), x.max()], y=[slope * x.min() + intercept, slope * x.max() + intercept],
                             mode="lines", line=dict(color=SUPPORT, dash="dash"), hoverinfo="skip"))
    fig.update_xaxes(title_text="Cuantiles de una normal")
    fig.update_yaxes(title_text="Cuantiles de los residuos (€)")
    return _titled(fig, "¿Son normales? (QQ-plot)", height=300)


def residuals_acf(resid: pd.DataFrame, max_lag: int = 20) -> go.Figure:
    x = resid["residual"].values - resid["residual"].mean()
    acf = [float(np.sum(x[k:] * x[:len(x) - k]) / np.sum(x * x)) for k in range(max_lag + 1)]
    band = 1.96 / np.sqrt(len(x))
    fig = go.Figure(go.Bar(x=list(range(max_lag + 1)), y=acf, marker_color=NAVY3, hoverinfo="skip"))
    fig.add_hline(y=band, line_dash="dot", line_color=MUTED)
    fig.add_hline(y=-band, line_dash="dot", line_color=MUTED)
    fig.update_xaxes(title_text="Retardo (semanas)", dtick=5)
    fig.update_yaxes(range=[-0.5, 1.05])
    return _titled(fig, "Autocorrelación de los residuos", height=300)


def test_forecast_chart(test_df: pd.DataFrame, baselines: pd.DataFrame, split_df: pd.DataFrame,
                        tail_weeks: int = 40) -> go.Figure:
    train_tail = split_df[split_df["split"] == "train"].tail(tail_weeks)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([test_df["date"], test_df["date"][::-1]]),
        y=pd.concat([test_df["upper95"], test_df["lower95"][::-1]]),
        fill="toself", fillcolor=SUPPORT_SOFT, line=dict(color="rgba(0,0,0,0)"), name="Intervalo al 95% del SARIMA",
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=train_tail["date"], y=train_tail["ventas"], name="Histórico",
                             line=dict(color=MUTED, width=1.3)))
    fig.add_trace(go.Scatter(x=test_df["date"], y=test_df["actual"], name="Ventas reales",
                             line=dict(color=INK, width=2.4)))
    fig.add_trace(go.Scatter(x=test_df["date"], y=test_df["forecast"], name="SARIMA",
                             line=dict(color=SUPPORT, width=2.4, dash="dash")))
    fig.add_trace(go.Scatter(x=baselines["date"], y=baselines["year_ago"], name="Año anterior",
                             line=dict(color=MUTED, width=1.8, dash="dot")))
    fig.add_trace(go.Scatter(x=baselines["date"], y=baselines["year_ago_growth"], name="Año anterior ajustado por crecimiento",
                             line=dict(color=NAVY3, width=1.8, dash="dot")))
    fig.add_vline(x=test_df["date"].iloc[0], line_dash="dash", line_color=MUTED)
    fig.update_yaxes(title_text="Ventas (€)")
    return _base_layout(fig, height=480)


def interval_width_chart(widths: pd.DataFrame) -> go.Figure:
    """Ancho del intervalo al 95% según la semana de horizonte: en las 52 semanas de test y en la previsión final."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=widths["horizon"], y=widths["width_test"], name="Modelo del test (52 semanas)",
                             line=dict(color=SUPPORT, width=2.6)))
    fig.add_trace(go.Scatter(x=widths["horizon"], y=widths["width_future"], name="Previsión final (20 semanas)",
                             line=dict(color=NAVY2, width=2.6, dash="dash")))
    fig.add_vline(x=20, line_dash="dot", line_color=MUTED)
    fig.update_xaxes(title_text="Semanas hacia delante")
    fig.update_yaxes(title_text="Ancho del intervalo (€)", rangemode="tozero")
    return _base_layout(fig, height=360)


def playground_horizon_chart(split_df: pd.DataFrame, future_df: pd.DataFrame, horizon: int, tail_weeks: int = 30) -> go.Figure:
    """Últimas semanas reales, la previsión a 20 semanas con su intervalo y la semana elegida resaltada."""
    hist_tail = split_df.tail(tail_weeks)
    selected = future_df.iloc[horizon - 1]
    last_date, last_sales = hist_tail["date"].iloc[-1], hist_tail["ventas"].iloc[-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(future_df["date"]) + list(future_df["date"][::-1]),
        y=list(future_df["upper95"]) + list(future_df["lower95"][::-1]),
        fill="toself", fillcolor=SUPPORT_SOFT, line=dict(color="rgba(0,0,0,0)"), name="Intervalo al 95%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist_tail["date"], y=hist_tail["ventas"], name="Histórico", line=dict(color=NAVY2, width=1.8)))
    fig.add_trace(go.Scatter(x=[last_date] + list(future_df["date"]), y=[last_sales] + list(future_df["forecast"]),
                             name="Previsión", line=dict(color=SUPPORT, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[selected["date"]], y=[selected["forecast"]], name="Semana elegida", mode="markers",
                             marker=dict(size=13, color=INK, line=dict(color="white", width=2))))
    fig.add_vline(x=last_date, line_dash="dash", line_color=MUTED)
    fig.update_yaxes(title_text="Ventas (€)")
    return _base_layout(fig, height=420)
