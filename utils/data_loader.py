"""Carga de artefactos con cache de Streamlit. app.py nunca importa
statsmodels/sklearn: todo lo que necesita para narrar y para el Playground
(elegir una semana del horizonte de 20) ya viene precalculado por
model/train.py en model/artifacts/."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "model" / "artifacts"


@st.cache_data(show_spinner=False)
def load_json(name: str):
    with open(ARTIFACTS / name, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(ARTIFACTS / name)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def artifacts_ready() -> bool:
    return (ARTIFACTS / "dataset_stats.json").exists() and (ARTIFACTS / "future_forecast.csv").exists()
