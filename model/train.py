"""
Entrena SARIMA(0,1,1)(0,1,1,52) replicando exactamente la metodologia del
notebook original (arima/01-ventas-semanales-retail) y guarda en
model/artifacts/ todo lo que app.py necesita para narrar el caso: series
para los graficos de exploracion/estacionariedad/ACF-PACF, diagnostico de
residuos, evaluacion sobre test (comparada contra una regla naive
estacional) y el forecast operativo a 20 semanas.

app.py no reentrena nada ni importa statsmodels: todo lo que necesita para
el Playground (elegir una semana del horizonte de 20) ya viene precalculado
en future_forecast.csv.

Ejecutar una sola vez (o si cambia el dataset):
    py -3.10 model/train.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "arima.xlsx"
ARTIFACTS = ROOT / "model" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

ORDER = (0, 1, 1)
SEASONAL_ORDER = (0, 1, 1, 52)
SEASONAL_PERIOD = 52
TRAIN_FRAC = 0.80
FUTURE_HORIZON = 20


def mape(y_real, y_pred):
    y_real, y_pred = np.asarray(y_real, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_real - y_pred) / y_real)) * 100)


def eval_metrics(y_real, y_pred):
    mae = float(mean_absolute_error(y_real, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_real, y_pred)))
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape(y_real, y_pred),
        "error_relativo": float(mae / np.mean(y_real) * 100),
    }


def adf_summary(series, name):
    result = adfuller(series.dropna())
    return {
        "name": name,
        "adf_stat": float(result[0]),
        "p_value": float(result[1]),
        "is_stationary": bool(result[1] < 0.05),
    }


def main():
    print("Cargando dataset...")
    df = pd.read_excel(DATA_PATH, parse_dates=["Fecha"], index_col="Fecha")
    df = df.asfreq("W-SUN")

    n_obs = len(df)
    split = int(n_obs * TRAIN_FRAC)
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

    # ------------------------------------------------------------------
    # 1) Estadisticos generales + descomposicion (Exploracion)
    # ------------------------------------------------------------------
    decomp = seasonal_decompose(df["Ventas"], model="additive", period=SEASONAL_PERIOD)
    decomp_df = pd.DataFrame({
        "date": df.index,
        "observed": decomp.observed.values,
        "trend": decomp.trend.values,
        "seasonal": decomp.seasonal.values,
        "resid": decomp.resid.values,
    })
    decomp_df.to_csv(ARTIFACTS / "decomposition.csv", index=False)

    series_split_df = pd.DataFrame({
        "date": df.index,
        "ventas": df["Ventas"].values,
        "split": ["train"] * len(train) + ["test"] * len(test),
    })
    series_split_df.to_csv(ARTIFACTS / "series_split.csv", index=False)

    # ------------------------------------------------------------------
    # 2) Estacionariedad — ADF original vs diferenciada (Metodologia)
    # ------------------------------------------------------------------
    train_diff = train["Ventas"].diff().dropna()
    adf_results = [
        adf_summary(train["Ventas"], "Serie original"),
        adf_summary(train_diff, "Serie diferenciada (d=1)"),
    ]
    with open(ARTIFACTS / "adf_results.json", "w", encoding="utf-8") as f:
        json.dump(adf_results, f, ensure_ascii=False, indent=2)

    diff_df = pd.DataFrame({"date": train_diff.index, "diff": train_diff.values})
    diff_df.to_csv(ARTIFACTS / "series_diff.csv", index=False)

    # ------------------------------------------------------------------
    # 3) ACF / PACF sobre la serie diferenciada (Metodologia)
    # ------------------------------------------------------------------
    n_lags = 60
    acf_vals = acf(train_diff, nlags=n_lags, fft=True)
    pacf_vals = pacf(train_diff, nlags=n_lags, method="ywm")
    conf_band = float(1.96 / np.sqrt(len(train_diff)))
    with open(ARTIFACTS / "acf_pacf.json", "w", encoding="utf-8") as f:
        json.dump({
            "lags": list(range(n_lags + 1)),
            "acf": [float(v) for v in acf_vals],
            "pacf": [float(v) for v in pacf_vals],
            "conf_band": conf_band,
        }, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 4) Modelo SARIMA sobre train, evaluado sobre test no visto
    # ------------------------------------------------------------------
    print("Entrenando SARIMA sobre train...")
    modelo = SARIMAX(
        train["Ventas"], order=ORDER, seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    resultado_fit = modelo.fit(disp=False)

    model_summary = {
        "params": [
            {"name": name, "coef": float(resultado_fit.params[name]),
             "std_err": float(resultado_fit.bse[name]), "p_value": float(resultado_fit.pvalues[name])}
            for name in resultado_fit.params.index
        ],
        "aic": float(resultado_fit.aic),
        "bic": float(resultado_fit.bic),
        "llf": float(resultado_fit.llf),
        "n_obs_train": int(len(train)),
    }
    with open(ARTIFACTS / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(model_summary, f, ensure_ascii=False, indent=2)

    # -- Diagnostico de residuos --------------------------------------------
    resid = resultado_fit.resid.dropna()
    lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
    qq_theoretical, qq_sample = stats.probplot(resid, dist="norm", fit=False)
    resid_acf_lags = min(20, len(resid) // 3)
    resid_acf_vals = acf(resid, nlags=resid_acf_lags, fft=True)
    residual_diag = {
        "ljung_box": [
            {"lag": int(idx), "stat": float(row["lb_stat"]), "p_value": float(row["lb_pvalue"])}
            for idx, row in lb.iterrows()
        ],
        "qq_theoretical": [float(v) for v in qq_theoretical],
        "qq_sample": [float(v) for v in qq_sample],
        "acf_lags": list(range(resid_acf_lags + 1)),
        "acf_values": [float(v) for v in resid_acf_vals],
        "acf_conf_band": float(1.96 / np.sqrt(len(resid))),
    }
    with open(ARTIFACTS / "residual_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(residual_diag, f, ensure_ascii=False, indent=2)

    resid_df = pd.DataFrame({"date": resid.index, "residual": resid.values})
    resid_df.to_csv(ARTIFACTS / "residuals.csv", index=False)

    # -- Forecast sobre el horizonte de test + baseline naive estacional ------
    pred = resultado_fit.get_forecast(steps=len(test))
    pred_media = pred.predicted_mean
    pred_ic = pred.conf_int()

    naive_forecast = df["Ventas"].shift(SEASONAL_PERIOD).loc[test.index]

    sarima_metrics = eval_metrics(test["Ventas"], pred_media)
    naive_metrics = eval_metrics(test["Ventas"], naive_forecast)

    metrics = {
        "test_weeks": int(len(test)),
        "test_mean_sales": float(test["Ventas"].mean()),
        "sarima": sarima_metrics,
        "naive_seasonal": naive_metrics,
        "improvement_vs_naive_pct": {
            "mae": float((1 - sarima_metrics["mae"] / naive_metrics["mae"]) * 100),
            "mape": float((1 - sarima_metrics["mape"] / naive_metrics["mape"]) * 100),
            "rmse": float((1 - sarima_metrics["rmse"] / naive_metrics["rmse"]) * 100),
        },
    }
    with open(ARTIFACTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    test_forecast_df = pd.DataFrame({
        "date": test.index,
        "actual": test["Ventas"].values,
        "forecast": pred_media.values,
        "lower95": pred_ic.iloc[:, 0].values,
        "upper95": pred_ic.iloc[:, 1].values,
        "naive_seasonal": naive_forecast.values,
    })
    test_forecast_df.to_csv(ARTIFACTS / "test_forecast.csv", index=False)

    # ------------------------------------------------------------------
    # 5) Reentrenamiento con toda la serie + forecast operativo (20 sem.)
    # ------------------------------------------------------------------
    print("Reentrenando con toda la serie...")
    modelo_final = SARIMAX(
        df["Ventas"], order=ORDER, seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    # Con los 260 datos completos, el heurístico por defecto de statsmodels para
    # arrancar la optimización de un SARIMA estacional (s=52) no converge ("Maximum
    # Likelihood optimization failed to converge") y produce un intervalo de
    # confianza degenerado (del orden de ±40 millones de EUR, aunque la media
    # prevista es correcta). Arrancar desde los parametros ya convergidos del
    # ajuste sobre train soluciona la convergencia sin cambiar el modelo.
    resultado_final = modelo_final.fit(disp=False, start_params=resultado_fit.params.values, maxiter=200)

    forecast = resultado_final.get_forecast(steps=FUTURE_HORIZON)
    fc_media = forecast.predicted_mean
    fc_ic = forecast.conf_int()
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(weeks=1), periods=FUTURE_HORIZON, freq="W-SUN")

    future_forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast": fc_media.values,
        "lower95": fc_ic.iloc[:, 0].values,
        "upper95": fc_ic.iloc[:, 1].values,
    })
    future_forecast_df.to_csv(ARTIFACTS / "future_forecast.csv", index=False)

    with open(ARTIFACTS / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_obs": int(n_obs),
            "start_date": df.index.min().strftime("%Y-%m-%d"),
            "end_date": df.index.max().strftime("%Y-%m-%d"),
            "mean_sales": float(df["Ventas"].mean()),
            "std_sales": float(df["Ventas"].std()),
            "min_sales": float(df["Ventas"].min()),
            "max_sales": float(df["Ventas"].max()),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "train_start": train.index.min().strftime("%Y-%m-%d"),
            "train_end": train.index.max().strftime("%Y-%m-%d"),
            "test_start": test.index.min().strftime("%Y-%m-%d"),
            "test_end": test.index.max().strftime("%Y-%m-%d"),
        }, f, ensure_ascii=False, indent=2)

    with open(ARTIFACTS / "model_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "order": list(ORDER),
            "seasonal_order": list(SEASONAL_ORDER),
            "future_horizon": FUTURE_HORIZON,
            "aic_final": float(resultado_final.aic),
            "bic_final": float(resultado_final.bic),
            "last_date": df.index.max().strftime("%Y-%m-%d"),
        }, f, ensure_ascii=False, indent=2)

    print("\nListo. Resumen (comparar a mano contra el notebook fuente):")
    print(f"  SARIMA        MAE={sarima_metrics['mae']:,.0f} EUR  RMSE={sarima_metrics['rmse']:,.0f} EUR  MAPE={sarima_metrics['mape']:.2f}%")
    print(f"  Naive (t-52)  MAE={naive_metrics['mae']:,.0f} EUR  RMSE={naive_metrics['rmse']:,.0f} EUR  MAPE={naive_metrics['mape']:.2f}%")
    print(f"  Mejora MAPE vs naive: {metrics['improvement_vs_naive_pct']['mape']:.1f}%")
    print(f"  Forecast medio proximas {FUTURE_HORIZON} semanas: {fc_media.mean():,.0f} EUR/semana")


if __name__ == "__main__":
    main()
