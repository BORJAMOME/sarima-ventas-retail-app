"""Comprobaciones que la narrativa necesita para no afirmar más de lo que los datos permiten.

Lee el dataset y los artefactos que ya genera train.py y guarda artefactos NUEVOS. No reentrena nada: los
artefactos de train.py no cambian.

  audit_checks.json       referencias más exigentes que «el año anterior» (ajustada por crecimiento), sesgo,
                          residuos sin el arranque del filtro, calibración y ensanchamiento del intervalo,
                          consistencia de la estacionalidad año a año y cifras descriptivas
  seasonal_by_year.csv    patrón estacional de cada año (una vez quitada la tendencia), alineado por semana
  baselines_test.csv      predicción de cada referencia sobre las 52 semanas de test
  residuals_clean.csv     residuos del ajuste sin las semanas de arranque
  interval_widths.csv     ancho del intervalo al 95 % según el horizonte (test y previsión final)

app.py solo lee estos archivos: en Streamlit Cloud no necesita statsmodels ni scipy.

Ejecutar una sola vez (o si cambia el dataset, después de train.py):
    py -3.10 model/export_audit.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro, wilcoxon
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "model" / "artifacts"
PERIOD = 52


def metrics(actual: pd.Series, forecast) -> dict:
    err = actual - forecast
    return {"mape": float((err.abs() / actual).mean() * 100), "mae": float(err.abs().mean()),
            "rmse": float(np.sqrt((err ** 2).mean())), "bias_pct": float(((forecast - actual) / actual).mean() * 100)}


def main():
    s = pd.read_excel(ROOT / "data" / "arima.xlsx", parse_dates=["Fecha"]).set_index("Fecha")["Ventas"].asfreq("W-SUN")
    stats = json.load(open(A / "dataset_stats.json"))
    n_train = stats["n_train"]
    train, test = s.iloc[:n_train], s.iloc[n_train:]
    tf = pd.read_csv(A / "test_forecast.csv", parse_dates=["date"])
    sarima = pd.Series(tf["forecast"].values, index=test.index)

    # -- Referencias sobre las mismas 52 semanas: solo con información disponible al terminar el entrenamiento --
    growth = float(train.iloc[-PERIOD:].mean() / train.iloc[-2 * PERIOD:-PERIOD].mean())
    year_ago = s.shift(PERIOD).loc[test.index]
    baselines = {
        "sarima": metrics(test, sarima),
        "year_ago": metrics(test, year_ago),
        "year_ago_growth": metrics(test, year_ago * growth),
        "last_value": metrics(test, pd.Series(train.iloc[-1], index=test.index)),
    }
    ape_s = (test - sarima).abs() / test
    ape_g = (test - year_ago * growth).abs() / test
    pd.DataFrame({"date": test.index, "year_ago": year_ago.values, "year_ago_growth": (year_ago * growth).values,
                  "last_value": train.iloc[-1]}).to_csv(A / "baselines_test.csv", index=False)

    # -- Residuos sin el arranque del filtro --
    # Con d=1 y D=1 (s=52) las primeras 53 observaciones son inicialización difusa: su «error» es la propia
    # venta (123.514 € la primera) y decae después según el término MA. Se aparta el arranque, SIEMPRE como un
    # tramo continuo (quitar observaciones sueltas del medio rompería la estructura temporal que mide
    # Ljung-Box), hasta que el residuo vuelve a la escala normal (3 desviaciones robustas). Como el resultado
    # depende algo del punto de corte, se guarda también el rango que salía con cualquier corte posterior.
    res = pd.read_csv(A / "residuals.csv", parse_dates=["date"])
    r = res["residual"]
    center = float(np.median(r.iloc[70:]))
    robust_sd = float(1.4826 * np.median(np.abs(r.iloc[70:] - center)))
    flagged = [i for i in range(70) if abs(r.iloc[i] - center) > 3 * robust_sd]
    burn_in = max(flagged) + 1 if flagged else 0

    def diag(x: pd.Series) -> dict:
        lb = acorr_ljungbox(x, lags=[10, 20], return_df=True)
        sw = shapiro(x)
        return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std()),
                "lb10": float(lb["lb_pvalue"].iloc[0]), "lb20": float(lb["lb_pvalue"].iloc[1]),
                "shapiro_w": float(sw[0]), "shapiro_p": float(sw[1]), "max_abs_sd": float(x.abs().max() / x.std())}

    clean = res.iloc[burn_in:].reset_index(drop=True)
    clean.to_csv(A / "residuals_clean.csv", index=False)
    cuts = [diag(r.iloc[k:]) for k in range(burn_in, 105)]
    residuals = {
        "n_total": int(len(r)), "burn_in": int(burn_in), "first_residual": float(r.iloc[0]),
        "first_date": res["date"].iloc[0].strftime("%Y-%m-%d"), "clean_start_date": clean["date"].iloc[0].strftime("%Y-%m-%d"),
        "clean": diag(clean["residual"]),
        "all_residuals": diag(r), "without_diffuse_only": diag(r.iloc[53:]),
        "sensitivity": {"cuts": f"{burn_in}-104", "lb10_min": min(c["lb10"] for c in cuts), "lb10_max": max(c["lb10"] for c in cuts),
                        "lb20_min": min(c["lb20"] for c in cuts), "lb20_max": max(c["lb20"] for c in cuts),
                        "share_lb_above_5pct": float(np.mean([min(c["lb10"], c["lb20"]) > 0.05 for c in cuts]))},
    }

    # -- Intervalo al 95 %: calibración en test y ensanchamiento con el horizonte --
    fu = pd.read_csv(A / "future_forecast.csv", parse_dates=["date"])
    w_test, w_future = tf["upper95"] - tf["lower95"], fu["upper95"] - fu["lower95"]
    pd.DataFrame({"horizon": range(1, len(w_test) + 1), "width_test": w_test.values,
                  "width_future": list(w_future.values) + [np.nan] * (len(w_test) - len(w_future))}
                 ).to_csv(A / "interval_widths.csv", index=False)
    inside = (tf["actual"] >= tf["lower95"]) & (tf["actual"] <= tf["upper95"])
    intervals = {
        "coverage": int(inside.sum()), "n": int(len(tf)),
        "half_width_pct_test": float((w_test / 2 / tf["forecast"]).mean() * 100),
        "test_width_w1": float(w_test.iloc[0]), "test_width_w52": float(w_test.iloc[-1]),
        "test_width_ratio": float(w_test.iloc[-1] / w_test.iloc[0]),
        "future_width_w1": float(w_future.iloc[0]), "future_width_w10": float(w_future.iloc[9]),
        "future_width_w20": float(w_future.iloc[-1]), "future_width_growth_pct": float((w_future.iloc[-1] / w_future.iloc[0] - 1) * 100),
    }

    # -- Consistencia de la estacionalidad: patrón de cada año sin tendencia, alineado por semana --
    dec = pd.read_csv(A / "decomposition.csv", parse_dates=["date"]).set_index("date")
    det = (dec["observed"] - dec["trend"]).dropna()
    d = pd.DataFrame({"v": det.values}, index=det.index)
    d["year"], d["week"] = d.index.year, d.index.isocalendar().week.values.astype(int)
    d = d[d["week"] <= PERIOD]
    piv = d.pivot_table(index="week", columns="year", values="v")
    full = [int(c) for c in piv.columns if piv[c].count() >= 50]
    piv.reset_index().to_csv(A / "seasonal_by_year.csv", index=False)
    pair = [piv[a].corr(piv[b]) for i, a in enumerate(full) for b in full[i + 1:]]
    seas = dec["seasonal"].iloc[:PERIOD]
    seasonal = {
        "full_years": full, "partial_years": [int(c) for c in piv.columns if c not in full],
        "corr_min": float(min(pair)), "corr_max": float(max(pair)), "corr_mean": float(np.mean(pair)),
        "amplitude_pct_of_mean": float((dec["seasonal"].max() - dec["seasonal"].min()) / s.mean() * 100),
        "peak_month": int(seas.idxmax().month), "low_month": int(seas.idxmin().month),
    }

    # -- Cifras descriptivas y lo que dicen ACF/PACF --
    yearly = s.groupby(s.index.year).mean()
    ap = json.load(open(A / "acf_pacf.json"))
    acf, band = np.array(ap["acf"]), ap["conf_band"]
    sig_lags = [i for i in range(1, len(acf)) if abs(acf[i]) > band]
    descriptive = {
        "yearly_mean": {int(k): float(v) for k, v in yearly.items()},
        "yearly_growth_pct": {int(k): float(v * 100) for k, v in yearly.pct_change().dropna().items()},
        "weekly_change_median_pct": float(s.pct_change().abs().median() * 100),
        "acf_lag1": float(acf[1]), "acf_lag52": float(acf[PERIOD]), "acf_band": float(band),
        "acf_significant_lags": sig_lags, "acf_significant_near_52": [i for i in sig_lags if 48 <= i <= 56],
    }

    checks = {
        "growth_last_train_year": growth, "baselines": baselines,
        "sarima_better_weeks_vs_growth": int((ape_s < ape_g).sum()), "n_test": int(len(test)),
        "wilcoxon_p_vs_growth": float(wilcoxon(ape_g, ape_s).pvalue),
        "mape_reduction_vs_growth_pct": float((1 - baselines["sarima"]["mape"] / baselines["year_ago_growth"]["mape"]) * 100),
        "residuals": residuals, "intervals": intervals, "seasonal": seasonal, "descriptive": descriptive,
    }
    with open(A / "audit_checks.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, ensure_ascii=False, indent=2)

    b = baselines
    print("audit_checks.json y 4 csv guardados")
    print(f"  MAPE: SARIMA {b['sarima']['mape']:.2f} | año anterior {b['year_ago']['mape']:.2f} | "
          f"año anterior x crecimiento ({(growth - 1) * 100:.1f}%) {b['year_ago_growth']['mape']:.2f} | última semana {b['last_value']['mape']:.2f}")
    print(f"  SARIMA mejor en {checks['sarima_better_weeks_vs_growth']}/52 semanas vs referencia ajustada (Wilcoxon p={checks['wilcoxon_p_vs_growth']:.2f})")
    c = residuals["clean"]
    sn = residuals["sensitivity"]
    print(f"  residuos: arranque {burn_in} semanas; Ljung-Box p10={c['lb10']:.3f} p20={c['lb20']:.3f}; Shapiro p={c['shapiro_p']:.2f}; rango con otros cortes p10 {sn['lb10_min']:.2f}-{sn['lb10_max']:.2f}, p20 {sn['lb20_min']:.2f}-{sn['lb20_max']:.2f}")
    print(f"  intervalo: cobertura {intervals['coverage']}/{intervals['n']}; +{intervals['future_width_growth_pct']:.1f}% en 20 sem., x{intervals['test_width_ratio']:.2f} en 52")
    print(f"  estacionalidad: años completos {full}, correlación {seasonal['corr_min']:.2f}-{seasonal['corr_max']:.2f}; ACF(52)={descriptive['acf_lag52']:.3f} (banda {band:.3f})")


if __name__ == "__main__":
    main()
