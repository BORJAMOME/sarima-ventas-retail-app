"""
SARIMA — Ventas Semanales Retail
Reportaje digital de datos: ¿hasta dónde puede llegar una previsión de ventas usando únicamente el pasado de
la propia serie? Del problema de negocio a la decisión, pasando por los datos, el método, la evidencia y sus límites.

Narrativa (11 beats):
  LEDE → 01 El problema → 02 Los datos → 03 ¿Qué cuentan cinco años de ventas? → 04 Del histórico a una
  previsión → 05 ¿Puede anticipar las ventas? → 06 ¿Qué queda sin explicar? → 07 ¿Qué pasará dentro de 1, 10 o
  20 semanas? → 08 Qué hemos aprendido → 09 ¿Qué podría hacer una empresa? → 10 Hasta dónde llega el modelo →
  11 El futuro no se puede conocer, pero sí estimar mejor

Regla de copy: primero se explica qué significa, después se pone el nombre técnico (en notas
«Detalle técnico» o entre paréntesis).

La composición vive en assets/editorial.css + components/editorial.py (sistema editorial reutilizable); aquí solo
hay contenido y datos. Toda cifra del texto sale de los artefactos de model/artifacts (train.py y export_audit.py)
o se calcula de ellos: nada está escrito a mano. La app no depende de statsmodels ni de scipy en ejecución.

Autor: Borja Mora Méndez
"""
import importlib
from pathlib import Path

import pandas as pd
import streamlit as st

from components import charts
from components import editorial as ed
from utils.data_loader import artifacts_ready, load_csv, load_json

# Streamlit recarga app.py al detectar cambios, pero mantiene en memoria los módulos locales ya
# importados. Tras un despliegue que modifica components/*.py y app.py a la vez, eso deja un
# app.py nuevo llamando a un módulo antiguo (AttributeError). Recargarlos en cada ejecución lo
# evita; el coste es despreciable.
importlib.reload(charts)
importlib.reload(ed)

ROOT = Path(__file__).resolve().parent
LECTURA = "12 min"
ACTUALIZADO = "Septiembre 2026"
FUENTE = "Elaboración propia"
PLOT = {"displayModeBar": False}

st.set_page_config(
    page_title="SARIMA · Ventas Semanales Retail",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Identidad del proyecto primero (paleta, tipografías); sistema editorial después
ed.load_css(ROOT, "style.css", "editorial.css")

if not artifacts_ready():
    st.error(
        "Los artefactos del modelo todavía no se han generado. Ejecuta `py -3.10 model/train.py` y "
        "`py -3.10 model/export_audit.py` desde la raíz del proyecto y recarga esta página."
    )
    st.stop()

dataset_stats = load_json("dataset_stats.json")
model_config = load_json("model_config.json")
model_summary = load_json("model_summary.json")
adf_results = load_json("adf_results.json")
acf_pacf = load_json("acf_pacf.json")
metrics = load_json("metrics.json")
audit = load_json("audit_checks.json")

decomp_df = load_csv("decomposition.csv")
split_df = load_csv("series_split.csv")
diff_df = load_csv("series_diff.csv")
test_df = load_csv("test_forecast.csv")
future_df = load_csv("future_forecast.csv")
baselines_df = load_csv("baselines_test.csv")
resid_clean = load_csv("residuals_clean.csv")
seasonal_df = load_csv("seasonal_by_year.csv")
widths_df = load_csv("interval_widths.csv")

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]


# ---------------------------------------------------------------- formato en español
def es(value: float, decimals: int = 1) -> str:
    """Número con coma decimal, como se escribe en español."""
    return f"{value:.{decimals}f}".replace("-", "−").replace(".", ",")


def sgn(value: float, decimals: int = 1) -> str:
    return f"{value:+.{decimals}f}".replace("-", "−").replace(".", ",")


def pct(value: float, decimals: int = 1) -> str:
    return f"{es(value, decimals)}%"


def eur(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".") + " €"


def fecha(ts, year: bool = True) -> str:
    return f"{ts.day} de {MESES[ts.month - 1]}" + (f" de {ts.year}" if year else "")


def pvalue(p: float) -> str:
    return "p &lt; 0,0001" if p < 0.0001 else f"p = {es(p, 2 if p >= 0.01 else 4)}"


# ---------------------------------------------------------------- cifras del texto
n_obs, n_train, n_test = dataset_stats["n_obs"], dataset_stats["n_train"], dataset_stats["n_test"]
start, end = pd.Timestamp(dataset_stats["start_date"]), pd.Timestamp(dataset_stats["end_date"])
test_start, test_end = pd.Timestamp(dataset_stats["test_start"]), pd.Timestamp(dataset_stats["test_end"])
n_years = round(n_obs / 52)
years_txt = {3: "tres", 4: "cuatro", 5: "cinco", 6: "seis"}.get(n_years, str(n_years))

sarima_m = metrics["sarima"]
year_ago_m = audit["baselines"]["year_ago"]
growth_m = audit["baselines"]["year_ago_growth"]
last_m = audit["baselines"]["last_value"]
improv = metrics["improvement_vs_naive_pct"]
growth_pct = (audit["growth_last_train_year"] - 1) * 100
wins, wilcoxon_p = audit["sarima_better_weeks_vs_growth"], audit["wilcoxon_p_vs_growth"]

desc = audit["descriptive"]
yearly = {int(k): v for k, v in desc["yearly_mean"].items()}
yoy = [v for v in desc["yearly_growth_pct"].values()]
seas = audit["seasonal"]
rd = audit["residuals"]
clean = rd["clean"]
sens = rd["sensitivity"]
iv = audit["intervals"]
params = {p["name"]: p for p in model_summary["params"]}
adf_raw, adf_diff = adf_results[0], adf_results[1]

# ============================================================ LEDE ==
ed.skip_link("contexto")
ed.topbar(
    "SARIMA · Ventas Retail",
    [("contexto", "Problema"), ("datos", "Datos"), ("metodo", "Método"), ("modelo", "Modelo"),
     ("diagnostico", "Diagnóstico"), ("playground", "Playground"), ("resultado", "Resultado"),
     ("decisiones", "Decisiones"), ("conclusion", "Conclusión")],
    back_url="https://borjamora.es/",
)
ed.anchor_scroll()
ed.lede(
    kicker="Machine Learning Case Study · Series temporales",
    headline="¿Cuánto venderás <em>la semana que viene</em>?",
    deck=(
        "Una cadena de supermercados necesita tomar decisiones antes de saber cuánto va a vender: cuánto pedir, "
        "cuánto personal necesitará y cómo prepararse para los picos de demanda. "
        f"Analicé {years_txt} años de ventas semanales para comprobar <b>hasta dónde puede llegar una previsión</b> "
        "utilizando únicamente el pasado de la propia serie."
    ),
    meta=[
        ("Autor", "Borja Mora Méndez"),
        ("Stack", "Python · statsmodels (SARIMA) · Streamlit"),
        ("Datos", f"{n_obs} semanas · {MESES[start.month - 1][:3]} {start.year} – {MESES[end.month - 1][:3]} {end.year}"),
        ("Actualizado", ACTUALIZADO),
        ("Lectura", LECTURA),
    ],
)

# ============================================================ 01 · EL PROBLEMA ==
ed.beat(
    "contexto", "01", "Problema", "El problema",
    deck=[
        "Una cadena de supermercados tiene que tomar decisiones sobre lo que todavía no ha ocurrido.",
        "Cuánto producto pedir. Cuánto personal necesitar. Cuándo prepararse para semanas de mayor demanda.",
        "Si compra de más, puede acumular stock innecesario. Si compra de menos, puede quedarse corto cuando las "
        "ventas suben.",
        "La pregunta no es si podemos saber exactamente cuánto venderemos. Es <b>si podemos anticiparlo</b>.",
    ],
)
ed.band(
    "La pregunta de negocio",
    '¿Podemos anticipar las ventas de las próximas semanas utilizando '
    '<span class="accent">únicamente lo que ya ocurrió</span>?',
    f"Para comprobarlo, trabajé con {years_txt} años de ventas semanales y reservé el último año para ver si el modelo "
    "era capaz de predecirlo sin haberlo visto. No podemos saber exactamente lo que va a pasar, pero quizá sí "
    "anticiparlo mejor.",
)

# ============================================================ 02 · LOS DATOS ==
ed.beat(
    "datos", "02", "Datos", "Los datos",
    deck=[
        f"El punto de partida es sorprendentemente sencillo: <b>{n_obs} semanas de ventas. Una única variable.</b>",
        f"Trabajé con las ventas semanales entre {MESES[start.month - 1]} de {start.year} y "
        f"{MESES[end.month - 1]} de {end.year}.",
    ],
)
ed.provenance([
    ("Dataset", "arima.xlsx"),
    ("Observaciones", f"{n_obs} semanas (domingos), del {fecha(start)} al {fecha(end)}"),
    ("Variable", "Ventas semanales (€)"),
    ("Venta media", f"{eur(dataset_stats['mean_sales'])} por semana"),
    ("Valores vacíos", "0 · sin huecos ni semanas duplicadas"),
    ("Variables externas", "ninguna"),
    ("Elaboración", FUENTE),
])
ed.passage(
    "El fichero no incluye promociones, festivos, climatología, campañas ni ninguna otra variable externa. Solo el "
    "historial de ventas.",
    "¿Es suficiente? Ese era precisamente el reto.",
    "Si existe una señal útil en esos cinco años —una tendencia, una estacionalidad o patrones que se repiten— "
    "debería estar escondida dentro de la propia serie.",
)

# ============================================================ 03 · ¿QUÉ CUENTAN CINCO AÑOS DE VENTAS? ==
ed.beat(
    "exploracion", "03", "Antes de modelar", "¿Qué me cuentan cinco años de ventas?",
    deck=[
        "Antes de construir una previsión, necesitaba saber si había algo que realmente pudiera predecirse.",
        "Al mirar la serie completa aparecen dos señales claras: <b>las ventas crecen con el tiempo y ciertos "
        "patrones se repiten cada año</b>. Eso importa porque una serie que simplemente sube y baja sin patrón sería "
        "mucho más difícil de anticipar.",
    ],
)
ed.subhead("¿Cómo se mueven las ventas?", level="wide")
with ed.split("serie", "8-4") as (viz, txt):
    with viz:
        st.plotly_chart(charts.full_series(split_df), use_container_width=True, config=PLOT)
        ed.caption("FIG. 01", "Ventas semanales (€), de enero de 2020 a diciembre de 2024.", FUENTE)
    with txt:
        ed.insight(
            f"Las ventas medias por semana pasan de {eur(yearly[min(yearly)])} en {min(yearly)} a "
            f"{eur(yearly[max(yearly)])} en {max(yearly)}: entre un {es(min(yoy), 0)}% y un {es(max(yoy), 0)}% más "
            f"cada año. Sobre esa tendencia, una semana típica cambia un {pct(desc['weekly_change_median_pct'])} "
            "respecto a la anterior."
        )

ed.subhead("Separar lo que ocurre", level="wide")
with ed.figure("componentes"):
    ed.cols([
        {"tag": "Tendencia", "title": "Hacia dónde van las ventas", "text": "Cómo evolucionan las ventas a largo plazo."},
        {"tag": "Estacionalidad", "title": "Lo que se repite", "text": "Qué patrones se repiten aproximadamente cada 52 semanas."},
        {"tag": "Residuo", "title": "Lo que queda", "text": "Lo que sobra después de explicar las dos anteriores."},
    ], count=3)
with ed.figure("descomposicion"):
    st.plotly_chart(charts.decomposition(decomp_df), use_container_width=True, config=PLOT)
    ed.caption("FIG. 02", "Descomposición aditiva de la serie con un periodo de 52 semanas.", FUENTE)
ed.insight("La señal más interesante está aquí: <b>la estacionalidad anual</b>. La pregunta es si se repite de verdad "
           "o si es un promedio que parece un patrón.")

ed.subhead("¿Se repite de verdad cada año?", level="wide")
with ed.split("estacionalidad", "8-4") as (viz, txt):
    with viz:
        st.plotly_chart(charts.seasonal_by_year(seasonal_df, seas["full_years"]), use_container_width=True, config=PLOT)
        ed.caption("FIG. 03", "Patrón de cada año una vez quitada la tendencia, alineado por semana del año.", FUENTE)
    with txt:
        ed.insight(
            f"Sí. Quitada la tendencia, el patrón de {', '.join(str(y) for y in seas['full_years'][:-1])} y "
            f"{seas['full_years'][-1]} coincide con una correlación de entre {es(seas['corr_min'], 2)} y "
            f"{es(seas['corr_max'], 2)}: las ventas suben hacia {MESES[seas['peak_month'] - 1]} y bajan hacia "
            f"{MESES[seas['low_month'] - 1]}, con una amplitud de alrededor del {pct(seas['amplitude_pct_of_mean'], 0)} "
            "de la venta media. Es una razón para pensar que el pasado puede contener información útil sobre el futuro."
        )
        ed.note(
            "<b>Detalle técnico.</b> La componente estacional de la descomposición es un promedio que se repite por "
            "construcción, así que la consistencia se comprueba año a año. Solo hay tres años completos: 2020 y 2024 "
            "tienen media serie a cada lado porque la tendencia usa una media móvil centrada de 52 semanas."
        )

# ============================================================ 04 · DEL HISTÓRICO A UNA PREVISIÓN ==
ed.beat(
    "metodo", "04", "Método", "Del histórico a una previsión",
    deck=(
        "En una serie temporal hay una regla que no puedo romper: <b>el futuro no puede enseñarle al modelo cómo "
        "predecir el pasado</b>. Por eso no dividí los datos al azar."
    ),
)
ed.route(["Ventas semanales", "Test ADF", "Diferenciación d=1, D=1", "ACF y PACF", "SARIMA(0,1,1)(0,1,1,52)",
          "Previsión a 20 semanas"])
ed.steps([
    ("Separé el pasado del futuro",
     f"Las primeras {n_train} semanas se utilizaron para entrenar. Las últimas {n_test} quedaron completamente fuera "
     "del entrenamiento para comprobar si el modelo acertaba con datos que nunca había visto."),
    ("Comprobé si la serie podía modelarse directamente",
     f"El test ADF indicó que la serie original no era estacionaria ({pvalue(adf_raw['p_value'])}). En términos "
     "sencillos: su comportamiento cambiaba con el tiempo y necesitaba ser transformada antes de modelarla."),
    ("Eliminé tendencia y estacionalidad",
     f"Apliqué una diferenciación normal (con ella, el ADF sí indica estacionariedad: {pvalue(adf_diff['p_value'])}) "
     "y otra estacional de 52 semanas, para trabajar con una serie más estable."),
    ("Busqué la estructura temporal",
     f"La ACF mostró un pico claro en la primera semana ({es(desc['acf_lag1'], 2)}), que apunta a un término de "
     "media móvil. En las 52 semanas, en cambio, apenas se mueve "
     f"({es(desc['acf_lag52'], 2)}, dentro de la banda de no significación): la estacionalidad anual no sale de ahí, "
     "sale de lo que vimos al explorar los datos. Con esa información propuse un candidato: "
     "<b>SARIMA(0,1,1)(0,1,1,52)</b>. Después lo comprobé con datos que el modelo nunca había visto."),
])

ed.subhead("¿Cómo dividimos el tiempo?", level="wide")
with ed.figure("corte"):
    st.plotly_chart(charts.train_test_split_chart(split_df), use_container_width=True, config=PLOT)
    ed.caption("FIG. 04", f"Las {n_train} primeras semanas para entrenar y las {n_test} últimas como test.", FUENTE)

ed.subhead("¿Por qué diferenciar?", level="wide")
with ed.split("estacionariedad", "7-5") as (viz, txt):
    with viz:
        st.plotly_chart(charts.stationarity_chart(split_df[split_df["split"] == "train"], diff_df),
                        use_container_width=True, config=PLOT)
        ed.caption("FIG. 05", "Serie de entrenamiento antes y después de diferenciarla una vez.", FUENTE)
    with txt:
        ed.insight(
            "Antes de diferenciar, la serie sube con el tiempo: su media no es estable. Después, oscila alrededor de "
            "cero. Eso es lo que necesita el modelo para poder aprender."
        )
        ed.note(f"<b>Detalle técnico.</b> ADF de la serie original: {pvalue(adf_raw['p_value'])}. ADF de la serie "
                f"diferenciada: {pvalue(adf_diff['p_value'])}. La diferenciación estacional (D=1) no se contrastó con un "
                "test específico: se eligió por el patrón anual que se ve en los datos.")

ed.subhead("¿Qué dicen la ACF y la PACF?", level="wide")
with ed.figure("acf"):
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.acf_chart(acf_pacf), use_container_width=True, config=PLOT)
    with right:
        st.plotly_chart(charts.pacf_chart(acf_pacf), use_container_width=True, config=PLOT)
    ed.caption("FIG. 06", "Memoria de la serie diferenciada. Las líneas de puntos marcan el límite de significación; "
                          "en ámbar, el retardo de 52 semanas.", FUENTE)
ed.insight(
    "El modelo final tiene solo dos parámetros: uno que corrige el error de la semana anterior y otro que corrige el "
    "error de hace un año (el mismo punto del ciclo estacional).",
    aside=(
        f"ma.L1 = {es(params['ma.L1']['coef'], 2)} ({pvalue(params['ma.L1']['p_value'])}) y ma.S.L52 = "
        f"{es(params['ma.S.L52']['coef'], 2)} ({pvalue(params['ma.S.L52']['p_value'])}). Ajustado con statsmodels "
        f"sobre las {n_train} semanas de entrenamiento."
    ),
)

# ============================================================ 05 · ¿PUEDE ANTICIPAR LAS VENTAS? ==
ed.beat(
    "modelo", "05", "Modelo", "¿Puede realmente anticipar las ventas?", weight="major",
    deck=[
        f"El modelo se enfrentó a las últimas {n_test} semanas sin haberlas visto durante el entrenamiento.",
        "El resultado:",
    ],
)
ed.metrics([
    ("Error porcentual (MAPE)", pct(sarima_m["mape"], 2),
     f"De cada 100 € que se venden, la previsión se equivoca de media en {es(sarima_m['mape'])} €."),
    ("Error medio (MAE)", eur(sarima_m["mae"]),
     f"De media, la previsión de una semana se desvía {eur(sarima_m['mae'])} sobre una venta media de "
     f"{eur(metrics['test_mean_sales'])} en el año de test."),
    ("Fallos grandes (RMSE)", eur(sarima_m["rmse"]),
     "Como el error medio, pero castiga más las semanas en que el modelo falla por mucho."),
])
ed.insight(
    "Pero una métrica aislada no dice mucho. Por eso lo comparé con una alternativa muy sencilla: "
    "<b>«la próxima semana será como la misma semana del año pasado»</b>."
)

ed.subhead("¿Y frente a repetir el año anterior?", level="wide")
with ed.figure("comparacion"):
    st.plotly_chart(charts.test_forecast_chart(test_df, baselines_df, split_df), use_container_width=True, config=PLOT)
    ed.caption("FIG. 07", f"Ventas reales y previsión de cada método sobre las {n_test} semanas de test. La banda es el "
                          "intervalo de predicción al 95% del SARIMA.", "Modelo entrenado en este proyecto")
ed.metrics([
    ("Repetir el año anterior", pct(year_ago_m["mape"], 2),
     f"Se queda corto {'todas las semanas' if abs(abs(year_ago_m['bias_pct']) - year_ago_m['mape']) < 0.01 else 'casi todas las semanas'} "
     f"({pct(year_ago_m['bias_pct'], 1)} de sesgo): las ventas crecen y esta regla "
     f"lo ignora. Frente a ella, el SARIMA reduce el MAPE un <b>{improv['mape']:.0f}%</b> (MAE {eur(year_ago_m['mae'])} → "
     f"{eur(sarima_m['mae'])}; RMSE {eur(year_ago_m['rmse'])} → {eur(sarima_m['rmse'])})."),
    ("Año anterior ajustado por crecimiento", pct(growth_m["mape"], 2),
     f"La misma regla, multiplicada por el crecimiento del último año de entrenamiento ({sgn(growth_pct)}%). Solo "
     "usa información disponible antes del test."),
    ("SARIMA", pct(sarima_m["mape"], 2),
     f"Mejor que la referencia ajustada en {wins} de {n_test} semanas: un {es(audit['mape_reduction_vs_growth_pct'], 0)}% "
     f"menos de error que no es concluyente con un año de test ({pvalue(wilcoxon_p)})."),
    ("Repetir la última semana", pct(last_m["mape"], 2),
     "Mantener la venta de la última semana de entrenamiento durante todo el año de test."),
])
ed.insight(
    "No se trataba de construir el modelo más complejo. Se trataba de comprobar si podía mejorar una forma muy "
    f"sencilla de anticipar las ventas, y lo hace: un <b>{improv['mape']:.0f}% menos de error</b> que repetir el año "
    "anterior. Pero la mejora no está en que el SARIMA sea sofisticado: está en <b>incorporar la tendencia y la "
    "estacionalidad</b>. Una regla de una línea que también las incorpora llega casi al mismo error.",
    aside=f"La referencia ajustada usa el crecimiento del último año de entrenamiento ({sgn(growth_pct)}%), calculado solo "
          f"con datos anteriores al test, así que no hay fuga de información. Con {n_test} semanas, la diferencia entre "
          f"ambas ({pct(sarima_m['mape'], 2)} frente a {pct(growth_m['mape'], 2)}) queda dentro del ruido.",
)

# ============================================================ 06 · ¿QUÉ QUEDA SIN EXPLICAR? ==
ed.beat(
    "diagnostico", "06", "Diagnóstico", "¿Qué queda sin explicar?",
    deck=[
        "Un forecast no termina cuando aparece una predicción. También hay que mirar qué está dejando atrás el modelo.",
        "Analicé los residuos: la diferencia entre lo que realmente ocurrió y lo que el modelo esperaba.",
    ],
)
with ed.figure("residuos"):
    r1, r2 = st.columns(2), st.columns(2)
    for col, fig in ((r1[0], charts.residuals_time(resid_clean)), (r1[1], charts.residuals_hist(resid_clean)),
                     (r2[0], charts.residuals_qq(resid_clean)), (r2[1], charts.residuals_acf(resid_clean))):
        with col:
            st.plotly_chart(fig, use_container_width=True, config=PLOT)
    ed.caption("FIG. 08", f"Residuos del ajuste sobre el entrenamiento, sin las {rd['burn_in']} primeras semanas.",
               "Modelo entrenado en este proyecto")
ed.insight(
    f"Tras apartar las primeras semanas, en las que el modelo aún no había «arrancado», el test de Ljung-Box no "
    f"detecta autocorrelación significativa ({pvalue(clean['lb10'])} en el lag 10 y {pvalue(clean['lb20'])} en el lag "
    "20) y los residuos quedan centrados cerca de cero. Pero no es un resultado limpio: son p-valores moderados, "
    f"que con otros puntos de corte del arranque oscilan entre {es(min(sens['lb10_min'], sens['lb20_min']), 2)} y "
    f"{es(max(sens['lb10_max'], sens['lb20_max']), 2)}, y "
    f"los residuos se parecen a una normal solo de forma aproximada (Shapiro-Wilk, {pvalue(clean['shapiro_p'])}).",
    aside=(
        f"Con d=1 y D=1 (s=52) las primeras 53 observaciones son inicialización, no error del modelo: el primer "
        f"residuo es {eur(rd['first_residual'])}, la propia venta de la primera semana. Se apartan las "
        f"{rd['burn_in']} primeras (hasta el {fecha(pd.Timestamp(rd['clean_start_date']) - pd.Timedelta(weeks=1))}), "
        f"siempre como un tramo continuo. Con todos los residuos, el lag "
        f"10 daría {pvalue(rd['all_residuals']['lb10'])}: un rastro de autocorrelación que en gran parte es ese arranque."
    ),
)
ed.insight("La precisión importa. Saber dónde deja de ser perfecta también.", tight=True)

# ============================================================ 07 · PLAYGROUND ==
ed.beat(
    "playground", "07", "Playground", "¿Qué pasará dentro de 1, 10 o 20 semanas?", weight="major",
    deck=[
        "El modelo no devuelve una única cifra como si el futuro estuviera escrito. Para cada semana genera una "
        "<b>predicción central</b> y un <b>intervalo de confianza del 95%</b>.",
        "Elige cualquier semana dentro de las próximas 20 y observa cómo cambia la previsión.",
    ],
)
with ed.split("playground", "4-8") as (pg_left, pg_right):
    with pg_left:
        horizon = st.slider("Semana del horizonte (1 = la próxima semana)", 1, model_config["future_horizon"], 10)
    selected = future_df.iloc[horizon - 1]
    width = selected["upper95"] - selected["lower95"]
    with pg_right:
        st.plotly_chart(charts.playground_horizon_chart(split_df, future_df, horizon), use_container_width=True, config=PLOT)
        ed.caption("FIG. 09", "Últimas 30 semanas reales y previsión a 20 semanas con su intervalo al 95%; el punto "
                              "negro es la semana elegida.", "Modelo final entrenado con las 260 semanas")

ed.metrics([
    (f"Semana {horizon} · {selected['date'].strftime('%d/%m/%Y')}", eur(selected["forecast"]),
     "Predicción central: el escenario más probable."),
    ("Intervalo al 95%", f"{eur(selected['lower95'])} – {eur(selected['upper95'])}",
     "El rango en el que esperamos que caiga la venta real."),
    ("Margen", f"±{pct(width / 2 / selected['forecast'] * 100)}", f"Ancho total del intervalo: {eur(width)}."),
], word=True)
ed.insight(
    f"La previsión no dice «venderás exactamente {eur(selected['forecast'])}». Dice: este es el escenario central y "
    "este es el margen de incertidumbre que deberíamos tener en cuenta al planificar."
)

ed.subhead("¿Cuánto crece la incertidumbre?", level="wide")
with ed.figure("incertidumbre"):
    st.plotly_chart(charts.interval_width_chart(widths_df), use_container_width=True, config=PLOT)
    ed.caption("FIG. 10", "Ancho del intervalo al 95% según las semanas hacia delante.", "Modelos entrenados en este proyecto")
ed.insight(
    f"Cuanto más lejos miramos, mayor es la incertidumbre, pero no de forma abrupta en este horizonte: dentro de las "
    f"próximas 20 semanas el intervalo se ensancha un {pct(iv['future_width_growth_pct'])} (de "
    f"{eur(iv['future_width_w1'])} a {eur(iv['future_width_w20'])}), mientras que en las {n_test} semanas del test "
    f"casi se duplica (×{es(iv['test_width_ratio'], 1)}).",
    aside=(
        f"En el test, el intervalo contuvo la venta real en {iv['coverage']} de {iv['n']} semanas (con un 95% de "
        f"confianza cabría esperar unas 49): sugiere que es algo prudente, con un margen medio de "
        f"±{pct(iv['half_width_pct_test'], 0)} frente a un error real del {pct(sarima_m['mape'])}."
    ),
)

# ============================================================ 08 · QUÉ HEMOS APRENDIDO ==
ed.beat(
    "resultado", "08", "Resultado", "El pasado contiene una señal útil", weight="major",
    deck=(
        f"Las {n_test} semanas de test muestran que el historial de ventas puede contener suficiente información "
        "para construir una previsión útil."
    ),
)
ed.subhead("En resumen")
ed.metrics([
    ("Error en el test", pct(sarima_m["mape"], 2),
     f"MAPE del SARIMA sobre {n_test} semanas nunca vistas (MAE {eur(sarima_m['mae'])}, RMSE {eur(sarima_m['rmse'])})."),
    ("Frente al año anterior", f"−{improv['mape']:.0f}%",
     f"Menos error que repetir la misma semana del año pasado ({pct(year_ago_m['mape'], 2)}, {eur(year_ago_m['mae'])}, "
     f"{eur(year_ago_m['rmse'])})."),
    ("Frente a la regla ajustada", f"−{es(audit['mape_reduction_vs_growth_pct'], 0)}%",
     f"Casi empate ({pct(growth_m['mape'], 2)}): con un año de test, la diferencia no es concluyente."),
    ("Intervalo al 95%", f"{iv['coverage']} de {iv['n']}", "Semanas de test en las que la venta real cayó dentro del intervalo."),
])
ed.insight(
    f"Pero el resultado más interesante no es el {es(sarima_m['mape'], 2)}%. Es que la previsión mejora cuando dejamos de asumir que el "
    "futuro será simplemente una repetición del pasado y <b>utilizamos también la tendencia y la estacionalidad que "
    "contiene la serie</b>. El SARIMA es una forma de hacerlo; una regla sencilla ajustada por crecimiento es otra, y "
    "en este test rinde casi igual."
)

# ============================================================ 09 · DECISIONES ==
ed.beat(
    "decisiones", "09", "Implicaciones", "¿Qué podría hacer una empresa con esta previsión?",
    deck=(
        "El modelo no decide cuánto comprar. Pero puede aportar una <b>referencia cuantitativa</b> para tomar esa "
        "decisión con menos incertidumbre."
    ),
)
with ed.figure("hipotesis"):
    ed.cards([
        {"tag": "Hipótesis · Inventario", "title": "Planificación de inventario", "points": [
            f"<b>Insight.</b> La previsión reduce el error frente a repetir el año anterior ({pct(sarima_m['mape'], 2)} "
            f"frente a {pct(year_ago_m['mape'], 2)}), aunque una regla ajustada por crecimiento llega a "
            f"{pct(growth_m['mape'], 2)}.",
            "<b>Hipótesis.</b> Utilizar la previsión como una de las referencias para planificar pedidos semanales, "
            "comparándola siempre con esa regla sencilla.",
            "<b>Qué medir.</b> Error de previsión, sobrestock y roturas de stock.",
        ]},
        {"tag": "Hipótesis · Personal", "title": "Planificación de personal", "points": [
            "<b>Insight.</b> La previsión permite anticipar semanas con mayor o menor volumen esperado.",
            "<b>Hipótesis.</b> Utilizar el forecast para anticipar necesidades de personal y ajustar recursos.",
            "<b>Qué medir.</b> Coste de personal, cobertura y desviación frente a la demanda real.",
        ]},
        {"tag": "Hipótesis · Incertidumbre", "title": "Planificar con incertidumbre", "points": [
            f"<b>Insight.</b> El intervalo de confianza aumenta cuanto más lejos está la fecha prevista "
            f"({sgn(iv['future_width_growth_pct'])}% en 20 semanas; ×{es(iv['test_width_ratio'], 1)} a 52).",
            "<b>Hipótesis.</b> Usar la predicción central como referencia a corto plazo y el rango completo para "
            "escenarios de planificación.",
            "<b>Qué medir.</b> Ancho del intervalo y error real a cada horizonte.",
        ]},
    ], count=3)
ed.subhead("Una advertencia importante")
ed.insight("Son <b>hipótesis de negocio</b>, no conclusiones del modelo. La previsión describe cómo se comportaron las "
           "ventas en estos cinco años; decidir qué hacer requiere probar la acción y medir su efecto.", tight=True)

# ============================================================ 10 · LIMITACIONES ==
ed.beat("limitaciones", "10", "Limitaciones", "Hasta dónde llega el modelo",
        deck="Una previsión útil no es una previsión perfecta.")
ed.cards([
    {"title": "Lo que el modelo sí puede hacer", "color": "var(--positive)", "points": [
        f"Predecir las ventas semanales agregadas con un <b>MAPE del {es(sarima_m['mape'], 2)}%</b> sobre las {n_test} "
        "semanas de test.",
        "Capturar la <b>tendencia y la estacionalidad anual</b> presentes en el histórico.",
        "<b>Mejorar una referencia sencilla</b> basada en repetir el año anterior.",
        "Mostrar un <b>intervalo de confianza</b> para cuantificar la incertidumbre.",
    ]},
    {"title": "Lo que el modelo no puede hacer", "color": "var(--negative)", "points": [
        "Anticipar <b>cambios que no aparecen en el histórico</b>: promociones, crisis, nuevas tiendas o cambios "
        "bruscos en el comportamiento.",
        "<b>Explicar por qué</b> aumentan o disminuyen las ventas.",
        "Predecir qué ocurrirá en una <b>tienda, categoría o producto</b> concreto.",
        "Garantizar el mismo nivel de precisión si <b>cambian las condiciones del negocio</b>.",
        f"Demostrar que es <b>mejor que una regla sencilla</b> ajustada por crecimiento: con {n_test} semanas, "
        f"{pct(sarima_m['mape'], 2)} frente a {pct(growth_m['mape'], 2)} no es concluyente.",
        "Asegurar que el resultado <b>se repita</b>: solo hay un año de test.",
    ]},
])
ed.passage(
    "La siguiente iteración natural sería incorporar variables externas como promociones, festivos o eventos, y "
    "trabajar con una granularidad mayor.",
    "Porque una cosa es predecir cuánto se venderá. <b>Otra muy distinta es entender por qué.</b>",
    aside=(
        f"Un único corte temporal ({n_test} semanas) y sin backtesting rodante: no se sabe cuánto varía el error en "
        "otros años. La previsión final se reajustó con las 260 semanas partiendo de los parámetros del ajuste de "
        "entrenamiento, porque el optimizador no convergía desde cero. Los intervalos suponen residuos normales y "
        "estables, y aquí solo lo son de forma aproximada."
    ),
)

# ============================================================ 11 · CONCLUSIÓN ==
ed.beat(
    "conclusion", "11", "Conclusión", "El futuro no se puede conocer. Pero sí se puede estimar mejor.", weight="major",
    deck=[
        f"Con {years_txt} años de ventas semanales, construí un modelo SARIMA capaz de anticipar las siguientes {n_test} "
        f"semanas con un MAPE del <b>{es(sarima_m['mape'], 2)}%</b> sobre datos que nunca había visto.",
        f"Frente a una previsión basada simplemente en repetir el año anterior, el error se redujo un "
        f"<b>{improv['mape']:.0f}%</b>. Pero una regla sencilla que también tiene en cuenta el crecimiento llega casi "
        "al mismo nivel: lo que funciona es incorporar la tendencia y la estacionalidad, no la complejidad del modelo.",
        "El aprendizaje más importante no es que el modelo acierte. Es que <b>una buena previsión también tiene que "
        "decir cuánto podemos confiar en ella</b>. El forecast aporta una estimación. El intervalo aporta el margen "
        "de incertidumbre. Y la decisión sigue estando en el negocio.",
    ],
)
ed.band(
    "Para cerrar",
    'No se trata de adivinar las ventas. Se trata de llegar a la próxima semana con '
    '<span class="accent">algo más que una intuición</span>.',
)
ed.colophon("Borja Mora Méndez", [
    ("Repositorio del proyecto", "https://github.com/BORJAMOME/sarima-ventas-retail-app"),
    ("Portfolio", "https://borjamora.es/"),
    ("LinkedIn", "https://www.linkedin.com/in/borjamoramendez/"),
    ("Contacto", "mailto:borja.mora.mendez@gmail.com"),
])
