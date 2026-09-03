"""
SARIMA — Predicción de Ventas Semanales en Retail
Case study interactivo en Streamlit: de la pregunta de negocio a la
decisión, pasando por los datos, el modelo y su explicabilidad.

Autor: Borja Mora Méndez
"""
from pathlib import Path

import streamlit as st

from components import charts, ui
from utils.data_loader import artifacts_ready, load_csv, load_json

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="SARIMA · Ventas Semanales Retail",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

with open(ROOT / "assets" / "style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if not artifacts_ready():
    st.error(
        "Los artefactos del modelo todavía no se han generado. "
        "Ejecuta `py -3.10 model/train.py` desde la raíz del proyecto y recarga esta página."
    )
    st.stop()

dataset_stats = load_json("dataset_stats.json")
model_config = load_json("model_config.json")
model_summary = load_json("model_summary.json")
adf_results = load_json("adf_results.json")
acf_pacf = load_json("acf_pacf.json")
metrics = load_json("metrics.json")
residual_diag = load_json("residual_diagnostics.json")

decomp_df = load_csv("decomposition.csv")
split_df = load_csv("series_split.csv")
diff_df = load_csv("series_diff.csv")
resid_df = load_csv("residuals.csv")
test_forecast_df = load_csv("test_forecast.csv")
future_df = load_csv("future_forecast.csv")

n_obs = dataset_stats["n_obs"]
n_years = round((split_df["date"].max() - split_df["date"].min()).days / 365, 1)
cv_pct = dataset_stats["std_sales"] / dataset_stats["mean_sales"] * 100
sarima_m = metrics["sarima"]
naive_m = metrics["naive_seasonal"]
improv = metrics["improvement_vs_naive_pct"]


def eur(value: float, sign: bool = False) -> str:
    """Euros con separador de miles europeo — solo sobre el número aislado,
    nunca aplicado a una frase completa (evita mezclar con comas gramaticales)."""
    spec = f"{value:+,.0f}" if sign else f"{value:,.0f}"
    return spec.replace(",", ".") + " €"


ui.nav()
ui.install_smooth_scroll()

# ============================================================ HERO ==
st.markdown(
    f"""
    <div id="top" class="hero-wrap">
      <p class="hero-kicker">Machine Learning Case Study · Series Temporales</p>
      <h1 class="hero-title">Pedir de más llena el almacén de producto que nadie compra. Pedir de menos vacía las estanterías justo cuando más se vende.</h1>
      <p class="hero-sub">Así que dejé que cinco años de ventas contaran su propia historia: acertaron 3 veces mejor que la intuición de repetir el año anterior.</p>
      <div class="hero-meta">
        <span class="hero-pill">Borja Mora Méndez</span>
        <span class="hero-pill">Python · statsmodels (SARIMA)</span>
        <span class="hero-pill">Streamlit</span>
        <span class="hero-pill">{n_obs} semanas de histórico</span>
      </div>
      <div class="hero-scroll-row">
        <a href="#contexto" class="hero-scroll">explorar el caso &#8595;</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================ CONTEXTO ==
ui.section_open("contexto")
ui.eyebrow("Contexto")
ui.h2("El problema")
ui.lead(
    "Una cadena de supermercados necesita anticipar las ventas de cada semana para "
    "planificar inventario, personal y promociones. Comprar de más inmoviliza dinero "
    "en almacén y genera merma; comprar de menos vacía estanterías justo cuando hay "
    "demanda. Ambos errores cuestan dinero, y la única forma de evitarlos es tener una "
    "estimación fiable de lo que se va a vender antes de que ocurra."
)
ui.kpi_grid([
    {"num": f"{n_obs}", "label": "semanas de histórico"},
    {"num": f"{n_years}", "label": "años de operación (2020–2024)"},
    {"num": f"±{cv_pct:.0f}%", "label": "variación semana a semana"},
    {"num": "0", "label": "variables externas disponibles"},
])
st.write("")
ui.question_block(
    "La pregunta de negocio",
    '¿Se pueden anticipar las ventas de las próximas semanas mirando '
    '<span class="accent">solo su propio pasado</span>?',
    "No usé encuestas, ni promociones anunciadas de antemano, ni ningún dato externo: "
    "solo la tendencia y el patrón que ya estaba escondido en el propio histórico de ventas.",
)
ui.section_close()

# ============================================================ DATOS ==
ui.section_open("datos")
ui.eyebrow("Materia prima")
ui.h2("Los datos")
ui.lead(
    f"Trabajé con {n_obs} observaciones semanales de ventas en euros, entre {dataset_stats['start_date']} "
    f"y {dataset_stats['end_date']}: una sola columna, sin huecos ni duplicados. El reto no estuvo en "
    "limpiar los datos, sino en decidir si esa única serie bastaba para predecir con precisión suficiente."
)
ui.kpi_grid([
    {"num": f"{n_obs}", "label": "semanas, frecuencia semanal"},
    {"num": eur(dataset_stats["mean_sales"]), "label": "venta media semanal"},
    {"num": "0", "label": "valores nulos"},
    {"num": "1", "label": "variable objetivo: ventas (€)"},
])
ui.pipeline(["Ventas semanales crudas", "Test de estacionariedad (ADF)", "Diferenciación d=1, D=1",
             "ACF / PACF", "SARIMA(0,1,1)(0,1,1,52)", "Forecast a 20 semanas"])
ui.finding(
    f"El rango de ventas va de {eur(dataset_stats['min_sales'])} a {eur(dataset_stats['max_sales'])} por "
    f"semana, con una variación media del <b>±{cv_pct:.0f}%</b> de una semana a otra. No hay ninguna señal "
    "externa registrada (promociones, festivos, climatología), así que tuve que trabajar solo con el "
    "propio pasado de la serie."
)
ui.section_close()

# ============================================================ EXPLORACIÓN ==
ui.section_open("exploracion")
ui.eyebrow("Antes de modelar")
ui.h2("¿Qué me dicen los datos?")
ui.lead(
    "Antes de entrenar nada, me hice dos preguntas simples: ¿cómo se mueven las ventas a lo largo del "
    "tiempo? Y si descompongo la serie, ¿aparece con claridad la estacionalidad anual que necesito para "
    "elegir un SARIMA en vez de un ARIMA simple?"
)
ui.h3("Evolución de las ventas — serie completa")
st.plotly_chart(charts.full_series(split_df), use_container_width=True, config={"displayModeBar": False})
ui.finding(
    f"Tendencia al alza sostenida a lo largo de los {n_years} años, con oscilaciones que se repiten con "
    "una cadencia claramente anual. Ninguna caída ni pico parece aislado del patrón general: buena señal "
    "antes de empezar a modelar."
)
ui.h3("Descomposición de la serie")
st.plotly_chart(charts.decomposition(decomp_df), use_container_width=True, config={"displayModeBar": False})
ui.finding(
    "La descomposición aditiva (periodo de 52 semanas) separa con nitidez una <b>tendencia</b> creciente, "
    "una <b>estacionalidad</b> anual que se repite prácticamente igual año tras año, y un <b>residuo</b> "
    "que oscila sin patrón visible alrededor de cero. Con esto confirmé que hacía falta un componente "
    "estacional (SARIMA): un ARIMA simple no me habría bastado."
)
ui.section_close()

# ============================================================ METODOLOGÍA ==
ui.section_open("metodologia")
ui.eyebrow("Cómo se llegó al modelo")
ui.h2("El camino hasta el modelo")
ui.lead(
    "En series temporales no se puede barajar el tiempo. Antes de nada, tuve que comprobar si la serie "
    "se podía modelar tal cual o había que transformarla, respetando siempre el orden en que ocurrieron "
    "las cosas."
)
ui.story_steps([
    ("Separé en el tiempo, no al azar",
     f"Reservé el {int(dataset_stats['n_test'])*100//n_obs}% final de las semanas "
     f"({dataset_stats['n_test']} semanas, de {dataset_stats['test_start']} a {dataset_stats['test_end']}) "
     "como test. El modelo solo podía aprender del pasado."),
    ("Comprobé que la serie no era estacionaria",
     f"El test de Dickey-Fuller (ADF) sobre la serie original dio p={adf_results[0]['p_value']:.4f}: no "
     "había evidencia de estacionariedad, así que no podía ajustar un SARIMA directamente."),
    ("Diferencié hasta conseguirlo",
     "Con una diferenciación de primer orden (d=1), el ADF bajó a p<0,0001: la serie diferenciada sí era "
     "estacionaria. Apliqué ese mismo criterio a la parte estacional y obtuve D=1."),
    ("Leí ACF y PACF para proponer un candidato",
     "Los picos alrededor de múltiplos de 52 semanas en la ACF confirmaban la estacionalidad anual. A "
     "partir de ahí propuse SARIMA(0,1,1)(0,1,1,52) como candidato, y lo validé después con los residuos, "
     "no de forma mecánica."),
])
st.write("")
st.plotly_chart(charts.train_test_split_chart(split_df), use_container_width=True, config={"displayModeBar": False})
ui.finding(
    f"Así dividí las {n_obs} semanas: {dataset_stats['n_train']} para entrenar y las últimas "
    f"{dataset_stats['n_test']} para comprobar honestamente si el modelo acertaba, nunca al revés."
)

with st.expander("Para quien quiera el detalle técnico — ADF, diferenciación, ACF/PACF"):
    st.markdown("Test de Dickey-Fuller Aumentado (ADF), antes y después de diferenciar:")
    adf_cols = st.columns(len(adf_results))
    for c, res in zip(adf_cols, adf_results):
        with c:
            st.metric(res["name"], "Estacionaria" if res["is_stationary"] else "No estacionaria",
                      f'p={res["p_value"]:.4f}')
    st.plotly_chart(charts.stationarity_chart(split_df[split_df["split"] == "train"], diff_df),
                     use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        "Con la serie ya estacionaria, usé la ACF y la PACF para orientar la elección de `p`, `d`, `q` "
        "(y su versión estacional `P`, `D`, `Q`, con periodo 52):"
    )
    st.plotly_chart(charts.acf_pacf_chart(acf_pacf), use_container_width=True, config={"displayModeBar": False})
ui.section_close()

# ============================================================ MODELO ==
ui.section_open("modelo")
ui.eyebrow("¿Funciona de verdad?")
ui.h2("SARIMA(0,1,1)(0,1,1,52)")
ui.lead(
    "Entrené el modelo solo con las semanas de train y lo evalué sobre las 52 semanas de test que nunca "
    "había visto. La pregunta que quería responder: ¿acierta lo suficiente como para sustituir a la "
    "intuición de \"esto se parece a lo del año pasado\"?"
)
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("MAE — error medio", eur(sarima_m["mae"]),
              f'-{improv["mae"]:.0f}% vs asumir "año pasado"', delta_color="inverse")
with m2:
    st.metric("MAPE — error %", f'{sarima_m["mape"]:.2f}%',
              f'-{improv["mape"]:.0f}% vs asumir "año pasado"', delta_color="inverse")
with m3:
    st.metric("RMSE", eur(sarima_m["rmse"]),
              f'-{improv["rmse"]:.0f}% vs asumir "año pasado"', delta_color="inverse")

with st.expander("¿Qué significan MAE, MAPE y RMSE, y qué hace exactamente el modelo?"):
    st.markdown(
        "- **MAE** — de media, en cuántos euros me equivoco por semana.\n"
        "- **MAPE** — el mismo error, en porcentaje. Permite comparar entre negocios de distinto tamaño; "
        "por debajo del 10% se considera un pronóstico bueno en retail.\n"
        "- **RMSE** — como el MAE, pero penaliza más los errores grandes.\n\n"
        "Me bastaron **2 parámetros** para capturar toda la dinámica de la serie:"
    )
    for p in model_summary["params"]:
        if p["name"] == "sigma2":
            continue
        label = "Corrección del error de la semana anterior" if p["name"] == "ma.L1" else \
                "Corrección del error de hace un año (mismo ciclo estacional)"
        st.markdown(f"- **{label}** (`{p['name']}` = {p['coef']:.3f}, p={p['p_value']:.4f}, significativo)")

st.write("")
ui.h3("Predicción vs realidad, sobre las 52 semanas de test")
st.plotly_chart(charts.test_forecast_chart(test_forecast_df, split_df), use_container_width=True,
                 config={"displayModeBar": False})
sarima_mape_txt = ui.stat(f"{sarima_m['mape']:.2f}%", "pos")
naive_mape_txt = ui.stat(f"{naive_m['mape']:.2f}%", "neg")
ui.finding(
    f"Sobre una venta media de test de {eur(metrics['test_mean_sales'])}/semana, mi SARIMA se desvía de "
    f"media solo {sarima_mape_txt}, frente al {naive_mape_txt} "
    "de asumir que la semana se repite igual que hace un año: más de <b>3 veces peor</b>. El patrón de "
    "crecimiento que capté con el modelo (la tendencia) es justo lo que a esa regla ingenua se le escapa."
)
ui.section_close()

# ============================================================ EXPLICABILIDAD ==
ui.section_open("explicabilidad")
ui.eyebrow("¿Por qué puedo fiarme?")
ui.h2("Explicabilidad y diagnóstico")
ui.lead(
    "El SARIMA no es una caja negra: sus 2 parámetros capturan exactamente la tendencia y la "
    "estacionalidad anual que ya vi en la descomposición. Para fiarme de la predicción, tenía que "
    "comprobar que no quedaba señal sin explicar en lo que sobra: los residuos."
)
st.plotly_chart(charts.residual_diagnostics(resid_df, residual_diag), use_container_width=True,
                 config={"displayModeBar": False})
lag10 = next(r for r in residual_diag["ljung_box"] if r["lag"] == 10)
lag20 = next(r for r in residual_diag["ljung_box"] if r["lag"] == 20)
ui.finding(
    f"El test de Ljung-Box no encontró autocorrelación significativa en el lag 20 "
    f"(p={lag20['p_value']:.4f}), y el histograma y el QQ-plot muestran residuos razonablemente centrados "
    f"en cero. <b>Soy honesto</b>: en el lag 10 sí aparece un rastro puntual de autocorrelación "
    f"(p={lag10['p_value']:.4f}). No es ruido perfectamente blanco, aunque no compromete la precisión "
    "que observé sobre el conjunto de test."
)
ui.section_close()

# ============================================================ PLAYGROUND ==
ui.section_open("playground")
ui.eyebrow("Pruébalo tú mismo")
ui.h2("Playground — explora el forecast a 20 semanas")
ui.lead(
    "Elige una semana dentro del horizonte que ya calculé y observa la predicción junto a su intervalo "
    "de confianza del 95%: fíjate en cómo esa incertidumbre crece cuanto más lejos miras."
)

pg_left, pg_right = st.columns([1, 1.3], gap="large")
with pg_left:
    horizon = st.slider("Semana del horizonte (1 = la próxima semana)", 1, model_config["future_horizon"], 10)
    selected = future_df.iloc[horizon - 1]
    width = selected["upper95"] - selected["lower95"]
    width_pct = width / selected["forecast"] * 100
    st.write("")
    ui.stat_card("Predicción central", eur(selected["forecast"]), selected["date"].strftime("%d/%m/%Y"))
    c1, c2 = st.columns(2)
    with c1:
        ui.stat_card("Límite inferior", eur(selected["lower95"]), "IC 95%", color=charts.NEGATIVE, value_size="1.3rem")
    with c2:
        ui.stat_card("Límite superior", eur(selected["upper95"]), "IC 95%", color=charts.POSITIVE, value_size="1.3rem")

with pg_right:
    st.plotly_chart(charts.playground_horizon_chart(split_df, future_df, horizon), use_container_width=True,
                     config={"displayModeBar": False})

st.write("")
if horizon <= 5:
    plazo_txt = (
        f"A tan corto plazo (semana {horizon}), el intervalo es estrecho: unos {eur(width)} de ancho "
        f"(±{width_pct/2:.1f}% sobre la predicción central). Suficiente precisión para comprometerte con "
        "un pedido concreto."
    )
elif horizon <= 14:
    plazo_txt = (
        f"A {horizon} semanas vista, el intervalo ya se ha ensanchado a unos {eur(width)} "
        f"(±{width_pct/2:.1f}%): todavía útil para dimensionar inventario y personal, con más margen de "
        "seguridad que a corto plazo."
    )
else:
    plazo_txt = (
        f"A {horizon} semanas vista, el intervalo llega a unos {eur(width)} de ancho "
        f"(±{width_pct/2:.1f}%). Bueno para el orden de magnitud y la planificación general, pero ya poco "
        "fiable para comprometerte con una cifra exacta de pedido."
    )
ui.finding(plazo_txt)
ui.section_close()

# ============================================================ RESULTADOS ==
ui.section_open("resultados")
ui.eyebrow("El hallazgo, resumido")
ui.h2("Resultados")
r1, r2, r3 = st.columns(3)
with r1:
    ui.stat_card("MAE", eur(sarima_m["mae"]), "SARIMA", color=charts.POSITIVE)
    st.write("")
    ui.stat_card("MAE", eur(naive_m["mae"]), 'asumir "año pasado"', color=charts.MUTED)
with r2:
    ui.stat_card("MAPE", f'{sarima_m["mape"]:.2f}%', "SARIMA", color=charts.POSITIVE)
    st.write("")
    ui.stat_card("MAPE", f'{naive_m["mape"]:.2f}%', 'asumir "año pasado"', color=charts.MUTED)
with r3:
    ui.stat_card("RMSE", eur(sarima_m["rmse"]), "SARIMA", color=charts.POSITIVE)
    st.write("")
    ui.stat_card("RMSE", eur(naive_m["rmse"]), 'asumir "año pasado"', color=charts.MUTED)
st.write("")
improv_mape_txt = ui.stat(f"{improv['mape']:.0f}%", "pos")
ui.finding(
    f"Respondo a la pregunta de negocio del principio: <b>sí</b>. El propio historial de ventas basta "
    f"para predecir con un error medio de solo {sarima_mape_txt} sobre datos nunca vistos, sin necesitar "
    f"ninguna señal externa, y mejora la intuición de repetir el año anterior en {improv_mape_txt} menos "
    "error."
)
ui.section_close()

# ============================================================ IMPACTO ==
ui.section_open("impacto", tight=True)
ui.impact_banner(
    f'Asumir que la semana "se repite como el año pasado" cuesta un '
    f'<span class="accent-neg">{improv["mape"]:.0f}% más de error</span> de previsión. '
    f'El modelo lo reduce a solo un <span class="accent-pos">{sarima_m["mape"]:.2f}%</span>.',
    quote='"El modelo no elimina la incertidumbre — la cuantifica, para planificar con ella en vez de a pesar de ella."',
)
ui.section_close()

# ============================================================ DECISIONES ==
ui.section_open("decisiones")
ui.eyebrow("¿Qué haría con esto?")
ui.h2("Decisiones que habilita")
ui.decision_flow(
    f'El error del modelo ({sarima_m["mape"]:.2f}%) es {improv["mape"]:.0f}% menor que repetir el año anterior',
    "Basar el pedido semanal a proveedores en el forecast SARIMA, no en la venta de la misma semana del año pasado",
    "Reducir sobrestock y roturas de stock",
    "MAPE de previsión semanal",
)
st.write("")
ui.decision_flow(
    "El intervalo de confianza se ensancha cuanto más lejos mira el forecast",
    "Usar el punto central para pedidos a 1-5 semanas y el rango completo (no una cifra exacta) para "
    "planificación a 3+ meses",
    "Evitar comprometerse con cifras exactas en horizontes largos",
    "Ancho del intervalo de confianza (95%)",
)
st.write("")
ui.decision_flow(
    f"Queda un rastro de autocorrelación en el lag 10 de los residuos (p={lag10['p_value']:.4f})",
    "Vigilar semanalmente el error real vs el previsto y reentrenar si se desvía de forma sostenida",
    "Detectar cambios estructurales (aperturas, competencia, hábitos) antes de que el forecast se degrade",
    "Error real vs esperado, en ventana móvil",
)
ui.section_close()

# ============================================================ LIMITACIONES ==
ui.section_open("limitaciones")
ui.eyebrow("Honestidad ante todo")
ui.h2("Limitaciones")
st.write("")
lc1, lc2 = st.columns(2, gap="large")
with lc1:
    st.markdown('<p class="limit-col-title">Lo que el modelo SÍ puede hacer</p>', unsafe_allow_html=True)
    st.markdown(
        f"""<ul class="limit-list">
        <li>Predecir ventas semanales agregadas con un error medio del {sarima_m['mape']:.2f}% sobre datos nunca vistos.</li>
        <li>Capturar tendencia y estacionalidad anual (52 semanas) sin necesitar ninguna variable externa.</li>
        <li>Superar ampliamente una previsión ingenua basada en repetir el año anterior ({improv['mape']:.0f}% menos error).</li>
        <li>Cuantificar la incertidumbre de cada semana futura con un intervalo de confianza del 95%.</li>
        </ul>""",
        unsafe_allow_html=True,
    )
with lc2:
    st.markdown('<p class="limit-col-title">Lo que el modelo NO puede hacer</p>', unsafe_allow_html=True)
    st.markdown(
        """<ul class="limit-list">
        <li>Anticipar eventos que rompan el patrón histórico (una promoción agresiva, una crisis, un nuevo local) — solo asume que el patrón se repite.</li>
        <li>Bajar a nivel de tienda, categoría o producto — trabaja sobre la cifra agregada semanal.</li>
        <li>Explicar por qué suben o bajan las ventas — solo el cuánto, no el porqué.</li>
        <li>Garantizar el mismo error si cambian las condiciones estructurales del negocio.</li>
        </ul>""",
        unsafe_allow_html=True,
    )
st.markdown(
    '<div class="limit-note"><p class="co-body">'
    "Para mejorar el modelo, incorporaría granularidad diaria o por tienda, y variables externas "
    "(promociones, festivos, eventos) que expliquen el porqué de las variaciones, no solo que las "
    "predigan: un camino natural para la siguiente iteración de este análisis. También quiero revisar el "
    f"pequeño rastro de autocorrelación detectado en el lag 10 de los residuos (p={lag10['p_value']:.4f}): "
    "no compromete la precisión observada, pero sugiere que queda un margen mínimo de estructura por capturar."
    "</p></div>",
    unsafe_allow_html=True,
)
ui.section_close()

# ============================================================ CONCLUSIÓN ==
ui.section_open("conclusion")
ui.eyebrow("Del dato a la decisión")
ui.h2("Conclusión")
ui.lead(
    f"Sí: el propio historial de ventas basta para predecir con un error medio del {sarima_m['mape']:.2f}% "
    f"sobre 52 semanas nunca vistas, {improv['mape']:.0f}% mejor que asumir que la semana se repite como "
    "el año pasado. El modelo no sustituye la decisión de comprar o de planificar personal: la apoya con "
    "mejor información y con una cifra de incertidumbre explícita en vez de una corazonada."
)
ui.section_close()

ui.footer_minimal(
    name="Borja Mora Méndez",
    repo_url="https://github.com/BORJAMOME/sarima-ventas-retail-app",
    linkedin_url="https://www.linkedin.com/in/borja-mora-mendez/",
    email="borja.mora.mendez@gmail.com",
)
