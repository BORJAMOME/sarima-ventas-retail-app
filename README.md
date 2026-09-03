# SARIMA — Predicción de Ventas Semanales en Retail

**Pedir de más llena el almacén de producto que nadie compra. Pedir de menos vacía las estanterías justo cuando más se vende.**

Así que dejamos que cinco años de ventas contaran su propia historia — y acertaron 3
veces mejor que la intuición de repetir el año anterior. Una aplicación interactiva
que recorre, paso a paso, cómo se construyó un modelo de previsión de ventas que no
necesita ninguna señal externa — ni promociones, ni festivos, ni eventos — para
anticipar la demanda semanal con un error medio del 2,21%.

No hace falta saber nada de Machine Learning para seguirla: empieza por el problema,
sigue por los datos, y termina dejándote explorar tú mismo el forecast de las próximas
20 semanas, con su intervalo de confianza.

## Ver la app

🔗 **Pendiente de desplegar en Streamlit Cloud**

## De qué trata, en dos frases

Una cadena retail necesita saber cuánto va a vender cada semana para planificar
inventario y personal. Se entrenó un modelo **SARIMA(0,1,1)(0,1,1,52)** sobre 208
semanas de histórico y se evaluó sobre 52 semanas nunca vistas.

**El resultado:** un error medio (MAPE) del **2,21%**, un 73,7% menos que asumir que
la semana se repite igual que el año anterior.

## Qué te vas a encontrar al recorrerla

1. **El problema** — por qué prever ventas a ciegas sale caro
2. **Los datos** — 260 semanas de ventas, sin ninguna variable externa
3. **Exploración** — tendencia y estacionalidad anual a simple vista
4. **Metodología** — estacionariedad, diferenciación, ACF/PACF
5. **El modelo** — SARIMA vs. asumir que "se repite como el año pasado"
6. **Explicabilidad** — qué capturan sus 2 únicos parámetros, y el diagnóstico de residuos
7. **Playground** — explora el forecast a 20 semanas semana a semana
8. **Resultados, decisiones y limitaciones** — honestas, con cifras reales

## Stack

- **Python** — pandas, numpy, statsmodels (SARIMAX), scikit-learn (métricas)
- **Streamlit** — la app interactiva
- **Plotly** — visualizaciones

`model/train.py` replica la metodología del notebook original una sola vez, en local,
y guarda en `model/artifacts/` todo lo que la app necesita para narrar y para el
Playground — `app.py` corre en Streamlit Cloud sin reentrenar nada ni depender de
`statsmodels`/`scikit-learn` en tiempo de ejecución.

## Ejecutar en local

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install statsmodels scikit-learn scipy openpyxl  # solo para regenerar artefactos
py -3.10 model/train.py
streamlit run app.py
```

## Fuente

Notebook original: [`03-Machine-Learning/04-series-temporales/arima/01-ventas-semanales-retail`](https://github.com/BORJAMOME/Data-Analytics-Portfolio/tree/main/03-Machine-Learning/04-series-temporales/arima/01-ventas-semanales-retail)
en [Data Analytics Portfolio](https://github.com/BORJAMOME/Data-Analytics-Portfolio).

---

**Autor:** Borja Mora Méndez · [LinkedIn](https://www.linkedin.com/in/borja-mora-mendez/) · [borja.mora.mendez@gmail.com](mailto:borja.mora.mendez@gmail.com)
