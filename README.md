# SARIMA — Ventas Semanales Retail

**¿Cuánto venderás la semana que viene?**

Una aplicación interactiva, con formato de reportaje de datos, que comprueba hasta dónde puede llegar una
previsión de ventas utilizando **únicamente el pasado de la propia serie**: cinco años de ventas semanales, una
sola variable, sin promociones, festivos ni ningún dato externo.

No hace falta saber nada de Machine Learning para seguirla: empieza por el problema, explica el método sin
tecnicismos (el detalle técnico va aparte, en notas al margen) y termina dejándote explorar el forecast de las
próximas 20 semanas con su intervalo de confianza.

## Ver la app

🔗 **[Abrir la app](https://sarima-ventas-retail.streamlit.app/)**

## De qué trata, en dos frases

Una cadena de supermercados necesita anticipar cuánto va a vender para planificar pedidos y personal. Entrené un
modelo **SARIMA(0,1,1)(0,1,1,52)** con 208 semanas de histórico y lo evalué sobre 52 semanas que nunca había visto.

**El resultado, con matices:**

- Error medio (MAPE) del **2,21%** en las 52 semanas de test, un 74% menos que repetir la misma semana del año
  anterior (8,39%).
- Pero esa referencia ignora que las ventas crecen (~9% al año). Con la misma regla **ajustada por el crecimiento
  del último año de entrenamiento**, el error es del 2,28%: prácticamente igual. Lo que aporta la mejora es
  incorporar la tendencia y la estacionalidad, no la complejidad del SARIMA.
- El intervalo del 95% contuvo la venta real en 52 de 52 semanas; dentro de las próximas 20 semanas se ensancha
  solo un 9,5%, mientras que en las 52 del test casi se duplica.

## Qué te vas a encontrar al recorrerla

Once capítulos, cada uno responde a una pregunta:

1. **El problema** — decidir sobre lo que todavía no ha ocurrido
2. **Los datos** — 260 semanas, una única variable
3. **¿Qué cuentan cinco años de ventas?** — tendencia y estacionalidad, y si esta se repite de verdad cada año
4. **Del histórico a una previsión** — corte temporal, ADF, diferenciación, ACF/PACF
5. **¿Puede anticipar las ventas?** — el SARIMA frente a tres referencias, incluida la ajustada por crecimiento
6. **¿Qué queda sin explicar?** — diagnóstico de residuos, sin el arranque del filtro
7. **¿Qué pasará dentro de 1, 10 o 20 semanas?** — Playground con predicción central e intervalo
8. **El pasado contiene una señal útil** — el resultado, con sus matices
9. **¿Qué podría hacer una empresa?** — hipótesis de negocio, no conclusiones del modelo
10. **Hasta dónde llega el modelo** — limitaciones
11. **Conclusión**

## Cómo está hecho

Python + [Streamlit](https://streamlit.io) para la aplicación, y [statsmodels](https://www.statsmodels.org/)
(SARIMAX) para el modelo. Todo lo pesado se calcula **una vez, en local**, y la app solo lee los resultados: en
Streamlit Cloud no reentrena nada ni depende de `statsmodels` ni de `scipy`.

- `model/train.py` replica la metodología del notebook original y guarda métricas, previsiones, residuos y el
  forecast a 20 semanas.
- `model/export_audit.py` calcula, sin reentrenar, lo que la narrativa necesita para no afirmar más de lo que
  los datos permiten: la referencia ajustada por crecimiento, los residuos sin el arranque del filtro, el
  ensanchamiento del intervalo y la consistencia de la estacionalidad año a año.

Nada del texto está escrito a mano: si cambian los datos, cambian las cifras.

Algunas decisiones de rigor que están en la app y conviene conocer:

- **Una referencia justa.** «Repetir el año anterior» pierde porque ignora el crecimiento; una regla igual de
  simple que sí lo tiene en cuenta llega casi al mismo error (2,28% frente a 2,21%, sin diferencia concluyente
  con un año de test).
- **La estacionalidad se comprueba año a año.** La componente estacional de una descomposición se repite por
  construcción, así que no prueba nada por sí sola. Aquí se compara el patrón de cada año completo
  (correlación 0,83–0,90).
- **La ACF no muestra un pico en el retardo 52.** El término estacional se justifica por el patrón anual de los
  datos y por su significación en el modelo, no por la ACF.
- **Residuos sin el arranque.** Con d=1 y D=1 (s=52) las primeras 53 observaciones son inicialización, no error
  del modelo, y contaminan Ljung-Box. Se apartan como un tramo continuo y se informa del rango que sale con otros
  cortes.

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Los resultados ya vienen calculados en `model/artifacts/`, así que no hace falta reentrenar nada. Solo si cambias
el dataset (`data/arima.xlsx`) necesitas regenerarlos:

```bash
pip install statsmodels scikit-learn scipy openpyxl   # solo para regenerar
python model/train.py
python model/export_audit.py
```

<details>
<summary>Estructura del proyecto, para quien quiera curiosear el código</summary>

```
app.py                    la aplicación — toda la narrativa, capítulo a capítulo
components/
  editorial.py              sistema editorial: un componente por función narrativa (lede, beat, figure…)
  ui.py                    bloques visuales heredados de versiones anteriores (ya no los usa app.py)
  charts.py                 gráficos, con la paleta de colores del proyecto
utils/
  data_loader.py             carga de artefactos (con cache de Streamlit)
model/
  train.py                    entrena el SARIMA y calcula métricas, residuos y el forecast a 20 semanas
  export_audit.py             comprobaciones adicionales sin reentrenar (referencias, residuos, intervalos…)
  artifacts/                   resultados ya calculados
data/                      el dataset original
assets/editorial.css       sistema de composición editorial (rejilla, niveles de ancho, ritmo, tokens) — reutilizable
assets/style.css           identidad de este proyecto (paleta y familias tipográficas)
```

</details>

## Fuente

Notebook original: [`03-Machine-Learning/04-series-temporales/arima/01-ventas-semanales-retail`](https://github.com/BORJAMOME/Data-Analytics-Portfolio/tree/main/03-Machine-Learning/04-series-temporales/arima/01-ventas-semanales-retail)
en [Data Analytics Portfolio](https://github.com/BORJAMOME/Data-Analytics-Portfolio).

### Sistema editorial (reutilizable en otros proyectos de ML)

`assets/editorial.css` convierte el contenedor de Streamlit en una rejilla con cinco niveles de ancho y una escala
de espacio y de tipografía editoriales; `components/editorial.py` expone un componente por función narrativa. Es
el mismo sistema que usan el resto de proyectos de ML del portfolio. Requiere `streamlit==1.58.0` (usa
`st.container(key=...)` y el DOM de esa versión).

---

**Autor:** Borja Mora Méndez · [LinkedIn](https://www.linkedin.com/in/borjamoramendez/) · [borja.mora.mendez@gmail.com](mailto:borja.mora.mendez@gmail.com)
