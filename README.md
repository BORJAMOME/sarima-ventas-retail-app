# ¿Cuánto venderás la semana que viene?

En este proyecto he analizado hasta qué punto es posible predecir las ventas utilizando únicamente datos históricos de la propia serie. Para ello, he trabajado con cinco años de ventas semanales, sin incorporar variables externas como promociones o festivos.


🔗 **[Abrir la app](https://sarima-ventas-retail.streamlit.app/)**

## Contexto

Una cadena de supermercados necesita anticipar cuánto va a vender para planificar pedidos y personal. Entrené un
modelo **SARIMA(0,1,1)(0,1,1,52)** con 208 semanas de histórico y lo evalué sobre 52 semanas que nunca había visto.

## Qué te encontrarás en este proyecto

1. **El problema**: decidir sobre lo que todavía no ha ocurrido
2. **Los datos**: 260 semanas, una única variable
3. **¿Qué cuentan cinco años de ventas?**: tendencia y estacionalidad, y si esta se repite de verdad cada año
4. **Del histórico a una previsión**: corte temporal, ADF, diferenciación, ACF/PACF
5. **¿Puede anticipar las ventas?**: el SARIMA frente a tres referencias, incluida la ajustada por crecimiento
6. **¿Qué queda sin explicar?**: diagnóstico de residuos, sin el arranque del filtro
7. **¿Qué pasará dentro de 1, 10 o 20 semanas?**: Playground con predicción central e intervalo
8. **El pasado contiene una señal útil**: el resultado, con sus matices
9. **¿Qué podría hacer una empresa?**: hipótesis de negocio, no conclusiones del modelo
10. **Hasta dónde llega el modelo**: limitaciones
11. **Conclusión**

## Cómo está hecho

Python + [Streamlit](https://streamlit.io) para la aplicación, y [statsmodels](https://www.statsmodels.org/)
(SARIMAX) para el modelo. Todo lo pesado se calcula **una vez, en local**, y la app solo lee los resultados: en
Streamlit Cloud no reentrena nada ni depende de `statsmodels` ni de `scipy`.

---

**Autor:** Borja Mora Méndez · [LinkedIn](https://www.linkedin.com/in/borjamoramendez/) · [borja.mora.mendez@gmail.com](mailto:borja.mora.mendez@gmail.com)
