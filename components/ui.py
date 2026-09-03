"""Bloques de HTML reutilizables para mantener el look editorial en toda
la app: eyebrows, títulos, KPI tiles, callouts de hallazgo, tarjetas...

Este archivo es idéntico byte a byte en los 4 proyectos que originaron
este skill. Cópialo tal cual a components/ui.py del proyecto nuevo y
cambia SOLO el string dentro de nav() (el nombre del proyecto). No
reescribas ninguna función salvo que necesites un componente que de
verdad no exista todavía — ver design-system.md, sección 2, "Patrones
no envueltos en función" para el candidato ya identificado (stat_card).
"""
import streamlit as st
import streamlit.components.v1 as components

PALETTE = {
    "ink": "#1D2638", "navy2": "#273A5F", "navy3": "#4A628E", "navy4": "#B9C5D6",
    "positive": "#6E7F5B", "negative": "#C2412E", "muted": "#6B7280",
}


_NAV_ITEMS = [
    ("contexto", "Contexto"), ("datos", "Datos"), ("exploracion", "Exploración"),
    ("modelo", "Modelo"), ("explicabilidad", "Explicabilidad"), ("playground", "Playground"),
    ("resultados", "Resultados"), ("decisiones", "Decisiones"),
]


def nav(portfolio_url: str = "https://borjamora.es/"):
    links = "".join(f'<a href="#{anchor}">{label}</a>' for anchor, label in _NAV_ITEMS)
    st.markdown(
        f"""
        <div class="co-nav">
          <div class="co-nav-left">
            <a class="co-nav-back" href="{portfolio_url}" target="_blank" rel="noopener">&#8592; Portfolio</a>
            <span class="co-nav-brand">SARIMA · Ventas Retail</span>
          </div>
          <div class="co-nav-links">{links}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def install_smooth_scroll():
    """Streamlit renderiza el contenido dentro de un contenedor con su propio
    scroll (`section[data-testid="stMain"]`), así que un `<a href="#id">` normal
    no lo desplaza, y `st.markdown` sanea los atributos `onclick`. Este componente
    inyecta un listener global (vía el iframe del componente, mismo origen) que
    intercepta los clics en enlaces `#ancla` y hace `scrollIntoView` a mano."""
    # Sin guarda de "ya instalado": Streamlit destruye y recrea el iframe de
    # este componente en cada rerun (al mover un slider, por ejemplo), lo que
    # mata el listener anterior. Reinstalar en cada rerun asegura que siempre
    # hay uno activo; los duplicados de reruns previos ya están muertos junto
    # con su iframe, así que no hay efecto doble.
    components.html(
        """
        <script>
        (function(){
          const doc = window.parent.document;
          if (doc.__coScrollHandler) {
            doc.removeEventListener('click', doc.__coScrollHandler);
          }
          doc.__coScrollHandler = function(e){
            const a = e.target.closest('a[href^="#"]');
            if (!a) return;
            const id = a.getAttribute('href').slice(1);
            const el = doc.getElementById(id);
            if (el) {
              e.preventDefault();
              el.scrollIntoView({behavior:'instant', block:'start'});
            }
          };
          doc.addEventListener('click', doc.__coScrollHandler);
        })();
        </script>
        """,
        height=0,
    )


def section_open(anchor: str, tight: bool = False):
    cls = "co-section tight" if tight else "co-section"
    st.markdown(f'<div id="{anchor}" class="{cls}">', unsafe_allow_html=True)


def section_close():
    st.markdown("</div>", unsafe_allow_html=True)


def eyebrow(text: str, muted: bool = False):
    cls = "eyebrow eyebrow-muted" if muted else "eyebrow"
    st.markdown(f'<p class="{cls}">{text}</p>', unsafe_allow_html=True)


def h2(text: str):
    st.markdown(f'<h2 class="co-h2">{text}</h2>', unsafe_allow_html=True)


def h3(text: str):
    st.markdown(f'<h3 class="co-h3">{text}</h3>', unsafe_allow_html=True)


def lead(text: str):
    st.markdown(f'<p class="co-lead">{text}</p>', unsafe_allow_html=True)


def body(text: str):
    st.markdown(f'<p class="co-body">{text}</p>', unsafe_allow_html=True)


def divider():
    st.markdown('<div class="co-divider"></div>', unsafe_allow_html=True)


def kpi_grid(items, cols: int = 4):
    """items: list of dicts {num, label, tone(optional: pos/neg)}"""
    cls = "kpi-grid" if cols == 4 else f"kpi-grid cols-{cols}"
    html = [f'<div class="{cls}">']
    for it in items:
        tone = f' {it["tone"]}' if it.get("tone") else ""
        html.append(
            f'<div class="kpi-tile"><div class="kpi-num{tone}">{it["num"]}</div>'
            f'<div class="kpi-label">{it["label"]}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def finding(text: str, tone: str = ""):
    """Callout de hallazgo: una sola apariencia visual (tarjeta plana
    #F2F0EB, sin barra lateral). `tone` ya no cambia el fondo — el color
    solo debe aparecer dentro del texto vía `stat()`, sobre la cifra que
    tiene un significado semántico real."""
    st.markdown(f'<div class="finding">{text}</div>', unsafe_allow_html=True)


def stat(text: str, tone: str = "") -> str:
    """Envuelve una cifra concreta en color semántico (pos/neg). Úsalo
    solo sobre el número, nunca sobre el texto completo de un callout."""
    cls = {"green": "pos", "pos": "pos", "red": "neg", "neg": "neg"}.get(tone, "")
    if not cls:
        return text
    return f'<span class="stat {cls}">{text}</span>'


def card_open(paper: bool = False):
    cls = "co-card paper" if paper else "co-card"
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def pipeline(steps):
    html = ['<div class="pipe-row">']
    for i, s in enumerate(steps):
        if i > 0:
            html.append('<span class="pipe-arrow">&#8594;</span>')
        html.append(f'<span class="pipe-step">{s}</span>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def story_steps(steps):
    """Metodología como recorrido narrativo, no como documentación técnica.
    steps: list of (title, text) — se numeran solos."""
    html = ['<div class="story-steps">']
    for i, (title, text) in enumerate(steps, start=1):
        html.append(
            f'<div class="story-step"><span class="ss-num">{i:02d}</span>'
            f'<span class="ss-title">{title}</span><span class="ss-text">{text}</span></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def question_block(eyebrow_text: str, question_html: str, sub: str = ""):
    """Composición de dos columnas: la pregunta (protagonista) a la
    izquierda, la explicación (apoyo) a la derecha, separadas por una
    regla vertical funcional. Sin `sub`, la pregunta ocupa las dos."""
    right = f'<p class="question-sub">{sub}</p>' if sub else ""
    grid_cls = "question-grid has-sub" if sub else "question-grid"
    st.markdown(
        f"""
        <div class="question-block">
          <p class="question-eyebrow">{eyebrow_text}</p>
          <div class="{grid_cls}">
            <p class="question-text">{question_html}</p>
            {right}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def impact_banner(text_html: str, quote: str = ""):
    """Misma composición de dos columnas que question_block(): la
    afirmación (protagonista) a un lado, la cita (apoyo) al otro."""
    right = f'<p class="impact-quote">{quote}</p>' if quote else ""
    grid_cls = "impact-grid has-quote" if quote else "impact-grid"
    st.markdown(
        f"""
        <div class="impact-banner">
          <div class="{grid_cls}">
            <p class="impact-text">{text_html}</p>
            {right}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, outline: bool = False):
    cls = "badge outline" if outline else "badge"
    return f'<span class="{cls}">{text}</span>'


def insight_card(num: str, title: str, text: str, so_what: str):
    """DEPRECADO como sección por defecto — ver audit.md B.4.1 y
    design-system.md sección 2. Se mantiene disponible para el caso
    concreto de 4-5 hallazgos que de verdad no encajen en Resultados
    ni en Decisiones. No lo uses como sección fija por costumbre."""
    st.markdown(
        f"""
        <div class="insight-card">
          <div class="insight-num">{num}</div>
          <div class="insight-body">
            <b class="h">{title}</b>
            <p>{text}</p>
            <p class="so-what">&#8594; {so_what}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def decision_flow(insight: str, action: str, objective: str, metric: str):
    st.markdown(
        f"""
        <div class="decision-flow">
          <div class="df-box"><span class="df-k">Insight</span><span class="df-v">{insight}</span></div>
          <div class="df-arrow">&#8594;</div>
          <div class="df-box"><span class="df-k">Acción</span><span class="df-v">{action}</span></div>
          <div class="df-arrow">&#8594;</div>
          <div class="df-box"><span class="df-k">Objetivo</span><span class="df-v">{objective}</span></div>
          <div class="df-arrow">&#8594;</div>
          <div class="df-box df-metric"><span class="df-k">Métrica</span><span class="df-v">{metric}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(title: str, value: str, subtitle: str = "", color: str = None, paper: bool = False,
              value_size: str = None, title_color: str = None):
    """VERIFICADO — adoptado en comparativa-modelos-aerolinea-app y
    conjoint-vuelos-app en sustitución del st.markdown crudo que antes
    se repetía a mano en cada proyecto (badges del Playground, tarjetas
    de resultado por clase/segmento). Ver audit.md, sección B.2 /
    design-system.md, sección 2. `value_size` (p.ej. "1.35rem") ajusta
    el tamaño del valor cuando la tarjeta vive en un grid apretado de
    3+ columnas — el kpi-num por defecto (clamp 1.6rem-2.3rem) no
    siempre cabe cómodo ahí; cada proyecto real usó un tamaño distinto
    (1.3-1.6rem) según el contenido de la tarjeta. `color` tiñe el
    VALOR (el modelo/predicción es lo protagonista); `title_color` tiñe
    y pone en negrita el TÍTULO en vez del valor (la categoría/segmento
    es lo protagonista y el número de abajo es neutro) — los 2 usos
    reales encontrados al adoptar el componente, ver audit.md."""
    cls = "co-card paper" if paper else "co-card"
    style_parts = ["margin:.35rem 0 .2rem;"]
    if color:
        style_parts.append(f"color:{color};")
    if value_size:
        style_parts.append(f"font-size:{value_size};")
    value_style = f' style="{" ".join(style_parts)}"' if style_parts else ""
    title_style = f' style="font-size:.72rem; font-weight:700; color:{title_color};"' if title_color \
        else ' style="font-size:.72rem;"'
    subtitle_html = f'<div class="kpi-label">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="{cls}" style="text-align:center; padding:1.1rem .8rem; height:100%;">'
        f'<div class="kpi-label"{title_style}>{title}</div>'
        f'<div class="kpi-num"{value_style}>{value}</div>'
        f'{subtitle_html}</div>',
        unsafe_allow_html=True,
    )


def footer_minimal(name: str, repo_url: str, linkedin_url: str, email: str,
                    portfolio_url: str = "https://borjamora.es/"):
    """Footer minimalista integrado con el fondo de la página: nombre a la
    izquierda, CTA centrado, enlaces discretos a la derecha (portfolio
    primero — es la salida más probable al terminar de leer el caso).
    Sin tarjeta oscura — no debe competir visualmente con el cierre."""
    st.markdown(
        f"""
        <div class="co-footer-mini">
          <span class="ff-name">{name}</span>
          <a class="ff-cta" href="{repo_url}" target="_blank">Repositorio del proyecto</a>
          <div class="ff-links">
            <a href="{portfolio_url}" target="_blank" rel="noopener">Portfolio</a>
            <a href="{linkedin_url}" target="_blank">LinkedIn</a>
            <a href="mailto:{email}">Contacto</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
