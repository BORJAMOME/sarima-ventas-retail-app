"""Editorial system — componentes de narrativa para case studies de ML en Streamlit.

Cada componente responde a UNA función narrativa (no a una «pieza de UI») y se coloca
en el nivel de la rejilla que le corresponde (ver assets/editorial.css):

    FULL      band()                          momentos de alto impacto (full bleed)
    WIDE      lede(), figure(), split(), cols(), cards()   figuras y composiciones asimétricas
    STORY     beat(), subhead(), passage(), insight(), note(), steps(), metrics()
              └─ dentro de STORY: READING (prosa) + ASIDE (notas al margen)

Estructura de una evidencia (PREGUNTA → VISUAL → CAPTION → INSIGHT → IMPLICACIÓN):

    ed.subhead("¿Puede el modelo anticipar los picos?")          # pregunta
    with ed.figure("picos"):                                     # visual + caption
        st.plotly_chart(...)
        ed.caption("FIG. 03", "Qué estamos viendo", source="Elaboración propia")
    ed.insight("La estacionalidad explica…", aside="detalle técnico")   # insight

El módulo NO contiene contenido ni identidad de proyecto: reutilizable tal cual en
cualquier proyecto de ML junto a assets/editorial.css. Los textos que recibe son HTML
de confianza (los escribe el autor).
"""
from contextlib import contextmanager
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def _md(html: str):
    st.markdown(html, unsafe_allow_html=True)


def _paras(value) -> list:
    if value is None or value == "":
        return []
    return [value] if isinstance(value, str) else list(value)


# ------------------------------------------------------------------ infraestructura
def load_css(root: Path, *files: str):
    """Inyecta las hojas de estilo en orden (identidad del proyecto primero, sistema después)."""
    css = "".join((root / "assets" / name).read_text(encoding="utf-8") for name in files)
    _md(f"<style>{css}</style>")


def anchor_scroll():
    """Streamlit renderiza el contenido en su propio contenedor con scroll, así que un
    `<a href="#id">` normal no lo desplaza. Este componente inyecta un listener global
    (vía el iframe del componente, mismo origen) que hace `scrollIntoView` a mano.
    Sin guarda de «ya instalado»: Streamlit recrea el iframe en cada ejecución."""
    components.html(
        """
        <script>
        (function(){
          const doc = window.parent.document;
          if (doc.__edScrollHandler) doc.removeEventListener('click', doc.__edScrollHandler);
          doc.__edScrollHandler = function(e){
            const a = e.target.closest('a[href^="#"]');
            if (!a) return;
            const el = doc.getElementById(a.getAttribute('href').slice(1));
            if (el){ e.preventDefault(); el.scrollIntoView({behavior:'instant', block:'start'}); }
          };
          doc.addEventListener('click', doc.__edScrollHandler);
        })();
        </script>
        """,
        height=0,
    )


def skip_link(target: str, label: str = "Saltar al contenido"):
    _md(f'<a class="ed-skip" href="#{target}">{label}</a>')


def topbar(brand: str, links: list, back_url: str, back_label: str = "&#8592; Portfolio"):
    """Barra superior FULL y pegajosa. links: [(ancla, etiqueta), ...]"""
    nav = "".join(f'<a href="#{anchor}">{label}</a>' for anchor, label in links)
    _md(
        f'<div class="ed-topbar ed-full"><div class="ed-topbar-inner">'
        f'<div class="ed-topbar-left"><a class="ed-back" href="{back_url}" target="_blank" rel="noopener">{back_label}</a>'
        f'<span class="ed-brand">{brand}</span></div>'
        f'<nav class="ed-topbar-links" aria-label="Secciones">{nav}</nav></div></div>'
    )


# ------------------------------------------------------------------ LEDE (WIDE)
def lede(kicker: str, headline: str, deck: str, meta: list, anchor: str = "top"):
    """El comienzo del reportaje: KICKER → HEADLINE → DECK → METADATA.
    meta: [(etiqueta, valor), ...]  — información de contexto, no badges."""
    rows = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in meta)
    _md(
        f'<header id="{anchor}" class="ed-lede ed-wide">'
        f'<p class="ed-kicker">{kicker}</p>'
        f'<h1 class="ed-headline">{headline}</h1>'
        f'<div class="ed-lede-foot"><p class="ed-deck">{deck}</p><dl class="ed-meta">{rows}</dl></div>'
        f"</header>"
    )


# ------------------------------------------------------------------ BEAT (STORY)
def beat(anchor: str, no: str, kicker: str, title: str, deck=None, weight: str = "normal"):
    """Apertura de un beat narrativo. weight: 'minor' | 'normal' | 'major' controla el silencio
    que lo precede (56-96 / 96-160 / 160-240 px): el espacio comunica «hemos cambiado de idea»."""
    cls = "ed-beat" + ("" if weight == "normal" else f" ed-beat--{weight}")
    deck_html = "".join(f'<p class="ed-deck">{p}</p>' for p in _paras(deck))
    _md(
        f'<section id="{anchor}" class="{cls}" aria-labelledby="{anchor}-title">'
        f'<p class="ed-beat-kicker"><span class="ed-beat-no">{no}</span><span>{kicker}</span></p>'
        f'<h2 class="ed-section-title" id="{anchor}-title">{title}</h2>{deck_html}</section>'
    )


def subhead(text: str, level: str = "story"):
    """La pregunta que abre una evidencia. `level` debe coincidir con el del bloque que introduce
    ('wide' si le sigue una figura o un split wide): así título y figura comparten eje."""
    cls = "ed-subhead" + (" ed-wide" if level == "wide" else "")
    _md(f'<h3 class="{cls}">{text}</h3>')


# ------------------------------------------------------------------ READING + ASIDE
def _aside_html(aside, label):
    if not aside:
        return ""
    lab = f'<span class="ed-aside-label">{label}</span>' if label else ""
    return f'<aside class="ed-aside">{lab}{aside}</aside>'


def passage(*paragraphs: str, aside: str = None, aside_label: str = "Detalle técnico", tight: bool = False):
    """Prosa a medida de lectura (~672 px). Con `aside`, la nota se cuelga al margen derecho
    (solo en el eje STORY; en columnas estrechas usa `note()`)."""
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    tight_cls = " ed-tight" if tight else ""
    if not aside:
        _md(f'<div class="ed-prose{tight_cls}">{body}</div>')
        return
    _md(f'<div class="ed-passage{tight_cls}"><div class="ed-prose">{body}</div>{_aside_html(aside, aside_label)}</div>')


def insight(text: str, aside: str = None, aside_label: str = "Detalle técnico", tight: bool = False):
    """La conclusión de una evidencia. Sin caja: escala, tinta y una regla corta."""
    tight_cls = " ed-tight" if tight else ""
    if not aside:
        _md(f'<p class="ed-insight{tight_cls}">{text}</p>')
        return
    _md(f'<div class="ed-passage{tight_cls}"><p class="ed-insight">{text}</p>{_aside_html(aside, aside_label)}</div>')


def note(text: str):
    """Nota en línea, para columnas estrechas donde no cabe un aside."""
    _md(f'<p class="ed-note">{text}</p>')


# ------------------------------------------------------------------ FIGURAS
@contextmanager
def figure(name: str, level: str = "wide"):
    """Contenedor de una figura. level: 'wide' (por defecto), 'story' o 'full' (full bleed)."""
    with st.container(key=f"ed-{level}-fig-{name}"):
        yield


def caption(no: str, text: str, source: str = None):
    """Pie de figura: FIG. 03 · descripción · fuente. Discreto y claramente distinto del cuerpo."""
    src = f'<span class="ed-source">{source}</span>' if source else ""
    _md(
        f'<div class="ed-caption" role="note"><span class="ed-fig-no">{no}</span>'
        f'<p class="ed-fig-text">{text}</p>{src}</div>'
    )


@contextmanager
def split(name: str, ratio: str = "4-8"):
    """Composición asimétrica sobre la rejilla de 12 columnas. ratio: '4-8', '8-4', '5-7', '7-5',
    '6-6' o '4-4-4'. La primera columna es la izquierda: la narrativa decide qué va en cada lado."""
    weights = [int(n) for n in ratio.split("-")]
    with st.container(key=f"ed-wide-split-{ratio}-{name}"):
        yield st.columns(weights, gap="small")


# ------------------------------------------------------------------ MÉTRICA → RESULTADO → INTERPRETACIÓN
def metrics(rows: list, word: bool = False):
    """rows: [(nombre, valor, interpretación_html, color_opcional), ...]
    Nunca una batería de KPI: cada valor va acompañado de lo que significa.
    word=True cuando el valor es una palabra (p. ej. un segmento) y no una cifra."""
    out = []
    for row in rows:
        name, value, text = row[0], row[1], row[2]
        color = row[3] if len(row) > 3 and row[3] else None
        style = f' style="color:{color}"' if color else ""
        out.append(
            f'<div class="ed-metric"><p class="ed-metric-name">{name}</p>'
            f'<p class="ed-metric-value"{style}>{value}</p><p class="ed-metric-text">{text}</p></div>'
        )
    _md(f'<div class="ed-metrics{" ed-metrics--word" if word else ""}">{"".join(out)}</div>')


# ------------------------------------------------------------------ PROCEDENCIA Y RUTA
def provenance(items: list):
    """Procedencia de los datos como información editorial (no como tarjeta de metadatos).
    items: [(etiqueta, valor), ...] — fuente, dataset, observaciones, variables, elaboración…"""
    rows = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in items)
    _md(f'<dl class="ed-meta ed-provenance">{rows}</dl>')


def route(items: list):
    """El recorrido de un vistazo, en una sola línea: a → b → c."""
    body = '<span class="ed-route-sep" aria-hidden="true">&#8594;</span>'.join(f"<span>{i}</span>" for i in items)
    _md(f'<p class="ed-route">{body}</p>')


# ------------------------------------------------------------------ COLUMNAS Y LISTAS (sin cajas)
def cols(items: list, count: int = None):
    """Columnas con regla superior. items: [{'tag':…, 'title':…, 'text':…, 'color': opcional}, ...]
    `color` tiñe la regla superior: úsalo solo cuando el color identifica una categoría (p. ej. un segmento)."""
    count = count or len(items)
    body = "".join(
        f'<div class="ed-col"' + (f' style="border-top-color:{it["color"]}"' if it.get("color") else "") + '>'
        + (f'<p class="ed-col-tag">{it["tag"]}</p>' if it.get("tag") else "")
        + f'<p class="ed-col-title">{it["title"]}</p><p class="ed-col-text">{it["text"]}</p></div>'
        for it in items
    )
    _md(f'<div class="ed-cols ed-cols--{count}" style="--cols:{count}">{body}</div>')


def cards(items: list, count: int = None):
    """Tarjetas para entidades paralelas y comparables cuyo COLOR las identifica (segmentos, escenarios,
    lo que sí / no puede hacer un modelo). Es la excepción a «reducir cajas»: solo tiene sentido cuando la
    agrupación es estructural. El color tiñe la barra superior, el fondo (7 %), el título y la cifra; el
    cuerpo va siempre en tinta para conservar el contraste.

    items: [{'tag': ..., 'title': ..., 'stat': '15,1%', 'stat_label': '978 clientes',
             'text': ... | 'points': [...], 'color': '#hex' | 'var(--positive)'}, ...]"""
    count = count or len(items)
    out = []
    for it in items:
        seg = f' style="--seg:{it["color"]}"' if it.get("color") else ""
        tag = f'<p class="ed-card-tag">{it["tag"]}</p>' if it.get("tag") else ""
        stat = ""
        if it.get("stat"):
            label = f'<span class="ed-card-stat-label">{it["stat_label"]}</span>' if it.get("stat_label") else ""
            stat = f'<p class="ed-card-stat"><span class="ed-card-stat-num">{it["stat"]}</span>{label}</p>'
        if it.get("points"):
            body = '<ul class="ed-card-list">' + "".join(f"<li>{pt}</li>" for pt in it["points"]) + "</ul>"
        else:
            body = f'<p class="ed-card-text">{it.get("text", "")}</p>'
        out.append(f'<article class="ed-card"{seg}>{tag}<p class="ed-card-title">{it["title"]}</p>{stat}{body}</article>')
    _md(f'<div class="ed-cards ed-cards--{count}" style="--cols:{count}">{"".join(out)}</div>')


def lists(items: list):
    """Dos (o más) listas con título. items: [{'title':…, 'points':[html, …]}, ...]"""
    body = "".join(
        f'<div class="ed-col"><p class="ed-col-title">{it["title"]}</p>'
        f'<ul class="ed-list">{"".join(f"<li>{p}</li>" for p in it["points"])}</ul></div>'
        for it in items
    )
    _md(f'<div class="ed-cols" style="--cols:{len(items)}">{body}</div>')


def steps(items: list):
    """El método como recorrido numerado. items: [(título, texto_html), ...]"""
    body = "".join(
        f'<div class="ed-step"><p class="ed-step-no">{i:02d}</p><p class="ed-step-title">{title}</p>'
        f'<p class="ed-step-text">{text}</p></div>'
        for i, (title, text) in enumerate(items, start=1)
    )
    _md(f'<div class="ed-steps">{body}</div>')


# ------------------------------------------------------------------ FULL BLEED
def band(eyebrow: str, lead: str, side: str = "", quote: bool = False):
    """Momento full bleed: la afirmación (protagonista) a la izquierda y su apoyo a la derecha.
    Úsalo pocas veces: es un momento editorial, no un componente de página."""
    side_html = f'<p class="ed-band-side{" quote" if quote else ""}">{side}</p>' if side else ""
    _md(
        f'<div class="ed-band ed-full"><div class="ed-band-inner">'
        f'<p class="ed-metadata ed-band-eyebrow">{eyebrow}</p>'
        f'<p class="ed-band-lead">{lead}</p>{side_html}</div></div>'
    )


# ------------------------------------------------------------------ COLOFÓN
def colophon(name: str, links: list):
    """Cierre de la pieza. links: [(etiqueta, url), ...]"""
    nav = "".join(f'<a href="{url}" target="_blank" rel="noopener">{label}</a>' for label, url in links)
    _md(f'<div class="ed-colophon" role="contentinfo"><p class="ed-colophon-name">{name}</p><nav aria-label="Enlaces">{nav}</nav></div>')
