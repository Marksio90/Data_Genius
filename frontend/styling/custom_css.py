"""
DataGenius PRO - Custom CSS (PRO+++)
Centralne stylowanie + lekkie komponenty HTML dla Streamlit.

Ten moduł:
- wstrzykuje spójny CSS z możliwością nadpisania zmiennych (CSS variables)
- zapewnia „compact mode”
- pozwala opcjonalnie ukryć branding Streamlit
- renderuje lekkie komponenty (metric card, badge, info box, status chip, header, pill button)
- dba o jednokrotne wstrzyknięcie CSS na sesję

Użycie:
    from ui.custom_css import load_custom_css, render_metric_card, render_badge, ...

    load_custom_css(
        theme_overrides={"--dg-primary": "#0ea5e9", "--dg-radius": "10px"},
        compact=True,
        hide_streamlit_branding=True,
    )

    render_metric_card("Dokładność", "0.913", "5-fold CV", "📈")
"""

from __future__ import annotations

from typing import Optional, Dict
import html
import streamlit as st

# Klucz sesyjny — aby nie duplikować stylów przy rerenderze
_DG_CSS_FLAG = "__dg_css_loaded_v2"

# =========================
# CSS bazowy (namespaced)
# =========================
_BASE_CSS = r"""
<style>
  /* ========= DataGenius Namespace ========= */
  :root {
    --dg-primary: #667eea;
    --dg-primary-2: #764ba2;
    --dg-text-1: #1f2937;
    --dg-text-2: #4b5563;
    --dg-muted: #6b7280;
    --dg-bg-1: #ffffff;
    --dg-bg-2: #f8f9fa;
    --dg-border: #e5e7eb;
    --dg-ok: #16a34a;
    --dg-warn: #f59e0b;
    --dg-danger: #ef4444;
    --dg-info: #0ea5e9;
    --dg-chip-bg-ok: #15a34a1a;
    --dg-chip-bg-warn: #f59e0b1a;
    --dg-radius: 12px;
    --dg-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    --dg-spacing: 1.5rem; /* zmniejsz w compact */
  }

  @media (prefers-color-scheme: dark) {
    :root {
      --dg-text-1: #e5e7eb;
      --dg-text-2: #cbd5e1;
      --dg-muted: #94a3b8;
      --dg-bg-1: #0b1220;
      --dg-bg-2: #111827;
      --dg-border: #1f2937;
      --dg-shadow: 0 2px 12px rgba(0,0,0,0.45);
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .dg-animate { animation: none !important; transition: none !important; }
  }

  /* ========= Layout / Typography ========= */
  .main { padding: var(--dg-spacing); }
  h1, h2, h3 { color: var(--dg-text-1); }
  h1 { font-weight: 800; margin-bottom: .75rem; }
  h2 { font-weight: 700; margin: 1.25rem 0 .75rem; }
  h3 { font-weight: 600; margin: 1rem 0 .5rem; }

  /* ========= Cards / Containers ========= */
  .dg-card {
    background: var(--dg-bg-1);
    border-radius: var(--dg-radius);
    box-shadow: var(--dg-shadow);
    padding: 1.25rem 1.25rem;
    border: 1px solid var(--dg-border);
  }
  .dg-card:hover { transform: translateY(-2px); transition: transform .2s ease; }

  /* ========= Metric card ========= */
  .dg-metric {
    display: grid; grid-template-columns: 56px 1fr; gap: .75rem; align-items: start;
  }
  .dg-metric .dg-icon { font-size: 2rem; line-height: 2rem; }
  .dg-metric .dg-title { font-size: .9rem; color: var(--dg-muted); font-weight: 600; }
  .dg-metric .dg-value { font-size: 2rem; font-weight: 800; color: var(--dg-primary); }
  .dg-metric .dg-sub { font-size: .85rem; color: var(--dg-text-2); }

  /* ========= Badges / Chips ========= */
  .dg-badge {
    display:inline-block; padding: 4px 10px; border-radius: 999px;
    font-size:.825rem; font-weight: 700; margin: 0 4px 4px 0;
    border: 1px solid var(--dg-border); background: var(--dg-bg-2); color: var(--dg-text-1);
  }
  .dg-badge.success { background:#d4edda; color:#155724; border-color:#c3e6cb; }
  .dg-badge.warning { background:#fff3cd; color:#856404; border-color:#ffeeba; }
  .dg-badge.danger  { background:#f8d7da; color:#721c24; border-color:#f5c6cb; }
  .dg-badge.info    { background:#d1ecf1; color:#0c5460; border-color:#bee5eb; }

  .dg-chip {
    display:inline-block; padding: 2px 10px; border-radius:999px; font-size:.8rem; font-weight:700;
    border:1px solid var(--dg-ok); color: var(--dg-ok); background: var(--dg-chip-bg-ok);
  }
  .dg-chip.warn { border-color: var(--dg-warn); color: var(--dg-warn); background: var(--dg-chip-bg-warn); }

  /* ========= Info boxes ========= */
  .dg-info, .dg-warn, .dg-success, .dg-error {
    border-left: 4px solid; padding: 1rem; border-radius: var(--dg-radius);
    margin: .75rem 0; background: var(--dg-bg-2);
  }
  .dg-info    { border-color: var(--dg-info); }
  .dg-warn    { border-color: var(--dg-warn); }
  .dg-success { border-color: var(--dg-ok); }
  .dg-error   { border-color: var(--dg-danger); }

  /* ========= Buttons (pill) ========= */
  .dg-pill {
    display:inline-flex; align-items:center; gap:.5rem; padding:.6rem 1rem; border-radius:999px;
    border:1px solid var(--dg-border); background: var(--dg-bg-1); color: var(--dg-text-1);
    font-weight:700; cursor:pointer; text-decoration:none;
    transition:transform .15s ease, box-shadow .15s ease;
    box-shadow: var(--dg-shadow);
  }
  .dg-pill:hover { transform: translateY(-1px); }
  .dg-pill.primary { background: linear-gradient(135deg, var(--dg-primary) 0%, var(--dg-primary-2) 100%); color: #fff; border: none; }

  /* ========= Tabs / Tables / Uploader / Progress / Spinner ========= */
  .stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; font-weight:700; }
  [data-testid="stDataFrame"] { border-radius: var(--dg-radius); overflow:hidden; }
  [data-testid="stFileUploader"] {
      border: 2px dashed var(--dg-border); border-radius: var(--dg-radius); padding: 2rem; text-align:center;
  }
  [data-testid="stFileUploader"]:hover {
      border-color: var(--dg-primary); background-color: rgba(102,126,234,0.05);
  }
  .stProgress > div > div > div > div {
      background: linear-gradient(90deg, var(--dg-primary) 0%, var(--dg-primary-2) 100%);
  }
  .stSpinner > div { border-top-color: var(--dg-primary) !important; }

  /* ========= Code / Plotly ========= */
  code { background: #0a0a0a0d; padding: 2px 6px; border-radius:4px; }
  .js-plotly-plot { border-radius: var(--dg-radius); overflow:hidden; }

  /* ========= Animations ========= */
  @keyframes dg-fade-in {
    from { opacity:0; transform: translateY(8px); }
    to   { opacity:1; transform: translateY(0); }
  }
  .dg-animate { animation: dg-fade-in .35s ease-out; }

  /* ========= Responsive ========= */
  @media (max-width: 768px) {
    .main { padding: calc(var(--dg-spacing) * .66); }
    h1 { font-size: 1.5rem; }
    h2 { font-size: 1.25rem; }
  }
</style>
"""


# =========================
# API: wstrzyknięcie CSS
# =========================
def load_custom_css(
    theme_overrides: Optional[Dict[str, str]] = None,
    compact: bool = False,
    hide_streamlit_branding: bool = True,
) -> None:
    """
    Wstrzykuj CSS do aplikacji Streamlit (jednorazowo na sesję).

    :param theme_overrides: np. {"--dg-primary": "#0ea5e9", "--dg-radius": "10px"}
    :param compact: True → mniejsze paddingi (gęstszy UI)
    :param hide_streamlit_branding: ukryj MainMenu i nagłówek Streamlit
    """
    if not st.session_state.get(_DG_CSS_FLAG):
        st.markdown(_BASE_CSS, unsafe_allow_html=True)
        st.session_state[_DG_CSS_FLAG] = True

    overrides = dict(theme_overrides or {})
    if compact:
        overrides["--dg-spacing"] = "1rem"

    if overrides:
        css_vars = "\n".join(
            [f"  {k}: {v};" for k, v in overrides.items() if k.startswith("--dg-")]
        )
        st.markdown(f"<style>:root {{\n{css_vars}\n}}</style>", unsafe_allow_html=True)

    if hide_streamlit_branding:
        st.markdown("<style>#MainMenu, header {visibility:hidden;}</style>", unsafe_allow_html=True)


# =========================
# Komponenty HTML
# =========================
def _escape(s: str) -> str:
    """Proste escapowanie, aby uniknąć wstrzyknięć HTML w tekstach."""
    return html.escape(s or "")


def render_metric_card(title: str, value: str, subtitle: str = "", icon: str = "") -> None:
    """Karta metryki (lekka, a11y)."""
    st.markdown(
        f"""
        <div class="dg-card dg-animate" role="group" aria-label="{_escape(title)} metric">
          <div class="dg-metric">
            <div class="dg-icon" aria-hidden="true">{_escape(icon)}</div>
            <div>
              <div class="dg-title">{_escape(title)}</div>
              <div class="dg-value">{_escape(value)}</div>
              <div class="dg-sub">{_escape(subtitle)}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_badge(text: str, type: str = "info") -> None:
    """Prosta odznaka: info | success | warning | danger."""
    type = (type or "info").lower()
    st.markdown(f'<span class="dg-badge {type}">{_escape(text)}</span>', unsafe_allow_html=True)


def render_info_box(content: str, type: str = "info") -> None:
    """Box informacyjny: info | warning | success | error."""
    mapping = {"info": "dg-info", "warning": "dg-warn", "success": "dg-success", "error": "dg-error"}
    cls = mapping.get(type.lower(), "dg-info")
    st.markdown(f'<div class="{cls} dg-animate">{content}</div>', unsafe_allow_html=True)


def render_status_chip(online: bool, label_ok: str = "AI: ONLINE", label_off: str = "AI: OFFLINE") -> None:
    """Mini „chip” statusu (np. AI ONLINE/OFFLINE)."""
    html_chip = (
        f'<span class="dg-chip">{_escape(label_ok)}</span>'
        if online
        else f'<span class="dg-chip warn">{_escape(label_off)}</span>'
    )
    st.markdown(f'<div style="text-align:right">{html_chip}</div>', unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "") -> None:
    """Nagłówek sekcji z cieniem."""
    st.markdown(
        f"""
        <div class="dg-animate" style="margin:.5rem 0 1rem;">
          <h2 style="margin:.25rem 0">{_escape(title)}</h2>
          <div style="color:var(--dg-text-2)">{_escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill_button(text: str, href: Optional[str] = None, primary: bool = False, right: bool = False) -> None:
    """
    Przyciski w formie pigułki (linki). Użyteczne do skrótów/nawigacji.
    - href: jeśli podasz, renderuje <a>; jeśli None — zwykły span (dekoracyjny).
    - primary: wypełniony gradientem.
    - right: wyrównanie do prawej.
    """
    cls = "dg-pill primary" if primary else "dg-pill"
    wrap = 'style="display:flex; justify-content:flex-end; margin:.25rem 0;"' if right else ""
    inner = f'<a class="{cls}" href="{_escape(href or "#")}" target="_self">{_escape(text)}</a>' if href else f'<span class="{cls}">{_escape(text)}</span>'
    st.markdown(f'<div {wrap}>{inner}</div>', unsafe_allow_html=True)
