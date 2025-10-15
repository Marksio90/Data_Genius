"""
icons.py — DataGenius PRO (PRO+++)
Lekka biblioteka ikon (emoji + inline SVG) dla Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List
import html

import streamlit as st


# === NAZWA_SEKCJI === Konfiguracja / modele danych ===

@dataclass(frozen=True)
class SvgIcon:
    """
    Definicja lekkiej ikony SVG.

    Attributes:
        viewbox: Atrybut viewBox SVG (np. '0 0 24 24')
        paths: Lista stringów <path d="..."> lub tagów linii/okręgów itp. bez nawiasów.
               Każdy element zostanie opakowany jako <path ... /> o ile nie zawiera
               jawnego tagu (np. '<circle .../>').
        stroke_linecap: Styl zakończeń linii ('round' domyślnie dla estetyki).
        stroke_linejoin: Styl łączeń ('round' domyślnie).
    """
    viewbox: str
    paths: List[str]
    stroke_linecap: str = "round"
    stroke_linejoin: str = "round"


# === NAZWA_SEKCJI === Emoji katalog (lekki) ===

_EMOJI: Dict[str, str] = {
    # główne sekcje
    "home": "🏠",
    "upload": "📤",
    "eda": "🔍",
    "ai": "🤖",
    "training": "🧪",
    "results": "📈",
    "monitoring": "📊",
    "registry": "📚",
    "mentor": "🎓",
    "forecast": "📉",
    "report": "📝",
    # statusy
    "ok": "✅",
    "warn": "⚠️",
    "error": "❌",
    "info": "ℹ️",
    # inne
    "feature": "🔥",
    "metrics": "🎯",
    "settings": "⚙️",
    "progress": "🧭",
    "refresh": "🔄",
    "clock": "🕘",
    "save": "💾",
}

# === NAZWA_SEKCJI === SVG katalog (bez zależności) ===
# Styl: linie 24×24, stroke='currentColor' (dziedziczy kolor z CSS/parenta).

_SVG: Dict[str, SvgIcon] = {
    "upload": SvgIcon(
        "0 0 24 24",
        paths=[
            # strzałka do góry
            "M12 16V4",
            "M7 9l5-5 5 5",
            # linia bazowa
            "M20 20H4a2 2 0 0 1-2-2v0a2 2 0 0 1 2-2h4",
        ],
    ),
    "eda": SvgIcon(
        "0 0 24 24",
        paths=[
            # prosta i czytelna lupa: okrąg + rączka
            '<circle cx="11" cy="11" r="8" />',
            '<line x1="21" y1="21" x2="16.65" y2="16.65" />',
        ],
    ),
    "ai": SvgIcon(
        "0 0 24 24",
        paths=[
            "M12 2v4",
            "M12 18v4",
            "M4.93 4.93l2.83 2.83",
            "M16.24 16.24l2.83 2.83",
            "M2 12h4",
            "M18 12h4",
            "M7 12a5 5 0 1 0 10 0 5 5 0 1 0-10 0z",
        ],
    ),
    "training": SvgIcon(
        "0 0 24 24",
        paths=[
            "M4 19V5a2 2 0 0 1 2-2h8",
            "M8 7h8",
            "M8 11h8",
            "M8 15h6",
            "M18 14l3 3-3 3",
        ],
    ),
    "results": SvgIcon(
        "0 0 24 24",
        paths=[
            "M4 20V10",
            "M10 20V4",
            "M16 20v-6",
            "M2 20h20",
        ],
    ),
    "monitoring": SvgIcon(
        "0 0 24 24",
        paths=[
            "M3 3h18v14H3z",
            "M7 12l3-3 3 2 4-4",
            "M3 21h18",
        ],
    ),
    "registry": SvgIcon(
        "0 0 24 24",
        paths=[
            "M6 3h12a2 2 0 0 1 2 2v14l-4-2-4 2-4-2-4 2V5a2 2 0 0 1 2-2z",
            "M8 7h8",
            "M8 11h8",
        ],
    ),
    "mentor": SvgIcon(
        "0 0 24 24",
        paths=[
            "M12 2l9 5-9 5-9-5 9-5z",
            "M4 12v5l8 4 8-4v-5",
        ],
    ),
    "feature": SvgIcon(
        "0 0 24 24",
        paths=[
            "M12 2l2.39 7.36H22l-6.19 4.49L17.82 22 12 17.77 6.18 22l2.01-8.15L2 9.36h7.61L12 2z",
        ],
    ),
    "metrics": SvgIcon(
        "0 0 24 24",
        paths=[
            "M4 19V5",
            "M10 19V9",
            "M16 19V13",
            "M2 19h20",
        ],
    ),
    "settings": SvgIcon(
        "0 0 24 24",
        paths=[
            "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z",
            "M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06A1.65 1.65 0 0 0 15 19.4a1.65 1.65 0 0 0-1 .6 1.65 1.65 0 0 0-.33 1.82l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 8.6 15a1.65 1.65 0 0 0-1-.6 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6c.26 0 .52-.06.76-.18A1.65 1.65 0 0 0 11.6 4a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9c0 .26.06.52.18.76.12.24.18.5.18.76s-.06.52-.18.76A1.65 1.65 0 0 0 19.4 15z",
        ],
    ),
    "refresh": SvgIcon(
        "0 0 24 24",
        paths=[
            "M21 12a9 9 0 1 1-2.64-6.36",
            "M21 3v6h-6",
        ],
    ),
    "play": SvgIcon(
        "0 0 24 24",
        paths=["M8 5v14l11-7-11-7z"],
    ),
    "stop": SvgIcon(
        "0 0 24 24",
        paths=["M6 6h12v12H6z"],
    ),
    "info": SvgIcon(
        "0 0 24 24",
        paths=[
            "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z",
            "M12 8h.01",
            "M11 12h2v6h-2z",
        ],
    ),
    "warn": SvgIcon(
        "0 0 24 24",
        paths=[
            "M10.29 3.86l-8.59 14.85A2 2 0 0 0 3.41 22h17.18a2 2 0 0 0 1.71-3.29L13.71 3.86a2 2 0 0 0-3.42 0z",
            "M12 9v4",
            "M12 17h.01",
        ],
    ),
    "error": SvgIcon(
        "0 0 24 24",
        paths=[
            "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z",
            "M15 9l-6 6",
            "M9 9l6 6",
        ],
    ),
    "ok": SvgIcon(
        "0 0 24 24",
        paths=[
            "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z",
            "M8 12l3 3 5-5",
        ],
    ),
}


# === NAZWA_SEKCJI === API emoji ===

@st.cache_data(show_spinner=False)
def get_emoji(name: str, default: str = "🔹") -> str:
    """
    Zwróć emoji po nazwie. Jeśli brak — `default`.
    """
    return _EMOJI.get(name.lower().strip(), default)


# === NAZWA_SEKCJI === API SVG ===

def _paths_to_svg_content(icon: SvgIcon, stroke: float) -> str:
    parts: List[str] = []
    for p in icon.paths:
        tag = p.strip()
        if tag.startswith("<"):  # użytkownik podał cały element (np. <circle .../>)
            parts.append(tag)
        else:
            parts.append(f'<path d="{html.escape(tag)}" />')
    inner = "\n    ".join(parts)
    return (
        f'viewBox="{icon.viewbox}" fill="none" stroke="currentColor" '
        f'stroke-width="{stroke}" stroke-linecap="{icon.stroke_linecap}" stroke-linejoin="{icon.stroke_linejoin}"'
    ), inner


@st.cache_data(show_spinner=False)
def get_svg(
    name: str,
    size: int = 20,
    color: str = "currentColor",
    stroke: float = 2.0,
    aria_label: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """
    Zwraca **string SVG** (inline) dla zadanej ikony.
    - size: rozmiar w px (szerokość/wysokość).
    - color: CSS (np. 'currentColor', '#0ea5e9').
    - stroke: grubość linii.
    - aria_label: tekst a11y, gdy podany doda role="img" i aria-label.
    - title: opcjonalny <title> jako tooltip i wsparcie a11y.
    """
    key = name.lower().strip()
    icon = _SVG.get(key)
    if icon is None:
        # fallback do emoji jako SVG-tekst
        emoji = get_emoji(key, default="🔹")
        return f'<span aria-hidden="true" style="font-size:{size}px;line-height:1">{emoji}</span>'

    attrs, inner = _paths_to_svg_content(icon, stroke)
    aria = f' role="img" aria-label="{html.escape(aria_label)}"' if aria_label else ' aria-hidden="true"'
    title_tag = f"<title>{html.escape(title)}</title>" if title else ""
    return f'<svg {attrs} width="{size}" height="{size}" style="color:{color}"{aria}>{title_tag}{inner}</svg>'


def render_svg(
    name: str,
    size: int = 20,
    color: str = "currentColor",
    stroke: float = 2.0,
    aria_label: Optional[str] = None,
    align: str = "center",
    title: Optional[str] = None,
) -> None:
    """
    Renderuje ikonę SVG w komponencie Markdown.
    - align: left | center | right — justowanie bloku.
    """
    align_css = {"left": "flex-start", "center": "center", "right": "flex-end"}.get(align, "center")
    html_block = f'<div style="display:flex;justify-content:{align_css}">{get_svg(name, size, color, stroke, aria_label, title)}</div>'
    st.markdown(html_block, unsafe_allow_html=True)


def icon_label(name: str, label: str, size: int = 18, color: str = "currentColor", gap_px: int = 8) -> str:
    """
    Zwraca gotowy HTML, który możesz osadzić w `st.markdown`: [SVG] [label].
    """
    svg = get_svg(name, size=size, color=color, title=label, aria_label=label)
    return f'<span style="display:inline-flex;align-items:center;gap:{gap_px}px">{svg}<span>{html.escape(label)}</span></span>'


# === NAZWA_SEKCJI === Narzędzia pomocnicze ===

def list_icons(kind: str = "all") -> List[str]:
    """
    Lista dostępnych ikon.
    - kind: 'emoji' | 'svg' | 'all'
    """
    k = kind.lower()
    if k == "emoji":
        return sorted(_EMOJI.keys())
    if k == "svg":
        return sorted(_SVG.keys())
    return sorted(set(_EMOJI.keys()) | set(_SVG.keys()))


def get_icon_html(
    name: str,
    label: Optional[str] = None,
    size: int = 18,
    color: str = "currentColor",
    stroke: float = 2.0,
) -> str:
    """
    Zwraca HTML ikony (SVG lub emoji fallback).
    - Jeśli `name` istnieje w _SVG → zwróci SVG z aria-label=label oraz <title>.
    - W przeciwnym razie zwróci emoji (aria-hidden).
    """
    if name.lower().strip() in _SVG:
        return get_svg(name, size=size, color=color, stroke=stroke, aria_label=label, title=label)
    emoji = get_emoji(name)
    return f'<span aria-hidden="true" style="font-size:{size}px;line-height:1">{emoji}</span>'


def icon_with_text(
    name: str,
    text: str,
    size: int = 18,
    color: str = "currentColor",
    gap_px: int = 8,
    stroke: float = 2.0,
) -> str:
    """
    Zwraca HTML: [ikona] [tekst], z pełnym fallbackiem i a11y.
    """
    icon_html = get_icon_html(name, label=text, size=size, color=color, stroke=stroke)
    return f'<span style="display:inline-flex;align-items:center;gap:{gap_px}px">{icon_html}<span>{html.escape(text)}</span></span>'
