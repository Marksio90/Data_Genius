"""
theme.py — DataGenius PRO (PRO+++)
Runtime Theme Manager: presety brandowe, zmienne CSS `--dg-*`, kompaktowość UI i prosty edytor.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple
import re

import streamlit as st

# Integracja z naszym CSS
try:
    from custom_css import load_custom_css
except Exception:  # fallback defensywny
    def load_custom_css(theme_overrides: Optional[Dict[str, str]] = None, compact: bool = False) -> None:
        st.markdown("<!-- custom_css not available; theme only partially applied -->", unsafe_allow_html=True)


# === NAZWA_SEKCJI === Dataclass i stałe bazowe ===

CSS_VAR_PREFIX = "--dg-"

@dataclass
class ThemeConfig:
    """
    Konfiguracja motywu (zestaw zmiennych CSS + flaga 'compact').
    - vars: słownik `--dg-*` → wartość (kolor/px/itp.)
    - name: nazwa motywu (dla UI/logiki)
    - compact: gęsty layout (mniejsze paddingi)
    """
    name: str = "custom"
    vars: Dict[str, str] = field(default_factory=dict)
    compact: bool = False


# Zmiennie wspierane przez custom_css (_BASE_CSS)
_BASE_VARS: Dict[str, str] = {
    "--dg-primary": "#667eea",
    "--dg-primary-2": "#764ba2",
    "--dg-text-1": "#1f2937",
    "--dg-text-2": "#4b5563",
    "--dg-muted": "#6b7280",
    "--dg-bg-1": "#ffffff",
    "--dg-bg-2": "#f8f9fa",
    "--dg-border": "#e5e7eb",
    "--dg-ok": "#16a34a",
    "--dg-warn": "#f59e0b",
    "--dg-danger": "#ef4444",
    "--dg-info": "#0ea5e9",
    "--dg-chip-bg-ok": "#15a34a1a",
    "--dg-chip-bg-warn": "#f59e0b1a",
    "--dg-radius": "12px",
    "--dg-shadow": "0 2px 8px rgba(0,0,0,0.08)",
    "--dg-spacing": "1.5rem",
}


# === NAZWA_SEKCJI === Presety brandowe (light-first; dark dziedziczony przez prefers-color-scheme) ===

PRESETS: Dict[str, Dict[str, str]] = {
    "ocean": {
        "--dg-primary": "#0ea5e9",
        "--dg-primary-2": "#2563eb",
        "--dg-bg-2": "#f3f7fb",
        "--dg-chip-bg-ok": "#10b98122",
        "--dg-chip-bg-warn": "#f59e0b22",
    },
    "emerald": {
        "--dg-primary": "#10b981",
        "--dg-primary-2": "#059669",
        "--dg-bg-2": "#f2fbf7",
        "--dg-chip-bg-ok": "#10b98122",
        "--dg-chip-bg-warn": "#f59e0b22",
    },
    "sunset": {
        "--dg-primary": "#f97316",
        "--dg-primary-2": "#ef4444",
        "--dg-bg-2": "#fff7f2",
        "--dg-chip-bg-ok": "#84cc1622",
        "--dg-chip-bg-warn": "#f59e0b22",
    },
    "graphite": {
        "--dg-primary": "#64748b",
        "--dg-primary-2": "#475569",
        "--dg-text-1": "#111827",
        "--dg-text-2": "#374151",
        "--dg-bg-2": "#f6f7f9",
    },
    "violet": {  # zbliżony do default
        "--dg-primary": "#667eea",
        "--dg-primary-2": "#764ba2",
        "--dg-bg-2": "#f8f7ff",
    },
}


# === NAZWA_SEKCJI === Utils (walidacja, merge, kolory) ===

_HEX_RE = re.compile(r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$")

def _is_css_var(key: str) -> bool:
    return isinstance(key, str) and key.startswith(CSS_VAR_PREFIX)

def _merge_vars(base: Dict[str, str], override: Optional[Dict[str, str]]) -> Dict[str, str]:
    merged = dict(base)
    if override:
        for k, v in override.items():
            if _is_css_var(k) and isinstance(v, str) and v.strip():
                merged[k] = v.strip()
    return merged

def _coerce_hex(s: str, default: str) -> str:
    if not isinstance(s, str):
        return default
    s = s.strip()
    return s if _HEX_RE.match(s) else default


# === NAZWA_SEKCJI === State manager (session_state) ===

def _init_state() -> None:
    st.session_state.setdefault("theme", asdict(ThemeConfig(name="violet", vars=_merge_vars(_BASE_VARS, PRESETS["violet"]), compact=False)))

def get_current_theme() -> ThemeConfig:
    """
    Pobierz aktualny motyw z sesji (inicjalizuje jeśli brak).
    """
    _init_state()
    d = st.session_state.get("theme", {})
    return ThemeConfig(name=d.get("name", "custom"), vars=d.get("vars", {}), compact=bool(d.get("compact", False)))

def set_theme(name: str, overrides: Optional[Dict[str, str]] = None, compact: Optional[bool] = None) -> ThemeConfig:
    """
    Ustaw motyw po nazwie (preset) + opcjonalne nadpisania i tryb compact.
    """
    _init_state()
    base = _merge_vars(_BASE_VARS, PRESETS.get(name, {}))
    final_vars = _merge_vars(base, overrides or {})
    cfg = ThemeConfig(name=name, vars=final_vars, compact=(compact if compact is not None else bool(st.session_state["theme"].get("compact", False))))
    st.session_state["theme"] = asdict(cfg)
    return cfg

def apply_theme(cfg: Optional[ThemeConfig] = None) -> ThemeConfig:
    """
    Zastosuj motyw (wstrzyknięcie CSS). Zwraca ostateczny ThemeConfig.
    """
    if cfg is None:
        cfg = get_current_theme()
    # sanity: tylko zmienne --dg- przepuszczamy do CSS
    safe_vars = {k: v for k, v in cfg.vars.items() if _is_css_var(k)}
    load_custom_css(theme_overrides=safe_vars, compact=cfg.compact)
    return cfg


# === NAZWA_SEKCJI === UI: panel przełączania motywu ===

def render_theme_switcher(expanded: bool = False) -> None:
    """
    Renderuje panel zmiany motywu (presety + szybkie korekty).
    Wywołuj we wstępnej sekcji strony (po set_page_config).
    """
    cfg = get_current_theme()

    with st.expander("🎨 Motyw aplikacji", expanded=expanded):
        c1, c2, c3 = st.columns([0.35, 0.25, 0.40])

        with c1:
            preset = st.selectbox("Preset", options=list(PRESETS.keys()), index=list(PRESETS.keys()).index(cfg.name) if cfg.name in PRESETS else 0)
            compact = st.toggle("Compact UI", value=cfg.compact, help="Zmniejsza przestrzenie (padding) w widokach.")
        with c2:
            st.caption("Kolory brandowe")
            col_primary = st.color_picker("Primary", value=cfg.vars.get("--dg-primary", _BASE_VARS["--dg-primary"]), key="thm_primary")
            col_primary2 = st.color_picker("Primary-2", value=cfg.vars.get("--dg-primary-2", _BASE_VARS["--dg-primary-2"]), key="thm_primary2")
        with c3:
            st.caption("Detale")
            radius = st.slider("Corner radius", min_value=6, max_value=20, value=int(cfg.vars.get("--dg-radius", "12px").replace("px", "") or 12))
            spacing = st.select_slider("Spacing", options=["1.0rem", "1.25rem", "1.5rem", "1.75rem", "2.0rem"], value=cfg.vars.get("--dg-spacing", "1.5rem"))

        # Zastosuj
        if st.button("✅ Zastosuj motyw", type="primary", use_container_width=True):
            overrides = {
                "--dg-primary": _coerce_hex(col_primary, cfg.vars.get("--dg-primary", _BASE_VARS["--dg-primary"])),
                "--dg-primary-2": _coerce_hex(col_primary2, cfg.vars.get("--dg-primary-2", _BASE_VARS["--dg-primary-2"])),
                "--dg-radius": f"{int(radius)}px",
                "--dg-spacing": spacing,
            }
            new_cfg = set_theme(preset, overrides=overrides, compact=compact)
            apply_theme(new_cfg)
            st.success(f"Zastosowano motyw: **{new_cfg.name}**")

        # Reset
        if st.button("↩️ Reset do presetów", use_container_width=True):
            new_cfg = set_theme(preset, overrides=None, compact=compact)
            apply_theme(new_cfg)
            st.info("Przywrócono wartości presetowe.")

        # Podgląd
        st.markdown("---")
        _theme_preview()


# === NAZWA_SEKCJI === Podgląd (mini komponenty pod ręką) ===

def _theme_preview() -> None:
    """
    Mały „live preview” wybranych elementów, by od razu zobaczyć kolory/faktury.
    """
    cfg = get_current_theme()
    st.caption("Podgląd palety i komponentów:")

    cA, cB, cC, cD = st.columns(4)
    with cA:
        st.markdown(f"""
        <div style="padding:12px;border-radius:8px;background:var(--dg-bg-2);border:1px solid var(--dg-border)">
          <div style="font-weight:700;color:var(--dg-primary)">Primary</div>
          <div style="height:8px;border-radius:6px;background:var(--dg-primary)"></div>
        </div>
        """, unsafe_allow_html=True)
    with cB:
        st.markdown(f"""
        <div style="padding:12px;border-radius:8px;background:var(--dg-bg-2);border:1px solid var(--dg-border)">
          <div style="font-weight:700;color:var(--dg-primary-2)">Primary-2</div>
          <div style="height:8px;border-radius:6px;background:var(--dg-primary-2)"></div>
        </div>
        """, unsafe_allow_html=True)
    with cC:
        st.markdown(f"""
        <div style="padding:12px;border-radius:8px;background:var(--dg-bg-2);border:1px solid var(--dg-border)">
          <div style="font-weight:700;color:var(--dg-ok)">OK</div>
          <div style="height:8px;border-radius:6px;background:var(--dg-ok)"></div>
        </div>
        """, unsafe_allow_html=True)
    with cD:
        st.markdown(f"""
        <div style="padding:12px;border-radius:8px;background:var(--dg-bg-2);border:1px solid var(--dg-border)">
          <div style="font-weight:700;color:var(--dg-warn)">WARN</div>
          <div style="height:8px;border-radius:6px;background:var(--dg-warn)"></div>
        </div>
        """, unsafe_allow_html=True)

    # Mini badges
    st.markdown(
        """
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px">
          <span class="dg-badge info">info</span>
          <span class="dg-badge success">success</span>
          <span class="dg-badge warning">warning</span>
          <span class="dg-badge danger">danger</span>
          <span class="dg-chip">AI: ONLINE</span>
          <span class="dg-chip warn">AI: OFFLINE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# === NAZWA_SEKCJI === API skrótowe dla stron ===

def ensure_and_apply(default_preset: str = "violet", compact: bool = False, overrides: Optional[Dict[str, str]] = None) -> ThemeConfig:
    """
    Jednolinijkowiec do wywołania na górze strony:
        cfg = ensure_and_apply("ocean", compact=True)
    - Jeśli brak stanu motywu → ustawi preset i zastosuje.
    - Jeśli stan istnieje → tylko zastosuje (chyba że podasz `overrides`, wtedy zmerge’uje i zapisze).
    """
    _init_state()
    if "theme" not in st.session_state or not st.session_state["theme"]:
        cfg = set_theme(default_preset, overrides=overrides, compact=compact)
    else:
        cur = get_current_theme()
        merged_vars = _merge_vars(cur.vars, overrides or {})
        cfg = ThemeConfig(name=cur.name, vars=merged_vars, compact=compact if compact is not None else cur.compact)
        st.session_state["theme"] = asdict(cfg)
    return apply_theme(cfg)
