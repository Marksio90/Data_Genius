# src/frontend/chat_interface.py
# === OPIS MODUŁU ===
# Moduł czatu PRO+++ dla DataGenius/Universal-Forecasting:
# - Streamlit UI (historia, konfiguracja, akcje)
# - Provider abstrakcyjny (OpenAI -> streaming / Offline fallback)
# - Bezpieczeństwo: sprawdzanie klucza w st.secrets/session/env
# - Stabilność: retry, limiter, obsługa wyjątków, walidacja
# - Eksport: zapis historii do JSON

from __future__ import annotations

import os
import json
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Protocol, Tuple

import streamlit as st

warnings.filterwarnings("ignore")

# === NAZWA_SEKCJI === Konfiguracja logowania (opcjonalnie przez Twój logger) ===
try:
    from src.utils.logger import get_logger  # Twój istniejący logger
    log = get_logger(__name__)
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("chat_interface")

# === NAZWA_SEKCJI === Dataclasses i protokoły ===

@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str
    created_at: float = field(default_factory=lambda: time.time())

@dataclass
class ChatConfig:
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 1024
    system_prompt: str = (
        "Jesteś asystentem danych (Senior DS/ML). Odpowiadasz zwięźle, konkretnie i bezpiecznie. "
        "Gdy proszą o kod, zwracasz kompletny, gotowy do uruchomienia fragment."
    )

class ChatProvider(Protocol):
    def chat_stream(
        self, messages: List[ChatMessage], cfg: ChatConfig
    ) -> Generator[str, None, None]:
        ...

# === NAZWA_SEKCJI === Pomocnicze: pobieranie klucza i trybu pracy ===

def _get_openai_api_key() -> Optional[str]:
    """
    Hierarchia źródeł:
    1) st.secrets["OPENAI_API_KEY"]
    2) st.session_state["openai_api_key"]
    3) os.environ["OPENAI_API_KEY"]
    """
    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        key = None

    if not key:
        key = st.session_state.get("openai_api_key")

    if not key:
        key = os.environ.get("OPENAI_API_KEY")

    return key

def _has_api_key() -> bool:
    return bool(_get_openai_api_key())

# === NAZWA_SEKCJI === Limiter (prosty throttle per sesja) ===

def _rate_limit(key: str, min_interval_sec: float = 1.0) -> None:
    """
    Wymusza odstęp czasowy między wywołaniami.
    Pod kluczem `key` przechowywany jest timestamp ostatniego użycia.
    """
    now = time.time()
    last_key = f"rl_{key}"
    last = st.session_state.get(last_key, 0.0)
    delta = now - last
    if delta < min_interval_sec:
        to_wait = round(min_interval_sec - delta, 2)
        raise RuntimeError(f"Zbyt częste wywołania. Odczekaj ~{to_wait}s.")
    st.session_state[last_key] = now

# === NAZWA_SEKCJI === Provider: OpenAI (streaming z defensywnym try/except) ===

class OpenAIChatProvider:
    """
    Provider OpenAI. Stara się korzystać z nowszego API; jeśli niedostępne, cofa się do kompatybilnych ścieżek.
    """
    def __init__(self) -> None:
        self.api_key = _get_openai_api_key()
        if not self.api_key:
            raise RuntimeError("Brak klucza OPENAI_API_KEY.")

        # Lazy import, by nie wymagać openai gdy offline
        try:
            import openai  # type: ignore
            self._openai = openai
            # Najczęstsze dwa style użycia (zależnie od wersji pakietu)
            if hasattr(openai, "OpenAI"):
                self.client = openai.OpenAI(api_key=self.api_key)  # type: ignore
            else:
                self.client = None
        except Exception as e:
            log.exception("Nie można zaimportować pakietu openai.")
            raise

    def _to_oai_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def chat_stream(
        self, messages: List[ChatMessage], cfg: ChatConfig
    ) -> Generator[str, None, None]:
        """
        Zwraca generator tokenów. Obsługuje zarówno nowe `responses.stream`, jak i starsze `chat.completions.create(stream=True)`.
        """
        # Retry/backoff
        retries = 2
        delay = 1.0

        for attempt in range(retries + 1):
            try:
                _rate_limit("openai_stream", min_interval_sec=0.5)
                oai_messages = self._to_oai_messages(messages)

                # Ścieżka A: nowe API (jeśli dostępne)
                if self.client and hasattr(self.client, "responses"):
                    with self.client.responses.stream(
                        model=cfg.model_name,
                        input=[{"role": "system", "content": cfg.system_prompt}, *oai_messages],
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        max_output_tokens=cfg.max_tokens,
                    ) as stream:
                        for event in stream:
                            try:
                                if event.type == "response.output_text.delta":  # type: ignore[attr-defined]
                                    yield event.delta  # type: ignore[attr-defined]
                                elif event.type == "response.completed":
                                    break
                            except Exception:
                                # Ciche pomijanie drobnych odchyleń wersji
                                pass
                        stream.close()
                    return

                # Ścieżka B: starsze Chat Completions (openai.ChatCompletion / chat.completions.create)
                if hasattr(self._openai, "ChatCompletion"):
                    # very old api
                    resp = self._openai.ChatCompletion.create(
                        model=cfg.model_name,
                        messages=[{"role": "system", "content": cfg.system_prompt}, *oai_messages],
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        max_tokens=cfg.max_tokens,
                        stream=True,
                        api_key=self.api_key,  # type: ignore
                    )
                    for chunk in resp:
                        delta = (chunk.get("choices", [{}])[0].get("delta", {}) or {}).get("content")
                        if delta:
                            yield delta
                    return

                # Ścieżka C: client.chat.completions
                if self.client and hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
                    resp = self.client.chat.completions.create(
                        model=cfg.model_name,
                        messages=[{"role": "system", "content": cfg.system_prompt}, *oai_messages],
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        max_tokens=cfg.max_tokens,
                        stream=True,
                    )
                    for chunk in resp:
                        try:
                            delta = chunk.choices[0].delta.content  # type: ignore
                            if delta:
                                yield delta
                        except Exception:
                            continue
                    return

                # Jeśli doszliśmy tutaj, brak obsługiwanej ścieżki
                raise RuntimeError("Nieobsługiwana wersja pakietu openai.")

            except Exception as e:
                log.warning(f"OpenAI attempt {attempt+1} failed: {e}")
                if attempt < retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise

# === NAZWA_SEKCJI === Provider: Offline (fallback) ===

class OfflineProvider:
    """
    Deterministyczny fallback bez zewnętrznych zależności.
    - Gdy brak klucza API, użytkownik nadal może testować UI i logikę przepływu.
    """
    def chat_stream(
        self, messages: List[ChatMessage], cfg: ChatConfig
    ) -> Generator[str, None, None]:
        # prosta reguła: podsumuj ostatnią wiadomość użytkownika
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        base = "Tryb OFFLINE — brak klucza OPENAI_API_KEY. Oto pomocny szkic odpowiedzi: "
        content = last_user.content if last_user else "(brak treści użytkownika)"
        pseudo = f"{base}{content[:800]}"
        # Udawane „streaming”
        for ch in pseudo:
            yield ch
            time.sleep(0.002)

# === NAZWA_SEKCJI === Fabryka providera ===

def _provider_factory() -> ChatProvider:
    try:
        if _has_api_key():
            return OpenAIChatProvider()
        return OfflineProvider()
    except Exception as e:
        log.error(f"Provider factory fallback to Offline due to: {e}")
        return OfflineProvider()

# === NAZWA_SEKCJI === Cache: ładowanie system prompt z pliku (opcjonalnie) ===

@st.cache_data(show_spinner=False, max_entries=8, ttl=3600)
def load_system_prompt_from_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        log.warning(f"Nie udało się wczytać system prompt z {path}: {e}")
        return None

# === NAZWA_SEKCJI === Sesyjna pamięć rozmowy ===

def _init_state() -> None:
    st.session_state.setdefault("chat_history", [])  # List[Dict]
    st.session_state.setdefault("chat_cfg", ChatConfig())
    st.session_state.setdefault("openai_api_key", st.session_state.get("openai_api_key", None))

def _append_message(role: str, content: str) -> None:
    msg = ChatMessage(role=role, content=content)
    st.session_state.chat_history.append(asdict(msg))

def _get_messages() -> List[ChatMessage]:
    return [ChatMessage(**m) for m in st.session_state.chat_history]

def _clear_chat() -> None:
    st.session_state.chat_history = []

# === NAZWA_SEKCJI === Eksport historii ===

def _export_history_json() -> bytes:
    data = {
        "exported_at": time.time(),
        "messages": st.session_state.chat_history,
        "config": asdict(st.session_state.chat_cfg),
    }
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

# === NAZWA_SEKCJI === UI główne czatu ===

def render_chat_ui(
    *,
    title: str = "🤖 AI Chat — PRO+++",
    system_prompt_path: Optional[str] = None,
    allow_user_prompt_edit: bool = True,
) -> None:
    """
    Renderuje kompletny interfejs czatu (Streamlit).
    Wpięcie w app.py: `from src.frontend.chat_interface import render_chat_ui; render_chat_ui()`
    """
    _init_state()

    st.header(title)
    st.caption("Interfejs czatu z obsługą OpenAI lub trybem Offline (fallback).")

    # — Panel konfiguracyjny
    with st.expander("⚙️ Ustawienia modelu", expanded=False):
        left, right = st.columns([2, 1])

        with left:
            # Klucz API — zgodnie z Twoją „security check” praktyką
            st.text_input(
                "OPENAI_API_KEY (lokalnie na czas sesji)",
                type="password",
                key="openai_api_key",
                help="Klucz zostanie użyty tylko w tej sesji Streamlit. Alternatywnie użyj st.secrets lub .env.",
            )

            # System prompt — z pliku lub własny
            loaded = load_system_prompt_from_file(system_prompt_path)
            default_prompt = loaded if loaded else st.session_state.chat_cfg.system_prompt
            if allow_user_prompt_edit:
                new_prompt = st.text_area(
                    "System Prompt",
                    value=default_prompt,
                    height=140,
                )
                st.session_state.chat_cfg.system_prompt = new_prompt
            else:
                st.code(default_prompt, language="markdown")

        with right:
            st.session_state.chat_cfg.model_name = st.selectbox(
                "Model",
                options=[
                    "gpt-4o-mini",
                    "gpt-4o",
                    "gpt-4.1-mini",
                    "gpt-4.1",
                    "o4-mini",
                ],
                index=0,
            )
            st.session_state.chat_cfg.temperature = float(
                st.slider("Temperature", 0.0, 1.0, value=float(st.session_state.chat_cfg.temperature), step=0.05)
            )
            st.session_state.chat_cfg.top_p = float(
                st.slider("top_p", 0.1, 1.0, value=float(st.session_state.chat_cfg.top_p), step=0.05)
            )
            st.session_state.chat_cfg.max_tokens = int(
                st.number_input("max_tokens", min_value=256, max_value=8192, value=int(st.session_state.chat_cfg.max_tokens), step=128)
            )

    # — Informacja o trybie
    if not _has_api_key():
        st.warning(
            "Działasz w **TRYBIE OFFLINE** (brak `OPENAI_API_KEY` w secrets/session/env). "
            "Możesz nadal testować UI — odpowiedzi będą szkicem (deterministycznym).",
            icon="⚠️",
        )
    else:
        st.success("Wykryto klucz OpenAI. Odpowiedzi będą generowane przez model.", icon="✅")

    # — Historia rozmowy
    for m in _get_messages():
        with st.chat_message(m.role):
            st.markdown(m.content)

    # — Pole wejścia użytkownika
    prompt = st.chat_input("Napisz wiadomość…")
    if prompt:
        _append_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Odpowiedź modelu (stream)
        provider = _provider_factory()
        assistant_container = st.chat_message("assistant")
        with assistant_container:
            placeholder = st.empty()
            full_reply = []
            try:
                for token in provider.chat_stream(_get_messages(), st.session_state.chat_cfg):
                    full_reply.append(token)
                    placeholder.markdown("".join(full_reply))
            except Exception as e:
                log.exception("Błąd podczas generowania odpowiedzi.")
                st.error(f"Nie udało się uzyskać odpowiedzi: {e}")
                return

            reply_text = "".join(full_reply).strip()
            if reply_text:
                _append_message("assistant", reply_text)

    # — Akcje (reset, eksport)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🧹 Wyczyść rozmowę", use_container_width=True):
            _clear_chat()
            st.experimental_rerun()
    with col2:
        data = _export_history_json()
        st.download_button(
            "💾 Eksportuj JSON",
            data=data,
            file_name="chat_history.json",
            mime="application/json",
            use_container_width=True,
        )
    with col3:
        st.caption(" ")

# === NAZWA_SEKCJI === Punkt wejścia (opcjonalny dla lokalnego uruchomienia) ===

if __name__ == "__main__":
    # Pozwala szybko odpalić moduł komendą: streamlit run src/frontend/chat_interface.py
    render_chat_ui()
