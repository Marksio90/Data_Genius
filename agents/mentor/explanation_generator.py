"""
DataGenius PRO+++++++++++++ — Explanation Generator (Enterprise)
Generates user-friendly, Polish explanations of EDA/ML results with LLM (safe fallbacks),
defensive guards, telemetry and stable output contract.

Output (AgentResult.data):
{
  "explanation": str,
  "content_type": str,
  "user_level": "beginner"|"intermediate"|"advanced",
  "telemetry": {"elapsed_ms": float, "llm_used": bool, "llm_latency_ms": float|null}
}
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd  # noqa: F401 (import kept for parity with project style)
from loguru import logger

from core.base_agent import BaseAgent, AgentResult
from core.llm_client import get_llm_client
from agents.mentor.prompt_templates import (
    EDA_EXPLANATION_TEMPLATE,
    ML_RESULTS_TEMPLATE,
)
try:
    # Opcjonalny szablon — nie zawsze obecny w starszych buildach
    from agents.mentor.prompt_templates import MODEL_COMPARISON_TEMPLATE  # type: ignore
except Exception:  # pragma: no cover
    MODEL_COMPARISON_TEMPLATE = None  # fallback w kodzie


# === KONFIG ===
@dataclass(frozen=True)
class ExplanationConfig:
    """Konfiguracja generatora wyjaśnień."""
    temperature: float = 0.6
    max_tokens: int = 1400
    timeout_s: float = 45.0
    # limity/specyfika per-typ
    eda_max_tokens: int = 1500
    ml_max_tokens: int = 1500
    model_cmp_max_tokens: int = 1500
    feature_imp_max_tokens: int = 1000
    metrics_max_tokens: int = 1200
    data_quality_max_tokens: int = 1000


class ExplanationGenerator(BaseAgent):
    """
    Generates clear, user-friendly explanations in Polish.
    Translates technical ML/DS concepts into simple language.
    Enterprise-grade: defensive guards, telemetry, safe LLM fallbacks.
    """

    def __init__(self, config: Optional[ExplanationConfig] = None) -> None:
        super().__init__(name="ExplanationGenerator", description="Generates user-friendly explanations")
        self.config = config or ExplanationConfig()
        self._log = logger.bind(agent=self.name)
        # LLM klient może nie być dostępny — działamy wtedy wyłącznie na fallbackach
        try:
            self.llm_client = get_llm_client()
        except Exception as e:
            self._log.warning(f"LLM client unavailable — running in offline mode. Reason: {e}")
            self.llm_client = None

    # === API GŁÓWNE ===
    def execute(
        self,
        content_type: str,
        content: Dict[str, Any],
        user_level: str = "beginner",
        **kwargs: Any
    ) -> AgentResult:
        """
        Generate explanation.

        Args:
            content_type: one of {"eda","ml_results","model_comparison","feature_importance","metrics","data_quality", other→generic}
            content: payload to explain (dict per orchestrator agents)
            user_level: "beginner" | "intermediate" | "advanced"

        Returns:
            AgentResult with:
              data = {
                "explanation": str,
                "content_type": str,
                "user_level": str,
                "telemetry": {"elapsed_ms": float, "llm_used": bool, "llm_latency_ms": float|None}
              }
        """
        t0 = time.perf_counter()
        result = AgentResult(agent_name=self.name)

        # Normalizacja i guardy
        ct = (content_type or "generic").strip().lower()
        lvl = (user_level or "beginner").strip().lower()
        if lvl not in {"beginner", "intermediate", "advanced"}:
            lvl = "beginner"

        llm_used = False
        llm_latency_ms: Optional[float] = None

        try:
            # Routing po typie
            if ct == "eda":
                explanation, llm_used, llm_latency_ms = self.explain_eda_results(content, lvl)
            elif ct == "ml_results":
                explanation, llm_used, llm_latency_ms = self.explain_ml_results(content, lvl)
            elif ct == "model_comparison":
                explanation, llm_used, llm_latency_ms = self.explain_model_comparison(
                    content if isinstance(content, list) else content.get("models", []), lvl
                )
            elif ct == "feature_importance":
                explanation, llm_used, llm_latency_ms = self.explain_feature_importance(content, lvl)
            elif ct == "metrics":
                # spodziewamy się {"metrics": {...}, "problem_type": "..."}
                metrics = content.get("metrics", content)
                problem_type = str(content.get("problem_type", "classification")).lower()
                explanation, llm_used, llm_latency_ms = self.explain_metrics(metrics, problem_type, lvl)
            elif ct == "data_quality":
                explanation, llm_used, llm_latency_ms = self.explain_data_quality(content, lvl)
            else:
                explanation, llm_used, llm_latency_ms = self.explain_generic(content, lvl)

            result.data = {
                "explanation": explanation,
                "content_type": ct,
                "user_level": lvl,
                "telemetry": {
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                    "llm_used": bool(llm_used),
                    "llm_latency_ms": None if llm_latency_ms is None else round(llm_latency_ms, 1),
                },
            }
            self._log.success(f"Explanation generated for '{ct}' (llm_used={llm_used})")

        except Exception as e:
            result.add_error(f"Explanation generation failed: {e}")
            self._log.exception(f"Explanation error: {e}")

        return result

    # === EDA ===
    def explain_eda_results(
        self,
        eda_results: Dict[str, Any],
        user_level: str = "beginner"
    ) -> Tuple[str, bool, Optional[float]]:
        """Explain EDA results in user-friendly way."""
        llm_used = False
        llm_latency = None
        try:
            summary = self._summarize_eda_results(eda_results)
            prompt = EDA_EXPLANATION_TEMPLATE.format(eda_results=summary, user_level=user_level)
            if self.llm_client is None:
                raise RuntimeError("LLM unavailable")
            t0 = time.perf_counter()
            resp = self._llm_generate(prompt, temperature=self.config.temperature, max_tokens=self.config.eda_max_tokens)
            llm_latency = (time.perf_counter() - t0) * 1000.0
            llm_used = True
            return (self._llm_text(resp), llm_used, llm_latency)
        except Exception as e:
            self._log.debug(f"EDA explanation via LLM failed, using fallback. Reason: {e}")
            return (self._fallback_eda_explanation(eda_results), llm_used, llm_latency)

    # === ML RESULTS ===
    def explain_ml_results(
        self,
        ml_results: Dict[str, Any],
        user_level: str = "beginner"
    ) -> Tuple[str, bool, Optional[float]]:
        """Explain ML training results."""
        llm_used = False
        llm_latency = None
        try:
            summary = self._summarize_ml_results(ml_results)
            prompt = ML_RESULTS_TEMPLATE.format(ml_results=summary, user_level=user_level)
            if self.llm_client is None:
                raise RuntimeError("LLM unavailable")
            t0 = time.perf_counter()
            resp = self._llm_generate(prompt, temperature=self.config.temperature, max_tokens=self.config.ml_max_tokens)
            llm_latency = (time.perf_counter() - t0) * 1000.0
            llm_used = True
            return (self._llm_text(resp), llm_used, llm_latency)
        except Exception as e:
            self._log.debug(f"ML results explanation via LLM failed, using fallback. Reason: {e}")
            return (self._fallback_ml_explanation(ml_results), llm_used, llm_latency)

    # === MODEL COMPARISON ===
    def explain_model_comparison(
        self,
        models_comparison: List[Dict[str, Any]],
        user_level: str = "beginner"
    ) -> Tuple[str, bool, Optional[float]]:
        """Explain comparison between different models."""
        llm_used = False
        llm_latency = None
        try:
            if MODEL_COMPARISON_TEMPLATE is None or self.llm_client is None:
                raise RuntimeError("LLM/template unavailable")
            compact = self._compact_model_comparison(models_comparison)
            prompt = MODEL_COMPARISON_TEMPLATE.format(models_comparison=compact, user_level=user_level)
            t0 = time.perf_counter()
            resp = self._llm_generate(
                prompt, temperature=self.config.temperature, max_tokens=self.config.model_cmp_max_tokens
            )
            llm_latency = (time.perf_counter() - t0) * 1000.0
            llm_used = True
            return (self._llm_text(resp), llm_used, llm_latency)
        except Exception as e:
            self._log.debug(f"Model comparison via LLM failed, using fallback. Reason: {e}")
            return (self._fallback_model_comparison(models_comparison), llm_used, llm_latency)

    # === FEATURE IMPORTANCE ===
    def explain_feature_importance(
        self,
        feature_importance: Dict[str, Any],
        user_level: str = "beginner"
    ) -> Tuple[str, bool, Optional[float]]:
        """Explain feature importance in simple terms."""
        prompt = (
            f"Wyjaśnij ważność cech (feature importance) po polsku, poziom: {user_level}\n\n"
            f"Dane o ważności cech (JSON):\n{json.dumps(feature_importance, ensure_ascii=False, indent=2)}\n\n"
            "Stwórz wyjaśnienie, które:\n"
            "1) tłumaczy, co oznacza 'ważność cechy',\n"
            "2) wskazuje 3–5 najważniejszych cech i krótko wyjaśnia ich wpływ,\n"
            "3) podaje praktyczne wnioski dla modelu/produkcji.\n"
            "Dostosuj styl do poziomu użytkownika."
        )
        llm_used = False
        llm_latency = None
        try:
            if self.llm_client is None:
                raise RuntimeError("LLM unavailable")
            t0 = time.perf_counter()
            resp = self._llm_generate(
                prompt, temperature=self.config.temperature, max_tokens=self.config.feature_imp_max_tokens
            )
            llm_latency = (time.perf_counter() - t0) * 1000.0
            llm_used = True
            return (self._llm_text(resp), llm_used, llm_latency)
        except Exception as e:
            self._log.debug(f"Feature importance via LLM failed, using fallback. Reason: {e}")
            return (self._fallback_feature_importance(feature_importance), llm_used, llm_latency)

    # === METRICS ===
    def explain_metrics(
        self,
        metrics: Dict[str, float],
        problem_type: str,
        user_level: str = "beginner"
    ) -> Tuple[str, bool, Optional[float]]:
        """Explain ML metrics in simple terms."""
        pt = (problem_type or "classification").lower().strip()
        prompt = (
            f"Wyjaśnij metryki modelu typu '{pt}' po polsku, poziom: {user_level}.\n\n"
            f"Metryki (JSON):\n{json.dumps(metrics, ensure_ascii=False, indent=2)}\n\n"
            "Dla każdej metryki podaj:\n"
            "• prostą definicję,\n"
            "• interpretację wartości (dobre/średnie/słabe),\n"
            "• praktyczny wniosek.\n"
            "Zachowaj zwięzły, rzeczowy styl i czytelne sekcje."
        )
        llm_used = False
        llm_latency = None
        try:
            if self.llm_client is None:
                raise RuntimeError("LLM unavailable")
            t0 = time.perf_counter()
            resp = self._llm_generate(
                prompt, temperature=self.config.temperature, max_tokens=self.config.metrics_max_tokens
            )
            llm_latency = (time.perf_counter() - t0) * 1000.0
            llm_used = True
            return (self._llm_text(resp), llm_used, llm_latency)
        except Exception as e:
            self._log.debug(f"Metrics explanation via LLM failed, using fallback. Reason: {e}")
            return (self._fallback_metrics_explanation(metrics, pt), llm_used, llm_latency)

    # === DATA QUALITY ===
    def explain_data_quality(
        self,
        quality_assessment: Dict[str, Any],
        user_level: str = "beginner"
    ) -> Tuple[str, bool, Optional[float]]:
        """Explain data quality assessment."""
        tone = "przyjazny, bez żargonu" if user_level == "beginner" else "techniczny, precyzyjny"
        prompt = (
            f"Wyjaśnij ocenę jakości danych po polsku (poziom: {user_level}, ton: {tone}).\n\n"
            f"Dane oceny (JSON):\n{json.dumps(quality_assessment, ensure_ascii=False, indent=2)}\n\n"
            "Podaj:\n"
            "1) ogólną ocenę (0–100) i krótką interpretację,\n"
            "2) najważniejsze problemy (priorytetowo),\n"
            "3) wpływ na model ML,\n"
            "4) konkretne rekomendacje naprawcze."
        )
        llm_used = False
        llm_latency = None
        try:
            if self.llm_client is None:
                raise RuntimeError("LLM unavailable")
            t0 = time.perf_counter()
            resp = self._llm_generate(
                prompt, temperature=self.config.temperature, max_tokens=self.config.data_quality_max_tokens
            )
            llm_latency = (time.perf_counter() - t0) * 1000.0
            llm_used = True
            return (self._llm_text(resp), llm_used, llm_latency)
        except Exception as e:
            self._log.debug(f"Data quality explanation via LLM failed, using fallback. Reason: {e}")
            return (self._fallback_quality_explanation(quality_assessment), llm_used, llm_latency)

    # === GENERIC ===
    def explain_generic(
        self,
        content: Dict[str, Any],
        user_level: str = "beginner"
    ) -> Tuple[str, bool, Optional[float]]:
        """Generic explanation for any content."""
        prompt = (
            f"Wyjaśnij poniższe wyniki analizy po polsku, poziom: {user_level}.\n\n"
            f"Dane (JSON):\n{json.dumps(content, ensure_ascii=False, indent=2)}\n\n"
            "Podaj najważniejsze wnioski, praktyczne implikacje i krótkie rekomendacje."
        )
        llm_used = False
        llm_latency = None
        try:
            if self.llm_client is None:
                raise RuntimeError("LLM unavailable")
            t0 = time.perf_counter()
            resp = self._llm_generate(prompt, temperature=self.config.temperature, max_tokens=self.config.max_tokens)
            llm_latency = (time.perf_counter() - t0) * 1000.0
            llm_used = True
            return (self._llm_text(resp), llm_used, llm_latency)
        except Exception as e:
            self._log.debug(f"Generic explanation via LLM failed, using fallback. Reason: {e}")
            return ("Nie udało się wygenerować wyjaśnienia LLM — podaję skrócone podsumowanie danych wejściowych:\n"
                    f"{self._brief_json(content)}", llm_used, llm_latency)

    # ==================== HELPERS: SUMMARIZERS ====================

    def _summarize_eda_results(self, eda_results: Dict[str, Any]) -> str:
        """Summarize EDA results for LLM prompt (odporne na brak pól)."""
        parts: List[str] = []

        # Data overview
        summary = eda_results.get("summary", {})
        shape = summary.get("dataset_shape") or summary.get("shape") or summary.get("n_rows_cols")
        if shape:
            parts.append(f"Dataset: {shape}")

        # Key findings
        findings = summary.get("key_findings") or summary.get("insights") or []
        if isinstance(findings, list) and findings:
            parts.append("Kluczowe odkrycia: " + ", ".join(map(str, findings[:5])))

        # Missing data (DataGenius MissingDataAnalyzer contract)
        missing = (
            eda_results.get("eda_results", {})
            .get("MissingDataAnalyzer")
        )
        if isinstance(missing, dict):
            msum = missing.get("summary", {}) or {}
            total_missing = int(msum.get("total_missing", 0))
            missing_pct = float(msum.get("missing_percentage", 0.0))
            parts.append(f"Braki danych: {total_missing} ({missing_pct:.2f}%)")

        # Outliers (DataGenius OutlierDetector contract)
        out = (
            eda_results.get("eda_results", {})
            .get("OutlierDetector")
        )
        if isinstance(out, dict):
            osum = out.get("summary", {}) or {}
            # kontrakt: total_outliers_rows_union
            total_outliers = int(osum.get("total_outliers_rows_union", osum.get("total_outliers", 0)))
            parts.append(f"Outliers (wiersze, unia): {total_outliers}")

        return "\n".join(parts) if parts else "Brak skondensowanych wyników EDA."

    def _summarize_ml_results(self, ml_results: Dict[str, Any]) -> str:
        """Summarize ML results for LLM prompt (odporne na brak pól)."""
        parts: List[str] = []
        summary = ml_results.get("summary", {}) or {}

        best_model = summary.get("best_model") or summary.get("model")
        if best_model:
            parts.append(f"Najlepszy model: {best_model}")

        best_score = summary.get("best_score") or summary.get("score")
        if best_score is not None:
            try:
                parts.append(f"Wynik: {float(best_score):.4f}")
            except Exception:
                parts.append(f"Wynik: {best_score}")

        insights = summary.get("key_insights") or summary.get("insights") or []
        if isinstance(insights, list) and insights:
            parts.append("Insights: " + ", ".join(map(str, insights[:3])))

        return "\n".join(parts) if parts else "Brak skondensowanych wyników ML."

    def _compact_model_comparison(self, models: List[Dict[str, Any]]) -> str:
        """Kompaktuje listę wyników modeli do zwięzłej tabelarycznej formy w tekście."""
        if not isinstance(models, list) or not models:
            return "Brak wyników modeli do porównania."
        rows: List[str] = []
        for i, m in enumerate(models[:8], 1):
            name = str(m.get("name", f"Model_{i}"))
            score = m.get("score", m.get("metric", None))
            try:
                score_txt = f"{float(score):.4f}" if score is not None else "N/A"
            except Exception:
                score_txt = str(score)
            extra = m.get("extra") or m.get("notes") or ""
            rows.append(f"{i}. {name} — score={score_txt} {(' | ' + str(extra)) if extra else ''}")
        return "\n".join(rows)

    # ==================== HELPERS: LLM ====================

    def _llm_generate(self, prompt: str, temperature: float, max_tokens: int) -> Any:
        """Abstrakcja nad klientem LLM — zgodna z różnymi implementacjami."""
        # Preferowana sygnatura: generate(prompt=..., temperature=..., max_tokens=..., timeout=...)
        if self.llm_client is None:
            raise RuntimeError("LLM client not initialized")
        try:
            return self.llm_client.generate(
                prompt=prompt,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                timeout=self.config.timeout_s,
            )
        except TypeError:
            # starsze klienty bez timeoutu
            return self.llm_client.generate(
                prompt=prompt,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )

    @staticmethod
    def _llm_text(resp: Any) -> str:
        """Wydobywa tekst z odpowiedzi LLM (obsługa .content / .text / string)."""
        if resp is None:
            return ""
        if isinstance(resp, str):
            return resp
        for attr in ("content", "text", "output"):
            if hasattr(resp, attr):
                val = getattr(resp, attr)
                if isinstance(val, str):
                    return val
        # fallback na repr — by zachować treść w logach
        return str(resp)

    @staticmethod
    def _brief_json(obj: Any, limit: int = 1200) -> str:
        try:
            s = json.dumps(obj, ensure_ascii=False, default=str, indent=2)
        except Exception:
            s = str(obj)
        return s if len(s) <= limit else s[:limit] + f"...(+{len(s)-limit} chars)"

    # ==================== FALLBACKS ====================

    def _fallback_eda_explanation(self, eda_results: Dict[str, Any]) -> str:
        explanation = "🔍 **Podsumowanie Analizy EDA**\n\n"
        summary = eda_results.get("summary", {}) or {}
        shape = summary.get("dataset_shape")
        if shape and isinstance(shape, (list, tuple)) and len(shape) == 2:
            explanation += f"- Rozmiar: **{shape[0]} wierszy × {shape[1]} kolumn**\n"

        findings = summary.get("key_findings") or []
        if isinstance(findings, list) and findings:
            explanation += "\n**Kluczowe odkrycia:**\n"
            for f in findings[:8]:
                explanation += f"- {f}\n"

        recs = summary.get("recommendations") or []
        if isinstance(recs, list) and recs:
            explanation += "\n**Rekomendacje:**\n"
            for r in recs[:8]:
                explanation += f"- {r}\n"
        return explanation

    def _fallback_ml_explanation(self, ml_results: Dict[str, Any]) -> str:
        explanation = "🤖 **Wyniki Trenowania Modelu**\n\n"
        summary = ml_results.get("summary", {}) or {}

        best_model = summary.get("best_model")
        if best_model:
            explanation += f"- Najlepszy model: **{best_model}**\n"

        score = summary.get("best_score")
        try:
            if score is not None:
                val = float(score)
                explanation += f"- Wynik: **{val:.2%}**\n"
                explanation += ("✅ Bardzo dobry wynik!\n" if val > 0.9 else
                                "👍 Dobry wynik.\n" if val > 0.75 else
                                "⚠️ Wynik wymaga poprawy.\n")
        except Exception:
            if score is not None:
                explanation += f"- Wynik: **{score}**\n"

        insights = summary.get("key_insights") or []
        if isinstance(insights, list) and insights:
            explanation += "\n**Kluczowe wnioski:**\n"
            for i in insights[:5]:
                explanation += f"- {i}\n"
        return explanation

    def _fallback_model_comparison(self, models: List[Dict[str, Any]]) -> str:
        explanation = "📊 **Porównanie Modeli**\n\n"
        if not models:
            return explanation + "Brak danych o modelach."
        for i, m in enumerate(models[:5], 1):
            name = str(m.get("name", f"Model {i}"))
            score = m.get("score", m.get("metric"))
            try:
                score_txt = f"{float(score):.2%}" if score is not None else "N/A"
            except Exception:
                score_txt = str(score)
            explanation += f"{i}. **{name}** — wynik: {score_txt}\n"
        return explanation

    def _fallback_feature_importance(self, importance: Dict[str, Any]) -> str:
        explanation = "🔑 **Ważność Cech**\n\n"
        top: List[Tuple[str, float]] = []

        if isinstance(importance, dict):
            # różne możliwe struktury wejściowe
            if "top_features" in importance and isinstance(importance["top_features"], list):
                for it in importance["top_features"]:
                    if isinstance(it, (list, tuple)) and len(it) >= 2:
                        top.append((str(it[0]), float(it[1])))
                    else:
                        top.append((str(it), float("nan")))
            elif "importances" in importance and isinstance(importance["importances"], dict):
                for k, v in importance["importances"].items():
                    try:
                        top.append((str(k), float(v)))
                    except Exception:
                        top.append((str(k), float("nan")))
        if top:
            explanation += "Najważniejsze cechy wpływające na predykcje:\n"
            for i, (f, v) in enumerate(sorted(top, key=lambda x: (x[1] if x[1] == x[1] else -1), reverse=True)[:5], 1):
                vtxt = f"{v:.4f}" if v == v else "N/A"
                explanation += f"{i}. {f} — {vtxt}\n"
        else:
            explanation += "Brak czytelnej listy istotności cech.\n"
        return explanation

    def _fallback_metrics_explanation(self, metrics: Dict[str, float], problem_type: str) -> str:
        explanation = "📈 **Metryki Modelu**\n\n"
        for k, v in (metrics or {}).items():
            try:
                explanation += f"- **{k}**: {float(v):.4f}\n"
            except Exception:
                explanation += f"- **{k}**: {v}\n"
        explanation += "\nInterpretacja metryk zależy od kontekstu problemu i danych walidacyjnych."
        return explanation

    def _fallback_quality_explanation(self, quality: Dict[str, Any]) -> str:
        explanation = "📊 **Ocena Jakości Danych**\n\n"
        score = quality.get("quality_score")
        if score is not None:
            try:
                v = float(score)
                explanation += f"Ogólna ocena: **{v:.1f}/100**\n"
                if v > 80:
                    explanation += "✅ Bardzo dobra jakość danych.\n"
                elif v > 60:
                    explanation += "👍 Dobra jakość, przyda się kilka poprawek.\n"
                else:
                    explanation += "⚠️ Dane wymagają istotnych korekt.\n"
            except Exception:
                explanation += f"Ogólna ocena: **{score}**\n"

        issues = quality.get("issues") or quality.get("problems") or []
        if isinstance(issues, list) and issues:
            explanation += "\n**Najważniejsze problemy:**\n"
            for it in issues[:8]:
                explanation += f"- {it}\n"

        recs = quality.get("recommendations") or []
        if isinstance(recs, list) and recs:
            explanation += "\n**Rekomendacje:**\n"
            for r in recs[:8]:
                explanation += f"- {r}\n"
        return explanation
