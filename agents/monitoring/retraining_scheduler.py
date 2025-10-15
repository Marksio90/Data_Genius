# === retraining_scheduler.py (PRO++++ / KOSMOS) ===
"""
DataGenius PRO++++ — Retraining Scheduler (KOSMOS)
Decyduje o retrainie na podstawie: driftu danych, spadku jakości, wieku modelu i wolumenu nowych danych.
Generuje rekomendowany harmonogram (cron + iCal VEVENT), obsługuje cooldown i limity tygodniowe,
a opcjonalnie uruchamia retraining przez MLOrchestrator. Zapisuje audyt do retraining_log.csv.

Najważniejsze cechy KOSMOS:
- Sygnalizacja: drift (% cech, top cechy, target drift), performance delta (accuracy/r2), wiek modelu, wolumen.
- Skoring decyzyjny: wagi drift/perf/age + progi priorytetów (low/medium/high).
- Triggery „twarde” (hard): critical drift, target drift, krytyczny wiek.
- Polityki operacyjne: cooldown dniowy, limit retrainów / 7 dni, minimalny wolumen próbek.
- Harmonogram: najbliższe okno w preferowanej godzinie + cron + iCal VEVENT (z TZ jeśli dostępny).
- Tryb natychmiastowy: jeśli dostarczono orchestrator + dane → uruchamia retrain i dołącza wynik.
- Bogaty kontrakt wyniku: {decision, schedule, signals, retrain_result, audit_log_path} + telemetry.
- Defensywa: bezpieczne parsowanie, odporność na braki, stabilne typy, brak twardych zależności spoza stdlib/pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Literal, Tuple
from pathlib import Path
from datetime import datetime, timedelta, timezone
import math
import json

import numpy as np
import pandas as pd
from loguru import logger

from core.base_agent import BaseAgent, AgentResult
from config.settings import settings

try:
    # Python 3.9+: stdlib
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:  # pragma: no cover
    ZoneInfo = None  # fallback – użyj UTC


# === POLITYKA RETRAINU ===
@dataclass(frozen=True)
class RetrainPolicy:
    # Drift (z DriftDetector)
    drift_warn_pct: float = 10.0          # % cech z driftem → ostrzeżenie
    drift_crit_pct: float = 30.0          # % cech z driftem → krytyczny trigger
    target_drift_triggers: bool = True    # drift targetu zawsze triggeruje

    # Performance (z PerformanceTracker.compare)
    max_acc_drop_pct: float = 2.0         # dopuszczalny spadek accuracy [p.p.] vs baseline
    max_f1_drop_pct: float = 2.0          # dopuszczalny spadek F1 [p.p.] (informacyjnie)
    max_r2_drop_abs: float = 0.03         # spadek R^2 (bezwzględny)

    # Wiek i wolumen
    age_warn_days: int = 14               # ostrzeżenie wieku modelu
    age_crit_days: int = 30               # krytyczny wiek modelu
    min_new_samples: int = 5_000          # minimalny wolumen nowych próbek

    # Higiena operacyjna
    cooldown_days: int = 3                # min przerwa między retrainami
    max_retrains_per_week: int = 2        # limit ochronny

    # Harmonogram preferowany
    preferred_hour: int = 2               # 02:30 lokalnie
    preferred_minute: int = 30
    days_of_week: Optional[List[int]] = None  # None=codziennie, albo lista 0..6 (Mon=0)
    timezone: str = getattr(settings, "TIMEZONE", "Europe/Warsaw")

    # Skoring decyzyjny
    weight_drift: float = 0.5
    weight_perf: float = 0.3
    weight_age: float = 0.2

    # Priorytety
    priority_high_threshold: float = 0.7
    priority_medium_threshold: float = 0.4


class RetrainingScheduler(BaseAgent):
    """
    Ocena konieczności retrainu + harmonogram + (opcjonalny) natychmiastowy retrain.
    """

    version: str = "3.4-kosmos"

    def __init__(self, policy: Optional[RetrainPolicy] = None):
        super().__init__(
            name="RetrainingScheduler",
            description="Schedules and optionally triggers ML retraining based on drift/metrics/age"
        )
        self.policy = policy or RetrainPolicy()
        # ścieżki
        self.metrics_path: Path = Path(getattr(settings, "METRICS_PATH", Path("metrics")))
        self.metrics_path.mkdir(parents=True, exist_ok=True)
        self.log_path: Path = self.metrics_path / "retraining_log.csv"

    # === GŁÓWNY INTERFEJS ===
    def execute(
        self,
        problem_type: Literal["classification", "regression"],
        *,
        drift_report: Optional[Dict[str, Any]] = None,
        performance_data: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        last_train_ts: Optional[str] = None,   # ISO8601
        new_samples: Optional[int] = None,
        force: bool = False,

        # ewentualny natychmiastowy retrain
        train_data: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None,
        orchestrator: Optional[Any] = None,    # oczekujemy .execute(data=..., target_column=..., problem_type=...)
        orchestrator_kwargs: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Zdecyduj o retrainie i wygeneruj harmonogram. Opcjonalnie uruchom retrain natychmiast.

        Minimalne wymagania: `problem_type`, a do decyzji najlepiej podać chociaż jeden sygnał:
        `drift_report` lub `performance_data` albo `model_path/last_train_ts` + `new_samples`.

        Gdy podasz `train_data` + `target_column` + `orchestrator`, i decyzja==True → wykona retrain.
        """
        res = AgentResult(agent_name=self.name)
        t0 = datetime.utcnow()

        try:
            # 1) Zbierz sygnały
            drift = self._extract_drift_signals(drift_report)
            perf = self._extract_perf_signals(performance_data, problem_type)
            age_days = self._model_age_days(model_path, last_train_ts)
            cooldown_ok, cooldown_info = self._check_cooldown()
            weekly_ok, weekly_info = self._check_weekly_limit()
            vol_ok = (new_samples or 0) >= self.policy.min_new_samples if new_samples is not None else True

            # 2) Wyceń skorygowany score i priorytet
            score, parts = self._compute_score(drift, perf, age_days)
            priority = self._priority_from_score(score)

            # 3) Triggery binarne
            triggers, hard_trigger = self._compute_triggers(drift, perf, age_days)

            # 4) Decyzja (lub override force)
            should_retrain = (
                (hard_trigger or score >= self.policy.priority_medium_threshold)
                and cooldown_ok and weekly_ok and vol_ok
            ) or bool(force)

            # 5) Harmonogram (zalecenie)
            next_local_iso, cron, window, vevent = self._recommend_schedule()

            # 6) Opcjonalny natychmiastowy retrain
            retrain_result: Optional[Dict[str, Any]] = None
            status = "DECIDED_NO_ACTION"
            run_ok = False
            if should_retrain and orchestrator is not None and train_data is not None and target_column:
                try:
                    self.logger.info("Starting immediate retraining via orchestrator…")
                    run_ok, retrain_result = self._run_immediate_retrain(
                        orchestrator=orchestrator,
                        train_data=train_data,
                        target_column=target_column,
                        problem_type=problem_type,
                        orchestrator_kwargs=orchestrator_kwargs or {}
                    )
                    status = "RETRAIN_OK" if run_ok else "RETRAIN_FAILED"
                except Exception as e:
                    status = "RETRAIN_FAILED"
                    self.logger.error(f"Immediate retrain error: {e}", exc_info=True)

            # 7) Audit log (append-only)
            self._append_log_record({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "problem_type": problem_type,
                "decision": bool(should_retrain),
                "priority": priority,
                "score": round(float(score), 6),
                "drift_pct": drift["pct"],
                "target_drift": drift["target_drift"],
                "perf_delta_primary": perf["primary_delta"],
                "age_days": int(age_days),
                "new_samples": int(new_samples or -1),
                "cooldown_ok": bool(cooldown_ok),
                "weekly_ok": bool(weekly_ok),
                "volume_ok": bool(vol_ok),
                "status": status
            })

            # 8) Telemetria
            telemetry = {
                "elapsed_s": round((datetime.utcnow() - t0).total_seconds(), 4),
                "version": self.version,
                "policy": {
                    "cooldown_days": self.policy.cooldown_days,
                    "max_retrains_per_week": self.policy.max_retrains_per_week,
                    "min_new_samples": self.policy.min_new_samples
                }
            }

            # 9) Uzasadnienie decyzji (krótka narracja)
            reasoning = self._build_reasoning(
                should_retrain=should_retrain, score=score, priority=priority, parts=parts,
                triggers=triggers, cooldown_ok=cooldown_ok, weekly_ok=weekly_ok, vol_ok=vol_ok, force=force
            )

            # 10) Zwróć wynik
            res.data = {
                "decision": {
                    "should_retrain": bool(should_retrain),
                    "priority": priority,
                    "score": float(score),
                    "triggers": triggers,
                    "score_parts": parts,
                    "force": bool(force),
                    "cooldown": cooldown_info,
                    "weekly_limit": weekly_info,
                    "volume_ok": bool(vol_ok),
                    "reasoning": reasoning
                },
                "schedule": {
                    "next_time_local_iso": next_local_iso,
                    "cron": cron,
                    "window": window,
                    "ical_vevent": vevent
                },
                "signals": {
                    "drift": drift,
                    "performance": perf,
                    "age_days": int(age_days),
                    "new_samples": int(new_samples or -1),
                },
                "retrain_result": retrain_result if run_ok else None,
                "audit_log_path": str(self.log_path),
                "telemetry": telemetry
            }

            msg = "Retraining scheduled" if should_retrain else "Retraining not required"
            self.logger.success(f"{msg}: priority={priority}, score={score:.3f}")

        except Exception as e:
            res.add_error(f"Retraining scheduling failed: {e}")
            self.logger.error(f"Retraining scheduling error: {e}", exc_info=True)

        return res

    # === EKSTRAKCJA SYGNAŁÓW ===
    def _extract_drift_signals(self, drift_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        pct = 0.0
        target_drift = False
        top_features: List[str] = []

        if isinstance(drift_report, dict) and drift_report:
            try:
                # Dopuszczamy kilka wariantów kluczy (elastyczność)
                dd = drift_report.get("data_drift", drift_report.get("drift", {}))
                if isinstance(dd, dict):
                    # Obsługa alternatywnych nazw
                    pct = float(
                        dd.get("drift_score", dd.get("drift_pct", dd.get("pct_drifted_features", 0.0)))
                    )
                    drifted = dd.get("drifted_features", dd.get("top_drifted_features", [])) or []
                    top_features = list(map(str, drifted[:5]))
                td = drift_report.get("target_drift", {})
                target_drift = bool(td.get("is_drift", td.get("drift", False)))
            except Exception:
                pass

        return {
            "pct": float(pct),                     # % cech z driftem
            "target_drift": bool(target_drift),    # drift targetu?
            "top_features": top_features
        }

    def _extract_perf_signals(self, performance_data: Optional[Dict[str, Any]], problem_type: str) -> Dict[str, Any]:
        """
        performance_data: oczekujemy struktury z PerformanceTracker.execute().data
        i węzłem .comparison[<primary_key>].delta
        """
        primary_delta = 0.0  # dodatni = poprawa; ujemny = spadek
        detail: Dict[str, Any] = {}
        if not isinstance(performance_data, dict) or not performance_data:
            return {"primary_delta": float(primary_delta), "detail": detail}

        comparison = performance_data.get("comparison")
        if not isinstance(comparison, dict):
            return {"primary_delta": float(primary_delta), "detail": detail}

        key = "accuracy" if problem_type == "classification" else "r2"
        try:
            node = comparison.get(key, {})
            primary_delta = float(node.get("delta", 0.0))
            # przekaż resztę porównań (np. rmse/mae)
            detail = {k: v for k, v in comparison.items() if isinstance(v, dict)}
        except Exception:
            pass

        return {"primary_delta": float(primary_delta), "detail": detail}

    def _model_age_days(self, model_path: Optional[str], last_train_ts: Optional[str]) -> int:
        # Ustal wiek modelu w dniach na podstawie mtime pliku albo last_train_ts (ISO)
        try:
            if last_train_ts:
                dt = datetime.fromisoformat(last_train_ts.replace("Z", "+00:00"))
                return max(0, (datetime.utcnow().replace(tzinfo=timezone.utc) - dt.astimezone(timezone.utc)).days)
        except Exception:
            pass
        try:
            if model_path:
                p = Path(model_path)
                if p.exists():
                    mtime = datetime.utcfromtimestamp(p.stat().st_mtime).replace(tzinfo=timezone.utc)
                    return max(0, (datetime.utcnow().replace(tzinfo=timezone.utc) - mtime).days)
        except Exception:
            pass
        return -1  # nieznany

    # === SCORE & TRIGGERS ===
    def _compute_score(self, drift: Dict[str, Any], perf: Dict[str, Any], age_days: int) -> Tuple[float, Dict[str, float]]:
        pol = self.policy
        # drift: normalizacja do progu krytycznego
        drift_part = min(1.0, float(drift["pct"]) / max(pol.drift_crit_pct, 1e-9))

        # performance: osobno klasyfikacja vs. regresja;
        # tutaj traktujemy delta <0 jako spadek — normalizujemy względem dopuszczalnego spadku.
        if perf["primary_delta"] >= 0:
            perf_part = 0.0
        else:
            # dla acc — delta jest w [0..1] jeśli pochodzi z PerformanceTracker; ale bywa interpretowana jako różnica bezwzględna.
            # Bezpiecznie potraktujmy 0.01 * max_acc_drop_pct jako „dopuszczalny” spadek.
            denom = max(1e-9, 0.01 * pol.max_acc_drop_pct if pol.max_acc_drop_pct else 0.02)
            perf_part = min(1.0, abs(perf["primary_delta"]) / denom)

        # age
        if age_days < 0:
            age_part = 0.0
        elif age_days >= pol.age_crit_days:
            age_part = 1.0
        else:
            age_part = max(0.0, (age_days - pol.age_warn_days) / max(pol.age_crit_days - pol.age_warn_days, 1))

        score = pol.weight_drift * drift_part + pol.weight_perf * perf_part + pol.weight_age * age_part
        parts = {"drift": round(float(drift_part), 6), "performance": round(float(perf_part), 6), "age": round(float(age_part), 6)}
        return float(score), parts

    def _priority_from_score(self, score: float) -> Literal["low", "medium", "high"]:
        if score >= self.policy.priority_high_threshold:
            return "high"
        if score >= self.policy.priority_medium_threshold:
            return "medium"
        return "low"

    def _compute_triggers(self, drift: Dict[str, Any], perf: Dict[str, Any], age_days: int) -> Tuple[List[str], bool]:
        pol = self.policy
        triggers: List[str] = []
        hard = False

        if drift["pct"] >= pol.drift_crit_pct:
            triggers.append(f"data_drift_critical({drift['pct']:.1f}%)")
            hard = True
        elif drift["pct"] >= pol.drift_warn_pct:
            triggers.append(f"data_drift_warn({drift['pct']:.1f}%)")

        if pol.target_drift_triggers and drift["target_drift"]:
            triggers.append("target_drift_detected")
            hard = True

        # performance: delta < 0 (spadek)
        if perf["primary_delta"] < 0:
            triggers.append(f"performance_drop({perf['primary_delta']:.4f})")

        if age_days >= pol.age_crit_days:
            triggers.append(f"model_age_critical({age_days}d)")
            hard = True
        elif age_days >= pol.age_warn_days:
            triggers.append(f"model_age_warn({age_days}d)")

        return triggers, hard

    # === COOL-DOWN & WEEKLY LIMIT ===
    def _check_cooldown(self) -> Tuple[bool, Dict[str, Any]]:
        days = self.policy.cooldown_days
        if days <= 0:
            return True, {"cooldown_days": 0, "last_ts": None}

        hist = self._read_log()
        if hist.empty:
            return True, {"cooldown_days": days, "last_ts": None}

        last = hist.sort_values("timestamp").iloc[-1]
        try:
            last_ts = pd.to_datetime(last["timestamp"], utc=True)
        except Exception:
            return True, {"cooldown_days": days, "last_ts": None}

        ok = (pd.Timestamp.utcnow() - last_ts) >= pd.Timedelta(days=days)
        return bool(ok), {"cooldown_days": days, "last_ts": last_ts.isoformat()}

    def _check_weekly_limit(self) -> Tuple[bool, Dict[str, Any]]:
        limit = self.policy.max_retrains_per_week
        if limit <= 0:
            return True, {"limit": 0, "count_last_7d": 0}

        hist = self._read_log()
        if hist.empty:
            return True, {"limit": limit, "count_last_7d": 0}

        now = pd.Timestamp.utcnow()
        recent = hist[pd.to_datetime(hist["timestamp"], utc=True) >= (now - pd.Timedelta(days=7))]
        count = int((recent["status"] == "RETRAIN_OK").sum())
        return bool(count < limit), {"limit": limit, "count_last_7d": count}

    # === HARMONOGRAM ===
    def _now_local(self) -> datetime:
        tz = self.policy.timezone
        if ZoneInfo:
            try:
                return datetime.now(ZoneInfo(tz))  # type: ignore
            except Exception:
                pass
        # fallback UTC
        return datetime.utcnow().replace(tzinfo=timezone.utc)

    def _recommend_schedule(self) -> Tuple[str, str, Dict[str, Any], str]:
        """
        Zwraca: (next_time_local_iso, cron, window_info, ical_vevent)
        """
        now = self._now_local()
        hour = int(self.policy.preferred_hour)
        minute = int(self.policy.preferred_minute)
        dow = self.policy.days_of_week  # None albo lista 0..6 (Mon=0)

        # wyznacz najbliższy slot
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)

        allowed = set(dow) if dow else None
        if allowed:
            while candidate.weekday() not in allowed:
                candidate += timedelta(days=1)

        # cron (DOW: Sun=0, Mon=1, ... w standardowym cronie; my mamy Mon=0..Sun=6 → mapuj)
        cron_dow = "*"
        if dow and len(dow) > 0:
            cron_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0}
            cron_vals = sorted(cron_map[d] for d in dow if d in cron_map)
            cron_dow = ",".join(str(v) for v in cron_vals)

        cron = f"{minute} {hour} * * {cron_dow}"

        window = {
            "hour": hour,
            "minute": minute,
            "days_of_week": dow if dow is not None else "daily",
            "timezone": self.policy.timezone
        }

        # iCal VEVENT (jednorazowe zdarzenie; system może sam dodać RRULE jeśli potrzebne)
        dtstart = candidate.strftime("%Y%m%dT%H%M%S")
        tzid = self.policy.timezone if ZoneInfo else "UTC"
        vevent = (
            "BEGIN:VEVENT\n"
            f"DTSTART;TZID={tzid}:{dtstart}\n"
            f"SUMMARY:Model Retraining ({self.name})\n"
            "DESCRIPTION:Recommended retraining window by RetrainingScheduler\n"
            "END:VEVENT"
        )

        return candidate.isoformat(), cron, window, vevent

    # === OPCJONALNY NATYCHMIASTOWY RETRAIN ===
    def _run_immediate_retrain(
        self,
        *,
        orchestrator: Any,
        train_data: pd.DataFrame,
        target_column: str,
        problem_type: str,
        orchestrator_kwargs: Dict[str, Any]
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        if not hasattr(orchestrator, "execute"):
            raise ValueError("Provided orchestrator does not expose .execute(...)")

        out = orchestrator.execute(
            data=train_data,
            target_column=target_column,
            problem_type=problem_type,
            **orchestrator_kwargs
        )
        ok = (hasattr(out, "is_success") and out.is_success()) or (isinstance(out, dict) and "ml_results" in out)
        payload = out.data if hasattr(out, "data") else (out if isinstance(out, dict) else None)
        return bool(ok), payload

    # === LOGI / HISTORIA ===
    def _append_log_record(self, record: Dict[str, Any]) -> None:
        df = pd.DataFrame([record])
        header_needed = not self.log_path.exists()
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.log_path, mode="a", header=header_needed, index=False, encoding="utf-8")
        except Exception as e:
            self.logger.warning(f"Failed to append retraining log: {e}")

    def _read_log(self) -> pd.DataFrame:
        if not self.log_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.log_path, encoding="utf-8")
            return df
        except Exception as e:
            self.logger.warning(f"Failed to read retraining log: {e}")
            return pd.DataFrame()

    # === UZASADNIENIE ===
    def _build_reasoning(
        self,
        *,
        should_retrain: bool,
        score: float,
        priority: str,
        parts: Dict[str, float],
        triggers: List[str],
        cooldown_ok: bool,
        weekly_ok: bool,
        vol_ok: bool,
        force: bool
    ) -> str:
        flags = []
        if not cooldown_ok: flags.append("cooldown_not_elapsed")
        if not weekly_ok: flags.append("weekly_limit_reached")
        if not vol_ok: flags.append("insufficient_new_samples")
        if force: flags.append("forced")

        parts_str = ", ".join([f"{k}={v:.2f}" for k, v in parts.items()])
        trig_str = ", ".join(triggers) if triggers else "none"
        gate_str = "OK" if (cooldown_ok and weekly_ok and vol_ok) else ("BLOCKED: " + ", ".join(flags))

        return (
            f"score={score:.3f} (parts: {parts_str}), priority={priority}; "
            f"triggers=[{trig_str}]; gates={gate_str}."
        )
