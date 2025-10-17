# agents/monitoring/alert_system.py
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  DataGenius PRO Master Enterprise ++++ — Alert System v6.0                ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  🚀 ENTERPRISE-GRADE ML MONITORING & ALERTING SYSTEM                      ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ✓ Multi-Channel Notifications (Email, Slack, SMS, Webhook)              ║
║  ✓ Intelligent Alert Deduplication                                       ║
║  ✓ Rule-Based & Statistical Detection                                    ║
║  ✓ Severity Classification (Info, Warning, Critical)                     ║
║  ✓ Alert Lifecycle Management                                            ║
║  ✓ Historical Tracking & Analytics                                       ║
║  ✓ Configurable Cooldown Periods                                         ║
║  ✓ Consecutive Violation Detection                                       ║
║  ✓ Rich Alert Context & Recommendations                                  ║
║  ✓ Export & Persistence Layer                                            ║
║  ✓ Thread-Safe Operations                                                ║
║  ✓ Async Notification Support                                            ║
║  ✓ Rate Limiting & Throttling                                            ║
║  ✓ Alert Aggregation & Batching                                          ║
╚════════════════════════════════════════════════════════════════════════════╝

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                   AlertManager Core                         │
    ├─────────────────────────────────────────────────────────────┤
    │  1. Alert Detection & Rule Evaluation                       │
    │  2. Deduplication & Rate Limiting                           │
    │  3. Severity Classification                                 │
    │  4. Multi-Channel Notification Routing                      │
    │  5. Alert Lifecycle Management                              │
    │  6. Historical Tracking & Analytics                         │
    │  7. Export & Persistence                                    │
    └─────────────────────────────────────────────────────────────┘

Alert Types:
    • DATA_DRIFT          → Distribution shift detection
    • CONCEPT_DRIFT       → Model assumption violations
    • PERFORMANCE_DEGRADATION → Metric decline
    • DATA_QUALITY        → Data integrity issues
    • ANOMALY            → Outlier detection
    • SYSTEM             → Infrastructure alerts
    • THRESHOLD          → Custom metric thresholds
    • CUSTOM             → User-defined alerts

Notification Channels:
    • EMAIL    → SMTP-based email delivery
    • SLACK    → Webhook integration
    • SMS      → Twilio integration
    • WEBHOOK  → Custom HTTP endpoints
    • LOG      → Structured logging
    • DATABASE → Persistence layer

Usage:
```python
    # Basic usage
    from agents.monitoring.alert_system import AlertManager, create_alert_manager
    
    # Create with default config
    manager = create_alert_manager()
    
    # Create custom alert
    alert = manager.create_data_drift_alert(
        feature_name='age',
        drift_score=0.35,
        threshold=0.25,
        model_name='customer_churn'
    )
    
    # Add custom rule
    from agents.monitoring.alert_system import AlertRule, AlertType, Severity
    
    rule = AlertRule(
        rule_id='custom_accuracy',
        rule_name='Low Accuracy',
        alert_type=AlertType.PERFORMANCE_DEGRADATION,
        severity=Severity.CRITICAL,
        metric_name='accuracy',
        operator='<',
        threshold=0.85
    )
    manager.add_rule(rule)
    
    # Evaluate metrics
    alerts = manager.evaluate_metric('accuracy', 0.80)
    
    # Get active alerts
    active = manager.get_active_alerts(severity=Severity.CRITICAL)
```
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import smtplib
import sys
import time
import warnings
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

import numpy as np
import pandas as pd
import requests

# ═══════════════════════════════════════════════════════════════════════════
# Logging Configuration
# ═══════════════════════════════════════════════════════════════════════════

try:
    from loguru import logger
    
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/alert_system_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        level="DEBUG"
    )
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Module Metadata
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "AlertType",
    "Severity",
    "NotificationChannel",
    "AlertStatus",
    "Alert",
    "AlertRule",
    "NotificationConfig",
    "AlertManager",
    "create_alert_manager",
    "create_default_rules"
]
__version__ = "6.0.0-enterprise"
__author__ = "DataGenius Enterprise Team"
__license__ = "Proprietary"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Enumerations
# ═══════════════════════════════════════════════════════════════════════════

class AlertType(str, Enum):
    """
    🎯 **Alert Type Classification**
    
    Defines categories of alerts for ML monitoring.
    """
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_QUALITY = "data_quality"
    ANOMALY = "anomaly"
    SYSTEM = "system"
    THRESHOLD = "threshold"
    CUSTOM = "custom"
    PREDICTION_ERROR = "prediction_error"
    TRAINING_FAILURE = "training_failure"


class Severity(str, Enum):
    """
    🚨 **Alert Severity Levels**
    
    Hierarchical severity classification.
    """
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    
    def __lt__(self, other):
        """Enable severity comparison."""
        order = {"info": 0, "warning": 1, "critical": 2, "emergency": 3}
        return order[self.value] < order[other.value]


class NotificationChannel(str, Enum):
    """
    📢 **Notification Delivery Channels**
    
    Available channels for alert delivery.
    """
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG = "log"
    DATABASE = "database"
    SMS = "sms"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"
    DISCORD = "discord"


class AlertStatus(str, Enum):
    """
    📊 **Alert Status Lifecycle**
    
    Tracks alert state transitions.
    """
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    EXPIRED = "expired"
    ESCALATED = "escalated"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Alert Data Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Alert:
    """
    🎯 **Individual Alert Instance**
    
    Represents a single alert with complete context and metadata.
    
    Attributes:
        alert_id: Unique identifier
        alert_type: Type of alert
        severity: Severity level
        title: Human-readable title
        message: Detailed description
        
        # Metrics
        metric_name: Associated metric name
        metric_value: Current metric value
        threshold: Threshold value (if applicable)
        
        # Context
        model_name: Associated model
        dataset_name: Associated dataset
        feature_names: Affected features
        
        # Additional data
        details: Additional contextual information
        recommendations: Actionable recommendations
        tags: Custom tags for categorization
        
        # Status tracking
        status: Current status
        created_at: Creation timestamp
        acknowledged_at: Acknowledgment timestamp
        resolved_at: Resolution timestamp
        
        # Notification tracking
        notified_channels: Successfully notified channels
        notification_failures: Failed notification attempts
        
        # Lifecycle
        ttl_hours: Time-to-live in hours
        expires_at: Expiration timestamp
    """
    
    alert_id: str
    alert_type: AlertType
    severity: Severity
    title: str
    message: str
    
    # Metrics
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    
    # Context
    model_name: Optional[str] = None
    dataset_name: Optional[str] = None
    feature_names: Optional[List[str]] = None
    
    # Additional data
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    
    # Status tracking
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    
    # Notification tracking
    notified_channels: List[NotificationChannel] = field(default_factory=list)
    notification_failures: Dict[NotificationChannel, str] = field(default_factory=dict)
    
    # Lifecycle
    ttl_hours: int = 72
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.expires_at is None and self.ttl_hours > 0:
            self.expires_at = self.created_at + timedelta(hours=self.ttl_hours)
        
        # Convert tags to set if list provided
        if isinstance(self.tags, list):
            self.tags = set(self.tags)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert alert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "feature_names": self.feature_names,
            "details": self.details,
            "recommendations": self.recommendations,
            "tags": list(self.tags),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_by": self.acknowledged_by,
            "resolved_by": self.resolved_by,
            "notified_channels": [c.value for c in self.notified_channels],
            "notification_failures": {k.value: v for k, v in self.notification_failures.items()},
            "ttl_hours": self.ttl_hours,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }
    
    def get_hash(self) -> str:
        """
        Generate unique hash for deduplication.
        
        Returns:
            MD5 hash string
        """
        key_components = [
            self.alert_type.value,
            str(self.metric_name),
            str(self.model_name),
            str(self.dataset_name),
            str(sorted(self.feature_names) if self.feature_names else [])
        ]
        key = "_".join(key_components)
        return hashlib.md5(key.encode()).hexdigest()
    
    def is_expired(self) -> bool:
        """Check if alert has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def add_tag(self, tag: str) -> None:
        """Add tag to alert."""
        self.tags.add(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove tag from alert."""
        self.tags.discard(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if alert has specific tag."""
        return tag in self.tags


@dataclass
class AlertRule:
    """
    📋 **Alert Rule Definition**
    
    Defines conditions for automatic alert triggering.
    
    Attributes:
        rule_id: Unique identifier
        rule_name: Human-readable name
        alert_type: Type of alert to create
        severity: Severity level for alerts
        
        # Condition
        metric_name: Metric to monitor
        operator: Comparison operator (>, <, >=, <=, ==, !=)
        threshold: Threshold value
        
        # Advanced conditions
        consecutive_violations: Required consecutive violations
        time_window: Time window for violations (minutes)
        min_samples: Minimum samples required
        
        # Notification settings
        enabled: Whether rule is active
        channels: Notification channels to use
        cooldown_minutes: Minimum time between alerts
        max_alerts_per_day: Maximum alerts per 24h period
        
        # Metadata
        description: Rule description
        created_at: Creation timestamp
        created_by: Creator identifier
        last_triggered: Last trigger timestamp
        trigger_count: Total trigger count
        
        # Suppression
        suppressed_until: Suppression end time
        suppression_reason: Reason for suppression
    """
    
    rule_id: str
    rule_name: str
    alert_type: AlertType
    severity: Severity
    
    # Condition
    metric_name: str
    operator: str  # >, <, >=, <=, ==, !=
    threshold: float
    
    # Advanced conditions
    consecutive_violations: int = 1
    time_window: Optional[int] = None  # Minutes
    min_samples: int = 1
    
    # Notification settings
    enabled: bool = True
    channels: List[NotificationChannel] = field(default_factory=list)
    cooldown_minutes: int = 60
    max_alerts_per_day: int = 100
    
    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    
    # Suppression
    suppressed_until: Optional[datetime] = None
    suppression_reason: Optional[str] = None
    
    # Daily alert tracking
    _daily_alert_count: int = field(default=0, repr=False)
    _last_reset_date: Optional[datetime] = field(default=None, repr=False)
    
    def evaluate(self, value: float) -> bool:
        """
        Evaluate if rule condition is met.
        
        Args:
            value: Current metric value
        
        Returns:
            True if rule triggered, False otherwise
        """
        if not self.enabled:
            return False
        
        if self.is_suppressed():
            return False
        
        operators = {
            '>': lambda x, y: x > y,
            '<': lambda x, y: x < y,
            '>=': lambda x, y: x >= y,
            '<=': lambda x, y: x <= y,
            '==': lambda x, y: abs(x - y) < 1e-10,
            '!=': lambda x, y: abs(x - y) >= 1e-10
        }
        
        op_func = operators.get(self.operator)
        if not op_func:
            logger.warning(f"Unknown operator: {self.operator}")
            return False
        
        return op_func(value, self.threshold)
    
    def is_in_cooldown(self) -> bool:
        """
        Check if rule is in cooldown period.
        
        Returns:
            True if in cooldown, False otherwise
        """
        if self.last_triggered is None:
            return False
        
        cooldown_delta = timedelta(minutes=self.cooldown_minutes)
        return datetime.now(timezone.utc) - self.last_triggered < cooldown_delta
    
    def is_suppressed(self) -> bool:
        """
        Check if rule is currently suppressed.
        
        Returns:
            True if suppressed, False otherwise
        """
        if self.suppressed_until is None:
            return False
        
        return datetime.now(timezone.utc) < self.suppressed_until
    
    def can_trigger(self) -> bool:
        """
        Check if rule can trigger (not in cooldown, under daily limit).
        
        Returns:
            True if can trigger, False otherwise
        """
        if not self.enabled or self.is_suppressed() or self.is_in_cooldown():
            return False
        
        # Reset daily counter if needed
        self._reset_daily_count_if_needed()
        
        return self._daily_alert_count < self.max_alerts_per_day
    
    def record_trigger(self) -> None:
        """Record a rule trigger."""
        self.last_triggered = datetime.now(timezone.utc)
        self.trigger_count += 1
        
        # Update daily count
        self._reset_daily_count_if_needed()
        self._daily_alert_count += 1
    
    def suppress(self, duration_hours: int, reason: str = "") -> None:
        """
        Suppress rule for specified duration.
        
        Args:
            duration_hours: Suppression duration in hours
            reason: Reason for suppression
        """
        self.suppressed_until = datetime.now(timezone.utc) + timedelta(hours=duration_hours)
        self.suppression_reason = reason
        logger.info(f"Rule '{self.rule_name}' suppressed until {self.suppressed_until}")
    
    def unsuppress(self) -> None:
        """Remove suppression from rule."""
        self.suppressed_until = None
        self.suppression_reason = None
        logger.info(f"Rule '{self.rule_name}' unsuppressed")
    
    def _reset_daily_count_if_needed(self) -> None:
        """Reset daily alert counter if date has changed."""
        now = datetime.now(timezone.utc)
        
        if self._last_reset_date is None or self._last_reset_date.date() < now.date():
            self._daily_alert_count = 0
            self._last_reset_date = now
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert rule to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "operator": self.operator,
            "threshold": self.threshold,
            "consecutive_violations": self.consecutive_violations,
            "time_window": self.time_window,
            "min_samples": self.min_samples,
            "enabled": self.enabled,
            "channels": [c.value for c in self.channels],
            "cooldown_minutes": self.cooldown_minutes,
            "max_alerts_per_day": self.max_alerts_per_day,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
            "trigger_count": self.trigger_count,
            "suppressed_until": self.suppressed_until.isoformat() if self.suppressed_until else None,
            "suppression_reason": self.suppression_reason
        }


@dataclass
class NotificationConfig:
    """
    ⚙️ **Notification Channel Configuration**
    
    Central configuration for all notification channels.
    
    Attributes:
        # Email (SMTP)
        email_enabled: Enable email notifications
        smtp_host: SMTP server host
        smtp_port: SMTP server port
        smtp_user: SMTP username
        smtp_password: SMTP password
        smtp_use_tls: Use TLS encryption
        email_from: Sender email address
        email_to: Recipient email addresses
        
        # Slack
        slack_enabled: Enable Slack notifications
        slack_webhook_url: Slack webhook URL
        slack_channel: Default Slack channel
        slack_username: Bot username
        slack_icon_emoji: Bot icon emoji
        
        # Webhook
        webhook_enabled: Enable webhook notifications
        webhook_url: Webhook endpoint URL
        webhook_headers: Custom HTTP headers
        webhook_timeout: Request timeout in seconds
        webhook_retry_count: Number of retry attempts
        
        # SMS (Twilio)
        sms_enabled: Enable SMS notifications
        twilio_account_sid: Twilio account SID
        twilio_auth_token: Twilio auth token
        twilio_from_number: Sender phone number
        sms_to_numbers: Recipient phone numbers
        
        # Microsoft Teams
        teams_enabled: Enable Teams notifications
        teams_webhook_url: Teams webhook URL
        
        # PagerDuty
        pagerduty_enabled: Enable PagerDuty integration
        pagerduty_api_key: PagerDuty API key
        pagerduty_service_key: Service integration key
        
        # Discord
        discord_enabled: Enable Discord notifications
        discord_webhook_url: Discord webhook URL
    """
    
    # Email (SMTP)
    email_enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)
    
    # Slack
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    slack_channel: str = "#alerts"
    slack_username: str = "DataGenius Alert Bot"
    slack_icon_emoji: str = ":warning:"
    
    # Webhook
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    webhook_timeout: int = 10
    webhook_retry_count: int = 3
    
    # SMS (Twilio)
    sms_enabled: bool = False
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    sms_to_numbers: List[str] = field(default_factory=list)
    
    # Microsoft Teams
    teams_enabled: bool = False
    teams_webhook_url: str = ""
    
    # PagerDuty
    pagerduty_enabled: bool = False
    pagerduty_api_key: str = ""
    pagerduty_service_key: str = ""
    
    # Discord
    discord_enabled: bool = False
    discord_webhook_url: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (without sensitive data)."""
        return {
            "email_enabled": self.email_enabled,
            "email_to": self.email_to,
            "slack_enabled": self.slack_enabled,
            "slack_channel": self.slack_channel,
            "webhook_enabled": self.webhook_enabled,
            "sms_enabled": self.sms_enabled,
            "teams_enabled": self.teams_enabled,
            "pagerduty_enabled": self.pagerduty_enabled,
            "discord_enabled": self.discord_enabled
        }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Alert Manager (Main System)
# ═══════════════════════════════════════════════════════════════════════════

class AlertManager:
    """
    🚀 **Alert Manager PRO Master Enterprise ++++**
    
    Central alert management system with enterprise features.
    
    Responsibilities:
      1. Alert detection & creation
      2. Rule evaluation & triggering
      3. Deduplication & rate limiting
      4. Multi-channel notification routing
      5. Alert lifecycle management
      6. Historical tracking & analytics
      7. Export & persistence
      8. Thread-safe operations
    
    Features:
      ✓ Multiple alert types
      ✓ Rule-based detection
      ✓ Intelligent deduplication
      ✓ Multi-channel notifications
      ✓ Alert lifecycle management
      ✓ Cooldown periods
      ✓ Consecutive violation tracking
      ✓ Daily alert limits
      ✓ Alert suppression
      ✓ Historical analytics
      ✓ Export capabilities
      ✓ Thread-safe
    
    Usage:
```python
        # Basic usage
        manager = AlertManager()
        
        # Create alert
        alert = manager.create_data_drift_alert(
            feature_name='age',
            drift_score=0.35,
            threshold=0.25
        )
        
        # Add rule
        rule = AlertRule(...)
        manager.add_rule(rule)
        
        # Evaluate metric
        alerts = manager.evaluate_metric('accuracy', 0.80)
        
        # Get active alerts
        active = manager.get_active_alerts()
```
    """
    
    version: str = __version__
    
    def __init__(
        self,
        notification_config: Optional[NotificationConfig] = None,
        max_history: int = 1000,
        enable_async: bool = False
    ):
        """
        Initialize alert manager.
        
        Args:
            notification_config: Notification configuration
            max_history: Maximum alerts to keep in history
            enable_async: Enable async notification delivery
        """
        self.notification_config = notification_config or NotificationConfig()
        self.max_history = max_history
        self.enable_async = enable_async
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: Deque[Alert] = deque(maxlen=max_history)
        
        # Rules
        self.rules: Dict[str, AlertRule] = {}
        
        # Violation tracking for consecutive violations
        self.violation_tracker: Dict[str, Deque[datetime]] = defaultdict(
            lambda: deque(maxlen=100)
        )
        
        # Deduplication cache
        self.dedup_cache: Dict[str, datetime] = {}
        self.dedup_window_minutes = 60
        
        # Thread safety
        self._lock = RLock()
        self._notification_lock = Lock()
        
        # Metrics
        self._metrics = {
            "total_alerts_created": 0,
            "total_alerts_suppressed": 0,
            "total_notifications_sent": 0,
            "total_notification_failures": 0,
            "alerts_by_type": defaultdict(int),
            "alerts_by_severity": defaultdict(int)
        }
        
        logger.info(f"✓ AlertManager v{self.version} initialized")
    
    # ───────────────────────────────────────────────────────────────────
    # Rule Management
    # ───────────────────────────────────────────────────────────────────
    
    def add_rule(self, rule: AlertRule) -> None:
        """
        Add alert rule.
        
        Args:
            rule: AlertRule to add
        """
        with self._lock:
            self.rules[rule.rule_id] = rule
            logger.info(f"✓ Added rule: {rule.rule_name} (ID: {rule.rule_id})")
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove alert rule.
        
        Args:
            rule_id: Rule ID to remove
        
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if rule_id in self.rules:
                rule_name = self.rules[rule_id].rule_name
                del self.rules[rule_id]
                logger.info(f"✓ Removed rule: {rule_name} (ID: {rule_id})")
                return True
            return False
    
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get rule by ID."""
        return self.rules.get(rule_id)
    
    def list_rules(
        self,
        enabled_only: bool = False,
        alert_type: Optional[AlertType] = None
    ) -> List[AlertRule]:
        """
        List all rules with optional filtering.
        
        Args:
            enabled_only: Return only enabled rules
            alert_type: Filter by alert type
        
        Returns:
            List of rules
        """
        rules = list(self.rules.values())
        
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        
        if alert_type:
            rules = [r for r in rules if r.alert_type == alert_type]
        
        return rules
    
    # ───────────────────────────────────────────────────────────────────
    # Metric Evaluation
    # ───────────────────────────────────────────────────────────────────
    
    def evaluate_metric(
        self,
        metric_name: str,
        metric_value: float,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Alert]:
        """
        Evaluate metric against all applicable rules.
        
        Args:
            metric_name: Name of the metric
            metric_value: Current value
            context: Additional context (model_name, etc.)
        
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        context = context or {}
        
        with self._lock:
            # Find applicable rules
            applicable_rules = [
                rule for rule in self.rules.values()
                if rule.metric_name == metric_name and rule.enabled
            ]
            
            for rule in applicable_rules:
                # Check if rule can trigger
                if not rule.can_trigger():
                    continue
                
                # Evaluate rule
                if rule.evaluate(metric_value):
                    # Track violation
                    violation_key = f"{rule.rule_id}_{metric_name}"
                    self.violation_tracker[violation_key].append(datetime.now(timezone.utc))
                    
                    # Check consecutive violations
                    time_window = rule.time_window or 60
                    recent_violations = [
                        dt for dt in self.violation_tracker[violation_key]
                        if datetime.now(timezone.utc) - dt < timedelta(minutes=time_window)
                    ]
                    
                    if len(recent_violations) >= rule.consecutive_violations:
                        # Create alert
                        alert = self._create_alert_from_rule(
                            rule, metric_value, context
                        )
                        
                        # Check deduplication
                        if not self._is_duplicate(alert):
                            triggered_alerts.append(alert)
                            self._register_alert(alert)
                            
                            # Update rule
                            rule.record_trigger()
                            
                            # Send notifications
                            self._send_notifications(alert, rule.channels)
                            
                            logger.warning(
                                f"⚠ Alert triggered: {alert.title} | "
                                f"metric={metric_name}={metric_value:.4f} | "
                                f"threshold={rule.threshold}"
                            )
                        else:
                            self._metrics["total_alerts_suppressed"] += 1
                            logger.debug(f"Alert suppressed (duplicate): {rule.rule_name}")
                else:
                    # Clear violations if value is back to normal
                    violation_key = f"{rule.rule_id}_{metric_name}"
                    if violation_key in self.violation_tracker:
                        self.violation_tracker[violation_key].clear()
        
        return triggered_alerts
    
    # ───────────────────────────────────────────────────────────────────
    # Alert Creation (Main Methods)
    # ───────────────────────────────────────────────────────────────────
    
    def create_alert(
        self,
        alert_type: AlertType,
        severity: Severity,
        title: str,
        message: str,
        channels: Optional[List[NotificationChannel]] = None,
        **kwargs
    ) -> Alert:
        """
        Create and register a custom alert.
        
        Args:
            alert_type: Type of alert
            severity: Severity level
            title: Alert title
            message: Alert message
            channels: Notification channels
            **kwargs: Additional alert parameters
        
        Returns:
            Created Alert object
        """
        alert_id = self._generate_alert_id()
        
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            **kwargs
        )
        
        with self._lock:
            # Check deduplication
            if self._is_duplicate(alert):
                self._metrics["total_alerts_suppressed"] += 1
                logger.info(f"Duplicate alert suppressed: {title}")
                return alert
            
            # Register alert
            self._register_alert(alert)
            
            # Update metrics
            self._metrics["total_alerts_created"] += 1
            self._metrics["alerts_by_type"][alert_type.value] += 1
            self._metrics["alerts_by_severity"][severity.value] += 1
        
        # Send notifications
        if channels is None:
            channels = [NotificationChannel.LOG]
        
        self._send_notifications(alert, channels)
        
        logger.info(
            f"✓ Alert created: {title} | "
            f"severity={severity.value} | "
            f"type={alert_type.value} | "
            f"id={alert_id}"
        )
        
        return alert
    
    def create_data_drift_alert(
        self,
        feature_name: str,
        drift_score: float,
        threshold: float,
        model_name: Optional[str] = None,
        drift_method: str = "PSI",
        channels: Optional[List[NotificationChannel]] = None
    ) -> Alert:
        """
        Create data drift alert.
        
        Args:
            feature_name: Name of drifted feature
            drift_score: Drift score value
            threshold: Threshold value
            model_name: Associated model name
            drift_method: Drift detection method
            channels: Notification channels
        
        Returns:
            Created Alert object
        """
        # Determine severity based on drift magnitude
        if drift_score > threshold * 2:
            severity = Severity.CRITICAL
        elif drift_score > threshold * 1.5:
            severity = Severity.WARNING
        else:
            severity = Severity.INFO
        
        return self.create_alert(
            alert_type=AlertType.DATA_DRIFT,
            severity=severity,
            title=f"Data Drift Detected: {feature_name}",
            message=(
                f"Feature '{feature_name}' shows significant drift "
                f"({drift_method} score: {drift_score:.4f}, threshold: {threshold:.4f})"
            ),
            metric_name=f"drift_score_{drift_method.lower()}",
            metric_value=drift_score,
            threshold=threshold,
            model_name=model_name,
            feature_names=[feature_name],
            channels=channels,
            recommendations=[
                "Review recent data changes and collection pipeline",
                "Check for upstream data source modifications",
                "Consider model retraining with recent data",
                "Investigate distribution shift causes",
                "Validate feature engineering pipeline"
            ],
            details={
                "drift_method": drift_method,
                "drift_magnitude": drift_score / threshold if threshold > 0 else float('inf'),
                "feature_name": feature_name
            }
        )
    
    def create_performance_alert(
        self,
        metric_name: str,
        current_value: float,
        baseline_value: float,
        degradation_pct: float,
        model_name: Optional[str] = None,
        channels: Optional[List[NotificationChannel]] = None
    ) -> Alert:
        """
        Create performance degradation alert.
        
        Args:
            metric_name: Name of performance metric
            current_value: Current metric value
            baseline_value: Baseline/expected value
            degradation_pct: Degradation percentage
            model_name: Associated model name
            channels: Notification channels
        
        Returns:
            Created Alert object
        """
        # Determine severity
        if degradation_pct > 25:
            severity = Severity.CRITICAL
        elif degradation_pct > 15:
            severity = Severity.WARNING
        else:
            severity = Severity.INFO
        
        return self.create_alert(
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=severity,
            title=f"Performance Degradation: {metric_name}",
            message=(
                f"Model performance declined by {degradation_pct:.1f}% "
                f"({metric_name}: {current_value:.4f} vs baseline: {baseline_value:.4f})"
            ),
            metric_name=metric_name,
            metric_value=current_value,
            threshold=baseline_value,
            model_name=model_name,
            channels=channels,
            recommendations=[
                "Retrain model with recent data",
                "Check for data drift in input features",
                "Review feature engineering changes",
                "Validate data quality and preprocessing",
                "Analyze prediction error patterns"
            ],
            details={
                "current_value": current_value,
                "baseline_value": baseline_value,
                "degradation_pct": degradation_pct,
                "absolute_change": baseline_value - current_value
            }
        )
    
    def create_data_quality_alert(
        self,
        issue_type: str,
        affected_features: List[str],
        severity: Severity = Severity.WARNING,
        model_name: Optional[str] = None,
        issue_details: Optional[Dict[str, Any]] = None,
        channels: Optional[List[NotificationChannel]] = None
    ) -> Alert:
        """
        Create data quality alert.
        
        Args:
            issue_type: Type of quality issue
            affected_features: List of affected features
            severity: Alert severity
            model_name: Associated model name
            issue_details: Additional issue details
            channels: Notification channels
        
        Returns:
            Created Alert object
        """
        feature_list = ', '.join(affected_features[:3])
        if len(affected_features) > 3:
            feature_list += f" (+{len(affected_features) - 3} more)"
        
        return self.create_alert(
            alert_type=AlertType.DATA_QUALITY,
            severity=severity,
            title=f"Data Quality Issue: {issue_type}",
            message=(
                f"Detected {issue_type} in {len(affected_features)} features: {feature_list}"
            ),
            model_name=model_name,
            feature_names=affected_features,
            channels=channels,
            recommendations=[
                "Review data collection and ingestion process",
                "Check data validation rules and constraints",
                "Investigate data source quality",
                "Update preprocessing pipeline",
                "Implement additional data quality checks"
            ],
            details={
                "issue_type": issue_type,
                "n_affected_features": len(affected_features),
                "affected_features": affected_features,
                **(issue_details or {})
            }
        )
    
    def create_anomaly_alert(
        self,
        anomaly_score: float,
        threshold: float,
        n_anomalies: int,
        total_samples: int,
        model_name: Optional[str] = None,
        anomaly_method: str = "IsolationForest",
        channels: Optional[List[NotificationChannel]] = None
    ) -> Alert:
        """
        Create anomaly detection alert.
        
        Args:
            anomaly_score: Anomaly score
            threshold: Threshold value
            n_anomalies: Number of detected anomalies
            total_samples: Total number of samples
            model_name: Associated model name
            anomaly_method: Anomaly detection method
            channels: Notification channels
        
        Returns:
            Created Alert object
        """
        anomaly_rate = (n_anomalies / total_samples * 100) if total_samples > 0 else 0
        
        # Determine severity
        if anomaly_rate > 10:
            severity = Severity.CRITICAL
        elif anomaly_rate > 5:
            severity = Severity.WARNING
        else:
            severity = Severity.INFO
        
        return self.create_alert(
            alert_type=AlertType.ANOMALY,
            severity=severity,
            title=f"Anomalies Detected ({anomaly_method})",
            message=(
                f"Detected {n_anomalies:,} anomalies ({anomaly_rate:.2f}%) "
                f"in {total_samples:,} samples "
                f"(score: {anomaly_score:.4f}, threshold: {threshold:.4f})"
            ),
            metric_name="anomaly_score",
            metric_value=anomaly_score,
            threshold=threshold,
            model_name=model_name,
            channels=channels,
            recommendations=[
                "Investigate anomalous samples for patterns",
                "Check for data corruption or collection issues",
                "Review outlier handling strategy",
                "Validate model assumptions and constraints",
                "Consider adjusting anomaly detection threshold"
            ],
            details={
                "n_anomalies": n_anomalies,
                "total_samples": total_samples,
                "anomaly_rate_pct": anomaly_rate,
                "anomaly_method": anomaly_method
            }
        )
    
    def create_system_alert(
        self,
        system_component: str,
        issue_description: str,
        severity: Severity = Severity.WARNING,
        error_details: Optional[str] = None,
        channels: Optional[List[NotificationChannel]] = None
    ) -> Alert:
        """
        Create system/infrastructure alert.
        
        Args:
            system_component: Affected system component
            issue_description: Description of the issue
            severity: Alert severity
            error_details: Additional error details
            channels: Notification channels
        
        Returns:
            Created Alert object
        """
        return self.create_alert(
            alert_type=AlertType.SYSTEM,
            severity=severity,
            title=f"System Issue: {system_component}",
            message=issue_description,
            channels=channels,
            recommendations=[
                "Check system logs for additional details",
                "Verify system resource availability",
                "Review recent configuration changes",
                "Contact system administrator if issue persists"
            ],
            details={
                "component": system_component,
                "error_details": error_details
            }
        )
    
    # ───────────────────────────────────────────────────────────────────
    # Alert Lifecycle Management
    # ───────────────────────────────────────────────────────────────────
    
    def acknowledge_alert(
        self,
        alert_id: str,
        user: Optional[str] = None
    ) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID to acknowledge
            user: User acknowledging the alert
        
        Returns:
            True if acknowledged, False if not found
        """
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now(timezone.utc)
                alert.acknowledged_by = user
                
                logger.info(
                    f"✓ Alert acknowledged: {alert_id}" +
                    (f" by {user}" if user else "")
                )
                return True
        
        return False
    
    def resolve_alert(
        self,
        alert_id: str,
        user: Optional[str] = None,
        resolution_note: Optional[str] = None
    ) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert ID to resolve
            user: User resolving the alert
            resolution_note: Optional resolution note
        
        Returns:
            True if resolved, False if not found
        """
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now(timezone.utc)
                alert.resolved_by = user
                
                if resolution_note:
                    alert.details["resolution_note"] = resolution_note
                
                # Move to history
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]
                
                logger.info(
                    f"✓ Alert resolved: {alert_id}" +
                    (f" by {user}" if user else "")
                )
                return True
        
        return False
    
    def escalate_alert(
        self,
        alert_id: str,
        new_severity: Severity,
        reason: str,
        additional_channels: Optional[List[NotificationChannel]] = None
    ) -> bool:
        """
        Escalate alert to higher severity.
        
        Args:
            alert_id: Alert ID to escalate
            new_severity: New severity level
            reason: Escalation reason
            additional_channels: Additional notification channels
        
        Returns:
            True if escalated, False if not found
        """
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                
                old_severity = alert.severity
                alert.severity = new_severity
                alert.status = AlertStatus.ESCALATED
                alert.details["escalation_reason"] = reason
                alert.details["original_severity"] = old_severity.value
                alert.details["escalated_at"] = datetime.now(timezone.utc).isoformat()
                
                logger.warning(
                    f"⚠ Alert escalated: {alert_id} | "
                    f"{old_severity.value} → {new_severity.value} | "
                    f"reason: {reason}"
                )
                
                # Send additional notifications
                if additional_channels:
                    self._send_notifications(alert, additional_channels)
                
                return True
        
        return False
    
    def suppress_alert(
        self,
        alert_id: str,
        reason: str
    ) -> bool:
        """
        Suppress an alert.
        
        Args:
            alert_id: Alert ID to suppress
            reason: Suppression reason
        
        Returns:
            True if suppressed, False if not found
        """
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.SUPPRESSED
                alert.details["suppression_reason"] = reason
                alert.details["suppressed_at"] = datetime.now(timezone.utc).isoformat()
                
                logger.info(f"Alert suppressed: {alert_id} | reason: {reason}")
                return True
        
        return False
    
    # ───────────────────────────────────────────────────────────────────
    # Alert Retrieval & Querying
    # ───────────────────────────────────────────────────────────────────
    
    def get_active_alerts(
        self,
        alert_type: Optional[AlertType] = None,
        severity: Optional[Severity] = None,
        model_name: Optional[str] = None,
        tags: Optional[Set[str]] = None
    ) -> List[Alert]:
        """
        Get active alerts with optional filtering.
        
        Args:
            alert_type: Filter by alert type
            severity: Filter by severity
            model_name: Filter by model name
            tags: Filter by tags (alerts must have all tags)
        
        Returns:
            List of filtered alerts
        """
        with self._lock:
            alerts = list(self.active_alerts.values())
        
        # Apply filters
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if model_name:
            alerts = [a for a in alerts if a.model_name == model_name]
        
        if tags:
            alerts = [a for a in alerts if tags.issubset(a.tags)]
        
        # Sort by severity and creation time
        severity_order = {
            Severity.EMERGENCY: 0,
            Severity.CRITICAL: 1,
            Severity.WARNING: 2,
            Severity.INFO: 3
        }
        alerts.sort(
            key=lambda x: (severity_order.get(x.severity, 99), -x.created_at.timestamp())
        )
        
        return alerts
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """
        Get specific alert by ID.
        
        Args:
            alert_id: Alert ID
        
        Returns:
            Alert if found, None otherwise
        """
        return self.active_alerts.get(alert_id)
    
    def get_alert_history(
        self,
        limit: int = 100,
        alert_type: Optional[AlertType] = None,
        severity: Optional[Severity] = None,
        since: Optional[datetime] = None
    ) -> List[Alert]:
        """
        Get alert history with optional filtering.
        
        Args:
            limit: Maximum number of alerts to return
            alert_type: Filter by alert type
            severity: Filter by severity
            since: Only return alerts created after this time
        
        Returns:
            List of historical alerts
        """
        with self._lock:
            history = list(self.alert_history)
        
        # Apply filters
        if alert_type:
            history = [a for a in history if a.alert_type == alert_type]
        
        if severity:
            history = [a for a in history if a.severity == severity]
        
        if since:
            history = [a for a in history if a.created_at >= since]
        
        # Sort by creation time (newest first)
        history.sort(key=lambda x: x.created_at, reverse=True)
        
        return history[:limit]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive alert statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            all_alerts = list(self.active_alerts.values()) + list(self.alert_history)
            
            if not all_alerts:
                return {
                    "total_alerts": 0,
                    "active_alerts": 0,
                    "by_type": {},
                    "by_severity": {},
                    "by_status": {},
                    "metrics": self._metrics.copy()
                }
            
            by_type = defaultdict(int)
            by_severity = defaultdict(int)
            by_status = defaultdict(int)
            
            for alert in all_alerts:
                by_type[alert.alert_type.value] += 1
                by_severity[alert.severity.value] += 1
                by_status[alert.status.value] += 1
            
            # Calculate average resolution time
            resolved_alerts = [
                a for a in all_alerts
                if a.resolved_at and a.created_at
            ]
            
            avg_resolution_time = None
            if resolved_alerts:
                resolution_times = [
                    (a.resolved_at - a.created_at).total_seconds() / 3600
                    for a in resolved_alerts
                ]
                avg_resolution_time = sum(resolution_times) / len(resolution_times)
            
            return {
                "total_alerts": len(all_alerts),
                "active_alerts": len(self.active_alerts),
                "resolved_alerts": len([a for a in all_alerts if a.status == AlertStatus.RESOLVED]),
                "by_type": dict(by_type),
                "by_severity": dict(by_severity),
                "by_status": dict(by_status),
                "oldest_active": (
                    min([a.created_at for a in self.active_alerts.values()]).isoformat()
                    if self.active_alerts else None
                ),
                "avg_resolution_time_hours": (
                    round(avg_resolution_time, 2) if avg_resolution_time else None
                ),
                "metrics": self._metrics.copy()
            }
    
    # ───────────────────────────────────────────────────────────────────
    # Notification System
    # ───────────────────────────────────────────────────────────────────
    
    def _send_notifications(
        self,
        alert: Alert,
        channels: List[NotificationChannel]
    ) -> None:
        """
        Send notifications through specified channels.
        
        Args:
            alert: Alert to send
            channels: Notification channels to use
        """
        if self.enable_async:
            # Async notification (non-blocking)
            import threading
            thread = threading.Thread(
                target=self._send_notifications_sync,
                args=(alert, channels),
                daemon=True
            )
            thread.start()
        else:
            # Sync notification
            self._send_notifications_sync(alert, channels)
    
    def _send_notifications_sync(
        self,
        alert: Alert,
        channels: List[NotificationChannel]
    ) -> None:
        """Synchronous notification delivery."""
        for channel in channels:
            try:
                if channel == NotificationChannel.EMAIL and self.notification_config.email_enabled:
                    self._send_email(alert)
                    alert.notified_channels.append(NotificationChannel.EMAIL)
                
                elif channel == NotificationChannel.SLACK and self.notification_config.slack_enabled:
                    self._send_slack(alert)
                    alert.notified_channels.append(NotificationChannel.SLACK)
                
                elif channel == NotificationChannel.WEBHOOK and self.notification_config.webhook_enabled:
                    self._send_webhook(alert)
                    alert.notified_channels.append(NotificationChannel.WEBHOOK)
                
                elif channel == NotificationChannel.LOG:
                    self._log_alert(alert)
                    alert.notified_channels.append(NotificationChannel.LOG)
                
                elif channel == NotificationChannel.SMS and self.notification_config.sms_enabled:
                    self._send_sms(alert)
                    alert.notified_channels.append(NotificationChannel.SMS)
                
                elif channel == NotificationChannel.TEAMS and self.notification_config.teams_enabled:
                    self._send_teams(alert)
                    alert.notified_channels.append(NotificationChannel.TEAMS)
                
                elif channel == NotificationChannel.DISCORD and self.notification_config.discord_enabled:
                    self._send_discord(alert)
                    alert.notified_channels.append(NotificationChannel.DISCORD)
                
                self._metrics["total_notifications_sent"] += 1
            
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                alert.notification_failures[channel] = error_msg
                self._metrics["total_notification_failures"] += 1
                logger.error(f"Failed to send notification via {channel.value}: {error_msg}")
    
    def _send_email(self, alert: Alert) -> None:
        """Send email notification."""
        if not self.notification_config.email_to:
            return
        
        with self._notification_lock:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            msg['From'] = self.notification_config.email_from
            msg['To'] = ", ".join(self.notification_config.email_to)
            
            # Severity colors
            severity_colors = {
                Severity.INFO: "#2196F3",
                Severity.WARNING: "#FF9800",
                Severity.CRITICAL: "#F44336",
                Severity.EMERGENCY: "#9C27B0"
            }
            
            # Create HTML body
            html_body = f"""
            <html>
              <body style="font-family: Arial, sans-serif;">
                <div style="border-left: 4px solid {severity_colors[alert.severity]}; padding-left: 20px;">
                  <h2 style="color: {severity_colors[alert.severity]};">
                    {alert.title}
                  </h2>
                  <p><strong>Severity:</strong> <span style="color: {severity_colors[alert.severity]};">{alert.severity.value.upper()}</span></p>
                  <p><strong>Type:</strong> {alert.alert_type.value}</p>
                  <p><strong>Time:</strong> {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                  <p><strong>Alert ID:</strong> {alert.alert_id}</p>
                  <p><strong>Message:</strong> {alert.message}</p>
                  
                  {f'<p><strong>Model:</strong> {alert.model_name}</p>' if alert.model_name else ''}
                  {f'<p><strong>Metric:</strong> {alert.metric_name} = {alert.metric_value:.4f}</p>' if alert.metric_name else ''}
                  {f'<p><strong>Threshold:</strong> {alert.threshold:.4f}</p>' if alert.threshold else ''}
                  
                  {f'''
                  <h3>Recommendations:</h3>
                  <ul style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
                    {''.join([f'<li>{r}</li>' for r in alert.recommendations])}
                  </ul>
                  ''' if alert.recommendations else ''}
                  
                  <hr>
                  <p style="font-size: 12px; color: #666;">
                    DataGenius PRO Alert System v{self.version}
                  </p>
                </div>
              </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send
            try:
                with smtplib.SMTP(
                    self.notification_config.smtp_host,
                    self.notification_config.smtp_port
                ) as server:
                    if self.notification_config.smtp_use_tls:
                        server.starttls()
                    server.login(
                        self.notification_config.smtp_user,
                        self.notification_config.smtp_password
                    )
                    server.send_message(msg)
                
                logger.debug(f"✓ Email sent for alert: {alert.alert_id}")
            except Exception as e:
                logger.error(f"Failed to send email: {e}")
                raise
    
    def _send_slack(self, alert: Alert) -> None:
        """Send Slack notification."""
        color_map = {
            Severity.INFO: "#36a64f",
            Severity.WARNING: "#ff9800",
            Severity.CRITICAL: "#f44336",
            Severity.EMERGENCY: "#9c27b0"
        }
        
        fields = [
            {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
            {"title": "Type", "value": alert.alert_type.value, "short": True},
        ]
        
        if alert.model_name:
            fields.append({"title": "Model", "value": alert.model_name, "short": True})
        
        if alert.metric_name and alert.metric_value is not None:
            fields.append({
                "title": "Metric",
                "value": f"{alert.metric_name}: {alert.metric_value:.4f}",
                "short": True
            })
        
        payload = {
            "channel": self.notification_config.slack_channel,
            "username": self.notification_config.slack_username,
            "icon_emoji": self.notification_config.slack_icon_emoji,
            "attachments": [{
                "color": color_map[alert.severity],
                "title": alert.title,
                "text": alert.message,
                "fields": fields,
                "footer": f"DataGenius PRO v{self.version}",
                "footer_icon": "https://platform.slack-edge.com/img/default_application_icon.png",
                "ts": int(alert.created_at.timestamp())
            }]
        }
        
        try:
            response = requests.post(
                self.notification_config.slack_webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            logger.debug(f"✓ Slack notification sent for alert: {alert.alert_id}")
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            raise
    
    def _send_webhook(self, alert: Alert) -> None:
        """Send webhook notification."""
        payload = alert.to_dict()
        
        for attempt in range(self.notification_config.webhook_retry_count):
            try:
                response = requests.post(
                    self.notification_config.webhook_url,
                    json=payload,
                    headers=self.notification_config.webhook_headers,
                    timeout=self.notification_config.webhook_timeout
                )
                response.raise_for_status()
                logger.debug(f"✓ Webhook sent for alert: {alert.alert_id}")
                return
            
            except Exception as e:
                if attempt < self.notification_config.webhook_retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to send webhook after {attempt + 1} attempts: {e}")
                    raise
    
    def _send_sms(self, alert: Alert) -> None:
        """Send SMS notification via Twilio."""
        try:
            from twilio.rest import Client
            
            client = Client(
                self.notification_config.twilio_account_sid,
                self.notification_config.twilio_auth_token
            )
            
            message_body = (
                f"[{alert.severity.value.upper()}] {alert.title}: "
                f"{alert.message[:100]}"
            )
            
            for number in self.notification_config.sms_to_numbers:
                client.messages.create(
                    body=message_body[:160],  # SMS character limit
                    from_=self.notification_config.twilio_from_number,
                    to=number
                )
            
            logger.debug(f"✓ SMS sent for alert: {alert.alert_id}")
        
        except ImportError:
            logger.warning("Twilio library not installed, cannot send SMS")
            raise
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            raise
    
    def _send_teams(self, alert: Alert) -> None:
        """Send Microsoft Teams notification."""
        color_map = {
            Severity.INFO: "0078D4",
            Severity.WARNING: "FFB900",
            Severity.CRITICAL: "D13438",
            Severity.EMERGENCY: "8661C5"
        }
        
        facts = [
            {"name": "Severity", "value": alert.severity.value.upper()},
            {"name": "Type", "value": alert.alert_type.value},
            {"name": "Time", "value": alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
        ]
        
        if alert.model_name:
            facts.append({"name": "Model", "value": alert.model_name})
        
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color_map[alert.severity],
            "summary": alert.title,
            "sections": [{
                "activityTitle": alert.title,
                "activitySubtitle": alert.message,
                "facts": facts,
                "markdown": True
            }]
        }
        
        try:
            response = requests.post(
                self.notification_config.teams_webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            logger.debug(f"✓ Teams notification sent for alert: {alert.alert_id}")
        except Exception as e:
            logger.error(f"Failed to send Teams notification: {e}")
            raise
    
    def _send_discord(self, alert: Alert) -> None:
        """Send Discord notification."""
        color_map = {
            Severity.INFO: 0x3498db,
            Severity.WARNING: 0xf39c12,
            Severity.CRITICAL: 0xe74c3c,
            Severity.EMERGENCY: 0x9b59b6
        }
        
        fields = [
            {"name": "Severity", "value": alert.severity.value.upper(), "inline": True},
            {"name": "Type", "value": alert.alert_type.value, "inline": True}
        ]
        
        if alert.model_name:
            fields.append({"name": "Model", "value": alert.model_name, "inline": True})
        
        payload = {
            "embeds": [{
                "title": alert.title,
                "description": alert.message,
                "color": color_map[alert.severity],
                "fields": fields,
                "footer": {
                    "text": f"DataGenius PRO v{self.version}"
                },
                "timestamp": alert.created_at.isoformat()
            }]
        }
        
        try:
            response = requests.post(
                self.notification_config.discord_webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            logger.debug(f"✓ Discord notification sent for alert: {alert.alert_id}")
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            raise
    
    def _log_alert(self, alert: Alert) -> None:
        """Log alert to logger."""
        log_level_map = {
            Severity.INFO: logging.INFO,
            Severity.WARNING: logging.WARNING,
            Severity.CRITICAL: logging.CRITICAL,
            Severity.EMERGENCY: logging.CRITICAL
        }
        
        log_msg = (
            f"ALERT [{alert.severity.value.upper()}] "
            f"{alert.title}: {alert.message}"
        )
        
        if alert.model_name:
            log_msg += f" | model={alert.model_name}"
        
        if alert.metric_name:
            log_msg += f" | {alert.metric_name}={alert.metric_value}"
        
        logger.log(log_level_map[alert.severity], log_msg)
    
    # ───────────────────────────────────────────────────────────────────
    # Helper Methods
    # ───────────────────────────────────────────────────────────────────
    
    def _create_alert_from_rule(
        self,
        rule: AlertRule,
        metric_value: float,
        context: Dict[str, Any]
    ) -> Alert:
        """
        Create alert from triggered rule.
        
        Args:
            rule: Triggered rule
            metric_value: Current metric value
            context: Additional context
        
        Returns:
            Created Alert object
        """
        alert_id = self._generate_alert_id()
        
        message = (
            f"{rule.description or rule.rule_name} triggered: "
            f"{rule.metric_name} {rule.operator} {rule.threshold} "
            f"(current: {metric_value:.4f})"
        )
        
        return Alert(
            alert_id=alert_id,
            alert_type=rule.alert_type,
            severity=rule.severity,
            title=rule.rule_name,
            message=message,
            metric_name=rule.metric_name,
            metric_value=metric_value,
            threshold=rule.threshold,
            model_name=context.get("model_name"),
            dataset_name=context.get("dataset_name"),
            details={
                "rule_id": rule.rule_id,
                "operator": rule.operator,
                **context
            }
        )
    
    def _register_alert(self, alert: Alert) -> None:
        """
        Register alert in active alerts and update dedup cache.
        
        Args:
            alert: Alert to register
        """
        self.active_alerts[alert.alert_id] = alert
        
        # Update deduplication cache
        alert_hash = alert.get_hash()
        self.dedup_cache[alert_hash] = datetime.now(timezone.utc)
    
    def _is_duplicate(self, alert: Alert) -> bool:
        """
        Check if alert is a duplicate within dedup window.
        
        Args:
            alert: Alert to check
        
        Returns:
            True if duplicate, False otherwise
        """
        alert_hash = alert.get_hash()
        
        if alert_hash in self.dedup_cache:
            last_seen = self.dedup_cache[alert_hash]
            time_diff = datetime.now(timezone.utc) - last_seen
            
            if time_diff < timedelta(minutes=self.dedup_window_minutes):
                return True
        
        return False
    
    def _generate_alert_id(self) -> str:
        """
        Generate unique alert ID.
        
        Returns:
            Unique alert ID
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        unique_suffix = str(uuid4())[:8]
        return f"ALERT-{timestamp}-{unique_suffix}"
    
    # ───────────────────────────────────────────────────────────────────
    # Maintenance & Utilities
    # ───────────────────────────────────────────────────────────────────
    
    def cleanup_expired_alerts(self) -> int:
        """
        Remove expired alerts from active alerts.
        
        Returns:
            Number of alerts cleaned up
        """
        with self._lock:
            expired = [
                alert_id for alert_id, alert in self.active_alerts.items()
                if alert.is_expired()
            ]
            
            for alert_id in expired:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.EXPIRED
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]
            
            if expired:
                logger.info(f"✓ Cleaned up {len(expired)} expired alerts")
            
            return len(expired)
    
    def clear_resolved_alerts(self, older_than_days: int = 30) -> int:
        """
        Clear resolved alerts older than specified days from history.
        
        Args:
            older_than_days: Age threshold in days
        
        Returns:
            Number of alerts cleared
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        
        with self._lock:
            old_len = len(self.alert_history)
            
            self.alert_history = deque(
                [a for a in self.alert_history if a.created_at > cutoff_date],
                maxlen=self.max_history
            )
            
            removed = old_len - len(self.alert_history)
            
            if removed > 0:
                logger.info(f"✓ Cleared {removed} old alerts from history")
            
            return removed
    
    def cleanup_dedup_cache(self) -> int:
        """
        Clean up old entries from deduplication cache.
        
        Returns:
            Number of entries removed
        """
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.dedup_window_minutes)
        
        with self._lock:
            old_len = len(self.dedup_cache)
            
            self.dedup_cache = {
                k: v for k, v in self.dedup_cache.items()
                if v > cutoff
            }
            
            removed = old_len - len(self.dedup_cache)
            
            if removed > 0:
                logger.debug(f"✓ Cleaned up {removed} dedup cache entries")
            
            return removed
    
    def export_alerts(
        self,
        filepath: Union[str, Path],
        include_history: bool = True
    ) -> None:
        """
        Export alerts to JSON file.
        
        Args:
            filepath: Output file path
            include_history: Include alert history
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            data = {
                "metadata": {
                    "version": self.version,
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "total_alerts": len(self.active_alerts) + len(self.alert_history)
                },
                "active_alerts": [alert.to_dict() for alert in self.active_alerts.values()],
                "rules": [rule.to_dict() for rule in self.rules.values()],
                "statistics": self.get_alert_statistics()
            }
            
            if include_history:
                data["alert_history"] = [alert.to_dict() for alert in self.alert_history]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Alerts exported to {filepath}")
    
    def import_rules(self, filepath: Union[str, Path]) -> int:
        """
        Import rules from JSON file.
        
        Args:
            filepath: Input file path
        
        Returns:
            Number of rules imported
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"Rule file not found: {filepath}")
            return 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        rules_data = data.get('rules', [])
        imported = 0
        
        for rule_dict in rules_data:
            try:
                # Convert string enums back
                rule_dict['alert_type'] = AlertType(rule_dict['alert_type'])
                rule_dict['severity'] = Severity(rule_dict['severity'])
                rule_dict['channels'] = [
                    NotificationChannel(c) for c in rule_dict.get('channels', [])
                ]
                
                # Convert datetime strings
                for date_field in ['created_at', 'last_triggered', 'suppressed_until']:
                    if rule_dict.get(date_field):
                        rule_dict[date_field] = datetime.fromisoformat(rule_dict[date_field])
                
                rule = AlertRule(**rule_dict)
                self.add_rule(rule)
                imported += 1
            
            except Exception as e:
                logger.warning(f"Failed to import rule {rule_dict.get('rule_id')}: {e}")
        
        logger.info(f"✓ Imported {imported} rules from {filepath}")
        return imported


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Pre-configured Rules
# ═══════════════════════════════════════════════════════════════════════════

def create_default_rules() -> List[AlertRule]:
    """
    Create comprehensive set of default alert rules.
    
    Returns:
        List of AlertRule objects
    """
    rules = [
        # ─── Performance Degradation Rules ───
        AlertRule(
            rule_id="perf_accuracy_drop_warning",
            rule_name="Accuracy Drop >10%",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=Severity.WARNING,
            metric_name="accuracy_drop_pct",
            operator=">",
            threshold=10,
            consecutive_violations=2,
            channels=[NotificationChannel.LOG, NotificationChannel.EMAIL],
            description="Model accuracy dropped by more than 10%"
        ),
        
        AlertRule(
            rule_id="perf_accuracy_drop_critical",
            rule_name="Accuracy Drop >20%",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=Severity.CRITICAL,
            metric_name="accuracy_drop_pct",
            operator=">",
            threshold=20,
            consecutive_violations=1,
            channels=[NotificationChannel.LOG, NotificationChannel.EMAIL, NotificationChannel.SLACK],
            description="Critical accuracy drop detected"
        ),
        
        # ─── Data Drift Rules ───
        AlertRule(
            rule_id="drift_psi_moderate",
            rule_name="Moderate Data Drift (PSI)",
            alert_type=AlertType.DATA_DRIFT,
            severity=Severity.WARNING,
            metric_name="psi_score",
            operator=">",
            threshold=0.2,
            consecutive_violations=1,
            channels=[NotificationChannel.LOG],
            description="PSI score indicates moderate drift"
        ),
        
        AlertRule(
            rule_id="drift_psi_high",
            rule_name="High Data Drift (PSI)",
            alert_type=AlertType.DATA_DRIFT,
            severity=Severity.CRITICAL,
            metric_name="psi_score",
            operator=">",
            threshold=0.25,
            consecutive_violations=1,
            channels=[NotificationChannel.LOG, NotificationChannel.SLACK],
            description="PSI score indicates significant drift"
        ),
        
        AlertRule(
            rule_id="drift_kl_divergence",
            rule_name="High KL Divergence",
            alert_type=AlertType.DATA_DRIFT,
            severity=Severity.WARNING,
            metric_name="kl_divergence",
            operator=">",
            threshold=0.1,
            consecutive_violations=2,
            channels=[NotificationChannel.LOG],
            description="Kullback-Leibler divergence exceeds threshold"
        ),
        
        # ─── Data Quality Rules ───
        AlertRule(
            rule_id="quality_missing_moderate",
            rule_name="Moderate Missing Values",
            alert_type=AlertType.DATA_QUALITY,
            severity=Severity.WARNING,
            metric_name="missing_pct",
            operator=">",
            threshold=20,
            consecutive_violations=1,
            channels=[NotificationChannel.LOG],
            description="More than 20% missing values detected"
        ),
        
        AlertRule(
            rule_id="quality_missing_high",
            rule_name="High Missing Values",
            alert_type=AlertType.DATA_QUALITY,
            severity=Severity.CRITICAL,
            metric_name="missing_pct",
            operator=">",
            threshold=40,
            consecutive_violations=1,
            channels=[NotificationChannel.LOG, NotificationChannel.EMAIL],
            description="More than 40% missing values detected"
        ),
        
        AlertRule(
            rule_id="quality_duplicates",
            rule_name="High Duplicate Rate",
            alert_type=AlertType.DATA_QUALITY,
            severity=Severity.WARNING,
            metric_name="duplicate_pct",
            operator=">",
            threshold=10,
            consecutive_violations=1,
            channels=[NotificationChannel.LOG],
            description="More than 10% duplicate records"
        ),
        
        # ─── Anomaly Detection Rules ───
        AlertRule(
            rule_id="anomaly_rate_moderate",
            rule_name="Moderate Anomaly Rate",
            alert_type=AlertType.ANOMALY,
            severity=Severity.WARNING,
            metric_name="anomaly_rate",
            operator=">",
            threshold=5,
            consecutive_violations=2,
            channels=[NotificationChannel.LOG],
            description="Anomaly rate exceeds 5%"
        ),
        
        AlertRule(
            rule_id="anomaly_rate_high",
            rule_name="High Anomaly Rate",
            alert_type=AlertType.ANOMALY,
            severity=Severity.CRITICAL,
            metric_name="anomaly_rate",
            operator=">",
            threshold=10,
            consecutive_violations=1,
            channels=[NotificationChannel.LOG, NotificationChannel.SLACK],
            description="Anomaly rate exceeds 10%"
        ),
        
        # ─── System Rules ───
        AlertRule(
            rule_id="system_prediction_latency",
            rule_name="High Prediction Latency",
            alert_type=AlertType.SYSTEM,
            severity=Severity.WARNING,
            metric_name="prediction_latency_ms",
            operator=">",
            threshold=1000,
            consecutive_violations=3,
            channels=[NotificationChannel.LOG],
            description="Prediction latency exceeds 1 second"
        ),
        
        AlertRule(
            rule_id="system_error_rate",
            rule_name="High Error Rate",
            alert_type=AlertType.SYSTEM,
            severity=Severity.CRITICAL,
            metric_name="error_rate_pct",
            operator=">",
            threshold=5,
            consecutive_violations=2,
            channels=[NotificationChannel.LOG, NotificationChannel.EMAIL, NotificationChannel.SLACK],
            description="System error rate exceeds 5%"
        )
    ]
    
    return rules


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════

def create_alert_manager(
    config: Optional[NotificationConfig] = None,
    include_default_rules: bool = True,
    max_history: int = 1000,
    enable_async: bool = False
) -> AlertManager:
    """
    Create alert manager with optional default rules.
    
    Args:
        config: Notification configuration
        include_default_rules: Add default alert rules
        max_history: Maximum alerts in history
        enable_async: Enable async notifications
    
    Returns:
        Configured AlertManager instance
    
    Example:
```python
        from agents.monitoring.alert_system import create_alert_manager
        
        # Basic usage
        manager = create_alert_manager()
        
        # With custom config
        config = NotificationConfig(
            email_enabled=True,
            email_to=['admin@example.com'],
            slack_enabled=True,
            slack_webhook_url='https://hooks.slack.com/...'
        )
        manager = create_alert_manager(config)
```
    """
    manager = AlertManager(
        notification_config=config,
        max_history=max_history,
        enable_async=enable_async
    )
    
    # Add default rules
    if include_default_rules:
        for rule in create_default_rules():
            manager.add_rule(rule)
        
        logger.info(f"✓ Added {len(create_default_rules())} default rules")
    
    return manager


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Module Initialization
# ═══════════════════════════════════════════════════════════════════════════

def _module_init():
    """Initialize module on import."""
    logger.info(f"✓ Alert System v{__version__} loaded")

_module_init()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION: Module Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Alert System v{__version__}")
    print(f"{'='*80}")
    
    # Create manager
    print("\n✓ Creating AlertManager...")
    manager = create_alert_manager()
    
    # Create test alert
    print("\n✓ Creating test alert...")
    alert = manager.create_data_drift_alert(
        feature_name="test_feature",
        drift_score=0.35,
        threshold=0.25,
        model_name="test_model"
    )
    
    print(f"  Alert ID: {alert.alert_id}")
    print(f"  Severity: {alert.severity.value}")
    print(f"  Status: {alert.status.value}")
    
    # Get statistics
    print("\n✓ Alert Statistics:")
    stats = manager.get_alert_statistics()
    for key, value in stats.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")
    
    print(f"\n{'='*80}")
    print("EXAMPLE USAGE:")
    print(f"{'='*80}")
    print("""
from agents.monitoring.alert_system import create_alert_manager

# Create manager
manager = create_alert_manager()

# Create custom alert
alert = manager.create_data_drift_alert(
    feature_name='age',
    drift_score=0.35,
    threshold=0.25,
    model_name='customer_churn'
)

# Evaluate metrics
alerts = manager.evaluate_metric('accuracy', 0.80)

# Get active alerts
active = manager.get_active_alerts(severity=Severity.CRITICAL)
    """)