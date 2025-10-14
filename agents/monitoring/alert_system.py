"""
DataGenius PRO - Alert System
Comprehensive alerting system for ML model monitoring and data quality.

Features:
- Multiple alert types (drift, performance, anomaly, data quality)
- Severity levels (info, warning, critical)
- Multiple notification channels (email, Slack, webhook, log)
- Alert deduplication
- Alert history tracking
- Configurable thresholds
- Rule-based and statistical detection
"""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import logging
import json
import hashlib
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

logger = logging.getLogger(__name__)


class AlertType(str, Enum):
    """Types of alerts."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_QUALITY = "data_quality"
    ANOMALY = "anomaly"
    SYSTEM = "system"
    THRESHOLD = "threshold"
    CUSTOM = "custom"


class Severity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class NotificationChannel(str, Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG = "log"
    DATABASE = "database"
    SMS = "sms"


class AlertStatus(str, Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Individual alert."""
    
    alert_id: str
    alert_type: AlertType
    severity: Severity
    title: str
    message: str
    
    # Metadata
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
    
    # Status tracking
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Notification tracking
    notified_channels: List[NotificationChannel] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "notified_channels": [c.value for c in self.notified_channels]
        }
    
    def get_hash(self) -> str:
        """Get unique hash for deduplication."""
        key = f"{self.alert_type}_{self.metric_name}_{self.model_name}_{self.dataset_name}"
        return hashlib.md5(key.encode()).hexdigest()


@dataclass
class AlertRule:
    """Rule for triggering alerts."""
    
    rule_id: str
    rule_name: str
    alert_type: AlertType
    severity: Severity
    
    # Condition
    metric_name: str
    operator: str  # >, <, >=, <=, ==, !=
    threshold: float
    
    # Optional advanced conditions
    consecutive_violations: int = 1  # Trigger after N consecutive violations
    time_window: Optional[int] = None  # Minutes
    
    # Notification settings
    enabled: bool = True
    channels: List[NotificationChannel] = field(default_factory=list)
    cooldown_minutes: int = 60  # Minimum time between alerts
    
    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    
    def evaluate(self, value: float) -> bool:
        """Evaluate if rule is triggered."""
        if not self.enabled:
            return False
        
        operators = {
            '>': lambda x, y: x > y,
            '<': lambda x, y: x < y,
            '>=': lambda x, y: x >= y,
            '<=': lambda x, y: x <= y,
            '==': lambda x, y: x == y,
            '!=': lambda x, y: x != y
        }
        
        op_func = operators.get(self.operator)
        if not op_func:
            logger.warning(f"Unknown operator: {self.operator}")
            return False
        
        return op_func(value, self.threshold)
    
    def is_in_cooldown(self) -> bool:
        """Check if rule is in cooldown period."""
        if self.last_triggered is None:
            return False
        
        cooldown_delta = timedelta(minutes=self.cooldown_minutes)
        return datetime.now() - self.last_triggered < cooldown_delta
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
            "enabled": self.enabled,
            "channels": [c.value for c in self.channels],
            "cooldown_minutes": self.cooldown_minutes,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
            "trigger_count": self.trigger_count
        }


@dataclass
class NotificationConfig:
    """Configuration for notification channels."""
    
    # Email
    email_enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)
    
    # Slack
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    slack_channel: str = "#alerts"
    
    # Webhook
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    
    # SMS (Twilio)
    sms_enabled: bool = False
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    sms_to_numbers: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "email_enabled": self.email_enabled,
            "email_to": self.email_to,
            "slack_enabled": self.slack_enabled,
            "slack_channel": self.slack_channel,
            "webhook_enabled": self.webhook_enabled,
            "sms_enabled": self.sms_enabled
        }


class AlertManager:
    """
    Central alert management system.
    
    Features:
    - Alert creation and tracking
    - Rule-based detection
    - Deduplication
    - Multi-channel notifications
    - Alert history
    """
    
    def __init__(
        self,
        notification_config: Optional[NotificationConfig] = None,
        max_history: int = 1000
    ):
        """Initialize alert manager."""
        self.notification_config = notification_config or NotificationConfig()
        self.max_history = max_history
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=max_history)
        
        # Rules
        self.rules: Dict[str, AlertRule] = {}
        
        # Violation tracking for consecutive violations
        self.violation_tracker: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        
        # Deduplication cache
        self.dedup_cache: Dict[str, datetime] = {}
        self.dedup_window_minutes = 60
        
        logger.info("AlertManager initialized")
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added rule: {rule.rule_name} (ID: {rule.rule_id})")
    
    def remove_rule(self, rule_id: str):
        """Remove an alert rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed rule: {rule_id}")
    
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
        
        # Find applicable rules
        applicable_rules = [
            rule for rule in self.rules.values()
            if rule.metric_name == metric_name and rule.enabled
        ]
        
        for rule in applicable_rules:
            # Check cooldown
            if rule.is_in_cooldown():
                continue
            
            # Evaluate rule
            if rule.evaluate(metric_value):
                # Track violation
                violation_key = f"{rule.rule_id}_{metric_name}"
                self.violation_tracker[violation_key].append(datetime.now())
                
                # Check consecutive violations
                recent_violations = [
                    dt for dt in self.violation_tracker[violation_key]
                    if datetime.now() - dt < timedelta(minutes=rule.time_window or 60)
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
                        rule.last_triggered = datetime.now()
                        rule.trigger_count += 1
                        
                        # Send notifications
                        self._send_notifications(alert, rule.channels)
                        
                        logger.warning(f"Alert triggered: {alert.title}")
            else:
                # Clear violations if value is back to normal
                violation_key = f"{rule.rule_id}_{metric_name}"
                if violation_key in self.violation_tracker:
                    self.violation_tracker[violation_key].clear()
        
        return triggered_alerts
    
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
        
        # Check deduplication
        if self._is_duplicate(alert):
            logger.info(f"Duplicate alert suppressed: {title}")
            return alert
        
        # Register alert
        self._register_alert(alert)
        
        # Send notifications
        if channels is None:
            channels = [NotificationChannel.LOG]
        
        self._send_notifications(alert, channels)
        
        logger.info(f"Alert created: {title} ({severity.value})")
        
        return alert
    
    def create_data_drift_alert(
        self,
        feature_name: str,
        drift_score: float,
        threshold: float,
        model_name: Optional[str] = None
    ) -> Alert:
        """Create data drift alert."""
        severity = Severity.CRITICAL if drift_score > threshold * 1.5 else Severity.WARNING
        
        return self.create_alert(
            alert_type=AlertType.DATA_DRIFT,
            severity=severity,
            title=f"Data Drift Detected: {feature_name}",
            message=f"Feature '{feature_name}' shows significant drift (score: {drift_score:.4f}, threshold: {threshold:.4f})",
            metric_name="drift_score",
            metric_value=drift_score,
            threshold=threshold,
            model_name=model_name,
            feature_names=[feature_name],
            recommendations=[
                "Review recent data changes",
                "Check data collection pipeline",
                "Consider model retraining",
                "Investigate distribution shift"
            ]
        )
    
    def create_performance_alert(
        self,
        metric_name: str,
        current_value: float,
        baseline_value: float,
        degradation_pct: float,
        model_name: Optional[str] = None
    ) -> Alert:
        """Create performance degradation alert."""
        severity = Severity.CRITICAL if degradation_pct > 20 else Severity.WARNING
        
        return self.create_alert(
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=severity,
            title=f"Performance Degradation: {metric_name}",
            message=f"Model performance declined by {degradation_pct:.1f}% ({metric_name}: {current_value:.4f} vs baseline: {baseline_value:.4f})",
            metric_name=metric_name,
            metric_value=current_value,
            threshold=baseline_value,
            model_name=model_name,
            recommendations=[
                "Retrain model with recent data",
                "Check for data drift",
                "Review feature engineering",
                "Validate data quality"
            ],
            details={
                "current_value": current_value,
                "baseline_value": baseline_value,
                "degradation_pct": degradation_pct
            }
        )
    
    def create_data_quality_alert(
        self,
        issue_type: str,
        affected_features: List[str],
        severity: Severity = Severity.WARNING
    ) -> Alert:
        """Create data quality alert."""
        return self.create_alert(
            alert_type=AlertType.DATA_QUALITY,
            severity=severity,
            title=f"Data Quality Issue: {issue_type}",
            message=f"Detected {issue_type} in {len(affected_features)} features: {', '.join(affected_features[:3])}{'...' if len(affected_features) > 3 else ''}",
            feature_names=affected_features,
            recommendations=[
                "Review data collection process",
                "Check data validation rules",
                "Investigate data source",
                "Update preprocessing pipeline"
            ],
            details={
                "issue_type": issue_type,
                "n_affected": len(affected_features)
            }
        )
    
    def create_anomaly_alert(
        self,
        anomaly_score: float,
        threshold: float,
        n_anomalies: int,
        model_name: Optional[str] = None
    ) -> Alert:
        """Create anomaly detection alert."""
        severity = Severity.CRITICAL if n_anomalies > 100 else Severity.WARNING
        
        return self.create_alert(
            alert_type=AlertType.ANOMALY,
            severity=severity,
            title=f"Anomalies Detected",
            message=f"Detected {n_anomalies} anomalies (score: {anomaly_score:.4f}, threshold: {threshold:.4f})",
            metric_name="anomaly_score",
            metric_value=anomaly_score,
            threshold=threshold,
            model_name=model_name,
            recommendations=[
                "Investigate anomalous samples",
                "Check for data corruption",
                "Review outlier handling",
                "Validate model assumptions"
            ],
            details={
                "n_anomalies": n_anomalies,
                "anomaly_rate": n_anomalies / 1000  # Assuming batch size
            }
        )
    
    def acknowledge_alert(self, alert_id: str, user: Optional[str] = None):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            
            logger.info(f"Alert acknowledged: {alert_id}" + (f" by {user}" if user else ""))
    
    def resolve_alert(self, alert_id: str, user: Optional[str] = None):
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            
            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert_id}" + (f" by {user}" if user else ""))
    
    def get_active_alerts(
        self,
        alert_type: Optional[AlertType] = None,
        severity: Optional[Severity] = None,
        model_name: Optional[str] = None
    ) -> List[Alert]:
        """Get active alerts with optional filtering."""
        alerts = list(self.active_alerts.values())
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if model_name:
            alerts = [a for a in alerts if a.model_name == model_name]
        
        # Sort by severity and creation time
        severity_order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
        alerts.sort(key=lambda x: (severity_order[x.severity], x.created_at), reverse=True)
        
        return alerts
    
    def get_alert_history(
        self,
        limit: int = 100,
        alert_type: Optional[AlertType] = None
    ) -> List[Alert]:
        """Get alert history."""
        history = list(self.alert_history)
        
        if alert_type:
            history = [a for a in history if a.alert_type == alert_type]
        
        return history[:limit]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        all_alerts = list(self.active_alerts.values()) + list(self.alert_history)
        
        if not all_alerts:
            return {
                "total_alerts": 0,
                "active_alerts": 0,
                "by_type": {},
                "by_severity": {},
                "by_status": {}
            }
        
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        by_status = defaultdict(int)
        
        for alert in all_alerts:
            by_type[alert.alert_type.value] += 1
            by_severity[alert.severity.value] += 1
            by_status[alert.status.value] += 1
        
        return {
            "total_alerts": len(all_alerts),
            "active_alerts": len(self.active_alerts),
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "by_status": dict(by_status),
            "oldest_active": min([a.created_at for a in self.active_alerts.values()]).isoformat() if self.active_alerts else None
        }
    
    def _create_alert_from_rule(
        self,
        rule: AlertRule,
        metric_value: float,
        context: Dict[str, Any]
    ) -> Alert:
        """Create alert from triggered rule."""
        alert_id = self._generate_alert_id()
        
        return Alert(
            alert_id=alert_id,
            alert_type=rule.alert_type,
            severity=rule.severity,
            title=rule.rule_name,
            message=f"{rule.description or rule.rule_name} triggered: {rule.metric_name} {rule.operator} {rule.threshold} (current: {metric_value:.4f})",
            metric_name=rule.metric_name,
            metric_value=metric_value,
            threshold=rule.threshold,
            model_name=context.get("model_name"),
            dataset_name=context.get("dataset_name"),
            details=context
        )
    
    def _register_alert(self, alert: Alert):
        """Register alert in active alerts and history."""
        self.active_alerts[alert.alert_id] = alert
        
        # Update deduplication cache
        alert_hash = alert.get_hash()
        self.dedup_cache[alert_hash] = datetime.now()
    
    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is a duplicate within dedup window."""
        alert_hash = alert.get_hash()
        
        if alert_hash in self.dedup_cache:
            last_seen = self.dedup_cache[alert_hash]
            time_diff = datetime.now() - last_seen
            
            if time_diff < timedelta(minutes=self.dedup_window_minutes):
                return True
        
        return False
    
    def _send_notifications(
        self,
        alert: Alert,
        channels: List[NotificationChannel]
    ):
        """Send notifications through specified channels."""
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
                
            except Exception as e:
                logger.error(f"Failed to send notification via {channel}: {e}")
    
    def _send_email(self, alert: Alert):
        """Send email notification."""
        if not self.notification_config.email_to:
            return
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
        msg['From'] = self.notification_config.email_from
        msg['To'] = ", ".join(self.notification_config.email_to)
        
        # Create body
        html_body = f"""
        <html>
          <body>
            <h2 style="color: {'red' if alert.severity == Severity.CRITICAL else 'orange'};">
              {alert.title}
            </h2>
            <p><strong>Severity:</strong> {alert.severity.value}</p>
            <p><strong>Type:</strong> {alert.alert_type.value}</p>
            <p><strong>Time:</strong> {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Message:</strong> {alert.message}</p>
            
            {f'<p><strong>Model:</strong> {alert.model_name}</p>' if alert.model_name else ''}
            {f'<p><strong>Metric:</strong> {alert.metric_name} = {alert.metric_value:.4f}</p>' if alert.metric_name else ''}
            
            {f'<h3>Recommendations:</h3><ul>' + ''.join([f'<li>{r}</li>' for r in alert.recommendations]) + '</ul>' if alert.recommendations else ''}
          </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, 'html'))
        
        # Send
        try:
            with smtplib.SMTP(self.notification_config.smtp_host, self.notification_config.smtp_port) as server:
                server.starttls()
                server.login(self.notification_config.smtp_user, self.notification_config.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent for alert: {alert.alert_id}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    def _send_slack(self, alert: Alert):
        """Send Slack notification."""
        color = {
            Severity.INFO: "#36a64f",
            Severity.WARNING: "#ff9800",
            Severity.CRITICAL: "#f44336"
        }
        
        payload = {
            "channel": self.notification_config.slack_channel,
            "username": "DataGenius Alert Bot",
            "icon_emoji": ":warning:",
            "attachments": [{
                "color": color[alert.severity],
                "title": alert.title,
                "text": alert.message,
                "fields": [
                    {"title": "Severity", "value": alert.severity.value, "short": True},
                    {"title": "Type", "value": alert.alert_type.value, "short": True},
                ],
                "footer": "DataGenius PRO",
                "ts": int(alert.created_at.timestamp())
            }]
        }
        
        if alert.model_name:
            payload["attachments"][0]["fields"].append(
                {"title": "Model", "value": alert.model_name, "short": True}
            )
        
        try:
            response = requests.post(
                self.notification_config.slack_webhook_url,
                json=payload
            )
            response.raise_for_status()
            logger.info(f"Slack notification sent for alert: {alert.alert_id}")
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    def _send_webhook(self, alert: Alert):
        """Send webhook notification."""
        payload = alert.to_dict()
        
        try:
            response = requests.post(
                self.notification_config.webhook_url,
                json=payload,
                headers=self.notification_config.webhook_headers
            )
            response.raise_for_status()
            logger.info(f"Webhook sent for alert: {alert.alert_id}")
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
    
    def _send_sms(self, alert: Alert):
        """Send SMS notification via Twilio."""
        try:
            from twilio.rest import Client
            
            client = Client(
                self.notification_config.twilio_account_sid,
                self.notification_config.twilio_auth_token
            )
            
            message_body = f"[{alert.severity.value.upper()}] {alert.title}: {alert.message}"
            
            for number in self.notification_config.sms_to_numbers:
                client.messages.create(
                    body=message_body[:160],  # SMS limit
                    from_=self.notification_config.twilio_from_number,
                    to=number
                )
            
            logger.info(f"SMS sent for alert: {alert.alert_id}")
        except ImportError:
            logger.warning("Twilio not installed, cannot send SMS")
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
    
    def _log_alert(self, alert: Alert):
        """Log alert to logger."""
        log_level = {
            Severity.INFO: logging.INFO,
            Severity.WARNING: logging.WARNING,
            Severity.CRITICAL: logging.CRITICAL
        }
        
        logger.log(
            log_level[alert.severity],
            f"ALERT [{alert.severity.value}] {alert.title}: {alert.message}"
        )
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"ALERT-{timestamp}"
    
    def export_alerts(self, filepath: Union[str, Path]):
        """Export alerts to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "active_alerts": [alert.to_dict() for alert in self.active_alerts.values()],
            "alert_history": [alert.to_dict() for alert in self.alert_history],
            "rules": [rule.to_dict() for rule in self.rules.values()],
            "statistics": self.get_alert_statistics(),
            "exported_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Alerts exported to {filepath}")
    
    def clear_resolved_alerts(self, older_than_days: int = 30):
        """Clear resolved alerts older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        # Filter history
        old_len = len(self.alert_history)
        self.alert_history = deque(
            [a for a in self.alert_history if a.created_at > cutoff_date],
            maxlen=self.max_history
        )
        
        removed = old_len - len(self.alert_history)
        logger.info(f"Cleared {removed} old alerts")


# Pre-configured alert rules
def create_default_rules() -> List[AlertRule]:
    """Create default alert rules."""
    rules = [
        # Performance degradation
        AlertRule(
            rule_id="perf_accuracy_drop",
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
        
        # Data drift
        AlertRule(
            rule_id="drift_high",
            rule_name="High Data Drift",
            alert_type=AlertType.DATA_DRIFT,
            severity=Severity.CRITICAL,
            metric_name="psi_score",
            operator=">",
            threshold=0.25,
            consecutive_violations=1,
            channels=[NotificationChannel.LOG, NotificationChannel.SLACK],
            description="PSI score indicates significant drift"
        ),
        
        # Data quality
        AlertRule(
            rule_id="missing_high",
            rule_name="High Missing Values",
            alert_type=AlertType.DATA_QUALITY,
            severity=Severity.WARNING,
            metric_name="missing_pct",
            operator=">",
            threshold=30,
            consecutive_violations=1,
            channels=[NotificationChannel.LOG],
            description="More than 30% missing values detected"
        ),
        
        # Anomalies
        AlertRule(
            rule_id="anomaly_rate_high",
            rule_name="High Anomaly Rate",
            alert_type=AlertType.ANOMALY,
            severity=Severity.WARNING,
            metric_name="anomaly_rate",
            operator=">",
            threshold=5,
            consecutive_violations=2,
            channels=[NotificationChannel.LOG],
            description="Anomaly rate exceeds 5%"
        )
    ]
    
    return rules


# Convenience functions
def create_alert_manager(config: Optional[NotificationConfig] = None) -> AlertManager:
    """Create alert manager with default rules."""
    manager = AlertManager(notification_config=config)
    
    # Add default rules
    for rule in create_default_rules():
        manager.add_rule(rule)
    
    return manager