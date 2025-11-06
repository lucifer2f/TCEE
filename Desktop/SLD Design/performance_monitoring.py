"""
Performance Monitoring and Reporting System for AI Excel Extraction Integration

This module provides comprehensive performance monitoring, metrics collection, and reporting
capabilities for the AI Excel extraction system integration layer.

Key Features:
- Real-time performance metrics collection
- Automated performance alerts and threshold monitoring
- Historical performance analysis and trending
- Comprehensive reporting with visualizations
- Performance regression detection
- Resource utilization monitoring
- Quality assurance metrics tracking
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    component: str  # e.g., "extraction", "calculation", "standards", "sld"
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert definition"""
    alert_id: str
    metric_name: str
    threshold_value: float
    comparison_operator: str  # ">", "<", ">=", "<=", "=="
    severity: str  # "info", "warning", "critical"
    message: str
    triggered_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    status: str = "active"  # "active", "resolved", "acknowledged"


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    report_id: str
    generated_at: datetime
    report_period: Tuple[datetime, datetime]
    summary_metrics: Dict[str, float]
    component_performance: Dict[str, Dict[str, float]]
    quality_metrics: Dict[str, float]
    alerts_summary: Dict[str, int]
    trends_analysis: Dict[str, Any]
    recommendations: List[str]


class MetricsCollector:
    """
    Collects and stores performance metrics from the integration system
    """

    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self.metrics_history: deque = deque(maxlen=max_history_size)
        self.current_session_metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)

    def record_metric(self, metric_name: str, value: float, unit: str, component: str, tags: Dict[str, str] = None):
        """
        Record a performance metric
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            component: System component name
            tags: Additional tags for categorization
        """
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            unit=unit,
            component=component,
            tags=tags or {}
        )
        
        self.metrics_history.append(metric)
        self.current_session_metrics[component].append(metric)
        
        self.logger.debug(f"Recorded metric: {metric_name}={value}{unit} for {component}")

    def record_integration_workflow_metrics(self, project_name: str, workflow_metrics: Dict[str, Any]):
        """
        Record metrics for a complete integration workflow
        
        Args:
            project_name: Name of the project being processed
            workflow_metrics: Dictionary containing workflow timing and quality metrics
        """
        timestamp = datetime.now()
        tags = {"project_name": project_name}
        
        # Record timing metrics
        for metric_name, value in workflow_metrics.get("timing_metrics", {}).items():
            self.record_metric(
                metric_name=f"workflow_{metric_name}",
                value=value,
                unit="seconds",
                component="integration",
                tags=tags
            )
        
        # Record quality metrics
        for metric_name, value in workflow_metrics.get("quality_metrics", {}).items():
            if isinstance(value, (int, float)):
                self.record_metric(
                    metric_name=f"quality_{metric_name}",
                    value=value,
                    unit="percent" if "rate" in metric_name or "score" in metric_name else "count",
                    component="quality",
                    tags=tags
                )
        
        # Record throughput metrics
        for metric_name, value in workflow_metrics.get("throughput_metrics", {}).items():
            self.record_metric(
                metric_name=f"throughput_{metric_name}",
                value=value,
                unit="items_per_second",
                component="performance",
                tags=tags
            )

    def get_metrics_by_component(self, component: str, time_window: Optional[timedelta] = None) -> List[PerformanceMetric]:
        """
        Get metrics for a specific component within a time window
        
        Args:
            component: Component name
            time_window: Optional time window to filter metrics
            
        Returns:
            List of metrics for the component
        """
        if time_window is None:
            # Return all metrics for component
            return [m for m in self.metrics_history if m.component == component]
        
        cutoff_time = datetime.now() - time_window
        return [
            m for m in self.metrics_history 
            if m.component == component and m.timestamp >= cutoff_time
        ]

    def get_metric_statistics(self, metric_name: str, component: str = None, time_window: Optional[timedelta] = None) -> Dict[str, float]:
        """
        Calculate statistics for a specific metric
        
        Args:
            metric_name: Name of the metric
            component: Optional component filter
            time_window: Optional time window
            
        Returns:
            Dictionary with statistics (count, min, max, avg, std)
        """
        metrics = self._filter_metrics(metric_name, component, time_window)
        
        if not metrics:
            return {"count": 0, "min": 0, "max": 0, "avg": 0, "std": 0}
        
        values = [m.value for m in metrics]
        
        # Calculate statistics
        count = len(values)
        min_val = min(values)
        max_val = max(values)
        avg_val = sum(values) / count
        
        # Calculate standard deviation
        variance = sum((x - avg_val) ** 2 for x in values) / count
        std_val = variance ** 0.5
        
        return {
            "count": count,
            "min": min_val,
            "max": max_val,
            "avg": avg_val,
            "std": std_val
        }

    def _filter_metrics(self, metric_name: str, component: str = None, time_window: Optional[timedelta] = None) -> List[PerformanceMetric]:
        """Filter metrics based on criteria"""
        filtered = self.metrics_history
        
        # Filter by metric name
        filtered = [m for m in filtered if m.metric_name == metric_name]
        
        # Filter by component
        if component:
            filtered = [m for m in filtered if m.component == component]
        
        # Filter by time window
        if time_window:
            cutoff_time = datetime.now() - time_window
            filtered = [m for m in filtered if m.timestamp >= cutoff_time]
        
        return filtered

    def export_metrics(self, file_path: str, time_window: Optional[timedelta] = None):
        """
        Export metrics to JSON file
        
        Args:
            file_path: Path to export file
            time_window: Optional time window to export
        """
        metrics_data = []
        
        for metric in self.metrics_history:
            if time_window is None or metric.timestamp >= datetime.now() - time_window:
                metrics_data.append(asdict(metric))
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_metrics": len(metrics_data),
            "metrics": metrics_data
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported {len(metrics_data)} metrics to {file_path}")


class AlertManager:
    """
    Manages performance alerts and threshold monitoring
    """

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts: List[PerformanceAlert] = []
        self.alert_rules: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    def add_alert_rule(self, metric_name: str, threshold: float, operator: str, 
                      severity: str, message: str, component: str = None):
        """
        Add a new alert rule
        
        Args:
            metric_name: Name of the metric to monitor
            threshold: Threshold value
            operator: Comparison operator (">", "<", ">=", "<=", "==")
            severity: Alert severity ("info", "warning", "critical")
            message: Alert message template
            component: Optional component filter
        """
        alert_id = f"{metric_name}_{component or 'all'}_{int(time.time())}"
        
        rule = {
            "alert_id": alert_id,
            "metric_name": metric_name,
            "threshold": threshold,
            "operator": operator,
            "severity": severity,
            "message": message,
            "component": component,
            "created_at": datetime.now()
        }
        
        self.alert_rules.append(rule)
        self.logger.info(f"Added alert rule: {alert_id}")

    def check_alerts(self) -> List[PerformanceAlert]:
        """
        Check current metrics against alert rules and trigger alerts
        
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        for rule in self.alert_rules:
            # Get recent metrics for this rule
            recent_metrics = self.metrics_collector._filter_metrics(
                rule["metric_name"], 
                rule["component"],
                timedelta(minutes=5)  # Check last 5 minutes
            )
            
            if not recent_metrics:
                continue
            
            # Check latest metric against threshold
            latest_metric = max(recent_metrics, key=lambda m: m.timestamp)
            
            if self._evaluate_condition(latest_metric.value, rule["threshold"], rule["operator"]):
                # Check if alert is already active
                existing_alert = self._find_active_alert(rule["alert_id"])
                
                if not existing_alert:
                    # Create new alert
                    alert = PerformanceAlert(
                        alert_id=rule["alert_id"],
                        metric_name=rule["metric_name"],
                        threshold_value=rule["threshold"],
                        comparison_operator=rule["operator"],
                        severity=rule["severity"],
                        message=rule["message"].format(
                            value=latest_metric.value,
                            threshold=rule["threshold"],
                            component=latest_metric.component
                        ),
                        triggered_at=datetime.now()
                    )
                    
                    self.alerts.append(alert)
                    triggered_alerts.append(alert)
                    
                    self.logger.warning(f"Alert triggered: {alert.alert_id} - {alert.message}")
        
        return triggered_alerts

    def _evaluate_condition(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate if a value meets the alert condition"""
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return abs(value - threshold) < 0.001  # Float comparison with tolerance
        else:
            return False

    def _find_active_alert(self, alert_id: str) -> Optional[PerformanceAlert]:
        """Find active alert by ID"""
        for alert in self.alerts:
            if alert.alert_id == alert_id and alert.status == "active":
                return alert
        return None

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        alert = self._find_active_alert(alert_id)
        if alert:
            alert.status = "acknowledged"
            self.logger.info(f"Alert acknowledged: {alert_id}")

    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        alert = self._find_active_alert(alert_id)
        if alert:
            alert.status = "resolved"
            alert.resolved_at = datetime.now()
            self.logger.info(f"Alert resolved: {alert_id}")

    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active alerts"""
        return [alert for alert in self.alerts if alert.status == "active"]

    def get_alerts_by_severity(self, severity: str) -> List[PerformanceAlert]:
        """Get alerts by severity level"""
        return [alert for alert in self.alerts if alert.severity == severity]


class PerformanceAnalyzer:
    """
    Analyzes performance trends and generates insights
    """

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)

    def analyze_trends(self, metric_name: str, component: str = None, 
                      time_window: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """
        Analyze trends for a specific metric
        
        Args:
            metric_name: Name of the metric to analyze
            component: Optional component filter
            time_window: Time window for analysis
            
        Returns:
            Trend analysis results
        """
        metrics = self.metrics_collector._filter_metrics(metric_name, component, time_window)
        
        if len(metrics) < 2:
            return {"trend": "insufficient_data", "analysis": "Need at least 2 data points"}
        
        # Sort metrics by timestamp
        metrics.sort(key=lambda m: m.timestamp)
        
        # Calculate trend
        values = [m.value for m in metrics]
        
        # Simple linear trend calculation
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        # Linear regression slope
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        # Determine trend direction
        if slope > 0.01:
            trend = "increasing"
        elif slope < -0.01:
            trend = "decreasing"
        else:
            trend = "stable"
        
        # Calculate volatility (coefficient of variation)
        avg_value = sum(values) / n
        if avg_value > 0:
            variance = sum((x - avg_value) ** 2 for x in values) / n
            volatility = (variance ** 0.5) / avg_value
        else:
            volatility = 0
        
        # Detect anomalies (values outside 2 standard deviations)
        std_dev = (variance ** 0.5) if 'variance' in locals() else 0
        anomalies = []
        
        if std_dev > 0:
            mean = avg_value
            for metric in metrics:
                z_score = abs(metric.value - mean) / std_dev
                if z_score > 2:
                    anomalies.append({
                        "timestamp": metric.timestamp.isoformat(),
                        "value": metric.value,
                        "z_score": z_score
                    })
        
        return {
            "trend": trend,
            "slope": slope,
            "volatility": volatility,
            "data_points": len(metrics),
            "avg_value": avg_value,
            "min_value": min(values),
            "max_value": max(values),
            "anomalies": anomalies,
            "analysis_period": {
                "start": metrics[0].timestamp.isoformat(),
                "end": metrics[-1].timestamp.isoformat()
            }
        }

    def compare_periods(self, metric_name: str, component: str = None,
                       current_period: timedelta = timedelta(hours=24),
                       previous_period: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """
        Compare performance between two time periods
        
        Args:
            metric_name: Name of the metric to compare
            component: Optional component filter
            current_period: Duration of current period
            previous_period: Duration of previous period
            
        Returns:
            Period comparison results
        """
        now = datetime.now()
        
        # Get metrics for current period
        current_metrics = self.metrics_collector._filter_metrics(
            metric_name, component, current_period
        )
        
        # Get metrics for previous period
        previous_start = now - current_period - previous_period
        previous_end = now - current_period
        previous_metrics = [
            m for m in self.metrics_collector._filter_metrics(metric_name, component)
            if previous_start <= m.timestamp <= previous_end
        ]
        
        # Calculate statistics for both periods
        current_stats = self._calculate_period_stats(current_metrics)
        previous_stats = self._calculate_period_stats(previous_metrics)
        
        # Calculate changes
        changes = {}
        for stat_name in ["avg", "min", "max", "count"]:
            if previous_stats[stat_name] != 0:
                change_pct = ((current_stats[stat_name] - previous_stats[stat_name]) / 
                            previous_stats[stat_name]) * 100
                changes[f"{stat_name}_change_percent"] = change_pct
            else:
                changes[f"{stat_name}_change_percent"] = 0 if current_stats[stat_name] == 0 else 100
        
        return {
            "current_period": {
                "start": (now - current_period).isoformat(),
                "end": now.isoformat(),
                "stats": current_stats
            },
            "previous_period": {
                "start": previous_start.isoformat(),
                "end": previous_end.isoformat(),
                "stats": previous_stats
            },
            "changes": changes,
            "significant_changes": self._identify_significant_changes(changes)
        }

    def _calculate_period_stats(self, metrics: List[PerformanceMetric]) -> Dict[str, float]:
        """Calculate statistics for a period"""
        if not metrics:
            return {"avg": 0, "min": 0, "max": 0, "count": 0}
        
        values = [m.value for m in metrics]
        return {
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values)
        }

    def _identify_significant_changes(self, changes: Dict[str, float]) -> List[str]:
        """Identify significant changes (threshold: 20%)"""
        significant = []
        
        for change_name, change_value in changes.items():
            if abs(change_value) > 20:
                direction = "increased" if change_value > 0 else "decreased"
                metric_name = change_name.replace("_change_percent", "")
                significant.append(f"{metric_name} {direction} by {abs(change_value):.1f}%")
        
        return significant

    def generate_performance_insights(self, time_window: timedelta = timedelta(hours=24)) -> List[str]:
        """
        Generate performance insights based on analysis
        
        Args:
            time_window: Time window for analysis
            
        Returns:
            List of performance insights
        """
        insights = []
        
        # Analyze key metrics
        key_metrics = [
            "workflow_total_integration_time",
            "quality_data_quality_score",
            "throughput_components_per_second",
            "quality_calculation_success_rate"
        ]
        
        for metric in key_metrics:
            try:
                trend_analysis = self.analyze_trends(metric, time_window=time_window)
                
                # Generate insights based on trend
                if trend_analysis["trend"] == "increasing" and "time" in metric:
                    insights.append(f"Integration time is increasing - may indicate performance degradation")
                elif trend_analysis["trend"] == "decreasing" and "score" in metric:
                    insights.append(f"Quality score is improving - good progress!")
                elif trend_analysis["volatility"] > 0.5:
                    insights.append(f"High volatility detected in {metric} - investigate consistency")
                elif trend_analysis["anomalies"]:
                    insights.append(f"Anomalies detected in {metric} - review outlier events")
                
            except Exception as e:
                self.logger.warning(f"Could not analyze metric {metric}: {e}")
        
        # Check alert activity
        # This would integrate with the alert manager in a real implementation
        insights.append("Monitor alert frequency for performance degradation patterns")
        
        return insights


class PerformanceReporter:
    """
    Generates comprehensive performance reports
    """

    def __init__(self, metrics_collector: MetricsCollector, analyzer: PerformanceAnalyzer):
        self.metrics_collector = metrics_collector
        self.analyzer = analyzer
        self.logger = logging.getLogger(__name__)

    def generate_comprehensive_report(self, report_period: Tuple[datetime, datetime] = None) -> PerformanceReport:
        """
        Generate a comprehensive performance report
        
        Args:
            report_period: Tuple of (start_time, end_time) for the report period
            
        Returns:
            PerformanceReport object
        """
        if report_period is None:
            # Default to last 24 hours
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            report_period = (start_time, end_time)
        
        start_time, end_time = report_period
        
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(start_time, end_time)
        
        # Analyze component performance
        component_performance = self._analyze_component_performance(start_time, end_time)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(start_time, end_time)
        
        # Get alerts summary
        alerts_summary = self._get_alerts_summary(start_time, end_time)
        
        # Perform trends analysis
        trends_analysis = self._perform_trends_analysis(start_time, end_time)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(summary_metrics, trends_analysis)
        
        # Create report
        report = PerformanceReport(
            report_id=f"perf_report_{int(time.time())}",
            generated_at=datetime.now(),
            report_period=report_period,
            summary_metrics=summary_metrics,
            component_performance=component_performance,
            quality_metrics=quality_metrics,
            alerts_summary=alerts_summary,
            trends_analysis=trends_analysis,
            recommendations=recommendations
        )
        
        self.logger.info(f"Generated performance report: {report.report_id}")
        return report

    def _calculate_summary_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, float]:
        """Calculate summary performance metrics"""
        time_window = end_time - start_time
        metrics = [
            m for m in self.metrics_collector.metrics_history
            if start_time <= m.timestamp <= end_time
        ]
        
        # Group metrics by component
        component_metrics = defaultdict(list)
        for metric in metrics:
            component_metrics[metric.component].append(metric.value)
        
        summary = {
            "total_metrics_collected": len(metrics),
            "unique_components": len(component_metrics),
            "report_duration_hours": time_window.total_seconds() / 3600
        }
        
        # Calculate averages for key metrics
        key_metrics = ["workflow_total_integration_time", "quality_data_quality_score", "throughput_components_per_second"]
        
        for metric_name in key_metrics:
            metric_values = [m.value for m in metrics if m.metric_name == metric_name]
            if metric_values:
                summary[f"avg_{metric_name}"] = sum(metric_values) / len(metric_values)
                summary[f"min_{metric_name}"] = min(metric_values)
                summary[f"max_{metric_name}"] = max(metric_values)
            else:
                summary[f"avg_{metric_name}"] = 0
                summary[f"min_{metric_name}"] = 0
                summary[f"max_{metric_name}"] = 0
        
        return summary

    def _analyze_component_performance(self, start_time: datetime, end_time: datetime) -> Dict[str, Dict[str, float]]:
        """Analyze performance by component"""
        time_window = timedelta(seconds=(end_time - start_time).total_seconds())
        metrics = [
            m for m in self.metrics_collector.metrics_history
            if start_time <= m.timestamp <= end_time
        ]
        
        component_performance = {}
        
        for component in set(m.component for m in metrics):
            component_metrics = [m for m in metrics if m.component == component]
            
            # Calculate component-specific metrics
            total_time = sum(m.value for m in component_metrics if "time" in m.metric_name)
            throughput = len(component_metrics) / time_window.total_seconds() if time_window.total_seconds() > 0 else 0
            
            component_performance[component] = {
                "total_operations": len(component_metrics),
                "avg_operation_time": total_time / len(component_metrics) if component_metrics else 0,
                "throughput_ops_per_second": throughput,
                "unique_metrics": len(set(m.metric_name for m in component_metrics))
            }
        
        return component_performance

    def _calculate_quality_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, float]:
        """Calculate quality-related metrics"""
        quality_metrics = [
            m for m in self.metrics_collector.metrics_history
            if start_time <= m.timestamp <= end_time and m.component == "quality"
        ]
        
        if not quality_metrics:
            return {}
        
        quality_summary = {}
        quality_names = set(m.metric_name for m in quality_metrics)
        
        for quality_name in quality_names:
            quality_values = [m.value for m in quality_metrics if m.metric_name == quality_name]
            quality_summary[quality_name] = {
                "avg": sum(quality_values) / len(quality_values),
                "min": min(quality_values),
                "max": max(quality_values),
                "count": len(quality_values)
            }
        
        return quality_summary

    def _get_alerts_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, int]:
        """Get summary of alerts in the period"""
        # This would integrate with AlertManager in a real implementation
        # For now, return placeholder data
        return {
            "total_alerts": 0,
            "active_alerts": 0,
            "critical_alerts": 0,
            "warning_alerts": 0,
            "resolved_alerts": 0
        }

    def _perform_trends_analysis(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Perform trends analysis for the report period"""
        time_window = end_time - start_time
        
        # Analyze trends for key metrics
        key_metrics = [
            "workflow_total_integration_time",
            "quality_data_quality_score",
            "throughput_components_per_second"
        ]
        
        trends = {}
        for metric in key_metrics:
            try:
                trends[metric] = self.analyzer.analyze_trends(metric, time_window=time_window)
            except Exception as e:
                trends[metric] = {"error": str(e)}
        
        return {
            "key_metric_trends": trends,
            "analysis_period_hours": time_window.total_seconds() / 3600
        }

    def _generate_recommendations(self, summary_metrics: Dict[str, float], trends_analysis: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Analyze summary metrics for recommendations
        avg_integration_time = summary_metrics.get("avg_workflow_total_integration_time", 0)
        if avg_integration_time > 10:
            recommendations.append("Integration time is high - consider optimizing calculation algorithms")
        
        avg_quality_score = summary_metrics.get("avg_quality_data_quality_score", 0)
        if avg_quality_score < 0.8:
            recommendations.append("Quality score is below target - review data extraction and validation processes")
        
        avg_throughput = summary_metrics.get("avg_throughput_components_per_second", 0)
        if avg_throughput < 1:
            recommendations.append("Processing throughput is low - consider scaling resources or optimizing workflows")
        
        # Analyze trends for recommendations
        trends = trends_analysis.get("key_metric_trends", {})
        
        integration_time_trend = trends.get("workflow_total_integration_time", {}).get("trend")
        if integration_time_trend == "increasing":
            recommendations.append("Integration time is trending upward - implement performance monitoring alerts")
        
        # Check for high volatility
        for metric_name, trend_data in trends.items():
            if trend_data.get("volatility", 0) > 0.5:
                recommendations.append(f"High volatility detected in {metric_name} - investigate consistency issues")
        
        return recommendations

    def export_report(self, report: PerformanceReport, file_path: str, format: str = "json"):
        """
        Export performance report to file
        
        Args:
            report: PerformanceReport object
            file_path: Path to export file
            format: Export format ("json", "html", "csv")
        """
        if format == "json":
            self._export_json_report(report, file_path)
        elif format == "html":
            self._export_html_report(report, file_path)
        elif format == "csv":
            self._export_csv_report(report, file_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported performance report to {file_path}")

    def _export_json_report(self, report: PerformanceReport, file_path: str):
        """Export report as JSON"""
        report_data = asdict(report)
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

    def _export_html_report(self, report: PerformanceReport, file_path: str):
        """Export report as HTML"""
        html_content = self._generate_html_report(report)
        with open(file_path, 'w') as f:
            f.write(html_content)

    def _export_csv_report(self, report: PerformanceReport, file_path: str):
        """Export report as CSV"""
        import csv
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write summary metrics
            writer.writerow(["Metric", "Value"])
            for metric, value in report.summary_metrics.items():
                writer.writerow([metric, value])
            
            writer.writerow([])  # Empty row for separation
            
            # Write recommendations
            writer.writerow(["Recommendations"])
            for i, rec in enumerate(report.recommendations, 1):
                writer.writerow([f"{i}. {rec}"])

    def _generate_html_report(self, report: PerformanceReport) -> str:
        """Generate HTML report content"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Report - {report.report_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ margin: 10px 0; }}
                .recommendation {{ background: #f0f8ff; padding: 10px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>Performance Report</h1>
            <p><strong>Report ID:</strong> {report.report_id}</p>
            <p><strong>Generated:</strong> {report.generated_at}</p>
            <p><strong>Period:</strong> {report.report_period[0]} to {report.report_period[1]}</p>
            
            <div class="section">
                <h2>Summary Metrics</h2>
                {self._format_metrics_html(report.summary_metrics)}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {self._format_recommendations_html(report.recommendations)}
            </div>
        </body>
        </html>
        """

    def _format_metrics_html(self, metrics: Dict[str, float]) -> str:
        """Format metrics as HTML"""
        html = ""
        for metric, value in metrics.items():
            html += f'<div class="metric">{metric}: {value}</div>'
        return html

    def _format_recommendations_html(self, recommendations: List[str]) -> str:
        """Format recommendations as HTML"""
        html = ""
        for rec in recommendations:
            html += f'<div class="recommendation">{rec}</div>'
        return html


# Performance monitoring system main class
class PerformanceMonitoringSystem:
    """
    Main performance monitoring system that orchestrates all components
    """

    def __init__(self, max_metrics_history: int = 10000):
        self.metrics_collector = MetricsCollector(max_metrics_history)
        self.alert_manager = AlertManager(self.metrics_collector)
        self.performance_analyzer = PerformanceAnalyzer(self.metrics_collector)
        self.performance_reporter = PerformanceReporter(self.metrics_collector, self.performance_analyzer)
        self.logger = logging.getLogger(__name__)

    def start_monitoring(self):
        """Start the performance monitoring system"""
        self.logger.info("Performance monitoring system started")
        
        # Initialize default alert rules
        self._setup_default_alerts()
        
        # Start alert checking (in a real implementation, this would be a background thread)
        self._check_alerts_periodically()

    def _setup_default_alerts(self):
        """Setup default alert rules"""
        self.alert_manager.add_alert_rule(
            metric_name="workflow_total_integration_time",
            threshold=10.0,
            operator=">",
            severity="warning",
            message="Integration time is high: {value}s > {threshold}s",
            component="integration"
        )
        
        self.alert_manager.add_alert_rule(
            metric_name="quality_data_quality_score",
            threshold=0.7,
            operator="<",
            severity="warning",
            message="Quality score is low: {value} < {threshold}",
            component="quality"
        )
        
        self.alert_manager.add_alert_rule(
            metric_name="throughput_components_per_second",
            threshold=0.5,
            operator="<",
            severity="critical",
            message="Throughput is critically low: {value} < {threshold}",
            component="performance"
        )

    def _check_alerts_periodically(self):
        """Periodically check for alerts (placeholder implementation)"""
        # In a real implementation, this would run in a background thread
        triggered_alerts = self.alert_manager.check_alerts()
        
        for alert in triggered_alerts:
            self.logger.warning(f"ALERT: {alert.message}")
            # In a real implementation, you might send notifications here

    def record_integration_performance(self, project_name: str, workflow_metrics: Dict[str, Any]):
        """Record performance metrics for an integration workflow"""
        self.metrics_collector.record_integration_workflow_metrics(project_name, workflow_metrics)
        
        # Check for alerts after recording
        self._check_alerts_periodically()

    def generate_performance_report(self, hours: int = 24) -> PerformanceReport:
        """Generate a performance report for the last N hours"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        return self.performance_reporter.generate_comprehensive_report((start_time, end_time))

    def get_current_alerts(self) -> List[PerformanceAlert]:
        """Get currently active alerts"""
        return self.alert_manager.get_active_alerts()

    def export_performance_data(self, hours: int = 24, file_path: str = None):
        """Export performance data to file"""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"performance_metrics_{timestamp}.json"
        
        time_window = timedelta(hours=hours)
        self.metrics_collector.export_metrics(file_path, time_window)
        
        return file_path


# Example usage and demonstration
def demo_performance_monitoring():
    """Demonstrate the performance monitoring system"""
    
    print("🔍 Starting Performance Monitoring System Demo")
    print("=" * 50)
    
    # Initialize monitoring system
    monitoring_system = PerformanceMonitoringSystem()
    monitoring_system.start_monitoring()
    
    # Simulate some integration workflows with performance data
    print("\n📊 Recording sample integration performance data...")
    
    # Sample workflow 1: Good performance
    workflow1_metrics = {
        "timing_metrics": {
            "total_integration_time": 3.2,
            "calculation_time": 1.1,
            "standards_validation_time": 0.8,
            "sld_integration_time": 0.9,
            "optimization_time": 0.4
        },
        "quality_metrics": {
            "data_quality_score": 0.92,
            "calculation_success_rate": 95.0,
            "standards_compliance_rate": 88.0,
            "optimization_suggestions": 3
        },
        "throughput_metrics": {
            "components_per_second": 2.1,
            "loads_processed_per_second": 1.5
        }
    }
    
    monitoring_system.record_integration_performance("Manufacturing_Plant_A", workflow1_metrics)
    
    # Sample workflow 2: Poor performance
    workflow2_metrics = {
        "timing_metrics": {
            "total_integration_time": 12.5,
            "calculation_time": 4.2,
            "standards_validation_time": 2.1,
            "sld_integration_time": 3.8,
            "optimization_time": 2.4
        },
        "quality_metrics": {
            "data_quality_score": 0.65,
            "calculation_success_rate": 72.0,
            "standards_compliance_rate": 68.0,
            "optimization_suggestions": 8
        },
        "throughput_metrics": {
            "components_per_second": 0.8,
            "loads_processed_per_second": 0.6
        }
    }
    
    monitoring_system.record_integration_performance("Complex_Distribution_B", workflow2_metrics)
    
    # Generate performance report
    print("\n📈 Generating performance report...")
    report = monitoring_system.generate_performance_report(hours=1)
    
    print(f"\n📋 Performance Report Summary:")
    print(f"   Report ID: {report.report_id}")
    print(f"   Period: {report.report_period[0]} to {report.report_period[1]}")
    print(f"   Total Metrics: {report.summary_metrics.get('total_metrics_collected', 0)}")
    
    # Display key metrics
    summary = report.summary_metrics
    if "avg_workflow_total_integration_time" in summary:
        print(f"   Avg Integration Time: {summary['avg_workflow_total_integration_time']:.2f}s")
    if "avg_quality_data_quality_score" in summary:
        print(f"   Avg Quality Score: {summary['avg_quality_data_quality_score']:.1%}")
    if "avg_throughput_components_per_second" in summary:
        print(f"   Avg Throughput: {summary['avg_throughput_components_per_second']:.2f} comp/s")
    
    # Display component performance
    print(f"\n🏗️ Component Performance:")
    for component, perf in report.component_performance.items():
        print(f"   {component}: {perf['total_operations']} ops, {perf['throughput_ops_per_second']:.2f} ops/s")
    
    # Display recommendations
    if report.recommendations:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Check for alerts
    active_alerts = monitoring_system.get_current_alerts()
    if active_alerts:
        print(f"\n🚨 Active Alerts ({len(active_alerts)}):")
        for alert in active_alerts:
            print(f"   [{alert.severity.upper()}] {alert.message}")
    else:
        print(f"\n✅ No active alerts")
    
    # Export data
    export_file = monitoring_system.export_performance_data(hours=1)
    print(f"\n💾 Performance data exported to: {export_file}")
    
    return monitoring_system


if __name__ == "__main__":
    # Run performance monitoring demo
    demo_performance_monitoring()