"""
Alerting and Reporting Module for Credit Score Monitoring
==========================================================
Generates automated alerts, email reports, and exports monitoring results.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


class MonitoringAlerts:
    """
    Manages alerting logic and thresholds for model monitoring.
    """
    
    def __init__(self, 
                 psi_moderate: float = 0.1,
                 psi_high: float = 0.25,
                 drift_pct_moderate: float = 20.0,
                 drift_pct_high: float = 50.0):
        """
        Initialize alerting system with configurable thresholds.
        
        Parameters:
        -----------
        psi_moderate : float
            PSI threshold for moderate drift warning
        psi_high : float
            PSI threshold for high drift alert
        drift_pct_moderate : float
            Percentage of features with drift for moderate warning
        drift_pct_high : float
            Percentage of features with drift for high alert
        """
        self.thresholds = {
            'psi_moderate': psi_moderate,
            'psi_high': psi_high,
            'drift_pct_moderate': drift_pct_moderate,
            'drift_pct_high': drift_pct_high
        }
        
        self.alert_history = []
    
    
    def evaluate_alerts(self, monitoring_results: Dict) -> List[Dict]:
        """
        Evaluate monitoring results and generate alerts.
        
        Parameters:
        -----------
        monitoring_results : Dict
            Results from CreditScoreMonitor.monitor_monthly()
        
        Returns:
        --------
        alerts : List[Dict]
            List of alert dictionaries
        """
        alerts = []
        month = monitoring_results['month']
        
        # 1. Check score PSI
        score_psi = monitoring_results.get('score_psi')
        if score_psi is not None:
            if score_psi >= self.thresholds['psi_high']:
                alerts.append({
                    'type': 'CRITICAL',
                    'category': 'Score Drift',
                    'month': month,
                    'metric': 'Score PSI',
                    'value': score_psi,
                    'threshold': self.thresholds['psi_high'],
                    'message': f"🚨 CRITICAL: Score PSI ({score_psi:.4f}) exceeds high threshold ({self.thresholds['psi_high']}). Immediate investigation required.",
                    'recommendation': "Review score distribution changes, check for data quality issues, consider model recalibration."
                })
            elif score_psi >= self.thresholds['psi_moderate']:
                alerts.append({
                    'type': 'WARNING',
                    'category': 'Score Drift',
                    'month': month,
                    'metric': 'Score PSI',
                    'value': score_psi,
                    'threshold': self.thresholds['psi_moderate'],
                    'message': f"⚠️  WARNING: Score PSI ({score_psi:.4f}) shows moderate drift. Monitor closely.",
                    'recommendation': "Continue monitoring, investigate if trend continues for 2+ months."
                })
        
        # 2. Check feature drift percentage
        drift_pct = (sum(monitoring_results['feature_drift_flags'].values()) / 
                    len(monitoring_results['feature_drift_flags']) * 100)
        
        if drift_pct >= self.thresholds['drift_pct_high']:
            alerts.append({
                'type': 'CRITICAL',
                'category': 'Feature Drift',
                'month': month,
                'metric': 'Drift Percentage',
                'value': drift_pct,
                'threshold': self.thresholds['drift_pct_high'],
                'message': f"🚨 CRITICAL: {drift_pct:.1f}% of features showing drift. Model stability compromised.",
                'recommendation': "Conduct comprehensive feature analysis, check data pipeline, prepare for model retraining."
            })
        elif drift_pct >= self.thresholds['drift_pct_moderate']:
            alerts.append({
                'type': 'WARNING',
                'category': 'Feature Drift',
                'month': month,
                'metric': 'Drift Percentage',
                'value': drift_pct,
                'threshold': self.thresholds['drift_pct_moderate'],
                'message': f"⚠️  WARNING: {drift_pct:.1f}% of features showing drift.",
                'recommendation': "Review drifting features, identify root causes, plan mitigation if needed."
            })
        
        # 3. Check individual high-impact features
        high_psi_features = {feat: psi for feat, psi in monitoring_results['feature_psi'].items()
                            if psi >= self.thresholds['psi_high']}
        
        if high_psi_features:
            top_3 = sorted(high_psi_features.items(), key=lambda x: x[1], reverse=True)[:3]
            feature_list = ', '.join([f"{feat} (PSI: {psi:.3f})" for feat, psi in top_3])
            
            alerts.append({
                'type': 'WARNING',
                'category': 'Feature Drift',
                'month': month,
                'metric': 'High PSI Features',
                'value': len(high_psi_features),
                'threshold': self.thresholds['psi_high'],
                'message': f"⚠️  WARNING: {len(high_psi_features)} feature(s) with high PSI. Top: {feature_list}",
                'recommendation': "Investigate these features for data quality issues or population changes."
            })
        
        # 4. Check volume anomalies
        record_count = monitoring_results['record_count']
        if record_count < 1000:  # Example threshold
            alerts.append({
                'type': 'INFO',
                'category': 'Volume',
                'month': month,
                'metric': 'Record Count',
                'value': record_count,
                'threshold': 1000,
                'message': f"ℹ️  INFO: Low volume detected ({record_count:,} records). May affect statistical significance.",
                'recommendation': "Verify data completeness, check for collection issues."
            })
        
        # Store alerts in history
        self.alert_history.extend(alerts)
        
        return alerts
    
    
    def generate_alert_summary(self, alerts: List[Dict]) -> str:
        """
        Generate formatted alert summary text.
        
        Parameters:
        -----------
        alerts : List[Dict]
            List of alert dictionaries
        
        Returns:
        --------
        summary : str
            Formatted alert summary
        """
        if not alerts:
            return "✓ No alerts triggered. Model monitoring within acceptable thresholds."
        
        critical = [a for a in alerts if a['type'] == 'CRITICAL']
        warnings = [a for a in alerts if a['type'] == 'WARNING']
        info = [a for a in alerts if a['type'] == 'INFO']
        
        summary = f"""
{'='*80}
MONITORING ALERT SUMMARY - {alerts[0]['month']}
{'='*80}

Total Alerts: {len(alerts)} (Critical: {len(critical)}, Warning: {len(warnings)}, Info: {len(info)})

"""
        
        if critical:
            summary += "🚨 CRITICAL ALERTS:\n"
            summary += "-" * 80 + "\n"
            for i, alert in enumerate(critical, 1):
                summary += f"{i}. {alert['message']}\n"
                summary += f"   → Recommendation: {alert['recommendation']}\n\n"
        
        if warnings:
            summary += "⚠️  WARNING ALERTS:\n"
            summary += "-" * 80 + "\n"
            for i, alert in enumerate(warnings, 1):
                summary += f"{i}. {alert['message']}\n"
                summary += f"   → Recommendation: {alert['recommendation']}\n\n"
        
        if info:
            summary += "ℹ️  INFORMATION:\n"
            summary += "-" * 80 + "\n"
            for i, alert in enumerate(info, 1):
                summary += f"{i}. {alert['message']}\n"
                summary += f"   → Recommendation: {alert['recommendation']}\n\n"
        
        summary += "="*80 + "\n"
        
        return summary
    
    
    def export_alerts_to_csv(self, output_path: str) -> None:
        """
        Export alert history to CSV file.
        
        Parameters:
        -----------
        output_path : str
            Path to save CSV file
        """
        if not self.alert_history:
            print("No alerts to export.")
            return
        
        df = pd.DataFrame(self.alert_history)
        df.to_csv(output_path, index=False)
        print(f"✓ Alerts exported to: {output_path}")
    
    
    def get_alert_statistics(self) -> pd.DataFrame:
        """
        Get statistics on alert history.
        
        Returns:
        --------
        stats : pd.DataFrame
            Alert statistics by type and category
        """
        if not self.alert_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.alert_history)
        
        stats = df.groupby(['type', 'category']).agg({
            'month': 'count',
            'value': ['mean', 'min', 'max']
        }).round(4)
        
        stats.columns = ['count', 'avg_value', 'min_value', 'max_value']
        stats = stats.reset_index()
        
        return stats


class MonitoringReporter:
    """
    Generates comprehensive monitoring reports.
    """
    
    def __init__(self):
        self.report_templates = {
            'monthly': self._monthly_report_template,
            'executive': self._executive_report_template,
            'technical': self._technical_report_template
        }
    
    
    def generate_monthly_report(self, monitoring_results: Dict, 
                               alerts: List[Dict],
                               output_path: Optional[str] = None) -> str:
        """
        Generate monthly monitoring report.
        
        Parameters:
        -----------
        monitoring_results : Dict
            Results from CreditScoreMonitor.monitor_monthly()
        alerts : List[Dict]
            List of alerts from MonitoringAlerts.evaluate_alerts()
        output_path : str, optional
            Path to save report
        
        Returns:
        --------
        report : str
            Formatted report text
        """
        report = self._monthly_report_template(monitoring_results, alerts)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"✓ Monthly report saved to: {output_path}")
        
        return report
    
    
    def _monthly_report_template(self, results: Dict, alerts: List[Dict]) -> str:
        """Template for monthly monitoring report."""
        
        month = results['month']
        record_count = results['record_count']
        score_psi = results.get('score_psi', 'N/A')
        
        # Calculate summary metrics
        total_features = len(results['feature_psi'])
        drifting_features = sum(results['feature_drift_flags'].values())
        drift_pct = (drifting_features / total_features * 100) if total_features > 0 else 0
        
        # Get top drifting features
        top_drift = sorted(results['feature_psi'].items(), 
                          key=lambda x: x[1], reverse=True)[:5]
        
        # Alert summary
        critical_alerts = len([a for a in alerts if a['type'] == 'CRITICAL'])
        warning_alerts = len([a for a in alerts if a['type'] == 'WARNING'])
        
        report = f"""
{'='*80}
CREDIT SCORE MODEL MONITORING REPORT
{'='*80}

Report Period: {month}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
EXECUTIVE SUMMARY
{'='*80}

Records Monitored: {record_count:,}
Score PSI: {score_psi if isinstance(score_psi, str) else f'{score_psi:.4f}'}
Features Monitored: {total_features}
Features with Drift: {drifting_features} ({drift_pct:.1f}%)

Alert Status:
  🚨 Critical: {critical_alerts}
  ⚠️  Warning: {warning_alerts}
  
Overall Assessment: {'🚨 REQUIRES ATTENTION' if critical_alerts > 0 else '⚠️  MONITORING REQUIRED' if warning_alerts > 0 else '✓ STABLE'}

{'='*80}
SCORE DISTRIBUTION ANALYSIS
{'='*80}

Score PSI: {score_psi if isinstance(score_psi, str) else f'{score_psi:.4f}'}
Status: {'🚨 High Drift' if isinstance(score_psi, float) and score_psi >= 0.25 else '⚠️  Moderate Drift' if isinstance(score_psi, float) and score_psi >= 0.1 else '✓ Stable'}

The score distribution PSI measures the stability of the model's output.
A PSI < 0.1 indicates stable predictions, while PSI > 0.25 suggests
significant distribution shift requiring investigation.

{'='*80}
FEATURE DRIFT ANALYSIS
{'='*80}

Top 5 Features by PSI:
"""
        
        for i, (feature, psi) in enumerate(top_drift, 1):
            status = "🚨" if psi >= 0.25 else "⚠️" if psi >= 0.1 else "✓"
            report += f"\n{i}. {feature:<30} PSI: {psi:.4f}  {status}"
        
        report += f"""

Drift Distribution:
  Stable (PSI < 0.1):        {sum(1 for p in results['feature_psi'].values() if p < 0.1):>3} features
  Moderate (0.1 ≤ PSI < 0.25): {sum(1 for p in results['feature_psi'].values() if 0.1 <= p < 0.25):>3} features
  High (PSI ≥ 0.25):          {sum(1 for p in results['feature_psi'].values() if p >= 0.25):>3} features

{'='*80}
ALERT DETAILS
{'='*80}
"""
        
        if not alerts:
            report += "\n✓ No alerts triggered. All monitoring metrics within acceptable thresholds.\n"
        else:
            for alert in alerts:
                report += f"\n[{alert['type']}] {alert['category']}\n"
                report += f"  {alert['message']}\n"
                report += f"  Recommendation: {alert['recommendation']}\n"
        
        report += f"""
{'='*80}
RECOMMENDATIONS
{'='*80}

"""
        
        if critical_alerts > 0:
            report += """
🚨 IMMEDIATE ACTION REQUIRED:
1. Investigate root causes of critical drift
2. Review data pipeline for quality issues
3. Assess need for model recalibration or retraining
4. Schedule emergency review with stakeholders
"""
        elif warning_alerts > 0:
            report += """
⚠️  MONITORING RECOMMENDED:
1. Continue close monitoring for next 2-3 months
2. Investigate warning features for trends
3. Prepare contingency plan if drift continues
4. Document findings for monthly review
"""
        else:
            report += """
✓ CONTINUE ROUTINE MONITORING:
1. Maintain current monitoring schedule
2. Review monthly trends
3. Update monitoring documentation
4. No immediate action required
"""
        
        report += f"""
{'='*80}
TECHNICAL DETAILS
{'='*80}

PSI Calculation Method: Quantile-based binning (10 bins for features, 20 for scores)
Development Baseline: Used as anchor for all PSI calculations
Thresholds: PSI Moderate=0.1, PSI High=0.25

For detailed feature-level analysis and visualizations, please refer to the
accompanying dashboard and technical appendix.

{'='*80}
END OF REPORT
{'='*80}
"""
        
        return report
    
    
    def generate_executive_summary(self, summary_df: pd.DataFrame,
                                   output_path: Optional[str] = None) -> str:
        """
        Generate executive summary for multiple months.
        
        Parameters:
        -----------
        summary_df : pd.DataFrame
            Summary dataframe from CreditScoreMonitor.batch_monitor()
        output_path : str, optional
            Path to save summary
        
        Returns:
        --------
        summary : str
            Formatted executive summary
        """
        start_month = summary_df['month'].iloc[0]
        end_month = summary_df['month'].iloc[-1]
        total_months = len(summary_df)
        
        # Calculate trends
        avg_score_psi = summary_df['score_psi'].mean()
        max_score_psi = summary_df['score_psi'].max()
        avg_drift_pct = summary_df['drift_percentage'].mean()
        max_drift_pct = summary_df['drift_percentage'].max()
        
        # Count months with issues
        months_with_score_drift = sum(summary_df['score_psi'] >= 0.1)
        months_with_high_feature_drift = sum(summary_df['drift_percentage'] > 20)
        
        summary = f"""
{'='*80}
EXECUTIVE SUMMARY: MODEL MONITORING OVERVIEW
{'='*80}

Period: {start_month} to {end_month} ({total_months} months)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
KEY FINDINGS
{'='*80}

Score Stability:
  Average Score PSI:    {avg_score_psi:.4f}
  Maximum Score PSI:    {max_score_psi:.4f}
  Months with Drift:    {months_with_score_drift}/{total_months}

Feature Stability:
  Average Drift %:      {avg_drift_pct:.1f}%
  Maximum Drift %:      {max_drift_pct:.1f}%
  Months with >20% Drift: {months_with_high_feature_drift}/{total_months}

{'='*80}
OVERALL ASSESSMENT
{'='*80}
"""
        
        if max_score_psi >= 0.25 or max_drift_pct > 50:
            summary += """
🚨 MODEL REQUIRES ATTENTION
- Significant drift detected in monitoring period
- Recommend comprehensive model review and potential retraining
- Schedule stakeholder meeting to discuss intervention strategy
"""
        elif avg_score_psi >= 0.1 or avg_drift_pct > 20:
            summary += """
⚠️  ENHANCED MONITORING RECOMMENDED
- Moderate drift trends observed
- Continue close monitoring with monthly reviews
- Prepare contingency plans for model refresh
"""
        else:
            summary += """
✓ MODEL PERFORMING WITHIN ACCEPTABLE PARAMETERS
- Drift metrics within normal ranges
- Continue routine quarterly reviews
- No immediate action required
"""
        
        summary += f"""
{'='*80}
MONTHLY BREAKDOWN
{'='*80}

"""
        
        for _, row in summary_df.iterrows():
            status = "🚨" if row['score_psi'] >= 0.25 or row['drift_percentage'] > 50 else \
                     "⚠️" if row['score_psi'] >= 0.1 or row['drift_percentage'] > 20 else "✓"
            summary += f"{row['month']}: Score PSI={row['score_psi']:.3f}, Drift={row['drift_percentage']:.1f}%  {status}\n"
        
        summary += f"\n{'='*80}\n"
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(summary)
            print(f"✓ Executive summary saved to: {output_path}")
        
        return summary
    
    
    def export_monitoring_data(self, summary_df: pd.DataFrame,
                              output_dir: str,
                              format: str = 'csv') -> None:
        """
        Export monitoring data in various formats.
        
        Parameters:
        -----------
        summary_df : pd.DataFrame
            Summary dataframe to export
        output_dir : str
            Directory to save exports
        format : str
            Export format ('csv', 'excel', 'json')
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'csv':
            output_path = Path(output_dir) / f'monitoring_summary_{timestamp}.csv'
            summary_df.to_csv(output_path, index=False)
            print(f"✓ Data exported to CSV: {output_path}")
        
        elif format == 'excel':
            output_path = Path(output_dir) / f'monitoring_summary_{timestamp}.xlsx'
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Add PSI detail sheet
                psi_cols = [col for col in summary_df.columns if col.startswith('psi_')]
                psi_data = summary_df[['month'] + psi_cols]
                psi_data.to_excel(writer, sheet_name='Feature PSI', index=False)
            
            print(f"✓ Data exported to Excel: {output_path}")
        
        elif format == 'json':
            output_path = Path(output_dir) / f'monitoring_summary_{timestamp}.json'
            summary_df.to_json(output_path, orient='records', indent=2)
            print(f"✓ Data exported to JSON: {output_path}")
        
        else:
            print(f"Unsupported format: {format}")


# Example usage
def example_alerting_workflow():
    """
    Demonstrate alerting and reporting workflow.
    """
    # Simulate monitoring results
    np.random.seed(42)
    
    monitoring_results = {
        'month': '2024-11',
        'record_count': 8543,
        'score_psi': 0.285,  # High drift
        'feature_psi': {
            'age': 0.08,
            'income': 0.32,  # High drift
            'credit_utilization': 0.15,
            'loan_tenure': 0.05,
            'payment_history': 0.22,
            'debt_to_income': 0.28  # High drift
        },
        'feature_drift_flags': {
            'age': False,
            'income': True,
            'credit_utilization': True,
            'loan_tenure': False,
            'payment_history': True,
            'debt_to_income': True
        },
        'score_drift_flag': True,
        'summary_stats': pd.DataFrame()
    }
    
    # Initialize alerting
    alert_system = MonitoringAlerts()
    
    # Evaluate alerts
    alerts = alert_system.evaluate_alerts(monitoring_results)
    
    # Print alert summary
    print(alert_system.generate_alert_summary(alerts))
    
    # Generate report
    reporter = MonitoringReporter()
    report = reporter.generate_monthly_report(monitoring_results, alerts)
    print(report)
    
    # Export alerts
    # alert_system.export_alerts_to_csv('alerts_history.csv')
    
    print("\n✓ Alerting workflow completed successfully!")
    
    return alert_system, reporter


if __name__ == "__main__":
    print("Monitoring Alerting and Reporting Module")
    print("=" * 80)
    alert_system, reporter = example_alerting_workflow()