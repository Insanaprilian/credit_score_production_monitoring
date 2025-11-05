# User Guide

## Getting Started

This guide will walk you through using the Credit Score Monitoring Framework from installation to generating your first monitoring report.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Data Preparation](#data-preparation)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Customization](#customization)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

---

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/credit-score-monitoring
cd credit-score-monitoring

# Install required packages
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.7.0
```

### Verify Installation

```python
from src.credit_score_monitor import CreditScoreMonitor
print("✓ Installation successful!")
```

---

## Quick Start

### 5-Minute Tutorial

```python
import pandas as pd
from src.credit_score_monitor import CreditScoreMonitor

# 1. Load your data
development_df = pd.read_csv('development_baseline.csv')
production_df = pd.read_csv('production_november.csv')

# 2. Initialize monitor
monitor = CreditScoreMonitor(
    development_df=development_df,
    score_col='score_prediction',
    month_col='month'
)

# 3. Run monitoring
results = monitor.monitor_monthly(production_df, '2024-11')

# 4. Generate report
monitor.generate_report(results, output_path='report_nov_2024.png')

print("✓ Monitoring complete! Check report_nov_2024.png")
```

---

## Data Preparation

### Required Data Format

Your data must include:

1. **Development/Baseline Dataset**
   - Historical data used for model training
   - Should be representative of "normal" conditions
   - Recommended: 10,000+ records

2. **Production Dataset**
   - Current month's scoring data
   - Must have same features as development
   - Include month identifier column

### Data Structure

```python
# Development DataFrame
development_df = pd.DataFrame({
    'customer_id': [1, 2, 3, ...],           # Unique identifier
    'age': [35, 42, 28, ...],                # Numeric features
    'income': [50000, 75000, 45000, ...],
    'credit_utilization': [0.3, 0.5, 0.2, ...],
    'employment_type': ['Full', 'Part', ...], # Categorical features
    'score_prediction': [680, 720, 650, ...], # Model output
    'month': ['development', 'development', ...]
})

# Production DataFrame (same structure)
production_df = pd.DataFrame({
    'customer_id': [10001, 10002, ...],
    'age': [37, 44, ...],
    'income': [52000, 78000, ...],
    # ... same columns as development
    'score_prediction': [685, 725, ...],
    'month': ['2024-11', '2024-11', ...]
})
```

### Data Quality Checks

Before monitoring, ensure:

```python
# 1. Check for required columns
required_cols = ['score_prediction', 'month']
assert all(col in development_df.columns for col in required_cols)

# 2. Check for missing values
print(f"Missing values: {development_df.isnull().sum().sum()}")

# 3. Verify data types
print(development_df.dtypes)

# 4. Check score range
assert development_df['score_prediction'].between(300, 850).all()

# 5. Verify consistent features
assert set(development_df.columns) == set(production_df.columns)
```

---

## Basic Usage

### Single Month Monitoring

```python
from src.credit_score_monitor import CreditScoreMonitor

# Initialize
monitor = CreditScoreMonitor(
    development_df=dev_df,
    score_col='score_prediction',
    month_col='month'
)

# Monitor specific month
november_data = prod_df[prod_df['month'] == '2024-11']
results = monitor.monitor_monthly(november_data, '2024-11')

# Print summary
print(f"Score PSI: {results['score_psi']:.4f}")
print(f"Features with drift: {sum(results['feature_drift_flags'].values())}")
```

### Batch Monitoring (Multiple Months)

```python
# Monitor all months at once
summary_df = monitor.batch_monitor(prod_df)

# View summary
print(summary_df[['month', 'score_psi', 'drift_percentage']])

# Export results
summary_df.to_csv('monitoring_summary.csv', index=False)
```

### Generate Visual Report

```python
# For single month
monitor.generate_report(
    results, 
    output_path='reports/november_2024_report.png'
)

# Create dashboard for multiple months
from src.monitoring_dashboard import MonitoringDashboard

dashboard = MonitoringDashboard()
dashboard.create_executive_summary(summary_df)
dashboard.plot_psi_trends(summary_df)
dashboard.plot_drift_heatmap(summary_df)
```

---

## Advanced Features

### Custom Feature Selection

```python
# Monitor specific features only
monitor = CreditScoreMonitor(
    development_df=dev_df,
    score_col='score_prediction',
    month_col='month',
    feature_cols=['age', 'income', 'credit_utilization']  # Specify features
)
```

### Alerting System

```python
from src.monitoring_alerts import MonitoringAlerts

# Initialize with custom thresholds
alert_system = MonitoringAlerts(
    psi_moderate=0.08,      # Lower threshold = more sensitive
    psi_high=0.20,
    drift_pct_moderate=15.0,
    drift_pct_high=40.0
)

# Evaluate alerts
alerts = alert_system.evaluate_alerts(results)

# Print alert summary
print(alert_system.generate_alert_summary(alerts))

# Export alerts
alert_system.export_alerts_to_csv('alerts/november_2024.csv')
```

### Custom Reports

```python
from src.monitoring_alerts import MonitoringReporter

reporter = MonitoringReporter()

# Generate monthly report
monthly_report = reporter.generate_monthly_report(
    results, 
    alerts,
    output_path='reports/monthly_report_nov_2024.txt'
)

# Generate executive summary (multi-month)
exec_summary = reporter.generate_executive_summary(
    summary_df,
    output_path='reports/executive_summary_q4_2024.txt'
)

# Export in multiple formats
reporter.export_monitoring_data(
    summary_df,
    output_dir='exports',
    format='excel'  # Options: 'csv', 'excel', 'json'
)
```

---

## Customization

### Adjust PSI Bins

```python
# Modify binning strategy for finer granularity
def custom_psi_calculation(monitor, expected, actual):
    psi, detail = monitor.calculate_psi(
        expected, 
        actual, 
        bins=20  # Default is 10 for features, 20 for scores
    )
    return psi, detail
```

### Custom Visualization Theme

```python
from src.monitoring_dashboard import MonitoringDashboard

dashboard = MonitoringDashboard()

# Customize colors
dashboard.colors = {
    'stable': '#00b894',      # Custom green
    'moderate': '#fdcb6e',    # Custom yellow
    'high': '#d63031',        # Custom red
    'primary': '#0984e3',
    'secondary': '#6c5ce7'
}

# Create dashboard with custom theme
dashboard.create_executive_summary(summary_df)
```

### Integrate with Existing Pipeline

```python
# Example: Airflow DAG
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def run_monthly_monitoring(**context):
    from src.credit_score_monitor import CreditScoreMonitor
    
    # Load data from your warehouse
    dev_df = load_from_warehouse('development_baseline')
    prod_df = load_from_warehouse('production_data', 
                                  month=context['execution_date'])
    
    # Run monitoring
    monitor = CreditScoreMonitor(dev_df)
    results = monitor.monitor_monthly(prod_df, context['execution_date'])
    
    # Send alerts if needed
    if results['score_psi'] >= 0.25:
        send_email_alert(results)
    
    return results

dag = DAG(
    'credit_score_monitoring',
    default_args={
        'owner': 'data-science',
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='Monthly credit score monitoring',
    schedule_interval='0 1 1 * *',  # 1st of every month at 1 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

monitoring_task = PythonOperator(
    task_id='run_monitoring',
    python_callable=run_monthly_monitoring,
    dag=dag,
)
```

---

## Troubleshooting

### Common Issues

#### 1. "Column not found" Error

```python
# Problem: Column name mismatch
KeyError: 'score_prediction'

# Solution: Check exact column names
print(df.columns.tolist())

# Rename if needed
df = df.rename(columns={'model_score': 'score_prediction'})
```

#### 2. PSI Calculation Returns NaN

```python
# Problem: Too many missing values
# Solution: Check data quality

# Identify problematic features
for col in df.columns:
    missing_pct = df[col].isnull().mean() * 100
    if missing_pct > 50:
        print(f"{col}: {missing_pct:.1f}% missing")

# Handle missing values
df = df.dropna(subset=['score_prediction'])  # Drop rows with missing scores
df['income'] = df['income'].fillna(df['income'].median())  # Impute features
```

#### 3. Memory Error with Large Datasets

```python
# Problem: Dataset too large for memory
MemoryError

# Solution: Process in chunks
chunk_size = 100000
results_list = []

for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    result = monitor.monitor_monthly(chunk, '2024-11')
    results_list.append(result)

# Aggregate results
final_results = aggregate_chunk_results(results_list)
```

#### 4. Visualization Not Displaying

```python
# Problem: Matplotlib backend issue
# Solution: Set backend explicitly

import matplotlib
matplotlib.use('Agg')  # For non-interactive backend
import matplotlib.pyplot as plt

# Then generate plots
monitor.generate_report(results, output_path='report.png')
```

---

## FAQ

### Q1: How often should I run monitoring?

**A:** 
- **Standard**: Monthly for most credit models
- **High-risk**: Weekly during volatile periods
- **Post-deployment**: Daily for first 2-3 months

### Q2: When should I update the development baseline?

**A:**
- **Rare**: Only with governance approval
- **Triggers**: Model retrained, major business change, regulatory update
- **Process**: Document rationale, version control, stakeholder signoff

### Q3: What PSI value means I must retrain?

**A:**
- PSI **≥ 0.25**: Immediate investigation required
- Consider retraining if:
  - PSI > 0.25 for 2+ consecutive months
  - Multiple critical features drifting
  - Actual performance metrics declining

### Q4: Can I use this for non-credit models?

**A:** Yes! PSI is applicable to any classification or regression model:
- Fraud detection
- Churn prediction
- Risk scoring
- Demand forecasting

Just ensure you have clear baseline and production datasets.

### Q5: How do I handle categorical feature drift?

**A:**
```python
# Categorical features are automatically handled
monitor = CreditScoreMonitor(development_df)

# PSI treats each category as a "bin"
categorical_psi, detail = monitor.calculate_psi(
    dev_df['employment_type'],
    prod_df['employment_type'],
    categorical=True  # Explicitly mark as categorical
)
```

### Q6: What if my production data has new categories?

**A:** The framework handles this by:
1. Creating union of all categories
2. Assigning small probability (0.0001) to missing categories
3. New categories contribute to PSI, indicating drift

### Q7: How do I export results for stakeholders?

**A:**
```python
from src.monitoring_alerts import MonitoringReporter

reporter = MonitoringReporter()

# Excel for business users
reporter.export_monitoring_data(summary_df, 'reports/', format='excel')

# CSV for data team
reporter.export_monitoring_data(summary_df, 'reports/', format='csv')

# Text report for email
report_text = reporter.generate_monthly_report(results, alerts)
send_email(recipients=['stakeholders@company.com'], body=report_text)
```

### Q8: Can I monitor model performance, not just drift?

**A:** This framework focuses on **drift detection** (input monitoring). For performance monitoring (output quality):

```python
# Add performance metrics to your workflow
from sklearn.metrics import roc_auc_score, accuracy_score

# If you have ground truth labels
if 'actual_default' in prod_df.columns:
    auc = roc_auc_score(prod_df['actual_default'], 
                       prod_df['score_prediction'])
    print(f"Production AUC: {auc:.3f}")
```

### Q9: How do I integrate with MLflow/MLOps platforms?

**A:**
```python
import mlflow

with mlflow.start_run():
    # Run monitoring
    results = monitor.monitor_monthly(prod_df, '2024-11')
    
    # Log metrics
    mlflow.log_metric("score_psi", results['score_psi'])
    mlflow.log_metric("drift_percentage", 
                     sum(results['feature_drift_flags'].values()) / 
                     len(results['feature_drift_flags']) * 100)
    
    # Log artifacts
    monitor.generate_report(results, 'temp_report.png')
    mlflow.log_artifact('temp_report.png')
```

---

## Best Practices

### 1. Establish Monitoring Cadence
- Set up automated monthly runs
- Calendar reminders for manual review
- Document monitoring schedule

### 2. Create Response Playbook
- Define actions for each alert level
- Assign responsibilities
- Set response time SLAs

### 3. Maintain Audit Trail
```python
# Log every monitoring run
import logging

logging.basicConfig(
    filename='monitoring_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

logging.info(f"Monitoring run: {results['month']}, PSI: {results['score_psi']}")
```

### 4. Regular Threshold Review
- Quarterly review of thresholds
- Adjust based on model performance
- Document threshold changes

### 5. Stakeholder Communication
- Monthly summary emails
- Quarterly deep dives
- Annual strategy review

---

## Support

### Getting Help

- **Documentation**: Check README.md and docs/ folder
- **Issues**: Open GitHub issue with:
  - Error message
  - Minimal reproducible example
  - Environment details

### Contributing

See CONTRIBUTING.md for:
- Code style guidelines
- Pull request process
- Development setup

---

*Last Updated: November 2024*
*User Guide Version: 1.0.0*