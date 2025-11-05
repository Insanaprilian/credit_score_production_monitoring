# credit_score_production_monitoring
Production-grade module to support credit score monitoring for one of the digital bank client, integrating PSI tracking and data drifting detection. 
This helps clients to remove manual monitoring and become part of regulatory monitoring

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Project Introduction
In production credit scoring, model performance/stability can degrade over time due to population drift, economic changes, or data quality issues. This framework provides automated monthly monitoring to detect stability issues before they impact business decisions.

## Business Problem
Credit models deployed in production face several challenges:

1. Population Drift: Customer demographics shift over time
2. Economic Changes: Macroeconomic conditions affect credit behavior
3. Data Quality: Pipeline changes can introduce anomalies
4. Regulatory Requirement: Need documented evidence of model stability

Without correct monitoring, these issues can drives to:
1. Inaccurate risk assessments
2. Poor lending decisions
3. Regulatory non-compliance
4. Reputational damage

## Solution
This module implements automatic Population Stability Index (PSI) based monitoring to:
1. Track feature and score distribution drift
2. Generate alerts when thresholds are breached
3. Produce executive reports and dashboards
4. Enable early detection of performance degradation

## Features in module
1. Automated PSI Calculation: Tracks both feature and score level stability
2. Configurable Thresholds: Customize alert levels based on risk appetite
3. Reporting: Executive summaries, technical reports, and visual dashboards
4. Batch Processing: Monitor multiple months simultaneously
5. Production-Ready: Includes error handling, logging, and data validation


## Architecture

```
┌─────────────────┐
│ Development     │ ← Baseline anchor (frozen)
│ Dataset         │
└────────┬────────┘
         │
         ↓
┌─────────────────────────────────────────┐
│  Credit Score Monitor                   │
│  ┌──────────────────────────────────┐  │
│  │ 1. Feature PSI Calculation       │  │
│  │ 2. Score Distribution PSI        │  │
│  │ 3. Drift Detection               │  │
│  │ 4. Alert Generation              │  │
│  └──────────────────────────────────┘  │
└──────────┬──────────────────────────────┘
           │
           ↓
    ┌──────┴──────┐
    │             │
    ↓             ↓
┌───────┐    ┌─────────┐
│Reports│    │Dashboard│
└───────┘    └─────────┘
```

## Methodology

### Population Stability Index (PSI)

PSI measures the shift in population distribution between two time periods:

```
PSI = Σ ((Actual% - Expected%) × ln(Actual% / Expected%))
```

**Interpretation:**
- `PSI < 0.1`: No significant change
- `0.1 ≤ PSI < 0.25`: Investigate and monitor
- `PSI ≥ 0.25`: Immediate action required

### Implementation Approach

1. Baseline Establishment: Use development/training dataset as anchor
2. Binning Strategy: 
   - Continuous variables: 10 quantile-based bins
   - Scores: 20 bins for granular tracking
3. Monthly Comparison: Compare production data against frozen baseline
4. Multi-level Monitoring: Track both individual features and overall score

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/credit-score-monitoring
cd credit-score-monitoring
pip install -r requirements.txt
```

### Basic Usage

```python
from credit_score_monitor import CreditScoreMonitor
import pandas as pd

# Load data
development_df = pd.read_csv('development_baseline.csv')
production_df = pd.read_csv('monthly_production.csv')

# Initialize monitor
monitor = CreditScoreMonitor(
    development_df=development_df,
    score_col='score_prediction',
    month_col='month'
)

# Monitor single month
results = monitor.monitor_monthly(
    production_df[production_df['month'] == '2024-11'],
    month_label='2024-11'
)

# Generate visual report
monitor.generate_report(results, output_path='report.png')
```

### Batch Monitoring

```python
# Monitor multiple months at once
summary_df = monitor.batch_monitor(production_df)
summary_df.to_csv('monitoring_summary.csv', index=False)

# Create dashboard
from monitoring_dashboard import MonitoringDashboard
dashboard = MonitoringDashboard()
dashboard.create_executive_summary(summary_df)
```

## Sample Output

### Monthly Monitoring Report
```
============================================================
MONITORING SUMMARY - 2024-11
============================================================

Records Monitored: 8,543

Score PSI: 0.0847 ✓

Features Monitored: 15
Features with Drift: 3
Drift Percentage: 20.0%

Top 5 Drifting Features:
1. credit_utilization       PSI: 0.1523
2. debt_to_income_ratio     PSI: 0.1247  
3. annual_income            PSI: 0.1089
4. age                      PSI: 0.0723
5. number_of_accounts       PSI: 0.0615

Overall Assessment: MONITORING REQUIRED
```

### Visual Dashboard
![Executive Dashboard](docs/images/executive_dashboard_sample.png)
