# credit_score_production_monitoring
module to support credit score performance stability for one of digital bank client, integrating PSI tracking and data drifting visualization. The module removed manual monitoring time and became a part of regulatory monitoring

## Project Introduction
In production credit scoring, model performance can degrade over time due to population drift, economic changes, or data quality issues. This framework provides automated monthly monitoring to detect stability issues before they impact business decisions.

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
This repos implement automatic Population Stability Index (PSI) based monitoring to:
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
