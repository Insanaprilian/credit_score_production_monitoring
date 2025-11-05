"""
Credit Scoring Model Monthly Monitoring Framework
==================================================
Monitors model stability through PSI calculation, drift detection, and performance tracking.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CreditScoreMonitor:
    """
    Main monitoring class for credit scoring models.
    Tracks feature drift, score distribution changes, and calculates PSI.
    """
    
    def __init__(self, development_df: pd.DataFrame, score_col: str = 'score_prediction', 
                 month_col: str = 'month', feature_cols: Optional[List[str]] = None):
        """
        Initialize the monitoring framework.
        
        Parameters:
        -----------
        development_df : pd.DataFrame
            Baseline/development dataset used as anchor for comparison
        score_col : str
            Column name for score predictions
        month_col : str
            Column name for month identifier
        feature_cols : List[str], optional
            List of feature columns to monitor. If None, auto-detect.
        """
        self.development_df = development_df.copy()
        self.score_col = score_col
        self.month_col = month_col
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            self.feature_cols = [col for col in development_df.columns 
                                if col not in [score_col, month_col]]
        else:
            self.feature_cols = feature_cols
        
        print(f"✓ Initialized monitoring for {len(self.feature_cols)} features")
        print(f"✓ Development baseline: {len(development_df)} records")
    
    
    def calculate_psi(self, expected: pd.Series, actual: pd.Series, 
                     bins: int = 10, categorical: bool = False) -> Tuple[float, pd.DataFrame]:
        """
        Calculate Population Stability Index (PSI).
        
        PSI Formula: Σ((%Actual - %Expected) * ln(%Actual / %Expected))
        
        PSI Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 ≤ PSI < 0.25: Moderate change, investigate
        - PSI ≥ 0.25: Significant change, model may need retraining
        
        Parameters:
        -----------
        expected : pd.Series
            Development/baseline population
        actual : pd.Series
            Current month population
        bins : int
            Number of bins for continuous variables
        categorical : bool
            Whether the variable is categorical
        
        Returns:
        --------
        psi_value : float
            PSI score
        psi_detail : pd.DataFrame
            Detailed PSI calculation by bin
        """
        # Remove missing values
        expected = expected.dropna()
        actual = actual.dropna()
        
        if len(expected) == 0 or len(actual) == 0:
            return np.nan, pd.DataFrame()
        
        if categorical or expected.dtype == 'object':
            # Categorical variable
            expected_counts = expected.value_counts(normalize=True)
            actual_counts = actual.value_counts(normalize=True)
            
            # Align categories
            all_categories = set(expected_counts.index) | set(actual_counts.index)
            expected_pct = pd.Series({cat: expected_counts.get(cat, 0.0001) for cat in all_categories})
            actual_pct = pd.Series({cat: actual_counts.get(cat, 0.0001) for cat in all_categories})
            
        else:
            # Continuous variable - use development quantiles
            try:
                _, bin_edges = pd.qcut(expected, q=bins, retbins=True, duplicates='drop')
            except:
                # If qcut fails, use regular binning
                bin_edges = np.linspace(expected.min(), expected.max(), bins + 1)
            
            # Ensure bins cover both distributions
            bin_edges[0] = min(expected.min(), actual.min()) - 0.001
            bin_edges[-1] = max(expected.max(), actual.max()) + 0.001
            
            expected_binned = pd.cut(expected, bins=bin_edges, include_lowest=True)
            actual_binned = pd.cut(actual, bins=bin_edges, include_lowest=True)
            
            expected_pct = expected_binned.value_counts(normalize=True)
            actual_pct = actual_binned.value_counts(normalize=True)
            
            # Align bins
            all_bins = expected_pct.index.union(actual_binned.value_counts(normalize=True).index)
            expected_pct = expected_pct.reindex(all_bins, fill_value=0.0001)
            actual_pct = actual_pct.reindex(all_bins, fill_value=0.0001)
        
        # Replace zeros with small value to avoid log(0)
        expected_pct = expected_pct.replace(0, 0.0001)
        actual_pct = actual_pct.replace(0, 0.0001)
        
        # Calculate PSI
        psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        psi_value = psi_values.sum()
        
        # Create detailed breakdown
        psi_detail = pd.DataFrame({
            'bin': expected_pct.index.astype(str),
            'expected_pct': expected_pct.values * 100,
            'actual_pct': actual_pct.values * 100,
            'psi_contribution': psi_values.values
        })
        
        return psi_value, psi_detail
    
    
    def monitor_monthly(self, current_month_df: pd.DataFrame, 
                       month_label: str) -> Dict:
        """
        Perform monthly monitoring on a single month's data.
        
        Parameters:
        -----------
        current_month_df : pd.DataFrame
            Current month's data to monitor
        month_label : str
            Label for the current month (e.g., '2024-11')
        
        Returns:
        --------
        results : Dict
            Dictionary containing all monitoring metrics
        """
        results = {
            'month': month_label,
            'record_count': len(current_month_df),
            'feature_psi': {},
            'score_psi': None,
            'feature_drift_flags': {},
            'score_drift_flag': False,
            'summary_stats': {}
        }
        
        print(f"\n{'='*60}")
        print(f"Monitoring Month: {month_label}")
        print(f"{'='*60}")
        print(f"Records: {len(current_month_df):,}")
        
        # 1. Calculate PSI for each feature
        print(f"\n📊 Feature PSI Analysis:")
        print(f"{'Feature':<30} {'PSI':>8} {'Status'}")
        print("-" * 60)
        
        for feature in self.feature_cols:
            if feature not in current_month_df.columns:
                continue
            
            is_categorical = (self.development_df[feature].dtype == 'object' or 
                            self.development_df[feature].nunique() < 10)
            
            psi_value, _ = self.calculate_psi(
                self.development_df[feature],
                current_month_df[feature],
                categorical=is_categorical
            )
            
            results['feature_psi'][feature] = psi_value
            
            # Flag based on PSI thresholds
            if pd.isna(psi_value):
                status = "⚠️  Missing Data"
                flag = True
            elif psi_value < 0.1:
                status = "✓ Stable"
                flag = False
            elif psi_value < 0.25:
                status = "⚠️  Moderate Drift"
                flag = True
            else:
                status = "🚨 High Drift"
                flag = True
            
            results['feature_drift_flags'][feature] = flag
            print(f"{feature:<30} {psi_value:>8.4f} {status}")
        
        # 2. Calculate PSI for score distribution
        if self.score_col in current_month_df.columns:
            score_psi, score_detail = self.calculate_psi(
                self.development_df[self.score_col],
                current_month_df[self.score_col],
                bins=20  # More granular for scores
            )
            
            results['score_psi'] = score_psi
            results['score_drift_flag'] = score_psi >= 0.1
            results['score_psi_detail'] = score_detail
            
            print(f"\n🎯 Score Distribution PSI: {score_psi:.4f}")
            if score_psi < 0.1:
                print("   Status: ✓ Stable")
            elif score_psi < 0.25:
                print("   Status: ⚠️  Moderate Drift")
            else:
                print("   Status: 🚨 High Drift - Review Required")
        
        # 3. Summary statistics comparison
        results['summary_stats'] = self._compare_summary_stats(current_month_df)
        
        # 4. Overall assessment
        drift_count = sum(results['feature_drift_flags'].values())
        total_features = len(results['feature_drift_flags'])
        drift_pct = (drift_count / total_features * 100) if total_features > 0 else 0
        
        print(f"\n📈 Overall Assessment:")
        print(f"   Features with drift: {drift_count}/{total_features} ({drift_pct:.1f}%)")
        print(f"   Score drift detected: {'Yes' if results['score_drift_flag'] else 'No'}")
        
        if drift_pct > 50 or results['score_drift_flag']:
            print(f"   ⚠️  ACTION REQUIRED: Investigate model stability")
        elif drift_pct > 20:
            print(f"   ℹ️  MONITOR: Some features showing drift")
        else:
            print(f"   ✓ Model appears stable")
        
        return results
    
    
    def _compare_summary_stats(self, current_df: pd.DataFrame) -> pd.DataFrame:
        """Compare summary statistics between development and current month."""
        stats_comparison = []
        
        for feature in self.feature_cols:
            if feature not in current_df.columns:
                continue
            
            if self.development_df[feature].dtype in ['int64', 'float64']:
                dev_mean = self.development_df[feature].mean()
                dev_std = self.development_df[feature].std()
                curr_mean = current_df[feature].mean()
                curr_std = current_df[feature].std()
                
                # Calculate percentage change
                mean_change = ((curr_mean - dev_mean) / dev_mean * 100) if dev_mean != 0 else 0
                std_change = ((curr_std - dev_std) / dev_std * 100) if dev_std != 0 else 0
                
                stats_comparison.append({
                    'feature': feature,
                    'dev_mean': dev_mean,
                    'curr_mean': curr_mean,
                    'mean_change_pct': mean_change,
                    'dev_std': dev_std,
                    'curr_std': curr_std,
                    'std_change_pct': std_change
                })
        
        return pd.DataFrame(stats_comparison)
    
    
    def batch_monitor(self, monthly_df: pd.DataFrame) -> pd.DataFrame:
        """
        Monitor multiple months at once.
        
        Parameters:
        -----------
        monthly_df : pd.DataFrame
            DataFrame with month column identifying different periods
        
        Returns:
        --------
        monitoring_summary : pd.DataFrame
            Summary of PSI metrics across all months
        """
        months = sorted(monthly_df[self.month_col].unique())
        all_results = []
        
        print(f"\n{'='*60}")
        print(f"BATCH MONITORING: {len(months)} months")
        print(f"{'='*60}")
        
        for month in months:
            month_data = monthly_df[monthly_df[self.month_col] == month]
            results = self.monitor_monthly(month_data, str(month))
            all_results.append(results)
        
        # Create summary DataFrame
        summary_data = []
        for result in all_results:
            row = {
                'month': result['month'],
                'record_count': result['record_count'],
                'score_psi': result['score_psi'],
                'features_with_drift': sum(result['feature_drift_flags'].values()),
                'total_features': len(result['feature_drift_flags']),
                'drift_percentage': (sum(result['feature_drift_flags'].values()) / 
                                   len(result['feature_drift_flags']) * 100) if result['feature_drift_flags'] else 0
            }
            # Add individual feature PSI
            for feature, psi in result['feature_psi'].items():
                row[f'psi_{feature}'] = psi
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by month
        summary_df = summary_df.sort_values('month')
        
        print(f"\n{'='*60}")
        print("MONITORING SUMMARY COMPLETE")
        print(f"{'='*60}")
        
        return summary_df
    
    
    def generate_report(self, monitoring_results: Dict, 
                       output_path: Optional[str] = None) -> None:
        """
        Generate visual monitoring report.
        
        Parameters:
        -----------
        monitoring_results : Dict
            Results from monitor_monthly()
        output_path : str, optional
            Path to save the report figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Feature PSI heatmap
        ax1 = fig.add_subplot(gs[0, :])
        feature_psi = pd.Series(monitoring_results['feature_psi']).sort_values(ascending=False)
        colors = ['red' if x >= 0.25 else 'orange' if x >= 0.1 else 'green' 
                 for x in feature_psi.values]
        ax1.barh(range(len(feature_psi)), feature_psi.values, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(feature_psi)))
        ax1.set_yticklabels(feature_psi.index, fontsize=8)
        ax1.set_xlabel('PSI Value')
        ax1.set_title(f'Feature PSI - {monitoring_results["month"]}', fontsize=14, fontweight='bold')
        ax1.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='Moderate Threshold')
        ax1.axvline(x=0.25, color='red', linestyle='--', alpha=0.5, label='High Threshold')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Score PSI breakdown
        if 'score_psi_detail' in monitoring_results and not monitoring_results['score_psi_detail'].empty:
            ax2 = fig.add_subplot(gs[1, 0])
            psi_detail = monitoring_results['score_psi_detail']
            x = range(len(psi_detail))
            width = 0.35
            ax2.bar([i - width/2 for i in x], psi_detail['expected_pct'], width, 
                   label='Development', alpha=0.7)
            ax2.bar([i + width/2 for i in x], psi_detail['actual_pct'], width,
                   label='Current Month', alpha=0.7)
            ax2.set_xlabel('Score Bins')
            ax2.set_ylabel('Percentage (%)')
            ax2.set_title(f'Score Distribution: Dev vs Current\nPSI = {monitoring_results["score_psi"]:.4f}',
                         fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=6)
        
        # 3. PSI contribution by bin
        if 'score_psi_detail' in monitoring_results and not monitoring_results['score_psi_detail'].empty:
            ax3 = fig.add_subplot(gs[1, 1])
            psi_detail = monitoring_results['score_psi_detail']
            colors_contrib = ['red' if x > 0.05 else 'orange' if x > 0.02 else 'green' 
                             for x in psi_detail['psi_contribution']]
            ax3.bar(range(len(psi_detail)), psi_detail['psi_contribution'], 
                   color=colors_contrib, alpha=0.7)
            ax3.set_xlabel('Score Bins')
            ax3.set_ylabel('PSI Contribution')
            ax3.set_title('PSI Contribution by Score Bin', fontsize=12, fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=6)
        
        # 4. Summary statistics table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Prepare summary text
        summary_text = f"""
        MONITORING SUMMARY - {monitoring_results['month']}
        {'='*70}
        
        Records Monitored: {monitoring_results['record_count']:,}
        
        Score PSI: {monitoring_results.get('score_psi', 'N/A'):.4f} {'🚨' if monitoring_results.get('score_drift_flag', False) else '✓'}
        
        Features Monitored: {len(monitoring_results['feature_psi'])}
        Features with Drift: {sum(monitoring_results['feature_drift_flags'].values())}
        Drift Percentage: {sum(monitoring_results['feature_drift_flags'].values()) / len(monitoring_results['feature_drift_flags']) * 100:.1f}%
        
        Top 5 Drifting Features:
        """
        
        top_5_drift = pd.Series(monitoring_results['feature_psi']).nlargest(5)
        for feat, psi in top_5_drift.items():
            flag = "🚨" if psi >= 0.25 else "⚠️" if psi >= 0.1 else "✓"
            summary_text += f"\n        {feat}: {psi:.4f} {flag}"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle(f'Credit Score Model Monitoring Report', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 Report saved to: {output_path}")
        
        plt.tight_layout()
        plt.show()


# Example usage function
def example_usage():
    """
    Example of how to use the monitoring framework.
    """
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_dev = 10000
    n_current = 5000
    
    # Development dataset (baseline)
    dev_data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_dev),
        'income': np.random.lognormal(10.5, 0.5, n_dev),
        'credit_utilization': np.random.beta(2, 5, n_dev),
        'loan_tenure': np.random.choice([12, 24, 36, 48, 60], n_dev),
        'employment_type': np.random.choice(['Permanent', 'Contract', 'Self-Employed'], n_dev),
        'score_prediction': np.random.normal(650, 80, n_dev),
        'month': 'development'
    })
    
    # Current month data (with some drift)
    current_data = pd.DataFrame({
        'age': np.random.normal(37, 11, n_current),  # Slight drift in age
        'income': np.random.lognormal(10.6, 0.55, n_current),  # Drift in income
        'credit_utilization': np.random.beta(2.5, 4.5, n_current),  # Moderate drift
        'loan_tenure': np.random.choice([12, 24, 36, 48, 60], n_current),
        'employment_type': np.random.choice(['Permanent', 'Contract', 'Self-Employed'], n_current),
        'score_prediction': np.random.normal(655, 85, n_current),  # Slight score drift
        'month': '2024-11'
    })
    
    # Initialize monitor
    monitor = CreditScoreMonitor(
        development_df=dev_data,
        score_col='score_prediction',
        month_col='month'
    )
    
    # Run monthly monitoring
    results = monitor.monitor_monthly(current_data, '2024-11')
    
    # Generate report
    monitor.generate_report(results)
    
    return monitor, results


if __name__ == "__main__":
    print("Credit Score Monitoring Framework")
    print("=" * 60)
    print("\nRunning example with synthetic data...")
    monitor, results = example_usage()