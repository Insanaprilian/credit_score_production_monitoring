"""
Visualization and Dashboard Module for Credit Score Monitoring
===============================================================
Creates interactive dashboards and trend visualizations for model monitoring.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


class MonitoringDashboard:
    """
    Creates comprehensive visualization dashboards for monitoring results.
    """
    
    def __init__(self):
        self.colors = {
            'stable': '#2ecc71',
            'moderate': '#f39c12',
            'high': '#e74c3c',
            'primary': '#3498db',
            'secondary': '#9b59b6'
        }
    
    def plot_psi_trends(self, summary_df: pd.DataFrame, 
                        features: Optional[List[str]] = None,
                        figsize: tuple = (16, 10)) -> None:
        """
        Plot PSI trends over time for multiple features.
        
        Parameters:
        -----------
        summary_df : pd.DataFrame
            Summary dataframe from batch_monitor()
        features : List[str], optional
            List of features to plot. If None, plots all.
        figsize : tuple
            Figure size
        """
        # Identify PSI columns
        psi_cols = [col for col in summary_df.columns if col.startswith('psi_')]
        
        if features:
            psi_cols = [f'psi_{feat}' for feat in features if f'psi_{feat}' in psi_cols]
        
        if not psi_cols:
            print("No PSI columns found in summary dataframe")
            return
        
        # Calculate number of subplots needed
        n_features = len(psi_cols)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, psi_col in enumerate(psi_cols):
            ax = axes[idx]
            feature_name = psi_col.replace('psi_', '')
            
            # Plot PSI trend
            ax.plot(summary_df['month'], summary_df[psi_col], 
                   marker='o', linewidth=2, markersize=8, color=self.colors['primary'])
            
            # Add threshold lines
            ax.axhline(y=0.1, color=self.colors['moderate'], 
                      linestyle='--', linewidth=1.5, alpha=0.7, label='Moderate (0.1)')
            ax.axhline(y=0.25, color=self.colors['high'], 
                      linestyle='--', linewidth=1.5, alpha=0.7, label='High (0.25)')
            
            # Color background based on PSI zones
            ax.axhspan(0, 0.1, alpha=0.1, color=self.colors['stable'])
            ax.axhspan(0.1, 0.25, alpha=0.1, color=self.colors['moderate'])
            ax.axhspan(0.25, ax.get_ylim()[1], alpha=0.1, color=self.colors['high'])
            
            ax.set_title(f'{feature_name}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Month', fontsize=9)
            ax.set_ylabel('PSI', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper right')
            
            # Rotate x-axis labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
            
            # Annotate points with high PSI
            for i, (month, psi) in enumerate(zip(summary_df['month'], summary_df[psi_col])):
                if psi >= 0.25:
                    ax.annotate(f'{psi:.2f}', xy=(i, psi), 
                              xytext=(0, 10), textcoords='offset points',
                              ha='center', fontsize=8, color=self.colors['high'],
                              fontweight='bold')
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Feature PSI Trends Over Time', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.show()
    
    
    def plot_score_psi_trend(self, summary_df: pd.DataFrame,
                            figsize: tuple = (14, 6)) -> None:
        """
        Plot score PSI trend with volume overlay.
        
        Parameters:
        -----------
        summary_df : pd.DataFrame
            Summary dataframe with score_psi and record_count
        figsize : tuple
            Figure size
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot 1: Score PSI trend
        ax1.plot(summary_df['month'], summary_df['score_psi'], 
                marker='o', linewidth=3, markersize=10, 
                color=self.colors['primary'], label='Score PSI')
        
        # Add threshold lines
        ax1.axhline(y=0.1, color=self.colors['moderate'], 
                   linestyle='--', linewidth=2, alpha=0.7, label='Moderate Threshold')
        ax1.axhline(y=0.25, color=self.colors['high'], 
                   linestyle='--', linewidth=2, alpha=0.7, label='High Threshold')
        
        # Color zones
        ax1.axhspan(0, 0.1, alpha=0.15, color=self.colors['stable'])
        ax1.axhspan(0.1, 0.25, alpha=0.15, color=self.colors['moderate'])
        ax1.axhspan(0.25, ax1.get_ylim()[1], alpha=0.15, color=self.colors['high'])
        
        ax1.set_ylabel('Score PSI', fontsize=12, fontweight='bold')
        ax1.set_title('Score Distribution Stability Over Time', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Annotate all points
        for i, (month, psi) in enumerate(zip(summary_df['month'], summary_df['score_psi'])):
            color = (self.colors['high'] if psi >= 0.25 
                    else self.colors['moderate'] if psi >= 0.1 
                    else self.colors['stable'])
            ax1.annotate(f'{psi:.3f}', xy=(i, psi), 
                        xytext=(0, 12), textcoords='offset points',
                        ha='center', fontsize=9, color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor=color, alpha=0.8))
        
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Volume trend
        ax2.bar(summary_df['month'], summary_df['record_count'], 
               color=self.colors['secondary'], alpha=0.6)
        ax2.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Records', fontsize=10)
        ax2.set_title('Monthly Volume', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add volume labels
        for i, (month, count) in enumerate(zip(summary_df['month'], summary_df['record_count'])):
            ax2.text(i, count, f'{count:,}', ha='center', va='bottom', 
                    fontsize=8, fontweight='bold')
        
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    
    def plot_drift_heatmap(self, summary_df: pd.DataFrame, 
                          figsize: tuple = (14, 8)) -> None:
        """
        Create a heatmap of PSI values across features and months.
        
        Parameters:
        -----------
        summary_df : pd.DataFrame
            Summary dataframe from batch_monitor()
        figsize : tuple
            Figure size
        """
        # Extract PSI columns
        psi_cols = [col for col in summary_df.columns if col.startswith('psi_')]
        
        if not psi_cols:
            print("No PSI columns found")
            return
        
        # Create PSI matrix
        psi_data = summary_df[['month'] + psi_cols].set_index('month')
        psi_data.columns = [col.replace('psi_', '') for col in psi_data.columns]
        
        # Transpose so features are rows and months are columns
        psi_matrix = psi_data.T
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create custom colormap: green -> yellow -> orange -> red
        colors_list = ['#2ecc71', '#f1c40f', '#f39c12', '#e74c3c', '#c0392b']
        n_bins = 100
        cmap = sns.blend_palette(colors_list, n_colors=n_bins, as_cmap=True)
        
        # Plot heatmap
        sns.heatmap(psi_matrix, annot=True, fmt='.3f', cmap=cmap,
                   vmin=0, vmax=0.4, center=0.175,
                   linewidths=0.5, linecolor='gray',
                   cbar_kws={'label': 'PSI Value', 'shrink': 0.8},
                   ax=ax)
        
        ax.set_title('Feature Drift Heatmap Across Months', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        
        # Add threshold reference
        threshold_text = "PSI Thresholds: <0.1 Stable | 0.1-0.25 Moderate | >0.25 High"
        fig.text(0.5, 0.02, threshold_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    
    def plot_drift_summary(self, summary_df: pd.DataFrame,
                          figsize: tuple = (16, 6)) -> None:
        """
        Create summary dashboard with drift statistics.
        
        Parameters:
        -----------
        summary_df : pd.DataFrame
            Summary dataframe from batch_monitor()
        figsize : tuple
            Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Drift percentage over time
        ax1 = axes[0]
        ax1.plot(summary_df['month'], summary_df['drift_percentage'],
                marker='o', linewidth=3, markersize=10, color=self.colors['high'])
        ax1.fill_between(range(len(summary_df)), summary_df['drift_percentage'],
                        alpha=0.3, color=self.colors['high'])
        ax1.axhline(y=20, color='orange', linestyle='--', linewidth=2, 
                   alpha=0.7, label='20% Threshold')
        ax1.axhline(y=50, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label='50% Threshold')
        ax1.set_title('Feature Drift Percentage', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Month', fontsize=10)
        ax1.set_ylabel('% Features Drifting', fontsize=10)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        
        # Plot 2: Number of features with drift
        ax2 = axes[1]
        bars = ax2.bar(summary_df['month'], summary_df['features_with_drift'],
                      color=self.colors['moderate'], alpha=0.7, label='Drifting')
        ax2.bar(summary_df['month'], 
               summary_df['total_features'] - summary_df['features_with_drift'],
               bottom=summary_df['features_with_drift'],
               color=self.colors['stable'], alpha=0.7, label='Stable')
        
        # Add value labels
        for i, (drift, total) in enumerate(zip(summary_df['features_with_drift'], 
                                               summary_df['total_features'])):
            ax2.text(i, total, f'{drift}/{total}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        
        ax2.set_title('Feature Drift Count', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Month', fontsize=10)
        ax2.set_ylabel('Number of Features', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        
        # Plot 3: Average PSI over time
        ax3 = axes[2]
        psi_cols = [col for col in summary_df.columns if col.startswith('psi_')]
        avg_psi = summary_df[psi_cols].mean(axis=1)
        
        ax3.plot(summary_df['month'], avg_psi,
                marker='s', linewidth=3, markersize=10, 
                color=self.colors['secondary'], label='Avg Feature PSI')
        ax3.plot(summary_df['month'], summary_df['score_psi'],
                marker='D', linewidth=3, markersize=8,
                color=self.colors['primary'], label='Score PSI')
        
        ax3.axhline(y=0.1, color=self.colors['moderate'], 
                   linestyle='--', linewidth=1.5, alpha=0.7)
        ax3.axhline(y=0.25, color=self.colors['high'],
                   linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax3.set_title('Average PSI Trends', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Month', fontsize=10)
        ax3.set_ylabel('PSI', fontsize=10)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        
        plt.suptitle('Monthly Drift Summary Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    
    def create_executive_summary(self, summary_df: pd.DataFrame,
                                figsize: tuple = (16, 10)) -> None:
        """
        Create an executive summary dashboard for stakeholders.
        
        Parameters:
        -----------
        summary_df : pd.DataFrame
            Summary dataframe from batch_monitor()
        figsize : tuple
            Figure size
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # Get latest month data
        latest = summary_df.iloc[-1]
        
        # 1. Key Metrics (Top Row)
        # Score PSI
        ax1 = fig.add_subplot(gs[0, 0])
        score_psi = latest['score_psi']
        color = (self.colors['high'] if score_psi >= 0.25 
                else self.colors['moderate'] if score_psi >= 0.1 
                else self.colors['stable'])
        ax1.text(0.5, 0.6, f"{score_psi:.4f}", ha='center', va='center',
                fontsize=48, fontweight='bold', color=color)
        ax1.text(0.5, 0.3, "Score PSI", ha='center', va='center',
                fontsize=16, color='gray')
        status = "🚨 HIGH" if score_psi >= 0.25 else "⚠️ MODERATE" if score_psi >= 0.1 else "✓ STABLE"
        ax1.text(0.5, 0.1, status, ha='center', va='center',
                fontsize=14, fontweight='bold', color=color)
        ax1.axis('off')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # Features Drifting
        ax2 = fig.add_subplot(gs[0, 1])
        drift_pct = latest['drift_percentage']
        color = 'red' if drift_pct > 50 else 'orange' if drift_pct > 20 else 'green'
        ax2.text(0.5, 0.6, f"{drift_pct:.1f}%", ha='center', va='center',
                fontsize=48, fontweight='bold', color=color)
        ax2.text(0.5, 0.3, "Features Drifting", ha='center', va='center',
                fontsize=16, color='gray')
        ax2.text(0.5, 0.1, f"{int(latest['features_with_drift'])}/{int(latest['total_features'])} features",
                ha='center', va='center', fontsize=12, color='gray')
        ax2.axis('off')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        # Records Monitored
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.text(0.5, 0.6, f"{int(latest['record_count']):,}", ha='center', va='center',
                fontsize=42, fontweight='bold', color=self.colors['primary'])
        ax3.text(0.5, 0.3, "Records Monitored", ha='center', va='center',
                fontsize=16, color='gray')
        ax3.text(0.5, 0.1, f"Latest: {latest['month']}", ha='center', va='center',
                fontsize=12, color='gray')
        ax3.axis('off')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        
        # 2. Score PSI Trend (Middle Row Left)
        ax4 = fig.add_subplot(gs[1, :2])
        ax4.plot(summary_df['month'], summary_df['score_psi'],
                marker='o', linewidth=3, markersize=10, color=self.colors['primary'])
        ax4.axhline(y=0.1, color=self.colors['moderate'], linestyle='--', alpha=0.7)
        ax4.axhline(y=0.25, color=self.colors['high'], linestyle='--', alpha=0.7)
        ax4.fill_between(range(len(summary_df)), 0, summary_df['score_psi'],
                        alpha=0.2, color=self.colors['primary'])
        ax4.set_title('Score PSI Trend', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Month', fontsize=11)
        ax4.set_ylabel('PSI', fontsize=11)
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Drift Percentage (Middle Row Right)
        ax5 = fig.add_subplot(gs[1, 2])
        colors_bar = ['red' if x > 50 else 'orange' if x > 20 else 'green' 
                     for x in summary_df['drift_percentage']]
        ax5.bar(summary_df['month'], summary_df['drift_percentage'],
               color=colors_bar, alpha=0.7)
        ax5.axhline(y=20, color='orange', linestyle='--', alpha=0.7, linewidth=2)
        ax5.axhline(y=50, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax5.set_title('Drift Percentage', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Month', fontsize=11)
        ax5.set_ylabel('%', fontsize=11)
        ax5.grid(True, alpha=0.3, axis='y')
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
        
        # 4. Top Drifting Features (Bottom Row)
        ax6 = fig.add_subplot(gs[2, :])
        psi_cols = [col for col in latest.index if col.startswith('psi_')]
        top_drift = latest[psi_cols].sort_values(ascending=False).head(10)
        top_drift.index = [idx.replace('psi_', '') for idx in top_drift.index]
        
        colors_drift = ['red' if x >= 0.25 else 'orange' if x >= 0.1 else 'green'
                       for x in top_drift.values]
        bars = ax6.barh(range(len(top_drift)), top_drift.values, color=colors_drift, alpha=0.7)
        ax6.set_yticks(range(len(top_drift)))
        ax6.set_yticklabels(top_drift.index, fontsize=10)
        ax6.set_xlabel('PSI Value', fontsize=11)
        ax6.set_title('Top 10 Features by PSI (Latest Month)', fontsize=14, fontweight='bold')
        ax6.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, linewidth=2)
        ax6.axvline(x=0.25, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax6.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_drift.values)):
            ax6.text(val, i, f' {val:.3f}', va='center', fontsize=9, fontweight='bold')
        
        plt.suptitle(f'Model Monitoring Executive Summary\nPeriod: {summary_df["month"].iloc[0]} to {latest["month"]}',
                    fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        plt.show()


# Example usage
def create_sample_visualizations():
    """
    Create sample visualizations with synthetic data.
    """
    # Generate synthetic monitoring summary
    months = pd.date_range('2024-01', '2024-11', freq='MS').strftime('%Y-%m').tolist()
    
    np.random.seed(42)
    summary_data = {
        'month': months,
        'record_count': np.random.randint(8000, 12000, len(months)),
        'score_psi': np.random.uniform(0.05, 0.3, len(months)),
        'features_with_drift': np.random.randint(2, 8, len(months)),
        'total_features': [15] * len(months),
    }
    
    # Add feature PSI columns
    features = ['age', 'income', 'credit_utilization', 'loan_tenure', 
                'payment_history', 'debt_to_income']
    for feat in features:
        summary_data[f'psi_{feat}'] = np.random.uniform(0.02, 0.35, len(months))
    
    summary_df = pd.DataFrame(summary_data)
    summary_df['drift_percentage'] = (summary_df['features_with_drift'] / 
                                     summary_df['total_features'] * 100)
    
    # Create dashboard
    dashboard = MonitoringDashboard()
    
    print("Creating visualizations...")
    print("\n1. Executive Summary Dashboard")
    dashboard.create_executive_summary(summary_df)
    
    print("\n2. Drift Summary Dashboard")
    dashboard.plot_drift_summary(summary_df)
    
    print("\n3. PSI Heatmap")
    dashboard.plot_drift_heatmap(summary_df)
    
    print("\n4. Score PSI Trend")
    dashboard.plot_score_psi_trend(summary_df)
    
    print("\n5. Feature PSI Trends")
    dashboard.plot_psi_trends(summary_df, features=features[:6])
    
    print("\n✓ All visualizations created successfully!")
    
    return dashboard, summary_df


if __name__ == "__main__":
    dashboard, summary_df = create_sample_visualizations()