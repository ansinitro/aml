"""
Visualization Module
Create comprehensive charts and graphs for air quality analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette(config.COLOR_PALETTE)


def plot_time_series(df, pollutants=None, save_path=None):
    """
    Plot time series for multiple pollutants
    """
    if pollutants is None:
        pollutants = [p for p in config.POLLUTANTS if p in df.columns]
    
    n_pollutants = len(pollutants)
    fig, axes = plt.subplots(n_pollutants, 1, figsize=(14, 3*n_pollutants))
    
    if n_pollutants == 1:
        axes = [axes]
    
    for idx, pollutant in enumerate(pollutants):
        ax = axes[idx]
        
        # Plot data
        ax.plot(df['date'], df[pollutant], linewidth=0.8, alpha=0.7)
        
        # Add WHO standard line if available
        if pollutant in config.WHO_STANDARDS:
            standards = config.WHO_STANDARDS[pollutant]
            if 'daily' in standards:
                ax.axhline(y=standards['daily'], color='red', linestyle='--', 
                          label=f"WHO Daily Standard ({standards['daily']} {standards['unit']})",
                          linewidth=1.5)
            if 'annual' in standards:
                ax.axhline(y=standards['annual'], color='orange', linestyle='--',
                          label=f"WHO Annual Standard ({standards['annual']} {standards['unit']})",
                          linewidth=1.5)
        
        ax.set_title(f'{pollutant} Levels Over Time - Aktobe', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel(f'{pollutant} ({config.WHO_STANDARDS.get(pollutant, {}).get("unit", "μg/m³")})', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"Saved time series plot to {save_path}")
    
    return fig


def plot_seasonal_boxplot(df, pollutant='PM2.5', groupby='month', save_path=None):
    """
    Create boxplot showing seasonal variation
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if groupby == 'month':
        df_plot = df.copy()
        df_plot['month_name'] = pd.to_datetime(df_plot['date']).dt.strftime('%b')
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        sns.boxplot(data=df_plot, x='month_name', y=pollutant, order=month_order, ax=ax)
        ax.set_xlabel('Month', fontsize=11)
    elif groupby == 'season':
        season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
        sns.boxplot(data=df, x='season', y=pollutant, order=season_order, ax=ax)
        ax.set_xlabel('Season', fontsize=11)
    
    # Add WHO standard line
    if pollutant in config.WHO_STANDARDS:
        standards = config.WHO_STANDARDS[pollutant]
        if 'daily' in standards:
            ax.axhline(y=standards['daily'], color='red', linestyle='--',
                      label=f"WHO Daily Standard", linewidth=2)
    
    ax.set_title(f'{pollutant} Distribution by {groupby.capitalize()} - Aktobe',
                fontsize=13, fontweight='bold')
    ax.set_ylabel(f'{pollutant} ({config.WHO_STANDARDS.get(pollutant, {}).get("unit", "μg/m³")})',
                 fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"Saved seasonal boxplot to {save_path}")
    
    return fig


def plot_correlation_heatmap(df, pollutants=None, weather_vars=None, save_path=None):
    """
    Create correlation heatmap
    """
    if pollutants is None:
        pollutants = [p for p in config.POLLUTANTS if p in df.columns]
    
    if weather_vars is None:
        weather_vars = ['temperature', 'humidity', 'wind_speed']
        weather_vars = [w for w in weather_vars if w in df.columns]
    
    all_vars = pollutants + weather_vars
    corr_matrix = df[all_vars].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax)
    
    ax.set_title('Correlation Matrix - Pollutants and Weather Variables',
                fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"Saved correlation heatmap to {save_path}")
    
    return fig


def plot_aqi_distribution(df, save_path=None):
    """
    Plot AQI distribution and categories
    """
    if 'AQI' not in df.columns:
        print("AQI column not found")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['AQI'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(df['AQI'].mean(), color='red', linestyle='--',
                   label=f"Mean: {df['AQI'].mean():.1f}", linewidth=2)
    axes[0].axvline(df['AQI'].median(), color='green', linestyle='--',
                   label=f"Median: {df['AQI'].median():.1f}", linewidth=2)
    axes[0].set_xlabel('AQI', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('AQI Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Category pie chart
    if 'AQI_category' in df.columns:
        category_counts = df['AQI_category'].value_counts()
        colors = ['green', 'yellow', 'orange', 'red', 'purple', 'maroon']
        axes[1].pie(category_counts, labels=category_counts.index, autopct='%1.1f%%',
                   colors=colors[:len(category_counts)], startangle=90)
        axes[1].set_title('AQI Category Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"Saved AQI distribution plot to {save_path}")
    
    return fig


def plot_heating_season_comparison(df, pollutants=None, save_path=None):
    """
    Compare pollution during heating vs non-heating season
    """
    if pollutants is None:
        pollutants = [p for p in config.POLLUTANTS[:4] if p in df.columns]
    
    heating_comparison = df.groupby('is_heating_season')[pollutants].mean()
    heating_comparison.index = ['Non-Heating', 'Heating']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(pollutants))
    width = 0.35
    
    ax.bar(x - width/2, heating_comparison.loc['Non-Heating'], width,
           label='Non-Heating Season', alpha=0.8)
    ax.bar(x + width/2, heating_comparison.loc['Heating'], width,
           label='Heating Season', alpha=0.8)
    
    ax.set_xlabel('Pollutant', fontsize=11)
    ax.set_ylabel('Average Concentration (μg/m³)', fontsize=11)
    ax.set_title('Pollution Levels: Heating vs Non-Heating Season - Aktobe',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pollutants)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"Saved heating season comparison to {save_path}")
    
    return fig


def plot_who_exceedance(df, save_path=None):
    """
    Visualize WHO standard exceedance
    """
    pollutants_with_standards = []
    exceedance_pct = []
    
    for pollutant in config.POLLUTANTS:
        if pollutant not in df.columns:
            continue
        
        exceedance_col = f'{pollutant}_exceeds_WHO_daily'
        if exceedance_col in df.columns:
            pct = (df[exceedance_col].sum() / len(df)) * 100
            pollutants_with_standards.append(pollutant)
            exceedance_pct.append(pct)
    
    if not pollutants_with_standards:
        print("No WHO exceedance data available")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(pollutants_with_standards, exceedance_pct, alpha=0.7, edgecolor='black')
    
    # Color bars based on severity
    for i, bar in enumerate(bars):
        if exceedance_pct[i] > 50:
            bar.set_color('darkred')
        elif exceedance_pct[i] > 25:
            bar.set_color('orange')
        else:
            bar.set_color('yellow')
    
    ax.set_xlabel('Pollutant', fontsize=11)
    ax.set_ylabel('Percentage of Days Exceeding WHO Daily Standard (%)', fontsize=11)
    ax.set_title('WHO Daily Standard Exceedance - Aktobe', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(exceedance_pct):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"Saved WHO exceedance plot to {save_path}")
    
    return fig


def create_all_visualizations(df, output_dir=None):
    """
    Generate all visualizations and save to output directory
    """
    if output_dir is None:
        output_dir = config.DATA_DIRS['figures']
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating all visualizations...")
    print("=" * 60)
    
    # Time series
    print("\n1. Time series plots...")
    plot_time_series(df, save_path=os.path.join(output_dir, '01_time_series.png'))
    plt.close()
    
    # Seasonal boxplots
    print("\n2. Seasonal boxplots...")
    for pollutant in ['PM2.5', 'PM10']:
        if pollutant in df.columns:
            plot_seasonal_boxplot(df, pollutant=pollutant, groupby='month',
                                save_path=os.path.join(output_dir, f'02_seasonal_{pollutant}.png'))
            plt.close()
    
    # Correlation heatmap
    print("\n3. Correlation heatmap...")
    plot_correlation_heatmap(df, save_path=os.path.join(output_dir, '03_correlation_heatmap.png'))
    plt.close()
    
    # AQI distribution
    print("\n4. AQI distribution...")
    plot_aqi_distribution(df, save_path=os.path.join(output_dir, '04_aqi_distribution.png'))
    plt.close()
    
    # Heating season comparison
    print("\n5. Heating season comparison...")
    plot_heating_season_comparison(df, save_path=os.path.join(output_dir, '05_heating_comparison.png'))
    plt.close()
    
    # WHO exceedance
    print("\n6. WHO exceedance...")
    plot_who_exceedance(df, save_path=os.path.join(output_dir, '06_who_exceedance.png'))
    plt.close()
    
    print("\n" + "=" * 60)
    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    from preprocessing import load_data
    
    df = load_data('aktobe_real_processed.csv', data_type='processed')
    create_all_visualizations(df)
