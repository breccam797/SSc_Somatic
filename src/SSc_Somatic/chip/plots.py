import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_hex
from matplotlib.colors import LinearSegmentedColormap
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from collections import defaultdict

from ..settings import cohort_values


# Cohort labels and metadata column names. These start empty and are filled
# from the study configuration by apply_settings(), which SSc_Somatic.configure() calls.
AGE_COL = None
CONTROL_LABEL = None
DLCO_PCT_COL = None
DURATION_COL = None
FVC_PCT_COL = None
GROUP_COL = None
SKIN_SCORE_COL = None
STATUS_COL = None
SUBTYPE_A = None
SUBTYPE_B = None
SUBTYPE_COL = None


def apply_settings(config):
    """
    Fill this module's cohort labels and column names from a StudyConfig.

    Called by SSc_Somatic.configure(); notebooks do not need to call it directly.
    """
    globals().update(cohort_values(config))


def save_and_show(output_filename, verbose=True, add_pdf_extension=False):
    """
    Save the current matplotlib figure if a file name is given, then display it.

    Parameters:
        output_filename: str or None, file to save to; nothing is saved when empty
        verbose: bool, print the saved path (default: True)
        add_pdf_extension: bool, append '.pdf' to output_filename (default: False)
    """
    if output_filename:
        path = f'{output_filename}.pdf' if add_pdf_extension else output_filename
        plt.savefig(path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Saved figure to {path}")
    plt.show()


def plot_vaf_and_gene_distribution(df, title_suffix="", vaf_cutoff=0.02, figsize=(10, 6), bins=30, color='purple'):
    """
    Create VAF histogram and gene frequency bar chart for WL driver variants.
    
    Parameters:
        df: Dataframe with VAF and Gene.refGene columns
        title_suffix: Additional text for plot titles
        vaf_cutoff: VAF threshold for vertical line
        figsize: Figure size (width, height)
        bins: Number of histogram bins
        color: Color for plots
    
    Returns:
        Dictionary with VAF stats and gene frequency data
    """
    min_vaf = df['VAF'].min()
    max_vaf = df['VAF'].max()
    max_vaf_sample = df.loc[df['VAF'] == max_vaf]
    gene_counts = df['Gene.refGene'].value_counts()
    print("VAF Summary Statistics:")
    print(f"Lowest VAF value: {min_vaf:.6f}")
    print(f"Highest VAF value: {max_vaf:.6f}")
    print("\nSample with highest VAF value:")
    print(max_vaf_sample)
    print("-" * 50)
    plt.figure(figsize=figsize)
    sns.histplot(df['VAF'], bins=bins, kde=True, color=color)
    plt.axvline(x=vaf_cutoff, color='teal', linestyle='--', linewidth=2)
    y_position = plt.ylim()[1] * 0.96
    x_position = vaf_cutoff + (plt.xlim()[1] - plt.xlim()[0]) * 0.02
    plt.text(x_position, y_position, f"{vaf_cutoff*100:.0f}% VAF cutoff", 
             color='black', fontsize=12, verticalalignment='top')
    base_title = 'Histogram of VAF values for WL driver genes'
    full_title = f"{base_title}{' ' + title_suffix if title_suffix else ''}"
    plt.title(full_title)
    plt.xlabel('VAF (proportion value)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    plt.figure(figsize=figsize)
    plt.bar(gene_counts.index, gene_counts.values, color=color)
    plt.xticks(rotation=90)
    plt.title('Number of samples with CH driver variants on each wl gene')
    plt.xlabel('CH gene')
    plt.ylabel('Number of samples')
    plt.show()
    return {
        'min_vaf': min_vaf,
        'max_vaf': max_vaf,
        'max_vaf_sample': max_vaf_sample,
        'total_samples': len(df),
        'gene_counts': gene_counts,
        'unique_genes': len(gene_counts)
    }


def plot_chip_variants_by_subtype(wl_df, snv_df, age_bins_normal, labels, age_bins=None, 
                                  age_groups_to_remove=None, 
                                  figsize=(12, 12), colors=None):
    """
    Create analysis and visualization of CHIP variants by SSc subtype.
    
    Parameters:
        wl_df: WL CHIP dataframe with statustype, SSc Type, Age columns
        snv_df: SNV dataframe with SSc Type, age_group, statustype columns
        age_bins_normal: Age bin counts for control samples
        labels: Age group labels
        age_bins: Bin edges for age grouping (if age_group column doesn't exist)
        age_groups_to_remove: Age groups to exclude from analysis
        figsize: Figure size (width, height)
        colors: Colors for Control, Diffuse, Limited subtypes
    
    Returns:
        Dictionary with processed dataframes and statistics
    """
    if age_groups_to_remove is None:
        age_groups_to_remove = ['0-9','10-19','80-89','90-99']
    if colors is None:
        colors = ['#1f998a','darkred','darkblue']
    wl_working = wl_df.copy()
    snv_working = snv_df.copy()
    if 'age_group' not in wl_working.columns:
        if 'Age_Group' in wl_working.columns:
            wl_working['age_group'] = wl_working['Age_Group']
        elif age_bins is not None:
            wl_working['age_group'] = pd.cut(wl_working[AGE_COL], bins=age_bins, labels=labels, right=False)
        else:
            raise ValueError("age_group column not found and age_bins not provided")
    wl_working[f'{SUBTYPE_COL} '] = wl_working[SUBTYPE_COL].fillna(CONTROL_LABEL)
    snv_limited = snv_working[snv_working[SUBTYPE_COL] == SUBTYPE_A]
    snv_diffuse = snv_working[snv_working[SUBTYPE_COL] == SUBTYPE_B]
    snv_limited_grouped = snv_limited.groupby(['age_group', STATUS_COL]).size().unstack(fill_value=0)
    snv_diffuse_grouped = snv_diffuse.groupby(['age_group', STATUS_COL]).size().unstack(fill_value=0)
    snv_limited_df = pd.DataFrame(snv_limited_grouped.reset_index())
    snv_limited_df.columns = ['age_group','num_samples']
    snv_diffuse_df = pd.DataFrame(snv_diffuse_grouped.reset_index())
    snv_diffuse_df.columns = ['age_group','num_samples']
    age_bins_limited = snv_limited_df['num_samples'].tolist()
    age_bins_diffuse = snv_diffuse_df['num_samples'].tolist()
    ssc_type_age_bin_counts = pd.DataFrame()
    ssc_type_age_bin_counts[CONTROL_LABEL] = age_bins_normal
    ssc_type_age_bin_counts[SUBTYPE_B] = age_bins_diffuse
    ssc_type_age_bin_counts[SUBTYPE_A] = age_bins_limited
    ssc_type_age_bin_counts.index = labels
    print('Age breakdown of completed samples for control and SSc samples:')
    print(ssc_type_age_bin_counts)
    wl_grouped_type = wl_working.groupby(['age_group', SUBTYPE_COL]).size().unstack(fill_value=0)
    wl_grouped_type_norm = wl_grouped_type.div(ssc_type_age_bin_counts, axis=0)
    print('\nAge breakdown of CHIP sample counts for SSc subtypes and normal samples:')
    print(wl_grouped_type)
    print('\nRate of CHIP in each age group for SSc subtypes and normal samples:')
    print(wl_grouped_type_norm)
    wl_grouped_norm_filtered = wl_grouped_type_norm.drop(age_groups_to_remove, axis=0)
    gene_column = 'Gene.refGene' if 'Gene.refGene' in wl_working.columns else 'gene'
    wl_subtype_grouped = wl_working.groupby([gene_column, GROUP_COL]).size().unstack(fill_value=0)
    wl_subtype_grouped_sorted = wl_subtype_grouped.sort_values(
        by=wl_subtype_grouped.sum(axis=0).index.tolist(), ascending=False
    )
    wl_subtype_totals = wl_subtype_grouped.sum(axis=0)
    wl_subtype_grouped_norm = wl_subtype_grouped.div(wl_subtype_totals, axis=1)
    wl_subtype_grouped_norm_sorted = wl_subtype_grouped_norm.sort_values(
        by=wl_subtype_grouped_norm.sum(axis=0).index.tolist(), ascending=False
    )
    print(f'\nTotal samples by subtype:')
    print(wl_subtype_totals)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    wl_subtype_grouped_sorted.plot(kind='bar', ax=ax1, color=colors, edgecolor='black')
    ax1.set_title('Number of Samples with CH Driver Variants by Gene and Subtype', fontsize=18)
    ax1.set_xlabel('CH Gene', fontsize=16)
    ax1.set_ylabel('Number of Samples', fontsize=16)
    ax1.tick_params(axis='x', rotation=45, labelsize=15)
    plt.setp(ax1.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax1.legend(title='Subtype', fontsize=14)
    wl_subtype_grouped_norm_sorted.plot(kind='bar', ax=ax2, color=colors, edgecolor='black')
    ax2.set_title('Proportion of Samples with CH Driver Variants by Gene and Subtype', fontsize=18)
    ax2.set_xlabel('CH Gene', fontsize=16)
    ax2.set_ylabel('Proportion of Samples', fontsize=16)
    ax2.tick_params(axis='x', rotation=45, labelsize=14)
    plt.setp(ax2.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax2.legend(title='Subtype', fontsize=14)
    plt.tight_layout()
    plt.show()
    return {
        'subtype_counts': wl_subtype_grouped_sorted,
        'subtype_proportions': wl_subtype_grouped_norm_sorted,
        'age_chip_rates': wl_grouped_type_norm,
        'age_chip_rates_filtered': wl_grouped_norm_filtered,
        'total_by_subtype': wl_subtype_totals,
        'age_breakdown': ssc_type_age_bin_counts
    }

def create_chip_occurrence_plot(merged_df, output_filename='ssc_chip_rate_line_plot.pdf', 
                                figsize=(10, 8), verbose=True):
    """
    Create logistic regression plot showing CHIP occurrence with age by SSc subtype.
    Includes 95% confidence intervals and R² values from linear regression.
    
    Parameters:
        merged_df: Dataframe with Age, wl_CHIP, statustype_subtype, sample_id columns
        output_filename: Path for saved figure (set to None to skip saving)
        figsize: Figure size as (width, height)
        verbose: Print summary statistics
    
    Returns:
        tuple: (figure, r2_results dictionary with R², correlation, slope, n for each subtype)
    """
    
    if verbose:
        print(f"Original data shape: {merged_df.shape}")
    merged_df_unique = merged_df.drop_duplicates(subset=['sample_id'], keep='first').copy()
    if verbose:
        print(f"Deduplicated data shape: {merged_df_unique.shape}")
        print(f"Removed {merged_df.shape[0] - merged_df_unique.shape[0]} duplicate sample_ids")
    merged_df_unique['is_wl'] = (merged_df_unique['wl_CHIP'] == 'WL').astype(int)
    r2_results = {}
    if verbose:
        print("Calculating R² values using linear regression:")
    for subtype in merged_df_unique[GROUP_COL].unique():
        subtype_data = merged_df_unique[merged_df_unique[GROUP_COL] == subtype].copy()
        subtype_data = subtype_data.dropna(subset=[AGE_COL, 'is_wl'])
        if len(subtype_data) > 1:
            X = subtype_data[AGE_COL].values.reshape(-1, 1)
            y = subtype_data['is_wl'].values
            try:
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                correlation = np.corrcoef(subtype_data[AGE_COL], subtype_data['is_wl'])[0, 1]
                r2_results[subtype] = {
                    'R2': r2,
                    'correlation': correlation,
                    'slope': model.coef_[0],
                    'intercept': model.intercept_,
                    'n_observations': len(subtype_data),
                    'mean_age': subtype_data[AGE_COL].mean(),
                    'mean_chip_rate': subtype_data['is_wl'].mean()
                }
                if verbose:
                    print(f"  {subtype}: R²={r2:.4f}, r={correlation:.4f}, slope={model.coef_[0]:.6f}, n={len(subtype_data)}")
            except Exception as e:
                print(f"Warning: Could not fit linear model for {subtype}: {e}")
                r2_results[subtype] = {
                    'R2': np.nan, 'correlation': np.nan, 'slope': np.nan, 'intercept': np.nan,
                    'n_observations': len(subtype_data),
                    'mean_age': subtype_data[AGE_COL].mean() if len(subtype_data) > 0 else np.nan,
                    'mean_chip_rate': subtype_data['is_wl'].mean() if len(subtype_data) > 0 else np.nan
                }
        else:
            print(f"Warning: {subtype} has insufficient data (n={len(subtype_data)})")
            r2_results[subtype] = {
                'R2': np.nan, 'correlation': np.nan, 'slope': np.nan, 'intercept': np.nan,
                'n_observations': len(subtype_data),
                'mean_age': subtype_data[AGE_COL].mean() if len(subtype_data) > 0 else np.nan,
                'mean_chip_rate': subtype_data['is_wl'].mean() if len(subtype_data) > 0 else np.nan
            }
    palette = {CONTROL_LABEL: '#1f998a', SUBTYPE_B: 'darkred', SUBTYPE_A: 'darkblue'}
    if verbose:
        print(f"Creating logistic regression plot with {len(merged_df_unique)} unique samples")
    plt.figure(figsize=figsize)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sns.lmplot(
            data=merged_df_unique,
            x=AGE_COL,
            y='is_wl',
            hue=GROUP_COL,
            height=8,
            aspect=1.2,
            palette=palette,
            scatter_kws={'alpha': 0.4, 's': 40},
            ci=95,
            logistic=True,
            legend=False
        )
    g.set_axis_labels(AGE_COL, 'Rate of CHIP', fontsize=14)
    g.fig.suptitle('CHIP occurrence with Age and SSc Subtype\n(Linear Regression R² values shown)', 
                   fontsize=16, y=0.98)
    g.set(ylim=(0.0, 0.30))
    g.set(xlim=(20, 79))
    ax = g.fig.axes[0]
    ax.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    legend_elements = []
    subtypes = [SUBTYPE_B, SUBTYPE_A, CONTROL_LABEL]
    legend_colors = {SUBTYPE_B: 'darkred', SUBTYPE_A: 'darkblue', CONTROL_LABEL: '#1f998a'}
    for subtype in subtypes:
        if subtype in r2_results:
            r2_val = r2_results[subtype]['R2']
            n_obs = r2_results[subtype]['n_observations']
            if not np.isnan(r2_val):
                label = f"{subtype} (R² = {r2_val:.3f}, n={n_obs})"
            else:
                label = f"{subtype} (R² = N/A, n={n_obs})"
        else:
            label = subtype
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=legend_colors[subtype], 
                                          markersize=8, label=label))
    legend = ax.legend(handles=legend_elements, title='Subtype (Linear Regression)', 
                       loc='upper left', frameon=True, fontsize=9, title_fontsize=10,
                       bbox_to_anchor=(0.02, 0.98))
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    save_and_show(output_filename, verbose)
    return g.fig, r2_results

def analyze_wl_disease_duration_boxplots(merged_df, output_filename='ssc_chip_disease_duration_boxplots.pdf',
                                         show_plots=True, figsize=(12, 8), verbose=True):
    """
    Analyze CHIP status and relationship with disease duration using boxplots.
    Performs pairwise Mann-Whitney U tests with Benjamini-Hochberg FDR correction.
    
    Parameters:
        merged_df: Dataframe with is_wl, statustype_subtype, Age, Disease Duration columns
        output_filename: Path for saved figure (set to None to skip saving)
        show_plots: Whether to display the plots
        figsize: Figure size as (width, height)
        verbose: Print summary statistics
    
    Returns:
        Dictionary with statistical_results, dataframe_with_groups, and interpretation
    """
    
    df = merged_df.copy()
    
    def get_highlight_color(row):
        if row['is_wl'] == 1 and row[GROUP_COL] == SUBTYPE_B:
            return f'WL {SUBTYPE_B}'
        elif row['is_wl'] == 1 and row[GROUP_COL] == SUBTYPE_A:
            return f'WL {SUBTYPE_A}'
        else:
            return 'Other'
    
    df['highlight_group'] = df.apply(get_highlight_color, axis=1)
    if 'age_at_diagnosis' not in df.columns:
        df['age_at_diagnosis'] = df[AGE_COL] - df[DURATION_COL]
    
    wl_limited = df[(df['is_wl'] == 1) & (df[GROUP_COL] == SUBTYPE_A)]
    wl_diffuse = df[(df['is_wl'] == 1) & (df[GROUP_COL] == SUBTYPE_B)]
    non_wl = df[df['is_wl'] == 0]
    all_wl = df[df['is_wl'] == 1]
    
    wl_limited_durations = wl_limited[DURATION_COL].dropna()
    wl_diffuse_durations = wl_diffuse[DURATION_COL].dropna()
    non_wl_durations = non_wl[DURATION_COL].dropna()
    all_wl_durations = all_wl[DURATION_COL].dropna()
    
    tests = [
        ('limited_vs_nonwl', wl_limited_durations, non_wl_durations),
        ('diffuse_vs_nonwl', wl_diffuse_durations, non_wl_durations),
        ('diffuse_vs_limited', wl_diffuse_durations, wl_limited_durations),
        ('overall_wl_vs_nonwl', all_wl_durations, non_wl_durations)
    ]
    
    pairwise_results = {}
    p_values = []
    test_names = []
    
    for test_name, group1, group2 in tests:
        if len(group1) > 0 and len(group2) > 0:
            u_stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
            pairwise_results[test_name] = {'u_stat': u_stat, 'p_val': p_val}
            p_values.append(p_val)
            test_names.append(test_name)
        else:
            pairwise_results[test_name] = {'u_stat': np.nan, 'p_val': np.nan}
            p_values.append(np.nan)
            test_names.append(test_name)
    
    valid_p_values = [p for p in p_values if not np.isnan(p)]
    if len(valid_p_values) > 0:
        rejected, p_corrected, _, _ = multipletests(valid_p_values, alpha=0.05, method='fdr_bh')
        corrected_idx = 0
        for i, test_name in enumerate(test_names):
            if not np.isnan(p_values[i]):
                pairwise_results[test_name]['p_val_corrected'] = p_corrected[corrected_idx]
                pairwise_results[test_name]['significant_corrected'] = rejected[corrected_idx]
                corrected_idx += 1
            else:
                pairwise_results[test_name]['p_val_corrected'] = np.nan
                pairwise_results[test_name]['significant_corrected'] = False
    
    stats_summary = {
        'wl_diffuse_median': np.median(wl_diffuse_durations) if len(wl_diffuse_durations) > 0 else np.nan,
        'wl_diffuse_iqr': (np.percentile(wl_diffuse_durations, 75) - np.percentile(wl_diffuse_durations, 25)) if len(wl_diffuse_durations) > 0 else np.nan,
        'wl_diffuse_n': len(wl_diffuse_durations),
        'wl_limited_median': np.median(wl_limited_durations) if len(wl_limited_durations) > 0 else np.nan,
        'wl_limited_iqr': (np.percentile(wl_limited_durations, 75) - np.percentile(wl_limited_durations, 25)) if len(wl_limited_durations) > 0 else np.nan,
        'wl_limited_n': len(wl_limited_durations),
        'non_wl_median': np.median(non_wl_durations) if len(non_wl_durations) > 0 else np.nan,
        'non_wl_iqr': (np.percentile(non_wl_durations, 75) - np.percentile(non_wl_durations, 25)) if len(non_wl_durations) > 0 else np.nan,
        'non_wl_n': len(non_wl_durations),
        'pairwise_tests': pairwise_results,
        'overall_test': pairwise_results['overall_wl_vs_nonwl']
    }
    
    if show_plots:
        non_wl_limited = df[(df['is_wl'] == 0) & (df[GROUP_COL] == SUBTYPE_A)][DURATION_COL].dropna()
        non_wl_diffuse = df[(df['is_wl'] == 0) & (df[GROUP_COL] == SUBTYPE_B)][DURATION_COL].dropna()
        wl_limited_plot = df[(df['is_wl'] == 1) & (df[GROUP_COL] == SUBTYPE_A)][DURATION_COL].dropna()
        wl_diffuse_plot = df[(df['is_wl'] == 1) & (df[GROUP_COL] == SUBTYPE_B)][DURATION_COL].dropna()
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        box_data = [non_wl_limited, non_wl_diffuse, wl_limited_plot, wl_diffuse_plot]
        box_labels = [f'Non-WL {SUBTYPE_A}', f'Non-WL {SUBTYPE_B}', f'WL {SUBTYPE_A}', f'WL {SUBTYPE_B}']
        
        bp = ax.boxplot(box_data, labels=box_labels, widths=0.3, patch_artist=False, showfliers=False)
        for element in ['boxes', 'whiskers', 'medians', 'caps']:
            plt.setp(bp[element], color='black')
        
        positions = [1, 2, 3, 4]
        colors = ['gray', 'gray', 'darkblue', 'darkred']
        np.random.seed(42)
        for data, pos, color in zip(box_data, positions, colors):
            if len(data) > 0:
                x_jitter = np.random.normal(pos, 0.05, len(data))
                ax.scatter(x_jitter, data, alpha=0.6, s=20, facecolors=color, edgecolors='none')
        
        ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_ylabel(DURATION_COL, fontsize=17)
        ax.set_xlabel('', fontsize=17)
        ax.tick_params(axis='both', which='major', labelsize=17)
        ax.grid(False)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_and_show(output_filename, verbose)
        
        if verbose:
            print("WL analysis results:")
            print(f"Group counts: {df['highlight_group'].value_counts().to_dict()}")
            print(f"Non-WL (n={stats_summary['non_wl_n']}): median={stats_summary['non_wl_median']:.1f}, IQR={stats_summary['non_wl_iqr']:.1f}")
            print(f"WL {SUBTYPE_B} (n={stats_summary['wl_diffuse_n']}): median={stats_summary['wl_diffuse_median']:.1f}, IQR={stats_summary['wl_diffuse_iqr']:.1f}")
            print(f"WL {SUBTYPE_A} (n={stats_summary['wl_limited_n']}): median={stats_summary['wl_limited_median']:.1f}, IQR={stats_summary['wl_limited_iqr']:.1f}")
            print("Pairwise Mann-Whitney U tests (FDR corrected):")
            test_descriptions = {
                'limited_vs_nonwl': f'WL {SUBTYPE_A} vs Non-WL',
                'diffuse_vs_nonwl': f'WL {SUBTYPE_B} vs Non-WL',
                'diffuse_vs_limited': f'WL {SUBTYPE_B} vs WL {SUBTYPE_A}',
                'overall_wl_vs_nonwl': 'Overall WL vs Non-WL'
            }
            for test_name, description in test_descriptions.items():
                result = pairwise_results[test_name]
                p_corr = result.get('p_val_corrected', np.nan)
                sig = result.get('significant_corrected', False)
                sig_marker = '**' if p_corr <= 0.01 else ('*' if p_corr <= 0.05 else 'ns')
                print(f"  {description}: U={result['u_stat']:.1f}, p={result['p_val']:.4f}, p_FDR={p_corr:.4f} {sig_marker}")
    
    return {
        'statistical_results': stats_summary,
        'dataframe_with_groups': df,
        'interpretation': 'independent' if len([r for r in pairwise_results.values() if r.get('significant_corrected', False)]) == 0 else 'associated'
    }

def plot_chip_variants_by_statustype(df, genes_of_interest, output_filename='ssc_chip_gene_statustype_bar_chart.pdf',
                                     figsize=(4, 12), alpha=0.8, verbose=True):
    """
    Create vertically stacked visualizations of CHIP driver variants by gene and status type subtype.
    
    Parameters:
        df: Dataframe containing Gene.refGene and statustype_subtype columns
        genes_of_interest: List of gene names to include in the plot
        output_filename: Path for saved figure (set to None to skip saving)
        figsize: Figure size as (width, height)
        alpha: Transparency level for bars
        verbose: Print summary statistics
    
    Returns:
        tuple: (grouped_counts DataFrame, normalized_proportions DataFrame)
    """
    status_order = [CONTROL_LABEL, SUBTYPE_A, SUBTYPE_B]
    color_palette = {CONTROL_LABEL: '#1F998A', SUBTYPE_A: 'darkblue', SUBTYPE_B: 'darkred'}
    
    df_filtered = df[df['Gene.refGene'].isin(genes_of_interest)]
    status_grouped = df_filtered.groupby(['Gene.refGene', GROUP_COL]).size().unstack(fill_value=0)
    
    available_status = [status for status in status_order if status in status_grouped.columns]
    status_grouped = status_grouped[available_status]
    status_grouped_sorted = status_grouped.sort_values(
        by=status_grouped.sum(axis=0).index.tolist(), ascending=False
    )
    
    status_grouped_totals = status_grouped.sum(axis=0)
    status_grouped_norm = status_grouped.div(status_grouped_totals, axis=1)
    status_grouped_norm_sorted = status_grouped_norm.sort_values(
        by=status_grouped_norm.sum(axis=0).index.tolist(), ascending=False
    )
    
    colors = [color_palette[status] for status in available_status]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    status_grouped_sorted.plot(kind='bar', ax=ax1, color=colors, edgecolor='black', alpha=alpha, width=0.6)
    ax1.set_title('Number of Samples with CH Driver Variants\nby Gene and Status Type Subtype', fontsize=18)
    ax1.set_xlabel('CH Gene', fontsize=18)
    ax1.set_ylabel('Number of Samples', fontsize=18)
    ax1.tick_params(axis='x', rotation=45, labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    ax1.legend(title='Status Type Subtype', fontsize=18, title_fontsize=18, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    status_grouped_norm_sorted.plot(kind='bar', ax=ax2, color=colors, edgecolor='black', alpha=alpha, width=0.6)
    ax2.set_title('Proportion of Samples with CH Driver Variants\nby Gene and Status Type Subtype', fontsize=18)
    ax2.set_xlabel('CH Gene', fontsize=18)
    ax2.set_ylabel('Proportion of Samples', fontsize=18)
    ax2.tick_params(axis='x', rotation=45, labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)
    ax2.legend(title='Status Type Subtype', fontsize=18, title_fontsize=18, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    
    save_and_show(output_filename, verbose)
    
    if verbose:
        print("Total samples by status type subtype:")
        for status, count in status_grouped_totals.items():
            print(f"  {status}: {count}")
        print(f"Genes with highest variant counts: {status_grouped_sorted.sum(axis=1).head(3).index.tolist()}")
    
    return status_grouped_sorted, status_grouped_norm_sorted

def create_chip_variants_plot(df, top_n=30, output_filename='ssc_chip_variants_bar_plot.pdf',
                              figsize=(7, 8), verbose=True):
    """
    Create horizontal stacked bar chart showing variant types by gene.
    
    Parameters:
        df: Dataframe with Gene.refGene, ExonicFunc.refGene, and Func.refGene columns
        top_n: Number of top genes to display
        output_filename: Path for saved figure (set to None to skip saving)
        figsize: Figure size as (width, height)
        verbose: Print summary statistics
    
    Returns:
        tuple: (plot_data DataFrame, color_map dictionary)
    """
    
    custom_colors = ['#0B1D3A', '#70A9A1', '#19605E', '#188F92', '#88C3CC', '#A0AEC0']
    
    def get_variant_category(row):
        exonic_func = row['ExonicFunc.refGene']
        func = row['Func.refGene']
        if pd.notna(exonic_func) and exonic_func != '.' and exonic_func != '':
            return exonic_func
        elif pd.notna(func) and func != '.' and func != '':
            return func
        else:
            return 'Unknown'
    
    def get_darkness(hex_color):
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return 0.299 * r + 0.587 * g + 0.114 * b
    
    df_working = df.copy()
    df_working['Variant_Category'] = df_working.apply(get_variant_category, axis=1)
    variant_counts = df_working.groupby(['Gene.refGene', 'Variant_Category']).size().unstack(fill_value=0)
    
    gene_totals = variant_counts.sum(axis=1).sort_values(ascending=True)
    top_genes = gene_totals.tail(top_n)
    plot_data = variant_counts.loc[top_genes.index]
    
    variant_categories = list(plot_data.columns)
    n_categories = len(variant_categories)
    
    category_totals = plot_data.sum(axis=0).sort_values(ascending=False)
    column_order = category_totals.index.tolist()
    plot_data = plot_data[column_order]
    
    sorted_colors = sorted(custom_colors, key=get_darkness)
    if n_categories <= len(sorted_colors):
        colors = sorted_colors[:n_categories]
    else:
        additional_colors = sns.color_palette("husl", n_categories - len(sorted_colors))
        colors = sorted_colors + [to_hex(c) for c in additional_colors]
    
    color_map = dict(zip(column_order, colors))
    
    fig, ax = plt.subplots(figsize=figsize)
    left = np.zeros(len(plot_data))
    
    for category in plot_data.columns:
        color = color_map[category]
        ax.barh(range(len(plot_data)), plot_data[category], 
                left=left, label=category, color=color, height=0.8)
        left += plot_data[category]
    
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data.index, fontsize=16)
    ax.set_xlabel('Number of variants', fontsize=16)
    ax.set_title('CHIP variants', fontsize=18, fontweight='bold', loc='left')
    ax.tick_params(axis='x', labelsize=16)
    
    if len(variant_categories) <= 6:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    
    max_variants = plot_data.sum(axis=1).max()
    ax.set_xlim(0, max_variants * 1.1)
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    save_and_show(output_filename, verbose)
    
    if verbose:
        print(f"Top {top_n} genes displayed")
        print(f"Variant categories: {variant_categories}")
        print(f"Total variants in plot: {plot_data.sum().sum()}")
    
    return plot_data, color_map

def plot_gene_vaf_violin(df, genes_of_interest, stats_df=None, min_variants=3, vaf_max=0.5,
                         output_filename='ssc_chip_gene_vaf_violin.pdf', figsize=(12, 8),
                         violin_width=0.8, verbose=True):
    """
    Create violin plots for VAF distribution by gene, including an 'All CHIP' distribution.
    
    Parameters:
        df: Dataframe with VAF and Gene.refGene columns
        genes_of_interest: List of specific genes to include
        stats_df: Optional DataFrame from analyze_vaf_distribution_differences for annotations
        min_variants: Minimum number of variants required per gene
        vaf_max: Maximum VAF value to display on y-axis
        output_filename: Path for saved figure (set to None to skip saving)
        figsize: Figure size as (width, height)
        violin_width: Width of violin plots (0-1)
        verbose: Print summary statistics
    
    Returns:
        tuple: (figure, axes, plot_data dictionary)
    """
    if 'VAF' not in df.columns:
        print("Error: VAF column not found in dataframe")
        return None, None, None
    
    clean_df = df.dropna(subset=['VAF', 'Gene.refGene']).copy()
    clean_df['VAF'] = pd.to_numeric(clean_df['VAF'], errors='coerce')
    clean_df = clean_df.dropna(subset=['VAF'])
    
    if clean_df['VAF'].max() > 1:
        clean_df['VAF'] = clean_df['VAF'] / 100
    
    all_chip_data = clean_df['VAF'].values
    filtered_df = clean_df[clean_df['Gene.refGene'].isin(genes_of_interest)]
    
    gene_counts = filtered_df['Gene.refGene'].value_counts()
    genes_with_min_variants = gene_counts[gene_counts >= min_variants].index
    final_genes = [gene for gene in genes_of_interest if gene in genes_with_min_variants]
    
    if len(final_genes) == 0:
        print(f"Error: No genes have at least {min_variants} variants")
        return None, None, None
    
    plot_data = filtered_df[filtered_df['Gene.refGene'].isin(final_genes)]
    
    all_labels = ['All CHIP'] + final_genes
    all_violin_data = [all_chip_data]
    for gene in final_genes:
        gene_data = plot_data[plot_data['Gene.refGene'] == gene]['VAF'].values
        all_violin_data.append(gene_data)
    
    all_chip_color = '#2C3E50'
    gene_colors = ['#3A4330', '#426737', '#75871B', '#BAB97D']
    individual_colors = (gene_colors * ((len(final_genes) // len(gene_colors)) + 1))[:len(final_genes)]
    colors = [all_chip_color] + individual_colors
    
    fig, ax = plt.subplots(figsize=figsize)
    violin_parts = ax.violinplot(all_violin_data, positions=range(len(all_labels)), 
                                 widths=violin_width, showmeans=False, showmedians=True)
    
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)
    
    violin_parts['cmedians'].set_colors('black')
    violin_parts['cmedians'].set_linewidth(2)
    
    if stats_df is not None and 'Significance_Level' in stats_df.columns:
        for i, gene in enumerate(final_genes):
            gene_stats = stats_df[stats_df['Gene'] == gene]
            if len(gene_stats) > 0:
                sig_level = gene_stats.iloc[0]['Significance_Level']
                if sig_level != 'ns':
                    y_pos = vaf_max * 0.95
                    ax.text(i + 1, y_pos, sig_level, ha='center', va='bottom', 
                            fontweight='bold', fontsize=10)
    
    ax.set_xlim(-0.5, len(all_labels) - 0.5)
    ax.set_ylim(0, vaf_max)
    ax.set_xlabel('Gene', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variant Allele Frequency (VAF)', fontsize=12, fontweight='bold')
    ax.set_title('VAF Distribution by Gene', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    plt.tight_layout()
    
    save_and_show(output_filename, verbose)
    
    if verbose:
        print(f"Genes displayed: {final_genes}")
        print(f"All CHIP mean VAF: {np.mean(all_chip_data):.4f}")
        print(f"All CHIP n: {len(all_chip_data)}")
    
    return fig, ax, {'all_chip': all_chip_data, 'genes': final_genes, 'data': all_violin_data}

def plot_tet2_vaf_by_subtype_violin(df, stats_df=None, min_variants=3, vaf_max=0.7,
                                    output_filename='ssc_chip_tet2_vaf_subtype_violin.pdf',
                                    figsize=(10, 8), violin_width=0.8, verbose=True):
    """
    Create violin plots for TET2 VAF distribution by statustype_subtype groups.
    
    Parameters:
        df: Dataframe with VAF, Gene.refGene, and statustype_subtype columns
        stats_df: Optional DataFrame from analyze_vaf_distribution_differences for annotations
        min_variants: Minimum number of variants required per subtype group
        vaf_max: Maximum VAF value to display on y-axis
        output_filename: Path for saved figure (set to None to skip saving)
        figsize: Figure size as (width, height)
        violin_width: Width of violin plots (0-1)
        verbose: Print summary statistics
    
    Returns:
        tuple: (figure, axes, subtype_data dictionary)
    """
    required_columns = ['VAF', 'Gene.refGene', GROUP_COL]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None, None, None
    
    clean_df = df.dropna(subset=required_columns).copy()
    clean_df['VAF'] = pd.to_numeric(clean_df['VAF'], errors='coerce')
    clean_df = clean_df.dropna(subset=['VAF'])
    
    if clean_df['VAF'].max() > 1:
        clean_df['VAF'] = clean_df['VAF'] / 100
    
    tet2_df = clean_df[clean_df['Gene.refGene'] == 'TET2'].copy()
    if len(tet2_df) == 0:
        print("Error: No TET2 variants found in the dataset")
        return None, None, None
    
    if verbose:
        print(f"Found {len(tet2_df)} TET2 variants total")
    
    subtype_counts = tet2_df[GROUP_COL].value_counts()
    desired_order = [CONTROL_LABEL, SUBTYPE_A, SUBTYPE_B]
    subtype_colors = {CONTROL_LABEL: '#259A8B', SUBTYPE_A: '#0C1D39', SUBTYPE_B: '#8A181A'}
    
    valid_subtypes = []
    for subtype in desired_order:
        if subtype in subtype_counts.index and subtype_counts[subtype] >= min_variants:
            valid_subtypes.append(subtype)
    
    if len(valid_subtypes) < 2:
        print(f"Error: Need at least 2 subtype groups with >= {min_variants} variants each")
        return None, None, None
    
    plot_df = tet2_df[tet2_df[GROUP_COL].isin(valid_subtypes)].copy()
    
    subtype_data = {}
    for subtype in valid_subtypes:
        subtype_data[subtype] = plot_df[plot_df[GROUP_COL] == subtype]['VAF'].values
    
    fig, ax = plt.subplots(figsize=figsize)
    violin_data = [subtype_data[subtype] for subtype in valid_subtypes]
    colors = [subtype_colors[subtype] for subtype in valid_subtypes]
    
    violin_parts = ax.violinplot(violin_data, positions=range(len(valid_subtypes)), 
                                 widths=violin_width, showmeans=False, showmedians=True)
    
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.8)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)
    
    violin_parts['cmedians'].set_colors('black')
    violin_parts['cmedians'].set_linewidth(2)
    
    np.random.seed(42)
    for i, subtype in enumerate(valid_subtypes):
        vaf_values = subtype_data[subtype]
        n_points = len(vaf_values)
        jitter_strength = violin_width * 0.1
        x_jitter = np.random.uniform(-jitter_strength, jitter_strength, n_points)
        x_positions = [i] * n_points + x_jitter
        ax.scatter(x_positions, vaf_values, color='black', s=25, alpha=0.7, zorder=3)
    
    if stats_df is not None and 'Significance_Level' in stats_df.columns:
        y_offset = vaf_max * 0.05
        current_y = vaf_max * 0.9
        for _, row in stats_df.iterrows():
            if row['Significance_Level'] != 'ns':
                if row['Group_1'] in valid_subtypes and row['Group_2'] in valid_subtypes:
                    pos1 = valid_subtypes.index(row['Group_1'])
                    pos2 = valid_subtypes.index(row['Group_2'])
                    ax.plot([pos1, pos2], [current_y, current_y], 'k-', linewidth=1)
                    ax.plot([pos1, pos1], [current_y - y_offset/4, current_y + y_offset/4], 'k-', linewidth=1)
                    ax.plot([pos2, pos2], [current_y - y_offset/4, current_y + y_offset/4], 'k-', linewidth=1)
                    mid_pos = (pos1 + pos2) / 2
                    ax.text(mid_pos, current_y + y_offset/2, row['Significance_Level'], 
                            ha='center', va='bottom', fontweight='bold', fontsize=10)
                    current_y += y_offset * 2
    
    ax.set_xlim(-0.5, len(valid_subtypes) - 0.5)
    ax.set_ylim(0, vaf_max)
    ax.set_xlabel('Status Type Subtype', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variant Allele Frequency (VAF)', fontsize=12, fontweight='bold')
    ax.set_title('TET2 VAF Distribution by Status Type Subtype', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(valid_subtypes)))
    ax.set_xticklabels(valid_subtypes, rotation=45, ha='right', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    plt.tight_layout()
    
    save_and_show(output_filename, verbose)
    
    if verbose:
        print(f"Subtypes displayed: {valid_subtypes}")
        for subtype in valid_subtypes:
            print(f"  {subtype}: n={len(subtype_data[subtype])}, mean VAF={np.mean(subtype_data[subtype]):.4f}")
    
    return fig, ax, subtype_data

def plot_gene_age_swarm(df, stats_results=None, genes=None,
                        age_col=None, gene_col='Gene.refGene', category_col=None,
                        output_filename='ssc_chip_gene_age_swarm.pdf', figsize=(12, 10), verbose=True):
    """
    Create 4-panel subplot with swarm plots and overlaid boxplots for gene age distributions.
    
    Parameters:
        df: Dataframe with age, gene, and statustype_subtype columns
        stats_results: Optional dictionary from analyze_gene_age_by_subtype for annotations
        genes: List of genes to plot
        age_col: Column name containing age values
        gene_col: Column name containing gene names
        category_col: Column name containing statustype_subtype categories
        output_filename: Path for saved figure (set to None to skip saving)
        figsize: Figure size as (width, height)
        verbose: Print summary statistics
    
    Returns:
        tuple: (figure, axes)
    """
    if genes is None:
        genes = ['DNMT3A', 'TET2', 'ASXL1', 'PPM1D']
    if age_col is None:
        age_col = AGE_COL
    if category_col is None:
        category_col = GROUP_COL
    category_order = [CONTROL_LABEL, SUBTYPE_A, SUBTYPE_B]
    category_colors = {CONTROL_LABEL: '#1F998A', SUBTYPE_A: 'darkblue', SUBTYPE_B: 'darkred'}
    
    all_ages = df[age_col].dropna()
    age_min, age_max = all_ages.min(), all_ages.max()
    age_padding = (age_max - age_min) * 0.05
    y_min, y_max = age_min - age_padding, age_max + age_padding
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for idx, gene in enumerate(genes):
        ax = axes[idx]
        gene_data = df[df[gene_col] == gene].copy()
        gene_data = gene_data.dropna(subset=[age_col, category_col])
        
        if len(gene_data) == 0:
            ax.text(0.5, 0.5, f'No data available\nfor {gene}', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, style='italic')
            ax.set_title(f'{gene} (n=0)', fontsize=14, fontweight='bold')
            continue
        
        present_categories = [cat for cat in category_order if cat in gene_data[category_col].unique()]
        
        sns.swarmplot(data=gene_data, x=category_col, y=age_col, 
                      ax=ax, size=8, alpha=0.6, order=present_categories,
                      palette=[category_colors[cat] for cat in present_categories])
        
        sns.boxplot(data=gene_data, x=category_col, y=age_col, 
                    ax=ax, width=0.3, order=present_categories,
                    boxprops=dict(alpha=0.3, facecolor='white', edgecolor='black', linewidth=2),
                    whiskerprops=dict(alpha=0.7, color='black', linewidth=3),
                    capprops=dict(alpha=0.7, color='black', linewidth=3),
                    medianprops=dict(alpha=0.8, color='black', linewidth=3),
                    flierprops=dict(alpha=0))
        
        ax.set_ylim(y_min, y_max)
        ax.set_title(f'{gene} (n={len(gene_data)})', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel(AGE_COL, fontsize=14, fontweight='bold')
        ax.set_xlabel('Status Type Subtype', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig.suptitle('Age Distribution by Status Type Subtype Across Target Genes', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.93)
    
    save_and_show(output_filename, verbose)
    
    if verbose:
        print(f"Genes plotted: {genes}")
        for gene in genes:
            gene_n = len(df[df[gene_col] == gene].dropna(subset=[age_col, category_col]))
            print(f"  {gene}: n={gene_n}")
    
    return fig, axes

def plot_variant_prevalence_heatmap(df, genes_of_interest=None, min_variant_count=3, top_n_per_gene=10,
                                    output_filename='ssc_chip_variant_heatmap.pdf', figsize=(12, 4),
                                    verbose=True):
    """
    Create heatmap showing variant prevalences across subtypes.
    
    Parameters:
        df: Dataframe with Gene.refGene, AAChange.refGene, and statustype_subtype columns
        genes_of_interest: List of genes to include (default: all genes)
        min_variant_count: Minimum occurrences for a variant to be included
        top_n_per_gene: Number of top variants to show per gene
        output_filename: Path for saved figure (set to None to skip saving)
        figsize: Figure size as (width, height)
        verbose: Print summary statistics
    
    Returns:
        tuple: (figure, axes, heatmap_data DataFrame, variant_info dictionary)
    """
    
    def parse_protein_change(aa_change):
        """Extract protein change from AAChange.refGene annotation."""
        if pd.isna(aa_change) or aa_change == '.':
            return None
        try:
            parts = str(aa_change).split(':')
            for part in parts:
                if part.startswith('p.'):
                    return part
            return None
        except Exception:
            return None
    
    plot_df = df.copy()
    plot_df['protein_change'] = plot_df['AAChange.refGene'].apply(parse_protein_change)
    plot_df = plot_df.dropna(subset=['protein_change'])
    plot_df = plot_df[plot_df['protein_change'] != '']
    
    if genes_of_interest:
        plot_df = plot_df[plot_df['Gene.refGene'].isin(genes_of_interest)]
        gene_list = genes_of_interest
    else:
        gene_list = plot_df['Gene.refGene'].unique().tolist()
    
    plot_df['gene_variant'] = plot_df['Gene.refGene'] + ':' + plot_df['protein_change']
    
    subtypes = [CONTROL_LABEL, SUBTYPE_A, SUBTYPE_B]
    subtype_totals = plot_df[GROUP_COL].value_counts()
    
    if verbose:
        print(f"Sample counts by subtype: {dict(subtype_totals)}")
    
    all_gene_variants = []
    variant_info = {}
    
    for gene in gene_list:
        gene_df = plot_df[plot_df['Gene.refGene'] == gene].copy()
        
        if len(gene_df) == 0:
            if verbose:
                print(f"No data found for {gene}")
            continue
        
        gene_variant_counts = {}
        gene_total_counts = {}
        
        for subtype in subtypes:
            subtype_data = gene_df[gene_df[GROUP_COL] == subtype]
            variant_count = subtype_data['protein_change'].value_counts()
            total_samples = subtype_totals.get(subtype, 1)
            variant_proportions = variant_count / total_samples
            gene_variant_counts[subtype] = variant_proportions
            gene_total_counts[subtype] = variant_count
        
        all_gene_variants_set = set()
        for subtype_variants in gene_variant_counts.values():
            all_gene_variants_set.update(subtype_variants.index)
        
        variant_total_counts = {}
        for variant in all_gene_variants_set:
            total_count = sum(gene_total_counts[subtype].get(variant, 0) for subtype in subtypes)
            variant_total_counts[variant] = total_count
        
        filtered_variants = [v for v, count in variant_total_counts.items() 
                            if count >= min_variant_count]
        
        variant_max_prevalence = {}
        for variant in filtered_variants:
            max_prev = max(gene_variant_counts[subtype].get(variant, 0) for subtype in subtypes)
            variant_max_prevalence[variant] = max_prev
        
        sorted_variants = sorted(variant_max_prevalence.items(), key=lambda x: x[1], reverse=True)
        selected_variants = [v[0] for v in sorted_variants[:top_n_per_gene]]
        
        if verbose:
            print(f"{gene}: {len(filtered_variants)} variants meeting threshold, showing top {len(selected_variants)}")
        
        for variant in selected_variants:
            full_variant_name = f"{gene}:{variant}"
            all_gene_variants.append(full_variant_name)
            variant_info[full_variant_name] = {
                'gene': gene,
                'variant': variant,
                'prevalences': {subtype: gene_variant_counts[subtype].get(variant, 0) 
                               for subtype in subtypes},
                'counts': {subtype: gene_total_counts[subtype].get(variant, 0) 
                          for subtype in subtypes},
                'total_count': sum(gene_total_counts[subtype].get(variant, 0) for subtype in subtypes)
            }
    
    if not all_gene_variants:
        print("No variants found meeting the criteria")
        return None, None, None, None
    
    heatmap_data = pd.DataFrame(index=subtypes, columns=all_gene_variants)
    for variant_name in all_gene_variants:
        for subtype in subtypes:
            heatmap_data.loc[subtype, variant_name] = variant_info[variant_name]['prevalences'][subtype]
    heatmap_data = heatmap_data.astype(float)
    
    colors_for_map = ['white', '#19605E']
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors_for_map, N=100)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    variant_labels = [v.split(':')[1] for v in all_gene_variants]
    sns.heatmap(heatmap_data, 
                annot=False,
                cmap=custom_cmap,
                cbar_kws={'label': 'Prevalence (Proportion)', 'shrink': 0.8},
                ax=ax,
                linewidths=0.5,
                linecolor='lightgray',
                xticklabels=variant_labels,
                yticklabels=subtypes)
    
    gene_boundaries = []
    current_gene = None
    current_col = 0
    
    for j, variant_name in enumerate(all_gene_variants):
        gene = variant_name.split(':')[0]
        if gene != current_gene:
            if current_gene is not None:
                ax.axvline(x=j, color='black', linewidth=2, alpha=0.7)
                gene_boundaries.append((current_col, j-1, current_gene))
            current_gene = gene
            current_col = j
    
    if current_gene is not None:
        gene_boundaries.append((current_col, len(all_gene_variants)-1, current_gene))
    
    for start_col, end_col, gene in gene_boundaries:
        middle_col = (start_col + end_col) / 2
        ax.text(middle_col + 0.5, len(subtypes) + 0.1, gene, 
                ha='center', va='top', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8, edgecolor='navy'))
    
    ax.set_title('CHIP Variant Prevalence Heatmap by Subtype', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Variants', fontsize=16, fontweight='bold')
    ax.set_ylabel('Subtype', fontsize=16, fontweight='bold')
    
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center', fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=14)
    
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    
    save_and_show(output_filename, verbose)
    
    if verbose:
        print(f"Total variants displayed: {len(all_gene_variants)}")
        print(f"Genes included: {', '.join(gene_list)}")
        print(f"Prevalence range: {heatmap_data.min().min():.3f} - {heatmap_data.max().max():.3f}")
        print("Top variant by subtype:")
        for subtype in subtypes:
            top_variant_col = heatmap_data.loc[subtype].idxmax()
            top_prevalence = heatmap_data.loc[subtype, top_variant_col]
            print(f"  {subtype}: {top_variant_col} (prevalence: {top_prevalence:.3f})")
    
    return fig, ax, heatmap_data, variant_info

def plot_gene_within_metric_coefficients(results, gene, output_filename=None, gene_color=None):
    """
    Plot coefficient results with within-metric FDR correction for a specified gene.
    Shows coefficients with 95% CI across clinical outcomes and disease subtypes.
    
    Parameters:
        results: dict, nested results from run_gene_within_metric_regression_analysis
        gene: str, gene name (e.g., 'TET2', 'DNMT3A')
        output_filename: str, optional filename for saving (without extension)
        gene_color: str, optional hex color for markers (defaults based on gene)
    
    Returns:
        matplotlib Figure object
    """
    if gene_color is None:
        color_map = {
            'TET2': '#ABC5C4',
            'DNMT3A': '#D4A574',
            'ASXL1': '#8B9DC3',
            'PPM1D': '#C9A9A6'
        }
        main_color = color_map.get(gene, '#7FA99B')
    else:
        main_color = gene_color
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 10))
    fig.suptitle(f'{gene} Clinical Effects by Disease Subtype\n(Within-Metric FDR Correction)', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    outcomes = [FVC_PCT_COL, DLCO_PCT_COL, SKIN_SCORE_COL]
    comparisons = [f'{gene}_vs_nonCHIP', f'{gene}_vs_non{gene}_CHIP']
    subtypes = [SUBTYPE_A, SUBTYPE_B, 'Combined']
    
    y_positions = {
        f'{gene}_vs_nonCHIP': [5.5, 4.5, 3.5],
        f'{gene}_vs_non{gene}_CHIP': [2.0, 1.0, 0.0]
    }
    
    for col, outcome in enumerate(outcomes):
        ax = axes[col]
        
        for comparison in comparisons:
            for i, subtype in enumerate(subtypes):
                result = results[outcome][comparison][subtype]
                y_pos = y_positions[comparison][i]
                
                coef = result['coef']
                ci_lower = result['ci_lower']
                ci_upper = result['ci_upper']
                pvalue_raw = result['pvalue_raw']
                pvalue_fdr = result['pvalue_fdr_within']
                n = int(result['n'])
                cohens_d = result['cohens_d']
                
                if not np.isnan(coef):
                    ax.scatter(coef, y_pos, s=250, color=main_color, 
                              edgecolor='black', linewidth=1.5, zorder=5, alpha=0.9)
                    
                    ax.plot([ci_lower, ci_upper], [y_pos, y_pos], 
                           color=main_color, linewidth=6, alpha=0.8, zorder=4,
                           solid_capstyle='round')
                    
                    # Significance marker: black asterisk centered over marker
                    if not np.isnan(pvalue_fdr) and pvalue_fdr < 0.05:
                        ax.text(coef, y_pos, '*', fontsize=24, ha='center', va='center', 
                               color='black', fontweight='bold', zorder=6)
                    elif pvalue_raw < 0.05:
                        ax.text(coef, y_pos, '*', fontsize=18, ha='center', va='center', 
                               color='black', fontweight='normal', zorder=6)
        
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
        ax.axhline(y=2.75, color='lightgray', linestyle='-', linewidth=1, alpha=0.5, zorder=0)
        
        y_ticks = [5.5, 4.5, 3.5, 2.0, 1.0, 0.0]
        y_labels = ['Combined', SUBTYPE_B, SUBTYPE_A, 'Combined', SUBTYPE_B, SUBTYPE_A]
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        
        ax.set_title(f'{outcome}', fontsize=18, fontweight='bold', pad=15)
        
        if 'FVC' in outcome or 'DLCO' in outcome:
            ax.set_xlabel('Coefficient (% predicted)', fontsize=16, fontweight='bold')
            ax.set_xlim(-45, 25)
        else:
            ax.set_xlabel(f'Coefficient ({SKIN_SCORE_COL} points)', fontsize=16, fontweight='bold')
            ax.set_xlim(-10, 15)
        
        ax.set_ylim(-0.7, 6.2)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Add y-axis group labels outside plot area
    fig.text(0.01, 0.65, f'{gene} vs\nNon-CHIP', fontsize=14, fontweight='bold', 
             va='center', ha='left')
    fig.text(0.01, 0.35, f'{gene} vs\nNon-{gene} CHIP', fontsize=14, fontweight='bold', 
             va='center', ha='left')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15, left=0.08, right=0.95)
    
    save_and_show(output_filename, add_pdf_extension=True)
    
    return fig


def plot_gene_comparison_panel(gene1_results, gene2_results, gene1_name, gene2_name,
                               comparison='vs_nonCHIP', subtype='Combined', use_fdr=True,
                               output_filename=None, gene1_color=None, gene2_color=None):
    """
    Create horizontal bar chart comparing two genes across clinical outcomes.
    
    Parameters:
        gene1_results: dict, results from run_gene_within_metric_regression_analysis for gene 1
        gene2_results: dict, results from run_gene_within_metric_regression_analysis for gene 2
        gene1_name: str, name of first gene (e.g., 'TET2')
        gene2_name: str, name of second gene (e.g., 'DNMT3A')
        comparison: str, which comparison to plot: 'vs_nonCHIP' or 'vs_nonGENE_CHIP'
        subtype: str, which subtype to plot: 'Limited', 'Diffuse', or 'Combined'
        use_fdr: bool, whether to use FDR-corrected p-values
        output_filename: str, optional filename for saving (without extension)
        gene1_color: str, optional hex color for gene 1
        gene2_color: str, optional hex color for gene 2
    
    Returns:
        matplotlib Figure object
    """
    color_map = {
        'TET2': '#558B89',
        'DNMT3A': '#3B4431',
        'ASXL1': '#5B7093',
        'PPM1D': '#8B6969'
    }
    
    if gene1_color is None:
        gene1_color = color_map.get(gene1_name, '#558B89')
    if gene2_color is None:
        gene2_color = color_map.get(gene2_name, '#3B4431')
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    outcome_keys = [FVC_PCT_COL, DLCO_PCT_COL, SKIN_SCORE_COL]
    outcome_labels = [FVC_PCT_COL, 'DLCO %predicted', SKIN_SCORE_COL]
    y_positions = np.arange(len(outcome_keys))
    bar_height = 0.35
    
    if comparison == 'vs_nonCHIP':
        gene1_comparison_key = f'{gene1_name}_vs_nonCHIP'
        gene2_comparison_key = f'{gene2_name}_vs_nonCHIP'
    else:
        gene1_comparison_key = f'{gene1_name}_vs_non{gene1_name}_CHIP'
        gene2_comparison_key = f'{gene2_name}_vs_non{gene2_name}_CHIP'
    
    print(f"Plotting comparison: {comparison}, subtype: {subtype}")
    print(f"Using {gene1_name} key: {gene1_comparison_key}")
    print(f"Using {gene2_name} key: {gene2_comparison_key}")
    print(f"Using {'FDR-corrected' if use_fdr else 'raw'} p-values")
    
    gene1_effects = []
    gene1_errors = []
    gene2_effects = []
    gene2_errors = []
    gene1_pvalues = []
    gene2_pvalues = []
    
    for i, outcome_key in enumerate(outcome_keys):
        # Gene 1 data
        try:
            gene1_result = gene1_results[outcome_key][gene1_comparison_key][subtype]
            gene1_coef = gene1_result['coef']
            gene1_ci_lower = gene1_result['ci_lower']
            gene1_ci_upper = gene1_result['ci_upper']
            gene1_pval = gene1_result['pvalue_fdr_within'] if use_fdr else gene1_result['pvalue_raw']
            
            gene1_effects.append(gene1_coef)
            gene1_err = [gene1_coef - gene1_ci_lower, gene1_ci_upper - gene1_coef]
            gene1_errors.append(gene1_err)
            gene1_pvalues.append(gene1_pval)
            
            print(f"{gene1_name} {outcome_key}: coef={gene1_coef:.2f}, p={gene1_pval:.4f}")
            
        except KeyError as e:
            print(f"Missing {gene1_name} data for {outcome_key}: {e}")
            gene1_effects.append(0)
            gene1_errors.append([0, 0])
            gene1_pvalues.append(1.0)
        
        # Gene 2 data
        try:
            gene2_result = gene2_results[outcome_key][gene2_comparison_key][subtype]
            gene2_coef = gene2_result['coef']
            gene2_ci_lower = gene2_result['ci_lower']
            gene2_ci_upper = gene2_result['ci_upper']
            gene2_pval = gene2_result['pvalue_fdr_within'] if use_fdr else gene2_result['pvalue_raw']
            
            gene2_effects.append(gene2_coef)
            gene2_err = [gene2_coef - gene2_ci_lower, gene2_ci_upper - gene2_coef]
            gene2_errors.append(gene2_err)
            gene2_pvalues.append(gene2_pval)
            
            print(f"{gene2_name} {outcome_key}: coef={gene2_coef:.2f}, p={gene2_pval:.4f}")
            
        except KeyError as e:
            print(f"Missing {gene2_name} data for {outcome_key}: {e}")
            gene2_effects.append(0)
            gene2_errors.append([0, 0])
            gene2_pvalues.append(1.0)
    
    gene1_errors = np.array(gene1_errors).T
    gene2_errors = np.array(gene2_errors).T
    
    bars1 = ax.barh(y_positions - bar_height/2, gene1_effects, bar_height,
                    xerr=gene1_errors, label=gene1_name, color=gene1_color,
                    alpha=0.8, capsize=8, error_kw={'linewidth': 3})
    
    bars2 = ax.barh(y_positions + bar_height/2, gene2_effects, bar_height,
                    xerr=gene2_errors, label=gene2_name, color=gene2_color,
                    alpha=0.8, capsize=8, error_kw={'linewidth': 3})
    
    # Significance markers: black asterisks centered on bars
    for i in range(len(outcome_keys)):
        if not np.isnan(gene1_pvalues[i]) and gene1_pvalues[i] < 0.05:
            ax.text(gene1_effects[i], y_positions[i] - bar_height/2, '*',
                   fontsize=20, ha='center', va='center', color='black', fontweight='bold')
        
        if not np.isnan(gene2_pvalues[i]) and gene2_pvalues[i] < 0.05:
            ax.text(gene2_effects[i], y_positions[i] + bar_height/2, '*',
                   fontsize=20, ha='center', va='center', color='black', fontweight='bold')
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, zorder=1)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(outcome_labels, fontsize=18)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    
    ax.set_xlabel('Effect Size (Coefficient)', fontsize=20, fontweight='bold')
    
    title_suffix = f" ({subtype} subtype, {comparison.replace('vs_', 'vs ')})"
    ax.set_title(f'Gene-Specific Clinical Effects{title_suffix}', fontsize=22, fontweight='bold', pad=20)
    
    ax.legend(loc='lower right', fontsize=18, frameon=False)
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    all_values = gene1_effects + gene2_effects
    all_errors_upper = list(gene1_errors[1]) + list(gene2_errors[1])
    all_errors_lower = list(gene1_errors[0]) + list(gene2_errors[0])
    max_val = max([v + e for v, e in zip(all_values, all_errors_upper)]) + 5
    min_val = min([v - e for v, e in zip(all_values, all_errors_lower)]) - 5
    ax.set_xlim(min_val, max_val)
    
    plt.tight_layout()
    
    save_and_show(output_filename, add_pdf_extension=True)
    
    return fig


def plot_ana_forest(results, output_filename=None, main_color='#ABC5C4'):
    """
    Create multi-panel forest plot for antibody logistic regression results.
    One subplot per antibody showing odds ratios with 95% CI.
    
    Parameters:
        results: list of result dicts from ana_logistic_regression
        output_filename: str, optional filename for saving (without extension)
        main_color: str, hex color for markers and CI lines
    
    Returns:
        matplotlib Figure object
    """
    
    antibody_results = defaultdict(list)
    for result in results:
        antibody_results[result['antibody']].append(result)
    
    sorted_antibodies = sorted(antibody_results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    subtype_order = [SUBTYPE_A, SUBTYPE_B, 'Combined']
    
    for idx, antibody in enumerate(sorted_antibodies[:4]):
        ax = axes[idx]
        antibody_data = antibody_results[antibody]
        
        antibody_data.sort(key=lambda x: subtype_order.index(x['subtype']) if x['subtype'] in subtype_order else 999)
        
        y_positions = np.arange(len(antibody_data))
        odds_ratios = [result['odds_ratio'] for result in antibody_data]
        ci_lower = [result['ci_lower'] for result in antibody_data]
        ci_upper = [result['ci_upper'] for result in antibody_data]
        pvalues = [result['pvalue'] for result in antibody_data]
        sample_sizes = [result['n'] for result in antibody_data]
        
        # Y-axis labels with sample size
        labels = [f"{result['display_name']} (n={result['n']})" for result in antibody_data]
        
        for i, (or_val, ci_low, ci_high, pval) in enumerate(zip(odds_ratios, ci_lower, ci_upper, pvalues)):
            ax.plot([ci_low, ci_high], [i, i], color=main_color, linewidth=4, alpha=0.7, zorder=2)
            ax.scatter(or_val, i, s=200, color=main_color, zorder=3, alpha=0.8)
            
            if pval < 0.05:
                ax.text(or_val, i, '*', fontsize=20, ha='center', va='center', 
                       color='black', fontweight='bold', zorder=4)
        
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, zorder=1)
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=18)
        ax.set_xlabel('Odds Ratio (95% CI)', fontsize=18)
        ax.set_title(f'{antibody.upper()}', fontsize=20, fontweight='bold', pad=20)
        
        ax.set_xscale('log')
        ax.grid(False)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_ylim(-0.5, len(antibody_data) - 0.5)
        ax.invert_yaxis()
    
    for idx in range(len(sorted_antibodies), 4):
        axes[idx].set_visible(False)
    
    plt.suptitle('Antibody Analysis: Forest Plot of Odds Ratios', 
                fontsize=22, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, wspace=0.4)
    
    save_and_show(output_filename, add_pdf_extension=True)
    
    return fig

def plot_chip_tet2_triple_metrics(plot_df, group_order, stats_results=None, 
                                   metrics=None, figsize=(20, 6), output_filename=None):
    """
    Create three horizontal subplots showing distribution of clinical metrics
    across CHIP status, subtype groups, and TET2-specific groups.
    
    Parameters:
        plot_df: DataFrame with 'group' column from prepare_chip_tet2_groups
        group_order: list of group names in plotting order
        stats_results: dict from compute_chip_tet2_group_statistics (optional, for reference)
        metrics: list of metric column names (default: DLCO, FVC, MRSS)
        figsize: tuple for figure size
        output_filename: str, optional filename for saving (without extension)
    
    Returns:
        fig, axes: matplotlib figure and axes objects
    """
    if metrics is None:
        metrics = [DLCO_PCT_COL, FVC_PCT_COL, SKIN_SCORE_COL]
    
    y_labels = [DLCO_PCT_COL, FVC_PCT_COL, SKIN_SCORE_COL]
    
    group_colors_styles = {
        f'Non-CHIP {SUBTYPE_A}': {'color': 'gray', 'style': 'strip', 'size': 4, 'alpha': 0.7},
        f'CHIP {SUBTYPE_A}': {'color': 'darkblue', 'style': 'swarm', 'size': 6, 'alpha': 1.0},
        f'{SUBTYPE_A} TET2': {'color': '#89C4CC', 'style': 'swarm', 'size': 6, 'alpha': 1.0},
        f'Non-CHIP {SUBTYPE_B}': {'color': 'gray', 'style': 'strip', 'size': 4, 'alpha': 0.7},
        f'CHIP {SUBTYPE_B}': {'color': 'darkred', 'style': 'swarm', 'size': 6, 'alpha': 1.0},
        f'{SUBTYPE_B} TET2': {'color': '#3B4431', 'style': 'swarm', 'size': 6, 'alpha': 1.0}
    }
    
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
    
    for i, (metric, y_label) in enumerate(zip(metrics, y_labels)):
        ax = axes[i]
        metric_df = plot_df.dropna(subset=[metric])
        
        if metric_df.empty:
            ax.text(0.5, 0.5, f'No data for {metric}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=16)
            ax.set_title(y_label, fontsize=20)
            continue
        
        sns.boxplot(
            data=metric_df, x='group', y=metric, order=group_order,
            color='white', showfliers=False, width=0.5, ax=ax
        )
        
        for patch in ax.patches:
            patch.set_edgecolor('black')
            patch.set_linewidth(2)
        
        for line in ax.lines:
            line.set_color('black')
            line.set_linewidth(2)
        
        for group in group_order:
            group_data_subset = metric_df[metric_df['group'] == group]
            if not group_data_subset.empty:
                style_info = group_colors_styles[group]
                
                if style_info['style'] == 'strip':
                    sns.stripplot(
                        data=group_data_subset, x='group', y=metric, order=group_order,
                        color=style_info['color'], size=style_info['size'],
                        alpha=style_info['alpha'], ax=ax
                    )
                else:
                    sns.swarmplot(
                        data=group_data_subset, x='group', y=metric, order=group_order,
                        color=style_info['color'], size=style_info['size'],
                        alpha=style_info['alpha'], ax=ax
                    )
        
        ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        
        ax.set_xlabel('')
        ax.set_ylabel(y_label, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='x', rotation=45)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    save_and_show(output_filename, add_pdf_extension=True)
    
    return fig, axes
