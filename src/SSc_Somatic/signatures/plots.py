"""
Plotting functions for SSc mutational signature analysis.
Functions organized by notebook workflow in chronological order of use.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as scipy_stats
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from ..settings import cohort_values


# Study settings. These start empty and are filled from the study
# configuration by apply_settings(), which SSc_Somatic.configure() calls.
RACE_ORDER = None
RACE_COLORS = None
DISEASE_ORDER = None
DISEASE_COLORS = None
DEFAULT_COLORS = None

# Cohort labels and metadata column names, also filled by apply_settings()
AGE_COL = None
CONTROL_KEY = None
CONTROL_LABEL = None
DLCO_PCT_COL = None
DURATION_COL = None
ETHNICITY_COL = None
FVC_PCT_COL = None
GROUP_COL = None
SKIN_SCORE_COL = None
SUBTYPE_A = None
SUBTYPE_A_KEY = None
SUBTYPE_B = None
SUBTYPE_B_KEY = None


def apply_settings(config):
    """
    Fill this module's study settings from a StudyConfig.

    Called by SSc_Somatic.configure(); notebooks do not need to call it directly.
    """
    global RACE_ORDER, RACE_COLORS, DISEASE_ORDER, DISEASE_COLORS, DEFAULT_COLORS
    RACE_ORDER = list(config.require('cohort', 'race_order'))
    RACE_COLORS = dict(config.require('cohort', 'race_colors'))
    DISEASE_ORDER = list(config.require('cohort', 'group_order'))
    DISEASE_COLORS = dict(config.require('cohort', 'group_colors'))
    globals().update(cohort_values(config))

    # Box colors keyed by lowercase group name, so they depend on the cohort labels
    DEFAULT_COLORS = {
        CONTROL_KEY: '#239A8B',
        SUBTYPE_A_KEY: 'darkblue',
        SUBTYPE_B_KEY: 'darkred',
        'sbs': 'lightblue',
        'indel': 'darkcyan',
        'age_group': 'lightgray'
    }


def _get_significance_marker(p_value):
    """Return significance marker string based on p-value."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    return ''


def _save_figure(fig, output_filename):
    """Save figure as PDF."""
    if output_filename:
        fig.savefig(f"{output_filename}.pdf", format='pdf', bbox_inches='tight', dpi=300)


def plot_variant_counts_by_group(variant_data, stats_df, output_filename=None):
    """
    Create boxplot of variant counts by sample group and mutation type.
    
    Parameters:
        variant_data: dict, output from compute_variant_stats_by_group
        stats_df: DataFrame, statistics from compute_variant_stats_by_group
        output_filename: str, base filename for saving (default: None)
    
    Returns:
        Tuple of (fig, ax)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    data_for_boxplot = [
        variant_data['sbs_normal'], variant_data['sbs_limited'], variant_data['sbs_diffuse'],
        variant_data['indel_normal'], variant_data['indel_limited'], variant_data['indel_diffuse']
    ]
    n_control_sbs = len(variant_data['sbs_normal'])
    n_limited_sbs = len(variant_data['sbs_limited'])
    n_diffuse_sbs = len(variant_data['sbs_diffuse'])
    n_control_indel = len(variant_data['indel_normal'])
    n_limited_indel = len(variant_data['indel_limited'])
    n_diffuse_indel = len(variant_data['indel_diffuse'])
    labels = [
        f'{CONTROL_LABEL}\n(n={n_control_sbs})', f'{SUBTYPE_A}\n(n={n_limited_sbs})', f'{SUBTYPE_B}\n(n={n_diffuse_sbs})',
        f'{CONTROL_LABEL}\n(n={n_control_indel})', f'{SUBTYPE_A}\n(n={n_limited_indel})', f'{SUBTYPE_B}\n(n={n_diffuse_indel})'
    ]
    box_plot = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
    for i, patch in enumerate(box_plot['boxes']):
        if i < 3:
            patch.set_facecolor(DEFAULT_COLORS['sbs'])
        else:
            patch.set_facecolor(DEFAULT_COLORS['indel'])
    for i, median_line in enumerate(box_plot['medians']):
        median_line.set_color('black')
        median_line.set_linewidth(2)
    ax.axvline(x=3.5, color='gray', linestyle='--', alpha=0.5)
    ylim = ax.get_ylim()
    ax.text(2, ylim[1] * 0.95, 'SBS', ha='center', fontsize=12, fontweight='bold')
    ax.text(5, ylim[1] * 0.95, 'Indel', ha='center', fontsize=12, fontweight='bold')
    ax.set_xlabel('Sample Groups', fontsize=12)
    ax.set_ylabel('Total Variant Count per Sample', fontsize=12)
    ax.set_title('Distribution of Variant Counts by Sample Groups and Mutation Type', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    _save_figure(fig, output_filename)
    return fig, ax


def plot_zscore_by_subtype_age(mutation_data, comparison_results, age_groups=None, 
                               figsize=(15, 12), output_filename=None):
    """
    Create boxplot of z-scored mutation counts by subtype and age group.
    
    Parameters:
        mutation_data: dict, output from compute_zscore_mutation_data
        comparison_results: dict, output from compute_subtype_comparison_stats
        age_groups: list, age groups to display (default: uses groups from mutation_data)
        figsize: tuple, figure size (default: (15, 12))
        output_filename: str, base filename for saving (default: None)
    
    Returns:
        Tuple of (fig, axes)
    """
    subtypes = mutation_data.get('_subtypes', [])
    if age_groups is None:
        age_groups = mutation_data.get('_age_groups', ['20-29', '30-39', '40-49'])
    subtype_display = comparison_results.get('subtype_display_map', {})
    subtype_colors = {}
    for subtype in subtypes:
        subtype_lower = str(subtype).lower()
        if CONTROL_KEY in subtype_lower or subtype_lower == 'hc':
            subtype_colors[subtype] = DEFAULT_COLORS[CONTROL_KEY]
        elif SUBTYPE_A_KEY in subtype_lower:
            subtype_colors[subtype] = DEFAULT_COLORS[SUBTYPE_A_KEY]
        elif SUBTYPE_B_KEY in subtype_lower:
            subtype_colors[subtype] = DEFAULT_COLORS[SUBTYPE_B_KEY]
        else:
            subtype_colors[subtype] = 'gray'
    mutation_types = [k for k in mutation_data.keys() if not k.startswith('_')]
    n_plots = len(mutation_types)
    fig_height = figsize[1] * (n_plots / 3)
    fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], fig_height))
    if n_plots == 1:
        axes = [axes]
    for idx, mut_type in enumerate(mutation_types):
        ax = axes[idx]
        mut_info = mutation_data[mut_type]
        data_df = mut_info['zscore']
        mean_val = mut_info['mean']
        std_val = mut_info['std']
        plot_data = []
        plot_colors = []
        plot_positions = []
        position = 1
        n_boxes_per_subtype = 1 + len(age_groups)
        for i, subtype in enumerate(subtypes):
            all_subset = data_df[data_df[GROUP_COL] == subtype]
            plot_data.append(all_subset['zscore'].values if len(all_subset) > 0 else np.array([]))
            plot_colors.append(subtype_colors[subtype])
            plot_positions.append(position)
            position += 1
            for age_group in age_groups:
                age_subset = data_df[(data_df[GROUP_COL] == subtype) & 
                                    (data_df['age_group'] == age_group)]
                plot_data.append(age_subset['zscore'].values if len(age_subset) > 0 else np.array([]))
                plot_colors.append(DEFAULT_COLORS['age_group'])
                plot_positions.append(position)
                position += 1
            if i < len(subtypes) - 1:
                position += 1
        box_plot = ax.boxplot(plot_data, positions=plot_positions, patch_artist=True, widths=0.8,
                             flierprops=dict(marker='o', markersize=3, markerfacecolor='black',
                                           markeredgecolor='black', alpha=0.8))
        for patch, color in zip(box_plot['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for median_line in box_plot['medians']:
            median_line.set_color('black')
            median_line.set_linewidth(2)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
        separator_positions = []
        pos = n_boxes_per_subtype + 0.5
        for i in range(len(subtypes) - 1):
            separator_positions.append(pos)
            pos += n_boxes_per_subtype + 1
        for sep_pos in separator_positions:
            ax.axvline(x=sep_pos, color='gray', linestyle='-', alpha=0.5, linewidth=1)
        ax.set_title(f'{mut_type}\n(Mean={mean_val:.1f}, SD={std_val:.1f})', fontweight='bold', fontsize=14)
        ax.set_ylabel('Z-Score', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
        x_labels = []
        x_positions = []
        pos = 1
        for i, subtype in enumerate(subtypes):
            n_all = len(data_df[data_df[GROUP_COL] == subtype])
            x_labels.append(f"All\n(n={n_all})")
            x_positions.append(pos)
            pos += 1
            for age_group in age_groups:
                n_age = len(data_df[(data_df[GROUP_COL] == subtype) & 
                                   (data_df['age_group'] == age_group)])
                x_labels.append(f"{age_group}\n(n={n_age})")
                x_positions.append(pos)
                pos += 1
            if i < len(subtypes) - 1:
                pos += 1
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha='right')
        ylim = ax.get_ylim()
        subtype_label_positions = []
        pos = 1
        for i, subtype in enumerate(subtypes):
            center_pos = pos + (n_boxes_per_subtype - 1) / 2
            subtype_label_positions.append((center_pos, subtype_display.get(subtype, subtype)))
            pos += n_boxes_per_subtype + (1 if i < len(subtypes) - 1 else 0)
        for center_pos, subtype_label in subtype_label_positions:
            ax.text(center_pos, ylim[1] * 0.92, subtype_label, ha='center', va='top',
                   fontweight='bold', fontsize=12)
    legend_elements = [
        Patch(facecolor=subtype_colors[subtype], alpha=0.6, 
              label=f'{subtype_display.get(subtype, subtype)}')
        for subtype in subtypes
    ]
    legend_elements.append(Patch(facecolor=DEFAULT_COLORS['age_group'], alpha=0.6, label='Age-specific'))
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)
    fig.suptitle('Z-Scored Mutation Counts by Subtype and Age Group', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_figure(fig, output_filename)
    return fig, axes


def plot_signature_distributions(df, signatures_list, sig_stats, output_filename=None):
    """
    Create violin plots with strip plots for signature activity distributions.
    
    Parameters:
        df: DataFrame, annotated signature data with statustype_subtype column
        signatures_list: list, signature column names to plot
        sig_stats: dict, output from compute_signature_stats
        output_filename: str, base filename for saving (default: None)
    
    Returns:
        Tuple of (fig, axes)
    """
    n_sigs = len(signatures_list)
    fig, axes = plt.subplots(1, n_sigs, figsize=(5 * n_sigs, 6))
    if n_sigs == 1:
        axes = [axes]
    order = sorted(df[GROUP_COL].dropna().unique())
    for i, sig in enumerate(signatures_list):
        ax = axes[i]
        sns.violinplot(data=df, x=GROUP_COL, y=sig, ax=ax, order=order)
        sns.stripplot(data=df, x=GROUP_COL, y=sig, ax=ax, order=order,
                     size=4, jitter=True, color='darkblue', alpha=0.6)
        ax.set_title(f'{sig} Activity Distribution', fontweight='bold', fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('Mutation Count', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        new_labels = []
        for subtype in order:
            n = sig_stats[sig]['n_per_subtype'].get(subtype, 0)
            med = sig_stats[sig]['medians'].get(subtype, 0)
            new_labels.append(f'{subtype}\n(n={n}, med={med:.0f})')
        ax.set_xticklabels(new_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save_figure(fig, output_filename)
    return fig, axes


def plot_clinical_effect_bars(effect_df, continuous_vars=None, binary_vars=None,
                               colors=None, figsize=(16, 6), output_filename=None):
    """
    Create grouped bar chart of effect sizes for continuous and binary variables.
    
    Parameters:
        effect_df: DataFrame, output from compute_signature_clinical_effects
        continuous_vars: list, continuous variable names (default: standard set)
        binary_vars: list, binary variable names (default: standard set)
        colors: list, colors for each signature (default: standard palette)
        figsize: tuple, figure size (default: (16, 6))
        output_filename: str, base filename for saving (default: None)
    
    Returns:
        Tuple of (fig, (ax1, ax2))
    """
    if continuous_vars is None:
        continuous_vars = [DLCO_PCT_COL, FVC_PCT_COL, SKIN_SCORE_COL, DURATION_COL, AGE_COL]
    if binary_vars is None:
        binary_vars = ['Black', ETHNICITY_COL, 'ana', 'anti-centromere', 'anti-topo', 'anti-RNA polymerase III']
    pivot_df = effect_df.pivot(index='variable', columns='signature', values='effect_size')
    pivot_df = pivot_df.fillna(0)
    available_continuous = [v for v in continuous_vars if v in pivot_df.index]
    available_binary = [v for v in binary_vars if v in pivot_df.index]
    signatures = pivot_df.columns.tolist()
    if colors is None:
        colors = ['#5F7C5B', '#B8A57A', '#9BA3B5', '#7DBDC4', '#3D6B6B']
    while len(colors) < len(signatures):
        colors.append('gray')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    label_map = {'Black': 'Race - Black', ETHNICITY_COL: 'Ethnicity - Hisp/Latin'}
    bar_width = 0.15
    if available_continuous:
        pivot_continuous = pivot_df.reindex(available_continuous)
        x_pos = np.arange(len(available_continuous))
        for i, sig in enumerate(signatures):
            offset = (i - len(signatures)/2 + 0.5) * bar_width
            ax1.bar(x_pos + offset, pivot_continuous[sig], bar_width,
                   label=sig, color=colors[i], alpha=0.9, edgecolor='black', linewidth=0.5)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)
        ax1.set_xlabel('Clinical Variables', fontsize=12, fontweight='bold')
        ax1.set_ylabel("Cohen's d", fontsize=12, fontweight='bold')
        ax1.set_title('Continuous Variables', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([label_map.get(var, var) for var in available_continuous],
                           rotation=45, ha='right', fontsize=10)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
    if available_binary:
        pivot_binary = pivot_df.reindex(available_binary)
        x_pos = np.arange(len(available_binary))
        for i, sig in enumerate(signatures):
            offset = (i - len(signatures)/2 + 0.5) * bar_width
            ax2.bar(x_pos + offset, pivot_binary[sig], bar_width,
                   label=sig, color=colors[i], alpha=0.9, edgecolor='black', linewidth=0.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)
        ax2.set_xlabel('Clinical Variables', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Log Odds Ratio', fontsize=12, fontweight='bold')
        ax2.set_title('Categorical Variables', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([label_map.get(var, var) for var in available_binary],
                           rotation=45, ha='right', fontsize=10)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=len(signatures), fontsize=10, frameon=True)
    plt.tight_layout()
    _save_figure(fig, output_filename)
    return fig, (ax1, ax2)



def plot_volcano_disease_comparisons(
    volcano_df,
    fdr_threshold=0.1,
    p_threshold_line=0.1,
    min_fold_change=None,
    figsize=(12, 8),
    marker_size=80,
    label_fontsize=8,
    fc_label_threshold=1.2,
    pval_label_threshold=0.75,
    output_filename=None
):
    """
    Volcano plot for disease comparisons with race, comparison, and cohort encoding.
    
    Parameters:
        volcano_df: DataFrame, output from prepare_volcano_disease_data
        fdr_threshold: float, FDR threshold for marking significant points (default: 0.1)
        p_threshold_line: float, nominal p-value for horizontal reference line (default: 0.1)
        min_fold_change: float, filter to fold changes above this value (default: None)
        figsize: tuple, figure dimensions (default: (12, 8))
        marker_size: float, base marker size (default: 80)
        label_fontsize: float, font size for signature labels (default: 8)
        fc_label_threshold: float, label signatures with FC above this (default: 1.2)
        pval_label_threshold: float, label signatures with -log10(p) above this (default: 0.75)
        output_filename: str, base filename for saving (default: None)
    
    Returns:
        Tuple of (fig, ax)
    """
    comparison_markers = {
        f'{SUBTYPE_B} vs {CONTROL_LABEL}': 'o',
        f'{SUBTYPE_A} vs {CONTROL_LABEL}': 's',
        f'{SUBTYPE_B} vs {SUBTYPE_A}': '^',
        f'Pooled SSc vs {CONTROL_LABEL}': 'D',
        f'{SUBTYPE_A}+{SUBTYPE_B} vs {CONTROL_LABEL}': 'D'
    }
    
    plot_df = volcano_df[volcano_df['p_value'].notna()].copy()
    
    if min_fold_change is not None:
        plot_df = plot_df[plot_df['fold_change'] >= min_fold_change].copy()
    
    if len(plot_df) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=ax.transAxes)
        _save_figure(fig, output_filename)
        return fig, ax
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for _, row in plot_df.iterrows():
        race = row['stratum']
        comparison = row['comparison_label']
        cohort = row['cohort']
        
        color = RACE_COLORS.get(race, '#333333')
        marker = comparison_markers.get(comparison, 'o')
        
        if cohort == 'Full':
            facecolor = color
            edgecolor = color
            linewidth = 1.0
        else:
            facecolor = color
            edgecolor = 'black'
            linewidth = 2.5
        
        ax.scatter(
            row['fold_change'], row['neg_log10_p'],
            c=[facecolor], edgecolors=[edgecolor],
            marker=marker, s=marker_size, linewidths=linewidth,
            alpha=0.85, zorder=3
        )
    
    neg_log10_threshold = -np.log10(p_threshold_line)
    ax.axhline(y=neg_log10_threshold, color='darkred', linestyle='--', linewidth=1.5, zorder=1)
    
    ax.axvline(x=1, color='lightgray', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
    
    label_mask = (plot_df['fold_change'] > fc_label_threshold) | (plot_df['neg_log10_p'] > pval_label_threshold)
    for _, row in plot_df[label_mask].iterrows():
        ax.annotate(
            row['signature'],
            (row['fold_change'], row['neg_log10_p']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=label_fontsize, alpha=0.9
        )
    
    legend_elements = []
    legend_elements.append(Line2D([0], [0], marker='', color='w', label='Race/Ethnicity:', markersize=0))
    for race, color in RACE_COLORS.items():
        legend_elements.append(Line2D(
            [0], [0], marker='o', color='w',
            markerfacecolor=color, markeredgecolor=color,
            markersize=10, label=race
        ))
    
    legend_elements.append(Line2D([0], [0], marker='', color='w', label='', markersize=0))
    legend_elements.append(Line2D([0], [0], marker='', color='w', label='Comparison:', markersize=0))
    for comp, marker in comparison_markers.items():
        if comp in plot_df['comparison_label'].values:
            legend_elements.append(Line2D(
                [0], [0], marker=marker, color='w',
                markerfacecolor='black', markeredgecolor='black',
                markersize=10, label=comp
            ))
    
    legend_elements.append(Line2D([0], [0], marker='', color='w', label='', markersize=0))
    legend_elements.append(Line2D([0], [0], marker='', color='w', label='Cohort:', markersize=0))
    legend_elements.append(Line2D(
        [0], [0], marker='o', color='w',
        markerfacecolor='black', markeredgecolor='black',
        markersize=10, linewidth=1.0, label='Full cohort'
    ))
    legend_elements.append(Line2D(
        [0], [0], marker='o', color='w',
        markerfacecolor='white', markeredgecolor='black',
        markersize=10, markeredgewidth=2.5, label='Baseline (thick edge)'
    ))
    
    legend_elements.append(Line2D([0], [0], marker='', color='w', label='', markersize=0))
    legend_elements.append(Line2D(
        [0], [0], color='darkred', linestyle='--',
        linewidth=1.5, label=f'p < {p_threshold_line}'
    ))
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, fontsize=9)
    
    ax.set_xlabel('Fold Change (Disease / Reference)', fontsize=12)
    ax.set_ylabel('-log$_{10}$(p-value)', fontsize=12)
    ax.set_title('Mutational Signatures Elevated in SSc', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    x_min, x_max = plot_df['fold_change'].min(), plot_df['fold_change'].max()
    x_range = x_max - x_min if x_max > x_min else 0.1
    ax.set_xlim(max(0.9, x_min - 0.1 * x_range), x_max + 0.15 * x_range)
    y_max = plot_df['neg_log10_p'].max()
    ax.set_ylim(-0.1, y_max * 1.1)
    
    plt.tight_layout()
    _save_figure(fig, output_filename)
    
    return fig, ax

def plot_clinical_effect_heatmap(
    effect_size_results,
    signatures=None,
    continuous_vars=None,
    categorical_vars=None,
    subtypes=None,
    ssc_specific_vars=None,
    figsize=(16, 10),
    vmin=-3,
    vmax=3,
    output_filename=None
):
    """
    Create heatmap of clinical variable associations with signature burden by disease subtype.
    
    Parameters:
        effect_size_results: dict, output from run_clinical_effect_analysis_by_subtype
        signatures: list, signature columns to display (default: ['SBS96A', 'SBS96B', 'ID83A'])
        continuous_vars: list, continuous clinical variables (default: standard set)
        categorical_vars: list, categorical clinical variables (default: standard set)
        subtypes: list, disease subtypes in column order (default: ['Control', 'Limited', 'Diffuse'])
        ssc_specific_vars: list, variables only applicable to SSc (default: standard set)
        figsize: tuple, figure dimensions (default: (16, 10))
        vmin: float, minimum effect size for colormap (default: -3)
        vmax: float, maximum effect size for colormap (default: 3)
        output_filename: str, base filename for saving (default: None)
    
    Returns:
        Tuple of (fig, ax)
    """
    if signatures is None:
        signatures = ['SBS96A', 'SBS96B', 'ID83A']
    if continuous_vars is None:
        continuous_vars = [AGE_COL, DURATION_COL, SKIN_SCORE_COL, DLCO_PCT_COL, FVC_PCT_COL]
    if categorical_vars is None:
        categorical_vars = ['Black', ETHNICITY_COL, 'ana', 'anti-centromere', 'anti-topo']
    if subtypes is None:
        subtypes = [CONTROL_LABEL, SUBTYPE_A, SUBTYPE_B]
    if ssc_specific_vars is None:
        ssc_specific_vars = [
            DURATION_COL, SKIN_SCORE_COL, DLCO_PCT_COL, FVC_PCT_COL,
            'anti-centromere', 'anti-topo', 'anti-RNA polymerase III'
        ]
    
    all_vars = continuous_vars + categorical_vars
    n_vars = len(all_vars)
    n_sigs = len(signatures)
    n_subtypes = len(subtypes)
    n_cols = n_sigs * n_subtypes
    
    matrix = np.full((n_vars, n_cols), np.nan)
    pval_matrix = np.full((n_vars, n_cols), np.nan)
    
    for subtype_idx, subtype in enumerate(subtypes):
        subtype_data = effect_size_results.get(subtype, {})
        if subtype_data is None:
            continue
        effect_df = subtype_data.get('effect_df', pd.DataFrame())
        
        for sig_idx, sig in enumerate(signatures):
            col_idx = subtype_idx * n_sigs + sig_idx
            
            for var_idx, var in enumerate(all_vars):
                if subtype == CONTROL_LABEL and var in ssc_specific_vars:
                    continue
                
                if len(effect_df) > 0:
                    row = effect_df[(effect_df['signature'] == sig) & (effect_df['variable'] == var)]
                    if len(row) > 0:
                        matrix[var_idx, col_idx] = row['effect_size'].values[0]
                        if 'p_value_fdr' in row.columns:
                            pval_matrix[var_idx, col_idx] = row['p_value_fdr'].values[0]
                        elif 'p_value' in row.columns:
                            pval_matrix[var_idx, col_idx] = row['p_value'].values[0]
    
    colors = ['#2a2d7c', 'white', '#8a181a']
    cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=256)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    masked_matrix = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_facecolor('white')
    
    for i in range(n_vars):
        for j in range(n_cols):
            if not np.isnan(matrix[i, j]) and not np.isnan(pval_matrix[i, j]):
                pval = pval_matrix[i, j]
                stars = _get_significance_marker(pval)
                if stars:
                    bg_val = matrix[i, j]
                    text_color = 'white' if abs(bg_val) > (vmax * 0.6) else 'black'
                    ax.text(j, i, stars, ha='center', va='center', fontsize=9, fontweight='bold', color=text_color)
    
    for i in range(n_vars + 1):
        ax.axhline(y=i - 0.5, color='white', linewidth=1)
    for j in range(n_cols + 1):
        ax.axvline(x=j - 0.5, color='white', linewidth=1)
    
    for subtype_idx in range(1, n_subtypes):
        x_pos = subtype_idx * n_sigs - 0.5
        ax.axvline(x=x_pos, color='black', linewidth=2)
    
    gap_y = len(continuous_vars) - 0.5
    ax.axhline(y=gap_y, color='black', linewidth=1.5, linestyle='-')
    
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(all_vars, fontsize=10)
    
    x_labels = signatures * n_subtypes
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    
    for subtype_idx, subtype in enumerate(subtypes):
        x_center = subtype_idx * n_sigs + (n_sigs - 1) / 2
        ax.text(x_center, -1.2, subtype, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Effect Size\n(Cohen's d / Log OR)", fontsize=10)
    
    ax.set_title('Clinical Associations with High Signature Burden', fontsize=14, fontweight='bold', pad=30)
    
    fig.text(0.02, 0.02, '* FDR < 0.05   ** FDR < 0.01   *** FDR < 0.001', fontsize=9, style='italic')
    
    plt.tight_layout()
    _save_figure(fig, output_filename)
    
    return fig, ax

def plot_signature_clinical_correlations(
    correlation_results,
    cohort='full',
    figsize_per_panel=(3.5, 3.5),
    marker_size=40,
    alpha=0.25,
    show_ci=True,
    ci_alpha=0.15,
    output_filename=None
):
    """
    Create scatter plots with linear trend lines for signature-clinical correlations.
    
    Statistics use Spearman rho; trend lines are linear regression for visualization.
    
    Parameters:
        correlation_results: dict, output from compute_signature_clinical_correlations
        cohort: str, 'full' or 'baseline' (default: 'full')
        figsize_per_panel: tuple, size per scatter panel (default: (3.5, 3.5))
        marker_size: float, scatter point size (default: 40)
        alpha: float, scatter point transparency (default: 0.25)
        show_ci: bool, show 95% confidence interval bands (default: True)
        ci_alpha: float, CI band transparency (default: 0.15)
        output_filename: str, base filename for saving (default: None)
    
    Returns:
        Tuple of (fig, axes)
    """
    config = correlation_results['config']
    signatures = config['signatures']
    clinical_vars = config['clinical_vars']
    race_groups = config['race_groups']
    
    if cohort == 'full':
        data_df = correlation_results['full_cohort_data']
        cohort_name = 'Full Cohort'
    elif cohort == 'baseline':
        data_df = correlation_results['baseline_cohort_data']
        if data_df is None or len(data_df) == 0:
            raise ValueError("No baseline cohort data available")
        cohort_name = f"Baseline (Duration <= {config['baseline_duration']} years)"
    else:
        raise ValueError(f"Invalid cohort: {cohort}. Use 'full' or 'baseline'")
    
    stats_df = correlation_results['stats_df']
    cohort_stats = stats_df[stats_df['cohort'] == cohort_name]
    
    n_rows = len(clinical_vars)
    n_cols = len(signatures)
    
    fig_width = figsize_per_panel[0] * n_cols + 2
    fig_height = figsize_per_panel[1] * n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    
    for row_idx, clin_var in enumerate(clinical_vars):
        for col_idx, sig in enumerate(signatures):
            ax = axes[row_idx, col_idx]
            
            for race in race_groups:
                subset = data_df[
                    (data_df['race_eth'] == race) &
                    data_df[sig].notna() &
                    data_df[clin_var].notna()
                ].copy()
                
                if len(subset) < 3:
                    continue
                
                x = subset[sig].values
                y = subset[clin_var].values
                color = RACE_COLORS.get(race, '#333333')
                
                ax.scatter(x, y, c=color, s=marker_size, alpha=alpha, edgecolors='none', zorder=2)
                
                stat_row = cohort_stats[
                    (cohort_stats['signature'] == sig) &
                    (cohort_stats['clinical_var'] == clin_var) &
                    (cohort_stats['race_eth'] == race)
                ]
                
                if len(stat_row) > 0 and not np.isnan(stat_row['slope'].values[0]):
                    slope = stat_row['slope'].values[0]
                    intercept = stat_row['intercept'].values[0]
                    
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, color=color, linewidth=2.5, linestyle='-', zorder=4)
                    
                    if show_ci:
                        n = len(x)
                        x_mean = np.mean(x)
                        ss_x = np.sum((x - x_mean) ** 2)
                        
                        residuals = y - (slope * x + intercept)
                        mse = np.sum(residuals ** 2) / (n - 2) if n > 2 else 0
                        
                        if mse > 0 and ss_x > 0:
                            se_line = np.sqrt(mse * (1/n + (x_line - x_mean)**2 / ss_x))
                            t_crit = scipy_stats.t.ppf(0.975, n - 2)
                            
                            ci_upper = y_line + t_crit * se_line
                            ci_lower = y_line - t_crit * se_line
                            
                            ax.fill_between(x_line, ci_lower, ci_upper, color=color, alpha=ci_alpha, zorder=3)
            
            if row_idx == n_rows - 1:
                ax.set_xlabel('Mutation Count', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(clin_var, fontsize=10)
            if row_idx == 0:
                ax.set_title(sig, fontsize=11, fontweight='bold')
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RACE_COLORS.get(race, '#333333'),
               markersize=10, label=race)
        for race in race_groups
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=True, fontsize=10)
    
    fig.suptitle(f'Signature-Clinical Correlations (Spearman): {cohort_name}', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.92, top=0.95)
    
    _save_figure(fig, output_filename)
    
    return fig, axes

def plot_cosine_similarity_heatmap(cosine_results, orientation='wide', figsize=None,
                                    vmin=0.0, vmax=1.0, annotation_fontsize=11,
                                    cbar_label='Cosine Similarity', title=None,
                                    ax=None, output_filename=None):
    """
    Plot cosine similarity heatmap between de novo and COSMIC signatures.

    Parameters:
        cosine_results: dict, output from compute_cosine_similarity_matrix
        orientation: str, 'wide' (de novo rows, COSMIC cols) or 'tall' (transposed)
        figsize: tuple, figure dimensions (default: auto-calculated)
        vmin: float, colormap minimum (default: 0.0)
        vmax: float, colormap maximum (default: 1.0)
        annotation_fontsize: float, font size for cell values (default: 11)
        cbar_label: str, colorbar label (default: 'Cosine Similarity')
        title: str, plot title (default: auto-generated)
        ax: matplotlib Axes, if provided plot onto this axis (default: None)
        output_filename: str, base filename for saving (default: None)

    Returns:
        Tuple of (fig, ax)
    """
    matrix = cosine_results['matrix'].copy()
    if orientation == 'tall':
        matrix = matrix.T
    cmap = LinearSegmentedColormap.from_list('white_to_teal', ['#ffffff', '#2ca089', '#002119'], N=256)
    created_fig = False
    if ax is None:
        if figsize is None:
            n_cols, n_rows = matrix.shape[1], matrix.shape[0]
            figsize = (max(4, n_cols * 1.4 + 2.5), max(3, n_rows * 0.9 + 1.5))
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()
    ax.set_rasterized(True)
    im = ax.imshow(matrix.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            text_color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=annotation_fontsize, fontweight='bold', color=text_color)
    for i in range(matrix.shape[0] + 1):
        ax.axhline(y=i - 0.5, color='white', linewidth=2)
    for j in range(matrix.shape[1] + 1):
        ax.axvline(x=j - 0.5, color='white', linewidth=2)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=10)
    if title is None:
        if orientation == 'wide':
            title = 'De Novo vs COSMIC Signature Similarity'
        else:
            title = 'COSMIC vs De Novo Signature Similarity'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)
    _save_figure(fig, output_filename)
    return fig, ax

def plot_forest_unadjusted_and_adjusted(
    stats_df,
    signatures=None,
    clinical_vars=None,
    race_groups=None,
    figsize=(12, 6),
    point_size=150,
    dodge_width=0.25,
    fdr_threshold=0.1,
    nominal_threshold=0.05,
    output_filename=None
):
    """
    Forest plot with two rows: unadjusted correlations and age-adjusted standardized betas.

    Parameters:
        stats_df: DataFrame, output from compute_signature_clinical_forest_stats
        signatures: list, signature names in display order (default: auto-detected)
        clinical_vars: list, clinical variables for columns (default: auto-detected)
        race_groups: list, race/ethnicity groups (default: RACE_ORDER)
        figsize: tuple, figure dimensions (default: (12, 6))
        point_size: float, scatter point size (default: 150)
        dodge_width: float, vertical offset between race groups (default: 0.25)
        fdr_threshold: float, FDR threshold for ** (default: 0.1)
        nominal_threshold: float, nominal p threshold for * (default: 0.05)
        output_filename: str, base filename for saving (default: None)

    Returns:
        Tuple of (fig, axes)
    """
    if signatures is None:
        signatures = stats_df['signature'].unique().tolist()
    if clinical_vars is None:
        clinical_vars = stats_df['clinical_var'].unique().tolist()
    if race_groups is None:
        race_groups = RACE_ORDER

    race_colors = {
        'White Non-Hispanic': '#D72000',
        'White Hispanic': '#9093A2',
        'Black': '#132157',
    }

    analysis_types = ['Unadjusted', 'Age-Adjusted']
    row_labels = {'Unadjusted': 'Unadjusted', 'Age-Adjusted': 'Age-Adjusted'}
    effect_labels = {'Unadjusted': 'Spearman rho', 'Age-Adjusted': 'Std. Beta (signature)'}

    n_rows = len(analysis_types)
    n_cols = len(clinical_vars)
    n_races = len(race_groups)
    dodge_offsets = np.linspace(-dodge_width, dodge_width, n_races)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, sharex=False)

    for row_idx, atype in enumerate(analysis_types):
        for col_idx, clin_var in enumerate(clinical_vars):
            ax = axes[row_idx, col_idx]
            y_base = {sig: len(signatures) - 1 - i for i, sig in enumerate(signatures)}
            for race_idx, race in enumerate(race_groups):
                color = race_colors.get(race, '#333333')
                y_offset = dodge_offsets[race_idx]
                for sig in signatures:
                    row = stats_df[
                        (stats_df['analysis_type'] == atype) &
                        (stats_df['signature'] == sig) &
                        (stats_df['clinical_var'] == clin_var) &
                        (stats_df['race_eth'] == race)
                    ]
                    if len(row) == 0:
                        continue
                    row = row.iloc[0]
                    if pd.isna(row['effect']):
                        continue
                    y = y_base[sig] + y_offset
                    x = row['effect']
                    ci_lower = row.get('ci_lower', np.nan)
                    ci_upper = row.get('ci_upper', np.nan)
                    if pd.notna(ci_lower) and pd.notna(ci_upper):
                        ax.plot([ci_lower, ci_upper], [y, y],
                                color=color, linewidth=4, alpha=0.7, zorder=2)
                    ax.scatter(x, y, s=point_size, c=color, zorder=3, alpha=0.8, edgecolors='none')
                    sig_label = ''
                    if row.get('sig_fdr', False):
                        sig_label = '**'
                    elif row.get('sig_nominal', False):
                        sig_label = '*'
                    if sig_label:
                        ax.text(x, y, sig_label, fontsize=14, ha='center', va='center',
                                color='black', fontweight='bold', zorder=4)
            ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
            ax.set_yticks(list(range(len(signatures))))
            ax.set_yticklabels(signatures[::-1], fontsize=11)
            ax.set_ylim(-0.5, len(signatures) - 0.5)
            if row_idx == 0:
                clin_label = clin_var.replace(' %Predicted', '').replace(' %predicted', '').replace('%', '')
                ax.set_title(clin_label, fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(row_labels[atype], fontsize=11, fontweight='bold')
            if row_idx == n_rows - 1:
                ax.set_xlabel(effect_labels[atype], fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.tick_params(width=0.5)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=race_colors[r],
               markersize=10, label=r) for r in race_groups
    ]
    legend_elements.extend([
        Line2D([0], [0], marker='', color='w', label=f'** FDR < {fdr_threshold}'),
        Line2D([0], [0], marker='', color='w', label=f'*  p < {nominal_threshold}'),
    ])
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98),
               frameon=True, fontsize=9, handletextpad=0.5,
               edgecolor='lightgray', fancybox=False)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85, hspace=0.25, wspace=0.3)
    _save_figure(fig, output_filename)
    return fig, axes
