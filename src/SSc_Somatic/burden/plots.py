"""
Plotting functions for SSc mutation burden analysis pipeline.
Functions organized by notebook workflow in chronological order of use.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy import stats

from ..settings import cohort_values


OLIVE = '#5E6B2F'
OLIVE_LIGHT = '#8A9A5B'
OLIVE_PALE = '#C5CFA6'
GREY = '#9E9E9E'

# Study settings. These start empty and are filled from the study
# configuration by apply_settings(), which SSc_Somatic.configure() calls.
DISEASE_COLORS = None
DISEASE_COLORS_LOWER = None
RACE_COLORS = None
DISEASE_ORDER = None
RACE_ORDER = None
ONSET_DECADES = None


# Cohort labels and metadata column names, also filled by apply_settings()
CONTROL_KEY = None
CONTROL_LABEL = None
DURATION_COL = None
SUBTYPE_A = None
SUBTYPE_A_KEY = None
SUBTYPE_B = None
SUBTYPE_B_KEY = None


def apply_settings(config):
    """
    Fill this module's study settings from a StudyConfig.

    Called by SSc_Somatic.configure(); notebooks do not need to call it directly.
    """
    global DISEASE_COLORS, DISEASE_COLORS_LOWER, RACE_COLORS
    global DISEASE_ORDER, RACE_ORDER, ONSET_DECADES

    DISEASE_COLORS = dict(config.require('cohort', 'group_colors'))
    DISEASE_COLORS_LOWER = {k.lower(): v for k, v in DISEASE_COLORS.items()}
    RACE_COLORS = dict(config.require('cohort', 'race_colors'))
    DISEASE_ORDER = list(config.require('cohort', 'group_order'))
    RACE_ORDER = list(config.require('cohort', 'race_order'))
    # Onset decades are the age groups that have a broad age group assigned
    ONSET_DECADES = list(config.require('age_groups', 'broad_groups'))
    globals().update(cohort_values(config))


FONT_SIZE_MIN = 18
FONT_SIZE_AXIS_LABEL = 20
FONT_SIZE_TITLE = 22


def set_default_plot_style():
    """Configure matplotlib defaults for consistent styling across multipanel figures."""
    plt.rcParams.update({
        'font.size': FONT_SIZE_MIN,
        'axes.titlesize': FONT_SIZE_TITLE,
        'axes.labelsize': FONT_SIZE_AXIS_LABEL,
        'xtick.labelsize': FONT_SIZE_MIN,
        'ytick.labelsize': FONT_SIZE_MIN,
        'legend.fontsize': FONT_SIZE_MIN,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def plot_mutation_burden(analysis_df, y_col="mutation_burden", group_col="disease_group",
                         figsize=(4.5, 5), ylim=None, ylabel=None, title=None,
                         save_path=None, dpi=300):
    """
    Box plot with overlaid jittered strip points for mutation burden across
    disease groups, styled to match the reference figure.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Must contain columns specified by `y_col` and `group_col`.
        Expected group_col values: 'control', 'limited', 'diffuse'.
    y_col : str
        Column name for the continuous y-axis variable.
    group_col : str
        Column name containing disease group labels.
    figsize : tuple
        Figure width and height in inches.
    ylim : tuple or None
        Explicit (ymin, ymax). If None, determined automatically.
    ylabel : str or None
        Y-axis label. If None, no label is drawn.
    title : str or None
        Plot title. If None, no title is drawn.
    save_path : str or None
        If provided, saves the figure to this path.
    dpi : int
        Resolution for saved figure.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """
    # Mapping from internal group labels to display labels
    group_order = [CONTROL_KEY, SUBTYPE_A_KEY, SUBTYPE_B_KEY]
    display_labels = {CONTROL_KEY: "HC", SUBTYPE_A_KEY: "lcSSc", SUBTYPE_B_KEY: "dcSSc"}
    colors = {CONTROL_KEY: "#66C2A5", SUBTYPE_A_KEY: "#1B3A4B", SUBTYPE_B_KEY: "#8B1A1A"}

    # Subset and order the data
    plot_data = analysis_df[analysis_df[group_col].isin(group_order)].copy()
    grouped = [plot_data.loc[plot_data[group_col] == g, y_col].dropna().values
               for g in group_order]

    fig, ax = plt.subplots(figsize=figsize)

    # Box plot: 
    bp = ax.boxplot(
        grouped,
        positions=range(1, len(group_order) + 1),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
        boxprops=dict(linewidth=1),
    )

    # Style each box with a semi-transparent fill matching the group color
    for patch, group_key in zip(bp["boxes"], group_order):
        face = colors[group_key]
        patch.set_facecolor(face)
        patch.set_alpha(0.30)
        patch.set_edgecolor("black")

    # Overlay jittered strip points
    rng = np.random.default_rng(42)
    for i, (group_key, values) in enumerate(zip(group_order, grouped), start=1):
        jitter = rng.normal(loc=0, scale=0.12, size=len(values))
        ax.scatter(
            i + jitter,
            values,
            s=12,
            alpha=0.6,
            color=colors[group_key],
            edgecolors="none",
            zorder=3,
        )

    # Axis formatting
    ax.set_xticks(range(1, len(group_order) + 1))
    ax.set_xticklabels([display_labels[g] for g in group_order], fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ylim is not None:
        ax.set_ylim(ylim)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=13)

    # Legend matching the reference figure style
    legend_handles = [
        mpatches.Patch(facecolor=colors[g], label=display_labels[g])
        for g in group_order
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True,
              fontsize=10, edgecolor="lightgray")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax

def _save_figure(fig: plt.Figure, output_filename: str) -> None:
    """
    Save figure in PDF format.

    Parameters:
        fig: Figure, matplotlib figure object
        output_filename: str, base filename without extension
    """
    fig.savefig(f'{output_filename}.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_filename}.pdf')


def assign_onset_decade(age):
    """
    Assign an age to its decade bin.

    Parameters:
        age: float, age value to bin

    Returns:
        str or np.nan, decade label (e.g., '40-49') or NaN if outside 20-79 range
    """
    if pd.isna(age):
        return np.nan
    for start in range(20, 80, 10):
        if start <= age < start + 10:
            return f'{start}-{start+9}'
    return np.nan


def bootstrap_ci(data, n_bootstrap=1000, statistic='mean', ci=0.95):
    """
    Calculate bootstrap confidence interval for a statistic.

    Parameters:
        data: array-like, input values
        n_bootstrap: int, number of resamples (default: 1000)
        statistic: str, 'mean' or 'median' (default: 'mean')
        ci: float, confidence level (default: 0.95)

    Returns:
        dict with keys 'estimate', 'ci_lower', 'ci_upper', 'n'
    """
    data = np.array(data)
    data = data[~np.isnan(data)]
    n = len(data)

    if n == 0:
        return {'estimate': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'n': 0}

    stat_func = np.mean if statistic == 'mean' else np.median
    point_estimate = stat_func(data)

    if n < 3:
        return {'estimate': point_estimate, 'ci_lower': np.nan, 'ci_upper': np.nan, 'n': n}

    rng = np.random.default_rng(42)
    boot_stats = np.array([stat_func(rng.choice(data, size=n, replace=True)) for _ in range(n_bootstrap)])

    alpha = 1 - ci
    ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return {'estimate': point_estimate, 'ci_lower': ci_lower, 'ci_upper': ci_upper, 'n': n}


def calculate_control_regressions(df, race_groups=None, age_col='current_age',
                                   burden_col='vcf_snv_count', group_col='disease_group',
                                   race_col='race_ethnicity', min_n=20):
    """
    Calculate age vs mutation burden regression lines for control samples by race.

    Parameters:
        df: pd.DataFrame, analysis dataframe
        race_groups: list or None, race groups to include (default: RACE_ORDER)
        age_col: str, column for age (default: 'current_age')
        burden_col: str, column for mutation burden (default: 'vcf_snv_count')
        group_col: str, column for disease group (default: 'disease_group')
        race_col: str, column for race/ethnicity (default: 'race_ethnicity')
        min_n: int, minimum controls required for regression (default: 20)

    Returns:
        dict, {race: {'slope': float, 'intercept': float, 'n': int}}
    """
    if race_groups is None:
        race_groups = RACE_ORDER

    control_regressions = {}
    for race in race_groups:
        mask = (df[race_col] == race) & (df[group_col] == CONTROL_KEY)
        controls = df[mask].dropna(subset=[age_col, burden_col])

        if len(controls) >= min_n:
            X = controls[age_col].values
            y = controls[burden_col].values
            slope, intercept = np.polyfit(X, y, 1)
            control_regressions[race] = {'slope': slope, 'intercept': intercept, 'n': len(controls)}

    return control_regressions


def calculate_subtype_statistics(df, subtype, duration_filter=None, n_bootstrap=1000):
    """
    Calculate mean and 95% CI for mutation burden by race and onset decade for a disease subtype.

    Parameters:
        df: pd.DataFrame, must contain 'race_ethnicity', 'disease_group', 'age_at_onset', 'disease_duration', 'vcf_snv_count'
        subtype: str, 'limited', 'diffuse', or 'both'
        duration_filter: tuple or None, (min_duration, max_duration) or None for all
        n_bootstrap: int, bootstrap iterations (default: 1000)

    Returns:
        pd.DataFrame with columns: race_ethnicity, onset_age_group, onset_group_midpoint, n, mean_burden, ci_lower, ci_upper
    """
    df = df.copy()
    df['onset_age_group'] = df['age_at_onset'].apply(assign_onset_decade)

    if subtype == 'both':
        cases = df[df['disease_group'].isin([SUBTYPE_A_KEY, SUBTYPE_B_KEY])].copy()
    else:
        cases = df[df['disease_group'] == subtype].copy()

    if duration_filter is not None:
        min_dur, max_dur = duration_filter
        cases = cases[(cases['disease_duration'] >= min_dur) & (cases['disease_duration'] <= max_dur)]
    else:
        cases = cases[cases['disease_duration'].notna()]

    results = []
    for race in RACE_ORDER:
        race_cases = cases[cases['race_ethnicity'] == race]
        for onset_grp in ONSET_DECADES:
            cell_data = race_cases[race_cases['onset_age_group'] == onset_grp]['vcf_snv_count'].values
            ci_result = bootstrap_ci(cell_data, n_bootstrap=n_bootstrap, statistic='mean')
            results.append({
                'race_ethnicity': race,
                'onset_age_group': onset_grp,
                'onset_group_midpoint': int(onset_grp.split('-')[0]) + 5,
                'n': ci_result['n'],
                'mean_burden': ci_result['estimate'],
                'ci_lower': ci_result['ci_lower'],
                'ci_upper': ci_result['ci_upper']
            })

    return pd.DataFrame(results)


def plot_burden_vs_age_by_disease(
    analysis_df: pd.DataFrame,
    stats_df: pd.DataFrame = None,
    disease_groups: List[str] = None,
    burden_col: str = 'vcf_snv_count',
    age_col: str = 'current_age',
    group_col: str = 'disease_group',
    figsize: Tuple[float, float] = (12, 4),
    output_filename: str = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot SNV burden vs age by disease group (3-panel figure).

    Parameters:
        analysis_df: DataFrame, preprocessed analysis data
        stats_df: DataFrame, correlation stats from compute_burden_age_correlation_by_disease()
        disease_groups: list, disease groups to plot (default: ['control', 'limited', 'diffuse'])
        burden_col: str, column for mutation burden (default: 'vcf_snv_count')
        age_col: str, column for age (default: 'current_age')
        group_col: str, column for disease group (default: 'disease_group')
        figsize: tuple, figure dimensions (default: (12, 4))
        output_filename: str, base filename for saving (saves .pdf)

    Returns:
        Tuple of (Figure, axes array)
    """
    if disease_groups is None:
        disease_groups = [CONTROL_KEY, SUBTYPE_A_KEY, SUBTYPE_B_KEY]

    titles = {CONTROL_KEY: CONTROL_LABEL, SUBTYPE_A_KEY: f'{SUBTYPE_A} SSc', SUBTYPE_B_KEY: f'{SUBTYPE_B} SSc'}

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    subset = analysis_df[analysis_df[group_col].isin(disease_groups)].dropna(subset=[age_col, burden_col])
    x_min = subset[age_col].min()
    x_max = subset[age_col].max()
    y_max = subset[burden_col].quantile(0.99) * 1.1

    for i, group in enumerate(disease_groups):
        ax = axes[i]
        group_df = analysis_df[analysis_df[group_col] == group].dropna(subset=[age_col, burden_col])

        ax.scatter(
            group_df[age_col],
            group_df[burden_col],
            c=DISEASE_COLORS.get(group.capitalize(), DISEASE_COLORS_LOWER.get(group, 'gray')),
            alpha=0.4,
            s=20,
            edgecolors='none'
        )

        if stats_df is not None:
            row = stats_df[stats_df['disease_group'] == group]
            if len(row) > 0 and pd.notna(row.iloc[0]['slope']):
                slope = row.iloc[0]['slope']
                intercept = row.iloc[0]['intercept']
                x_line = np.linspace(group_df[age_col].min(), group_df[age_col].max(), 100)
                y_line = slope * x_line + intercept
                color = DISEASE_COLORS.get(group.capitalize(), DISEASE_COLORS_LOWER.get(group, 'gray'))
                ax.plot(x_line, y_line, color=color, linewidth=2.5, linestyle='-')

                r = row.iloc[0]['spearman_r']
                p_val = row.iloc[0]['p_value']
                n = row.iloc[0]['n']
                print(f'{titles.get(group, group)}: r={r:.3f}, p={p_val:.2e}, n={n}')

        ax.set_xlabel('Current Age (years)', fontsize=10)
        ax.set_ylabel('SNV Count' if i == 0 else '', fontsize=10)
        color = DISEASE_COLORS.get(group.capitalize(), DISEASE_COLORS_LOWER.get(group, 'gray'))
        ax.set_title(titles.get(group, group.capitalize()), fontsize=11, fontweight='bold', color=color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(0, y_max)

    plt.tight_layout()

    if output_filename:
        _save_figure(fig, output_filename)

    return fig, axes


def plot_burden_vs_age_by_race_within_disease(
    analysis_df: pd.DataFrame,
    stats_df: pd.DataFrame = None,
    disease_groups: List[str] = None,
    race_groups: List[str] = None,
    burden_col: str = 'vcf_snv_count',
    age_col: str = 'current_age',
    group_col: str = 'disease_group',
    race_col: str = 'race_ethnicity',
    min_n_scatter: int = 5,
    min_n_regression: int = 10,
    figsize: Tuple[float, float] = (12, 4),
    output_filename: str = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot SNV burden vs age by race within each disease group (3-panel figure).

    Parameters:
        analysis_df: DataFrame, preprocessed analysis data
        stats_df: DataFrame, correlation stats from compute_burden_age_correlation_by_race_within_disease()
        disease_groups: list, disease groups to plot (default: ['control', 'limited', 'diffuse'])
        race_groups: list, race/ethnicity groups to plot (default: White NH, White H, Black)
        burden_col: str, column for mutation burden (default: 'vcf_snv_count')
        age_col: str, column for age (default: 'current_age')
        group_col: str, column for disease group (default: 'disease_group')
        race_col: str, column for race/ethnicity (default: 'race_ethnicity')
        min_n_scatter: int, minimum n to show scatter points (default: 5)
        min_n_regression: int, minimum n to show regression line (default: 10)
        figsize: tuple, figure dimensions (default: (12, 4))
        output_filename: str, base filename for saving (saves .pdf)

    Returns:
        Tuple of (Figure, axes array)
    """
    if disease_groups is None:
        disease_groups = [CONTROL_KEY, SUBTYPE_A_KEY, SUBTYPE_B_KEY]
    if race_groups is None:
        race_groups = ['White Non-Hispanic', 'White Hispanic', 'Black']

    titles = {CONTROL_KEY: CONTROL_LABEL, SUBTYPE_A_KEY: f'{SUBTYPE_A} SSc', SUBTYPE_B_KEY: f'{SUBTYPE_B} SSc'}

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    subset = analysis_df[analysis_df[group_col].isin(disease_groups)].dropna(subset=[age_col, burden_col])
    x_min = subset[age_col].min()
    x_max = subset[age_col].max()
    y_max = subset[burden_col].quantile(0.99) * 1.1

    race_handles = []
    race_labels_legend = []

    for i, group in enumerate(disease_groups):
        ax = axes[i]
        group_df = analysis_df[analysis_df[group_col] == group].dropna(subset=[age_col, burden_col])

        for race in race_groups:
            race_df = group_df[group_df[race_col] == race]

            if len(race_df) < min_n_scatter:
                continue

            scatter = ax.scatter(
                race_df[age_col],
                race_df[burden_col],
                c=RACE_COLORS.get(race, 'gray'),
                alpha=0.4,
                s=15,
                edgecolors='none',
                label=race
            )

            if i == 0 and race not in race_labels_legend:
                race_handles.append(scatter)
                race_labels_legend.append(race)

            if stats_df is not None and len(race_df) >= min_n_regression:
                row = stats_df[(stats_df['disease_group'] == group) & (stats_df['race_ethnicity'] == race)]
                if len(row) > 0 and pd.notna(row.iloc[0]['slope']):
                    slope = row.iloc[0]['slope']
                    intercept = row.iloc[0]['intercept']
                    x_line = np.linspace(race_df[age_col].min(), race_df[age_col].max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, color=RACE_COLORS.get(race, 'gray'), linewidth=2, linestyle='-')

                    r = row.iloc[0]['spearman_r']
                    p_val = row.iloc[0]['p_value']
                    n = row.iloc[0]['n']
                    race_abbrev = race.replace('Non-Hispanic', 'NH').replace('Hispanic', 'H')
                    print(f'{titles.get(group, group)} | {race_abbrev}: r={r:.3f}, p={p_val:.2e}, n={n}')

        ax.set_xlabel('Current Age (years)', fontsize=10)
        ax.set_ylabel('SNV Count' if i == 0 else '', fontsize=10)
        color = DISEASE_COLORS.get(group.capitalize(), DISEASE_COLORS_LOWER.get(group, 'gray'))
        ax.set_title(titles.get(group, group.capitalize()), fontsize=11, fontweight='bold', color=color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(0, y_max)

    if race_handles:
        fig.legend(
            race_handles,
            race_labels_legend,
            loc='upper right',
            bbox_to_anchor=(0.99, 0.99),
            frameon=True,
            fontsize=8,
            title='Race/Ethnicity',
            title_fontsize=9
        )

    plt.tight_layout()

    if output_filename:
        _save_figure(fig, output_filename)

    return fig, axes


def plot_burden_by_onset_duration_race(
    analysis_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    race_groups: List[str] = None,
    burden_col: str = 'vcf_snv_count',
    age_col: str = 'current_age',
    group_col: str = 'disease_group',
    race_col: str = 'race_ethnicity',
    min_n_plot: int = 3,
    figsize: Tuple[float, float] = (16, 14),
    output_filename: str = None,
    ylim: Tuple[float, float] = (0, 500)
) -> Tuple[plt.Figure, Dict]:
    """
    Plot mutation burden by onset age group and duration category with 95% CIs.

    Parameters:
        analysis_df: DataFrame, preprocessed analysis data
        stats_df: DataFrame, output from compute_burden_stats_by_onset_duration_race()
        race_groups: list, race/ethnicity groups (default: ['White Non-Hispanic', 'White Hispanic', 'Black'])
        burden_col: str, column for mutation burden (default: 'vcf_snv_count')
        age_col: str, column for current age (default: 'current_age')
        group_col: str, column for disease group (default: 'disease_group')
        race_col: str, column for race/ethnicity (default: 'race_ethnicity')
        min_n_plot: int, minimum n to plot points (default: 3)
        figsize: tuple, figure dimensions (default: (16, 14))
        output_filename: str, base filename for saving (saves .pdf)
        ylim: tuple, y-axis range of every panel (default: (0, 500); the
            protein-altering SNV figure uses (25, 175))

    Returns:
        Tuple of (Figure, control_regressions dict)
    """
    if race_groups is None:
        race_groups = ['White Non-Hispanic', 'White Hispanic', 'Black']

    control_regressions = calculate_control_regressions(
        analysis_df, race_groups=race_groups,
        age_col=age_col, burden_col=burden_col,
        group_col=group_col, race_col=race_col
    )

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    panel_labels = ['A', 'B', 'C', 'D']
    duration_categories = ['Baseline (<=2yr)', 'Duration 3-5yr', 'Duration 6-8yr', 'All Durations']
    age_range = np.linspace(20, 80, 100)

    for panel_idx, dur_cat in enumerate(duration_categories):
        ax = axes[panel_idx]
        panel_data = stats_df[stats_df['duration_category'] == dur_cat]

        for race in race_groups:
            if race in control_regressions:
                reg = control_regressions[race]
                y_pred = reg['slope'] * age_range + reg['intercept']
                ax.plot(
                    age_range, y_pred,
                    color=RACE_COLORS.get(race, 'gray'),
                    linewidth=2,
                    linestyle='--',
                    alpha=0.4,
                    zorder=1
                )

        for j, race in enumerate(race_groups):
            race_data = panel_data[panel_data['race_ethnicity'] == race].copy()
            race_data = race_data[race_data['n'] >= min_n_plot]

            if len(race_data) == 0:
                continue

            x_positions = race_data['onset_group_midpoint'].values + (j - 1) * 3
            y_values = race_data['mean_burden'].values
            y_lower = race_data['ci_lower'].values
            y_upper = race_data['ci_upper'].values
            n_values = race_data['n'].values

            yerr_lower = y_values - y_lower
            yerr_upper = y_upper - y_values

            ax.errorbar(
                x_positions, y_values,
                yerr=[yerr_lower, yerr_upper],
                fmt='o',
                color=RACE_COLORS.get(race, 'gray'),
                markersize=8,
                capsize=4,
                capthick=2,
                linewidth=2,
                alpha=0.8,
                label=f'{race}' if panel_idx == 0 else None
            )

            for x, y, n in zip(x_positions, y_values, n_values):
                if n < 100:
                    ax.annotate(
                        f'{n}', (x, y),
                        textcoords='offset points',
                        xytext=(5, 5),
                        fontsize=8,
                        alpha=0.7
                    )

        ax.set_xlim(15, 85)
        ax.set_ylim(*ylim)

        decade_midpoints = [25, 35, 45, 55, 65, 75]
        ax.set_xticks(decade_midpoints)
        ax.set_xticklabels(ONSET_DECADES)

        ax.set_xlabel('Age of Onset (years)', fontsize=10)
        ax.set_ylabel('SNV Count', fontsize=10)
        ax.set_title(f'{panel_labels[panel_idx]}. {dur_cat}', fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)

        total_n = panel_data['n'].sum()
        print(f"  {panel_labels[panel_idx]}. {dur_cat}: total n={total_n}")

    axes[0].legend(loc='lower left', fontsize=9)
    plt.tight_layout()

    if output_filename:
        _save_figure(fig, output_filename)

    return fig, control_regressions

def plot_panel_a_onset_distributions(ax_limited, ax_diffuse, df, panel_labels=('A1', 'A2')):
    """
    Plot age of onset KDE distributions for Limited and Diffuse SSc, stratified by race.

    Parameters:
        ax_limited: matplotlib Axes, axes for Limited distribution
        ax_diffuse: matplotlib Axes, axes for Diffuse distribution
        df: pd.DataFrame, must contain 'disease_group', 'race_ethnicity', 'age_at_onset'
        panel_labels: tuple, labels for the two panels (default: ('A1', 'A2'))
    """
    df_cases = df[df['disease_group'].isin([SUBTYPE_A_KEY, SUBTYPE_B_KEY])].copy()
    df_cases = df_cases[df_cases['race_ethnicity'].isin(RACE_ORDER)]

    linestyles = {'White Non-Hispanic': '-', 'White Hispanic': '--', 'Black': ':'}

    for ax, group, label in [(ax_limited, SUBTYPE_A_KEY, panel_labels[0]),
                              (ax_diffuse, SUBTYPE_B_KEY, panel_labels[1])]:
        group_data = df_cases[df_cases['disease_group'] == group]

        for race in RACE_ORDER:
            race_data = group_data[group_data['race_ethnicity'] == race]['age_at_onset'].dropna()
            if len(race_data) > 10:
                try:
                    kde = gaussian_kde(race_data, bw_method=0.3)
                    x_range = np.linspace(10, 85, 200)
                    density = kde(x_range)
                    ax.plot(x_range, density, color=RACE_COLORS[race],
                           linestyle=linestyles[race], linewidth=2.5,
                           label=f'{race} (n={len(race_data)})')
                    ax.fill_between(x_range, density, alpha=0.15, color=RACE_COLORS[race])
                except Exception:
                    pass

        ax.set_xlabel('Age of Onset (years)')
        ax.set_ylabel('Density')
        subtype_name = f'{SUBTYPE_A} SSc' if group == SUBTYPE_A_KEY else f'{SUBTYPE_B} SSc'
        ax.set_title(f'{label}. Age of Onset - {subtype_name}', fontweight='bold')
        ax.legend(loc='upper right', frameon=False, fontsize=FONT_SIZE_MIN - 2)
        ax.set_xlim(10, 85)


def plot_panel_b_duration_scatter(ax, df, panel_label='B'):
    """
    Plot mutation burden vs disease duration scatter with regression lines by race.

    Parameters:
        ax: matplotlib Axes
        df: pd.DataFrame, must contain 'disease_group', 'race_ethnicity', 'disease_duration', 'vcf_snv_count'
        panel_label: str, panel label for title (default: 'B')
    """
    cases = df[df['disease_group'].isin([SUBTYPE_A_KEY, SUBTYPE_B_KEY])].copy()
    cases = cases[cases['race_ethnicity'].isin(RACE_ORDER)]
    cases = cases.dropna(subset=['disease_duration', 'vcf_snv_count'])

    for race in RACE_ORDER:
        race_data = cases[cases['race_ethnicity'] == race]
        ax.scatter(race_data['disease_duration'], race_data['vcf_snv_count'],
                  color=RACE_COLORS[race], alpha=0.8, s=30,
                  label=f'{race} (n={len(race_data)})')

        if len(race_data) >= 20:
            x = race_data['disease_duration'].values
            y = race_data['vcf_snv_count'].values
            slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=RACE_COLORS[race], linewidth=2.5, linestyle='-', alpha=0.8)

    ax.set_xlabel(f'{DURATION_COL} (years)')
    ax.set_ylabel('SNV Count (Mutation Burden)')
    ax.set_title(f'{panel_label}. Mutation Burden vs {DURATION_COL}', fontweight='bold')
    ax.legend(loc='upper right', frameon=False, fontsize=FONT_SIZE_MIN - 2)


def plot_burden_by_onset_subpanel(ax, df, control_regressions, subtype, duration_filter,
                                   subtype_label, show_legend=True):
    """
    Plot mean burden with 95% CI by onset decade for a specific subtype and duration window.

    Parameters:
        ax: matplotlib Axes
        df: pd.DataFrame, analysis dataframe
        control_regressions: dict, pre-calculated control regression parameters from calculate_control_regressions
        subtype: str, 'limited', 'diffuse', or 'both'
        duration_filter: tuple or None, (min_dur, max_dur) or None for all
        subtype_label: str, title label (e.g., 'Limited SSc - Duration <=2yr')
        show_legend: bool, whether to display legend (default: True)
    """
    stats_df = calculate_subtype_statistics(df, subtype, duration_filter)

    age_range = np.linspace(20, 80, 100)
    for race in RACE_ORDER:
        if race in control_regressions:
            reg = control_regressions[race]
            y_pred = reg['slope'] * age_range + reg['intercept']
            ax.plot(age_range, y_pred, color=RACE_COLORS[race], linewidth=2, linestyle='--', alpha=0.35, zorder=1)

    x_offsets = {'White Non-Hispanic': -2, 'White Hispanic': 0, 'Black': 2}

    for race in RACE_ORDER:
        race_data = stats_df[stats_df['race_ethnicity'] == race].copy()
        race_data = race_data[race_data['n'] >= 3]
        if len(race_data) == 0:
            continue

        x_positions = race_data['onset_group_midpoint'].values + x_offsets[race]
        y_values = race_data['mean_burden'].values
        y_lower = race_data['ci_lower'].values
        y_upper = race_data['ci_upper'].values

        yerr_lower = np.where(np.isnan(y_values - y_lower), 0, y_values - y_lower)
        yerr_upper = np.where(np.isnan(y_upper - y_values), 0, y_upper - y_values)

        ax.errorbar(x_positions, y_values, yerr=[yerr_lower, yerr_upper],
                   fmt='o', color=RACE_COLORS[race], markersize=8,
                   capsize=4, capthick=2, linewidth=2, alpha=0.85,
                   label=race if show_legend else None)

    ax.set_xlim(18, 82)
    ax.set_ylim(0, 500)
    decade_midpoints = [25, 35, 45, 55, 65, 75]
    ax.set_xticks(decade_midpoints)
    ax.set_xticklabels(ONSET_DECADES)
    ax.set_xlabel('Age of Onset (years)')
    ax.set_ylabel('Mean SNV Count')
    ax.set_title(subtype_label, fontweight='bold')

    if show_legend:
        ax.legend(loc='upper right', frameon=False, fontsize=FONT_SIZE_MIN - 2)

    total_n = stats_df['n'].sum()
    print(f"  {subtype_label}: total n={total_n}")


def plot_panel_c_to_f(axes, df, control_regressions, duration_filter, panel_label, duration_label):
    """
    Plot 3-subpanel figure (Limited, Diffuse, All SSc) for a given duration window.

    Parameters:
        axes: list of 3 matplotlib Axes
        df: pd.DataFrame, analysis dataframe
        control_regressions: dict, pre-calculated control regressions
        duration_filter: tuple or None, (min_dur, max_dur) or None
        panel_label: str, panel letter (e.g., 'C')
        duration_label: str, description of duration window (e.g., 'Duration <=2yr')
    """
    subtypes = [
        (SUBTYPE_A_KEY, f'{panel_label}1. {SUBTYPE_A} SSc - {duration_label}'),
        (SUBTYPE_B_KEY, f'{panel_label}2. {SUBTYPE_B} SSc - {duration_label}'),
        ('both', f'{panel_label}3. All SSc Cases - {duration_label}')
    ]

    for i, (subtype, title) in enumerate(subtypes):
        show_legend = (i == 0)
        plot_burden_by_onset_subpanel(axes[i], df, control_regressions, subtype, duration_filter,
                                      title, show_legend=show_legend)


def create_full_multipanel_figure(df, save_path=None, ylim=None):
    """
    Create the complete multi-panel figure with panels A through F.

    Parameters:
        df: pd.DataFrame, analysis dataframe
        save_path: str or None, path to save PDF (default: None)
        ylim: tuple or None, y-axis range applied to every panel after plotting
            (default: None keeps each panel's own range; the protein-altering SNV
            figure uses (25, 175))

    Returns:
        matplotlib Figure
    """
    set_default_plot_style()
    control_regressions = calculate_control_regressions(df)

    fig = plt.figure(figsize=(21, 36))
    gs = fig.add_gridspec(6, 3, hspace=0.35, wspace=0.25)

    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[0, 2])
    axes_c = [fig.add_subplot(gs[1, i]) for i in range(3)]
    axes_d = [fig.add_subplot(gs[2, i]) for i in range(3)]
    axes_e = [fig.add_subplot(gs[3, i]) for i in range(3)]
    axes_f = [fig.add_subplot(gs[4, i]) for i in range(3)]

    print("Plotting Panel A: age of onset distributions")
    plot_panel_a_onset_distributions(ax_a1, ax_a2, df)

    print("Plotting Panel B: mutation burden vs disease duration")
    plot_panel_b_duration_scatter(ax_b, df)

    print("Plotting Panel C: early disease (<=2 years)")
    plot_panel_c_to_f(axes_c, df, control_regressions, (0, 2), 'C', 'Duration <=2yr')

    print("Plotting Panel D: duration 3-5 years")
    plot_panel_c_to_f(axes_d, df, control_regressions, (3, 5), 'D', 'Duration 3-5yr')

    print("Plotting Panel E: duration 6-10 years")
    plot_panel_c_to_f(axes_e, df, control_regressions, (6, 10), 'E', 'Duration 6-10yr')

    print("Plotting Panel F: all durations")
    plot_panel_c_to_f(axes_f, df, control_regressions, None, 'F', 'All Durations')

    for ax in fig.axes:
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_MIN)
        if ylim is not None:
            ax.set_ylim(*ylim)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")

    return fig

def plot_magma_geneset_forest(
    gsa_path: str,
    label_map: dict = None,
    ci_level: float = 1.96,
    p_threshold: float = 0.05,
    figsize: tuple = (6.5, 4),
    save_path: str = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Forest plot of MAGMA competitive gene-set analysis BETAs.

    Parameters:
        gsa_path: str, path to MAGMA .gsa.out file
        label_map: dict, {VARIABLE: display_label} for y-axis labels (default: auto-generated)
        ci_level: float, z-score for confidence interval (default: 1.96 for 95% CI)
        p_threshold: float, significance threshold for asterisk annotation (default: 0.05)
        figsize: tuple, figure dimensions (default: (6.5, 4))
        save_path: str, file path to save figure (default: None)
        dpi: int, resolution for saved figure (default: 300)

    Returns:
        matplotlib Figure object
    """
    with open(gsa_path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    header = lines[0].split()
    rows = [l.split() for l in lines[1:]]
    gsa = pd.DataFrame(rows, columns=header)
    gsa['BETA'] = gsa['BETA'].astype(float)
    gsa['SE'] = gsa['SE'].astype(float)
    gsa['P'] = gsa['P'].astype(float)
    gsa['NGENES'] = gsa['NGENES'].astype(int)

    gsa['ci_lower'] = gsa['BETA'] - ci_level * gsa['SE']
    gsa['ci_upper'] = gsa['BETA'] + ci_level * gsa['SE']

    if label_map is None:
        label_map = {
            'NK_SOMATIC_GENES': 'NK Somatic Genes\n(FCGR3A, NKG7, KLRF2)',
            'GO_0002323': 'NK Cell Activation In\nImmune Response (GO:0002323)',
            'GO_0043320': 'NK Cell Degranulation\n(GO:0043320)',
            'GO_0042267': 'NK Cell Mediated\nCytotoxicity (GO:0042267)',
            'GO_0019730': 'Antimicrobial Humoral\nResponse (GO:0019730)'
        }

    gsa['label'] = gsa['VARIABLE'].map(label_map).fillna(gsa['VARIABLE'])
    gsa = gsa.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    y_positions = np.arange(len(gsa))

    for i, row in gsa.iterrows():
        ax.plot(
            [row['ci_lower'], row['ci_upper']], [i, i],
            color=OLIVE, linewidth=1.8, solid_capstyle='round', zorder=2
        )
        ax.plot(
            row['BETA'], i,
            'o', color=OLIVE, markersize=8, markeredgecolor='white',
            markeredgewidth=0.8, zorder=3
        )
        if row['P'] < p_threshold:
            ax.annotate(
                '*', xy=(row['ci_upper'] + 0.03, i),
                fontsize=14, fontweight='bold', color='black', va='center'
            )

    ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [f"{row['label']}  (n = {row['NGENES']})" for _, row in gsa.iterrows()],
        fontsize=8.5
    )
    ax.set_xlabel('MAGMA BETA (excess GWAS signal)', fontsize=10)
    ax.set_title(
        'MAGMA competitive gene-set analysis\n(SSc GWAS, Lopez-Isac et al. 2019)',
        fontsize=11, fontweight='bold', loc='left', pad=10
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', labelsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    return fig


def plot_convergence_panel(
    somatic_results: dict,
    magma_results: dict,
    term_order: list = None,
    term_labels: dict = None,
    somatic_threshold: float = 0.05,
    magma_threshold: float = 0.05,
    figsize: tuple = (10, 4),
    save_path: str = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Side-by-side bar plot of somatic ORA FDR and MAGMA gene-set p-values for somatic-germline convergence.

    Parameters:
        somatic_results: dict, {GO_ID: FDR} from somatic ORA
        magma_results: dict, {GO_ID: p-value} from MAGMA gene-set analysis
        term_order: list, GO IDs in desired y-axis order (default: sorted by somatic FDR)
        term_labels: dict, {GO_ID: display_label} for y-axis (default: auto-generated)
        somatic_threshold: float, significance line for somatic panel (default: 0.05)
        magma_threshold: float, significance line for MAGMA panel (default: 0.05)
        figsize: tuple, figure dimensions (default: (10, 4))
        save_path: str, file path to save figure (default: None)
        dpi: int, resolution (default: 300)

    Returns:
        matplotlib Figure object
    """
    if term_labels is None:
        term_labels = {
            'GO:0002323': 'NK Cell Activation In\nImmune Response',
            'GO:0043320': 'NK Cell\nDegranulation',
            'GO:0042267': 'NK Cell Mediated\nCytotoxicity',
            'GO:0019730': 'Antimicrobial\nHumoral Response'
        }

    if term_order is None:
        term_order = sorted(somatic_results.keys(), key=lambda x: somatic_results[x])

    labels = [term_labels.get(t, t) for t in term_order]
    somatic_vals = [-np.log10(somatic_results[t]) for t in term_order]
    magma_vals = [-np.log10(magma_results[t]) for t in term_order]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.05)
    ax_somatic = fig.add_subplot(gs[0])
    ax_magma = fig.add_subplot(gs[1])

    y_pos = np.arange(len(term_order))
    bar_height = 0.55

    ax_somatic.barh(
        y_pos, somatic_vals, height=bar_height,
        color=OLIVE, edgecolor='white', linewidth=0.5
    )
    ax_somatic.axvline(
        x=-np.log10(somatic_threshold), color='grey',
        linestyle='--', linewidth=0.8, alpha=0.7
    )
    ax_somatic.set_xlim(max(somatic_vals) * 1.15, 0)
    ax_somatic.set_yticks(y_pos)
    ax_somatic.set_yticklabels(labels, fontsize=8.5, ha='center')
    ax_somatic.yaxis.set_ticks_position('none')
    for label in ax_somatic.get_yticklabels():
        label.set_position((1.0, 0))
        label.set_ha('center')
    ax_somatic.tick_params(axis='y', pad=55)
    ax_somatic.set_xlabel('$-\\log_{10}$(FDR)', fontsize=9)
    ax_somatic.set_title('Somatic ORA\n(VAF < 0.35)', fontsize=10, fontweight='bold', loc='center')
    ax_somatic.spines['top'].set_visible(False)
    ax_somatic.spines['right'].set_visible(False)
    ax_somatic.spines['left'].set_visible(False)
    ax_somatic.tick_params(axis='x', labelsize=8)

    ax_magma.barh(
        y_pos, magma_vals, height=bar_height,
        color=OLIVE_LIGHT, edgecolor='white', linewidth=0.5
    )
    ax_magma.axvline(
        x=-np.log10(magma_threshold), color='grey',
        linestyle='--', linewidth=0.8, alpha=0.7
    )
    ax_magma.set_xlim(0, max(magma_vals) * 1.15)
    ax_magma.set_yticks(y_pos)
    ax_magma.set_yticklabels([])
    ax_magma.set_xlabel('$-\\log_{10}$(P)', fontsize=9)
    ax_magma.set_title('GWAS MAGMA\n(Lopez-Isac 2019)', fontsize=10, fontweight='bold', loc='center')
    ax_magma.spines['top'].set_visible(False)
    ax_magma.spines['left'].set_visible(False)
    ax_magma.spines['right'].set_visible(False)
    ax_magma.tick_params(axis='x', labelsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    return fig


def plot_magma_gene_heatmap(
    genes_out_path: str,
    go_term_entrez_ids: list,
    somatic_entrez_ids: list = None,
    entrez_to_symbol: dict = None,
    gene_loc_path: str = None,
    go_term_label: str = 'GO:0042267 (NK Cell Mediated Cytotoxicity)',
    figsize: tuple = (3.5, 8),
    save_path: str = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Heatmap of gene-level MAGMA Z-scores for genes in a GO term, with somatic NK genes highlighted.

    Parameters:
        genes_out_path: str, path to MAGMA .genes.out file
        go_term_entrez_ids: list, Entrez IDs of genes in the GO term
        somatic_entrez_ids: list, Entrez IDs of somatic NK genes to highlight (default: [2214, 4818, 100431172])
        entrez_to_symbol: dict, {entrez_id: gene_symbol} mapping (default: built from gene_loc_path)
        gene_loc_path: str, path to NCBI37.3.gene.loc for symbol lookup
        go_term_label: str, title label for the GO term
        figsize: tuple, figure dimensions (default: (3.5, 8))
        save_path: str, file path to save figure (default: None)
        dpi: int, resolution (default: 300)

    Returns:
        matplotlib Figure object
    """
    if somatic_entrez_ids is None:
        somatic_entrez_ids = [2214, 4818, 100431172]

    genes = pd.read_csv(genes_out_path, sep=r'\s+', comment='#')
    genes.columns = [c.strip() for c in genes.columns]
    genes['GENE'] = genes['GENE'].astype(int)

    go_term_entrez_ids = [int(x) for x in go_term_entrez_ids]
    somatic_entrez_ids = [int(x) for x in somatic_entrez_ids]

    subset = genes[genes['GENE'].isin(go_term_entrez_ids)].copy()
    missing = set(go_term_entrez_ids) - set(subset['GENE'])
    if missing:
        print(f"Note: {len(missing)} genes not in MAGMA results (no SNP coverage): {missing}")

    if entrez_to_symbol is None and gene_loc_path is not None:
        loc = pd.read_csv(gene_loc_path, sep=r'\s+', header=None,
                          names=['ENTREZ', 'CHR', 'START', 'STOP', 'STRAND', 'SYMBOL'])
        entrez_to_symbol = dict(zip(loc['ENTREZ'], loc['SYMBOL']))

    if entrez_to_symbol is not None:
        subset['SYMBOL'] = subset['GENE'].map(entrez_to_symbol).fillna(subset['GENE'].astype(str))
    else:
        subset['SYMBOL'] = subset['GENE'].astype(str)

    subset['is_somatic'] = subset['GENE'].isin(somatic_entrez_ids)
    subset = subset.sort_values('ZSTAT', ascending=True).reset_index(drop=True)

    fig, (ax_heat, ax_annot) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={'width_ratios': [6, 1], 'wspace': 0.08}
    )

    z_vals = subset['ZSTAT'].values.reshape(-1, 1)
    z_max = max(abs(z_vals.min()), abs(z_vals.max()), 2.5)

    cmap_colors = ['#2166AC', '#67A9CF', '#D1E5F0', '#F7F7F7', '#C5CFA6', OLIVE_LIGHT, OLIVE]
    cmap = mcolors.LinearSegmentedColormap.from_list('gwas_diverge', cmap_colors, N=256)
    norm = mcolors.TwoSlopeNorm(vmin=-z_max, vcenter=0, vmax=z_max)

    im = ax_heat.imshow(
        z_vals, aspect='auto', cmap=cmap, norm=norm,
        interpolation='nearest'
    )

    ax_heat.set_yticks(range(len(subset)))
    y_labels = []
    for _, row in subset.iterrows():
        label = row['SYMBOL']
        if row['is_somatic']:
            label = f"{label} *"
        y_labels.append(label)
    ax_heat.set_yticklabels(y_labels, fontsize=8)
    ax_heat.set_xticks([])
    ax_heat.set_title(f'MAGMA Z-score\n{go_term_label}', fontsize=9, fontweight='bold', pad=10)

    for i, row in subset.iterrows():
        if row['is_somatic']:
            ax_heat.get_yticklabels()[i].set_fontweight('bold')
            ax_heat.get_yticklabels()[i].set_color(OLIVE)

    for i, row in subset.iterrows():
        z = row['ZSTAT']
        text_color = 'white' if abs(z) > z_max * 0.65 else 'black'
        ax_heat.text(0, i, f'{z:.2f}', ha='center', va='center',
                     fontsize=7.5, color=text_color)

    annot_vals = subset['is_somatic'].astype(int).values.reshape(-1, 1)
    annot_cmap = mcolors.ListedColormap(['#F7F7F7', OLIVE])
    ax_annot.imshow(annot_vals, aspect='auto', cmap=annot_cmap,
                    interpolation='nearest', vmin=0, vmax=1)
    ax_annot.set_xticks([0])
    ax_annot.set_xticklabels(['Somatic\nhit'], fontsize=7, rotation=0)
    ax_annot.set_yticks([])
    ax_annot.tick_params(axis='x', length=0, pad=2)

    cbar = fig.colorbar(im, ax=ax_heat, orientation='horizontal',
                        fraction=0.04, pad=0.08, shrink=0.8)
    cbar.set_label('MAGMA Z-score', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax_heat.spines['top'].set_visible(False)
    ax_heat.spines['right'].set_visible(False)
    ax_heat.spines['bottom'].set_visible(False)
    ax_annot.spines['top'].set_visible(False)
    ax_annot.spines['right'].set_visible(False)
    ax_annot.spines['bottom'].set_visible(False)
    ax_annot.spines['left'].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    return fig


def create_cohort_dist_plots(df, save_path=None):
    """
    Two-panel cohort distribution figure.
    Panel A: current age distributions by race (White vs Black) and disease group.
    Panel B: age of onset distributions by race and SSc subtype.

    Parameters:
        df: pd.DataFrame, analysis dataframe with 'race_ethnicity', 'disease_group',
            'current_age', 'age_at_onset'
        save_path: str or None, path to save figure (default: None)

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    race_groups_plot = ['White Non-Hispanic', 'White Hispanic', 'Black']
    df_comp = df[df['race_ethnicity'].isin(race_groups_plot)].copy()
    df_comp['race_plot'] = df_comp['race_ethnicity'].apply(lambda x: 'White' if 'White' in x else x)

    ax_a = axes[0]
    for race in ['White', 'Black']:
        race_data = df_comp[df_comp['race_plot'] == race]
        for group in [CONTROL_KEY, SUBTYPE_A_KEY, SUBTYPE_B_KEY]:
            group_data = race_data[race_data['disease_group'] == group]['current_age'].dropna()
            if len(group_data) > 10:
                try:
                    kde = gaussian_kde(group_data, bw_method=0.3)
                    x_range = np.linspace(10, 95, 200)
                    density = kde(x_range)
                    linestyle = '-' if race == 'White' else '--'
                    alpha = 0.7 if race == 'White' else 0.5
                    ax_a.plot(x_range, density, color=DISEASE_COLORS_LOWER[group],
                             linestyle=linestyle, linewidth=2, alpha=alpha)
                    ax_a.fill_between(x_range, density,
                                     alpha=0.2 if race == 'White' else 0.1,
                                     color=DISEASE_COLORS_LOWER[group])
                except Exception:
                    pass

    legend_elements_a = [
        Line2D([0], [0], color=DISEASE_COLORS_LOWER[CONTROL_KEY], linewidth=2, label=CONTROL_LABEL),
        Line2D([0], [0], color=DISEASE_COLORS_LOWER[SUBTYPE_A_KEY], linewidth=2, label=SUBTYPE_A),
        Line2D([0], [0], color=DISEASE_COLORS_LOWER[SUBTYPE_B_KEY], linewidth=2, label=SUBTYPE_B),
        Line2D([0], [0], color='gray', linewidth=2, linestyle='-', label='White'),
        Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='Black'),
    ]
    ax_a.legend(handles=legend_elements_a, loc='upper right', frameon=False)
    ax_a.set_xlabel('Current Age (years)', fontsize=11)
    ax_a.set_ylabel('Density', fontsize=11)
    ax_a.set_title('A. Cohort Age Distribution by Race and Disease', fontsize=12, fontweight='bold')
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    ax_b = axes[1]
    df_cases = df[df['disease_group'].isin([SUBTYPE_A_KEY, SUBTYPE_B_KEY])].copy()
    df_cases = df_cases[df_cases['race_ethnicity'].isin(race_groups_plot)]
    df_cases['race_plot'] = df_cases['race_ethnicity'].apply(lambda x: 'White' if 'White' in x else x)

    for race in ['White', 'Black']:
        race_data = df_cases[df_cases['race_plot'] == race]
        for group in [SUBTYPE_A_KEY, SUBTYPE_B_KEY]:
            group_data = race_data[race_data['disease_group'] == group]['age_at_onset'].dropna()
            if len(group_data) > 10:
                try:
                    kde = gaussian_kde(group_data, bw_method=0.3)
                    x_range = np.linspace(10, 85, 200)
                    density = kde(x_range)
                    linestyle = '-' if race == 'White' else '--'
                    alpha = 0.7 if race == 'White' else 0.5
                    ax_b.plot(x_range, density, color=DISEASE_COLORS_LOWER[group],
                             linestyle=linestyle, linewidth=2, alpha=alpha)
                    ax_b.fill_between(x_range, density,
                                     alpha=0.2 if race == 'White' else 0.1,
                                     color=DISEASE_COLORS_LOWER[group])
                except Exception:
                    pass

    legend_elements_b = [
        Line2D([0], [0], color=DISEASE_COLORS_LOWER[SUBTYPE_A_KEY], linewidth=2, label=SUBTYPE_A),
        Line2D([0], [0], color=DISEASE_COLORS_LOWER[SUBTYPE_B_KEY], linewidth=2, label=SUBTYPE_B),
        Line2D([0], [0], color='gray', linewidth=2, linestyle='-', label='White'),
        Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='Black'),
    ]
    ax_b.legend(handles=legend_elements_b, loc='upper right', frameon=False)
    ax_b.set_xlabel('Age of Onset (years)', fontsize=11)
    ax_b.set_ylabel('Density', fontsize=11)
    ax_b.set_title('B. Age of Onset Distribution by Race and SSc Subtype', fontsize=12, fontweight='bold')
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig
