"""
Analysis functions for SSc mutation burden analysis pipeline.
Functions organized by notebook workflow in chronological order of use.
"""

import os
import pandas as pd
import numpy as np
import gseapy as gp
from scipy.stats import spearmanr, mannwhitneyu, chi2_contingency, fisher_exact
from typing import List, Dict
from collections import defaultdict
from scipy import stats
import statsmodels.formula.api as smf

from ..metadata import SAMPLE_ID_COLUMN
from ..settings import cohort_values


# Study settings. These start empty and are filled from the study
# configuration by apply_settings(), which SSc_Somatic.configure() calls.
PROTEIN_ALTERING_FUNCS = None
RACE_ORDER = None
NK_GOBP_SIGNIFICANT = None
NK_GOBP_SUBTHRESHOLD = None
NK_GOBP_ALL = None
NK_SOMATIC_GENES = None
GSEAPY_GOBP_LIBRARY = None
NK_VAF_THRESHOLD = None
VCF_GENE_COL = None
VCF_FUNC_COL = None
VCF_VAF_COL = None
VCF_SAMPLE_COL = None
VCF_VARIANT_ID_COL = None
META_SAMPLE_COL = None
META_SUBTYPE_COL = None
NK_COMPARISON_CONTINUOUS = None
NK_COMPARISON_CATEGORICAL = None
GWAS_SNP_COL = None
GWAS_CHR_COL = None
GWAS_BP_COL = None
GWAS_P_COL = None
GWAS_A1_COL = None
GWAS_A2_COL = None
GWAS_OR_COL = None
GWAS_N_TOTAL = None
GWAS_SUMSTATS_PATH = None
gwas_output_root = None


# Cohort labels and metadata column names, also filled by apply_settings()
AGE_COL = None
CASE_LABEL = None
CONTROL_KEY = None
CONTROL_LABEL = None
DURATION_COL = None
ETHNICITY_COL = None
GROUP_COL = None
META_ID_COL = None
RACE_COL = None
STATUS_COL = None
SUBTYPE_A = None
SUBTYPE_A_KEY = None
SUBTYPE_B = None
SUBTYPE_B_KEY = None
SUBTYPE_COL = None


def apply_settings(config):
    """
    Fill this module's study settings from a StudyConfig.

    Called by SSc_Somatic.configure(); notebooks do not need to call it directly.
    """
    global PROTEIN_ALTERING_FUNCS, RACE_ORDER
    global NK_GOBP_SIGNIFICANT, NK_GOBP_SUBTHRESHOLD, NK_GOBP_ALL
    global NK_SOMATIC_GENES, GSEAPY_GOBP_LIBRARY, NK_VAF_THRESHOLD
    global VCF_GENE_COL, VCF_FUNC_COL, VCF_VAF_COL, VCF_SAMPLE_COL, VCF_VARIANT_ID_COL
    global META_SAMPLE_COL, META_SUBTYPE_COL
    global NK_COMPARISON_CONTINUOUS, NK_COMPARISON_CATEGORICAL
    global GWAS_SNP_COL, GWAS_CHR_COL, GWAS_BP_COL, GWAS_P_COL
    global GWAS_A1_COL, GWAS_A2_COL, GWAS_OR_COL, GWAS_N_TOTAL
    global GWAS_SUMSTATS_PATH, gwas_output_root

    PROTEIN_ALTERING_FUNCS = list(config.require('protein_altering_functions'))
    RACE_ORDER = list(config.require('cohort', 'race_order'))

    NK_GOBP_SIGNIFICANT = dict(config.require('enrichment', 'nk_terms_significant'))
    NK_GOBP_SUBTHRESHOLD = dict(config.require('enrichment', 'nk_terms_subthreshold'))
    NK_GOBP_ALL = {**NK_GOBP_SIGNIFICANT, **NK_GOBP_SUBTHRESHOLD}
    NK_SOMATIC_GENES = list(config.require('enrichment', 'nk_somatic_genes'))
    GSEAPY_GOBP_LIBRARY = config.require('enrichment', 'gseapy_library')
    NK_VAF_THRESHOLD = config.require('enrichment', 'nk_vaf_threshold')
    NK_COMPARISON_CONTINUOUS = list(config.require('enrichment', 'nk_comparison_continuous'))
    NK_COMPARISON_CATEGORICAL = list(config.require('enrichment', 'nk_comparison_categorical'))

    VCF_GENE_COL = config.require('variant_columns', 'gene')
    VCF_FUNC_COL = config.require('variant_columns', 'function')
    VCF_VAF_COL = config.require('variant_columns', 'vaf')
    VCF_SAMPLE_COL = config.require('variant_columns', 'sample')
    VCF_VARIANT_ID_COL = config.require('variant_columns', 'variant_id')

    META_SAMPLE_COL = SAMPLE_ID_COLUMN
    META_SUBTYPE_COL = config.require('cohort', 'group_column')

    GWAS_SNP_COL = config.require('gwas', 'columns', 'snp')
    GWAS_CHR_COL = config.require('gwas', 'columns', 'chromosome')
    GWAS_BP_COL = config.require('gwas', 'columns', 'position')
    GWAS_P_COL = config.require('gwas', 'columns', 'p_value')
    GWAS_A1_COL = config.require('gwas', 'columns', 'effect_allele')
    GWAS_A2_COL = config.require('gwas', 'columns', 'other_allele')
    GWAS_OR_COL = config.require('gwas', 'columns', 'odds_ratio')
    GWAS_N_TOTAL = config.require('gwas', 'total_sample_size')
    GWAS_SUMSTATS_PATH = str(config.path('reference', 'gwas_summary_statistics'))
    gwas_output_root = str(config.output_path('mutation', 'gwas_validation_dir', create_parent=False))
    globals().update(cohort_values(config))


# Mutation burden analysis

def create_analysis_df(
    meta_df,
    mutation_data=None,
    mutation_source='meta',
    sample_id_col='sample_id',
    subtype_col=None,
    age_col=None,
    onset_col='age_of_onset',
    duration_col=None,
    race_col=None,
    ethnicity_col=None,
    burden_col=None
):
    """
    Preprocess metadata and mutation data into a standardized analysis-ready dataframe.

    Supports three mutation burden sources:
        'meta': synonymous + nonsynonymous SNV columns already in meta_df
        'sbs': SigProfiler SBS96 matrix (sum of 96 trinucleotide contexts)
        'vcf': aggregated SNV counts from VCF

    Parameters:
        meta_df: pd.DataFrame, sample metadata with clinical and demographic columns
        mutation_data: pd.DataFrame or None, external mutation source (required for 'sbs' or 'vcf')
        mutation_source: str, 'meta', 'sbs', or 'vcf' (default: 'meta')
        sample_id_col: str, sample identifier column in metadata (default: 'sample_id')
        subtype_col: str, disease subtype column (default: 'statustype_subtype')
        age_col: str, current age column (default: 'Age')
        onset_col: str, age of onset column (default: 'age_of_onset')
        duration_col: str, disease duration column (default: 'Disease Duration')
        race_col: str, race column (default: 'race')
        ethnicity_col: str, ethnicity column (default: 'Ethnicity')
        burden_col: str or None, mutation count column for 'vcf' source (default: 'snv_count_vcf')

    Returns:
        pd.DataFrame with standardized columns: sample_id, mutation_burden, vcf_snv_count,
        disease_group, current_age, age_at_onset, disease_duration, race_clean,
        ethnicity_clean, race_ethnicity
    """
    if subtype_col is None:
        subtype_col = GROUP_COL
    if age_col is None:
        age_col = AGE_COL
    if duration_col is None:
        duration_col = DURATION_COL
    if race_col is None:
        race_col = RACE_COL
    if ethnicity_col is None:
        ethnicity_col = ETHNICITY_COL
    meta_clean = meta_df.copy()
    meta_clean[sample_id_col] = meta_clean[sample_id_col].astype(str).str.strip()

    if mutation_source == 'meta':
        if 'synonymous SNV' not in meta_clean.columns or 'nonsynonymous SNV' not in meta_clean.columns:
            raise ValueError(
                "mutation_source='meta' requires 'synonymous SNV' and 'nonsynonymous SNV' columns in meta_df"
            )
        meta_clean['mutation_burden'] = (
            pd.to_numeric(meta_clean['synonymous SNV'], errors='coerce').fillna(0) +
            pd.to_numeric(meta_clean['nonsynonymous SNV'], errors='coerce').fillna(0)
        )
        merged = meta_clean
        source_description = "metadata (synonymous + nonsynonymous SNV)"

    elif mutation_source == 'sbs':
        if mutation_data is None:
            raise ValueError("mutation_source='sbs' requires mutation_data (SBS96 matrix)")
        if 'MutationType' not in mutation_data.columns:
            raise ValueError("SBS matrix must have 'MutationType' column")

        sbs_transposed = mutation_data.set_index('MutationType').T
        sbs_transposed['mutation_burden'] = sbs_transposed.sum(axis=1)
        sbs_transposed = sbs_transposed.reset_index()
        sbs_transposed = sbs_transposed.rename(columns={'index': 'sbs_sample_id'})
        sbs_transposed['sample_id_extracted'] = (
            sbs_transposed['sbs_sample_id']
            .str.split('_')
            .str[0]
            .astype(str)
            .str.strip()
        )

        merged = meta_clean.merge(
            sbs_transposed[['sample_id_extracted', 'mutation_burden', 'sbs_sample_id']],
            left_on=sample_id_col,
            right_on='sample_id_extracted',
            how='inner'
        )
        source_description = "SigProfiler SBS96 matrix (sum of 96 trinucleotide contexts)"

    elif mutation_source == 'vcf':
        if mutation_data is None:
            raise ValueError("mutation_source='vcf' requires mutation_data (aggregated SNV counts)")
        if burden_col is None:
            burden_col = 'snv_count_vcf'

        vcf_sample_col = None
        for candidate in ['SAMPLE', 'sample_id', 'Sample', 'sample']:
            if candidate in mutation_data.columns:
                vcf_sample_col = candidate
                break
        if vcf_sample_col is None:
            raise ValueError(
                "Could not find sample ID column in mutation_data. "
                "Expected one of: SAMPLE, sample_id, Sample, sample"
            )
        if burden_col not in mutation_data.columns:
            raise ValueError(f"burden_col '{burden_col}' not found in mutation_data")

        vcf_clean = mutation_data[[vcf_sample_col, burden_col]].copy()
        vcf_clean[vcf_sample_col] = vcf_clean[vcf_sample_col].astype(str).str.strip()
        vcf_clean = vcf_clean.rename(columns={burden_col: 'mutation_burden'})

        merged = meta_clean.merge(
            vcf_clean,
            left_on=sample_id_col,
            right_on=vcf_sample_col,
            how='inner'
        )
        source_description = f"VCF-derived counts ({burden_col})"

    else:
        raise ValueError(f"Invalid mutation_source: '{mutation_source}'. Must be 'meta', 'sbs', or 'vcf'")

    if subtype_col in merged.columns:
        merged['disease_group'] = merged[subtype_col].astype(str).str.lower().str.strip()
    else:
        raise ValueError(f"subtype_col '{subtype_col}' not found in merged data")

    merged['current_age'] = pd.to_numeric(merged[age_col], errors='coerce')
    merged['age_at_onset'] = pd.to_numeric(merged[onset_col], errors='coerce')
    merged['disease_duration'] = pd.to_numeric(merged[duration_col], errors='coerce')

    merged['race_clean'] = merged[race_col].fillna('Unknown').astype(str).str.strip()
    merged['ethnicity_clean'] = merged[ethnicity_col].fillna('Unknown').astype(str).str.strip()

    def assign_race_ethnicity(row):
        race = row['race_clean']
        ethnicity = row['ethnicity_clean']
        if race == 'White':
            if ethnicity in ['Hisp/Latin', 'Hispanic', 'Latino', 'Latina']:
                return 'White Hispanic'
            else:
                return 'White Non-Hispanic'
        elif race == 'Black':
            return 'Black'
        elif race == 'Asian':
            return 'Asian'
        else:
            return 'Other'

    merged['race_ethnicity'] = merged.apply(assign_race_ethnicity, axis=1)
    merged['vcf_snv_count'] = merged['mutation_burden']

    print(f"Mutation burden source: {source_description}")
    print(f"Input samples in metadata: {len(meta_df)}")
    if mutation_source != 'meta':
        print(f"Input samples in mutation data: {len(mutation_data)}")
    print(f"Successfully merged samples: {len(merged)}")

    if mutation_source == 'sbs':
        unmatched_meta = set(meta_clean[sample_id_col]) - set(sbs_transposed['sample_id_extracted'])
        unmatched_mut = set(sbs_transposed['sample_id_extracted']) - set(meta_clean[sample_id_col])
        if len(unmatched_meta) > 0:
            print(f"WARNING: {len(unmatched_meta)} metadata samples not in SBS matrix")
            print(f"  Examples: {list(unmatched_meta)[:5]}")
        if len(unmatched_mut) > 0:
            print(f"WARNING: {len(unmatched_mut)} SBS samples not in metadata")
            print(f"  Examples: {list(unmatched_mut)[:5]}")
    elif mutation_source == 'vcf':
        unmatched_meta = set(meta_clean[sample_id_col]) - set(vcf_clean[vcf_sample_col])
        unmatched_mut = set(vcf_clean[vcf_sample_col]) - set(meta_clean[sample_id_col])
        if len(unmatched_meta) > 0:
            print(f"WARNING: {len(unmatched_meta)} metadata samples not in VCF counts")
            print(f"  Examples: {list(unmatched_meta)[:5]}")
        if len(unmatched_mut) > 0:
            print(f"WARNING: {len(unmatched_mut)} VCF samples not in metadata")
            print(f"  Examples: {list(unmatched_mut)[:5]}")

    print(f"\nDisease group distribution:")
    print(merged['disease_group'].value_counts().to_string())
    print(f"\nRace/ethnicity distribution:")
    print(merged['race_ethnicity'].value_counts().to_string())
    print(f"\nMutation burden summary:")
    print(f"  Mean:   {merged['mutation_burden'].mean():.1f}")
    print(f"  Median: {merged['mutation_burden'].median():.1f}")
    print(f"  Std:    {merged['mutation_burden'].std():.1f}")
    print(f"  Range:  {merged['mutation_burden'].min():.0f} - {merged['mutation_burden'].max():.0f}")

    print(f"\nMissing values in key columns:")
    key_cols = ['mutation_burden', 'current_age', 'age_at_onset', 'disease_group', 'race_ethnicity']
    for col in key_cols:
        n_missing = merged[col].isna().sum()
        pct_missing = 100 * n_missing / len(merged)
        print(f"  {col}: {n_missing} ({pct_missing:.1f}%)")

    return merged


def prepare_analysis_df(analysis_df, burden_col='mutation_burden'):
    """
    Validate and prepare analysis dataframe for figure generation.

    Parameters:
        analysis_df: pd.DataFrame, preprocessed dataframe from preprocess_and_merge_data()
        burden_col: str, mutation burden column name (default: 'mutation_burden')

    Returns:
        pd.DataFrame, validated dataframe with 'vcf_snv_count' column ensured
    """
    df = analysis_df.copy()

    if burden_col in df.columns:
        df['vcf_snv_count'] = df[burden_col]
    elif 'vcf_snv_count' in df.columns:
        pass
    elif 'mutation_burden' in df.columns:
        df['vcf_snv_count'] = df['mutation_burden']
    else:
        raise ValueError(
            f"Could not find mutation burden column. "
            f"Expected '{burden_col}', 'mutation_burden', or 'vcf_snv_count'"
        )

    required_cols = ['disease_group', 'current_age', 'race_ethnicity']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Ensure data was preprocessed with preprocess_and_merge_data()"
        )

    df['disease_group'] = df['disease_group'].astype(str).str.lower().str.strip()

    valid_groups = {CONTROL_KEY, SUBTYPE_A_KEY, SUBTYPE_B_KEY}
    actual_groups = set(df['disease_group'].unique())
    invalid_groups = actual_groups - valid_groups - {'nan', 'none', ''}
    if invalid_groups:
        print(f"WARNING: unexpected disease groups found: {invalid_groups}")

    return df

def compute_burden_age_correlation_by_disease(
    analysis_df: pd.DataFrame,
    disease_groups: List[str] = None,
    burden_col: str = 'vcf_snv_count',
    age_col: str = 'current_age',
    group_col: str = 'disease_group'
) -> pd.DataFrame:
    """
    Compute Spearman correlations between mutation burden and age for each disease group.

    Parameters:
        analysis_df: DataFrame, preprocessed analysis data
        disease_groups: list, disease groups to analyze (default: ['control', 'limited', 'diffuse'])
        burden_col: str, column name for mutation burden (default: 'vcf_snv_count')
        age_col: str, column name for age (default: 'current_age')
        group_col: str, column name for disease group (default: 'disease_group')

    Returns:
        DataFrame with columns: disease_group, n, spearman_r, p_value, slope, intercept
    """
    if disease_groups is None:
        disease_groups = [CONTROL_KEY, SUBTYPE_A_KEY, SUBTYPE_B_KEY]

    results = []
    for group in disease_groups:
        group_df = analysis_df[analysis_df[group_col] == group].dropna(subset=[age_col, burden_col])
        n = len(group_df)

        if n < 10:
            results.append({
                'disease_group': group, 'n': n, 'spearman_r': np.nan,
                'p_value': np.nan, 'slope': np.nan, 'intercept': np.nan
            })
            continue

        r, p_val = spearmanr(group_df[age_col], group_df[burden_col])
        slope, intercept = np.polyfit(group_df[age_col], group_df[burden_col], 1)

        results.append({
            'disease_group': group, 'n': n, 'spearman_r': r,
            'p_value': p_val, 'slope': slope, 'intercept': intercept
        })

    return pd.DataFrame(results)


def compute_burden_age_correlation_by_race_within_disease(
    analysis_df: pd.DataFrame,
    disease_groups: List[str] = None,
    race_groups: List[str] = None,
    burden_col: str = 'vcf_snv_count',
    age_col: str = 'current_age',
    group_col: str = 'disease_group',
    race_col: str = 'race_ethnicity',
    min_n: int = 10
) -> pd.DataFrame:
    """
    Compute Spearman correlations between mutation burden and age stratified by race within disease groups.

    Parameters:
        analysis_df: DataFrame, preprocessed analysis data
        disease_groups: list, disease groups to analyze (default: ['control', 'limited', 'diffuse'])
        race_groups: list, race/ethnicity groups to analyze (default: RACE_ORDER from config)
        burden_col: str, column name for mutation burden (default: 'vcf_snv_count')
        age_col: str, column name for age (default: 'current_age')
        group_col: str, column name for disease group (default: 'disease_group')
        race_col: str, column name for race/ethnicity (default: 'race_ethnicity')
        min_n: int, minimum sample size for correlation (default: 10)

    Returns:
        DataFrame with columns: disease_group, race_ethnicity, n, spearman_r, p_value, slope, intercept
    """
    if disease_groups is None:
        disease_groups = [CONTROL_KEY, SUBTYPE_A_KEY, SUBTYPE_B_KEY]
    if race_groups is None:
        race_groups = ['White Non-Hispanic', 'White Hispanic', 'Black', 'Asian', 'Other']

    results = []
    for group in disease_groups:
        group_df = analysis_df[analysis_df[group_col] == group]

        for race in race_groups:
            race_df = group_df[group_df[race_col] == race].dropna(subset=[age_col, burden_col])
            n = len(race_df)

            if n < min_n:
                results.append({
                    'disease_group': group, 'race_ethnicity': race, 'n': n,
                    'spearman_r': np.nan, 'p_value': np.nan, 'slope': np.nan, 'intercept': np.nan
                })
                continue

            r, p_val = spearmanr(race_df[age_col], race_df[burden_col])
            slope, intercept = np.polyfit(race_df[age_col], race_df[burden_col], 1)

            results.append({
                'disease_group': group, 'race_ethnicity': race, 'n': n,
                'spearman_r': r, 'p_value': p_val, 'slope': slope, 'intercept': intercept
            })

    return pd.DataFrame(results)

def compute_bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 95,
    statistic: str = 'mean'
) -> Dict:
    """
    Calculate bootstrap confidence interval for a statistic.

    Parameters:
        data: array-like, data to bootstrap
        n_bootstrap: int, number of bootstrap samples (default: 1000)
        ci: float, confidence interval percentage (default: 95)
        statistic: str, 'mean' or 'median' (default: 'mean')

    Returns:
        Dict with keys: estimate, ci_lower, ci_upper, n
    """
    data = np.array(data)
    data = data[~np.isnan(data)]
    n = len(data)

    if n < 3:
        return {'estimate': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'n': n}

    stat_func = np.mean if statistic == 'mean' else np.median

    boot_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(stat_func(sample))

    boot_stats = np.array(boot_stats)
    alpha = (100 - ci) / 2
    ci_lower = np.percentile(boot_stats, alpha)
    ci_upper = np.percentile(boot_stats, 100 - alpha)

    return {
        'estimate': stat_func(data),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n': n
    }

def compute_burden_stats_by_onset_duration_race(
    analysis_df: pd.DataFrame,
    race_groups: List[str] = None,
    onset_bins: List[str] = None,
    burden_col: str = 'vcf_snv_count',
    onset_col: str = 'age_at_onset',
    duration_col: str = 'disease_duration',
    group_col: str = 'disease_group',
    race_col: str = 'race_ethnicity',
    n_bootstrap: int = 1000
) -> pd.DataFrame:
    """
    Calculate mean burden and 95% CI for each race x onset group x duration category.

    Parameters:
        analysis_df: DataFrame, preprocessed analysis data
        race_groups: list, race/ethnicity groups (default: ['White Non-Hispanic', 'White Hispanic', 'Black'])
        onset_bins: list, onset age group labels (default: ['20-29', '30-39', ..., '70-79'])
        burden_col: str, column for mutation burden (default: 'vcf_snv_count')
        onset_col: str, column for age at onset (default: 'age_at_onset')
        duration_col: str, column for disease duration (default: 'disease_duration')
        group_col: str, column for disease group (default: 'disease_group')
        race_col: str, column for race/ethnicity (default: 'race_ethnicity')
        n_bootstrap: int, number of bootstrap samples (default: 1000)

    Returns:
        DataFrame with columns: duration_category, race_ethnicity, onset_age_group,
        onset_group_midpoint, median_onset_age, n, mean_burden, ci_lower, ci_upper
    """
    if race_groups is None:
        race_groups = ['White Non-Hispanic', 'White Hispanic', 'Black']
    if onset_bins is None:
        onset_bins = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']

    df = analysis_df.copy()

    def assign_onset_decade(age):
        if pd.isna(age):
            return np.nan
        for start in range(20, 80, 10):
            if start <= age < start + 10:
                return f'{start}-{start+9}'
        return np.nan

    df['onset_age_group'] = df[onset_col].apply(assign_onset_decade)
    cases = df[df[group_col].isin([SUBTYPE_A_KEY, SUBTYPE_B_KEY])].copy()

    duration_defs = {
        'Baseline (<=2yr)': cases[duration_col] <= 2,
        'Duration 3-5yr': (cases[duration_col] >= 3) & (cases[duration_col] <= 5),
        'Duration 6-8yr': (cases[duration_col] >= 6) & (cases[duration_col] <= 8),
        'All Durations': cases[duration_col].notna()
    }

    results = []
    for dur_name, dur_mask in duration_defs.items():
        dur_cases = cases[dur_mask]
        for race in race_groups:
            race_cases = dur_cases[dur_cases[race_col] == race]
            for onset_grp in onset_bins:
                cell_data = race_cases[race_cases['onset_age_group'] == onset_grp][burden_col].values
                ci_result = compute_bootstrap_ci(cell_data, n_bootstrap=n_bootstrap, statistic='mean')
                onset_ages = race_cases[race_cases['onset_age_group'] == onset_grp][onset_col].values
                median_onset_age = np.nanmedian(onset_ages) if len(onset_ages) > 0 else np.nan
                results.append({
                    'duration_category': dur_name,
                    'race_ethnicity': race,
                    'onset_age_group': onset_grp,
                    'onset_group_midpoint': int(onset_grp.split('-')[0]) + 5,
                    'median_onset_age': median_onset_age,
                    'n': ci_result['n'],
                    'mean_burden': ci_result['estimate'],
                    'ci_lower': ci_result['ci_lower'],
                    'ci_upper': ci_result['ci_upper']
                })

    return pd.DataFrame(results)


def create_burden_summary_table(
    stats_df: pd.DataFrame,
    min_n: int = 5
) -> pd.DataFrame:
    """
    Create formatted burden summary table from onset/duration/race statistics.

    Parameters:
        stats_df: DataFrame, output from compute_burden_stats_by_onset_duration_race()
        min_n: int, minimum sample size to include (default: 5)

    Returns:
        DataFrame pivoted with duration_category and race_ethnicity as index,
        onset_age_group as columns, values as 'Mean (95% CI)' strings
    """
    table_df = stats_df[stats_df['n'] >= min_n].copy()
    table_df['Mean (95% CI)'] = table_df.apply(
        lambda row: f"{row['mean_burden']:.1f} ({row['ci_lower']:.1f}-{row['ci_upper']:.1f})",
        axis=1
    )
    pivot = table_df.pivot_table(
        index=['duration_category', 'race_ethnicity'],
        columns='onset_age_group',
        values='Mean (95% CI)',
        aggfunc='first'
    )
    return pivot

def test_case_vs_control_elevation(df):
    """
    Test whether cases have significantly higher mutation burden than
    predicted by the age-matched control regression, within each race group.

    For each case, calculates a residual (observed - predicted from controls)
    and uses a one-sample t-test to assess whether the mean residual > 0.
    """

    race_groups = ['White Non-Hispanic', 'White Hispanic', 'Black']

    control_regressions = {}
    for race in race_groups:
        control_mask = (df['race_ethnicity'] == race) & (df['disease_group'] == CONTROL_KEY)
        controls = df[control_mask].dropna(subset=['current_age', 'vcf_snv_count'])

        if len(controls) >= 20:
            slope, intercept = np.polyfit(controls['current_age'], controls['vcf_snv_count'], 1)
            control_regressions[race] = {'slope': slope, 'intercept': intercept, 'n': len(controls)}

    results = []

    for duration_label, duration_filter in [('Baseline (<=2yr)', lambda x: x <= 2),
                                             ('All Durations', lambda x: x.notna())]:
        print(f"\n{duration_label}")
        print("-" * 50)

        for race in race_groups:
            if race not in control_regressions:
                continue

            reg = control_regressions[race]

            case_mask = (
                (df['race_ethnicity'] == race) &
                (df['disease_group'].isin([SUBTYPE_A_KEY, SUBTYPE_B_KEY])) &
                (df['vcf_snv_count'].notna()) &
                (df['current_age'].notna())
            )

            if duration_label == 'Baseline (<=2yr)':
                case_mask = case_mask & (df['disease_duration'] <= 2)

            cases = df[case_mask].copy()

            if len(cases) < 10:
                print(f"  {race}: insufficient cases (n={len(cases)}), skipping")
                continue

            cases['predicted_burden'] = reg['slope'] * cases['current_age'] + reg['intercept']
            cases['residual'] = cases['vcf_snv_count'] - cases['predicted_burden']

            mean_residual = cases['residual'].mean()
            se_residual = cases['residual'].std() / np.sqrt(len(cases))
            t_stat, p_two_sided = stats.ttest_1samp(cases['residual'], 0)
            p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2

            ci_lower = mean_residual - 1.96 * se_residual
            ci_upper = mean_residual + 1.96 * se_residual

            mean_predicted = cases['predicted_burden'].mean()
            pct_elevation = (mean_residual / mean_predicted) * 100

            sig_label = "p < 0.05" if p_one_sided < 0.05 else "n.s."

            print(f"\n  {race} (n={len(cases)}):")
            print(f"    Observed mean: {cases['vcf_snv_count'].mean():.1f}")
            print(f"    Predicted mean (controls): {mean_predicted:.1f}")
            print(f"    Elevation: {mean_residual:.1f} SNVs ({pct_elevation:.1f}%), 95% CI: ({ci_lower:.1f}, {ci_upper:.1f})")
            print(f"    One-sided p = {p_one_sided:.4f} ({sig_label})")

            results.append({
                'duration_category': duration_label,
                'race_ethnicity': race,
                'n_cases': len(cases),
                'mean_observed': cases['vcf_snv_count'].mean(),
                'mean_predicted': mean_predicted,
                'mean_elevation': mean_residual,
                'elevation_ci_lower': ci_lower,
                'elevation_ci_upper': ci_upper,
                'pct_elevation': pct_elevation,
                'p_value_one_sided': p_one_sided
            })

    return pd.DataFrame(results), control_regressions


def test_young_vs_old_elevation(df):
    """
    Test whether the case-control elevation in mutation burden is larger
    in young-onset (<40) versus older-onset (>=40) patients, within each
    race group. Restricted to baseline cases (duration <= 2 years).

    Uses residuals from the control regression and a two-sample t-test
    (one-sided: young > older).
    """

    race_groups = ['White Non-Hispanic', 'White Hispanic', 'Black']

    control_regressions = {}
    for race in race_groups:
        control_mask = (df['race_ethnicity'] == race) & (df['disease_group'] == CONTROL_KEY)
        controls = df[control_mask].dropna(subset=['current_age', 'vcf_snv_count'])

        if len(controls) >= 20:
            slope, intercept = np.polyfit(controls['current_age'], controls['vcf_snv_count'], 1)
            control_regressions[race] = {'slope': slope, 'intercept': intercept}

    results = []

    for race in race_groups:
        if race not in control_regressions:
            continue

        reg = control_regressions[race]

        case_mask = (
            (df['race_ethnicity'] == race) &
            (df['disease_group'].isin([SUBTYPE_A_KEY, SUBTYPE_B_KEY])) &
            (df['disease_duration'] <= 2) &
            (df['vcf_snv_count'].notna()) &
            (df['age_at_onset'].notna())
        )

        cases = df[case_mask].copy()
        cases['predicted_burden'] = reg['slope'] * cases['current_age'] + reg['intercept']
        cases['residual'] = cases['vcf_snv_count'] - cases['predicted_burden']

        young = cases[cases['age_at_onset'] < 40]['residual']
        older = cases[cases['age_at_onset'] >= 40]['residual']

        print(f"\n{race}")
        print("-" * 50)
        print(f"  Young onset (<40):  n={len(young)}, mean elevation={young.mean():.1f} (SD={young.std():.1f})")
        print(f"  Older onset (>=40): n={len(older)}, mean elevation={older.mean():.1f} (SD={older.std():.1f})")

        if len(young) >= 5 and len(older) >= 5:
            t_stat, p_two_sided = stats.ttest_ind(young, older)
            p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
            diff = young.mean() - older.mean()
            sig_label = "p < 0.05" if p_one_sided < 0.05 else "n.s."

            print(f"  Difference (young - older): {diff:.1f} SNVs")
            print(f"  One-sided p = {p_one_sided:.4f} ({sig_label})")

            results.append({
                'race_ethnicity': race,
                'n_young': len(young),
                'n_older': len(older),
                'mean_elevation_young': young.mean(),
                'mean_elevation_older': older.mean(),
                'difference': diff,
                'p_value': p_one_sided
            })
        else:
            print(f"  Insufficient sample size for comparison")

    return pd.DataFrame(results)

def test_duration_effect(df):
    """
    Test whether disease duration is associated with increased mutation burden
    within each race group using multiple regression (burden ~ age + duration).

    Also fits an interaction model to assess whether the duration effect
    differs across race groups. Results are cross-sectional, not causal.
    """

    race_groups = ['White Non-Hispanic', 'White Hispanic', 'Black']

    results = []

    for race in race_groups:
        case_mask = (
            (df['race_ethnicity'] == race) &
            (df['disease_group'].isin([SUBTYPE_A_KEY, SUBTYPE_B_KEY])) &
            (df['vcf_snv_count'].notna()) &
            (df['current_age'].notna()) &
            (df['disease_duration'].notna())
        )
        cases = df[case_mask].copy()

        if len(cases) < 30:
            print(f"{race}: insufficient cases (n={len(cases)}), skipping")
            continue

        model = smf.ols('vcf_snv_count ~ current_age + disease_duration', data=cases).fit()

        duration_coef = model.params['disease_duration']
        duration_se = model.bse['disease_duration']
        duration_p = model.pvalues['disease_duration']
        duration_ci = model.conf_int().loc['disease_duration']

        sig_label = "p < 0.05" if duration_p < 0.05 else "n.s."

        print(f"\n{race} (n={len(cases)})")
        print(f"  Duration coeff: {duration_coef:.2f} SNVs/yr (SE: {duration_se:.2f})")
        print(f"  95% CI: ({duration_ci[0]:.2f}, {duration_ci[1]:.2f})")
        print(f"  p = {duration_p:.4f} ({sig_label})")

        results.append({
            'race_ethnicity': race,
            'n': len(cases),
            'duration_coef': duration_coef,
            'duration_se': duration_se,
            'duration_ci_lower': duration_ci[0],
            'duration_ci_upper': duration_ci[1],
            'duration_p': duration_p
        })

    all_cases = df[
        (df['race_ethnicity'].isin(race_groups)) &
        (df['disease_group'].isin([SUBTYPE_A_KEY, SUBTYPE_B_KEY])) &
        (df['vcf_snv_count'].notna()) &
        (df['current_age'].notna()) &
        (df['disease_duration'].notna())
    ].copy()

    interaction_model = smf.ols(
        'vcf_snv_count ~ current_age + disease_duration * C(race_ethnicity, Treatment("White Non-Hispanic"))',
        data=all_cases
    ).fit()

    print("\nRace by duration interaction model:")
    print(interaction_model.summary().tables[1])

    return pd.DataFrame(results), interaction_model

def parse_gff_attributes(attr_string: str) -> dict:
    """Parse GFF3 attribute column into a dictionary."""
    attributes = {}
    for item in attr_string.split(';'):
        if '=' in item:
            key, value = item.split('=', 1)
            attributes[key.strip()] = value.strip()
    return attributes


def merge_and_sum_regions(regions: list) -> int:
    """Merge overlapping genomic regions and return total length."""
    if not regions:
        return 0

    by_chrom = defaultdict(list)
    for chrom, start, end in regions:
        by_chrom[chrom].append((start, end))

    total_length = 0
    for chrom, intervals in by_chrom.items():
        intervals.sort()
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        for start, end in merged:
            total_length += (end - start + 1)

    return total_length


def load_cds_lengths_from_gff(gff_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Parse GFF file and calculate total CDS length per gene.

    Parameters:
        gff_path: str, path to GFF3 annotation file
        verbose: bool, print progress and summary statistics (default: True)

    Returns:
        DataFrame with columns: gene, cds_length_bp, cds_length_kb
    """
    if verbose:
        print(f"Parsing GFF file: {gff_path}")

    transcript_to_gene = {}
    gene_cds_regions = defaultdict(list)
    n_lines = 0
    n_cds = 0

    with open(gff_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            n_lines += 1
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            chrom, source, feature_type, start, end, score, strand, phase, attributes_str = fields
            start, end = int(start), int(end)
            attributes = parse_gff_attributes(attributes_str)

            if feature_type in ['mRNA', 'transcript', 'primary_transcript']:
                transcript_id = attributes.get('ID', '')
                gene_name = attributes.get('gene', '')
                if transcript_id and gene_name:
                    transcript_to_gene[transcript_id] = gene_name

            elif feature_type == 'CDS':
                n_cds += 1
                parent = attributes.get('Parent', '')
                gene_name = attributes.get('gene', '')
                if not gene_name and parent:
                    gene_name = transcript_to_gene.get(parent, None)
                if gene_name:
                    gene_cds_regions[gene_name].append((chrom, start, end))

            if verbose and n_lines % 500000 == 0:
                print(f"  Processed {n_lines:,} lines...")

    if verbose:
        print(f"  Parsed {n_lines:,} lines")
        print(f"  Found {n_cds:,} CDS entries")
        print(f"  Genes with CDS data: {len(gene_cds_regions):,}")

    gene_lengths = {}
    for gene, regions in gene_cds_regions.items():
        gene_lengths[gene] = merge_and_sum_regions(regions)

    df = pd.DataFrame([
        {'gene': gene, 'cds_length_bp': length}
        for gene, length in gene_lengths.items()
    ])
    df['cds_length_kb'] = df['cds_length_bp'] / 1000

    if verbose:
        print(f"  Median CDS length: {df['cds_length_bp'].median():,.0f} bp")
        print(f"  Mean CDS length: {df['cds_length_bp'].mean():,.0f} bp")
        print(f"  Range: {df['cds_length_bp'].min():,.0f} - {df['cds_length_bp'].max():,.0f} bp")

    return df


def calculate_length_normalized_rates(
    variants_df: pd.DataFrame,
    cds_lengths_df: pd.DataFrame,
    sample_col: str = None,
    gene_col: str = None
) -> pd.DataFrame:
    """
    Calculate length-normalized mutation rates per gene.

    Parameters:
        variants_df: DataFrame, protein-altering variants
        cds_lengths_df: DataFrame, output from load_cds_lengths_from_gff()
        sample_col: str, sample column name
        gene_col: str, gene column name

    Returns:
        DataFrame with per-gene mutation counts, CDS length, and normalized rates
        (mutations_per_kb, samples_per_kb), sorted by samples_per_kb descending
    """
    if sample_col is None:
        sample_col = VCF_SAMPLE_COL
    if gene_col is None:
        gene_col = VCF_GENE_COL
    gene_mutation_counts = variants_df.groupby(gene_col).agg({
        sample_col: ['count', 'nunique']
    }).reset_index()
    gene_mutation_counts.columns = [gene_col, 'n_mutations', 'n_samples']

    merged = gene_mutation_counts.merge(
        cds_lengths_df,
        left_on=gene_col,
        right_on='gene',
        how='left'
    )

    n_missing = merged['cds_length_bp'].isna().sum()
    if n_missing > 0:
        print(f"  Warning: {n_missing} genes missing CDS length data (using median length)")

    median_length_kb = cds_lengths_df['cds_length_kb'].median()
    mask = merged['cds_length_kb'].isna()
    merged.loc[mask, 'cds_length_kb'] = median_length_kb
    merged.loc[mask, 'cds_length_bp'] = median_length_kb * 1000
    merged.loc[mask, 'length_imputed'] = True
    merged['length_imputed'] = merged['length_imputed'].fillna(False)

    merged['mutations_per_kb'] = merged['n_mutations'] / merged['cds_length_kb']
    merged['samples_per_kb'] = merged['n_samples'] / merged['cds_length_kb']

    return merged.sort_values('samples_per_kb', ascending=False)


def get_query_and_background_genes(
    filtered_df: pd.DataFrame,
    cds_lengths_df: pd.DataFrame,
    min_samples: int = 2,
    percentile_threshold: int = 90
) -> tuple:
    """
    Get query genes using length-normalized selection (combined method)
    and background genes from all observed genes.

    The combined method first requires a gene to be mutated in >= min_samples
    unique samples, then selects the top percentile by samples_per_kb rate.
    This filters out genes whose high sample count is driven by long CDS length
    rather than biological signal.

    Parameters:
        filtered_df: DataFrame, protein-altering variants with GENE and SAMPLE columns
        cds_lengths_df: DataFrame, output from load_cds_lengths_from_gff()
        min_samples: int, minimum unique samples a gene must be mutated in (default: 2)
        percentile_threshold: int, percentile cutoff for samples_per_kb rate (default: 90)

    Returns:
        Tuple of (query_genes list, background_genes list, n_samples int, gene_stats DataFrame)
    """
    n_samples = filtered_df[VCF_SAMPLE_COL].nunique()
    background_genes = filtered_df[VCF_GENE_COL].dropna().unique().tolist()
    gene_stats = calculate_length_normalized_rates(filtered_df, cds_lengths_df)

    if len(gene_stats) > 0 and gene_stats['n_samples'].max() >= min_samples:
        eligible = gene_stats[gene_stats['n_samples'] >= min_samples]
        if len(eligible) > 0:
            cutoff = np.percentile(eligible['samples_per_kb'].dropna(), percentile_threshold)
            query_genes = eligible[eligible['samples_per_kb'] >= cutoff][VCF_GENE_COL].tolist()
        else:
            query_genes = []
    else:
        query_genes = []

    return query_genes, background_genes, n_samples, gene_stats


def merge_variants_with_analysis(
    variants_df: pd.DataFrame,
    analysis_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge variant-level data with sample-level metadata.

    Parameters:
        variants_df: DataFrame, variant-level data with SAMPLE column
        analysis_df: DataFrame, sample-level data from create_mutation_burden_analysis_df()

    Returns:
        DataFrame with variant rows annotated by sample metadata
    """
    sample_cols = [VCF_SAMPLE_COL, 'disease_group', 'race_ethnicity',
                   'age_at_onset', 'disease_duration', 'current_age']
    available_cols = [c for c in sample_cols if c in analysis_df.columns]
    merged = variants_df.merge(analysis_df[available_cols], on=VCF_SAMPLE_COL, how='inner')
    return merged


def filter_protein_altering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to protein-altering variants only using PROTEIN_ALTERING_FUNCS.

    Parameters:
        df: DataFrame, variant-level data with FUNC column

    Returns:
        DataFrame filtered to protein-altering functional categories
    """
    return df[df[VCF_FUNC_COL].isin(PROTEIN_ALTERING_FUNCS)].copy()


def run_gobp_enrichment(
    query_genes: list,
    background_genes: list,
    analysis_name: str,
    gene_sets: str = None,
    min_query_genes: int = 10
) -> pd.DataFrame:
    """
    Run GO:BP over-representation analysis via gseapy.

    Parameters:
        query_genes: list, genes of interest (length-normalized, mutated in >= N samples)
        background_genes: list, all genes in the dataset for hypergeometric test
        analysis_name: str, label for this analysis
        gene_sets: str, gseapy gene set library name (default: from config)
        min_query_genes: int, minimum query genes required to run (default: 10)

    Returns:
        DataFrame with enrichment results including Term, P-value, Adjusted P-value,
        Odds Ratio, Overlap, and analysis metadata columns
    """
    if gene_sets is None:
        gene_sets = GSEAPY_GOBP_LIBRARY
    query_genes = [g for g in query_genes if pd.notna(g) and g != '']
    background_genes = [g for g in background_genes if pd.notna(g) and g != '']

    print(f"Analysis: {analysis_name}")
    print(f"  Query genes: {len(query_genes)}, Background genes: {len(background_genes)}")

    if len(query_genes) < min_query_genes:
        print(f"  WARNING: Too few query genes (<{min_query_genes}). Skipping.")
        return pd.DataFrame()

    try:
        enr = gp.enrich(
            gene_list=query_genes,
            gene_sets=gene_sets,
            background=background_genes,
            outdir=None,
            verbose=False
        )
        results = enr.results.copy()

        if len(results) == 0:
            print(f"  No enriched terms found.")
            return pd.DataFrame()

        results['analysis_name'] = analysis_name
        results['n_query_genes'] = len(query_genes)
        results['n_background_genes'] = len(background_genes)

        if 'Overlap' in results.columns:
            results['overlap_count'] = results['Overlap'].apply(
                lambda x: int(x.split('/')[0]) if isinstance(x, str) and '/' in x else 0
            )
            results['term_size'] = results['Overlap'].apply(
                lambda x: int(x.split('/')[1]) if isinstance(x, str) and '/' in x else 0
            )

        pval_col = 'P-value' if 'P-value' in results.columns else 'p_value'
        adjp_col = 'Adjusted P-value' if 'Adjusted P-value' in results.columns else 'adjusted_p_value'
        results = results.sort_values(pval_col)

        n_sig = (results[adjp_col] < 0.05).sum()
        print(f"  Terms tested: {len(results)}, Significant (FDR<0.05): {n_sig}")
        return results

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return pd.DataFrame()


def define_analysis_groups(merged_df: pd.DataFrame) -> dict:
    """
    Define all analysis subgroups and their filtered dataframes.

    Parameters:
        merged_df: DataFrame, merged variants + analysis data (already protein-altering filtered)

    Returns:
        Dict of {analysis_name: filtered DataFrame}
    """
    analyses = {}

    baseline_mask = (
        (merged_df['disease_group'].isin([SUBTYPE_A_KEY, SUBTYPE_B_KEY])) &
        (merged_df['disease_duration'] <= 2)
    )
    baseline = merged_df[baseline_mask]
    analyses['1_baseline_all_races'] = baseline.copy()

    young_baseline = baseline[
        (baseline['age_at_onset'] < 40) &
        (baseline['race_ethnicity'].isin(['Black', 'White Non-Hispanic']))
    ]
    analyses['2_baseline_young_black_wnh'] = young_baseline.copy()
    analyses['3_baseline_older_60plus'] = baseline[baseline['age_at_onset'] >= 60].copy()
    analyses['4_baseline_limited'] = baseline[baseline['disease_group'] == SUBTYPE_A_KEY].copy()
    analyses['5_baseline_diffuse'] = baseline[baseline['disease_group'] == SUBTYPE_B_KEY].copy()

    all_ssc = merged_df[merged_df['disease_group'].isin([SUBTYPE_A_KEY, SUBTYPE_B_KEY])]
    analyses['6_all_ssc'] = all_ssc.copy()
    analyses['7_all_limited'] = all_ssc[all_ssc['disease_group'] == SUBTYPE_A_KEY].copy()
    analyses['8_all_diffuse'] = all_ssc[all_ssc['disease_group'] == SUBTYPE_B_KEY].copy()

    young_diffuse = all_ssc[
        (all_ssc['disease_group'] == SUBTYPE_B_KEY) & (all_ssc['age_at_onset'] < 40)
    ]
    analyses['9_young_diffuse'] = young_diffuse.copy()

    controls = merged_df[merged_df['disease_group'] == CONTROL_KEY]
    analyses['10_controls_all'] = controls.copy()
    analyses['11_controls_young'] = controls[controls['current_age'] < 40].copy()

    return analyses


def run_full_gobp_pipeline(
    variants_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    gff_path: str,
    output_dir: str = '.',
    min_samples: int = 2,
    percentile_threshold: int = 90,
    gene_sets: str = None
) -> tuple:
    """
    Run GO:BP enrichment pipeline with length-normalized gene selection
    for all defined analysis subgroups.

    Parameters:
        variants_df: DataFrame, variant-level data with GENE, FUNC, SAMPLE columns
        analysis_df: DataFrame, sample-level data with disease_group, race_ethnicity, etc.
        gff_path: str, path to GFF3 annotation file for CDS length extraction
        output_dir: str, directory to save output TSV files (default: '.')
        min_samples: int, minimum samples a gene must be mutated in (default: 2)
        percentile_threshold: int, percentile cutoff for length-normalized rate (default: 90)
        gene_sets: str, gseapy gene set library name (default: from config)

    Returns:
        Tuple of (results_dict {name: DataFrame}, summary_df DataFrame)
    """
    if gene_sets is None:
        gene_sets = GSEAPY_GOBP_LIBRARY
    os.makedirs(output_dir, exist_ok=True)

    print(f"GO:BP Enrichment Pipeline | Library: {gene_sets} | Min samples: {min_samples}")
    print(f"  Length normalization percentile: {percentile_threshold}")

    cds_lengths_df = load_cds_lengths_from_gff(gff_path, verbose=True)
    cds_lengths_df.to_csv(os.path.join(output_dir, 'cds_lengths.tsv'), sep='\t', index=False)

    merged = merge_variants_with_analysis(variants_df, analysis_df)
    merged = filter_protein_altering(merged)
    print(f"  Protein-altering variants: {len(merged)}, Samples: {merged[VCF_SAMPLE_COL].nunique()}")

    _, global_background, _, _ = get_query_and_background_genes(merged, cds_lengths_df, min_samples=1)
    print(f"  Global background genes: {len(global_background)}\n")

    analysis_groups = define_analysis_groups(merged)
    all_results = {}
    summary_data = []

    for analysis_name, group_df in analysis_groups.items():
        query_genes, _, n_samples, gene_stats = get_query_and_background_genes(
            group_df, cds_lengths_df,
            min_samples=min_samples,
            percentile_threshold=percentile_threshold
        )

        print(f"  Query genes (normalized): {len(query_genes)}")

        results = run_gobp_enrichment(
            query_genes=query_genes,
            background_genes=global_background,
            analysis_name=analysis_name,
            gene_sets=gene_sets
        )
        all_results[analysis_name] = results

        pval_col = 'P-value' if len(results) > 0 and 'P-value' in results.columns else 'p_value'
        adjp_col = 'Adjusted P-value' if len(results) > 0 and 'Adjusted P-value' in results.columns else 'adjusted_p_value'

        if len(results) > 0:
            summary_data.append({
                'analysis': analysis_name,
                'n_samples': n_samples,
                'n_query_genes': len(query_genes),
                'n_terms_tested': len(results),
                'n_sig_pval': (results[pval_col] < 0.05).sum(),
                'n_sig_adjp': (results[adjp_col] < 0.05).sum(),
                'top_term': results.iloc[0].get('Term', 'N/A'),
                'top_pval': results.iloc[0][pval_col]
            })
            results.to_csv(os.path.join(output_dir, f'gobp_{analysis_name}.tsv'), sep='\t', index=False)
        else:
            summary_data.append({
                'analysis': analysis_name, 'n_samples': n_samples,
                'n_query_genes': len(query_genes), 'n_terms_tested': 0,
                'n_sig_pval': 0, 'n_sig_adjp': 0, 'top_term': 'N/A', 'top_pval': np.nan
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'gobp_summary.tsv'), sep='\t', index=False)
    print(f"\nPipeline complete. Results saved to: {output_dir}")
    return all_results, summary_df


def run_age_matched_comparison(
    variants_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    gff_path: str,
    age_bins: list = None,
    min_samples: int = 2,
    percentile_threshold: int = 90,
    gene_sets: str = None,
    output_dir: str = '.'
) -> dict:
    """
    Run age-matched GO:BP enrichment comparing SSc cases vs controls
    with length-normalized gene selection.

    Parameters:
        variants_df: DataFrame, variant-level data
        analysis_df: DataFrame, sample-level data
        gff_path: str, path to GFF3 annotation file for CDS length extraction
        age_bins: list of tuples, age bins for matching (default: [(0,40),(40,60),(60,100)])
        min_samples: int, minimum samples for query genes (default: 2)
        percentile_threshold: int, percentile cutoff for length-normalized rate (default: 90)
        gene_sets: str, gene set library (default: from config)
        output_dir: str, output directory (default: '.')

    Returns:
        Dict of {age_bin_label: {'cases': results_df, 'controls': results_df}}
    """
    if gene_sets is None:
        gene_sets = GSEAPY_GOBP_LIBRARY
    if age_bins is None:
        age_bins = [(0, 40), (40, 60), (60, 100)]
    os.makedirs(output_dir, exist_ok=True)

    print("Age-matched case vs control comparison")

    cds_lengths_df = load_cds_lengths_from_gff(gff_path, verbose=True)

    merged = merge_variants_with_analysis(variants_df, analysis_df)
    merged = filter_protein_altering(merged)
    _, global_background, _, _ = get_query_and_background_genes(merged, cds_lengths_df, min_samples=1)

    results = {}
    for age_min, age_max in age_bins:
        bin_label = f"age_{age_min}_{age_max}"
        results[bin_label] = {}
        print(f"\n  Age bin: {age_min}-{age_max}")

        cases = merged[
            (merged['disease_group'].isin([SUBTYPE_A_KEY, SUBTYPE_B_KEY])) &
            (merged['age_at_onset'] >= age_min) &
            (merged['age_at_onset'] < age_max)
        ]
        controls = merged[
            (merged['disease_group'] == CONTROL_KEY) &
            (merged['current_age'] >= age_min) &
            (merged['current_age'] < age_max)
        ]

        for label, group_df in [('cases', cases), ('controls', controls)]:
            query, _, n, _ = get_query_and_background_genes(
                group_df, cds_lengths_df,
                min_samples=min_samples,
                percentile_threshold=percentile_threshold
            )
            print(f"    {label}: {n} samples, {len(query)} query genes")
            if len(query) >= 10:
                results[bin_label][label] = run_gobp_enrichment(
                    query_genes=query, background_genes=global_background,
                    analysis_name=f'{bin_label}_{label}', gene_sets=gene_sets
                )
            else:
                print(f"    Skipping {label} - too few query genes")
                results[bin_label][label] = pd.DataFrame()

    return results


def compare_enrichment_case_vs_control(
    case_results: pd.DataFrame,
    control_results: pd.DataFrame,
    p_threshold: float = 0.05,
    or_ratio_threshold: float = 1.5
) -> pd.DataFrame:
    """
    Compare enrichment between cases and controls for each GO term.

    Parameters:
        case_results: DataFrame, enrichment results for cases
        control_results: DataFrame, enrichment results for controls
        p_threshold: float, significance threshold (default: 0.05)
        or_ratio_threshold: float, minimum OR ratio for case-stronger category (default: 1.5)

    Returns:
        DataFrame with per-term comparison: p-values, odds ratios, and classification
        category (Case-specific, Control-specific, Both significant, Not significant)
    """
    if len(case_results) == 0:
        print("No case results to compare")
        return pd.DataFrame()

    term_col = 'Term' if 'Term' in case_results.columns else 'term'
    pval_col = 'P-value' if 'P-value' in case_results.columns else 'p_value'
    adjp_col = 'Adjusted P-value' if 'Adjusted P-value' in case_results.columns else 'adjusted_p_value'
    or_col = 'Odds Ratio' if 'Odds Ratio' in case_results.columns else 'odds_ratio'

    all_terms = set(case_results[term_col].tolist())
    ctrl_term_col = None
    if len(control_results) > 0:
        ctrl_term_col = 'Term' if 'Term' in control_results.columns else 'term'
        all_terms.update(control_results[ctrl_term_col].tolist())

    comparison_data = []
    for term in all_terms:
        row = {'term': term}

        case_match = case_results[case_results[term_col] == term]
        if len(case_match) > 0:
            row['case_pval'] = case_match[pval_col].values[0]
            row['case_adjp'] = case_match[adjp_col].values[0]
            row['case_or'] = case_match[or_col].values[0] if or_col in case_match.columns else np.nan
            row['case_significant'] = row['case_adjp'] < p_threshold
        else:
            row.update({'case_pval': 1.0, 'case_adjp': 1.0, 'case_or': 1.0, 'case_significant': False})

        if len(control_results) > 0 and ctrl_term_col is not None:
            ctrl_match = control_results[control_results[ctrl_term_col] == term]
            if len(ctrl_match) > 0:
                c_pval = 'P-value' if 'P-value' in control_results.columns else 'p_value'
                c_adjp = 'Adjusted P-value' if 'Adjusted P-value' in control_results.columns else 'adjusted_p_value'
                c_or = 'Odds Ratio' if 'Odds Ratio' in control_results.columns else 'odds_ratio'
                row['control_pval'] = ctrl_match[c_pval].values[0]
                row['control_adjp'] = ctrl_match[c_adjp].values[0]
                row['control_or'] = ctrl_match[c_or].values[0] if c_or in ctrl_match.columns else np.nan
                row['control_significant'] = row['control_adjp'] < p_threshold
            else:
                row.update({'control_pval': 1.0, 'control_adjp': 1.0, 'control_or': 1.0, 'control_significant': False})
        else:
            row.update({'control_pval': 1.0, 'control_adjp': 1.0, 'control_or': 1.0, 'control_significant': False})

        row['or_ratio'] = row['case_or'] / row['control_or'] if row['control_or'] > 0 else np.inf
        row['log_pval_diff'] = -np.log10(row['case_pval']) - (-np.log10(row['control_pval']))

        if row['case_significant'] and not row['control_significant']:
            row['category'] = 'Case-specific'
        elif row['case_significant'] and row['control_significant']:
            if row['or_ratio'] >= or_ratio_threshold:
                row['category'] = 'Both significant, stronger in cases'
            elif row['or_ratio'] <= 1 / or_ratio_threshold:
                row['category'] = 'Both significant, stronger in controls'
            else:
                row['category'] = 'Both significant, similar'
        elif not row['case_significant'] and row['control_significant']:
            row['category'] = f'{CONTROL_LABEL}-specific'
        else:
            row['category'] = 'Not significant in either'

        comparison_data.append(row)

    return pd.DataFrame(comparison_data).sort_values('case_pval')


def identify_control_enriched_terms(
    results_dict: dict,
    p_threshold: float = 0.05,
    use_adjusted: bool = True
) -> set:
    """
    Identify GO terms significantly enriched in control analyses.

    Parameters:
        results_dict: dict, {analysis_name: results_df} from pipeline
        p_threshold: float, significance threshold (default: 0.05)
        use_adjusted: bool, use adjusted p-value (default: True)

    Returns:
        Set of GO term strings enriched in any control analysis
    """
    control_terms = set()
    control_keys = [k for k in results_dict.keys() if CONTROL_KEY in k.lower()]

    for key in control_keys:
        results = results_dict[key]
        if len(results) == 0:
            continue
        term_col = 'Term' if 'Term' in results.columns else 'term'
        pval_col = ('Adjusted P-value' if use_adjusted and 'Adjusted P-value' in results.columns
                     else 'P-value' if 'P-value' in results.columns else 'p_value')
        sig_terms = results[results[pval_col] < p_threshold][term_col].tolist()
        control_terms.update(sig_terms)

    print(f"Identified {len(control_terms)} control-enriched terms (p < {p_threshold})")
    return control_terms


def get_clean_disease_enriched_terms(
    results_dict: dict,
    p_threshold: float = 0.05,
    use_adjusted: bool = True
) -> tuple:
    """
    Get disease-specific enrichment by removing control-enriched terms from case results.

    Parameters:
        results_dict: dict, {analysis_name: results_df} from pipeline
        p_threshold: float, significance threshold (default: 0.05)
        use_adjusted: bool, use adjusted p-value (default: True)

    Returns:
        Tuple of (filtered_results dict, summary DataFrame, control_terms set)
    """
    control_terms = identify_control_enriched_terms(results_dict, p_threshold, use_adjusted)

    filtered_results = {}
    summary_data = []
    case_keys = [k for k in results_dict.keys() if CONTROL_KEY not in k.lower()]

    for key in case_keys:
        results = results_dict[key]
        if len(results) == 0:
            filtered_results[key] = results
            summary_data.append({
                'analysis': key, 'n_total_terms': 0, 'n_sig_adjp': 0,
                'top_term': 'N/A', 'top_adjp': np.nan
            })
            continue

        term_col = 'Term' if 'Term' in results.columns else 'term'
        pval_col = ('Adjusted P-value' if use_adjusted and 'Adjusted P-value' in results.columns
                     else 'P-value' if 'P-value' in results.columns else 'p_value')

        filtered = results[~results[term_col].isin(control_terms)].copy()
        filtered_results[key] = filtered

        n_removed = len(results) - len(filtered)
        n_sig = (filtered[pval_col] < p_threshold).sum() if len(filtered) > 0 else 0
        print(f"{key}: removed {n_removed} control terms, {n_sig} significant disease-specific terms remain")

        summary_data.append({
            'analysis': key,
            'n_total_terms': len(filtered),
            'n_sig_adjp': n_sig,
            'top_term': filtered.iloc[0][term_col] if len(filtered) > 0 else 'N/A',
            'top_adjp': filtered.iloc[0][pval_col] if len(filtered) > 0 else np.nan
        })

    return filtered_results, pd.DataFrame(summary_data), control_terms


def run_complete_gobp_analysis(
    variants_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    gff_path: str,
    output_dir: str = './gobp_analysis',
    min_samples: int = 2,
    percentile_threshold: int = 90,
    gene_sets: str = None
) -> dict:
    """
    Run the full GO:BP pipeline: length-normalized standard enrichment,
    age-matched comparison, and disease-specific filtering.

    Parameters:
        variants_df: DataFrame, variant-level data
        analysis_df: DataFrame, sample-level data
        gff_path: str, path to GFF3 annotation file for CDS length extraction
        output_dir: str, base output directory (default: './gobp_analysis')
        min_samples: int, minimum samples for query genes (default: 2)
        percentile_threshold: int, percentile cutoff for length-normalized rate (default: 90)
        gene_sets: str, gene set library (default: from config)

    Returns:
        Dict with keys: standard_results, standard_summary, age_matched,
        filtered_results, filter_summary, control_terms_removed
    """
    if gene_sets is None:
        gene_sets = GSEAPY_GOBP_LIBRARY
    os.makedirs(output_dir, exist_ok=True)
    all_outputs = {}

    results, summary = run_full_gobp_pipeline(
        variants_df, analysis_df, gff_path,
        output_dir=os.path.join(output_dir, 'standard'),
        min_samples=min_samples,
        percentile_threshold=percentile_threshold,
        gene_sets=gene_sets
    )
    all_outputs['standard_results'] = results
    all_outputs['standard_summary'] = summary

    age_results = run_age_matched_comparison(
        variants_df, analysis_df, gff_path,
        output_dir=os.path.join(output_dir, 'age_matched_comparison'),
        min_samples=min_samples,
        percentile_threshold=percentile_threshold,
        gene_sets=gene_sets
    )
    comparisons = {}
    for bin_label, bin_results in age_results.items():
        comparisons[bin_label] = compare_enrichment_case_vs_control(
            bin_results.get('cases', pd.DataFrame()),
            bin_results.get('controls', pd.DataFrame())
        )
    all_outputs['age_matched'] = {'age_matched_results': age_results, 'comparisons': comparisons}

    filtered_results, filter_summary, control_terms = get_clean_disease_enriched_terms(results)
    all_outputs['filtered_results'] = filtered_results
    all_outputs['filter_summary'] = filter_summary
    all_outputs['control_terms_removed'] = control_terms

    os.makedirs(os.path.join(output_dir, 'disease_specific'), exist_ok=True)
    for key, res in filtered_results.items():
        if len(res) > 0:
            res.to_csv(os.path.join(output_dir, 'disease_specific', f'clean_gobp_{key}.tsv'),
                       sep='\t', index=False)

    print(f"\nComplete analysis saved to: {output_dir}")
    return all_outputs

# NK cell gene set characterization

def _resolve_gseapy_term_names(
    go_term_dict: dict,
    gseapy_library: str = None
) -> dict:
    """
    Map GO term IDs to their full gseapy library term names.

    Parameters:
        go_term_dict: dict, {GO_ID: term_name} from config
        gseapy_library: str, gseapy library name (default: from config)

    Returns:
        Dict of {GO_ID: full_gseapy_term_string} for matched terms
    """
    if gseapy_library is None:
        gseapy_library = GSEAPY_GOBP_LIBRARY
    library = gp.get_library(gseapy_library)
    go_id_to_gseapy = {}
    for full_term_name in library.keys():
        for go_id in go_term_dict.keys():
            if go_id in full_term_name:
                go_id_to_gseapy[go_id] = full_term_name
                break
    matched = set(go_id_to_gseapy.keys())
    unmatched = set(go_term_dict.keys()) - matched
    if unmatched:
        print(f"WARNING: {len(unmatched)} GO terms not found in {gseapy_library}: {unmatched}")
    return go_id_to_gseapy


def get_nk_geneset_membership(
    gene_list: list = None,
    go_term_dict: dict = None,
    gseapy_library: str = None
) -> pd.DataFrame:
    """
    Look up which genes belong to which NK cell GO:BP terms in the gseapy library.

    Parameters:
        gene_list: list, gene symbols to check (default: NK_SOMATIC_GENES from config)
        go_term_dict: dict, {GO_ID: name} terms to query (default: NK_GOBP_SIGNIFICANT)
        gseapy_library: str, gseapy library name (default: from config)

    Returns:
        DataFrame with columns: gene, go_id, go_term_name, gseapy_term, is_member
    """
    if gseapy_library is None:
        gseapy_library = GSEAPY_GOBP_LIBRARY
    if gene_list is None:
        gene_list = NK_SOMATIC_GENES
    if go_term_dict is None:
        go_term_dict = NK_GOBP_SIGNIFICANT

    library = gp.get_library(gseapy_library)
    go_id_to_gseapy = _resolve_gseapy_term_names(go_term_dict, gseapy_library)

    rows = []
    for go_id, short_name in go_term_dict.items():
        gseapy_term = go_id_to_gseapy.get(go_id)
        term_genes = set(library.get(gseapy_term, [])) if gseapy_term else set()
        for gene in gene_list:
            rows.append({
                'gene': gene,
                'go_id': go_id,
                'go_term_name': short_name,
                'gseapy_term': gseapy_term if gseapy_term else 'NOT FOUND',
                'is_member': gene in term_genes
            })

    result = pd.DataFrame(rows)
    n_members = result['is_member'].sum()
    print(f"Gene-term membership: {n_members} of {len(rows)} gene-term pairs are members")
    return result


def count_variants_per_nk_goterm(
    tidy_vcf_df: pd.DataFrame,
    case_ids: list,
    membership_df: pd.DataFrame = None,
    vaf_threshold: float = None,
    protein_altering_funcs: list = None
) -> pd.DataFrame:
    """
    Count somatic variants per NK cell GO:BP term in disease samples at low VAF.

    Parameters:
        tidy_vcf_df: DataFrame, annotated variant data with GENE, FUNC, vaf, SAMPLE columns
        case_ids: list, sample IDs to include (disease cases only)
        membership_df: DataFrame, output of get_nk_geneset_membership() (default: computed)
        vaf_threshold: float, maximum VAF to include (default: from config)
        protein_altering_funcs: list, functional categories to include (default: from config)

    Returns:
        DataFrame with columns: go_id, go_term_name, n_variants, n_unique_variant_ids,
        n_genes_hit, n_patients, genes_hit
    """
    if vaf_threshold is None:
        vaf_threshold = NK_VAF_THRESHOLD
    if protein_altering_funcs is None:
        protein_altering_funcs = PROTEIN_ALTERING_FUNCS
    if membership_df is None:
        membership_df = get_nk_geneset_membership()

    case_ids_str = [str(s) for s in case_ids]
    mask = (
        (tidy_vcf_df[VCF_SAMPLE_COL].astype(str).isin(case_ids_str)) &
        (tidy_vcf_df[VCF_VAF_COL] < vaf_threshold) &
        (tidy_vcf_df[VCF_FUNC_COL].isin(protein_altering_funcs))
    )
    filtered = tidy_vcf_df[mask].copy()

    member_pairs = membership_df[membership_df['is_member']][['gene', 'go_id', 'go_term_name']].copy()
    nk_variants = filtered.merge(member_pairs, left_on=VCF_GENE_COL, right_on='gene', how='inner')

    if len(nk_variants) == 0:
        print("WARNING: No qualifying variants found in NK gene sets")
        return pd.DataFrame()

    summary = nk_variants.groupby(['go_id', 'go_term_name']).agg(
        n_variants=(VCF_VARIANT_ID_COL, 'size'),
        n_unique_variant_ids=(VCF_VARIANT_ID_COL, 'nunique'),
        n_genes_hit=(VCF_GENE_COL, 'nunique'),
        n_patients=(VCF_SAMPLE_COL, 'nunique'),
        genes_hit=(VCF_GENE_COL, lambda x: ', '.join(sorted(x.unique())))
    ).reset_index()

    print(f"NK GO term variant summary ({len(summary)} terms with hits):")
    for _, row in summary.iterrows():
        print(f"  {row['go_term_name']}: {row['n_variants']} variants, "
              f"{row['n_patients']} patients, genes: {row['genes_hit']}")
    return summary


# NK mutation carrier clinical comparison

def classify_nk_mutation_carriers(
    tidy_vcf_df: pd.DataFrame,
    case_ids: list,
    nk_genes: list = None,
    vaf_threshold: float = None,
    protein_altering_funcs: list = None
) -> set:
    """
    Identify disease samples carrying protein-altering somatic variants in NK genes at low VAF.

    Parameters:
        tidy_vcf_df: DataFrame, annotated variant data
        case_ids: list, disease sample IDs
        nk_genes: list, NK cell gene symbols (default: NK_SOMATIC_GENES from config)
        vaf_threshold: float, maximum VAF (default: from config)
        protein_altering_funcs: list, functional categories (default: from config)

    Returns:
        Set of SAMPLE IDs classified as NK mutation carriers
    """
    if vaf_threshold is None:
        vaf_threshold = NK_VAF_THRESHOLD
    if nk_genes is None:
        nk_genes = NK_SOMATIC_GENES
    if protein_altering_funcs is None:
        protein_altering_funcs = PROTEIN_ALTERING_FUNCS

    case_ids_str = [str(s) for s in case_ids]
    mask = (
        (tidy_vcf_df[VCF_SAMPLE_COL].astype(str).isin(case_ids_str)) &
        (tidy_vcf_df[VCF_GENE_COL].isin(nk_genes)) &
        (tidy_vcf_df[VCF_VAF_COL] < vaf_threshold) &
        (tidy_vcf_df[VCF_FUNC_COL].isin(protein_altering_funcs))
    )
    carrier_ids = set(tidy_vcf_df.loc[mask, VCF_SAMPLE_COL].astype(str).unique())
    print(f"NK mutation carriers: {len(carrier_ids)} of {len(case_ids_str)} cases "
          f"({len(carrier_ids)/len(case_ids_str)*100:.1f}%)")
    return carrier_ids


def compare_nk_carriers_vs_noncarriers(
    meta_df: pd.DataFrame,
    carrier_ids: set,
    case_ids: list,
    continuous_cols: list = None,
    categorical_cols: list = None
) -> pd.DataFrame:
    """
    Compare clinical and demographic features between NK mutation carriers and non-carriers.

    Parameters:
        meta_df: DataFrame, patient metadata
        carrier_ids: set, sample IDs of NK carriers (from classify_nk_mutation_carriers)
        case_ids: list, all disease case sample IDs
        continuous_cols: list, continuous variables to compare (default: from config)
        categorical_cols: list, categorical variables to compare (default: from config)

    Returns:
        DataFrame with variable, carrier/non-carrier summary stats, and p-value per row.
        Continuous: median (IQR), Mann-Whitney U p-value.
        Categorical: count (%), chi-square or Fisher's exact p-value.
    """
    if continuous_cols is None:
        continuous_cols = NK_COMPARISON_CONTINUOUS
    if categorical_cols is None:
        categorical_cols = NK_COMPARISON_CATEGORICAL

    case_ids_str = [str(s) for s in case_ids]
    df = meta_df[meta_df[META_SAMPLE_COL].astype(str).isin(case_ids_str)].copy()
    df['nk_carrier'] = df[META_SAMPLE_COL].astype(str).isin(carrier_ids)

    carriers = df[df['nk_carrier']]
    non_carriers = df[~df['nk_carrier']]
    print(f"Carriers: {len(carriers)}, Non-carriers: {len(non_carriers)}")

    results = []

    for col in continuous_cols:
        c_vals = pd.to_numeric(carriers[col], errors='coerce').dropna()
        nc_vals = pd.to_numeric(non_carriers[col], errors='coerce').dropna()

        c_med = c_vals.median()
        c_q1 = c_vals.quantile(0.25)
        c_q3 = c_vals.quantile(0.75)
        nc_med = nc_vals.median()
        nc_q1 = nc_vals.quantile(0.25)
        nc_q3 = nc_vals.quantile(0.75)

        if len(c_vals) >= 2 and len(nc_vals) >= 2:
            stat, p_val = mannwhitneyu(c_vals, nc_vals, alternative='two-sided')
        else:
            p_val = np.nan

        results.append({
            'variable': col,
            'type': 'continuous',
            'carrier_n': len(c_vals),
            'carrier_summary': f"{c_med:.1f} ({c_q1:.1f}-{c_q3:.1f})",
            'noncarrier_n': len(nc_vals),
            'noncarrier_summary': f"{nc_med:.1f} ({nc_q1:.1f}-{nc_q3:.1f})",
            'test': 'Mann-Whitney U',
            'p_value': p_val
        })

    for col in categorical_cols:
        c_counts = carriers[col].fillna('Unknown').value_counts()
        nc_counts = non_carriers[col].fillna('Unknown').value_counts()
        all_cats = sorted(set(c_counts.index) | set(nc_counts.index))

        contingency = pd.DataFrame({
            'carrier': [c_counts.get(cat, 0) for cat in all_cats],
            'noncarrier': [nc_counts.get(cat, 0) for cat in all_cats]
        }, index=all_cats)

        if contingency.shape == (2, 2):
            _, p_val = fisher_exact(contingency.values)
            test_name = "Fisher's exact"
        elif contingency.shape[0] >= 2:
            chi2, p_val, _, _ = chi2_contingency(contingency.values)
            test_name = 'Chi-square'
        else:
            p_val = np.nan
            test_name = 'N/A'

        c_total = carriers[col].notna().sum()
        nc_total = non_carriers[col].notna().sum()
        c_str = '; '.join([f"{cat}: {c_counts.get(cat, 0)}" for cat in all_cats])
        nc_str = '; '.join([f"{cat}: {nc_counts.get(cat, 0)}" for cat in all_cats])

        results.append({
            'variable': col,
            'type': 'categorical',
            'carrier_n': c_total,
            'carrier_summary': c_str,
            'noncarrier_n': nc_total,
            'noncarrier_summary': nc_str,
            'test': test_name,
            'p_value': p_val
        })

    results_df = pd.DataFrame(results)
    print(f"\nClinical comparison ({len(results_df)} variables):")
    for _, row in results_df.iterrows():
        sig = '*' if row['p_value'] < 0.05 else ''
        print(f"  {row['variable']}: p={row['p_value']:.4f} {sig}")
    return results_df


# NK variant detail tracking

def get_nk_variant_details(
    tidy_vcf_df: pd.DataFrame,
    case_ids: list,
    membership_df: pd.DataFrame = None,
    nk_genes: list = None,
    vaf_threshold: float = None,
    protein_altering_funcs: list = None
) -> pd.DataFrame:
    """
    Extract all qualifying variants in NK genes with per-variant frequency and GO term membership.

    Parameters:
        tidy_vcf_df: DataFrame, annotated variant data
        case_ids: list, disease sample IDs
        membership_df: DataFrame, gene-term membership (default: computed)
        nk_genes: list, NK cell gene symbols (default: NK_SOMATIC_GENES)
        vaf_threshold: float, maximum VAF (default: from config)
        protein_altering_funcs: list, functional categories (default: from config)

    Returns:
        DataFrame with columns: variant_id, GENE, SAMPLE, vaf, FUNC, CADD_PHRED,
        SIFT, POLYPHEN, variant_frequency (n patients sharing this variant_id),
        and GO term membership columns
    """
    if vaf_threshold is None:
        vaf_threshold = NK_VAF_THRESHOLD
    if nk_genes is None:
        nk_genes = NK_SOMATIC_GENES
    if protein_altering_funcs is None:
        protein_altering_funcs = PROTEIN_ALTERING_FUNCS
    if membership_df is None:
        membership_df = get_nk_geneset_membership()

    case_ids_str = [str(s) for s in case_ids]
    mask = (
        (tidy_vcf_df[VCF_SAMPLE_COL].astype(str).isin(case_ids_str)) &
        (tidy_vcf_df[VCF_GENE_COL].isin(nk_genes)) &
        (tidy_vcf_df[VCF_VAF_COL] < vaf_threshold) &
        (tidy_vcf_df[VCF_FUNC_COL].isin(protein_altering_funcs))
    )
    detail_cols = [VCF_VARIANT_ID_COL, VCF_GENE_COL, VCF_SAMPLE_COL, VCF_VAF_COL,
                   VCF_FUNC_COL, 'CADD_PHRED', 'SIFT', 'POLYPHEN']
    available_cols = [c for c in detail_cols if c in tidy_vcf_df.columns]
    details = tidy_vcf_df.loc[mask, available_cols].copy()

    if len(details) == 0:
        print("WARNING: No qualifying NK cell variants found")
        return pd.DataFrame()

    freq = details.groupby(VCF_VARIANT_ID_COL)[VCF_SAMPLE_COL].nunique().reset_index()
    freq.columns = [VCF_VARIANT_ID_COL, 'variant_frequency']
    details = details.merge(freq, on=VCF_VARIANT_ID_COL, how='left')

    member_only = membership_df[membership_df['is_member']]
    for go_id in membership_df['go_id'].unique():
        term_genes = set(member_only[member_only['go_id'] == go_id]['gene'])
        col_name = f"in_{go_id}"
        details[col_name] = details[VCF_GENE_COL].isin(term_genes)

    details = details.sort_values([VCF_GENE_COL, VCF_VARIANT_ID_COL, VCF_SAMPLE_COL])

    n_variants = details[VCF_VARIANT_ID_COL].nunique()
    n_patients = details[VCF_SAMPLE_COL].nunique()
    print(f"NK variant details: {n_variants} unique variants across {n_patients} patients")
    for gene in nk_genes:
        gene_df = details[details[VCF_GENE_COL] == gene]
        if len(gene_df) > 0:
            print(f"  {gene}: {gene_df[VCF_VARIANT_ID_COL].nunique()} variants, "
                  f"{gene_df[VCF_SAMPLE_COL].nunique()} patients")
    return details


# GWAS validation / FUMA-MAGMA input preparation

def validate_gwas_sumstats(
    gwas_path: str = None
) -> pd.DataFrame:
    """
    Load and validate Lopez-Isac SSc GWAS summary statistics for FUMA compatibility.

    Parameters:
        gwas_path: str, path to summary statistics file (default: from config)

    Returns:
        DataFrame with validated GWAS summary statistics
    """
    if gwas_path is None:
        gwas_path = GWAS_SUMSTATS_PATH

    print(f"Loading GWAS summary statistics: {gwas_path}")
    gwas_df = pd.read_csv(gwas_path, sep=r'\s+', dtype={GWAS_CHR_COL: str})

    required = [GWAS_SNP_COL, GWAS_CHR_COL, GWAS_BP_COL, GWAS_P_COL]
    missing = [c for c in required if c not in gwas_df.columns]
    if missing:
        raise ValueError(f"Missing required GWAS columns: {missing}")

    n_snps = len(gwas_df)
    n_missing_p = gwas_df[GWAS_P_COL].isna().sum()
    p_range = (gwas_df[GWAS_P_COL].min(), gwas_df[GWAS_P_COL].max())
    n_chrs = gwas_df[GWAS_CHR_COL].nunique()
    n_gw_sig = (gwas_df[GWAS_P_COL] < 5e-8).sum()
    n_missing_a2 = (gwas_df[GWAS_A2_COL] == '?').sum() if GWAS_A2_COL in gwas_df.columns else 0

    print(f"  Total SNPs: {n_snps:,}")
    print(f"  Chromosomes: {n_chrs}")
    print(f"  Missing p-values: {n_missing_p}")
    print(f"  P-value range: {p_range[0]:.2e} to {p_range[1]:.4f}")
    print(f"  Genome-wide significant (p<5e-8): {n_gw_sig}")
    if n_missing_a2 > 0:
        print(f"  SNPs with missing A2 ('?'): {n_missing_a2:,} (FUMA tolerates this)")
    print("  FUMA compatibility: PASS")
    return gwas_df


def prepare_fuma_sumstats(
    gwas_df: pd.DataFrame = None,
    output_path: str = None
) -> str:
    """
    Write FUMA-formatted GWAS summary statistics file.

    Parameters:
        gwas_df: DataFrame, validated GWAS data (default: loaded via validate_gwas_sumstats)
        output_path: str, output file path (default: gwas_output_root/fuma_sumstats.txt)

    Returns:
        str, path to written file
    """
    if gwas_df is None:
        gwas_df = validate_gwas_sumstats()
    if output_path is None:
        os.makedirs(gwas_output_root, exist_ok=True)
        output_path = os.path.join(gwas_output_root, 'fuma_ssc_sumstats.txt')

    fuma_cols = [GWAS_SNP_COL, GWAS_CHR_COL, GWAS_BP_COL, GWAS_A1_COL, GWAS_A2_COL,
                 GWAS_P_COL, GWAS_OR_COL]
    available = [c for c in fuma_cols if c in gwas_df.columns]
    output_df = gwas_df[available].copy()

    n_before = len(output_df)
    output_df = output_df.dropna(subset=[GWAS_SNP_COL, GWAS_P_COL])
    n_dropped = n_before - len(output_df)

    output_df.to_csv(output_path, sep='\t', index=False)
    print(f"FUMA summary stats written: {output_path}")
    print(f"  SNPs: {len(output_df):,} (dropped {n_dropped} with missing SNP/P)")
    print(f"  FUMA upload note: set N = {GWAS_N_TOTAL} in the web form")
    return output_path


def prepare_magma_geneset_file(
    nk_genes: list = None,
    go_term_dict: dict = None,
    gseapy_library: str = None,
    output_path: str = None,
    include_full_go_terms: bool = True
) -> str:
    """
    Write custom gene set file in MAGMA format for FUMA upload.

    Parameters:
        nk_genes: list, somatic NK genes (default: NK_SOMATIC_GENES from config)
        go_term_dict: dict, {GO_ID: name} for full GO term sets (default: NK_GOBP_SIGNIFICANT)
        gseapy_library: str, gseapy library for resolving full GO term gene lists (default: from config)
        output_path: str, output file path (default: gwas_output_root/magma_nk_genesets.txt)
        include_full_go_terms: bool, also write full GO term gene sets for comparison (default: True)

    Returns:
        str, path to written file
    """
    if gseapy_library is None:
        gseapy_library = GSEAPY_GOBP_LIBRARY
    if nk_genes is None:
        nk_genes = NK_SOMATIC_GENES
    if go_term_dict is None:
        go_term_dict = NK_GOBP_SIGNIFICANT
    if output_path is None:
        os.makedirs(gwas_output_root, exist_ok=True)
        output_path = os.path.join(gwas_output_root, 'magma_nk_genesets.txt')

    lines = []
    lines.append('NK_SOMATIC_GENES\t' + '\t'.join(nk_genes))

    if include_full_go_terms:
        library = gp.get_library(gseapy_library)
        go_id_to_gseapy = _resolve_gseapy_term_names(go_term_dict, gseapy_library)
        for go_id, short_name in go_term_dict.items():
            gseapy_term = go_id_to_gseapy.get(go_id)
            if gseapy_term is None:
                continue
            term_genes = library[gseapy_term]
            set_name = go_id.replace(':', '_')
            lines.append(f'{set_name}\t' + '\t'.join(term_genes))

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"MAGMA gene set file written: {output_path}")
    print(f"  Gene sets: {len(lines)}")
    for line in lines:
        parts = line.split('\t')
        print(f"    {parts[0]}: {len(parts)-1} genes")
    return output_path
