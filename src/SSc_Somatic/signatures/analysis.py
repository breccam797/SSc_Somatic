"""
Analysis functions for SSc mutational signature analysis.
Functions organized by notebook workflow in chronological order of use.
"""

import re
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency
from statsmodels.stats.multitest import multipletests
from scipy import stats as scipy_stats

from ..settings import cohort_values


# Study settings. These start empty and are filled from the study
# configuration by apply_settings(), which SSc_Somatic.configure() calls.
DISEASE_ORDER = None
RACE_ORDER = None
DISEASE_COLORS = None
RACE_COLORS = None
DEFAULT_SIGNATURE_FAMILIES = None


# Cohort labels and metadata column names, also filled by apply_settings()
AGE_COL = None
CONTROL_KEY = None
CONTROL_LABEL = None
DLCO_PCT_COL = None
DURATION_COL = None
ETHNICITY_COL = None
FVC_PCT_COL = None
GROUP_COL = None
RACE_COL = None
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
    global DISEASE_ORDER, RACE_ORDER, DISEASE_COLORS, RACE_COLORS, DEFAULT_SIGNATURE_FAMILIES
    DISEASE_ORDER = list(config.require('cohort', 'group_order'))
    RACE_ORDER = list(config.require('cohort', 'race_order'))
    DISEASE_COLORS = dict(config.require('cohort', 'group_colors'))
    RACE_COLORS = dict(config.require('cohort', 'race_colors'))
    DEFAULT_SIGNATURE_FAMILIES = {
        family: list(signatures)
        for family, signatures in config.require('signatures', 'families').items()
    }
    globals().update(cohort_values(config))


def assign_race_ethnicity(row, race_col=None, ethnicity_col=None):
    """
    Assign race/ethnicity category based on race and ethnicity columns.
    
    Parameters:
        row: Series, row from DataFrame
        race_col: str, column name for race (default: 'race')
        ethnicity_col: str, column name for ethnicity (default: 'Ethnicity')
    
    Returns:
        str, race/ethnicity category
    """
    if race_col is None:
        race_col = RACE_COL
    if ethnicity_col is None:
        ethnicity_col = ETHNICITY_COL
    race = str(row.get(race_col, '')).strip() if pd.notna(row.get(race_col)) else ''
    ethnicity = str(row.get(ethnicity_col, '')).strip() if pd.notna(row.get(ethnicity_col)) else ''
    race_lower = race.lower()
    eth_lower = ethnicity.lower()
    if race_lower == 'black':
        return 'Black'
    elif race_lower == 'white':
        if 'non' in eth_lower:
            return 'White Non-Hispanic'
        elif any(term in eth_lower for term in ['hisp', 'latin']):
            return 'White Hispanic'
        else:
            return 'White Non-Hispanic'
    else:
        return 'Other'


def create_race_eth_column(df, race_col=None, ethnicity_col=None):
    """
    Add race_eth column to DataFrame.
    
    Parameters:
        df: DataFrame with race and ethnicity columns
        race_col: str, column name for race (default: 'race')
        ethnicity_col: str, column name for ethnicity (default: 'Ethnicity')
    
    Returns:
        DataFrame with race_eth column added
    """
    if race_col is None:
        race_col = RACE_COL
    if ethnicity_col is None:
        ethnicity_col = ETHNICITY_COL
    df = df.copy()
    df['race_eth'] = df.apply(
        lambda row: assign_race_ethnicity(row, race_col=race_col, ethnicity_col=ethnicity_col),
        axis=1
    )
    return df


def get_available_signatures(df):
    """
    Auto-detect signature columns in DataFrame.
    
    Parameters:
        df: DataFrame to scan for signature columns
    
    Returns:
        Dict with keys 'sbs', 'id', 'dbs', 'other' containing lists of column names
    """
    found = {'sbs': [], 'id': [], 'dbs': [], 'other': []}
    skip_cols = ['sample_id', 'Sample_ID', 'SampleID', 'Sample']
    for col in df.columns:
        if col in skip_cols:
            continue
        if re.match(r'^SBS\d+', col, re.IGNORECASE):
            found['sbs'].append(col)
        elif re.match(r'^ID\d+', col, re.IGNORECASE):
            found['id'].append(col)
        elif re.match(r'^DBS\d+', col, re.IGNORECASE):
            found['dbs'].append(col)
    return found


def compute_variant_totals_by_group(mat_df, sample_ids):
    """
    Compute total variant counts per sample for samples in specified group.
    
    Parameters:
        mat_df: DataFrame, mutation matrix with MutationType column and sample columns
        sample_ids: list, sample IDs to include
    
    Returns:
        ndarray, total variant counts per sample
    """
    available_cols = [col for col in mat_df.columns[1:] if col in sample_ids]
    if len(available_cols) == 0:
        available_cols = [col for col in mat_df.columns[1:] if any(sid in col for sid in sample_ids)]
    if len(available_cols) > 0:
        return mat_df[available_cols].sum(axis=0).values
    return np.array([])


def compute_variant_stats_by_group(mat_sbs_df, mat_id_df, normal_ids, limited_ids, diffuse_ids):
    """
    Compute variant count statistics by sample group and mutation type.
    
    Parameters:
        mat_sbs_df: DataFrame, SBS mutation matrix
        mat_id_df: DataFrame, indel mutation matrix
        normal_ids: list, control sample IDs
        limited_ids: list, limited SSc sample IDs
        diffuse_ids: list, diffuse SSc sample IDs
    
    Returns:
        Tuple of (variant_data dict, stats_df DataFrame)
    """
    variant_data = {
        'sbs_normal': compute_variant_totals_by_group(mat_sbs_df, normal_ids),
        'sbs_limited': compute_variant_totals_by_group(mat_sbs_df, limited_ids),
        'sbs_diffuse': compute_variant_totals_by_group(mat_sbs_df, diffuse_ids),
        'indel_normal': compute_variant_totals_by_group(mat_id_df, normal_ids),
        'indel_limited': compute_variant_totals_by_group(mat_id_df, limited_ids),
        'indel_diffuse': compute_variant_totals_by_group(mat_id_df, diffuse_ids)
    }
    def calc_stats(data, group_name, mutation_type):
        if len(data) == 0:
            return {
                'Group': group_name, 'Mutation_Type': mutation_type, 'Count': 0,
                'Mean': np.nan, 'Median': np.nan, 'Std_Dev': np.nan,
                'Min': np.nan, 'Max': np.nan, 'Q1': np.nan, 'Q3': np.nan
            }
        return {
            'Group': group_name, 'Mutation_Type': mutation_type, 'Count': len(data),
            'Mean': np.mean(data), 'Median': np.median(data),
            'Std_Dev': np.std(data, ddof=1) if len(data) > 1 else 0,
            'Min': np.min(data), 'Max': np.max(data),
            'Q1': np.percentile(data, 25), 'Q3': np.percentile(data, 75)
        }
    all_stats = [
        calc_stats(variant_data['sbs_normal'], CONTROL_LABEL, 'SBS'),
        calc_stats(variant_data['sbs_limited'], SUBTYPE_A, 'SBS'),
        calc_stats(variant_data['sbs_diffuse'], SUBTYPE_B, 'SBS'),
        calc_stats(variant_data['indel_normal'], CONTROL_LABEL, 'Indel'),
        calc_stats(variant_data['indel_limited'], SUBTYPE_A, 'Indel'),
        calc_stats(variant_data['indel_diffuse'], SUBTYPE_B, 'Indel')
    ]
    stats_df = pd.DataFrame(all_stats)
    numerical_cols = ['Mean', 'Median', 'Std_Dev', 'Min', 'Max', 'Q1', 'Q3']
    stats_df[numerical_cols] = stats_df[numerical_cols].round(2)
    return variant_data, stats_df


def compute_zscore_mutation_data(mutation_dfs, meta_df, sample_id_from_column=False,
                                  sample_id_delimiter='_', sample_id_position=0,
                                  age_min=20, age_max=79):
    """
    Compute z-scored mutation data with metadata annotations.
    
    Parameters:
        mutation_dfs: dict, mutation type names to DataFrames
        meta_df: DataFrame, metadata with sample_id, age_group, statustype_subtype
        sample_id_from_column: bool, extract sample_id from column names (default: False)
        sample_id_delimiter: str, delimiter for splitting column names (default: '_')
        sample_id_position: int, position of sample_id after split (default: 0)
        age_min: int, minimum age to include (default: 20)
        age_max: int, maximum age to include (default: 79)
    
    Returns:
        Tuple of (mutation_data dict, stats_df DataFrame)
    """
    if not mutation_dfs:
        raise ValueError("mutation_dfs dictionary cannot be empty")
    clean_meta = meta_df.dropna(subset=['age_group', GROUP_COL, AGE_COL]).copy()
    clean_meta = clean_meta[(clean_meta[AGE_COL] >= age_min) & (clean_meta[AGE_COL] <= age_max)]
    def get_totals_with_metadata(df, meta_subset):
        sample_totals = df.iloc[:, 1:].sum(axis=0)
        if sample_id_from_column:
            clean_ids = []
            for col in sample_totals.index:
                parts = col.split(sample_id_delimiter)
                clean_ids.append(parts[sample_id_position] if len(parts) > sample_id_position else col)
        else:
            clean_ids = sample_totals.index.tolist()
        totals_df = pd.DataFrame({'sample_id': clean_ids, 'total_count': sample_totals.values})
        merged_df = pd.merge(totals_df, meta_subset[['sample_id', 'age_group', GROUP_COL]],
                            on='sample_id', how='inner')
        return merged_df
    def calculate_zscores(data_df):
        mean_val = data_df['total_count'].mean()
        std_val = data_df['total_count'].std(ddof=1)
        data_df = data_df.copy()
        data_df['zscore'] = (data_df['total_count'] - mean_val) / std_val
        return data_df, mean_val, std_val
    mutation_data = {}
    for mut_type, mut_df in mutation_dfs.items():
        data = get_totals_with_metadata(mut_df, clean_meta)
        if len(data) == 0:
            continue
        zscore_data, mean_val, std_val = calculate_zscores(data)
        mutation_data[mut_type] = {'data': data, 'zscore': zscore_data, 'mean': mean_val, 'std': std_val}
    if len(mutation_data) == 0:
        raise ValueError("No mutation data available after merging")
    subtypes = sorted(clean_meta[GROUP_COL].dropna().unique())
    age_groups = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
    available_age_groups = [ag for ag in age_groups if ag in clean_meta['age_group'].values]
    stats_list = []
    for mut_type, mut_info in mutation_data.items():
        zscore_df = mut_info['zscore']
        for subtype in subtypes:
            all_subset = zscore_df[zscore_df[GROUP_COL] == subtype]
            stats_list.append({
                'Mutation_Type': mut_type, 'Subtype': subtype, 'Age_Group': 'All',
                'Count': len(all_subset),
                'Mean_ZScore': all_subset['zscore'].mean() if len(all_subset) > 0 else np.nan,
                'Median_ZScore': all_subset['zscore'].median() if len(all_subset) > 0 else np.nan,
                'Std_ZScore': all_subset['zscore'].std(ddof=1) if len(all_subset) > 1 else np.nan,
                'Mean_Raw_Count': all_subset['total_count'].mean() if len(all_subset) > 0 else np.nan,
                'Median_Raw_Count': all_subset['total_count'].median() if len(all_subset) > 0 else np.nan
            })
            for age_group in available_age_groups:
                subset = zscore_df[(zscore_df[GROUP_COL] == subtype) & (zscore_df['age_group'] == age_group)]
                stats_list.append({
                    'Mutation_Type': mut_type, 'Subtype': subtype, 'Age_Group': age_group,
                    'Count': len(subset),
                    'Mean_ZScore': subset['zscore'].mean() if len(subset) > 0 else np.nan,
                    'Median_ZScore': subset['zscore'].median() if len(subset) > 0 else np.nan,
                    'Std_ZScore': subset['zscore'].std(ddof=1) if len(subset) > 1 else np.nan,
                    'Mean_Raw_Count': subset['total_count'].mean() if len(subset) > 0 else np.nan,
                    'Median_Raw_Count': subset['total_count'].median() if len(subset) > 0 else np.nan
                })
    stats_df = pd.DataFrame(stats_list)
    numerical_cols = ['Mean_ZScore', 'Median_ZScore', 'Std_ZScore', 'Mean_Raw_Count', 'Median_Raw_Count']
    stats_df[numerical_cols] = stats_df[numerical_cols].round(3)
    mutation_data['_subtypes'] = subtypes
    mutation_data['_age_groups'] = available_age_groups
    return mutation_data, stats_df


def compute_subtype_comparison_stats(mutation_data, age_groups=None):
    """
    Compute statistical comparisons between subtypes and age groups.
    
    Parameters:
        mutation_data: dict, output from compute_zscore_mutation_data
        age_groups: list, age groups to compare (default: uses groups from mutation_data)
    
    Returns:
        Dict with 'kruskal_results', 'age_vs_all_results', 'subtype_comparison_results'
    """
    subtypes = mutation_data.get('_subtypes', [])
    if age_groups is None:
        age_groups = mutation_data.get('_age_groups', ['20-29', '30-39', '40-49'])
    subtype_display = {}
    for subtype in subtypes:
        subtype_lower = str(subtype).lower()
        if CONTROL_KEY in subtype_lower or subtype_lower == 'hc':
            subtype_display[subtype] = 'HC'
        elif SUBTYPE_A_KEY in subtype_lower:
            subtype_display[subtype] = SUBTYPE_A
        elif SUBTYPE_B_KEY in subtype_lower:
            subtype_display[subtype] = SUBTYPE_B
        else:
            subtype_display[subtype] = str(subtype)
    results = {'kruskal_results': [], 'age_vs_all_results': [], 'subtype_comparison_results': [], 'subtype_display_map': subtype_display}
    mutation_types = [k for k in mutation_data.keys() if not k.startswith('_')]
    for mut_type in mutation_types:
        mut_info = mutation_data[mut_type]
        data_df = mut_info['data']
        for subtype in subtypes:
            subtype_subset = data_df[data_df[GROUP_COL] == subtype]
            group_data, group_names = [], []
            if len(subtype_subset) > 0:
                group_data.append(subtype_subset['total_count'].values)
                group_names.append('All')
            for age_group in age_groups:
                age_subset = subtype_subset[subtype_subset['age_group'] == age_group]
                if len(age_subset) > 0:
                    group_data.append(age_subset['total_count'].values)
                    group_names.append(age_group)
            if len(group_data) >= 2 and all(len(d) > 0 for d in group_data):
                h_stat, p_value = kruskal(*group_data)
                medians = {name: np.median(d) for name, d in zip(group_names, group_data)}
                results['kruskal_results'].append({'mutation_type': mut_type, 'subtype': subtype_display[subtype], 'h_stat': h_stat, 'p_value': p_value, 'medians': medians})
        for subtype in subtypes:
            subtype_subset = data_df[data_df[GROUP_COL] == subtype]
            if len(subtype_subset) == 0:
                continue
            all_ages_data = subtype_subset['total_count'].values
            for age_group in age_groups:
                age_subset = subtype_subset[subtype_subset['age_group'] == age_group]
                if len(age_subset) == 0:
                    continue
                age_data = age_subset['total_count'].values
                try:
                    u_stat, p_value = mannwhitneyu(age_data, all_ages_data, alternative='two-sided')
                    results['age_vs_all_results'].append({'mutation_type': mut_type, 'subtype': subtype_display[subtype], 'age_group': age_group, 'u_stat': u_stat, 'p_value': p_value, 'age_median': np.median(age_data), 'all_median': np.median(all_ages_data), 'n_age': len(age_data), 'n_all': len(all_ages_data)})
                except Exception:
                    pass
        seen_pairs = set()
        for test_subtype in subtypes:
            for ref_subtype in subtypes:
                if test_subtype == ref_subtype:
                    continue
                pair = tuple(sorted([test_subtype, ref_subtype]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                test_data = data_df[data_df[GROUP_COL] == test_subtype]['total_count'].values
                ref_data = data_df[data_df[GROUP_COL] == ref_subtype]['total_count'].values
                if len(test_data) > 0 and len(ref_data) > 0:
                    try:
                        u_stat, p_value = mannwhitneyu(test_data, ref_data, alternative='two-sided')
                        results['subtype_comparison_results'].append({'mutation_type': mut_type, 'comparison': f"{subtype_display[test_subtype]} vs {subtype_display[ref_subtype]}", 'age_group': 'All', 'u_stat': u_stat, 'p_value': p_value, 'test_median': np.median(test_data), 'ref_median': np.median(ref_data), 'n_test': len(test_data), 'n_ref': len(ref_data)})
                    except Exception:
                        pass
                for age_group in age_groups:
                    test_subset = data_df[(data_df[GROUP_COL] == test_subtype) & (data_df['age_group'] == age_group)]
                    ref_subset = data_df[(data_df[GROUP_COL] == ref_subtype) & (data_df['age_group'] == age_group)]
                    test_age_data, ref_age_data = test_subset['total_count'].values, ref_subset['total_count'].values
                    if len(test_age_data) > 0 and len(ref_age_data) > 0:
                        try:
                            u_stat, p_value = mannwhitneyu(test_age_data, ref_age_data, alternative='two-sided')
                            results['subtype_comparison_results'].append({'mutation_type': mut_type, 'comparison': f"{subtype_display[test_subtype]} vs {subtype_display[ref_subtype]}", 'age_group': age_group, 'u_stat': u_stat, 'p_value': p_value, 'test_median': np.median(test_age_data), 'ref_median': np.median(ref_age_data), 'n_test': len(test_age_data), 'n_ref': len(ref_age_data)})
                        except Exception:
                            pass
    return results


def compute_signature_stats(df, signatures_list):
    """
    Compute statistics for signature activity distributions by subtype.
    
    Parameters:
        df: DataFrame, annotated signature data with statustype_subtype column
        signatures_list: list, signature column names to analyze
    
    Returns:
        Dict with signature names as keys, containing counts and medians per subtype
    """
    sig_stats = {}
    subtypes = df[GROUP_COL].dropna().unique()
    for sig in signatures_list:
        sig_stats[sig] = {'counts': df[GROUP_COL].value_counts().to_dict(), 'medians': {}, 'means': {}, 'n_per_subtype': {}}
        for subtype in subtypes:
            subset = df[df[GROUP_COL] == subtype]
            sig_stats[sig]['medians'][subtype] = subset[sig].median()
            sig_stats[sig]['means'][subtype] = subset[sig].mean()
            sig_stats[sig]['n_per_subtype'][subtype] = len(subset)
    return sig_stats


def stratified_mannwhitney(df, signature_col, group1, group2, strata_col='race_eth', group_col=None):
    """
    Run Mann-Whitney U test comparing two groups within each stratum.
    
    Parameters:
        df: DataFrame, prepared analysis data
        signature_col: str, signature column to test
        group1: str or list, first group (or list to pool)
        group2: str or list, second group (reference)
        strata_col: str, stratification column (default: 'race_eth')
        group_col: str, column containing disease groups (default: 'statustype_subtype')
    
    Returns:
        DataFrame with results per stratum
    """
    if group_col is None:
        group_col = GROUP_COL
    results = []
    if isinstance(group1, str):
        group1 = [group1]
    if isinstance(group2, str):
        group2 = [group2]
    g1_label = '+'.join(group1) if len(group1) > 1 else group1[0]
    g2_label = '+'.join(group2) if len(group2) > 1 else group2[0]
    comparison = f'{g1_label} vs {g2_label}'
    strata = [s for s in RACE_ORDER if s in df[strata_col].unique()]
    for stratum in strata:
        stratum_df = df[df[strata_col] == stratum].dropna(subset=[signature_col])
        data1 = stratum_df[stratum_df[group_col].isin(group1)][signature_col].values
        data2 = stratum_df[stratum_df[group_col].isin(group2)][signature_col].values
        if len(data1) < 3 or len(data2) < 3:
            results.append({'signature': signature_col, 'stratum': stratum, 'comparison': comparison, 'group1': g1_label, 'group2': g2_label, 'median_group1': np.median(data1) if len(data1) > 0 else np.nan, 'median_group2': np.median(data2) if len(data2) > 0 else np.nan, 'fold_change': np.nan, 'U_statistic': np.nan, 'p_value': np.nan, 'n_group1': len(data1), 'n_group2': len(data2)})
            continue
        med1, med2 = np.median(data1), np.median(data2)
        fc = med1 / med2 if med2 > 0 else np.nan
        try:
            U, p = mannwhitneyu(data1, data2, alternative='two-sided')
        except Exception:
            U, p = np.nan, np.nan
        results.append({'signature': signature_col, 'stratum': stratum, 'comparison': comparison, 'group1': g1_label, 'group2': g2_label, 'median_group1': med1, 'median_group2': med2, 'fold_change': fc, 'U_statistic': U, 'p_value': p, 'n_group1': len(data1), 'n_group2': len(data2)})
    return pd.DataFrame(results)


def stratified_kruskal_wallis(df, signature_col, strata_col='race_eth', group_col=None):
    """
    Perform Kruskal-Wallis test within each stratum (3-group comparison).
    
    Parameters:
        df: DataFrame, prepared analysis data
        signature_col: str, signature column to test
        strata_col: str, stratification column (default: 'race_eth')
        group_col: str, column containing disease groups (default: 'statustype_subtype')
    
    Returns:
        DataFrame with results per stratum
    """
    if group_col is None:
        group_col = GROUP_COL
    results = []
    strata = [s for s in RACE_ORDER if s in df[strata_col].unique()]
    for stratum in strata:
        stratum_df = df[df[strata_col] == stratum].dropna(subset=[signature_col])
        groups_data = {}
        for group in DISEASE_ORDER:
            gdata = stratum_df[stratum_df[group_col] == group][signature_col].values
            if len(gdata) > 0:
                groups_data[group] = gdata
        if len(groups_data) < 2:
            results.append({'stratum': stratum, 'signature': signature_col, 'H_statistic': np.nan, 'p_value': np.nan, 'n_per_group': {g: len(d) for g, d in groups_data.items()}, 'medians': {g: np.median(d) for g, d in groups_data.items()}})
            continue
        try:
            H, p = kruskal(*groups_data.values())
        except Exception:
            H, p = np.nan, np.nan
        results.append({'stratum': stratum, 'signature': signature_col, 'H_statistic': H, 'p_value': p, 'n_per_group': {g: len(d) for g, d in groups_data.items()}, 'medians': {g: np.median(d) for g, d in groups_data.items()}})
    return pd.DataFrame(results)


def prepare_stratified_analysis_data(meta_df, signature_dfs, exclude_other=True, disease_duration_max=None, disease_duration_min=None, age_min=None, age_max=None, race_col=None, ethnicity_col=None, verbose=True):
    """
    Prepare merged dataset for stratified analysis.
    
    Parameters:
        meta_df: DataFrame, metadata with sample_id, race, Ethnicity, statustype_subtype
        signature_dfs: dict, signature DataFrames keyed by name
        exclude_other: bool, exclude race_eth='Other' samples (default: True)
        disease_duration_max: float, max disease duration in years (default: None)
        disease_duration_min: float, min disease duration in years (default: None)
        age_min: float, minimum age (default: None)
        age_max: float, maximum age (default: None)
        race_col: str, column name for race (default: 'race')
        ethnicity_col: str, column name for ethnicity (default: 'Ethnicity')
        verbose: bool, print progress info (default: True)
    
    Returns:
        DataFrame ready for analysis
    """
    if race_col is None:
        race_col = RACE_COL
    if ethnicity_col is None:
        ethnicity_col = ETHNICITY_COL
    df = meta_df.copy()
    if verbose:
        print(f"Input {race_col} distribution:")
        print(df[race_col].value_counts())
    df['race_eth'] = df.apply(lambda row: assign_race_ethnicity(row, race_col=race_col, ethnicity_col=ethnicity_col), axis=1)
    if verbose:
        print(f"\nAssigned race_eth distribution:")
        print(df['race_eth'].value_counts())
    for name, sig_df in signature_dfs.items():
        if sig_df is None:
            continue
        sample_col = None
        for possible in ['sample_id', 'Sample_ID', 'SampleID', 'Sample']:
            if possible in sig_df.columns:
                sample_col = possible
                break
        if sample_col is None:
            print(f"Warning: No sample_id column found in {name}, skipping")
            continue
        sig_copy = sig_df.copy()
        if sample_col != 'sample_id':
            sig_copy = sig_copy.rename(columns={sample_col: 'sample_id'})
        sig_cols = [c for c in sig_copy.columns if c != 'sample_id']
        df = pd.merge(df, sig_copy[['sample_id'] + sig_cols], on='sample_id', how='left')
    valid_subtypes = [CONTROL_LABEL, SUBTYPE_A, SUBTYPE_B]
    n_before = len(df)
    df = df[df[GROUP_COL].isin(valid_subtypes)].copy()
    if verbose:
        print(f"\nFiltered to valid subtypes: {n_before} -> {len(df)}")
    if exclude_other:
        n_before = len(df)
        df = df[df['race_eth'] != 'Other'].copy()
        if verbose:
            print(f"Excluded 'Other' race_eth: {n_before} -> {len(df)}")
    if disease_duration_max is not None or disease_duration_min is not None:
        dur_col = None
        for possible in [DURATION_COL, 'disease_duration', 'DiseaseDuration']:
            if possible in df.columns:
                dur_col = possible
                break
        if dur_col:
            n_before = len(df)
            is_control = df[GROUP_COL] == CONTROL_LABEL
            has_dur = df[dur_col].notna()
            if disease_duration_max is not None and disease_duration_min is not None:
                dur_filter = is_control | (has_dur & (df[dur_col] >= disease_duration_min) & (df[dur_col] <= disease_duration_max))
            elif disease_duration_max is not None:
                dur_filter = is_control | (has_dur & (df[dur_col] <= disease_duration_max))
            else:
                dur_filter = is_control | (has_dur & (df[dur_col] >= disease_duration_min))
            df = df[dur_filter].copy()
            if verbose:
                print(f"Disease duration filter: {n_before} -> {len(df)}")
    if age_min is not None or age_max is not None:
        age_col = None
        for possible in [AGE_COL, 'age', 'AGE']:
            if possible in df.columns:
                age_col = possible
                break
        if age_col:
            n_before = len(df)
            if age_min is not None and age_max is not None:
                df = df[(df[age_col] >= age_min) & (df[age_col] <= age_max)].copy()
            elif age_min is not None:
                df = df[df[age_col] >= age_min].copy()
            else:
                df = df[df[age_col] <= age_max].copy()
            if verbose:
                print(f"Age filter: {n_before} -> {len(df)}")
    if verbose:
        print("\nFinal sample sizes (race_eth x disease_subtype):")
        crosstab = pd.crosstab(df['race_eth'], df[GROUP_COL], margins=True)
        col_order = [c for c in DISEASE_ORDER if c in crosstab.columns] + ['All']
        row_order = [r for r in RACE_ORDER if r in crosstab.index] + ['All']
        crosstab = crosstab.reindex(index=row_order, columns=col_order)
        print(crosstab)
    return df


def run_all_disease_comparisons(df, signatures, strata_col='race_eth', signature_families=None):
    """
    Run all standard disease comparisons for a set of signatures.
    
    Parameters:
        df: DataFrame, prepared analysis data
        signatures: list, signature columns to test
        strata_col: str, stratification column (default: 'race_eth')
        signature_families: dict, mapping for FDR correction grouping (default: uses DEFAULT_SIGNATURE_FAMILIES)
    
    Returns:
        Dict with comparison results DataFrames, FDR-corrected within signature families
    """
    results = {}
    available = [s for s in signatures if s in df.columns]
    
    if not available:
        print("No valid signatures found")
        return results
    
    print(f"Running comparisons for {len(available)} signatures")
    
    dc_results = [stratified_mannwhitney(df, sig, SUBTYPE_B, CONTROL_LABEL) for sig in available]
    results['diffuse_vs_control'] = apply_fdr_correction_by_family(
        pd.concat(dc_results, ignore_index=True), signature_families=signature_families
    )
    
    lc_results = [stratified_mannwhitney(df, sig, SUBTYPE_A, CONTROL_LABEL) for sig in available]
    results['limited_vs_control'] = apply_fdr_correction_by_family(
        pd.concat(lc_results, ignore_index=True), signature_families=signature_families
    )
    
    dl_results = [stratified_mannwhitney(df, sig, SUBTYPE_B, SUBTYPE_A) for sig in available]
    results['diffuse_vs_limited'] = apply_fdr_correction_by_family(
        pd.concat(dl_results, ignore_index=True), signature_families=signature_families
    )
    
    pooled_results = [stratified_mannwhitney(df, sig, [SUBTYPE_A, SUBTYPE_B], CONTROL_LABEL) for sig in available]
    results['pooled_ssc_vs_control'] = apply_fdr_correction_by_family(
        pd.concat(pooled_results, ignore_index=True), signature_families=signature_families
    )
    
    kw_results = [stratified_kruskal_wallis(df, sig) for sig in available]
    results['kruskal_wallis'] = apply_fdr_correction_by_family(
        pd.concat(kw_results, ignore_index=True), signature_families=signature_families
    )
    
    return results


def run_baseline_disease_analysis(meta_df, signature_dfs, signatures=None, duration_cutoff=3.0,
                                   exclude_other=True, signature_families=None, verbose=True):
    """
    Analyze baseline SSc patients (disease duration <= cutoff).
    
    Parameters:
        meta_df: DataFrame, metadata
        signature_dfs: dict, signature DataFrames
        signatures: list, signatures to analyze (auto-detected if None)
        duration_cutoff: float, max disease duration in years (default: 3.0)
        exclude_other: bool, exclude race_eth='Other' (default: True)
        signature_families: dict, mapping for FDR correction grouping (default: uses DEFAULT_SIGNATURE_FAMILIES)
        verbose: bool, print progress (default: True)
    
    Returns:
        Dict with analysis results
    """
    if verbose:
        print(f"\nBaseline SSc analysis (disease duration <= {duration_cutoff} years)")
    
    df = prepare_stratified_analysis_data(
        meta_df, signature_dfs, exclude_other=exclude_other,
        disease_duration_max=duration_cutoff, verbose=verbose
    )
    
    if signatures is None:
        detected = get_available_signatures(df)
        signatures = detected['sbs'] + detected['id']
    
    available = [s for s in signatures if s in df.columns]
    
    if verbose:
        print(f"\nAnalyzing {len(available)} signatures")
    
    results = {
        'data': df,
        'config': {'duration_cutoff': duration_cutoff, 'signatures': available}
    }
    
    comparisons = run_all_disease_comparisons(df, available, signature_families=signature_families)
    results.update(comparisons)
    
    if verbose:
        print_significant_findings(results, f"Baseline SSc (<= {duration_cutoff}yr)")
    
    return results


def _compare_young_vs_older(df_young, df_older, signatures, age_cutoff):
    """Compare young vs older SSc patients within each stratum."""
    results = []
    strata = [s for s in RACE_ORDER if s in df_young['race_eth'].unique()]
    for sig in signatures:
        if sig not in df_young.columns or sig not in df_older.columns:
            continue
        for stratum in strata:
            young_ssc = df_young[(df_young['race_eth'] == stratum) & (df_young[GROUP_COL].isin([SUBTYPE_A, SUBTYPE_B]))].dropna(subset=[sig])[sig].values
            older_ssc = df_older[(df_older['race_eth'] == stratum) & (df_older[GROUP_COL].isin([SUBTYPE_A, SUBTYPE_B]))].dropna(subset=[sig])[sig].values
            if len(young_ssc) < 3 or len(older_ssc) < 3:
                results.append({'signature': sig, 'stratum': stratum, 'comparison': f'Young (<{age_cutoff}) vs Older SSc', 'median_young': np.nan, 'median_older': np.nan, 'fold_change': np.nan, 'p_value': np.nan, 'n_young': len(young_ssc), 'n_older': len(older_ssc)})
                continue
            med_young, med_older = np.median(young_ssc), np.median(older_ssc)
            fc = med_young / med_older if med_older > 0 else np.nan
            try:
                _, p = mannwhitneyu(young_ssc, older_ssc, alternative='two-sided')
            except Exception:
                p = np.nan
            results.append({'signature': sig, 'stratum': stratum, 'comparison': f'Young (<{age_cutoff}) vs Older SSc', 'median_young': med_young, 'median_older': med_older, 'fold_change': fc, 'p_value': p, 'n_young': len(young_ssc), 'n_older': len(older_ssc)})
    return pd.DataFrame(results)


def run_young_baseline_analysis(meta_df, signature_dfs, signatures=None, age_cutoff=40,
                                 duration_cutoff=3.0, compare_to_older=True, exclude_other=True,
                                 signature_families=None, verbose=True):
    """
    Analyze young baseline SSc patients.
    
    Parameters:
        meta_df: DataFrame, metadata
        signature_dfs: dict, signature DataFrames
        signatures: list, signatures to analyze (auto-detected if None)
        age_cutoff: float, age threshold (< cutoff = young) (default: 40)
        duration_cutoff: float, max disease duration (default: 3.0)
        compare_to_older: bool, also compare young vs older (default: True)
        exclude_other: bool, exclude race_eth='Other' (default: True)
        signature_families: dict, mapping for FDR correction grouping (default: uses DEFAULT_SIGNATURE_FAMILIES)
        verbose: bool, print progress (default: True)
    
    Returns:
        Dict with analysis results
    """
    if verbose:
        print(f"\nYoung baseline SSc analysis (age < {age_cutoff}, duration <= {duration_cutoff} years)")
    
    df_young = prepare_stratified_analysis_data(
        meta_df, signature_dfs, exclude_other=exclude_other,
        disease_duration_max=duration_cutoff, age_max=age_cutoff - 0.001, verbose=verbose
    )
    
    if signatures is None:
        detected = get_available_signatures(df_young)
        signatures = detected['sbs'] + detected['id']
    
    available = [s for s in signatures if s in df_young.columns]
    
    if verbose:
        print(f"\nAnalyzing {len(available)} signatures in young patients")
    
    results = {
        'data_young': df_young,
        'config': {
            'age_cutoff': age_cutoff,
            'duration_cutoff': duration_cutoff,
            'signatures': available,
            'n_young': len(df_young)
        }
    }
    
    young_pooled = [stratified_mannwhitney(df_young, sig, [SUBTYPE_A, SUBTYPE_B], CONTROL_LABEL) for sig in available]
    results['young_pooled_vs_control'] = apply_fdr_correction_by_family(
        pd.concat(young_pooled, ignore_index=True), signature_families=signature_families
    )
    
    young_dl = [stratified_mannwhitney(df_young, sig, SUBTYPE_B, SUBTYPE_A) for sig in available]
    results['young_diffuse_vs_limited'] = apply_fdr_correction_by_family(
        pd.concat(young_dl, ignore_index=True), signature_families=signature_families
    )
    
    young_dc = [stratified_mannwhitney(df_young, sig, SUBTYPE_B, CONTROL_LABEL) for sig in available]
    results['young_diffuse_vs_control'] = apply_fdr_correction_by_family(
        pd.concat(young_dc, ignore_index=True), signature_families=signature_families
    )
    
    young_lc = [stratified_mannwhitney(df_young, sig, SUBTYPE_A, CONTROL_LABEL) for sig in available]
    results['young_limited_vs_control'] = apply_fdr_correction_by_family(
        pd.concat(young_lc, ignore_index=True), signature_families=signature_families
    )
    
    if compare_to_older:
        if verbose:
            print("\nComparing young vs older baseline SSc...")
        
        df_older = prepare_stratified_analysis_data(
            meta_df, signature_dfs, exclude_other=exclude_other,
            disease_duration_max=duration_cutoff, age_min=age_cutoff, verbose=verbose
        )
        
        results['data_older'] = df_older
        results['config']['n_older'] = len(df_older)
        
        young_vs_older = _compare_young_vs_older(df_young, df_older, available, age_cutoff)
        results['young_vs_older_ssc'] = apply_fdr_correction_by_family(
            young_vs_older, signature_families=signature_families
        )
    
    if verbose:
        print_significant_findings(results, f"Young baseline SSc (<{age_cutoff}yr)")
    
    return results


def run_comprehensive_disease_analysis(meta_df, signature_dfs, signatures=None, baseline_duration=3.0,
                                        early_duration=2.0, young_age=40, exclude_other=True,
                                        signature_families=None, verbose=True):
    """
    Run comprehensive SSc mutational signature analysis.
    
    Parameters:
        meta_df: DataFrame, metadata
        signature_dfs: dict, signature DataFrames
        signatures: list, signatures to analyze (auto-detected if None)
        baseline_duration: float, baseline disease duration cutoff (default: 3.0)
        early_duration: float, early disease duration cutoff (default: 2.0)
        young_age: float, young age cutoff (default: 40)
        exclude_other: bool, exclude race_eth='Other' (default: True)
        signature_families: dict, mapping for FDR correction grouping (default: uses DEFAULT_SIGNATURE_FAMILIES)
        verbose: bool, print progress (default: True)
    
    Returns:
        Dict with all analysis results organized by question, including 'summary'
    """
    if verbose:
        print("Comprehensive SSc mutational signature analysis")
        print(f"Parameters: baseline <= {baseline_duration}yr, early <= {early_duration}yr, young < {young_age}yr")
        print("FDR correction applied within signature families (SBS, ID separately)\n")
    
    all_results = {}
    
    if verbose:
        print("Q1: Full cohort analysis")
    
    df_full = prepare_stratified_analysis_data(
        meta_df, signature_dfs, exclude_other=exclude_other, verbose=verbose
    )
    
    if signatures is None:
        detected = get_available_signatures(df_full)
        signatures = detected['sbs'] + detected['id']
    
    available = [s for s in signatures if s in df_full.columns]
    
    if verbose:
        print(f"\nAnalyzing {len(available)} signatures: {available}")
    
    all_results['q1_full_cohort'] = {
        'data': df_full,
        'config': {'signatures': available},
        **run_all_disease_comparisons(df_full, available, signature_families=signature_families)
    }
    
    all_results['q2_baseline'] = run_baseline_disease_analysis(
        meta_df, signature_dfs, signatures=available,
        duration_cutoff=baseline_duration, exclude_other=exclude_other,
        signature_families=signature_families, verbose=verbose
    )
    
    if early_duration != baseline_duration:
        all_results['q2_early_disease'] = run_baseline_disease_analysis(
            meta_df, signature_dfs, signatures=available,
            duration_cutoff=early_duration, exclude_other=exclude_other,
            signature_families=signature_families, verbose=verbose
        )
    
    all_results['q3_young_baseline'] = run_young_baseline_analysis(
        meta_df, signature_dfs, signatures=available,
        age_cutoff=young_age, duration_cutoff=baseline_duration,
        compare_to_older=True, exclude_other=exclude_other,
        signature_families=signature_families, verbose=verbose
    )
    
    if verbose:
        summary = summarize_comprehensive_analysis_findings(all_results)
        all_results['summary'] = summary
    
    return all_results

def print_significant_findings(results, title):
    """Print summary of significant findings from analysis results."""
    print(f"\nSignificant findings: {title}")
    comparisons = [('pooled_ssc_vs_control', f'SSc (pooled) vs {CONTROL_LABEL}'), ('diffuse_vs_control', f'{SUBTYPE_B} vs {CONTROL_LABEL}'), ('limited_vs_control', f'{SUBTYPE_A} vs {CONTROL_LABEL}'), ('diffuse_vs_limited', f'{SUBTYPE_B} vs {SUBTYPE_A}'), ('young_pooled_vs_control', f'Young SSc vs {CONTROL_LABEL}'), ('young_diffuse_vs_limited', f'Young {SUBTYPE_B} vs {SUBTYPE_A}'), ('young_vs_older_ssc', 'Young vs Older SSc')]
    any_sig = False
    for key, label in comparisons:
        if key not in results or results[key] is None:
            continue
        df = results[key]
        sig_results = df[df['p_value'] < 0.05]
        if len(sig_results) == 0:
            continue
        any_sig = True
        print(f"\n{label}:")
        for _, row in sig_results.iterrows():
            fc = row.get('fold_change', np.nan)
            direction = 'UP' if fc > 1 else 'DOWN' if fc < 1 else '='
            p_fdr = row.get('p_value_fdr', 1)
            stars = '***' if p_fdr < 0.001 else '**' if p_fdr < 0.01 else '*' if p_fdr < 0.05 else ''
            print(f"  {row['signature']:12} | {row['stratum']:20} | FC={fc:.2f} {direction:4} | p={row['p_value']:.4f}{stars}")
    if not any_sig:
        print("  No significant findings (p < 0.05)")
    print("\n  * = FDR<0.05, ** = FDR<0.01, *** = FDR<0.001")

def compute_signature_clinical_effects(signature_df, meta_df, signature_cols, metadata_cols, percentile=75, age_group_filter=None, filter_col=None, filter_value=None, verbose=True):
    """
    Compute effect sizes between high vs low signature groups for clinical variables.
    
    Parameters:
        signature_df: DataFrame with sample_id and signature columns
        meta_df: DataFrame with sample_id and clinical columns
        signature_cols: list, signature column names
        metadata_cols: list, clinical variable column names
        percentile: float, threshold percentile for high group (default: 75)
        age_group_filter: str, filter on broad_age_group column (default: None)
        filter_col: str, additional filter column (default: None)
        filter_value: str, value to filter on (default: None)
        verbose: bool, print progress (default: True)
    
    Returns:
        Tuple of (effect_df, summary_table)
    """
    if verbose:
        print("Signature clinical effect analysis")
    meta_df = meta_df.copy()
    if RACE_COL in meta_df.columns and ('Black' not in meta_df.columns or meta_df['Black'].dtype == 'object'):
        meta_df['Black'] = meta_df[RACE_COL].apply(lambda x: 1 if x == 'Black' else 0)
    if ETHNICITY_COL in meta_df.columns and (meta_df[ETHNICITY_COL].dtype == 'object' or meta_df[ETHNICITY_COL].dtype.name == 'category'):
        meta_df[ETHNICITY_COL] = meta_df[ETHNICITY_COL].apply(lambda x: 1 if x == 'Hisp/Latin' else 0 if x == 'Non-Hisp' else np.nan)
    for ab_col in ['ana', 'anti-centromere', 'anti-topo', 'anti-RNA polymerase III']:
        if ab_col in meta_df.columns and 2 in meta_df[ab_col].dropna().unique():
            meta_df[ab_col] = meta_df[ab_col].map({1: 1, 2: 0})
    df = pd.merge(signature_df, meta_df, on='sample_id', how='inner')
    if verbose:
        print(f"Merged dataset shape: {df.shape}")
    if age_group_filter is not None and 'broad_age_group' in df.columns:
        initial_n = len(df)
        df = df[df['broad_age_group'] == age_group_filter].copy()
        if verbose:
            print(f"Filtered by age group '{age_group_filter}': {initial_n} -> {len(df)}")
    if filter_col and filter_value and filter_col in df.columns:
        initial_n = len(df)
        df = df[df[filter_col] == filter_value].copy()
        if verbose:
            print(f"Filtered by {filter_col} = '{filter_value}': {initial_n} -> {len(df)}")
    available_metadata = [col for col in metadata_cols if col in df.columns]
    available_signatures = [sig for sig in signature_cols if sig in df.columns]
    if not available_signatures:
        print("Error: No valid signature columns found")
        return pd.DataFrame(), pd.DataFrame()
    results = []
    for sig in available_signatures:
        sig_data = df.dropna(subset=[sig]).copy()
        if len(sig_data) < 10:
            continue
        nonzero_values = sig_data[sig_data[sig] > 0][sig]
        if len(nonzero_values) < 10:
            continue
        threshold = np.percentile(nonzero_values, percentile)
        sig_data['is_top'] = sig_data[sig] >= threshold
        n_top, n_bottom = sig_data['is_top'].sum(), (~sig_data['is_top']).sum()
        if n_top < 5 or n_bottom < 5:
            continue
        for var in available_metadata:
            var_data = sig_data.dropna(subset=[var]).copy()
            if len(var_data) < 10:
                continue
            top_group = var_data[var_data['is_top']][var]
            bottom_group = var_data[~var_data['is_top']][var]
            if len(top_group) < 3 or len(bottom_group) < 3:
                continue
            effect_size, p_value, effect_type = _calculate_single_effect_size(var, var_data, top_group, bottom_group)
            results.append({'signature': sig, 'variable': var, 'effect_size': effect_size, 'p_value': p_value, 'effect_type': effect_type, 'n_top': len(top_group), 'n_bottom': len(bottom_group), 'threshold': threshold})
    if not results:
        return pd.DataFrame(), pd.DataFrame()
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        reject, pvals_corrected, _, _ = multipletests(results_df['p_value'], alpha=0.05, method='fdr_bh')
        results_df['p_value_fdr'] = pvals_corrected
        results_df['significant_fdr'] = reject
    summary_table = create_effect_summary_table(results_df, signature_cols)
    return results_df, summary_table


def _calculate_single_effect_size(var, var_data, top_group, bottom_group):
    """Calculate effect size for a single variable comparison."""
    binary_vars = ['Black', ETHNICITY_COL, 'ana', 'anti-centromere', 'anti-topo', 'anti-RNA polymerase III']
    if pd.api.types.is_numeric_dtype(var_data[var]):
        unique_vals = var_data[var].dropna().unique()
        is_binary_categorical = set(unique_vals).issubset({0, 1, 0.0, 1.0})
        if is_binary_categorical and var in binary_vars:
            n_pos_top, n_pos_bottom = (top_group == 1).sum(), (bottom_group == 1).sum()
            n_neg_top, n_neg_bottom = (top_group == 0).sum(), (bottom_group == 0).sum()
            odds_ratio = (n_pos_top * n_neg_bottom) / (n_pos_bottom * n_neg_top + 1e-8)
            effect_size = np.log(odds_ratio) if odds_ratio > 0 else 0
            try:
                _, p_value, _, _ = chi2_contingency([[n_neg_bottom, n_neg_top], [n_pos_bottom, n_pos_top]])
            except Exception:
                p_value = 1.0
            effect_type = "Log Odds Ratio"
        else:
            top_mean, bottom_mean = top_group.mean(), bottom_group.mean()
            pooled_std = np.sqrt(((len(top_group)-1)*top_group.std()**2 + (len(bottom_group)-1)*bottom_group.std()**2) / (len(top_group) + len(bottom_group) - 2))
            effect_size = (top_mean - bottom_mean) / pooled_std if pooled_std > 0 else 0
            try:
                _, p_value = mannwhitneyu(top_group, bottom_group, alternative='two-sided')
            except Exception:
                p_value = 1.0
            effect_type = "Cohen's d"
    else:
        try:
            contingency_table = pd.crosstab(var_data[var], var_data['is_top'])
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            n = contingency_table.sum().sum()
            if contingency_table.shape == (2, 2):
                a, b, c, d = contingency_table.iloc[0, 1], contingency_table.iloc[0, 0], contingency_table.iloc[1, 1], contingency_table.iloc[1, 0]
                odds_ratio = (a * d) / (b * c + 1e-8)
                effect_size = np.log(odds_ratio) if odds_ratio > 0 else 0
                effect_type = "Log Odds Ratio"
            else:
                effect_size = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                effect_type = "Cramer's V"
        except Exception:
            effect_size, p_value, effect_type = 0, 1.0, "Error"
    return effect_size, p_value, effect_type


def create_effect_summary_table(effect_df, signature_cols=None):
    """
    Create formatted summary table for effect size results.
    
    Parameters:
        effect_df: DataFrame, output from compute_signature_clinical_effects
        signature_cols: list, signature columns to include (default: auto-detect)
    
    Returns:
        DataFrame with formatted summary
    """
    if len(effect_df) == 0:
        return pd.DataFrame()
    if signature_cols is None:
        signature_cols = effect_df['signature'].unique().tolist()
    variable_meanings = {
        AGE_COL: {'positive': 'Older in high signature', 'negative': 'Younger in high signature'},
        DURATION_COL: {'positive': 'Longer duration in high signature', 'negative': 'Shorter duration in high signature'},
        SKIN_SCORE_COL: {'positive': f'Higher {SKIN_SCORE_COL} in high signature', 'negative': f'Lower {SKIN_SCORE_COL} in high signature'},
        DLCO_PCT_COL: {'positive': 'Higher in high signature', 'negative': 'Lower in high signature'},
        FVC_PCT_COL: {'positive': 'Higher in high signature', 'negative': 'Lower in high signature'},
        'Black': {'positive': 'Black patients in high signature', 'negative': 'Non-Black in high signature'},
        ETHNICITY_COL: {'positive': 'Hisp/Latin in high signature', 'negative': 'Non-Hisp in high signature'},
        'ana': {'positive': 'Positive in high signature', 'negative': 'Negative in high signature'},
        'anti-centromere': {'positive': 'Positive in high signature', 'negative': 'Negative in high signature'},
        'anti-topo': {'positive': 'Positive in high signature', 'negative': 'Negative in high signature'},
        'anti-RNA polymerase III': {'positive': 'Positive in high signature', 'negative': 'Negative in high signature'}
    }
    variables = effect_df['variable'].unique()
    summary_rows = []
    for var in variables:
        var_data = effect_df[effect_df['variable'] == var]
        effect_type = var_data['effect_type'].iloc[0] if 'effect_type' in var_data.columns else 'Unknown'
        meanings = variable_meanings.get(var, {'positive': 'Higher in high signature', 'negative': 'Lower in high signature'})
        row = {'Variable': var, 'Effect Size Type': effect_type, 'Positive Means': meanings.get('positive', 'Category 1 enriched'), 'Negative Means': meanings.get('negative', 'Category 0 enriched')}
        for sig in signature_cols:
            sig_row = var_data[var_data['signature'] == sig]
            if len(sig_row) > 0:
                es = sig_row['effect_size'].iloc[0]
                pval_fdr = sig_row['p_value_fdr'].iloc[0] if 'p_value_fdr' in sig_row.columns else sig_row['p_value'].iloc[0]
                stars = '***' if pval_fdr < 0.001 else '**' if pval_fdr < 0.01 else '*' if pval_fdr < 0.05 else ''
                row[sig] = f"{es:+.2f}{stars}"
            else:
                row[sig] = 'N/A'
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def run_clinical_effect_analysis_by_subtype(signature_df, meta_df, signature_cols, metadata_cols, subtypes=None, subtype_col=None, age_group_filter=None, percentile=75, verbose=True):
    """
    Run clinical effect analysis separately for each disease subtype.
    
    Parameters:
        signature_df: DataFrame with sample_id and signature columns
        meta_df: DataFrame with sample_id and clinical columns
        signature_cols: list, signature column names
        metadata_cols: list, clinical variable column names
        subtypes: list, subtypes to analyze (default: ['Diffuse', 'Limited', 'Control'])
        subtype_col: str, column containing subtype info (default: 'statustype_subtype')
        age_group_filter: str, filter on broad_age_group (default: None)
        percentile: float, threshold percentile (default: 75)
        verbose: bool, print progress (default: True)
    
    Returns:
        Dict with subtype names as keys, containing 'effect_df' and 'summary_table'
    """
    if subtype_col is None:
        subtype_col = GROUP_COL
    if subtypes is None:
        subtypes = [SUBTYPE_B, SUBTYPE_A, CONTROL_LABEL]
    all_results = {}
    for subtype in subtypes:
        if verbose:
            filter_info = f" ({age_group_filter})" if age_group_filter else ""
            print(f"\nAnalyzing {subtype} subtype{filter_info}")
        try:
            effect_df, summary_table = compute_signature_clinical_effects(signature_df=signature_df, meta_df=meta_df, signature_cols=signature_cols, metadata_cols=metadata_cols, percentile=percentile, age_group_filter=age_group_filter, filter_col=subtype_col, filter_value=subtype, verbose=verbose)
            all_results[subtype] = {'effect_df': effect_df, 'summary_table': summary_table}
        except Exception as e:
            print(f"Error analyzing {subtype} subtype: {e}")
            all_results[subtype] = None
    if verbose:
        print("\nAll subtype analyses complete")
    return all_results

def prepare_volcano_disease_data(comprehensive_results):
    """
    Prepare combined comparison data for volcano plotting from comprehensive analysis.
    
    Parameters:
        comprehensive_results: dict, output from run_comprehensive_disease_analysis
    
    Returns:
        DataFrame with columns: signature, stratum, comparison, fold_change, p_value,
        p_value_fdr, cohort, comparison_label, neg_log10_p
    """
    comp_key_to_label = {
        'diffuse_vs_control': f'{SUBTYPE_B} vs {CONTROL_LABEL}',
        'limited_vs_control': f'{SUBTYPE_A} vs {CONTROL_LABEL}',
        'diffuse_vs_limited': f'{SUBTYPE_B} vs {SUBTYPE_A}',
        'pooled_ssc_vs_control': f'Pooled SSc vs {CONTROL_LABEL}'
    }
    
    all_comparisons = []
    
    if 'q1_full_cohort' in comprehensive_results:
        q1 = comprehensive_results['q1_full_cohort']
        for comp_key, comp_label in comp_key_to_label.items():
            if comp_key in q1 and q1[comp_key] is not None and isinstance(q1[comp_key], pd.DataFrame):
                df = q1[comp_key].copy()
                df['cohort'] = 'Full'
                df['comparison_label'] = comp_label
                all_comparisons.append(df)
    
    if 'q2_baseline' in comprehensive_results:
        q2 = comprehensive_results['q2_baseline']
        for comp_key, comp_label in comp_key_to_label.items():
            if comp_key in q2 and q2[comp_key] is not None and isinstance(q2[comp_key], pd.DataFrame):
                df = q2[comp_key].copy()
                df['cohort'] = 'Baseline'
                df['comparison_label'] = comp_label
                all_comparisons.append(df)
    
    if not all_comparisons:
        raise ValueError(f"No comparison data found. Available keys: {list(comprehensive_results.keys())}")
    
    combined_df = pd.concat(all_comparisons, ignore_index=True)
    combined_df['neg_log10_p'] = -np.log10(combined_df['p_value'].clip(lower=1e-300))
    
    return combined_df

def summarize_comprehensive_analysis_findings(all_results, fdr_threshold=0.05, p_threshold=0.05):
    """
    Generate interpretive summary of comprehensive disease analysis results.
    
    Parameters:
        all_results: dict, output from run_comprehensive_disease_analysis
        fdr_threshold: float, FDR significance threshold (default: 0.05)
        p_threshold: float, nominal p-value threshold (default: 0.05)
    
    Returns:
        Dict with 'summary_df' and 'interpretation_text'
    """
    questions = {
        'q1_full_cohort': {
            'label': 'Q1: Full Cohort',
            'question': f'Are mutational signatures different between disease groups ({CONTROL_LABEL} vs {SUBTYPE_A} vs {SUBTYPE_B}) across the entire cohort, stratified by race/ethnicity?'
        },
        'q2_baseline': {
            'label': 'Q2: Baseline SSc',
            'question': 'Are signatures different at disease baseline (duration <= cutoff), reducing confounding from disease progression and treatment?'
        },
        'q2_early_disease': {
            'label': 'Q2b: Early Disease',
            'question': 'Same as Q2 but with stricter duration cutoff for very early disease.'
        },
        'q3_young_baseline': {
            'label': 'Q3: Young Baseline',
            'question': 'Are signatures different in young baseline patients? Do young SSc patients differ from older SSc patients?'
        }
    }
    
    comparison_keys = [
        ('pooled_ssc_vs_control', f'SSc (pooled) vs {CONTROL_LABEL}'),
        ('diffuse_vs_control', f'{SUBTYPE_B} vs {CONTROL_LABEL}'),
        ('limited_vs_control', f'{SUBTYPE_A} vs {CONTROL_LABEL}'),
        ('diffuse_vs_limited', f'{SUBTYPE_B} vs {SUBTYPE_A}'),
        ('young_pooled_vs_control', f'Young SSc vs {CONTROL_LABEL}'),
        ('young_vs_older_ssc', 'Young vs Older SSc')
    ]
    
    summary_rows = []
    interpretations = []
    
    interpretations.append("COMPREHENSIVE SSc MUTATIONAL SIGNATURE ANALYSIS SUMMARY")
    interpretations.append("FDR correction: Benjamini-Hochberg, applied within signature families (SBS and ID corrected separately) per comparison type")
    interpretations.append("")
    
    for q_key, q_info in questions.items():
        if q_key not in all_results:
            continue
        
        results = all_results[q_key]
        config = results.get('config', {})
        
        n_samples = len(results.get('data', results.get('data_young', pd.DataFrame())))
        
        sig_findings_raw = []
        sig_findings_fdr = []
        
        for comp_key, comp_label in comparison_keys:
            if comp_key not in results or results[comp_key] is None:
                continue
            
            df = results[comp_key]
            if not isinstance(df, pd.DataFrame) or len(df) == 0:
                continue
            
            raw_sig = df[df['p_value'] < p_threshold]
            for _, row in raw_sig.iterrows():
                fc = row.get('fold_change', np.nan)
                direction = 'elevated' if fc > 1 else 'reduced' if fc < 1 else 'unchanged'
                sig_findings_raw.append({
                    'comparison': comp_label,
                    'signature': row['signature'],
                    'stratum': row['stratum'],
                    'fold_change': fc,
                    'direction': direction,
                    'p_value': row['p_value'],
                    'p_value_fdr': row.get('p_value_fdr', np.nan),
                    'signature_family': row.get('signature_family', 'unknown')
                })
            
            if 'p_value_fdr' in df.columns:
                fdr_sig = df[df['p_value_fdr'] < fdr_threshold]
                for _, row in fdr_sig.iterrows():
                    fc = row.get('fold_change', np.nan)
                    direction = 'elevated' if fc > 1 else 'reduced' if fc < 1 else 'unchanged'
                    sig_findings_fdr.append({
                        'comparison': comp_label,
                        'signature': row['signature'],
                        'stratum': row['stratum'],
                        'fold_change': fc,
                        'direction': direction,
                        'p_value': row['p_value'],
                        'p_value_fdr': row['p_value_fdr'],
                        'signature_family': row.get('signature_family', 'unknown')
                    })
        
        summary_rows.append({
            'Analysis': q_info['label'],
            'N_Samples': n_samples,
            'Sig_Raw': len(sig_findings_raw),
            'Sig_FDR': len(sig_findings_fdr)
        })
        
        interp = _generate_question_interpretation(
            q_info, sig_findings_raw, sig_findings_fdr, n_samples, config, p_threshold, fdr_threshold
        )
        interpretations.append(interp)
    
    summary_df = pd.DataFrame(summary_rows)
    interpretation_text = '\n\n'.join(interpretations)
    
    print(interpretation_text)
    
    return {'summary_df': summary_df, 'interpretation_text': interpretation_text, 'details': all_results}


def _generate_question_interpretation(q_info, sig_raw, sig_fdr, n_samples, config, p_thresh, fdr_thresh):
    """
    Generate interpretive text for a single analysis question.
    
    Parameters:
        q_info: dict with 'label' and 'question'
        sig_raw: list of significant findings at raw p threshold
        sig_fdr: list of significant findings at FDR threshold
        n_samples: int, number of samples analyzed
        config: dict, analysis configuration
        p_thresh: float, raw p-value threshold
        fdr_thresh: float, FDR threshold
    
    Returns:
        str, formatted interpretation text
    """
    lines = []
    lines.append(f"{q_info['label']}")
    lines.append(f"Question: {q_info['question']}")
    lines.append(f"Samples analyzed: {n_samples}")
    
    if config:
        config_items = []
        if 'duration_cutoff' in config:
            config_items.append(f"duration <= {config['duration_cutoff']}yr")
        if 'age_cutoff' in config:
            config_items.append(f"age < {config['age_cutoff']}yr")
        if config_items:
            lines.append(f"Filters: {', '.join(config_items)}")
    
    if len(sig_fdr) > 0:
        lines.append(f"Result: {len(sig_fdr)} findings significant after FDR correction (FDR < {fdr_thresh})")
        
        by_comparison = {}
        for f in sig_fdr:
            comp = f['comparison']
            if comp not in by_comparison:
                by_comparison[comp] = []
            by_comparison[comp].append(f)
        
        for comp, findings in by_comparison.items():
            elevated = [f for f in findings if f['direction'] == 'elevated']
            reduced = [f for f in findings if f['direction'] == 'reduced']
            
            if elevated:
                sigs = ', '.join([
                    f"{f['signature']} ({f['stratum']}, FC={f['fold_change']:.2f}, FDR={f['p_value_fdr']:.3f})"
                    for f in elevated
                ])
                lines.append(f"  {comp} [elevated]: {sigs}")
            if reduced:
                sigs = ', '.join([
                    f"{f['signature']} ({f['stratum']}, FC={f['fold_change']:.2f}, FDR={f['p_value_fdr']:.3f})"
                    for f in reduced
                ])
                lines.append(f"  {comp} [reduced]: {sigs}")
        
        lines.append("Interpretation: Evidence for signature differences between groups.")
    
    elif len(sig_raw) > 0:
        lines.append(f"Result: {len(sig_raw)} nominally significant findings (p < {p_thresh}), none survive FDR correction")
        
        top_findings = sorted(sig_raw, key=lambda x: x['p_value'])[:5]
        for f in top_findings:
            lines.append(
                f"  {f['comparison']}: {f['signature']} ({f['stratum']}, {f['signature_family']}) "
                f"FC={f['fold_change']:.2f}, p={f['p_value']:.4f}, FDR={f['p_value_fdr']:.3f}"
            )
        
        lines.append("Interpretation: Suggestive trends but insufficient evidence after multiple testing correction.")
    
    else:
        lines.append(f"Result: No significant findings (p < {p_thresh})")
        lines.append("Interpretation: No evidence for signature differences between groups in this analysis.")
    
    return '\n'.join(lines)

def get_signature_family(signature, signature_families=None):
    """
    Determine which family a signature belongs to.
    
    Parameters:
        signature: str, signature name
        signature_families: dict, mapping of family names to signature lists (default: uses DEFAULT_SIGNATURE_FAMILIES)
    
    Returns:
        str, family name ('sbs', 'id', or 'unknown')
    """
    if signature_families is None:
        signature_families = DEFAULT_SIGNATURE_FAMILIES
    
    all_sbs = signature_families.get('de_novo_sbs', []) + signature_families.get('cosmic_sbs', [])
    all_id = signature_families.get('de_novo_id', []) + signature_families.get('cosmic_id', [])
    
    if signature in all_sbs:
        return 'sbs'
    elif signature in all_id:
        return 'id'
    elif signature.upper().startswith('SBS'):
        return 'sbs'
    elif signature.upper().startswith('ID'):
        return 'id'
    elif signature.upper().startswith('DBS'):
        return 'dbs'
    else:
        return 'unknown'


def apply_fdr_correction_by_family(results_df, p_col='p_value', signature_col='signature', signature_families=None):
    """
    Apply Benjamini-Hochberg FDR correction within each signature family.
    
    Parameters:
        results_df: DataFrame with p-value and signature columns
        p_col: str, name of p-value column (default: 'p_value')
        signature_col: str, name of signature column (default: 'signature')
        signature_families: dict, mapping of family names to signature lists (default: uses DEFAULT_SIGNATURE_FAMILIES)
    
    Returns:
        DataFrame with added 'p_value_fdr' and 'signature_family' columns
    """
    df = results_df.copy()
    
    df['signature_family'] = df[signature_col].apply(
        lambda x: get_signature_family(x, signature_families)
    )
    
    df['p_value_fdr'] = np.nan
    
    for family in df['signature_family'].unique():
        family_mask = (df['signature_family'] == family) & (df[p_col].notna())
        
        if family_mask.sum() == 0:
            continue
        
        _, pvals_fdr, _, _ = multipletests(
            df.loc[family_mask, p_col], alpha=0.05, method='fdr_bh'
        )
        df.loc[family_mask, 'p_value_fdr'] = pvals_fdr
    
    return df

def compute_signature_clinical_correlations(
    meta_df,
    signature_dfs,
    signatures=None,
    clinical_vars=None,
    race_groups=None,
    baseline_duration=3.0,
    duration_col=None,
    min_samples=3,
    verbose=True
):
    """
    Compute Spearman correlation statistics between signature burden and clinical variables.
    
    Parameters:
        meta_df: DataFrame, metadata with sample_id, race, Ethnicity, clinical variables
        signature_dfs: dict, signature DataFrames keyed by name
        signatures: list, signature columns to analyze (default: ['SBS96A', 'SBS96B', 'ID83A'])
        clinical_vars: list, clinical variable columns (default: ['DLCO %Predicted', 'FVC %predicted', 'MRSS'])
        race_groups: list, race/ethnicity groups to stratify by (default: RACE_ORDER from config)
        baseline_duration: float, max disease duration for baseline cohort (default: 3.0)
        duration_col: str, column name for disease duration (default: 'Disease Duration')
        min_samples: int, minimum samples required for correlation (default: 3)
        verbose: bool, print progress info (default: True)
    
    Returns:
        Dict with 'full_cohort_data', 'baseline_cohort_data', 'stats_df', 'config'
    """
    if duration_col is None:
        duration_col = DURATION_COL
    if signatures is None:
        signatures = ['SBS96A', 'SBS96B', 'ID83A']
    if clinical_vars is None:
        clinical_vars = [DLCO_PCT_COL, FVC_PCT_COL, SKIN_SCORE_COL]
    if race_groups is None:
        race_groups = RACE_ORDER
    
    df = meta_df.copy()
    
    for df_key, sig_df in signature_dfs.items():
        if not isinstance(sig_df, pd.DataFrame):
            continue
        sample_col = None
        for possible in ['sample_id', 'Sample_ID', 'SampleID', 'Sample']:
            if possible in sig_df.columns:
                sample_col = possible
                break
        if sample_col is None:
            continue
        if sample_col != 'sample_id':
            sig_df = sig_df.rename(columns={sample_col: 'sample_id'})
        cols_to_merge = ['sample_id'] + [c for c in sig_df.columns if c != 'sample_id' and c not in df.columns]
        if len(cols_to_merge) > 1:
            df = df.merge(sig_df[cols_to_merge], on='sample_id', how='left')
    
    df = create_race_eth_column(df)
    df = df[df['race_eth'].isin(race_groups)].copy()
    
    available_sigs = [s for s in signatures if s in df.columns]
    if not available_sigs:
        raise ValueError(f"No signatures found in data. Available columns: {df.columns.tolist()}")
    
    available_clinical = [c for c in clinical_vars if c in df.columns]
    if not available_clinical:
        raise ValueError(f"No clinical variables found. Available: {df.columns.tolist()}")
    
    has_baseline = False
    df_baseline = pd.DataFrame()
    if duration_col in df.columns:
        df_baseline = df[df[duration_col] <= baseline_duration].copy()
        has_baseline = len(df_baseline) > 0
    elif verbose:
        print(f"Warning: {duration_col} not found. Skipping baseline cohort.")
    
    if verbose:
        print(f"Analyzing {len(available_sigs)} signatures: {available_sigs}")
        print(f"Clinical variables: {available_clinical}")
        print(f"Race/ethnicity groups: {race_groups}")
        print(f"\nFull cohort: {len(df)} samples")
        if has_baseline:
            print(f"Baseline cohort (duration <= {baseline_duration} years): {len(df_baseline)} samples")
    
    def compute_cohort_correlations(data_df, cohort_name):
        """Compute correlations for a single cohort."""
        records = []
        for clin_var in available_clinical:
            for sig in available_sigs:
                for race in race_groups:
                    subset = data_df[
                        (data_df['race_eth'] == race) &
                        data_df[sig].notna() &
                        data_df[clin_var].notna()
                    ].copy()
                    
                    if len(subset) < min_samples:
                        records.append({
                            'cohort': cohort_name,
                            'signature': sig,
                            'clinical_var': clin_var,
                            'race_eth': race,
                            'n': len(subset),
                            'r': np.nan,
                            'r_squared': np.nan,
                            'p_value': np.nan,
                            'slope': np.nan,
                            'intercept': np.nan,
                            'slope_se': np.nan
                        })
                        continue
                    
                    x = subset[sig].values
                    y = subset[clin_var].values
                    
                    rho, p_value = scipy_stats.spearmanr(x, y)
                    slope, intercept, _, _, std_err = scipy_stats.linregress(x, y)
                    
                    records.append({
                        'cohort': cohort_name,
                        'signature': sig,
                        'clinical_var': clin_var,
                        'race_eth': race,
                        'n': len(subset),
                        'r': rho,
                        'r_squared': rho ** 2,
                        'p_value': p_value,
                        'slope': slope,
                        'intercept': intercept,
                        'slope_se': std_err
                    })
        return records
    
    stats_full = compute_cohort_correlations(df, 'Full Cohort')
    
    if has_baseline:
        stats_baseline = compute_cohort_correlations(
            df_baseline, f'Baseline (Duration <= {baseline_duration} years)'
        )
        all_stats = stats_full + stats_baseline
    else:
        all_stats = stats_full
    
    stats_df = pd.DataFrame(all_stats)
    
    for cohort in stats_df['cohort'].unique():
        cohort_mask = stats_df['cohort'] == cohort
        valid_mask = cohort_mask & stats_df['p_value'].notna()
        
        if valid_mask.sum() > 0:
            _, pvals_fdr, _, _ = multipletests(
                stats_df.loc[valid_mask, 'p_value'].values, alpha=0.05, method='fdr_bh'
            )
            stats_df.loc[valid_mask, 'p_value_fdr'] = pvals_fdr
    
    stats_df['sig_nominal'] = stats_df['p_value'] < 0.05
    stats_df['sig_fdr'] = stats_df['p_value_fdr'] < 0.1
    stats_df = stats_df.sort_values(['cohort', 'p_value'], ascending=[True, True])
    
    return {
        'full_cohort_data': df,
        'baseline_cohort_data': df_baseline if has_baseline else None,
        'stats_df': stats_df,
        'config': {
            'signatures': available_sigs,
            'clinical_vars': available_clinical,
            'race_groups': race_groups,
            'baseline_duration': baseline_duration,
            'min_samples': min_samples
        }
    }


def summarize_signature_clinical_correlations(stats_df, fdr_threshold=0.1, nominal_threshold=0.05):
    """
    Generate formatted summary of correlation analysis results.
    
    Parameters:
        stats_df: DataFrame, output from compute_signature_clinical_correlations
        fdr_threshold: float, FDR significance threshold (default: 0.1)
        nominal_threshold: float, nominal p-value threshold (default: 0.05)
    
    Returns:
        Dict with 'summary_text', 'fdr_significant', 'nominally_significant'
    """
    lines = []
    fdr_sig_all = []
    nominal_sig_all = []
    
    for cohort in stats_df['cohort'].unique():
        cohort_df = stats_df[stats_df['cohort'] == cohort]
        
        lines.append(f"Correlation clinical-signature summary: {cohort.upper()}")
        
        fdr_sig = cohort_df[cohort_df['p_value_fdr'] < fdr_threshold].copy()
        fdr_sig_all.append(fdr_sig)
        
        lines.append(f"\nFDR-Significant Correlations (FDR < {fdr_threshold}): {len(fdr_sig)}")
        if len(fdr_sig) > 0:
            for _, row in fdr_sig.iterrows():
                direction = "+" if row['r'] > 0 else "-"
                lines.append(
                    f"  {row['signature']} vs {row['clinical_var']} [{row['race_eth']}]: "
                    f"R={row['r']:.3f} ({direction}), R²={row['r_squared']:.3f}, "
                    f"p={row['p_value']:.2e}, FDR={row['p_value_fdr']:.3f}, n={row['n']}"
                )
        
        nominal_only = cohort_df[
            (cohort_df['p_value'] < nominal_threshold) &
            (cohort_df['p_value_fdr'] >= fdr_threshold)
        ].copy()
        nominal_sig_all.append(nominal_only)
        
        lines.append(f"\nNominally Significant Only (p < {nominal_threshold}, FDR >= {fdr_threshold}): {len(nominal_only)}")
        if len(nominal_only) > 0:
            for _, row in nominal_only.iterrows():
                direction = "+" if row['r'] > 0 else "-"
                lines.append(
                    f"  {row['signature']} vs {row['clinical_var']} [{row['race_eth']}]: "
                    f"R={row['r']:.3f} ({direction}), R²={row['r_squared']:.3f}, "
                    f"p={row['p_value']:.2e}, FDR={row['p_value_fdr']:.3f}, n={row['n']}"
                )
        
        lines.append("\nSummary by Clinical Variable:")
        for clin_var in cohort_df['clinical_var'].unique():
            subset = cohort_df[cohort_df['clinical_var'] == clin_var]
            n_fdr = (subset['p_value_fdr'] < fdr_threshold).sum()
            n_nom = (subset['p_value'] < nominal_threshold).sum()
            mean_r2 = subset['r_squared'].mean()
            lines.append(f"  {clin_var}: {n_fdr} FDR-sig, {n_nom} nominal-sig, mean R²={mean_r2:.3f}")
        
        lines.append("")
    
    summary_text = '\n'.join(lines)
    print(summary_text)
    
    return {
        'summary_text': summary_text,
        'fdr_significant': pd.concat(fdr_sig_all, ignore_index=True) if fdr_sig_all else pd.DataFrame(),
        'nominally_significant': pd.concat(nominal_sig_all, ignore_index=True) if nominal_sig_all else pd.DataFrame()
    }

def compute_cosine_similarity_matrix(denovo_df, cosmic_df):
    """
    Compute pairwise cosine similarity between de novo and COSMIC signature profiles.

    Parameters:
        denovo_df: DataFrame, de novo signatures (96 mutation types x N signatures)
        cosmic_df: DataFrame, COSMIC signatures (96 mutation types x M signatures)

    Returns:
        Dict with keys:
            'matrix': DataFrame (de novo rows x COSMIC columns) of cosine similarities
            'denovo_signatures': list of de novo signature names
            'cosmic_signatures': list of COSMIC signature names
    """
    common_types = denovo_df.index.intersection(cosmic_df.index)
    if len(common_types) != 96:
        print(f"Warning: {len(common_types)} shared mutation types (expected 96)")
    denovo_aligned = denovo_df.loc[common_types]
    cosmic_aligned = cosmic_df.loc[common_types]
    denovo_sigs = denovo_aligned.columns.tolist()
    cosmic_sigs = cosmic_aligned.columns.tolist()
    matrix = np.zeros((len(denovo_sigs), len(cosmic_sigs)))
    for i, dn in enumerate(denovo_sigs):
        vec_dn = denovo_aligned[dn].values.astype(float)
        norm_dn = np.linalg.norm(vec_dn)
        for j, cs in enumerate(cosmic_sigs):
            vec_cs = cosmic_aligned[cs].values.astype(float)
            norm_cs = np.linalg.norm(vec_cs)
            if norm_dn > 0 and norm_cs > 0:
                matrix[i, j] = np.dot(vec_dn, vec_cs) / (norm_dn * norm_cs)
            else:
                matrix[i, j] = 0.0
    result_df = pd.DataFrame(matrix, index=denovo_sigs, columns=cosmic_sigs)
    return {
        'matrix': result_df,
        'denovo_signatures': denovo_sigs,
        'cosmic_signatures': cosmic_sigs
    }

def compute_signature_clinical_forest_stats(
    meta_df,
    signature_dfs,
    signatures=None,
    clinical_vars=None,
    race_groups=None,
    age_col=None,
    min_samples=3,
    fdr_threshold=0.1,
    nominal_threshold=0.05,
    baseline_duration=None,
    duration_col=None,
    sample_ids=None
):
    """
    Compute unadjusted and age-adjusted association statistics for forest plots.

    Parameters:
        meta_df: DataFrame, metadata with sample_id, race, Ethnicity, clinical variables
        signature_dfs: dict, signature DataFrames keyed by name
        signatures: list, signature columns to analyze (default: ['SBS96A', 'SBS96B'])
        clinical_vars: list, clinical variable columns (default: ['DLCO %Predicted', 'FVC %predicted', 'MRSS'])
        race_groups: list, race/ethnicity groups (default: RACE_ORDER)
        age_col: str, column name for age (default: 'Age')
        min_samples: int, minimum samples for regression (default: 3)
        fdr_threshold: float, FDR threshold for significance (default: 0.1)
        nominal_threshold: float, nominal p-value threshold (default: 0.05)
        baseline_duration: float, if provided filter SSc to disease duration <= this value;
            controls are always retained (default: None)
        duration_col: str, column name for disease duration (default: 'Disease Duration')
        sample_ids: list, if provided restrict analysis to these sample IDs (default: None, use all)

    Returns:
        DataFrame with columns: analysis_type, clinical_var, signature, race_eth, n,
        effect, effect_se, ci_lower, ci_upper, r, p_value, p_fdr, sig_nominal, sig_fdr.
        Unadjusted effect is Spearman rho with Fisher z-transform CIs.
        Age-adjusted effect is standardized beta from OLS regression
    """
    if age_col is None:
        age_col = AGE_COL
    if duration_col is None:
        duration_col = DURATION_COL
    if signatures is None:
        signatures = ['SBS96A', 'SBS96B']
    if clinical_vars is None:
        clinical_vars = [DLCO_PCT_COL, FVC_PCT_COL, SKIN_SCORE_COL]
    if race_groups is None:
        race_groups = RACE_ORDER

    df = meta_df.copy()
    for df_key, sig_df in signature_dfs.items():
        if not isinstance(sig_df, pd.DataFrame):
            continue
        sample_col = None
        for candidate in ['sample_id', 'Sample_ID', 'SampleID', 'Sample']:
            if candidate in sig_df.columns:
                sample_col = candidate
                break
        if sample_col is None:
            continue
        if sample_col != 'sample_id':
            sig_df = sig_df.rename(columns={sample_col: 'sample_id'})
        cols_to_merge = ['sample_id'] + [c for c in sig_df.columns if c != 'sample_id' and c not in df.columns]
        if len(cols_to_merge) > 1:
            df = df.merge(sig_df[cols_to_merge], on='sample_id', how='left')

    df = create_race_eth_column(df)
    df = df[df['race_eth'].isin(race_groups)].copy()
    if sample_ids is not None:
        df = df[df['sample_id'].isin(sample_ids)].copy()
        
    if baseline_duration is not None and duration_col in df.columns:
        is_control = df[GROUP_COL] == CONTROL_LABEL
        has_dur = df[duration_col].notna() & (df[duration_col] <= baseline_duration)
        df = df[is_control | has_dur].copy()

    available_sigs = [s for s in signatures if s in df.columns]
    available_clinical = [c for c in clinical_vars if c in df.columns]

    if not available_sigs:
        raise ValueError(f"No signatures found. Available: {df.columns.tolist()}")
    if not available_clinical:
        raise ValueError(f"No clinical vars found. Available: {df.columns.tolist()}")

    records = []
    for clin_var in available_clinical:
        for sig in available_sigs:
            for race in race_groups:
                subset = df[
                    (df['race_eth'] == race) &
                    df[sig].notna() &
                    df[clin_var].notna() &
                    df[age_col].notna()
                ].copy()
                n = len(subset)
                if n < min_samples:
                    for atype in ['Unadjusted', 'Age-Adjusted']:
                        records.append({
                            'analysis_type': atype, 'clinical_var': clin_var,
                            'signature': sig, 'race_eth': race, 'n': n,
                            'effect': np.nan, 'effect_se': np.nan,
                            'ci_lower': np.nan, 'ci_upper': np.nan,
                            'r': np.nan, 'p_value': np.nan
                        })
                    continue
                x_sig = subset[sig].values
                y_clin = subset[clin_var].values
                x_age = subset[age_col].values
                rho, p_val = scipy_stats.spearmanr(x_sig, y_clin)
                records.append({
                    'analysis_type': 'Unadjusted', 'clinical_var': clin_var,
                    'signature': sig, 'race_eth': race, 'n': n,
                    'effect': rho, 'effect_se': np.nan,
                    'ci_lower': np.tanh(np.arctanh(rho) - 1.96 / np.sqrt(n - 3)) if n > 3 else np.nan,
                    'ci_upper': np.tanh(np.arctanh(rho) + 1.96 / np.sqrt(n - 3)) if n > 3 else np.nan,
                    'r': rho, 'p_value': p_val
                })
                x_sig_z = (x_sig - x_sig.mean()) / x_sig.std(ddof=1) if x_sig.std(ddof=1) > 0 else np.zeros_like(x_sig)
                x_age_z = (x_age - x_age.mean()) / x_age.std(ddof=1) if x_age.std(ddof=1) > 0 else np.zeros_like(x_age)
                y_clin_z = (y_clin - y_clin.mean()) / y_clin.std(ddof=1) if y_clin.std(ddof=1) > 0 else np.zeros_like(y_clin)
                X_mat = np.column_stack([np.ones(n), x_sig_z, x_age_z])
                try:
                    betas, residuals, rank, sv = np.linalg.lstsq(X_mat, y_clin_z, rcond=None)
                    std_beta_sig = betas[1]
                    y_pred = X_mat @ betas
                    resid = y_clin_z - y_pred
                    dof = n - 3
                    mse = np.sum(resid ** 2) / dof if dof > 0 else np.nan
                    XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
                    se_beta_sig = np.sqrt(mse * XtX_inv[1, 1]) if mse > 0 else np.nan
                    t_stat = std_beta_sig / se_beta_sig if se_beta_sig > 0 else np.nan
                    p_adj = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), dof)) if not np.isnan(t_stat) else np.nan
                    t_crit_adj = scipy_stats.t.ppf(0.975, dof)
                    ci_lower_adj = std_beta_sig - t_crit_adj * se_beta_sig if not np.isnan(se_beta_sig) else np.nan
                    ci_upper_adj = std_beta_sig + t_crit_adj * se_beta_sig if not np.isnan(se_beta_sig) else np.nan
                except np.linalg.LinAlgError:
                    std_beta_sig, se_beta_sig, p_adj = np.nan, np.nan, np.nan
                    ci_lower_adj, ci_upper_adj = np.nan, np.nan
                records.append({
                    'analysis_type': 'Age-Adjusted', 'clinical_var': clin_var,
                    'signature': sig, 'race_eth': race, 'n': n,
                    'effect': std_beta_sig, 'effect_se': se_beta_sig,
                    'ci_lower': ci_lower_adj, 'ci_upper': ci_upper_adj,
                    'r': np.nan, 'p_value': p_adj
                })

    stats_df = pd.DataFrame(records)
    for atype in stats_df['analysis_type'].unique():
        for sig in available_sigs:
            mask = (stats_df['analysis_type'] == atype) & (stats_df['signature'] == sig) & stats_df['p_value'].notna()
            if mask.sum() > 0:
                _, pvals_fdr, _, _ = multipletests(stats_df.loc[mask, 'p_value'].values, alpha=0.05, method='fdr_bh')
                stats_df.loc[mask, 'p_fdr'] = pvals_fdr
    stats_df['sig_nominal'] = stats_df['p_value'] < nominal_threshold
    stats_df['sig_fdr'] = stats_df.get('p_fdr', pd.Series(dtype=float)) < fdr_threshold
    return stats_df
