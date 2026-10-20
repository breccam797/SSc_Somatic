import os
import re
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from scipy import stats
from scipy.stats import f_oneway, kruskal, chi2_contingency, fisher_exact,mannwhitneyu,ttest_ind,false_discovery_control
import matplotlib.pyplot as plt
from statsmodels.formula.api import logit, ols
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from itertools import combinations
from collections import defaultdict
from scipy.stats import pearsonr

from ..settings import cohort_values
from .plots import save_and_show


# Cohort labels and metadata column names. These start empty and are filled
# from the study configuration by apply_settings(), which SSc_Somatic.configure() calls.
AGE_COL = None
CASE_LABEL = None
CONTROL_LABEL = None
DLCO_PCT_COL = None
DURATION_COL = None
FVC_PCT_COL = None
GROUP_COL = None
META_ID_COL = None
RACE_COL = None
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


def parse_vcf_format_field(format_str: str, data_str: str, field_name: str) -> Optional[str]:
    """
    Extract a specific field from VCF FORMAT/sample data columns.
    
    Parameters:
        format_str: FORMAT column (e.g., "GT:AD:AF:DP:F1R2:F2R1:SB")
        data_str: Sample data (e.g., "0/1:45,23:0.34:68:20,15:25,18:10,15,8,12")
        field_name: Field name to extract (e.g., 'AD', 'DP', 'SB')
    
    Returns:
        Raw field value as string or None if not found
    """
    if pd.isna(format_str) or pd.isna(data_str):
        return None
    try:
        format_fields = str(format_str).split(':')
        data_fields = str(data_str).split(':')
        if len(format_fields) != len(data_fields):
            return None
        field_dict = dict(zip(format_fields, data_fields))
        return field_dict.get(field_name)
    except Exception:
        return None


def extract_allele_depths(format_str: str, data_str: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract reference and alternate allele depths from VCF FORMAT data."""
    ad_str = parse_vcf_format_field(format_str, data_str, 'AD')
    if ad_str is None:
        return None, None
    try:
        ad_values = ad_str.split(',')
        if len(ad_values) >= 2:
            return int(ad_values[0]), int(ad_values[1])
        return None, None
    except (ValueError, IndexError):
        return None, None


def extract_strand_bias(format_str: str, data_str: str) -> Optional[List[int]]:
    """Extract strand bias values from VCF FORMAT data."""
    sb_str = parse_vcf_format_field(format_str, data_str, 'SB')
    if sb_str is None:
        return None
    try:
        return [int(x) for x in sb_str.split(',')]
    except (ValueError, AttributeError):
        return None


def process_snv_data(snv_df, meta_df, bins, labels):
    """
    Merge SNV data with metadata and create age groups and statustype_subtype columns.
    
    Parameters:
        snv_df: SNV dataframe with sample_id column
        meta_df: Metadata with Patient_ID, Age, statustype, SSc Type columns
        bins: Bin edges for age grouping
        labels: Labels for age groups
    
    Returns:
        Processed dataframe with merged metadata and derived columns
    """
    snv_anno_df = snv_df.merge(meta_df, how='left', left_on='sample_id', right_on=META_ID_COL)
    snv_anno_df['age_group'] = pd.cut(snv_anno_df[AGE_COL], bins=bins, labels=labels, right=False)
    snv_anno_df[GROUP_COL] = snv_anno_df[STATUS_COL].apply(
        lambda x: CONTROL_LABEL if x == CONTROL_LABEL else None
    )
    snv_anno_df.loc[
        (snv_anno_df[STATUS_COL] == CASE_LABEL) & (snv_anno_df[SUBTYPE_COL] == SUBTYPE_A),
        GROUP_COL
    ] = SUBTYPE_A
    snv_anno_df.loc[
        (snv_anno_df[STATUS_COL] == CASE_LABEL) & (snv_anno_df[SUBTYPE_COL] == SUBTYPE_B),
        GROUP_COL
    ] = SUBTYPE_B
    return snv_anno_df


def concatenate_wl_files(vcf_root, sample_ids, output_filename='ssc_CHIP_tidy_table_WL.txt', verbose=True,
                         output_dir=None, file_pattern='{sample_id}.annovar.varsOI.wl.txt'):
    """
    Concatenate *wl.txt files for specified sample IDs into a single tidy table.

    Parameters:
        vcf_root: Directory containing the wl.txt files
        sample_ids: List of sample IDs to process
        output_filename: Name of output file
        verbose: Print progress information
        output_dir: Directory for the output file (default: None, which writes into vcf_root)
        file_pattern: Per-sample file name, with {sample_id} as the placeholder
            (default: '{sample_id}.annovar.varsOI.wl.txt')

    Returns:
        tuple: (combined_df, output_path)
    """
    if not os.path.exists(vcf_root):
        print(f"ERROR: Directory {vcf_root} does not exist")
        return pd.DataFrame(), ""
    if not sample_ids or not isinstance(sample_ids, (list, tuple)):
        print("ERROR: sample_ids must be a non-empty list or tuple of sample IDs")
        return pd.DataFrame(), ""
    if verbose:
        print(f"Looking for {len(sample_ids)} sample files in: {vcf_root}")
        print(f"Expected filename pattern: {file_pattern}")
    dataframes = []
    file_info = []
    missing_files = []
    for sample_id in sample_ids:
        expected_filename = file_pattern.format(sample_id=sample_id)
        file_path = os.path.join(vcf_root, expected_filename)
        if not os.path.exists(file_path):
            missing_files.append(expected_filename)
            if verbose:
                print(f"WARNING: File not found: {expected_filename}")
            continue
        try:
            if verbose:
                print(f"Reading: {expected_filename}")
            df = pd.read_csv(file_path, sep='\t', low_memory=False)
            df['source_file'] = expected_filename
            df['sample_id'] = sample_id
            dataframes.append(df)
            file_info.append({
                'file': expected_filename,
                'sample_id': sample_id,
                'variants': len(df)
            })
            if verbose:
                print(f"  -> {len(df)} variants from sample {sample_id}")
        except Exception as e:
            print(f"WARNING: Could not read {file_path}: {e}")
            continue
    if missing_files:
        print(f"\nWARNING: {len(missing_files)} files were not found:")
        for missing_file in missing_files:
            print(f"  - {missing_file}")
    if not dataframes:
        print("ERROR: No files could be successfully read")
        return pd.DataFrame(), ""
    if verbose:
        print(f"\nConcatenating {len(dataframes)} dataframes...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    output_path = os.path.join(output_dir if output_dir is not None else vcf_root, output_filename)
    try:
        combined_df.to_csv(output_path, sep='\t', index=False)
        if verbose:
            print(f"Successfully saved combined data to: {output_path}")
    except Exception as e:
        print(f"ERROR saving file: {e}")
        return combined_df, ""
    if verbose:
        print(f"\nConcatenation summary:")
        print(f"Sample IDs requested: {len(sample_ids)}")
        print(f"Files successfully processed: {len(dataframes)}")
        print(f"Files missing/failed: {len(sample_ids) - len(dataframes)}")
        print(f"Total variants: {len(combined_df):,}")
        print(f"Unique samples: {combined_df['sample_id'].nunique()}")
        print(f"Output file: {output_path}")
        print(f"\nPer-sample variant counts:")
        sample_counts = combined_df['sample_id'].value_counts().sort_index()
        for sample, count in sample_counts.items():
            print(f"  {sample}: {count:,} variants")
    return combined_df, output_path


def filter_variants_by_dp(wl_df, min_dp_threshold=20, verbose=True):
    """
    Filter variants based on minimum depth (DP) threshold extracted from Otherinfo11.
    
    Parameters:
        wl_df: Input dataframe with Otherinfo11 column
        min_dp_threshold: Minimum DP value for filtering
        verbose: Print statistics
    
    Returns:
        Filtered dataframe with DP >= min_dp_threshold
    """
    df = wl_df.copy()
    if verbose:
        print("Extracting DP values from Otherinfo11...")
    df['DP_value'] = df['Otherinfo11'].apply(
        lambda x: int(re.search(r'DP=(\d+)', x).group(1)) 
        if pd.notna(x) and isinstance(x, str) and re.search(r'DP=(\d+)', x) 
        else None
    )
    initial_count = len(df)
    dp_available_count = df['DP_value'].notna().sum()
    dp_missing_count = df['DP_value'].isna().sum()
    if verbose:
        print(f"Initial dataset: {initial_count} variants")
        print(f"Variants with DP values: {dp_available_count}")
        print(f"Variants missing DP values: {dp_missing_count}")
    if dp_available_count > 0:
        dp_stats = df['DP_value'].describe()
        if verbose:
            print(f"\nDP value statistics:")
            print(f"Mean: {dp_stats['mean']:.1f}")
            print(f"Median: {dp_stats['50%']:.1f}")
            print(f"Min: {dp_stats['min']:.0f}")
            print(f"Max: {dp_stats['max']:.0f}")
            print(f"Values < {min_dp_threshold}: {(df['DP_value'] < min_dp_threshold).sum()}")
            print(f"Values >= {min_dp_threshold}: {(df['DP_value'] >= min_dp_threshold).sum()}")
    df_filtered = df[df['DP_value'] >= min_dp_threshold].copy()
    final_count = len(df_filtered)
    removed_count = initial_count - final_count
    if verbose:
        print(f"\nFiltering results:")
        print(f"Variants kept: {final_count}")
        print(f"Variants removed: {removed_count}")
        print(f"Retention rate: {(final_count/initial_count)*100:.1f}%")
    return df_filtered


def check_sb_requirements(sb_list):
    """
    Check if strand bias meets requirement: at least 1 read in each direction.
    SB format: [ref_forward, ref_reverse, alt_forward, alt_reverse]
    """
    if sb_list is None or len(sb_list) != 4:
        return False
    ref_fwd, ref_rev, alt_fwd, alt_rev = sb_list
    ref_both_directions = ref_fwd >= 1 and ref_rev >= 1
    alt_both_directions = alt_fwd >= 1 and alt_rev >= 1
    return ref_both_directions and alt_both_directions


def filter_variants_by_ad_sb(df_filtered, format_col='Otherinfo12', sample_col='Otherinfo13',
                             min_alt_depth=3, auto_detect_columns=True, verbose=True,
                             keep_intermediate_columns=False):
    """
    Filter variants by allele depth (AD) and strand bias (SB) requirements.
    
    Parameters:
        df_filtered: Input dataframe (typically after DP filtering)
        format_col: Column with VCF FORMAT info
        sample_col: Column with sample data
        min_alt_depth: Minimum alternate allele depth
        auto_detect_columns: Auto-detect FORMAT vs sample columns
        verbose: Print statistics
        keep_intermediate_columns: Retain ref_depth, alt_depth, sb_values, sb_pass
    
    Returns:
        Filtered dataframe with VAF column added
    """
    if verbose:
        print("Extracting AD and SB values...")
    if auto_detect_columns:
        if verbose:
            print("Checking Otherinfo12 and Otherinfo13 structure...")
            print("Sample Otherinfo12:", str(df_filtered['Otherinfo12'].iloc[0])[:100])
            print("Sample Otherinfo13:", str(df_filtered['Otherinfo13'].iloc[0])[:100])
        sample_12 = str(df_filtered['Otherinfo12'].iloc[0])
        sample_13 = str(df_filtered['Otherinfo13'].iloc[0])
        if ':' in sample_12 and not any(char.isdigit() for char in sample_12.split(':')[0]):
            format_col, sample_col = 'Otherinfo12', 'Otherinfo13'
            if verbose:
                print("Using Otherinfo12 as FORMAT, Otherinfo13 as sample data")
        elif ':' in sample_13 and not any(char.isdigit() for char in sample_13.split(':')[0]):
            format_col, sample_col = 'Otherinfo13', 'Otherinfo12'
            if verbose:
                print("Using Otherinfo13 as FORMAT, Otherinfo12 as sample data")
        else:
            format_col, sample_col = 'Otherinfo12', 'Otherinfo13'
            if verbose:
                print("Assuming Otherinfo12 as FORMAT, Otherinfo13 as sample data")
    if verbose:
        print(f"FORMAT column: {format_col}")
        print(f"Sample column: {sample_col}")
    df_working = df_filtered.copy()
    if verbose:
        print("Extracting allele depths (AD)...")
    ad_results = df_working.apply(
        lambda row: extract_allele_depths(row[format_col], row[sample_col]), axis=1
    )
    df_working['ref_depth'] = ad_results.apply(lambda x: x[0])
    df_working['alt_depth'] = ad_results.apply(lambda x: x[1])
    if verbose:
        print("Extracting strand bias (SB)...")
    df_working['sb_values'] = df_working.apply(
        lambda row: extract_strand_bias(row[format_col], row[sample_col]), axis=1
    )
    initial_count = len(df_working)
    ad_available = (df_working['ref_depth'].notna() & df_working['alt_depth'].notna()).sum()
    sb_available = df_working['sb_values'].notna().sum()
    if verbose:
        print(f"\nAfter DP filtering: {initial_count} variants")
        print(f"Variants with AD values: {ad_available}")
        print(f"Variants with SB values: {sb_available}")
    if ad_available > 0 and verbose:
        print(f"\nAD statistics:")
        print(f"Reference depth - Mean: {df_working['ref_depth'].mean():.1f}, Median: {df_working['ref_depth'].median():.1f}")
        print(f"Alternate depth - Mean: {df_working['alt_depth'].mean():.1f}, Median: {df_working['alt_depth'].median():.1f}")
        print(f"Alt depth < {min_alt_depth}: {(df_working['alt_depth'] < min_alt_depth).sum()}")
        print(f"Alt depth >= {min_alt_depth}: {(df_working['alt_depth'] >= min_alt_depth).sum()}")
    df_working['sb_pass'] = df_working['sb_values'].apply(check_sb_requirements)
    if sb_available > 0 and verbose:
        sb_pass_count = df_working['sb_pass'].sum()
        sb_fail_count = (df_working['sb_values'].notna() & ~df_working['sb_pass']).sum()
        print(f"\nSB statistics:")
        print(f"Variants passing SB requirement (>=1 read each direction): {sb_pass_count}")
        print(f"Variants failing SB requirement: {sb_fail_count}")
    if verbose:
        print(f"\nCalculating VAF (Variant Allele Frequency)...")
    df_working['VAF'] = [
        alt / (alt + ref) if pd.notna(alt) and pd.notna(ref) and (alt + ref) > 0 else None
        for alt, ref in zip(df_working['alt_depth'], df_working['ref_depth'])
    ]
    vaf_available = df_working['VAF'].notna().sum()
    if vaf_available > 0 and verbose:
        vaf_stats = df_working['VAF'].describe()
        print(f"VAF statistics:")
        print(f"Mean: {vaf_stats['mean']:.3f}")
        print(f"Median: {vaf_stats['50%']:.3f}")
        print(f"Min: {vaf_stats['min']:.3f}")
        print(f"Max: {vaf_stats['max']:.3f}")
    if verbose:
        print(f"\nApplying filters: AD >= {min_alt_depth} AND SB >= 1 read each direction...")
    ad_filter = df_working['alt_depth'] >= min_alt_depth
    sb_filter = df_working['sb_pass'] == True
    combined_filter = ad_filter & sb_filter
    df_final = df_working[combined_filter].copy()
    final_count = len(df_final)
    removed_count = initial_count - final_count
    if verbose:
        print(f"\nFinal filtering results:")
        print(f"Variants after DP filter: {initial_count}")
        print(f"Variants after AD + SB filters: {final_count}")
        print(f"Variants removed by AD/SB filters: {removed_count}")
        print(f"Overall retention rate: {(final_count/initial_count)*100:.1f}%")
        ad_only_fail = (df_working['alt_depth'] < min_alt_depth).sum()
        sb_only_fail = (~df_working['sb_pass']).sum()
        print(f"\nBreakdown:")
        print(f"Failed AD requirement (alt_depth < {min_alt_depth}): {ad_only_fail}")
        print(f"Failed SB requirement: {sb_only_fail}")
    if not keep_intermediate_columns:
        df_final = df_final.drop(['ref_depth', 'alt_depth', 'sb_values', 'sb_pass'], axis=1)
    if verbose:
        print(f"\nFinal dataset shape: {df_final.shape}")
        print(f"VAF column added to final dataset: {'VAF' in df_final.columns}")
    if 'VAF' in df_final.columns and verbose:
        final_vaf_stats = df_final['VAF'].describe()
        print(f"\nFinal dataset VAF statistics:")
        print(f"Count: {final_vaf_stats['count']:.0f}")
        print(f"Mean: {final_vaf_stats['mean']:.3f}")
        print(f"Std: {final_vaf_stats['std']:.3f}")
        print(f"25%: {final_vaf_stats['25%']:.3f}")
        print(f"50%: {final_vaf_stats['50%']:.3f}")
        print(f"75%: {final_vaf_stats['75%']:.3f}")
    if removed_count > 0 and verbose:
        print("\nSample of variants removed by AD/SB filters:")
        removed_variants = df_working[~combined_filter][['Gene.refGene', 'Chr', 'Start', 'ref_depth', 'alt_depth', 'VAF', 'sb_pass']].head()
        print(removed_variants)
    return df_final


def process_wl_chip_data_complete_ages(df_final, meta_df, bins, labels):
    """
    Process WL CHIP data: clean sample IDs, add metadata, create subgroup classifications.
    Excludes samples with missing age data.
    
    Parameters:
        df_final: Dataframe with sample_id and VAF columns
        meta_df: Metadata with Patient_ID, statustype, Age, SSc Type columns
        bins: Bin edges for age grouping
        labels: Labels for age groups
    
    Returns:
        tuple: (wl_final_anno_df, chip_ids, big_chip_ids, excluded_count)
    """
    df_working = df_final.copy()
    df_working['sample_id'] = df_working['sample_id'].str.split('.').str[0]
    df_working['wl_CHIP'] = 'WL'
    wl_final_anno_df = df_working.merge(meta_df, how='left', left_on='sample_id', right_on=META_ID_COL)
    wl_final_anno_df = wl_final_anno_df.loc[
        wl_final_anno_df[STATUS_COL].isin([CASE_LABEL, CONTROL_LABEL])
    ]
    samples_before = len(wl_final_anno_df)
    wl_final_anno_df = wl_final_anno_df.dropna(subset=[AGE_COL])
    excluded_count = samples_before - len(wl_final_anno_df)
    print(f"Excluded {excluded_count} samples with missing age; {len(wl_final_anno_df)} remaining")
    wl_final_anno_df['Age_Group'] = pd.cut(
        wl_final_anno_df[AGE_COL], bins=bins, labels=labels, right=False
    )
    chip_ids = wl_final_anno_df['sample_id'].tolist()
    big_chip_ids = wl_final_anno_df.loc[wl_final_anno_df['VAF'] >= 0.10]['sample_id'].tolist()
    wl_final_anno_df[GROUP_COL] = wl_final_anno_df[STATUS_COL].apply(
        lambda x: CONTROL_LABEL if x == CONTROL_LABEL else None
    )
    wl_final_anno_df.loc[
        (wl_final_anno_df[STATUS_COL] == CASE_LABEL) & 
        (wl_final_anno_df[SUBTYPE_COL] == SUBTYPE_A),
        GROUP_COL
    ] = SUBTYPE_A
    wl_final_anno_df.loc[
        (wl_final_anno_df[STATUS_COL] == CASE_LABEL) & 
        (wl_final_anno_df[SUBTYPE_COL] == SUBTYPE_B),
        GROUP_COL
    ] = SUBTYPE_B
    return wl_final_anno_df, chip_ids, big_chip_ids, excluded_count


def comprehensive_chip_count_sample_analysis(df):
    """
    Perform descriptive statistics and statistical tests between statustype_subtype groups.
    
    Parameters:
        df: Input dataframe with genomic and clinical data
    
    Returns:
        Dictionary containing descriptive_stats, statistical_tests, and additional_analyses
    """
    columns_of_interest = [
        'gender', RACE_COL, 'ethnicity', AGE_COL, DURATION_COL, 
        'ana', 'anti-centromere', 'anti-topo', 'anti-RNA polymerase III',
        SKIN_SCORE_COL, FVC_PCT_COL, DLCO_PCT_COL
    ]
    print("Step 1: Cleaning column names...")
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.rstrip()
    for col in columns_of_interest:
        if col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str).str.rstrip()
    print(f"Cleaned {len(df_clean.columns)} column names")
    if GROUP_COL not in df_clean.columns:
        print(f"Warning: '{GROUP_COL}' column not found!")
        return None
    print(f"\nOriginal dataset: {len(df_clean)} rows")
    rows_with_missing_status = df_clean[GROUP_COL].isna().sum()
    print(f"Rows with missing {GROUP_COL}: {rows_with_missing_status}")
    df_clean = df_clean[df_clean[GROUP_COL].notna()].copy()
    print(f"Dataset after removing missing {GROUP_COL}: {len(df_clean)} rows")
    print("\nStep 2: Creating sample-level dataset...")
    sample_df = df_clean.drop_duplicates(subset=['sample_id'], keep='first').copy()
    print(f"Sample-level dataset: {len(sample_df)} rows")
    print(f"Unique samples (with valid {GROUP_COL}): {df_clean['sample_id'].nunique()}")
    groups = sample_df[GROUP_COL].value_counts()
    print(f"\nStatustype_subtype groups:")
    for group, count in groups.items():
        print(f"  {group}: {count} samples")
    print(f"\nDescriptive statistics by {GROUP_COL}")
    results = {}
    numerical_cols = []
    categorical_cols = []
    for col in columns_of_interest:
        if col in sample_df.columns:
            numeric_version = pd.to_numeric(sample_df[col], errors='coerce')
            if numeric_version.notna().sum() / sample_df[col].notna().sum() > 0.5:
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
        else:
            print(f"Warning: Column '{col}' not found in dataset")
    print(f"\nNumerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")
    desc_stats = {}
    for group in sample_df[GROUP_COL].unique():
        group_data = sample_df[sample_df[GROUP_COL] == group]
        desc_stats[group] = {}
        print(f"\nGroup: {group} (N={len(group_data)})")
        for col in numerical_cols:
            if col in group_data.columns:
                numeric_data = pd.to_numeric(group_data[col], errors='coerce')
                valid_data = numeric_data.dropna()
                if len(valid_data) > 0:
                    mean_val = valid_data.mean()
                    median_val = valid_data.median()
                    std_val = valid_data.std()
                    desc_stats[group][col] = {
                        'type': 'numerical',
                        'n_valid': len(valid_data),
                        'n_missing': len(numeric_data) - len(valid_data),
                        'mean': mean_val,
                        'median': median_val,
                        'std': std_val,
                        'min': valid_data.min(),
                        'max': valid_data.max()
                    }
                    print(f"{col}: N={len(valid_data)}, Mean={mean_val:.2f}, Median={median_val:.2f}, SD={std_val:.2f}")
                else:
                    desc_stats[group][col] = {'type': 'numerical', 'n_valid': 0, 'n_missing': len(group_data)}
                    print(f"{col}: No valid numerical data")
        for col in categorical_cols:
            if col in group_data.columns:
                value_counts = group_data[col].value_counts(dropna=False)
                desc_stats[group][col] = {
                    'type': 'categorical',
                    'n_total': len(group_data),
                    'value_counts': value_counts.to_dict()
                }
                print(f"{col}:")
                for value, count in value_counts.items():
                    pct = (count / len(group_data)) * 100
                    print(f"  {value}: {count} ({pct:.1f}%)")
    results['descriptive_stats'] = desc_stats
    print("\nStatistical tests between groups")
    test_results = {}
    test_data = sample_df.copy()
    unique_groups = test_data[GROUP_COL].unique()
    print(f"Testing differences between {len(unique_groups)} groups: {list(unique_groups)}")
    print("\nNumerical variables:")
    for col in numerical_cols:
        if col in test_data.columns:
            print(f"\nTesting {col}:")
            group_data_list = []
            group_names = []
            for group in unique_groups:
                group_subset = test_data[test_data[GROUP_COL] == group]
                numeric_data = pd.to_numeric(group_subset[col], errors='coerce').dropna()
                if len(numeric_data) > 0:
                    group_data_list.append(numeric_data)
                    group_names.append(group)
                    print(f"  {group}: N={len(numeric_data)}")
            if len(group_data_list) >= 2:
                if len(group_data_list) == 2:
                    normal_group1 = len(group_data_list[0]) < 5000 and stats.shapiro(group_data_list[0])[1] > 0.05
                    normal_group2 = len(group_data_list[1]) < 5000 and stats.shapiro(group_data_list[1])[1] > 0.05
                    if normal_group1 and normal_group2:
                        stat, p_value = stats.ttest_ind(group_data_list[0], group_data_list[1])
                        test_name = "Independent t-test"
                    else:
                        stat, p_value = stats.mannwhitneyu(group_data_list[0], group_data_list[1], alternative='two-sided')
                        test_name = "Mann-Whitney U test"
                else:
                    all_normal = all(len(group) < 5000 and stats.shapiro(group)[1] > 0.05 for group in group_data_list)
                    if all_normal:
                        stat, p_value = f_oneway(*group_data_list)
                        test_name = "One-way ANOVA"
                    else:
                        stat, p_value = kruskal(*group_data_list)
                        test_name = "Kruskal-Wallis test"
                test_results[col] = {
                    'type': 'numerical',
                    'test': test_name,
                    'statistic': stat,
                    'p_value': p_value,
                    'groups_tested': group_names
                }
                print(f"  Test: {test_name}")
                print(f"  Statistic: {stat:.4f}")
                print(f"  P-value: {p_value:.6f}")
                if p_value < 0.05:
                    print(f"  Result: Significant difference (p < 0.05)")
                else:
                    print(f"  Result: No significant difference (p >= 0.05)")
            else:
                print(f"  Cannot perform test: insufficient data")
                test_results[col] = {'type': 'numerical', 'test': 'insufficient_data'}
    print("\nCategorical variables:")
    for col in categorical_cols:
        if col in test_data.columns:
            print(f"\nTesting {col}:")
            contingency_table = pd.crosstab(test_data[GROUP_COL], test_data[col], dropna=False)
            print(f"  Contingency table shape: {contingency_table.shape}")
            print(contingency_table)
            if contingency_table.size > 0:
                chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)
                min_expected = expected.min()
                if min_expected >= 5 and contingency_table.shape == (2, 2):
                    test_name = "Chi-square test"
                    stat, p_value = chi2_stat, chi2_p
                elif contingency_table.shape == (2, 2):
                    oddsratio, p_value = fisher_exact(contingency_table)
                    stat = oddsratio
                    test_name = "Fisher's exact test"
                else:
                    test_name = "Chi-square test"
                    stat, p_value = chi2_stat, chi2_p
                test_results[col] = {
                    'type': 'categorical',
                    'test': test_name,
                    'statistic': stat,
                    'p_value': p_value,
                    'contingency_table': contingency_table
                }
                print(f"  Test: {test_name}")
                print(f"  Statistic: {stat:.4f}")
                print(f"  P-value: {p_value:.6f}")
                if p_value < 0.05:
                    print(f"  Result: Significant association (p < 0.05)")
                else:
                    print(f"  Result: No significant association (p >= 0.05)")
            else:
                print(f"  Cannot perform test: empty contingency table")
                test_results[col] = {'type': 'categorical', 'test': 'insufficient_data'}
    results['statistical_tests'] = test_results
    print("\nAdditional analyses")
    sample_counts = df_clean['sample_id'].value_counts()
    samples_with_multiple = sample_counts[sample_counts > 1]
    print(f"Number of sample_ids with more than one occurrence: {len(samples_with_multiple)}")
    most_frequent_sample = sample_counts.index[0]
    most_frequent_count = sample_counts.iloc[0]
    print(f"Sample with most occurrences: {most_frequent_sample} ({most_frequent_count} occurrences)")
    if 'Gene.refGene' in df_clean.columns:
        gene_counts = df_clean['Gene.refGene'].value_counts()
        most_frequent_gene = gene_counts.index[0]
        most_frequent_gene_count = gene_counts.iloc[0]
        print(f"Gene.refGene with most occurrences: {most_frequent_gene} ({most_frequent_gene_count} occurrences)")
    else:
        print("Gene.refGene column not found")
        most_frequent_gene = None
        most_frequent_gene_count = None
    results['additional_analyses'] = {
        'total_samples_with_valid_status': df_clean['sample_id'].nunique(),
        'total_rows_with_valid_status': len(df_clean),
        'samples_with_multiple_occurrences': len(samples_with_multiple),
        'sample_with_most_occurrences': {
            'sample_id': most_frequent_sample,
            'count': most_frequent_count
        },
        'gene_with_most_occurrences': {
            'gene': most_frequent_gene,
            'count': most_frequent_gene_count
        }
    }
    print(f"\nAnalysis complete!")
    print(f"Total unique samples (with valid {GROUP_COL}): {df_clean['sample_id'].nunique()}")
    print(f"Total rows in filtered dataset: {len(df_clean)}")
    return results


def create_pvalue_summary(results):
    """
    Create a summary table of all p-values from statistical tests.
    
    Parameters:
        results: Dictionary from comprehensive_chip_count_sample_analysis()
    
    Returns:
        DataFrame with Variable, Test, P-value, Significant columns sorted by p-value
    """
    if 'statistical_tests' not in results:
        return None
    summary_data = []
    for variable, test_info in results['statistical_tests'].items():
        if 'p_value' in test_info:
            summary_data.append({
                'Variable': variable,
                'Test': test_info['test'],
                'P-value': test_info['p_value'],
                'Significant': 'Yes' if test_info['p_value'] < 0.05 else 'No'
            })
    summary_df = pd.DataFrame(summary_data)
    return summary_df.sort_values('P-value')


def summarize_significant_findings(results, alpha=0.05):
    """
    Create a comprehensive summary of significant findings from the analysis.
    
    Parameters:
        results: Results dictionary from comprehensive_chip_count_sample_analysis()
        alpha: Significance threshold (default: 0.05)
    
    Returns:
        Dictionary with summary_stats, significant_findings, all_findings, summary_table
    """
    print("Summary of significant findings")
    if 'statistical_tests' not in results or 'descriptive_stats' not in results:
        print("Error: Results dictionary missing required sections")
        return None
    significant_findings = []
    all_findings = []
    for variable, test_info in results['statistical_tests'].items():
        if 'p_value' in test_info:
            finding = {
                'variable': variable,
                'test': test_info['test'],
                'p_value': test_info['p_value'],
                'significant': test_info['p_value'] < alpha,
                'test_type': test_info['type']
            }
            if 'statistic' in test_info:
                finding['statistic'] = test_info['statistic']
            all_findings.append(finding)
            if finding['significant']:
                significant_findings.append(finding)
    total_tests = len(all_findings)
    significant_count = len(significant_findings)
    print(f"Total statistical tests performed: {total_tests}")
    print(f"Significant findings (p < {alpha}): {significant_count}")
    print(f"Percentage significant: {(significant_count/total_tests)*100:.1f}%" if total_tests > 0 else "No tests performed")
    if significant_count == 0:
        print(f"\nNo significant differences found between {GROUP_COL} groups.")
        return {
            'summary_stats': {
                'total_tests': total_tests,
                'significant_count': 0,
                'percentage_significant': 0
            },
            'significant_findings': [],
            'all_findings': all_findings
        }
    print(f"\nSignificant findings (p < {alpha}):")
    significant_summary = []
    for finding in sorted(significant_findings, key=lambda x: x['p_value']):
        variable = finding['variable']
        p_val = finding['p_value']
        test_type = finding['test']
        print(f"\n{variable.upper()}:")
        print(f"  Test: {test_type}")
        print(f"  P-value: {p_val:.6f}")
        desc_stats = results['descriptive_stats']
        if finding['test_type'] == 'numerical':
            print(f"  Finding: Significant difference in {variable} between groups")
            group_summaries = []
            for group, group_stats in desc_stats.items():
                if variable in group_stats and group_stats[variable]['type'] == 'numerical':
                    stats_dict = group_stats[variable]
                    if stats_dict['n_valid'] > 0:
                        mean_val = stats_dict['mean']
                        median_val = stats_dict['median']
                        n_val = stats_dict['n_valid']
                        group_summaries.append({
                            'group': group,
                            'n': n_val,
                            'mean': mean_val,
                            'median': median_val
                        })
                        print(f"    {group}: N={n_val}, Mean={mean_val:.2f}, Median={median_val:.2f}")
            if len(group_summaries) > 1:
                means_sorted = sorted(group_summaries, key=lambda x: x['mean'])
                highest_group = means_sorted[-1]['group']
                lowest_group = means_sorted[0]['group']
                print(f"    Highest mean: {highest_group} ({means_sorted[-1]['mean']:.2f})")
                print(f"    Lowest mean: {lowest_group} ({means_sorted[0]['mean']:.2f})")
        elif finding['test_type'] == 'categorical':
            print(f"  Finding: Significant association between {variable} and {GROUP_COL}")
            if 'contingency_table' in results['statistical_tests'][variable]:
                cont_table = results['statistical_tests'][variable]['contingency_table']
                print(f"    Contingency table:")
                percentages = cont_table.div(cont_table.sum(axis=1), axis=0) * 100
                for group in cont_table.index:
                    print(f"    {group}:")
                    for category in cont_table.columns:
                        count = cont_table.loc[group, category]
                        pct = percentages.loc[group, category]
                        print(f"      {category}: {count} ({pct:.1f}%)")
        significant_summary.append({
            'variable': variable,
            'p_value': p_val,
            'test': test_type,
            'test_type': finding['test_type']
        })
    print(f"\nSummary table:")
    summary_table = pd.DataFrame([
        {
            'Variable': f['variable'],
            'Test Type': f['test_type'].title(),
            'Statistical Test': f['test'],
            'P-value': f'{f["p_value"]:.6f}',
            'Significance': '***' if f['p_value'] < 0.001 else '**' if f['p_value'] < 0.01 else '*'
        }
        for f in sorted(significant_findings, key=lambda x: x['p_value'])
    ])
    print(summary_table.to_string(index=False))
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05")
    print("\nClinical interpretation:")
    numerical_sig = [f for f in significant_findings if f['test_type'] == 'numerical']
    categorical_sig = [f for f in significant_findings if f['test_type'] == 'categorical']
    if numerical_sig:
        print(f"Numerical variables with significant group differences:")
        for f in numerical_sig:
            print(f"  - {f['variable']}: Groups differ significantly (p={f['p_value']:.6f})")
    if categorical_sig:
        print(f"Categorical variables with significant group associations:")
        for f in categorical_sig:
            print(f"  - {f['variable']}: Distribution differs across groups (p={f['p_value']:.6f})")
    return {
        'summary_stats': {
            'total_tests': total_tests,
            'significant_count': significant_count,
            'percentage_significant': (significant_count/total_tests)*100 if total_tests > 0 else 0
        },
        'significant_findings': significant_summary,
        'all_findings': all_findings,
        'summary_table': summary_table if significant_count > 0 else None
    }


def merge_snv_with_chip_and_metadata(snv_df, wl_final_anno_df, meta_df):
    """
    Merge SNV data with CHIP annotations and metadata to create a comprehensive dataset.
    
    Parameters:
        snv_df: SNV dataframe with sample_id column
        wl_final_anno_df: WL CHIP dataframe with Sample and Gene.refGene columns
        meta_df: Metadata dataframe with Patient_ID, Age, Disease Duration columns
    
    Returns:
        Merged dataframe with SNV data, metadata, CHIP annotations, and derived variables
    """
    snv_anno_df = snv_df.merge(meta_df, how='left', left_on='sample_id', right_on=META_ID_COL)
    chip_to_merge = pd.DataFrame()
    chip_to_merge['sample_id'] = wl_final_anno_df['Sample']
    chip_to_merge['wl_CHIP'] = 'WL'
    chip_to_merge['CHIP_gene'] = wl_final_anno_df['Gene.refGene']
    merged_df = snv_anno_df.merge(chip_to_merge, on='sample_id', how='left')
    merged_df['wl_CHIP'] = merged_df['wl_CHIP'].fillna('No_CHIP')
    merged_df['time_since_diagnosis'] = merged_df[AGE_COL] - merged_df[DURATION_COL]
    print(f"Merged dataset shape: {merged_df.shape}")
    print(f"Number of unique samples: {len(merged_df['sample_id'].unique())}")
    return merged_df


def create_snv_anno_ssc_dataframe(snv_df, meta_df, age_bins, age_labels):
    """
    Create annotated SNV dataframe with metadata and age groups for SSc subtype analysis.
    
    Parameters:
        snv_df: SNV dataframe with sample_id column
        meta_df: Metadata dataframe with Patient_ID, Age, SSc Type, statustype columns
        age_bins: Bin edges for age grouping
        age_labels: Labels for age groups
    
    Returns:
        Merged dataframe filtered to SSc Index Cases with age groups
    """
    snv_anno_df = snv_df.merge(meta_df, how='left', left_on='sample_id', right_on=META_ID_COL)
    snv_anno_df['age_group'] = pd.cut(snv_anno_df[AGE_COL], bins=age_bins, labels=age_labels, right=False)
    snv_anno_ssc_df = snv_anno_df[snv_anno_df[STATUS_COL] == CASE_LABEL].copy()
    print(f"Original SNV dataset: {len(snv_df)} samples")
    print(f"After metadata merge: {len(snv_anno_df)} samples")
    print(f"SSc samples only: {len(snv_anno_ssc_df)} samples")
    print(f"\nAge group distribution:")
    print(snv_anno_ssc_df['age_group'].value_counts().sort_index())
    return snv_anno_ssc_df

def calculate_control_age_distribution(meta_df, age_bins, age_labels):
    """
    Calculate the distribution of control samples across age groups.
    
    Parameters:
        meta_df: Metadata dataframe with Age and statustype columns
        age_bins: Bin edges for age grouping
        age_labels: Labels for age groups corresponding to bins
    
    Returns:
        List of control sample counts per age group
    """
    control_samples = meta_df[meta_df[STATUS_COL] == CONTROL_LABEL].copy()
    control_samples['age_group'] = pd.cut(control_samples[AGE_COL], bins=age_bins, labels=age_labels, right=False)
    control_age_counts = control_samples['age_group'].value_counts().sort_index()
    age_bins_normal = []
    for label in age_labels:
        if label in control_age_counts.index:
            age_bins_normal.append(control_age_counts[label])
        else:
            age_bins_normal.append(0)
    print(f"{CONTROL_LABEL} sample distribution by age group:")
    for label, count in zip(age_labels, age_bins_normal):
        print(f"  {label}: {count} samples")
    print(f"Total control samples: {sum(age_bins_normal)}")
    return age_bins_normal

def summarize_disease_duration_by_chip_and_subtype(merged_df, verbose=True):
    """
    Summarize time since diagnosis stratified by CHIP status and SSc subtype.
    
    Parameters:
        merged_df: Dataframe with is_wl, statustype_subtype, and time_since_diagnosis columns
        verbose: Print summary statistics
    
    Returns:
        Dictionary with group names as keys and dict of n, median, mean, std as values
    """
    if verbose:
        print("Checking data...")
        print(f"Total rows: {len(merged_df)}")
        print(f"is_wl value counts:")
        print(merged_df['is_wl'].value_counts().to_string())
        print(f"{GROUP_COL} value counts:")
        print(merged_df[GROUP_COL].value_counts().to_string())
    groups = {
        'chip_neg_lcSSc': (0, SUBTYPE_A),
        'chip_neg_dcSSc': (0, SUBTYPE_B),
        'chip_pos_lcSSc': (1, SUBTYPE_A),
        'chip_pos_dcSSc': (1, SUBTYPE_B)
    }
    results = {}
    for group_name, (chip_status, subtype) in groups.items():
        values = merged_df[
            (merged_df['is_wl'] == chip_status) & 
            (merged_df[GROUP_COL] == subtype)
        ]['time_since_diagnosis'].dropna()
        results[group_name] = {
            'n': len(values),
            'median': values.median() if len(values) > 0 else None,
            'mean': values.mean() if len(values) > 0 else None,
            'std': values.std() if len(values) > 0 else None,
            'values': values
        }
    if verbose:
        print(f"\nDisease duration by CHIP status and subtype:")
        for group_name, stats in results.items():
            if stats['n'] > 0:
                print(f"  {group_name}: n={stats['n']}, median={stats['median']:.1f}, mean={stats['mean']:.1f}")
            else:
                print(f"  {group_name}: n=0")
    return results

def calculate_adjusted_odds_ratio(df, test_column, group_column, group1, group2, adjust_vars=None, verbose=True):
    """
    Calculate adjusted odds ratio using logistic regression with optional covariates.
    
    Parameters:
        df: Dataframe containing all required columns
        test_column: Column name for binary outcome variable
        group_column: Column name for group classification
        group1: Name of first group (coded as 1)
        group2: Name of second group (coded as 0, reference)
        adjust_vars: List of covariates to adjust for (default: ['Age'])
        verbose: Print debug information on error
    
    Returns:
        Dictionary with odds_ratio, ci_lower, ci_upper, p_value, model, figure, formula, and model_df
    """
    
    if adjust_vars is None:
        adjust_vars = [AGE_COL]
    
    model_df = df[df[group_column].isin([group1, group2])].copy()
    model_df['group'] = (model_df[group_column] == group1).astype(int)
    
    if model_df[test_column].dtype == 'object':
        model_df[test_column] = model_df[test_column].map({
            'TRUE': 1, 'FALSE': 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0
        })
    elif model_df[test_column].dtype == bool:
        model_df[test_column] = model_df[test_column].astype(int)
    
    rename_dict = {}
    for var in adjust_vars:
        clean_name = var.strip().replace(' ', '_').replace('-', '_')
        if clean_name != var:
            rename_dict[var] = clean_name
            model_df[clean_name] = model_df[var]
    
    if rename_dict:
        adjust_vars = [rename_dict.get(var, var) for var in adjust_vars]
    
    null_handled_vars = []
    for var in adjust_vars.copy():
        if model_df[var].isnull().any():
            null_handled_vars.append(var)
            null_count = model_df[var].isnull().sum()
            total_count = len(model_df)
            if null_count / total_count < 0.1:
                model_df = model_df.dropna(subset=[var])
            else:
                missing_var = f'{var}_missing'
                model_df[missing_var] = model_df[var].isnull().astype(int)
                if model_df[var].dtype.kind in 'ifc':
                    model_df[var] = model_df[var].fillna(model_df[var].median())
                else:
                    mode_value = model_df[var].mode()[0]
                    model_df[var] = model_df[var].fillna(mode_value)
                adjust_vars.append(missing_var)
    
    formula = f"{test_column} ~ group + " + " + ".join(adjust_vars)
    
    try:
        model = logit(formula, data=model_df).fit(disp=0)
        odds_ratio = np.exp(model.params['group'])
        ci_lower, ci_upper = np.exp(model.conf_int().loc['group'])
        p_value = model.pvalues['group']
        
        plt.figure(figsize=(8, 5))
        plt.errorbar(x=[odds_ratio], y=[0.5], 
                     xerr=[[odds_ratio - ci_lower], [ci_upper - odds_ratio]],
                     fmt='o', color='darkblue', capsize=5, markersize=10)
        plt.axvline(x=1, color='gray', linestyle='--', linewidth=2, label='No Effect (OR=1)')
        
        max_distance = max(abs(odds_ratio - 1), abs(ci_lower - 1), abs(ci_upper - 1))
        padding = max_distance * 0.2
        plot_range = max_distance + padding
        plt.xlim(1 - plot_range, 1 + plot_range)
        
        original_adjust_vars = [var for var in adjust_vars if not var.endswith('_missing')]
        adjust_text = " and ".join(original_adjust_vars)
        plt.title(f'Odds Ratio: {group1} vs {group2}\nAdjusted for {adjust_text}', fontsize=16)
        plt.xlabel('Odds Ratio', fontsize=16)
        plt.yticks([])
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.3)
        
        plt.text(0.02, 0.98, 
                 f'Odds Ratio: {odds_ratio:.2f}\n95% CI: ({ci_lower:.2f}, {ci_upper:.2f})\np-value: {p_value:.4f}',
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 fontsize=16,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return {
            'odds_ratio': odds_ratio,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'model': model,
            'figure': plt.gcf(),
            'handled_missing': null_handled_vars,
            'formula': formula,
            'model_df': model_df,
            'sample_size': len(model_df)
        }
    
    except Exception as e:
        if verbose:
            print(f"Error: {str(e)}")
            print(f"Formula attempted: {formula}")
            print(f"Columns in dataframe: {model_df.columns.tolist()}")
            print(f"Adjustment variables after processing: {adjust_vars}")
            null_counts = model_df[[test_column, 'group'] + adjust_vars].isnull().sum()
            print(f"Null counts in key columns:\n{null_counts}")
        raise

def run_combined_chip_analysis(df, odds_ratio_function, output_filename='ssc_chip_forest_plot.pdf', 
                               figsize=(8, 12), verbose=True):
    """
    Run odds ratio analysis for CHIP across 4 group comparisons and display forest plot.
    
    Parameters:
        df: Merged dataframe with is_wl, statustype, statustype_subtype, Age, gender columns
        odds_ratio_function: Function to calculate adjusted odds ratios (e.g., calculate_adjusted_odds_ratio)
        output_filename: Path for saved figure (set to None to skip saving)
        figsize: Figure size as (width, height)
        verbose: Print summary statistics
    
    Returns:
        Dictionary with results, figure, successful_analyses count, and total_analyses count
    """
    analyses = [
        {'name': f'{CASE_LABEL} vs {CONTROL_LABEL}', 'group_column': STATUS_COL, 'group1': CASE_LABEL, 'group2': CONTROL_LABEL, 'adjust_vars': [AGE_COL, 'gender'], 'color': '#3A4330'},
        {'name': f'{SUBTYPE_B} vs {SUBTYPE_A}', 'group_column': GROUP_COL, 'group1': SUBTYPE_B, 'group2': SUBTYPE_A, 'adjust_vars': [AGE_COL, 'gender'], 'color': '#426737'},
        {'name': f'{SUBTYPE_A} vs {CONTROL_LABEL}', 'group_column': GROUP_COL, 'group1': SUBTYPE_A, 'group2': CONTROL_LABEL, 'adjust_vars': [AGE_COL, 'gender'], 'color': '#75871B'},
        {'name': f'{SUBTYPE_B} vs {CONTROL_LABEL}', 'group_column': GROUP_COL, 'group1': SUBTYPE_B, 'group2': CONTROL_LABEL, 'adjust_vars': [AGE_COL, 'gender'], 'color': '#BAB97D'}
    ]
    
    all_results = {}
    successful_results = []
    
    if verbose:
        print("Running CHIP odds ratio analysis using is_wl column")
        print(f"is_wl distribution: {df['is_wl'].value_counts().to_dict()}")
    
    df_working = df.copy()
    
    for i, analysis in enumerate(analyses):
        if verbose:
            print(f"Analysis {i+1}: {analysis['name']}")
        try:
            if analysis['group_column'] in df_working.columns:
                subset = df_working[df_working[analysis['group_column']].isin([analysis['group1'], analysis['group2']])].copy()
                complete_cases = subset.dropna(subset=['is_wl', analysis['group_column']] + analysis['adjust_vars'])
                if verbose:
                    print(f"  Complete cases: {len(complete_cases)}")
                if len(complete_cases) < 10:
                    if verbose:
                        print(f"  Too few cases - skipping")
                    continue
                result = odds_ratio_function(
                    df=df_working,
                    test_column='is_wl',
                    group_column=analysis['group_column'],
                    group1=analysis['group1'],
                    group2=analysis['group2'],
                    adjust_vars=analysis['adjust_vars']
                )
                if 'figure' in result:
                    plt.close(result['figure'])
                result['analysis_name'] = analysis['name']
                result['color'] = analysis['color']
                all_results[analysis['name']] = result
                successful_results.append(result)
                if verbose:
                    print(f"  OR = {result['odds_ratio']:.2f}, p = {result['p_value']:.4f}")
            else:
                if verbose:
                    print(f"  Column '{analysis['group_column']}' not found")
        except Exception as e:
            if verbose:
                print(f"  Error: {str(e)}")
    
    if not successful_results:
        print("No successful analyses to plot")
        return {'results': all_results, 'figure': None, 'successful_analyses': 0, 'total_analyses': len(analyses)}
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    n_results = len(successful_results)
    y_positions = np.arange(n_results)
    
    odds_ratios = [result['odds_ratio'] for result in successful_results]
    first_result = successful_results[0]
    if 'confidence_interval' in first_result:
        ci_lower = [result['confidence_interval'][0] for result in successful_results]
        ci_upper = [result['confidence_interval'][1] for result in successful_results]
    elif 'ci_lower' in first_result and 'ci_upper' in first_result:
        ci_lower = [result['ci_lower'] for result in successful_results]
        ci_upper = [result['ci_upper'] for result in successful_results]
    elif 'conf_int' in first_result:
        ci_lower = [result['conf_int'][0] for result in successful_results]
        ci_upper = [result['conf_int'][1] for result in successful_results]
    else:
        ci_lower = [max(0.01, or_val - 0.1) for or_val in odds_ratios]
        ci_upper = [or_val + 0.1 for or_val in odds_ratios]
    
    pvalues = [result['p_value'] for result in successful_results]
    labels = [result['analysis_name'] for result in successful_results]
    colors = [result['color'] for result in successful_results]
    
    for i, (or_val, ci_low, ci_high, pval, color) in enumerate(zip(odds_ratios, ci_lower, ci_upper, pvalues, colors)):
        ax.plot([ci_low, ci_high], [i, i], color=color, linewidth=6, alpha=0.7, zorder=2)
        ax.scatter(or_val, i, s=250, color=color, zorder=3, alpha=0.8)
        if pval < 0.05:
            ax.text(or_val, i + 0.15, '*', fontsize=24, ha='center', va='bottom', color='black', fontweight='bold')
    
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=16)
    ax.set_xlabel('Odds Ratio (95% Confidence Interval)', fontsize=16)
    ax.set_title('CHIP Analysis: Odds Ratios Across Group Comparisons\n(Adjusted for Age and Gender)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylim(-0.5, n_results - 0.5)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    save_and_show(output_filename, verbose)
    
    if verbose:
        print("Analysis summary:")
        for i, result in enumerate(successful_results):
            sig = '**' if result['p_value'] < 0.01 else ('*' if result['p_value'] < 0.05 else '')
            sample_size = result.get('sample_size', result.get('n', 'N/A'))
            print(f"  {result['analysis_name']}: OR={result['odds_ratio']:.3f} (95% CI: {ci_lower[i]:.3f}-{ci_upper[i]:.3f}) p={result['p_value']:.4f} {sig}")
        significant_results = [r for r in successful_results if r['p_value'] < 0.05]
        print(f"Significant findings: {len(significant_results)}/{len(successful_results)}")
    
    return {
        'results': all_results,
        'figure': fig,
        'successful_analyses': len(successful_results),
        'total_analyses': len(analyses)
    }

def add_chip_gene_and_deduplicate(wl_final_merged_df, wl_final_anno_df, verbose=True):
    """
    Add CHIP gene information to merged dataframe and deduplicate both dataframes.
    
    Parameters:
        wl_final_merged_df: Merged WL dataframe with sample_id and Sample_ID columns
        wl_final_anno_df: Annotated WL dataframe with Gene.refGene and VAF columns
        verbose: Print summary statistics
    
    Returns:
        tuple: (wl_final_merged_df with CHIP_gene added and deduplicated, 
                wl_final_anno_df deduplicated by variant)
    """
    merged_df = wl_final_merged_df.copy()
    anno_df = wl_final_anno_df.copy()
    
    if verbose:
        print(f"Initial merged_df shape: {merged_df.shape}")
        print(f"Initial anno_df shape: {anno_df.shape}")
    
    merged_df = merged_df.merge(
        anno_df[['sample_id', 'Sample_ID', 'Gene.refGene', 'VAF']], 
        on=['sample_id', 'Sample_ID'], 
        how='left'
    )
    merged_df['CHIP_gene'] = merged_df['Gene.refGene'].copy()
    
    anno_df = anno_df.drop_duplicates(
        subset=['Sample_ID', 'Start', 'Gene.refGene', 'VAF'], 
        keep='first'
    )
    merged_df = merged_df.drop_duplicates(subset=['sample_id'], keep='first')
    
    if verbose:
        print(f"Final merged_df shape: {merged_df.shape}")
        print(f"Final anno_df shape: {anno_df.shape}")
        print(f"Unique samples in merged_df: {merged_df['sample_id'].nunique()}")
        print(f"Unique variants in anno_df: {len(anno_df)}")
    
    return merged_df, anno_df

def analyze_chip_gene_proportion_differences(df, genes_of_interest, verbose=True):
    """
    Test whether CHIP gene proportions differ significantly between status type groups.
    Significant results indicate that certain CHIP genes are enriched or depleted
    in specific disease subtypes relative to others.
    
    Parameters:
        df: Dataframe containing Gene.refGene and statustype_subtype columns
        genes_of_interest: List of gene names to include in the analysis
        verbose: Print summary statistics
    
    Returns:
        Dictionary with proportion_table, individual_tests DataFrame, pairwise_tests DataFrame, and status_totals
    """
    
    df_filtered = df[df['Gene.refGene'].isin(genes_of_interest)]
    if df_filtered.empty:
        print("Warning: No data found for the specified genes of interest")
        return None
    
    contingency_table = pd.crosstab(df_filtered['Gene.refGene'], df_filtered[GROUP_COL])
    status_totals = contingency_table.sum(axis=0)
    proportion_table = contingency_table.div(status_totals, axis=1)
    
    if verbose:
        print(f"Total samples analyzed: {len(df_filtered)}")
        print(f"Number of genes: {len(contingency_table.index)}")
        print(f"Status types: {list(contingency_table.columns)}")
        print("Total variants per status type:")
        for status, total in status_totals.items():
            print(f"  {status}: {total}")
        print("Gene proportions within each status type:")
        print(proportion_table.round(3))
    
    # Individual gene tests: chi-square test for each gene
    # Tests if gene's proportion of total CHIP variants differs across status groups
    individual_results = []
    status_types = list(contingency_table.columns)
    
    for gene in contingency_table.index:
        gene_counts = contingency_table.loc[gene].values
        gene_proportions = proportion_table.loc[gene].values
        total_gene_variants = gene_counts.sum()
        
        if total_gene_variants < 3:
            continue
        
        # Chi-square goodness-of-fit: observed vs expected if gene had equal proportion across groups
        expected_props = total_gene_variants / status_totals.sum()
        expected_counts = status_totals * expected_props
        chi2_stat = np.sum((gene_counts - expected_counts)**2 / expected_counts)
        dof = len(status_types) - 1
        p_val = 1 - stats.chi2.cdf(chi2_stat, dof)
        
        max_prop_idx = np.argmax(gene_proportions)
        dominant_status = status_types[max_prop_idx]
        max_proportion = gene_proportions[max_prop_idx]
        prop_cv = np.std(gene_proportions) / np.mean(gene_proportions) if np.mean(gene_proportions) > 0 else 0
        
        result = {
            'Gene': gene,
            'Total_Variants': total_gene_variants,
            'Proportion_CV': prop_cv,
            'Chi2_Statistic': chi2_stat,
            'P_Value': p_val,
            'Highest_Prop_Status': dominant_status,
            'Highest_Proportion': max_proportion,
            'Significant_Raw': p_val < 0.05
        }
        for i, status in enumerate(status_types):
            result[f'Prop_{status}'] = gene_proportions[i]
        individual_results.append(result)
    
    individual_df = pd.DataFrame(individual_results)
    
    if len(individual_df) > 0:
        # FDR correction for multiple gene comparisons
        rejected, p_corrected, _, _ = multipletests(individual_df['P_Value'], method='fdr_bh')
        individual_df['P_Corrected_FDR'] = p_corrected
        individual_df['Significant_FDR'] = rejected
        individual_df = individual_df.sort_values('P_Corrected_FDR')
        
        if verbose:
            print(f"Individual gene proportion tests ({len(individual_df)} genes):")
            display_cols = ['Gene', 'P_Value', 'P_Corrected_FDR', 'Significant_FDR', 
                           'Highest_Prop_Status', 'Highest_Proportion'] + [f'Prop_{s}' for s in status_types]
            print(individual_df[display_cols].head(10).round(4))
            significant_genes = individual_df[individual_df['Significant_FDR']]['Gene'].tolist()
            print(f"Genes with significantly different proportions (FDR < 0.05): {len(significant_genes)}")
            if significant_genes:
                print(f"  {significant_genes}")
    
    # Pairwise tests: chi-square test between each pair of status groups
    # Tests if gene proportion differs between specific group pairs (e.g., Diffuse vs Limited)
    pairwise_results = []
    if len(status_types) >= 2:
        for gene in contingency_table.index:
            gene_counts = contingency_table.loc[gene]
            total_gene_variants = gene_counts.sum()
            if total_gene_variants < 3:
                continue
            
            gene_result = {'Gene': gene, 'Total_Variants': total_gene_variants}
            for status1, status2 in combinations(status_types, 2):
                count1 = gene_counts[status1]
                count2 = gene_counts[status2]
                total1 = status_totals[status1]
                total2 = status_totals[status2]
                prop1 = count1 / total1 if total1 > 0 else 0
                prop2 = count2 / total2 if total2 > 0 else 0
                
                # Build 2x2 contingency table: rows = gene present/absent, cols = status groups
                obs_table = np.array([
                    [count1, count2],
                    [total1 - count1, total2 - count2]
                ])
                
                if total1 > 0 and total2 > 0 and obs_table.sum() > 0:
                    try:
                        # Use Fisher's exact for small counts, chi-square otherwise
                        if obs_table.min() < 5:
                            _, p_val = fisher_exact(obs_table)
                            chi2_stat = np.nan
                        else:
                            chi2_stat, p_val, _, _ = chi2_contingency(obs_table)
                        
                        gene_result[f'{status1}_vs_{status2}_Prop1'] = prop1
                        gene_result[f'{status1}_vs_{status2}_Prop2'] = prop2
                        gene_result[f'{status1}_vs_{status2}_PropDiff'] = prop1 - prop2
                        gene_result[f'{status1}_vs_{status2}_Chi2'] = chi2_stat
                        gene_result[f'{status1}_vs_{status2}_P'] = p_val
                        gene_result[f'{status1}_vs_{status2}_Sig'] = p_val < 0.05
                    except (ValueError, ZeroDivisionError):
                        gene_result[f'{status1}_vs_{status2}_Prop1'] = prop1
                        gene_result[f'{status1}_vs_{status2}_Prop2'] = prop2
                        gene_result[f'{status1}_vs_{status2}_PropDiff'] = prop1 - prop2
                        gene_result[f'{status1}_vs_{status2}_Chi2'] = np.nan
                        gene_result[f'{status1}_vs_{status2}_P'] = np.nan
                        gene_result[f'{status1}_vs_{status2}_Sig'] = False
                else:
                    gene_result[f'{status1}_vs_{status2}_Prop1'] = prop1
                    gene_result[f'{status1}_vs_{status2}_Prop2'] = prop2
                    gene_result[f'{status1}_vs_{status2}_PropDiff'] = prop1 - prop2
                    gene_result[f'{status1}_vs_{status2}_Chi2'] = np.nan
                    gene_result[f'{status1}_vs_{status2}_P'] = np.nan
                    gene_result[f'{status1}_vs_{status2}_Sig'] = False
            pairwise_results.append(gene_result)
    
    pairwise_df = pd.DataFrame(pairwise_results)
    
    if verbose and len(pairwise_df) > 0:
        comparison_cols = [col for col in pairwise_df.columns if col.endswith('_Sig')]
        if comparison_cols:
            significant_pairs = pairwise_df[pairwise_df[comparison_cols].any(axis=1)]
            if len(significant_pairs) > 0:
                print(f"Genes with significant pairwise proportion differences (p < 0.05):")
                print(significant_pairs[['Gene'] + comparison_cols].head(10))
                prop_diff_cols = [col for col in pairwise_df.columns if '_PropDiff' in col]
                if prop_diff_cols:
                    sig_genes = significant_pairs['Gene'].tolist()
                    print("Proportion differences for significant genes:")
                    print(pairwise_df[pairwise_df['Gene'].isin(sig_genes)][['Gene'] + prop_diff_cols].round(4))
            else:
                print("No significant pairwise proportion differences found (p < 0.05)")
    
    if verbose:
        print("Summary:")
        if len(individual_df) > 0:
            n_sig_individual = sum(individual_df['Significant_FDR'])
            print(f"  Genes with significantly different proportions across groups: {n_sig_individual}/{len(individual_df)} (FDR < 0.05)")
            if n_sig_individual > 0:
                top_genes = individual_df[individual_df['Significant_FDR']].head(3)
                for _, row in top_genes.iterrows():
                    print(f"    {row['Gene']}: highest in {row['Highest_Prop_Status']} ({row['Highest_Proportion']:.3f})")
        if len(pairwise_df) > 0 and comparison_cols:
            n_genes_with_pairwise_sig = sum(pairwise_df[comparison_cols].any(axis=1))
            print(f"  Genes with significant pairwise differences: {n_genes_with_pairwise_sig}/{len(pairwise_df)} (p < 0.05)")
    
    return {
        'proportion_table': proportion_table,
        'individual_tests': individual_df if len(individual_df) > 0 else None,
        'pairwise_tests': pairwise_df if len(pairwise_df) > 0 else None,
        'status_totals': status_totals
    }

def analyze_vaf_distribution_differences(df, analysis_type='gene', genes_of_interest=None,
                                         gene_for_subtype='TET2', min_variants=3, alpha=0.05,
                                         verbose=True):
    """
    Perform statistical tests comparing VAF distributions across genes or subtype groups.
    
    Parameters:
        df: Dataframe with VAF, Gene.refGene, and optionally statustype_subtype columns
        analysis_type: 'gene' to compare genes vs overall CHIP, 'subtype' to compare subtypes for a specific gene
        genes_of_interest: List of genes for gene analysis (required if analysis_type='gene')
        gene_for_subtype: Gene to analyze across subtypes (default: TET2, used if analysis_type='subtype')
        min_variants: Minimum variants required per group
        alpha: Significance level for statistical tests
        verbose: Print summary statistics
    
    Returns:
        Dictionary with test results, including 'stats_df' DataFrame for plot annotations
    """
    
    clean_df = df.dropna(subset=['VAF', 'Gene.refGene']).copy()
    clean_df['VAF'] = pd.to_numeric(clean_df['VAF'], errors='coerce')
    clean_df = clean_df.dropna(subset=['VAF'])
    
    if clean_df['VAF'].max() > 1:
        clean_df['VAF'] = clean_df['VAF'] / 100
    
    if analysis_type == 'gene':
        # Compare each gene's VAF distribution to overall CHIP distribution
        # Uses Mann-Whitney U test (non-parametric test for comparing two distributions)
        if genes_of_interest is None:
            print("Error: genes_of_interest required for gene analysis")
            return None
        
        all_chip_data = clean_df['VAF'].values
        all_chip_mean = np.mean(all_chip_data)
        filtered_df = clean_df[clean_df['Gene.refGene'].isin(genes_of_interest)]
        
        gene_counts = filtered_df['Gene.refGene'].value_counts()
        valid_genes = [g for g in genes_of_interest if g in gene_counts.index and gene_counts[g] >= min_variants]
        
        if len(valid_genes) == 0:
            print(f"Error: No genes have at least {min_variants} variants")
            return None
        
        stats_results = []
        for gene in valid_genes:
            gene_data = filtered_df[filtered_df['Gene.refGene'] == gene]['VAF'].values
            gene_mean = np.mean(gene_data)
            
            # Mann-Whitney U: tests if gene VAF distribution differs from overall CHIP VAF distribution
            # Null hypothesis: the two distributions are identical
            try:
                statistic, p_value = mannwhitneyu(gene_data, all_chip_data, alternative='two-sided')
            except ValueError:
                statistic, p_value = np.nan, 1.0
            
            stats_results.append({
                'Gene': gene,
                'Gene_N': len(gene_data),
                'Gene_Mean_VAF': gene_mean,
                'All_CHIP_Mean_VAF': all_chip_mean,
                'Mean_Difference': gene_mean - all_chip_mean,
                'Mann_Whitney_U': statistic,
                'P_Value': p_value
            })
        
        stats_df = pd.DataFrame(stats_results)
        
        # Benjamini-Hochberg FDR correction: controls false discovery rate across multiple tests
        # Adjusts p-values to account for multiple comparisons
        if len(stats_df) > 1:
            _, p_corrected, _, _ = multipletests(stats_df['P_Value'], method='fdr_bh')
            stats_df['Adjusted_P_Value'] = p_corrected
        else:
            stats_df['Adjusted_P_Value'] = stats_df['P_Value']
        
        stats_df['Significant'] = stats_df['Adjusted_P_Value'] < alpha
        stats_df['Significance_Level'] = stats_df['Adjusted_P_Value'].apply(
            lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        )
        stats_df = stats_df.sort_values('Adjusted_P_Value')
        
        if verbose:
            print("Gene VAF vs All CHIP VAF comparison")
            print(f"Test: Mann-Whitney U (non-parametric)")
            print(f"Correction: Benjamini-Hochberg FDR")
            print(f"Significance level: alpha = {alpha}")
            print(f"Overall CHIP mean VAF: {all_chip_mean:.4f}, n = {len(all_chip_data)}")
            print(stats_df[['Gene', 'Gene_N', 'Gene_Mean_VAF', 'Mean_Difference', 
                           'P_Value', 'Adjusted_P_Value', 'Significance_Level']].round(4).to_string(index=False))
            n_sig = stats_df['Significant'].sum()
            print(f"Significant differences: {n_sig}/{len(stats_df)} genes (FDR < {alpha})")
        
        return {
            'analysis_type': 'gene',
            'stats_df': stats_df,
            'all_chip_mean': all_chip_mean,
            'all_chip_n': len(all_chip_data)
        }
    
    elif analysis_type == 'subtype':
        # Compare VAF distributions across status subtypes for a specific gene
        # Uses Kruskal-Wallis (3+ groups) or Mann-Whitney U (2 groups) for overall test
        # Uses pairwise Mann-Whitney U tests for post-hoc comparisons
        if GROUP_COL not in clean_df.columns:
            print(f"Error: {GROUP_COL} column required for subtype analysis")
            return None
        
        gene_df = clean_df[clean_df['Gene.refGene'] == gene_for_subtype].copy()
        if len(gene_df) == 0:
            print(f"Error: No {gene_for_subtype} variants found")
            return None
        
        subtype_counts = gene_df[GROUP_COL].value_counts()
        desired_order = [CONTROL_LABEL, SUBTYPE_A, SUBTYPE_B]
        valid_subtypes = [s for s in desired_order if s in subtype_counts.index and subtype_counts[s] >= min_variants]
        
        if len(valid_subtypes) < 2:
            print(f"Error: Need at least 2 subtype groups with >= {min_variants} variants")
            return None
        
        subtype_data = {}
        subtype_stats = {}
        for subtype in valid_subtypes:
            vaf_data = gene_df[gene_df[GROUP_COL] == subtype]['VAF'].values
            subtype_data[subtype] = vaf_data
            subtype_stats[subtype] = {'n': len(vaf_data), 'mean': np.mean(vaf_data), 'median': np.median(vaf_data)}
        
        # Overall test: determines if any differences exist across all groups
        # Kruskal-Wallis: non-parametric alternative to one-way ANOVA for 3+ groups
        # Mann-Whitney U: non-parametric test for 2 groups
        if len(valid_subtypes) >= 3:
            vaf_groups = [subtype_data[s] for s in valid_subtypes]
            overall_stat, overall_p = kruskal(*vaf_groups)
            overall_test = 'Kruskal-Wallis'
        else:
            overall_stat, overall_p = mannwhitneyu(
                subtype_data[valid_subtypes[0]], subtype_data[valid_subtypes[1]], alternative='two-sided')
            overall_test = 'Mann-Whitney U'
        
        # Pairwise comparisons: identify which specific groups differ
        # Mann-Whitney U test for each pair of groups
        pairwise_results = []
        
        for s1, s2 in combinations(valid_subtypes, 2):
            try:
                stat, p_val = mannwhitneyu(subtype_data[s1], subtype_data[s2], alternative='two-sided')
                mean_diff = subtype_stats[s1]['mean'] - subtype_stats[s2]['mean']
            except ValueError:
                stat, p_val, mean_diff = np.nan, 1.0, np.nan
            
            pairwise_results.append({
                'Group_1': s1,
                'Group_2': s2,
                'Group_1_N': subtype_stats[s1]['n'],
                'Group_2_N': subtype_stats[s2]['n'],
                'Group_1_Mean': subtype_stats[s1]['mean'],
                'Group_2_Mean': subtype_stats[s2]['mean'],
                'Mean_Difference': mean_diff,
                'Mann_Whitney_U': stat,
                'P_Value': p_val
            })
        
        pairwise_df = pd.DataFrame(pairwise_results)
        
        # Benjamini-Hochberg FDR correction for pairwise comparisons
        if len(pairwise_df) > 1:
            _, p_corrected, _, _ = multipletests(pairwise_df['P_Value'], method='fdr_bh')
            pairwise_df['Adjusted_P_Value'] = p_corrected
        else:
            pairwise_df['Adjusted_P_Value'] = pairwise_df['P_Value']
        
        pairwise_df['Significant'] = pairwise_df['Adjusted_P_Value'] < alpha
        pairwise_df['Significance_Level'] = pairwise_df['Adjusted_P_Value'].apply(
            lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        )
        
        if verbose:
            print(f"{gene_for_subtype} VAF comparison across subtypes")
            print(f"Overall test: {overall_test}")
            print(f"Statistic: {overall_stat:.4f}, p-value: {overall_p:.6f}")
            print(f"Significant overall difference: {'Yes' if overall_p < alpha else 'No'}")
            print(f"\nPairwise comparisons (Mann-Whitney U, FDR corrected):")
            print(pairwise_df[['Group_1', 'Group_2', 'Group_1_N', 'Group_2_N', 
                              'Mean_Difference', 'P_Value', 'Adjusted_P_Value', 
                              'Significance_Level']].round(4).to_string(index=False))
        
        return {
            'analysis_type': 'subtype',
            'gene': gene_for_subtype,
            'overall_test': overall_test,
            'overall_statistic': overall_stat,
            'overall_p_value': overall_p,
            'stats_df': pairwise_df,
            'subtype_stats': subtype_stats
        }
    
    else:
        print(f"Error: Unknown analysis_type '{analysis_type}'. Use 'gene' or 'subtype'")
        return None
    
def analyze_gene_age_by_subtype(df, genes=None,
                                 age_col=None, gene_col='Gene.refGene',
                                 category_col=None, alpha=0.05, verbose=True):
    """
    Perform statistical tests comparing age distributions across subtypes for each gene.
    
    Parameters:
        df: Dataframe with age, gene, and statustype_subtype columns
        genes: List of genes to analyze
        age_col: Column name containing age values
        gene_col: Column name containing gene names
        category_col: Column name containing statustype_subtype categories
        alpha: Significance level for statistical tests
        verbose: Print summary statistics
    
    Returns:
        Dictionary with statistical results for each gene, including group stats,
        overall test (Kruskal-Wallis or Mann-Whitney U), and Diffuse vs Limited comparison
    """
    if genes is None:
        genes = ['DNMT3A', 'TET2', 'ASXL1', 'PPM1D']
    if age_col is None:
        age_col = AGE_COL
    if category_col is None:
        category_col = GROUP_COL
    category_order = [CONTROL_LABEL, SUBTYPE_A, SUBTYPE_B]
    stats_results = {}
    
    for gene in genes:
        gene_data = df[df[gene_col] == gene].copy()
        gene_data = gene_data.dropna(subset=[age_col, category_col])
        
        gene_stats = {
            'n_total': len(gene_data),
            'groups': {},
            'overall_test': {},
            'diffuse_vs_limited': {}
        }
        
        if len(gene_data) == 0:
            stats_results[gene] = gene_stats
            continue
        
        present_categories = [cat for cat in category_order if cat in gene_data[category_col].unique()]
        
        # Descriptive statistics for each group
        for category in present_categories:
            cat_data = gene_data[gene_data[category_col] == category][age_col]
            gene_stats['groups'][category] = {
                'n': len(cat_data),
                'median': cat_data.median(),
                'mean': cat_data.mean(),
                'std': cat_data.std(),
                'q25': cat_data.quantile(0.25),
                'q75': cat_data.quantile(0.75)
            }
        
        # Overall test: Kruskal-Wallis (3+ groups) or Mann-Whitney U (2 groups)
        # Non-parametric tests comparing medians across groups
        if len(present_categories) >= 2:
            group_data = [gene_data[gene_data[category_col] == cat][age_col].values 
                          for cat in present_categories]
            try:
                if len(present_categories) == 2:
                    # Mann-Whitney U: compares two independent samples
                    statistic, p_value = stats.mannwhitneyu(group_data[0], group_data[1], 
                                                            alternative='two-sided')
                    test_name = 'Mann-Whitney U'
                else:
                    # Kruskal-Wallis: non-parametric one-way ANOVA for 3+ groups
                    statistic, p_value = stats.kruskal(*group_data)
                    test_name = 'Kruskal-Wallis'
                
                gene_stats['overall_test'] = {
                    'test_name': test_name,
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'groups_tested': present_categories
                }
            except Exception as e:
                gene_stats['overall_test'] = {'error': str(e)}
        else:
            gene_stats['overall_test'] = {'test_name': 'Not performed', 'reason': 'Only one group present'}
        
        # Pairwise comparison: Diffuse vs Limited
        # Mann-Whitney U test with effect size (rank-biserial correlation)
        if SUBTYPE_B in present_categories and SUBTYPE_A in present_categories:
            try:
                diffuse_data = gene_data[gene_data[category_col] == SUBTYPE_B][age_col].values
                limited_data = gene_data[gene_data[category_col] == SUBTYPE_A][age_col].values
                
                mw_statistic, mw_p_value = stats.mannwhitneyu(diffuse_data, limited_data, 
                                                              alternative='two-sided')
                
                # Rank-biserial correlation: effect size measure for Mann-Whitney U
                # Ranges from -1 to 1, indicates direction and magnitude of difference
                n1, n2 = len(diffuse_data), len(limited_data)
                effect_size = 1 - (2 * mw_statistic) / (n1 * n2)
                
                gene_stats['diffuse_vs_limited'] = {
                    'test_name': 'Mann-Whitney U',
                    'diffuse_n': n1,
                    'limited_n': n2,
                    'diffuse_median': np.median(diffuse_data),
                    'limited_median': np.median(limited_data),
                    'median_difference': np.median(diffuse_data) - np.median(limited_data),
                    'statistic': mw_statistic,
                    'p_value': mw_p_value,
                    'significant': mw_p_value < alpha,
                    'effect_size_rank_biserial': effect_size
                }
            except Exception as e:
                gene_stats['diffuse_vs_limited'] = {'error': str(e)}
        else:
            missing = [g for g in [SUBTYPE_B, SUBTYPE_A] if g not in present_categories]
            gene_stats['diffuse_vs_limited'] = {'test_name': 'Not performed', 
                                                 'reason': f'Missing groups: {", ".join(missing)}'}
        
        stats_results[gene] = gene_stats
    
    if verbose:
        print("Gene age by subtype analysis")
        print(f"Significance level: alpha = {alpha}")
        for gene, results in stats_results.items():
            print(f"\n{gene.upper()}")
            if results['n_total'] == 0:
                print("  No data available")
                continue
            
            print(f"  Total n: {results['n_total']}")
            print("  Group statistics:")
            for group, grp_stats in results['groups'].items():
                print(f"    {group}: n={grp_stats['n']}, median={grp_stats['median']:.1f}, "
                      f"mean={grp_stats['mean']:.1f} +/- {grp_stats['std']:.1f}")
            
            overall = results['overall_test']
            if 'error' not in overall and overall.get('test_name') != 'Not performed':
                sig_marker = '*' if overall['significant'] else 'ns'
                print(f"  Overall test ({overall['test_name']}): "
                      f"stat={overall['statistic']:.3f}, p={overall['p_value']:.4f} {sig_marker}")
            
            dvl = results['diffuse_vs_limited']
            if 'error' not in dvl and dvl.get('test_name') != 'Not performed':
                sig_marker = '*' if dvl['significant'] else 'ns'
                print(f"  {SUBTYPE_B} vs {SUBTYPE_A}: median diff={dvl['median_difference']:.1f} years, "
                      f"p={dvl['p_value']:.4f} {sig_marker}, effect size={dvl['effect_size_rank_biserial']:.3f}")
    
    return stats_results

def preprocess_clinical_tertiles(df, column_name, prefix=None, verbose=True):
    """
    Create tertile classifications for a clinical variable (DLCO, FVC, MRSS, etc.).
    Tertiles divide the data into three equal-sized groups based on the value distribution.
    
    Parameters:
        df: Dataframe containing the clinical variable
        column_name: Name of column to create tertiles for (e.g., 'DLCO %Predicted', 'FVC %predicted', 'MRSS')
        prefix: Prefix for new column names (default: derived from column_name)
        verbose: Print summary statistics
    
    Returns:
        tuple: (modified dataframe, tertile_info dictionary with boundaries and counts)
    """
    if column_name not in df.columns:
        raise ValueError(f"Required column '{column_name}' not found in dataframe")
    
    if prefix is None:
        prefix = column_name.split()[0].replace('%', '').replace(' ', '_')
    
    first_col = f'{prefix}_first_tertile'
    second_col = f'{prefix}_second_tertile'
    third_col = f'{prefix}_third_tertile'
    
    df[first_col] = 0
    df[second_col] = 0
    df[third_col] = 0
    
    original_type = df[column_name].dtype
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    
    valid_mask = df[column_name].notna()
    n_valid = valid_mask.sum()
    
    tertile_info = {
        'column': column_name,
        'prefix': prefix,
        'n_valid': n_valid,
        'n_missing': (~valid_mask).sum(),
        'boundaries': [None, None],
        'counts': {'first': 0, 'second': 0, 'third': 0}
    }
    
    if n_valid == 0:
        if verbose:
            print(f"Warning: No valid {column_name} values found")
        return df, tertile_info
    
    valid_values = df.loc[valid_mask, column_name]
    
    # Calculate tertile boundaries using 33rd and 67th percentiles
    # This divides data into three roughly equal-sized groups
    tertile_boundaries = valid_values.quantile([1/3, 2/3]).values
    tertile_info['boundaries'] = tertile_boundaries.tolist()
    
    # First tertile: lowest 33% of values
    first_condition = (df[column_name] <= tertile_boundaries[0]) & valid_mask
    df.loc[first_condition, first_col] = 1
    tertile_info['counts']['first'] = first_condition.sum()
    
    # Second tertile: middle 33% of values
    second_condition = ((df[column_name] > tertile_boundaries[0]) & 
                        (df[column_name] <= tertile_boundaries[1]) & valid_mask)
    df.loc[second_condition, second_col] = 1
    tertile_info['counts']['second'] = second_condition.sum()
    
    # Third tertile: highest 33% of values
    third_condition = (df[column_name] > tertile_boundaries[1]) & valid_mask
    df.loc[third_condition, third_col] = 1
    tertile_info['counts']['third'] = third_condition.sum()
    
    if verbose:
        print(f"{column_name} tertile preprocessing:")
        print(f"  Valid values: {n_valid}, Missing: {tertile_info['n_missing']}")
        print(f"  Boundaries: first <= {tertile_boundaries[0]:.2f}, "
              f"second <= {tertile_boundaries[1]:.2f}, third > {tertile_boundaries[1]:.2f}")
        print(f"  Counts: first={tertile_info['counts']['first']}, "
              f"second={tertile_info['counts']['second']}, third={tertile_info['counts']['third']}")
        print(f"  Added columns: {first_col}, {second_col}, {third_col}")
    
    return df, tertile_info


def preprocess_dlco_binary(df, low_threshold=75.0, bad_threshold=40.0, verbose=True):
    """
    Create binary DLCO classification variables based on clinical thresholds.
    
    Parameters:
        df: Dataframe containing DLCO %Predicted column
        low_threshold: Threshold for low DLCO classification (default: 75.0)
        bad_threshold: Threshold for bad/severe DLCO classification (default: 40.0)
        verbose: Print summary statistics
    
    Returns:
        tuple: (modified dataframe, classification_info dictionary)
    """
    if DLCO_PCT_COL not in df.columns:
        raise ValueError(f"Required column '{DLCO_PCT_COL}' not found in dataframe")
    
    df['DLCO_pred_low'] = 0
    df['DLCO_pred_bad'] = 0
    
    df[DLCO_PCT_COL] = pd.to_numeric(df[DLCO_PCT_COL], errors='coerce')
    
    valid_mask = df[DLCO_PCT_COL].notna()
    n_valid = valid_mask.sum()
    
    # Low DLCO: values below the low threshold (impaired diffusion capacity)
    condition_low = (df[DLCO_PCT_COL] < low_threshold) & valid_mask
    df.loc[condition_low, 'DLCO_pred_low'] = 1
    
    # Bad/severe DLCO: values below the bad threshold (severely impaired)
    condition_bad = (df[DLCO_PCT_COL] < bad_threshold) & valid_mask
    df.loc[condition_bad, 'DLCO_pred_bad'] = 1
    
    classification_info = {
        'n_valid': n_valid,
        'n_missing': (~valid_mask).sum(),
        'low_threshold': low_threshold,
        'bad_threshold': bad_threshold,
        'n_low': condition_low.sum(),
        'n_bad': condition_bad.sum(),
        'n_normal': (valid_mask & ~condition_low).sum()
    }
    
    if verbose:
        print("DLCO binary classification:")
        print(f"  Valid values: {n_valid}, Missing: {classification_info['n_missing']}")
        print(f"  Normal (>= {low_threshold}): {classification_info['n_normal']}")
        print(f"  Low (< {low_threshold}): {classification_info['n_low']}")
        print(f"  Bad/Severe (< {bad_threshold}): {classification_info['n_bad']}")
        print(f"  Added columns: DLCO_pred_low, DLCO_pred_bad")
    
    return df, classification_info


def standardize_column_names(df, verbose=True):
    """
    Standardize column names by removing trailing spaces.
    
    Parameters:
        df: Dataframe to standardize
        verbose: Print changes made
    
    Returns:
        Modified dataframe with standardized column names
    """
    renamed = {}
    for col in df.columns:
        if col != col.strip():
            renamed[col] = col.strip()
    
    if renamed:
        df.rename(columns=renamed, inplace=True)
        if verbose:
            for old, new in renamed.items():
                print(f"  Renamed '{old}' to '{new}'")
    elif verbose:
        print("  No column names needed standardization")
    
    return df

def summarize_clinical_tertiles(df, column_name, prefix=None, verbose=True):
    """
    Generate descriptive statistics and distribution summary for tertile-classified clinical variables.
    
    Parameters:
        df: Dataframe with tertile columns created by preprocess_clinical_tertiles
        column_name: Name of the original clinical variable column
        prefix: Prefix used for tertile columns (default: derived from column_name)
        verbose: Print detailed summary
    
    Returns:
        Dictionary with descriptive stats, tertile stats, and distribution info
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in dataframe")
    
    if prefix is None:
        prefix = column_name.split()[0].replace('%', '').replace(' ', '_')
    
    first_col = f'{prefix}_first_tertile'
    second_col = f'{prefix}_second_tertile'
    third_col = f'{prefix}_third_tertile'
    
    for col in [first_col, second_col, third_col]:
        if col not in df.columns:
            raise ValueError(f"Tertile column '{col}' not found. Run preprocess_clinical_tertiles first.")
    
    valid_mask = df[column_name].notna()
    
    # Overall descriptive statistics
    overall_stats = df[column_name].describe().to_dict()
    
    # Tertile-specific statistics
    tertile_stats = {}
    tertile_names = ['first', 'second', 'third']
    tertile_cols = [first_col, second_col, third_col]
    
    for name, col in zip(tertile_names, tertile_cols):
        mask = df[col] == 1
        if mask.sum() > 0:
            values = df.loc[mask, column_name]
            tertile_stats[name] = {
                'n': mask.sum(),
                'mean': values.mean(),
                'median': values.median(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max()
            }
        else:
            tertile_stats[name] = {'n': 0}
    
    # Verify mutual exclusivity (each patient should be in exactly one tertile or none)
    tertile_sum = df[first_col] + df[second_col] + df[third_col]
    n_multiple = (tertile_sum > 1).sum()
    n_none = ((tertile_sum == 0) & valid_mask).sum()
    
    results = {
        'column': column_name,
        'overall_stats': overall_stats,
        'tertile_stats': tertile_stats,
        'n_valid': valid_mask.sum(),
        'n_missing': (~valid_mask).sum(),
        'n_multiple_tertiles': n_multiple,
        'n_unassigned': n_none
    }
    
    if verbose:
        print(f"\n{column_name} tertile summary:")
        print(f"Overall distribution (n={results['n_valid']}):")
        print(f"  Mean: {overall_stats['mean']:.2f}, Std: {overall_stats['std']:.2f}")
        print(f"  Median: {overall_stats['50%']:.2f}")
        print(f"  Range: {overall_stats['min']:.2f} - {overall_stats['max']:.2f}")
        print(f"  IQR: {overall_stats['25%']:.2f} - {overall_stats['75%']:.2f}")
        
        print(f"\nBy tertile:")
        for name in tertile_names:
            stats = tertile_stats[name]
            if stats['n'] > 0:
                print(f"  {name.capitalize()} (n={stats['n']}): "
                      f"mean={stats['mean']:.2f}, range={stats['min']:.2f}-{stats['max']:.2f}")
            else:
                print(f"  {name.capitalize()}: no patients")
        
        if n_multiple > 0:
            print(f"\nWarning: {n_multiple} patients assigned to multiple tertiles")
        if n_none > 0:
            print(f"Note: {n_none} patients with valid values not assigned to any tertile")
    
    return results


def analyze_tertiles_by_chip_status(df, column_name, prefix=None, chip_col='is_wl', verbose=True):
    """
    Compare tertile distributions between CHIP-positive and CHIP-negative patients.
    Uses chi-square test to assess association between CHIP status and tertile membership.
    
    Parameters:
        df: Dataframe with tertile columns and CHIP status
        column_name: Name of the original clinical variable column
        prefix: Prefix used for tertile columns (default: derived from column_name)
        chip_col: Column name for CHIP status (default: 'is_wl')
        verbose: Print detailed summary
    
    Returns:
        Dictionary with contingency table, chi-square test results, and group statistics
    """
    if prefix is None:
        prefix = column_name.split()[0].replace('%', '').replace(' ', '_')
    
    first_col = f'{prefix}_first_tertile'
    second_col = f'{prefix}_second_tertile'
    third_col = f'{prefix}_third_tertile'
    
    for col in [first_col, second_col, third_col, chip_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
    
    # Create single tertile column for analysis
    df_analysis = df.copy()
    df_analysis['tertile'] = None
    df_analysis.loc[df_analysis[first_col] == 1, 'tertile'] = 'First'
    df_analysis.loc[df_analysis[second_col] == 1, 'tertile'] = 'Second'
    df_analysis.loc[df_analysis[third_col] == 1, 'tertile'] = 'Third'
    
    # Filter to patients with both tertile assignment and CHIP status
    analysis_mask = df_analysis['tertile'].notna() & df_analysis[chip_col].notna()
    df_filtered = df_analysis[analysis_mask]
    
    # Create contingency table
    contingency = pd.crosstab(df_filtered[chip_col], df_filtered['tertile'], 
                              margins=True, margins_name='Total')
    
    # Chi-square test: tests if CHIP status is independent of tertile membership
    # Significant result indicates CHIP status is associated with clinical severity
    contingency_no_margins = pd.crosstab(df_filtered[chip_col], df_filtered['tertile'])
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_no_margins)
    
    # Calculate proportions within each CHIP group
    proportions = contingency_no_margins.div(contingency_no_margins.sum(axis=1), axis=0)
    
    results = {
        'column': column_name,
        'contingency_table': contingency,
        'proportions': proportions,
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'expected_counts': expected,
        'significant': p_value < 0.05,
        'n_analyzed': len(df_filtered)
    }
    
    if verbose:
        print(f"\n{column_name} tertiles by CHIP status:")
        print(f"Chi-square test: statistic={chi2_stat:.3f}, p={p_value:.4f}")
        print(f"Result: {'Significant association' if p_value < 0.05 else 'No significant association'} "
              f"between CHIP status and {column_name} tertile")
        
        print(f"\nContingency table:")
        print(contingency)
        
        print(f"\nProportions within CHIP groups:")
        print(proportions.round(3))
        
        # Interpretation
        if p_value < 0.05:
            chip_pos_first = proportions.loc[1, 'First'] if 1 in proportions.index else 0
            chip_neg_first = proportions.loc[0, 'First'] if 0 in proportions.index else 0
            if chip_pos_first > chip_neg_first:
                print(f"\nCHIP-positive patients more likely to be in first (lowest) tertile")
            else:
                print(f"\nCHIP-negative patients more likely to be in first (lowest) tertile")
    
    return results

def prepare_ana_data(df):
    """
    Prepare data for antibody logistic regression analysis.
    Filters to Limited/Diffuse cases and converts antibody variables to binary format.
    
    Parameters:
        df: DataFrame with antibody and clinical data
    
    Returns:
        analysis_df: DataFrame filtered and with binary columns
        binary_cols: list of binary antibody column names
    """
    filtered_df = df[df[GROUP_COL].isin([SUBTYPE_A, SUBTYPE_B])].copy()
    
    antibody_cols = ['ana', 'anti-centromere', 'anti-topo', 'anti-RNA polymerase III']
    
    for col in antibody_cols:
        if col in filtered_df.columns:
            filtered_df[f'{col}_binary'] = filtered_df[col].apply(
                lambda x: 1 if x in [1, 1.0] else (0 if x in [2, 2.0] else np.nan)
            )
        else:
            print(f"Warning: Column '{col}' not found in dataset")
    
    filtered_df['is_wl_binary'] = filtered_df['is_wl'].apply(
        lambda x: 1 if x in [1, 1.0] else 0
    )
    
    binary_cols = [f'{col}_binary' for col in antibody_cols if f'{col}_binary' in filtered_df.columns]
    
    required_cols = ['is_wl_binary', AGE_COL]
    analysis_df = filtered_df.dropna(subset=required_cols)
    
    print(f"Original dataset: {len(df)} rows")
    print(f"After filtering for {SUBTYPE_A}/{SUBTYPE_B}: {len(filtered_df)} rows")
    print(f"After removing missing values in Age/is_wl: {len(analysis_df)} rows")
    print(f"\nSubtype distribution:")
    print(analysis_df[GROUP_COL].value_counts())
    print(f"\nis_wl positive distribution:")
    print(analysis_df['is_wl_binary'].value_counts())
    
    for col in binary_cols:
        valid_data = analysis_df[col].dropna()
        if len(valid_data) > 0:
            print(f"\n{col.replace('_binary', '')} distribution (n={len(valid_data)}):")
            print(valid_data.value_counts())
    
    return analysis_df, binary_cols


def ana_logistic_regression(df, subtype, antibody_col):
    """
    Run logistic regression for a specific subtype and antibody type.
    Model: logit(P(is_wl = 1)) = B0 + B1(antibody_binary) + B2(Age)
    
    Parameters:
        df: DataFrame containing prepared data
        subtype: 'Limited', 'Diffuse', or 'Combined'
        antibody_col: binary antibody column to use as predictor
    
    Returns:
        dict with odds ratio, CI, p-value, and sample info; None if model fails
    """
    if subtype == 'Combined':
        subtype_df = df[df[GROUP_COL].isin([SUBTYPE_A, SUBTYPE_B])].copy()
        display_name = 'Combined'
    else:
        subtype_df = df[df[GROUP_COL] == subtype].copy()
        display_name = subtype
    
    subtype_df = subtype_df.dropna(subset=[antibody_col, AGE_COL])
    
    if len(subtype_df) < 10:
        print(f"Warning: Small sample size for {display_name} - {antibody_col} (n={len(subtype_df)})")
        return None
    
    if subtype_df[antibody_col].nunique() < 2:
        print(f"Warning: No variation in {antibody_col} for {display_name}")
        return None
    
    X = subtype_df[[antibody_col, AGE_COL]]
    y = subtype_df['is_wl_binary']
    X = sm.add_constant(X)
    
    try:
        model = sm.Logit(y, X)
        results = model.fit(disp=0)
        
        coef = results.params[antibody_col]
        pvalue = results.pvalues[antibody_col]
        conf_int = results.conf_int().loc[antibody_col]
        
        odds_ratio = np.exp(coef)
        or_ci_lower = np.exp(conf_int[0])
        or_ci_upper = np.exp(conf_int[1])
        
        antibody_name = antibody_col.replace('_binary', '')
        
        print(f"\n{display_name} - {antibody_name} Results:")
        print(f"Sample size: {len(subtype_df)}, Positive cases: {int(subtype_df[antibody_col].sum())}")
        print(f"Odds Ratio: {odds_ratio:.3f} (95% CI: {or_ci_lower:.3f}-{or_ci_upper:.3f}), P-value: {pvalue:.4f}")
        
        return {
            'subtype': subtype,
            'display_name': display_name,
            'antibody': antibody_name,
            'antibody_col': antibody_col,
            'n': len(subtype_df),
            'n_positive': int(subtype_df[antibody_col].sum()),
            'odds_ratio': odds_ratio,
            'ci_lower': or_ci_lower,
            'ci_upper': or_ci_upper,
            'pvalue': pvalue,
            'coef': coef,
            'model_summary': results.summary()
        }
        
    except Exception as e:
        print(f"Error fitting model for {display_name} - {antibody_col}: {str(e)}")
        return None


def run_ana_analysis(df):
    """
    Run antibody analysis pipeline: data prep, logistic regression, and results summary.
    Analyzes Limited, Diffuse, and Combined groups for each antibody type.
    
    Parameters:
        df: DataFrame with antibody and clinical data
    
    Returns:
        list of result dictionaries from logistic regression
    """
    print("CHIP Analysis: Antibody Status Predicting is_wl")
    print("Adjusting for Age as covariate")
    print(f"Analyzing {SUBTYPE_A}, {SUBTYPE_B}, and Combined groups\n")
    
    if 'sample_id' in df.columns:
        original_rows = len(df)
        df = df.drop_duplicates(subset=['sample_id'], keep='first')
        final_rows = len(df)
        duplicates_removed = original_rows - final_rows
        
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate sample_id rows (kept first occurrence)")
        print(f"Analyzing {final_rows} unique samples\n")
    
    print("Step 1: Preparing data...")
    analysis_df, binary_cols = prepare_ana_data(df)
    
    if not binary_cols:
        print("No valid antibody columns found for analysis")
        return []
    
    print("\nStep 2: Running logistic regression analyses...")
    results = []
    analysis_groups = [SUBTYPE_A, SUBTYPE_B, 'Combined']
    
    for antibody_col in binary_cols:
        antibody_name = antibody_col.replace('_binary', '')
        print(f"\n--- Analyzing {antibody_name} ---")
        
        for subtype in analysis_groups:
            result = ana_logistic_regression(analysis_df, subtype, antibody_col)
            if result:
                results.append(result)
    
    if results:
        print_ana_results_summary(results)
    else:
        print("No valid results to analyze")
    
    return results


def print_ana_results_summary(results):
    """
    Print summary of antibody analysis results.
    
    Parameters:
        results: list of result dictionaries from ana_logistic_regression
    """
    
    print("\nAnalysis Summary")
    print("Format: Antibody - Group: OR (95% CI) p-value")
    
    antibody_results = defaultdict(list)
    for result in results:
        antibody_results[result['antibody']].append(result)
    
    subtype_order = [SUBTYPE_A, SUBTYPE_B, 'Combined']
    
    for antibody in sorted(antibody_results.keys()):
        antibody_group = antibody_results[antibody]
        print(f"\n{antibody.upper()}:")
        
        antibody_group.sort(key=lambda x: subtype_order.index(x['subtype']) if x['subtype'] in subtype_order else 999)
        
        for result in antibody_group:
            significance = "**" if result['pvalue'] < 0.01 else ("*" if result['pvalue'] < 0.05 else "")
            print(f"  {result['display_name']:<25}: OR={result['odds_ratio']:.3f} "
                  f"(95% CI: {result['ci_lower']:.3f}-{result['ci_upper']:.3f}) "
                  f"p={result['pvalue']:.4f} {significance}")
            print(f"{'':27}  n={result['n']}, {result['n_positive']} positive")
    
    significant_results = [r for r in results if r['pvalue'] < 0.05]
    if significant_results:
        print(f"\nSignificant findings (p < 0.05):")
        for result in significant_results:
            direction = "increased" if result['odds_ratio'] > 1 else "decreased"
            print(f"  {result['antibody']} in {result['display_name']}: {direction} odds of is_wl=1")
            print(f"    OR={result['odds_ratio']:.3f} (95% CI: {result['ci_lower']:.3f}-{result['ci_upper']:.3f}), p={result['pvalue']:.4f}")
    else:
        print(f"\nNo statistically significant findings (all p >= 0.05)")
    
    print(f"\nAnalysis statistics:")
    print(f"Total analyses completed: {len(results)}")
    print(f"Antibodies tested: {len(set(r['antibody'] for r in results))}")
    print(f"Groups analyzed: {len(set(r['subtype'] for r in results))}")
    print(f"Significant findings: {len(significant_results)} ({len(significant_results)/len(results)*100:.1f}%)")

def prepare_chip_tet2_groups(df):
    """
    Prepare dataframe with CHIP/TET2 group assignments for analysis.
    Creates six groups: Non-CHIP/CHIP/TET2 for both Limited and Diffuse subtypes.
    
    Parameters:
        df: DataFrame with is_wl, statustype_subtype, and CHIP_gene columns
    
    Returns:
        plot_df: DataFrame with 'group' column assigned
        group_order: list of group names in plotting order
    """
    plot_df = df.copy()
    plot_df['chip_status'] = plot_df['is_wl'].map({0: 'Non-CHIP', 1: 'CHIP'})
    plot_df['subtype'] = plot_df[GROUP_COL].str.extract(rf'({SUBTYPE_A}|{SUBTYPE_B})', expand=False)
    plot_df = plot_df[plot_df['subtype'].isin([SUBTYPE_A, SUBTYPE_B])]
    
    def assign_group(row):
        if pd.isna(row['CHIP_gene']) or row['CHIP_gene'] != 'TET2':
            return row['chip_status'] + ' ' + row['subtype']
        else:
            return row['subtype'] + ' TET2'
    
    plot_df['group'] = plot_df.apply(assign_group, axis=1)
    
    group_order = [f'Non-CHIP {SUBTYPE_A}', f'CHIP {SUBTYPE_A}', f'{SUBTYPE_A} TET2', 
                   f'Non-CHIP {SUBTYPE_B}', f'CHIP {SUBTYPE_B}', f'{SUBTYPE_B} TET2']
    
    plot_df = plot_df[plot_df['group'].isin(group_order)]
    
    return plot_df, group_order


def compute_chip_tet2_group_statistics(df, metrics=None, alpha=0.05):
    """
    Compute Kruskal-Wallis and pairwise Mann-Whitney U tests for CHIP/TET2 groups.
    
    Parameters:
        df: DataFrame with clinical metrics and group assignments
        metrics: list of metric column names (default: DLCO, FVC, MRSS)
        alpha: significance level for FDR correction
    
    Returns:
        stats_results: dict with test results for each metric
        plot_df: prepared DataFrame with group assignments
        group_order: list of group names
    """
    
    if metrics is None:
        metrics = [DLCO_PCT_COL, FVC_PCT_COL, SKIN_SCORE_COL]
    
    plot_df, group_order = prepare_chip_tet2_groups(df)
    stats_results = {}
    
    for metric in metrics:
        print(f"\nStatistical analysis for {metric}")
        
        metric_df = plot_df.dropna(subset=[metric])
        
        if metric_df.empty:
            stats_results[metric] = {'error': 'No data available'}
            continue
        
        group_data = {}
        for group in group_order:
            group_subset = metric_df[metric_df['group'] == group][metric].dropna()
            if len(group_subset) > 0:
                group_data[group] = group_subset
                print(f"{group}: n={len(group_subset)}, median={group_subset.median():.2f} "
                      f"(IQR: {group_subset.quantile(0.25):.2f}-{group_subset.quantile(0.75):.2f})")
        
        if len(group_data) < 2:
            stats_results[metric] = {'error': 'Insufficient groups for comparison'}
            continue
        
        kw_statistic, kw_pvalue = stats.kruskal(*group_data.values())
        print(f"\nKruskal-Wallis test: H={kw_statistic:.3f}, p={kw_pvalue:.4f}")
        
        stats_results[metric] = {
            'kruskal_wallis': {'statistic': kw_statistic, 'p_value': kw_pvalue},
            'group_medians': {group: data.median() for group, data in group_data.items()},
            'group_sizes': {group: len(data) for group, data in group_data.items()},
            'pairwise_comparisons': {}
        }
        
        if kw_pvalue < alpha:
            print(f"Kruskal-Wallis significant (p < {alpha}), performing pairwise Mann-Whitney U tests")
            
            group_names = list(group_data.keys())
            pvalues = []
            comparison_info = []
            
            for group1, group2 in combinations(group_names, 2):
                mw_statistic, mw_pvalue = stats.mannwhitneyu(
                    group_data[group1], group_data[group2], alternative='two-sided')
                
                group1_median = group_data[group1].median()
                group2_median = group_data[group2].median()
                lower_group = group1 if group1_median < group2_median else group2
                
                pvalues.append(mw_pvalue)
                comparison_info.append({
                    'group1': group1, 'group2': group2,
                    'statistic': mw_statistic, 'p_value': mw_pvalue,
                    'lower_median_group': lower_group
                })
            
            rejected, pvals_corrected, _, _ = multipletests(pvalues, alpha=alpha, method='fdr_bh')
            
            print(f"BH FDR correction applied ({len(pvalues)} comparisons)")
            
            for i, info in enumerate(comparison_info):
                comparison_key = f"{info['group1']}_vs_{info['group2']}"
                stats_results[metric]['pairwise_comparisons'][comparison_key] = {
                    'statistic': info['statistic'],
                    'p_value': info['p_value'],
                    'p_value_corrected': pvals_corrected[i],
                    'significant_fdr': rejected[i],
                    'lower_median_group': info['lower_median_group']
                }
                
                sig_marker = "***" if rejected[i] else ""
                print(f"  {info['group1']} vs {info['group2']}: U={info['statistic']:.1f}, "
                      f"p={info['p_value']:.4f}, p_adj={pvals_corrected[i]:.4f} {sig_marker}")
        else:
            print(f"Kruskal-Wallis not significant (p >= {alpha})")
    
    return stats_results, plot_df, group_order
    
    
def test_associations_flexible_with_direction_no_significance(df, variables_to_test, group1_mask, group2_mask, 
                                                              comparison_name, group1_name, group2_name):
    """
    Test associations between two groups without significance interpretation.
    Significance determined after FDR correction.
    
    Parameters:
        df: DataFrame with data
        variables_to_test: list of variable names to test
        group1_mask, group2_mask: boolean masks for each group
        comparison_name: descriptive name for this comparison
        group1_name, group2_name: names for interpretation
    
    Returns:
        DataFrame with test results (without significance interpretation)
    """
    analysis_df = df.copy()
    analysis_df.columns = analysis_df.columns.str.strip()
    
    for var in variables_to_test:
        var_clean = var.strip()
        if var_clean in analysis_df.columns:
            if analysis_df[var_clean].dtype == 'object':
                analysis_df[var_clean] = analysis_df[var_clean].astype(str).str.strip()
    
    group1_data = analysis_df[group1_mask]
    group2_data = analysis_df[group2_mask]
    
    print(f"\n{comparison_name}")
    print(f"{group1_name} N: {len(group1_data)}")
    print(f"{group2_name} N: {len(group2_data)}")
    
    results = []
    
    for variable in variables_to_test:
        var_clean = variable.strip()
        
        if var_clean not in analysis_df.columns:
            print(f"Warning: {var_clean} not found in dataframe")
            continue
        
        group1_var = group1_data[var_clean].dropna()
        group2_var = group2_data[var_clean].dropna()
        
        if len(group1_var) == 0 or len(group2_var) == 0:
            results.append({
                'Comparison': comparison_name, 'Variable': var_clean, 'Test_Type': 'Insufficient_Data',
                'Group1_Name': group1_name, 'Group2_Name': group2_name,
                'Group1_N': len(group1_var), 'Group2_N': len(group2_var),
                'Test_Statistic': np.nan, 'P_Value': np.nan, 'Effect_Size': np.nan,
                'Group1_Summary': 'No data', 'Group2_Summary': 'No data',
                'Group1_Mean': np.nan, 'Group2_Mean': np.nan, 'Mean_Difference': np.nan
            })
            continue
        
        is_numeric = pd.api.types.is_numeric_dtype(analysis_df[var_clean])
        
        if is_numeric:
            g1_mean, g1_std, g1_median = group1_var.mean(), group1_var.std(), group1_var.median()
            g2_mean, g2_std, g2_median = group2_var.mean(), group2_var.std(), group2_var.median()
            mean_diff = g1_mean - g2_mean
            
            try:
                statistic, p_value = mannwhitneyu(group1_var, group2_var, alternative='two-sided')
                test_type = 'Mann-Whitney U'
                effect_size = 1 - (2 * statistic) / (len(group1_var) * len(group2_var))
            except Exception:
                try:
                    statistic, p_value = ttest_ind(group1_var, group2_var, equal_var=False)
                    test_type = 'Welch t-test'
                    pooled_std = np.sqrt(((len(group1_var) - 1) * g1_std**2 + 
                                         (len(group2_var) - 1) * g2_std**2) / 
                                        (len(group1_var) + len(group2_var) - 2))
                    effect_size = (g1_mean - g2_mean) / pooled_std if pooled_std > 0 else np.nan
                except Exception:
                    statistic, p_value, effect_size = np.nan, np.nan, np.nan
                    test_type = 'Test_Failed'
            
            g1_summary = f"Mean+/-SD: {g1_mean:.2f}+/-{g1_std:.2f}, Median: {g1_median:.2f}"
            g2_summary = f"Mean+/-SD: {g2_mean:.2f}+/-{g2_std:.2f}, Median: {g2_median:.2f}"
            
        else:
            combined_data = pd.concat([
                pd.DataFrame({'group': 'Group1', var_clean: group1_var}),
                pd.DataFrame({'group': 'Group2', var_clean: group2_var})
            ])
            contingency_table = pd.crosstab(combined_data['group'], combined_data[var_clean], 
                                           dropna=False, margins=False)
            
            g1_props = (contingency_table.loc['Group1'] / contingency_table.loc['Group1'].sum() * 100).round(1) if 'Group1' in contingency_table.index else pd.Series(dtype=float)
            g2_props = (contingency_table.loc['Group2'] / contingency_table.loc['Group2'].sum() * 100).round(1) if 'Group2' in contingency_table.index else pd.Series(dtype=float)
            
            if contingency_table.shape == (2, 2):
                if (contingency_table < 5).any().any():
                    try:
                        statistic, p_value = fisher_exact(contingency_table)
                        test_type = 'Fisher Exact'
                        effect_size = statistic
                    except Exception:
                        statistic, p_value, effect_size = np.nan, np.nan, np.nan
                        test_type = 'Test_Failed'
                else:
                    try:
                        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                        statistic = chi2_stat
                        test_type = 'Chi-square'
                        n = contingency_table.sum().sum()
                        effect_size = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
                    except Exception:
                        statistic, p_value, effect_size = np.nan, np.nan, np.nan
                        test_type = 'Test_Failed'
            else:
                try:
                    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                    statistic = chi2_stat
                    test_type = 'Chi-square'
                    n = contingency_table.sum().sum()
                    effect_size = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
                except Exception:
                    statistic, p_value, effect_size = np.nan, np.nan, np.nan
                    test_type = 'Test_Failed'
            
            g1_summary = "; ".join([f"{cat}: {prop}%" for cat, prop in g1_props.items()]) if len(g1_props) > 0 else "No data"
            g2_summary = "; ".join([f"{cat}: {prop}%" for cat, prop in g2_props.items()]) if len(g2_props) > 0 else "No data"
            g1_mean, g2_mean, mean_diff = np.nan, np.nan, np.nan
        
        results.append({
            'Comparison': comparison_name, 'Variable': var_clean, 'Test_Type': test_type,
            'Group1_Name': group1_name, 'Group2_Name': group2_name,
            'Group1_N': len(group1_var), 'Group2_N': len(group2_var),
            'Test_Statistic': statistic, 'P_Value': p_value, 'Effect_Size': effect_size,
            'Group1_Summary': g1_summary, 'Group2_Summary': g2_summary,
            'Group1_Mean': g1_mean, 'Group2_Mean': g2_mean, 'Mean_Difference': mean_diff
        })
        
        print(f"{var_clean:25} | {test_type:15} | p = {p_value:.4f}")
    
    return pd.DataFrame(results)


def apply_fdr_correction_and_interpret(all_results):
    """
    Apply Benjamini-Hochberg FDR correction and add direction interpretations.
    
    Parameters:
        all_results: DataFrame with P_Value column from association tests
    
    Returns:
        DataFrame with FDR-corrected p-values and direction interpretations
    """
    valid_pvalues_mask = ~pd.isna(all_results['P_Value'])
    valid_pvalues = all_results.loc[valid_pvalues_mask, 'P_Value'].values
    
    if len(valid_pvalues) == 0:
        print("No valid p-values found for FDR correction")
        all_results['P_Value_FDR'] = np.nan
        all_results['Significant_Original'] = False
        all_results['Significant_FDR'] = False
        all_results['Direction_Interpretation'] = "No valid tests"
        return all_results
    
    print(f"\nApplying Benjamini-Hochberg FDR correction to {len(valid_pvalues)} tests")
    
    fdr_corrected = false_discovery_control(valid_pvalues, method='bh')
    
    all_results['P_Value_FDR'] = np.nan
    all_results.loc[valid_pvalues_mask, 'P_Value_FDR'] = fdr_corrected
    all_results['Significant_Original'] = all_results['P_Value'] < 0.05
    all_results['Significant_FDR'] = all_results['P_Value_FDR'] < 0.05
    
    n_sig_original = all_results['Significant_Original'].sum()
    n_sig_fdr = all_results['Significant_FDR'].sum()
    
    print(f"Original significant results (p < 0.05): {n_sig_original}/{len(valid_pvalues)} ({n_sig_original/len(valid_pvalues)*100:.1f}%)")
    print(f"FDR-corrected significant results (q < 0.05): {n_sig_fdr}/{len(valid_pvalues)} ({n_sig_fdr/len(valid_pvalues)*100:.1f}%)")
    print(f"Results filtered out by FDR correction: {n_sig_original - n_sig_fdr}")
    
    direction_interpretations = []
    
    for idx, row in all_results.iterrows():
        if pd.isna(row['P_Value_FDR']):
            direction_interpretations.append("No valid test")
        elif row['Test_Type'] == 'Insufficient_Data':
            direction_interpretations.append("Insufficient data")
        elif row['Test_Type'] == 'Test_Failed':
            direction_interpretations.append("Test failed")
        elif row['Significant_FDR']:
            if not pd.isna(row['Mean_Difference']):
                diff = row['Mean_Difference']
                if diff > 0:
                    direction_interpretations.append(f"{row['Group1_Name']} higher than {row['Group2_Name']} (delta = {diff:.2f}) [FDR significant]")
                else:
                    direction_interpretations.append(f"{row['Group2_Name']} higher than {row['Group1_Name']} (delta = {abs(diff):.2f}) [FDR significant]")
            else:
                try:
                    g1_summary = row['Group1_Summary']
                    g2_summary = row['Group2_Summary']
                    if 'No data' in g1_summary or 'No data' in g2_summary:
                        direction_interpretations.append("FDR significant (details unavailable)")
                    else:
                        direction_interpretations.append(f"Significant association between {row['Group1_Name']} and {row['Group2_Name']} [FDR significant]")
                except Exception:
                    direction_interpretations.append("FDR significant (details unavailable)")
        else:
            if row['Significant_Original']:
                direction_interpretations.append("Originally significant, but not after FDR correction")
            else:
                direction_interpretations.append("No significant difference")
    
    all_results['Direction_Interpretation'] = direction_interpretations
    
    return all_results

def get_unique_sample_mask_excluding_gene_samples(df, base_mask, gene, gene_col='Gene.refGene'):
    """
    For a given mask, keep only one occurrence per sample_id and exclude samples
    that share a sample_id with the specified gene to prevent group contamination.
    
    Parameters:
        df: DataFrame with genetic variant data
        base_mask: Boolean mask for initial filtering
        gene: str, gene name to exclude (e.g., 'DNMT3A', 'TET2')
        gene_col: str, column name containing gene identifiers
    
    Returns:
        Boolean mask for unique samples excluding gene-positive samples
    """
    gene_sample_ids = set(df[df[gene_col] == gene]['sample_id'].unique())
    subset_df = df[base_mask].copy()
    subset_df = subset_df[~subset_df['sample_id'].isin(gene_sample_ids)]
    unique_samples = subset_df.drop_duplicates(subset=['sample_id'], keep='first')
    unique_mask = df.index.isin(unique_samples.index)
    return unique_mask


def print_group_medians(df_subset, group_name, variables_to_test):
    """Print median values for numeric variables or mode for categorical variables in a group."""
    print(f"\n  {group_name} (n={len(df_subset)}) - Median values:")
    if len(df_subset) == 0:
        print("    No data available")
        return
    for var in variables_to_test:
        if var in df_subset.columns:
            if df_subset[var].dtype in ['int64', 'float64', 'int32', 'float32']:
                median_val = df_subset[var].median()
                n_non_null = df_subset[var].notna().sum()
                print(f"    {var}: {median_val:.4f} (n={n_non_null})")
            else:
                mode_val = df_subset[var].mode()
                if len(mode_val) > 0:
                    n_non_null = df_subset[var].notna().sum()
                    print(f"    {var}: {mode_val.iloc[0]} (most common, n={n_non_null})")
        else:
            print(f"    {var}: Variable not found in data")


def run_gene_pairwise_analysis_with_fdr(df, gene, variables_to_test, percentile_thresholds=None,
                                        gene_col='Gene.refGene', include_controls=True):
    """
    Run all pairwise comparison analyses for a specified gene with FDR correction.
    Includes duplicate filtering, optional control exclusion, and VAF-based comparisons.
    
    Parameters:
        df: DataFrame with genetic variant data
        gene: str, gene name to analyze (e.g., 'TET2', 'DNMT3A')
        variables_to_test: list, variable names to test for associations
        percentile_thresholds: list, percentile thresholds for high/low VAF comparisons (0-100)
        gene_col: str, column name containing gene identifiers
        include_controls: bool, if False excludes Control subtype (TET2 analysis typically excludes)
    
    Returns:
        all_results: DataFrame with FDR-corrected results
        df_clean: DataFrame after duplicate removal and filtering
    """
    if percentile_thresholds is None:
        percentile_thresholds = [50]
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip()
    
    print(f"FILTERING TECHNICAL DUPLICATES FOR {gene} ANALYSIS")
    original_count = len(df_clean)
    print(f"Original dataset size: {original_count}")
    
    if include_controls:
        df_clean = df_clean.drop_duplicates(subset=['sample_id', 'VAF', gene_col], keep='first')
    else:
        df_clean = df_clean.drop_duplicates(subset=['sample_id', 'VAF'], keep='first')
    
    filtered_count = len(df_clean)
    removed_count = original_count - filtered_count
    print(f"After removing duplicates: {filtered_count}")
    print(f"Duplicates removed: {removed_count}")
    
    if not include_controls:
        print(f"\nFiltering to {SUBTYPE_A} and {SUBTYPE_B} cases only")
        before_subtype_filter = len(df_clean)
        df_clean = df_clean[df_clean[GROUP_COL].isin([SUBTYPE_A, SUBTYPE_B])]
        after_subtype_filter = len(df_clean)
        print(f"Before subtype filtering: {before_subtype_filter}")
        print(f"After limiting to {SUBTYPE_A}/{SUBTYPE_B}: {after_subtype_filter}")
        print(f"{CONTROL_LABEL} cases removed: {before_subtype_filter - after_subtype_filter}")
    
    subtype_counts = df_clean[GROUP_COL].value_counts()
    print(f"{SUBTYPE_A} cases: {subtype_counts.get(SUBTYPE_A, 0)}")
    print(f"{SUBTYPE_B} cases: {subtype_counts.get(SUBTYPE_B, 0)}")
    if include_controls:
        print(f"{CONTROL_LABEL} cases: {subtype_counts.get(CONTROL_LABEL, 0)}")
    
    gene_filtered = df_clean[df_clean[gene_col] == gene]
    print(f"\n{gene} variants after filtering: {len(gene_filtered)}")
    gene_by_subtype = gene_filtered[GROUP_COL].value_counts()
    print(f"  - {gene} {SUBTYPE_A}: {gene_by_subtype.get(SUBTYPE_A, 0)}")
    print(f"  - {gene} {SUBTYPE_B}: {gene_by_subtype.get(SUBTYPE_B, 0)}")
    if include_controls:
        print(f"  - {gene} {CONTROL_LABEL}: {gene_by_subtype.get(CONTROL_LABEL, 0)}")
    
    if include_controls:
        gene_sample_ids = set(df_clean[df_clean[gene_col] == gene]['sample_id'].unique())
        print(f"\nUnique samples with {gene} mutations: {len(gene_sample_ids)}")
        print(f"These samples will be excluded from non-{gene} comparison groups")
    
    gene_vafs = df_clean[df_clean[gene_col] == gene]['VAF']
    vaf_thresholds = {}
    print(f"\nCalculating VAF thresholds from {gene} data (n={len(gene_vafs)})")
    for percentile in percentile_thresholds:
        high_threshold = gene_vafs.quantile(1 - percentile/100)
        low_threshold = gene_vafs.quantile(percentile/100)
        vaf_thresholds[percentile] = {'high_threshold': high_threshold, 'low_threshold': low_threshold}
        print(f"Percentile {percentile}%: Top {percentile}% >= {high_threshold:.4f}, Bottom {percentile}% <= {low_threshold:.4f}")
    
    comparisons = []
    comparison_counter = 1
    
    print(f"\nRunning comparisons with group characteristics")
    
    # Comparison 1: Gene vs non-Gene (all subtypes or Limited+Diffuse only)
    print(f"\n{comparison_counter}. {gene} vs non-{gene}")
    gene_mask = df_clean[gene_col] == gene
    if include_controls:
        non_gene_base_mask = df_clean[gene_col] != gene
        non_gene_mask = get_unique_sample_mask_excluding_gene_samples(df_clean, non_gene_base_mask, gene, gene_col)
    else:
        non_gene_mask = df_clean[gene_col] != gene
    
    print_group_medians(df_clean[gene_mask], f"{gene} group", variables_to_test)
    print_group_medians(df_clean[non_gene_mask], f"Non-{gene} group", variables_to_test)
    
    comp = test_associations_flexible_with_direction_no_significance(
        df_clean, variables_to_test, gene_mask, non_gene_mask,
        f"{comparison_counter}. {gene} vs non-{gene}", gene, f"non-{gene}")
    comparisons.append(comp)
    comparison_counter += 1
    
    # Comparison 2: Limited Gene vs Diffuse Gene
    print(f"\n{comparison_counter}. {SUBTYPE_A} {gene} vs {SUBTYPE_B} {gene}")
    limited_gene_mask = (df_clean[gene_col] == gene) & (df_clean[GROUP_COL] == SUBTYPE_A)
    diffuse_gene_mask = (df_clean[gene_col] == gene) & (df_clean[GROUP_COL] == SUBTYPE_B)
    
    print_group_medians(df_clean[limited_gene_mask], f"{SUBTYPE_A} {gene} group", variables_to_test)
    print_group_medians(df_clean[diffuse_gene_mask], f"{SUBTYPE_B} {gene} group", variables_to_test)
    
    comp = test_associations_flexible_with_direction_no_significance(
        df_clean, variables_to_test, limited_gene_mask, diffuse_gene_mask,
        f"{comparison_counter}. {SUBTYPE_A} {gene} vs {SUBTYPE_B} {gene}", f"{SUBTYPE_A} {gene}", f"{SUBTYPE_B} {gene}")
    comparisons.append(comp)
    comparison_counter += 1
    
    # Additional control comparisons if controls included
    if include_controls:
        control_gene_mask = (df_clean[gene_col] == gene) & (df_clean[GROUP_COL] == CONTROL_LABEL)
        
        print(f"\n{comparison_counter}. {SUBTYPE_A} {gene} vs {CONTROL_LABEL} {gene}")
        comp = test_associations_flexible_with_direction_no_significance(
            df_clean, variables_to_test, limited_gene_mask, control_gene_mask,
            f"{comparison_counter}. {SUBTYPE_A} {gene} vs {CONTROL_LABEL} {gene}", f"{SUBTYPE_A} {gene}", f"{CONTROL_LABEL} {gene}")
        comparisons.append(comp)
        comparison_counter += 1
        
        print(f"\n{comparison_counter}. {SUBTYPE_B} {gene} vs {CONTROL_LABEL} {gene}")
        comp = test_associations_flexible_with_direction_no_significance(
            df_clean, variables_to_test, diffuse_gene_mask, control_gene_mask,
            f"{comparison_counter}. {SUBTYPE_B} {gene} vs {CONTROL_LABEL} {gene}", f"{SUBTYPE_B} {gene}", f"{CONTROL_LABEL} {gene}")
        comparisons.append(comp)
        comparison_counter += 1
    
    # Comparisons: Gene vs non-Gene within each subtype
    print(f"\n{comparison_counter}. {SUBTYPE_A} {gene} vs non-{gene} ({SUBTYPE_A} cases)")
    if include_controls:
        limited_non_gene_base_mask = (df_clean[gene_col] != gene) & (df_clean[GROUP_COL] == SUBTYPE_A)
        limited_non_gene_mask = get_unique_sample_mask_excluding_gene_samples(df_clean, limited_non_gene_base_mask, gene, gene_col)
    else:
        limited_non_gene_mask = (df_clean[gene_col] != gene) & (df_clean[GROUP_COL] == SUBTYPE_A)
    
    print_group_medians(df_clean[limited_gene_mask], f"{SUBTYPE_A} {gene} group", variables_to_test)
    print_group_medians(df_clean[limited_non_gene_mask], f"{SUBTYPE_A} non-{gene} group", variables_to_test)
    
    comp = test_associations_flexible_with_direction_no_significance(
        df_clean, variables_to_test, limited_gene_mask, limited_non_gene_mask,
        f"{comparison_counter}. {SUBTYPE_A} {gene} vs non-{gene} ({SUBTYPE_A})", f"{SUBTYPE_A} {gene}", f"{SUBTYPE_A} non-{gene}")
    comparisons.append(comp)
    comparison_counter += 1
    
    print(f"\n{comparison_counter}. {SUBTYPE_B} {gene} vs non-{gene} ({SUBTYPE_B} cases)")
    if include_controls:
        diffuse_non_gene_base_mask = (df_clean[gene_col] != gene) & (df_clean[GROUP_COL] == SUBTYPE_B)
        diffuse_non_gene_mask = get_unique_sample_mask_excluding_gene_samples(df_clean, diffuse_non_gene_base_mask, gene, gene_col)
    else:
        diffuse_non_gene_mask = (df_clean[gene_col] != gene) & (df_clean[GROUP_COL] == SUBTYPE_B)
    
    print_group_medians(df_clean[diffuse_gene_mask], f"{SUBTYPE_B} {gene} group", variables_to_test)
    print_group_medians(df_clean[diffuse_non_gene_mask], f"{SUBTYPE_B} non-{gene} group", variables_to_test)
    
    comp = test_associations_flexible_with_direction_no_significance(
        df_clean, variables_to_test, diffuse_gene_mask, diffuse_non_gene_mask,
        f"{comparison_counter}. {SUBTYPE_B} {gene} vs non-{gene} ({SUBTYPE_B})", f"{SUBTYPE_B} {gene}", f"{SUBTYPE_B} non-{gene}")
    comparisons.append(comp)
    comparison_counter += 1
    
    if include_controls:
        print(f"\n{comparison_counter}. {CONTROL_LABEL} {gene} vs non-{gene} ({CONTROL_LABEL} cases)")
        control_non_gene_base_mask = (df_clean[gene_col] != gene) & (df_clean[GROUP_COL] == CONTROL_LABEL)
        control_non_gene_mask = get_unique_sample_mask_excluding_gene_samples(df_clean, control_non_gene_base_mask, gene, gene_col)
        
        comp = test_associations_flexible_with_direction_no_significance(
            df_clean, variables_to_test, control_gene_mask, control_non_gene_mask,
            f"{comparison_counter}. {CONTROL_LABEL} {gene} vs non-{gene} ({CONTROL_LABEL})", f"{CONTROL_LABEL} {gene}", f"{CONTROL_LABEL} non-{gene}")
        comparisons.append(comp)
        comparison_counter += 1
    
    # VAF-based comparisons
    for percentile in percentile_thresholds:
        high_threshold = vaf_thresholds[percentile]['high_threshold']
        low_threshold = vaf_thresholds[percentile]['low_threshold']
        
        print(f"\nRunning VAF-based comparisons for {percentile}th percentile threshold")
        
        # High vs Low VAF within Limited
        print(f"\n{comparison_counter}. Top {percentile}% vs Bottom {percentile}% VAF {SUBTYPE_A} {gene}")
        high_vaf_limited_mask = ((df_clean[gene_col] == gene) & 
                                 (df_clean[GROUP_COL] == SUBTYPE_A) & 
                                 (df_clean['VAF'] >= high_threshold))
        low_vaf_limited_mask = ((df_clean[gene_col] == gene) & 
                                (df_clean[GROUP_COL] == SUBTYPE_A) & 
                                (df_clean['VAF'] <= low_threshold))
        
        print_group_medians(df_clean[high_vaf_limited_mask], f"High VAF {SUBTYPE_A} {gene} (top {percentile}%, >={high_threshold:.4f})", variables_to_test)
        print_group_medians(df_clean[low_vaf_limited_mask], f"Low VAF {SUBTYPE_A} {gene} (bottom {percentile}%, <={low_threshold:.4f})", variables_to_test)
        
        comp = test_associations_flexible_with_direction_no_significance(
            df_clean, variables_to_test, high_vaf_limited_mask, low_vaf_limited_mask,
            f"{comparison_counter}. Top {percentile}% vs Bottom {percentile}% VAF {SUBTYPE_A} {gene} (>={high_threshold:.4f} vs <={low_threshold:.4f})",
            f"High VAF {SUBTYPE_A} {gene} (top {percentile}%)", f"Low VAF {SUBTYPE_A} {gene} (bottom {percentile}%)")
        comparisons.append(comp)
        comparison_counter += 1
        
        # High vs Low VAF within Diffuse
        print(f"\n{comparison_counter}. Top {percentile}% vs Bottom {percentile}% VAF {SUBTYPE_B} {gene}")
        high_vaf_diffuse_mask = ((df_clean[gene_col] == gene) & 
                                 (df_clean[GROUP_COL] == SUBTYPE_B) & 
                                 (df_clean['VAF'] >= high_threshold))
        low_vaf_diffuse_mask = ((df_clean[gene_col] == gene) & 
                                (df_clean[GROUP_COL] == SUBTYPE_B) & 
                                (df_clean['VAF'] <= low_threshold))
        
        print_group_medians(df_clean[high_vaf_diffuse_mask], f"High VAF {SUBTYPE_B} {gene} (top {percentile}%, >={high_threshold:.4f})", variables_to_test)
        print_group_medians(df_clean[low_vaf_diffuse_mask], f"Low VAF {SUBTYPE_B} {gene} (bottom {percentile}%, <={low_threshold:.4f})", variables_to_test)
        
        comp = test_associations_flexible_with_direction_no_significance(
            df_clean, variables_to_test, high_vaf_diffuse_mask, low_vaf_diffuse_mask,
            f"{comparison_counter}. Top {percentile}% vs Bottom {percentile}% VAF {SUBTYPE_B} {gene} (>={high_threshold:.4f} vs <={low_threshold:.4f})",
            f"High VAF {SUBTYPE_B} {gene} (top {percentile}%)", f"Low VAF {SUBTYPE_B} {gene} (bottom {percentile}%)")
        comparisons.append(comp)
        comparison_counter += 1
        
        # High vs Low VAF across all subtypes
        print(f"\n{comparison_counter}. Top {percentile}% vs Bottom {percentile}% VAF All {gene}")
        high_vaf_all_mask = (df_clean[gene_col] == gene) & (df_clean['VAF'] >= high_threshold)
        low_vaf_all_mask = (df_clean[gene_col] == gene) & (df_clean['VAF'] <= low_threshold)
        
        print_group_medians(df_clean[high_vaf_all_mask], f"High VAF All {gene} (top {percentile}%, >={high_threshold:.4f})", variables_to_test)
        print_group_medians(df_clean[low_vaf_all_mask], f"Low VAF All {gene} (bottom {percentile}%, <={low_threshold:.4f})", variables_to_test)
        
        comp = test_associations_flexible_with_direction_no_significance(
            df_clean, variables_to_test, high_vaf_all_mask, low_vaf_all_mask,
            f"{comparison_counter}. Top {percentile}% vs Bottom {percentile}% VAF All {gene} (>={high_threshold:.4f} vs <={low_threshold:.4f})",
            f"High VAF All {gene} (top {percentile}%)", f"Low VAF All {gene} (bottom {percentile}%)")
        comparisons.append(comp)
        comparison_counter += 1
        
        if include_controls:
            # High VAF vs non-gene within Limited
            print(f"\n{comparison_counter}. Top {percentile}% VAF {SUBTYPE_A} {gene} vs non-{gene} {SUBTYPE_A}")
            comp = test_associations_flexible_with_direction_no_significance(
                df_clean, variables_to_test, high_vaf_limited_mask, limited_non_gene_mask,
                f"{comparison_counter}. Top {percentile}% VAF {SUBTYPE_A} {gene} vs non-{gene} {SUBTYPE_A} (>={high_threshold:.4f})",
                f"High VAF {SUBTYPE_A} {gene} (top {percentile}%)", f"non-{gene} {SUBTYPE_A}")
            comparisons.append(comp)
            comparison_counter += 1
            
            # High VAF vs non-gene across all
            print(f"\n{comparison_counter}. Top {percentile}% VAF all {gene} vs all non-{gene}")
            comp = test_associations_flexible_with_direction_no_significance(
                df_clean, variables_to_test, high_vaf_all_mask, non_gene_mask,
                f"{comparison_counter}. Top {percentile}% VAF all {gene} vs all non-{gene} (>={high_threshold:.4f})",
                f"High VAF {gene} (top {percentile}%)", f"non-{gene}")
            comparisons.append(comp)
            comparison_counter += 1
            
            # High vs Low VAF in Control
            print(f"\n{comparison_counter}. Top {percentile}% vs Bottom {percentile}% VAF {gene} in {CONTROL_LABEL} patients")
            high_vaf_control_mask = ((df_clean[gene_col] == gene) & 
                                     (df_clean[GROUP_COL] == CONTROL_LABEL) & 
                                     (df_clean['VAF'] >= high_threshold))
            low_vaf_control_mask = ((df_clean[gene_col] == gene) & 
                                    (df_clean[GROUP_COL] == CONTROL_LABEL) & 
                                    (df_clean['VAF'] <= low_threshold))
            
            comp = test_associations_flexible_with_direction_no_significance(
                df_clean, variables_to_test, high_vaf_control_mask, low_vaf_control_mask,
                f"{comparison_counter}. Top {percentile}% vs Bottom {percentile}% VAF {gene} in {CONTROL_LABEL} patients",
                f"High VAF {CONTROL_LABEL} {gene} (top {percentile}%)", f"Low VAF {CONTROL_LABEL} {gene} (bottom {percentile}%)")
            comparisons.append(comp)
            comparison_counter += 1
    
    all_results_raw = pd.concat(comparisons, ignore_index=True)
    all_results_final = apply_fdr_correction_and_interpret(all_results_raw)
    
    print(f"\nAnalysis complete")
    print(f"Final dataset size: {len(df_clean)}")
    print(f"Total comparisons performed: {len(comparisons)}")
    print(f"Total statistical tests: {len(all_results_final)}")
    
    return all_results_final, df_clean


def run_gene_within_metric_regression_analysis(df, gene, gene_col='Gene.refGene'):
    """
    Perform gene-specific regression analysis with within-metric FDR correction.
    Tests gene vs non-CHIP and gene vs other CHIP across clinical outcomes.
    
    Parameters:
        df: DataFrame with genetic variant data and clinical outcomes
        gene: str, gene name to analyze (e.g., 'TET2', 'DNMT3A')
        gene_col: str, column name containing gene identifiers
    
    Returns:
        dict: Nested results dictionary with structure [outcome][comparison][subtype]
    """
    df = df.copy()
    gene_status_col = f'{gene}_status'
    df[gene_status_col] = (df[gene_col] == gene).astype(int)
    df['non_CHIP'] = (df['is_wl'] == 0).astype(int)
    df[f'non_{gene}_CHIP'] = ((df['is_wl'] == 1) & (df[gene_col] != gene)).astype(int)
    
    outcomes = [FVC_PCT_COL, DLCO_PCT_COL, SKIN_SCORE_COL]
    subtypes = [SUBTYPE_A, SUBTYPE_B, 'Combined']
    comparisons = [f'{gene}_vs_nonCHIP', f'{gene}_vs_non{gene}_CHIP']
    
    results = {}
    
    print(f"{gene} Analysis with within-metric FDR correction")
    print("Approach: FDR correction applied separately within each clinical metric")
    print("- FVC: 6 tests (2 comparisons x 3 subtypes)")
    print("- DLCO: 6 tests (2 comparisons x 3 subtypes)")
    print(f"- {SKIN_SCORE_COL}: 6 tests (2 comparisons x 3 subtypes)")
    
    for outcome in outcomes:
        print(f"\nAnalyzing: {outcome}")
        results[outcome] = {}
        p_values_for_outcome = []
        test_info_for_outcome = []
        
        for comparison in comparisons:
            results[outcome][comparison] = {}
            
            for subtype in subtypes:
                if subtype == 'Combined':
                    subset_df = df[df[GROUP_COL].isin([SUBTYPE_A, SUBTYPE_B])].copy()
                else:
                    subset_df = df[df[GROUP_COL] == subtype].copy()
                
                if comparison == f'{gene}_vs_nonCHIP':
                    analysis_df = subset_df[(subset_df[gene_status_col] == 1) | 
                                           (subset_df['non_CHIP'] == 1)].copy()
                else:
                    analysis_df = subset_df[(subset_df[gene_status_col] == 1) | 
                                           (subset_df[f'non_{gene}_CHIP'] == 1)].copy()
                
                comparison_var = gene_status_col
                analysis_df = analysis_df.dropna(subset=[outcome, AGE_COL])
                
                if len(analysis_df) < 10:
                    results[outcome][comparison][subtype] = {
                        'coef': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
                        'pvalue_raw': np.nan, 'pvalue_fdr_within': np.nan,
                        'f_stat': np.nan, 'f_pvalue': np.nan, 'n': len(analysis_df),
                        'cohens_d': np.nan, 'r_squared': np.nan
                    }
                    continue
                
                if subtype == 'Combined':
                    formula = f"Q('{outcome}') ~ {comparison_var} + {AGE_COL} + C({GROUP_COL})"
                else:
                    formula = f"Q('{outcome}') ~ {comparison_var} + {AGE_COL}"
                
                try:
                    model = ols(formula, data=analysis_df).fit()
                    coef = model.params[comparison_var]
                    ci = model.conf_int().loc[comparison_var]
                    pvalue_raw = model.pvalues[comparison_var]
                    f_stat = model.fvalue
                    f_pvalue = model.f_pvalue
                    
                    gene_group = analysis_df[analysis_df[comparison_var] == 1][outcome]
                    control_group = analysis_df[analysis_df[comparison_var] == 0][outcome]
                    pooled_std = np.sqrt(((len(gene_group) - 1) * gene_group.var() + 
                                         (len(control_group) - 1) * control_group.var()) / 
                                        (len(gene_group) + len(control_group) - 2))
                    cohens_d = (gene_group.mean() - control_group.mean()) / pooled_std if pooled_std > 0 else np.nan
                    
                    results[outcome][comparison][subtype] = {
                        'coef': coef,
                        'ci_lower': ci[0],
                        'ci_upper': ci[1],
                        'pvalue_raw': pvalue_raw,
                        'pvalue_fdr_within': np.nan,
                        'f_stat': f_stat,
                        'f_pvalue': f_pvalue,
                        'n': len(analysis_df),
                        'r_squared': model.rsquared,
                        'cohens_d': cohens_d,
                        'formula': formula,
                        'model_summary': str(model.summary())
                    }
                    
                    p_values_for_outcome.append(pvalue_raw)
                    test_info_for_outcome.append((comparison, subtype))
                    
                except Exception as e:
                    print(f"  Error in {comparison} {subtype}: {e}")
                    results[outcome][comparison][subtype] = {
                        'coef': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
                        'pvalue_raw': np.nan, 'pvalue_fdr_within': np.nan,
                        'f_stat': np.nan, 'f_pvalue': np.nan, 'n': len(analysis_df),
                        'cohens_d': np.nan, 'r_squared': np.nan
                    }
        
        if len(p_values_for_outcome) > 0:
            valid_p_values = [p for p in p_values_for_outcome if not np.isnan(p)]
            if len(valid_p_values) > 0:
                rejected, pvals_corrected, _, _ = multipletests(valid_p_values, alpha=0.05, method='fdr_bh')
                valid_idx = 0
                for i, (comparison, subtype) in enumerate(test_info_for_outcome):
                    if not np.isnan(p_values_for_outcome[i]):
                        results[outcome][comparison][subtype]['pvalue_fdr_within'] = pvals_corrected[valid_idx]
                        valid_idx += 1
        
        print(f"\nResults for {outcome}:")
        print(f"{'Test':<35} {'n':<5} {'Coef':<8} {'Raw p':<8} {'FDR p':<8} {'F-stat':<8} {'F p-val':<8} {'R2':<6}")
        
        sig_raw = 0
        sig_fdr = 0
        
        for comparison in comparisons:
            comp_label = comparison.replace(f'{gene}_vs_', '').replace('_', '-')
            for subtype in subtypes:
                result = results[outcome][comparison][subtype]
                if not np.isnan(result['pvalue_raw']):
                    raw_sig = "*" if result['pvalue_raw'] < 0.05 else ""
                    fdr_sig = "+" if (not np.isnan(result['pvalue_fdr_within']) and 
                                      result['pvalue_fdr_within'] < 0.05) else ""
                    if raw_sig:
                        sig_raw += 1
                    if fdr_sig:
                        sig_fdr += 1
                    
                    test_name = f"{comp_label} {subtype}"
                    print(f"{test_name:<35} {int(result['n']):<5} "
                          f"{result['coef']:<8.2f} "
                          f"{result['pvalue_raw']:<7.4f}{raw_sig} "
                          f"{result['pvalue_fdr_within']:<7.4f}{fdr_sig} "
                          f"{result['f_stat']:<8.2f} "
                          f"{result['f_pvalue']:<8.4f} "
                          f"{result['r_squared']:<6.3f}")
        
        print(f"\nSummary for {outcome}:")
        print(f"  Raw significant (p<0.05): {sig_raw}/6")
        print(f"  FDR significant (q<0.05): {sig_fdr}/6")
    
    return results


def create_model_performance_summary(results, gene):
    """
    Create summary of F-statistics and model performance for gene regression results.
    
    Parameters:
        results: dict, nested results from run_gene_within_metric_regression_analysis
        gene: str, gene name for labeling
    """
    print(f"\nF-statistics and model performance summary - {gene}")
    
    outcomes = [FVC_PCT_COL, DLCO_PCT_COL, SKIN_SCORE_COL]
    comparisons = [f'{gene}_vs_nonCHIP', f'{gene}_vs_non{gene}_CHIP']
    subtypes = [SUBTYPE_A, SUBTYPE_B, 'Combined']
    
    print(f"{'Model':<40} {'F-statistic':<12} {'F p-value':<10} {'R2':<8} {'Model Sig':<10}")
    
    significant_models = 0
    total_models = 0
    
    for outcome in outcomes:
        for comparison in comparisons:
            comp_short = comparison.replace(f'{gene}_vs_', '').replace('_', '-')
            for subtype in subtypes:
                result = results[outcome][comparison][subtype]
                
                if not np.isnan(result['f_stat']):
                    total_models += 1
                    f_sig = "Yes" if result['f_pvalue'] < 0.05 else "No"
                    if f_sig == "Yes":
                        significant_models += 1
                    
                    model_name = f"{outcome[:4]} {comp_short} {subtype}"
                    print(f"{model_name:<40} {result['f_stat']:<12.3f} "
                          f"{result['f_pvalue']:<10.4f} {result['r_squared']:<8.3f} {f_sig:<10}")
    
    print(f"\nOverall model significance: {significant_models}/{total_models} models have significant F-tests")
    print(f"This indicates how many models explain significant variance overall")


def summarize_results_with_direction(results_df, significance_threshold=0.05):
    """
    Create a clean summary of all comparison results with direction interpretation.
    
    Parameters:
        results_df: DataFrame with comparison results including P_Value and Direction_Interpretation columns
        significance_threshold: float, p-value threshold for significance
    """
    print("\nSummary of all comparisons with direction/trends")
    
    comparisons = results_df['Comparison'].unique()
    
    for comparison in comparisons:
        comp_results = results_df[results_df['Comparison'] == comparison]
        significant_results = comp_results[comp_results['P_Value'] < significance_threshold]
        
        print(f"\n{comparison}")
        
        if len(significant_results) > 0:
            print("Significant associations:")
            for _, row in significant_results.sort_values('P_Value').iterrows():
                effect_str = f"Effect: {row['Effect_Size']:.3f}" if not pd.isna(row['Effect_Size']) else "Effect: N/A"
                print(f"  - {row['Variable']:20} | p = {row['P_Value']:.4f} | {effect_str}")
                print(f"    Direction: {row['Direction_Interpretation']}")
        else:
            print("No significant associations found")
        
        print(f"Total variables tested: {len(comp_results)}")
        print(f"Significant associations: {len(significant_results)}")
    
    print(f"\nSummary table: All significant findings with clinical interpretation")
    
    significant_overall = results_df[results_df['P_Value'] < significance_threshold].sort_values('P_Value')
    
    if len(significant_overall) > 0:
        print(f"{'Comparison':<45} | {'Variable':<20} | {'P-Value':<8} | {'Direction/Trend'}")
        for _, row in significant_overall.iterrows():
            comp_short = row['Comparison'].split('.')[0] + "." if '.' in row['Comparison'] else row['Comparison'][:40]
            print(f"{comp_short:<45} | {row['Variable']:<20} | {row['P_Value']:<8.4f} | {row['Direction_Interpretation']}")
    else:
        print("No significant associations found across any comparisons")
    
    print(f"\nSummary counts")
    for comparison in comparisons:
        comp_results = results_df[results_df['Comparison'] == comparison]
        significant_count = len(comp_results[comp_results['P_Value'] < significance_threshold])
        total_count = len(comp_results)
        print(f"{comparison}: {significant_count}/{total_count} significant")
        
def check_chip_gene_confounding_quick(df, gene='TET2', outcome=None,
                                       confounder=None, gene_col_name='CHIP_gene',
                                       verbose=True):
    """
    Quick check for classic confounding triangle in CHIP gene analysis.
    Tests if confounder is associated with both gene status and outcome.
    
    Parameters:
        df: Dataframe with gene, outcome, and confounder columns
        gene: Gene name to analyze
        outcome: Clinical outcome variable
        confounder: Potential confounding variable
        gene_col_name: Column name containing gene names
        verbose: Print results
    
    Returns:
        Dictionary with association tests and confounding determination
    """
    if outcome is None:
        outcome = FVC_PCT_COL
    if confounder is None:
        confounder = DURATION_COL
    
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    gene_status_col = f'{gene}_status'
    df[gene_status_col] = (df[gene_col_name] == gene).astype(int)
    
    results = {
        'gene': gene,
        'outcome': outcome,
        'confounder': confounder,
        'confounder_gene_p': np.nan,
        'confounder_outcome_r': np.nan,
        'confounder_outcome_p': np.nan,
        'confounding_detected': False,
        'recommendation': ''
    }
    
    # Check confounder-gene association
    gene_pos = df[df[gene_status_col] == 1][confounder].dropna()
    gene_neg = df[df[gene_status_col] == 0][confounder].dropna()
    
    confounder_gene_associated = False
    if len(gene_pos) > 0 and len(gene_neg) > 0:
        _, t_p = ttest_ind(gene_pos, gene_neg)
        results['confounder_gene_p'] = t_p
        results['gene_pos_mean'] = gene_pos.mean()
        results['gene_neg_mean'] = gene_neg.mean()
        confounder_gene_associated = t_p < 0.05
    
    # Check confounder-outcome association
    disease_df = df[df[GROUP_COL].isin([SUBTYPE_A, SUBTYPE_B])]
    clean_data = disease_df[[confounder, outcome]].dropna()
    
    confounder_outcome_associated = False
    if len(clean_data) > 10:
        r, p = pearsonr(clean_data[confounder], clean_data[outcome])
        results['confounder_outcome_r'] = r
        results['confounder_outcome_p'] = p
        confounder_outcome_associated = p < 0.05
    
    # Determine confounding
    results['confounding_detected'] = confounder_gene_associated and confounder_outcome_associated
    
    if results['confounding_detected']:
        results['recommendation'] = f'Include {confounder} as covariate'
    elif confounder_gene_associated:
        results['recommendation'] = f'{confounder} associated with {gene} but not {outcome} - consider including'
    elif confounder_outcome_associated:
        results['recommendation'] = f'{confounder} affects {outcome} but not {gene} - not a confounder'
    else:
        results['recommendation'] = f'{confounder} adjustment not necessary'
    
    if verbose:
        print(f"Confounding check: {confounder} in {gene}-{outcome} analysis")
        if not np.isnan(results['confounder_gene_p']):
            print(f"  {confounder} vs {gene}: {results['gene_pos_mean']:.1f} vs {results['gene_neg_mean']:.1f}, p={results['confounder_gene_p']:.4f}")
        if not np.isnan(results['confounder_outcome_r']):
            print(f"  {confounder} vs {outcome}: r={results['confounder_outcome_r']:.3f}, p={results['confounder_outcome_p']:.4f}")
        print(f"  Confounding detected: {results['confounding_detected']}")
        print(f"  Recommendation: {results['recommendation']}")
    
    return results
