"""
Load the input tables for one analysis, as described by a study configuration.

load_study() is the single entry point used by the notebooks. It replaces the
separate import modules each analysis folder used to have, and it reads files
only when it is called, never at import time.

Three analyses are supported:
    chip        CHIP analysis, using the CHIP section of the configuration
    mutation    mutation burden analysis, using the mutation section
    signatures  mutational signature analysis, using the mutation section's
                sample files plus the signatures section
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .metadata import (
    AGE_GROUP_COLUMN, SAMPLE_ID_COLUMN, add_derived_columns, annotate_samples, assign_groups, clean_dataframe,
)


ANALYSES = ('chip', 'mutation', 'signatures')

# Configuration section holding the passing-sample files for each analysis
RUN_SECTION = {'chip': 'chip', 'mutation': 'mutation', 'signatures': 'mutation'}


@dataclass
class StudyData:
    """
    Tables and sample ID lists for one analysis.

    Attributes that could not be loaded, for example because the passing-sample
    files have not been generated yet, are None or empty.
    """
    analysis: str
    exclusions: Dict[str, List[str]] = field(default_factory=dict)
    design: Optional[pd.DataFrame] = None
    passing_ids: List[str] = field(default_factory=list)
    metadata: Optional[pd.DataFrame] = None
    samples: Optional[pd.DataFrame] = None
    snv_table: Optional[pd.DataFrame] = None
    case_ids: List[str] = field(default_factory=list)
    control_ids: List[str] = field(default_factory=list)
    group_ids: Dict[str, List[str]] = field(default_factory=dict)
    gene_constraint: Optional[pd.DataFrame] = None
    whitelist_calls: Optional[pd.DataFrame] = None
    whitelist_merged: Optional[pd.DataFrame] = None
    processed_metadata: Optional[pd.DataFrame] = None
    signatures: Dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def sample_ids(self):
        """Case IDs followed by control IDs."""
        return self.case_ids + self.control_ids


def _read_sample_list(path, config, sep=','):
    """Read a file with a sample ID column and return the IDs as a list of strings."""
    sample_col = SAMPLE_ID_COLUMN
    df = pd.read_csv(path, sep=sep, dtype='str')
    return df[sample_col].tolist()


def load_exclusion_lists(config):
    """
    Load the sample exclusion lists named in the inputs section.

    Returns:
        dict with keys 'panel_of_normals', 'multi_antibody' and 'low_plate',
        each a list of sample IDs
    """
    return {
        'panel_of_normals': _read_sample_list(config.path('inputs', 'panel_of_normals_samples'), config, sep='\t'),
        'multi_antibody': _read_sample_list(config.path('inputs', 'multi_antibody_samples'), config),
        'low_plate': _read_sample_list(config.path('inputs', 'low_plate_samples'), config),
    }


def load_design(config, analysis, exclusions):
    """
    Load the passing-sample list for an analysis and apply the configured exclusions.

    Parameters:
        config: StudyConfig
        analysis: str, one of ANALYSES
        exclusions: dict returned by load_exclusion_lists

    Returns:
        Tuple of (design DataFrame, list of passing sample IDs), or (None, [])
        with a printed warning if the passing-sample file does not exist yet
    """
    sample_col = SAMPLE_ID_COLUMN
    path = config.path(RUN_SECTION[analysis], 'passing_sample_ids')
    if not os.path.exists(path):
        print(f"WARNING: {path} not found. Run the find-passing-samples notebook to generate it.")
        return None, []

    design = clean_dataframe(pd.read_csv(path, dtype={sample_col: str}))
    rules = (
        ('exclude_multi_antibody', 'multi_antibody'),
        ('exclude_panel_of_normals', 'panel_of_normals'),
        ('exclude_low_plate', 'low_plate'),
    )
    for setting, list_name in rules:
        if config.get('exclusions', setting, default=False):
            design = design[~design[sample_col].isin(exclusions[list_name])]
    return design, design[sample_col].unique().tolist()


def load_metadata(config, passing_ids):
    """
    Load participant metadata for the passing samples.

    Sample IDs are read as text, and whitespace is stripped from the status column.

    Returns:
        DataFrame of metadata rows whose sample ID is in passing_ids
    """
    meta_sample_col = config.require('cohort', 'metadata_sample_column')
    status_col = config.require('cohort', 'status_column')
    meta = clean_dataframe(pd.read_csv(config.path('inputs', 'metadata'), dtype={meta_sample_col: str}))
    meta = meta.loc[meta[meta_sample_col].isin(passing_ids)].copy()
    meta[status_col] = meta[status_col].str.strip()
    return meta


def load_snv_table(config, analysis):
    """
    Load the per-sample SNV count table for an analysis.

    The separator comes from the snv_table_separator setting of the analysis's
    configuration section and defaults to a tab.

    Returns:
        DataFrame, or None with a printed warning if the file does not exist yet
    """
    section = RUN_SECTION[analysis]
    sample_col = SAMPLE_ID_COLUMN
    path = config.path(section, 'snv_table')
    if not os.path.exists(path):
        print(f"WARNING: {path} not found. Run the find-passing-samples notebook to generate it.")
        return None
    sep = config.get(section, 'snv_table_separator', default='\t')
    return clean_dataframe(pd.read_csv(path, sep=sep, dtype={sample_col: str}))


def load_gene_constraint(config):
    """Load the gnomAD gene constraint table (tab-separated)."""
    return clean_dataframe(pd.read_csv(config.path('reference', 'gene_constraint'), sep='\t'))


def load_blood_sample_ids(config):
    """Return the IDs of samples sequenced from blood, as strings."""
    sample_col = SAMPLE_ID_COLUMN
    df = pd.read_csv(config.path('inputs', 'blood_sample_ids'), dtype={sample_col: str})
    return df[sample_col].astype(str).tolist()


def _read_output_if_present(config, keys, sep):
    """Read a previously generated table, or return None if it does not exist yet."""
    path = config.output_path(*keys, create_parent=False)
    if not os.path.exists(path):
        return None
    return clean_dataframe(pd.read_csv(path, sep=sep))


def load_whitelist_tables(config):
    """
    Load the CHIP whitelist tidy tables written by the CHIP tables notebook.

    Returns:
        Tuple of (whitelist calls, merged whitelist table); each is None if not generated yet
    """
    return (
        _read_output_if_present(config, ('chip', 'whitelist_tidy_table'), sep='\t'),
        _read_output_if_present(config, ('chip', 'whitelist_merged_table'), sep='\t'),
    )


def load_processed_metadata(config):
    """Load the processed CHIP metadata written by the CHIP tables notebook, or None if not generated yet."""
    return _read_output_if_present(config, ('chip', 'processed_metadata'), sep=',')


def load_signature_design(config):
    """
    Load the SigProfiler design file.

    Each row describes one signature run and gives sig_id, solution_sigs,
    extraction_path, assignment_path and input_matrix. The first row is the
    SBS run and the second row is the indel run.
    """
    return clean_dataframe(pd.read_csv(config.path('signatures', 'design_file')))


def _sample_ids_from_names(names, config):
    """Recover sample IDs from SigProfiler sample names using the configured delimiter and position."""
    delimiter = config.require('signatures', 'sample_name_delimiter')
    position = config.require('signatures', 'sample_name_position')
    return names.str.split(delimiter).str[position]


def load_signature_activities(row, samples, sample_ids, config, variant=False, verbose=False):
    """
    Load extraction and assignment activities for one row of the signature design file.

    Parameters:
        row: Series, one row of the signature design file
        samples: DataFrame, annotated sample table used for annotation columns
        sample_ids: list, sample IDs to keep
        config: StudyConfig
        variant: bool, if True merge only the sample and age columns (default: False)
        verbose: bool, if True print processing information (default: False)

    Returns:
        dict with keys 'ex_df', 'assign_df', 'anno_ex_df' and 'anno_assign_df'
    """
    sample_col = SAMPLE_ID_COLUMN
    age_col = config.require('cohort', 'clinical_columns', 'age')

    sig_id = str(row['sig_id']) if pd.notna(row['sig_id']) else ''
    sig_id_short = sig_id.replace('_', '')
    solution_sigs = str(row['solution_sigs']) if pd.notna(row['solution_sigs']) else ''
    extraction_path = str(row['extraction_path']) if pd.notna(row['extraction_path']) else ''
    assignment_path = str(row['assignment_path']) if pd.notna(row['assignment_path']) else ''
    if verbose:
        print(f"Processing: {sig_id} (short: {sig_id_short})")
    if not extraction_path or not assignment_path:
        raise ValueError(f"Missing essential paths for row with sig_id: {sig_id}")

    extraction_file_path = (extraction_path + '/All_Solutions/' +
                            sig_id_short + '_' + solution_sigs + '_Signatures/Activities/' +
                            sig_id_short + '_S' + solution_sigs + '_NMF_Activities.txt')
    assignment_file_path = assignment_path + 'Activities/Decompose_Solution_Activities.txt'
    ex_df = pd.read_csv(extraction_file_path, sep='\t', dtype={'Samples': str})
    assign_df = pd.read_csv(assignment_file_path, sep='\t', dtype={'Samples': str})

    ex_df[sample_col] = _sample_ids_from_names(ex_df['Samples'], config)
    assign_df[sample_col] = _sample_ids_from_names(assign_df['Samples'], config)
    del ex_df['Samples']
    del assign_df['Samples']
    ex_df = ex_df[[sample_col] + [col for col in ex_df.columns if col != sample_col]]
    assign_df = assign_df[[sample_col] + [col for col in assign_df.columns if col != sample_col]]

    if variant:
        merge_columns = [sample_col, age_col]
        available_columns = [col for col in merge_columns if col in samples.columns]
        if sample_col not in available_columns:
            raise ValueError(f"{sample_col} column is required but not found in the sample table")
    else:
        merge_columns = [
            sample_col,
            config.require('cohort', 'status_column'),
            config.require('cohort', 'group_column'),
            age_col,
            AGE_GROUP_COLUMN,
        ]
        missing_columns = [col for col in merge_columns if col not in samples.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in the sample table: {missing_columns}")
        available_columns = merge_columns

    anno_ex_df = ex_df.merge(samples[available_columns], on=sample_col, how='left')
    anno_assign_df = assign_df.merge(samples[available_columns], on=sample_col, how='left')
    return {
        'ex_df': ex_df.loc[ex_df[sample_col].isin(sample_ids)],
        'assign_df': assign_df.loc[assign_df[sample_col].isin(sample_ids)],
        'anno_ex_df': anno_ex_df.loc[anno_ex_df[sample_col].isin(sample_ids)],
        'anno_assign_df': anno_assign_df.loc[anno_assign_df[sample_col].isin(sample_ids)],
    }


def load_all_signatures(design, samples, sample_ids, config, verbose=False):
    """
    Load signature activities for the SBS and indel runs and the input mutation matrices.

    Returns:
        dict with keys ex_sbs_df, assign_sbs_df, anno_ex_sbs_df, anno_assign_sbs_df,
        ex_id_df, assign_id_df, anno_ex_id_df, anno_assign_id_df, mat_sbs_df, mat_id_df
    """
    sbs = load_signature_activities(design.iloc[0], samples, sample_ids, config, verbose=verbose)
    indel = load_signature_activities(design.iloc[1], samples, sample_ids, config, verbose=verbose)
    suffix = config.require('signatures', 'matrix_column_suffix')
    mat_paths = design['input_matrix'].tolist()
    mat_sbs_df = pd.read_csv(mat_paths[0], sep='\t')
    mat_id_df = pd.read_csv(mat_paths[1], sep='\t')
    mat_sbs_df.columns = mat_sbs_df.columns.str.replace(suffix, '', regex=False)
    mat_id_df.columns = mat_id_df.columns.str.replace(suffix, '', regex=False)
    return {
        'ex_sbs_df': sbs['ex_df'],
        'assign_sbs_df': sbs['assign_df'],
        'anno_ex_sbs_df': sbs['anno_ex_df'],
        'anno_assign_sbs_df': sbs['anno_assign_df'],
        'ex_id_df': indel['ex_df'],
        'assign_id_df': indel['assign_df'],
        'anno_ex_id_df': indel['anno_ex_df'],
        'anno_assign_id_df': indel['anno_assign_df'],
        'mat_sbs_df': mat_sbs_df,
        'mat_id_df': mat_id_df,
    }


def load_signature_profiles(filepath):
    """
    Load signature profiles from a SigProfiler output file.

    Parameters:
        filepath: str or Path, tab-separated file with a MutationType column

    Returns:
        DataFrame with MutationType as index and one column per signature
    """
    return pd.read_csv(filepath, sep='\t').set_index('MutationType')


def _ids_where(df, column, value, id_column):
    """Return the unique IDs in id_column for rows where column equals value."""
    return df.loc[df[column] == value, id_column].unique().tolist()


def load_study(config, analysis):
    """
    Load everything an analysis notebook needs.

    Steps: read the exclusion lists, read the passing samples and apply the
    configured exclusions, read the metadata for those samples, then load the
    analysis-specific tables. For the CHIP analysis, cohort groups are assigned
    on the metadata. For the mutation and signature analyses, the SNV table is
    merged with the metadata into an annotated sample table, and group IDs are
    taken from that table.

    Parameters:
        config: StudyConfig
        analysis: str, one of 'chip', 'mutation', 'signatures'

    Returns:
        StudyData
    """
    if analysis not in ANALYSES:
        raise ValueError(f"analysis must be one of {ANALYSES}, got {analysis!r}")

    meta_sample_col = config.require('cohort', 'metadata_sample_column')
    status_col = config.require('cohort', 'status_column')
    group_col = config.require('cohort', 'group_column')
    subtypes = config.require('cohort', 'subtypes')

    data = StudyData(analysis=analysis, exclusions=load_exclusion_lists(config))
    data.design, data.passing_ids = load_design(config, analysis, data.exclusions)
    if data.design is None:
        return data

    data.metadata = load_metadata(config, data.passing_ids)
    data.case_ids = _ids_where(data.metadata, status_col, config.require('cohort', 'case_label'), meta_sample_col)
    data.control_ids = _ids_where(data.metadata, status_col, config.require('cohort', 'control_label'), meta_sample_col)
    data.snv_table = load_snv_table(config, analysis)
    if analysis in ('chip', 'mutation'):
        data.gene_constraint = load_gene_constraint(config)

    if analysis == 'chip':
        data.metadata = assign_groups(data.metadata, config)
        data.group_ids = {s: _ids_where(data.metadata, group_col, s, meta_sample_col) for s in subtypes}
        data.whitelist_calls, data.whitelist_merged = load_whitelist_tables(config)
        data.processed_metadata = load_processed_metadata(config)
        return data

    if data.snv_table is None:
        return data
    samples = clean_dataframe(annotate_samples(data.snv_table, data.metadata, config))
    data.samples = add_derived_columns(samples, config)
    data.group_ids = {s: _ids_where(data.samples, group_col, s, meta_sample_col) for s in subtypes}

    if analysis == 'signatures':
        design = load_signature_design(config)
        data.signatures = load_all_signatures(design, data.samples, data.sample_ids, config)
    return data
