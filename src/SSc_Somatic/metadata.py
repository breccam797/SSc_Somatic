"""
Assign cohort groups and derived annotation columns to sample tables.

Group labels, column names and age bins come from the study configuration,
so these functions work for any cohort that defines a case label, a control
label and a set of case subtypes.
"""

import pandas as pd


# Name of the sample ID column in sample lists, SNV tables and variant tables
SAMPLE_ID_COLUMN = 'sample_id'

# Names of the columns these functions add
AGE_GROUP_COLUMN = 'age_group'
BROAD_AGE_GROUP_COLUMN = 'broad_age_group'


def clean_dataframe(df):
    """Remove unnamed index columns and strip trailing whitespace from column names."""
    unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    df.columns = df.columns.str.rstrip()
    return df


def assign_groups(df, config):
    """
    Add the cohort group column (for example Control, Limited or Diffuse).

    Controls receive the control label. Cases receive the value of their
    subtype column when it is one of the configured subtypes. All other rows,
    including cases without a recognized subtype, receive None.

    Parameters:
        df: DataFrame with the configured status and subtype columns
        config: StudyConfig

    Returns:
        DataFrame with the configured group column added
    """
    status_col = config.require('cohort', 'status_column')
    subtype_col = config.require('cohort', 'subtype_column')
    group_col = config.require('cohort', 'group_column')
    case_label = config.require('cohort', 'case_label')
    control_label = config.require('cohort', 'control_label')

    df[group_col] = df[status_col].apply(lambda x: control_label if x == control_label else None)
    for subtype in config.require('cohort', 'subtypes'):
        df.loc[(df[status_col] == case_label) & (df[subtype_col] == subtype), group_col] = subtype
    return df


def annotate_samples(snv_df, meta_df, config):
    """
    Merge per-sample variant counts with participant metadata and add age groups and cohort groups.

    Parameters:
        snv_df: DataFrame with one row per sample and the configured sample column
        meta_df: DataFrame of participant metadata with the configured metadata sample column
        config: StudyConfig

    Returns:
        DataFrame with one row per row of snv_df, metadata columns, the age group
        column and the cohort group column
    """
    sample_col = SAMPLE_ID_COLUMN
    meta_sample_col = config.require('cohort', 'metadata_sample_column')
    age_col = config.require('cohort', 'clinical_columns', 'age')

    annotated = snv_df.merge(meta_df, how='left', left_on=sample_col, right_on=meta_sample_col)
    annotated[AGE_GROUP_COLUMN] = pd.cut(
        annotated[age_col], bins=config.age_bins, labels=config.age_labels, right=False
    )
    return assign_groups(annotated, config)


def add_derived_columns(df, config):
    """
    Add the broad age group, age of onset and configured indicator columns.

    Age of onset is computed as age minus disease duration. Indicator columns
    are 1 where the source column equals the configured value and 0 otherwise.

    Parameters:
        df: DataFrame returned by annotate_samples
        config: StudyConfig

    Returns:
        DataFrame with the derived columns added
    """
    clinical = config.require('cohort', 'clinical_columns')
    df[BROAD_AGE_GROUP_COLUMN] = df[AGE_GROUP_COLUMN].map(config.require('age_groups', 'broad_groups'))
    df[clinical['age_of_onset']] = df[clinical['age']] - df[clinical['disease_duration']]
    for name, spec in (config.get('cohort', 'indicator_columns') or {}).items():
        df[name] = (df[spec['column']] == spec['value']).astype(int)
    return df
