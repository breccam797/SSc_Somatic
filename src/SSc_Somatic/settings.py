"""
Apply a study configuration to the analysis and plotting modules.

Several modules use study-specific values such as group order, colors,
column names and output folders. Those values live in the configuration
file; configure() copies them into each module so the functions can use them.
Call it once per notebook, right after load_config().
"""


def cohort_values(config):
    """
    Return the cohort labels and metadata column names that analysis modules use.

    The analyses compare a control group with two case subtypes (each subtype
    against controls, and the subtypes against each other), so exactly two
    subtypes must be configured. SUBTYPE_A is the first configured subtype and
    SUBTYPE_B the second. The *_KEY values are lowercase forms of the labels,
    used where the code lowercases group names.

    Parameters:
        config: StudyConfig

    Returns:
        dict of setting name to value
    """
    subtypes = list(config.require('cohort', 'subtypes'))
    if len(subtypes) != 2:
        raise ValueError(f"cohort.subtypes must list exactly two subtypes, got {subtypes}")
    control_label = config.require('cohort', 'control_label')
    clinical = config.require('cohort', 'clinical_columns')
    return {
        'CASE_LABEL': config.require('cohort', 'case_label'),
        'CONTROL_LABEL': control_label,
        'SUBTYPE_A': subtypes[0],
        'SUBTYPE_B': subtypes[1],
        'CONTROL_KEY': control_label.lower(),
        'SUBTYPE_A_KEY': subtypes[0].lower(),
        'SUBTYPE_B_KEY': subtypes[1].lower(),
        'STATUS_COL': config.require('cohort', 'status_column'),
        'GROUP_COL': config.require('cohort', 'group_column'),
        'SUBTYPE_COL': config.require('cohort', 'subtype_column'),
        'META_ID_COL': config.require('cohort', 'metadata_sample_column'),
        'AGE_COL': clinical['age'],
        'DURATION_COL': clinical['disease_duration'],
        'SKIN_SCORE_COL': clinical['skin_score'],
        'FVC_PCT_COL': clinical['fvc_percent_predicted'],
        'DLCO_PCT_COL': clinical['dlco_percent_predicted'],
        'RACE_COL': config.require('cohort', 'race_column'),
        'ETHNICITY_COL': config.require('cohort', 'ethnicity_column'),
    }


def configure(config):
    """
    Apply a StudyConfig to every module that uses study settings.

    Parameters:
        config: StudyConfig returned by load_config()

    Returns:
        The same StudyConfig, so the call can be written as
        config = configure(load_config(path))
    """
    from .burden import analysis as burden_analysis
    from .burden import plots as burden_plots
    from .chip import analysis as chip_analysis
    from .chip import plots as chip_plots
    from .signatures import analysis as signature_analysis
    from .signatures import plots as signature_plots

    modules = (chip_analysis, chip_plots, burden_analysis, burden_plots,
               signature_analysis, signature_plots)
    for module in modules:
        module.apply_settings(config)
    return config
