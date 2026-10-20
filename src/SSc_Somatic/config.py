"""
Load a study configuration file and resolve the file paths it contains.

A configuration file is a YAML document such as config/example_SSc.yaml.
Relative paths in the file are resolved against data_root, and generated
files are placed under output_root, so the same code runs on any machine
once the configuration points at that machine's data.
"""

from pathlib import Path

import yaml


# Settings that name input files or folders, checked by StudyConfig.check_inputs
INPUT_PATH_SETTINGS = (
    ('inputs', 'metadata'),
    ('inputs', 'panel_of_normals_samples'),
    ('inputs', 'multi_antibody_samples'),
    ('inputs', 'low_plate_samples'),
    ('inputs', 'blood_sample_ids'),
    ('reference', 'gene_constraint'),
    ('reference', 'gene_annotation_gff'),
    ('reference', 'magma_gene_locations'),
    ('reference', 'gwas_summary_statistics'),
    ('chip', 'passing_sample_ids'),
    ('chip', 'snv_table'),
    ('chip', 'whitelist_calls_dir'),
    ('mutation', 'passing_sample_ids'),
    ('mutation', 'snv_table'),
    ('mutation', 'annovar_dir'),
    ('mutation', 'annovar_filtered_dir'),
    ('signatures', 'design_file'),
    ('signatures', 'denovo_signature_profiles'),
    ('signatures', 'cosmic_signature_profiles'),
)


class StudyConfig:
    """
    Settings for one study, read from a YAML configuration file.

    Values are read with get(), for example config.get('cohort', 'case_label').
    Input file locations are read with path(), and generated file locations
    with output_path(); both return absolute pathlib.Path objects.
    """

    def __init__(self, settings, config_file, data_root=None):
        """
        Parameters:
            settings: dict, parsed contents of the configuration file
            config_file: Path, location of the configuration file
            data_root: str or Path or None, overrides data_root from the file
        """
        self.settings = settings
        self.config_file = Path(config_file).resolve()

        root = data_root if data_root is not None else settings.get('data_root')
        if root is None:
            raise ValueError(f"No data_root set in {self.config_file} and none passed in")
        root = Path(root).expanduser()
        if not root.is_absolute():
            root = self.config_file.parent / root
        self.data_root = root.resolve()

        output_root = Path(settings.get('output_root', 'outputs')).expanduser()
        if not output_root.is_absolute():
            output_root = self.data_root / output_root
        self.output_root = output_root.resolve()

    def get(self, *keys, default=None):
        """
        Return a nested setting, or default if any key along the way is missing.

        Parameters:
            *keys: str, sequence of keys, e.g. ('cohort', 'subtypes')
            default: value returned when the setting is absent
        """
        value = self.settings
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    def require(self, *keys):
        """Return a nested setting, raising KeyError with the full key path if it is missing."""
        missing = object()
        value = self.get(*keys, default=missing)
        if value is missing:
            raise KeyError(f"Setting {'.'.join(keys)} is missing from {self.config_file}")
        return value

    def path(self, *keys):
        """Return an input path from the configuration, resolved against data_root."""
        value = Path(str(self.require(*keys))).expanduser()
        return value if value.is_absolute() else self.data_root / value

    def output_path(self, *keys, create_parent=True):
        """
        Return a generated-file path from the configuration, resolved against output_root.

        Parameters:
            *keys: str, key path under the outputs section, e.g. ('chip', 'whitelist_tidy_table')
            create_parent: bool, create the containing folder if it does not exist
        """
        value = Path(str(self.require('outputs', *keys))).expanduser()
        full_path = value if value.is_absolute() else self.output_root / value
        if create_parent:
            full_path.parent.mkdir(parents=True, exist_ok=True)
        return full_path

    def output_dir(self, *keys):
        """Return a generated-file folder from the configuration, creating it if needed."""
        folder = self.output_path(*keys, create_parent=False)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def age_bins(self):
        """Age bin edges, e.g. [0, 10, ..., 100]."""
        return list(self.require('age_groups', 'bin_edges'))

    @property
    def age_labels(self):
        """Age bin labels matching age_bins, e.g. ['0-9', '10-19', ...]."""
        edges = self.age_bins
        return [f'{low}-{high - 1}' for low, high in zip(edges[:-1], edges[1:])]

    def check_inputs(self):
        """
        Report which configured input paths exist.

        Checks every setting listed in INPUT_PATH_SETTINGS that is present in
        the configuration file.

        Returns:
            dict with keys 'found' and 'missing', each a list of (setting, path) tuples
        """
        found, missing = [], []
        for keys in INPUT_PATH_SETTINGS:
            if self.get(*keys) is None:
                continue
            full_path = self.path(*keys)
            (found if full_path.exists() else missing).append(('.'.join(keys), full_path))
        return {'found': found, 'missing': missing}

    def __repr__(self):
        return f"StudyConfig({self.config_file.name}, data_root={self.data_root})"


def load_config(config_file, data_root=None):
    """
    Read a study configuration file.

    Parameters:
        config_file: str or Path, path to a YAML configuration file
        data_root: str or Path or None, overrides the data_root in the file,
            which is useful when the same configuration is used on several machines

    Returns:
        StudyConfig
    """
    config_file = Path(config_file)
    with open(config_file, encoding='utf-8') as handle:
        settings = yaml.safe_load(handle) or {}
    return StudyConfig(settings, config_file, data_root=data_root)
