"""
SSc_Somatic: somatic mutation, CHIP and mutational signature analysis for
exome-sequenced cohorts, with a systemic sclerosis cohort as the worked example.

Typical use in a notebook:

    from SSc_Somatic import configure, load_config, load_study
    config = configure(load_config('config/example_SSc.yaml'))
    data = load_study(config, 'chip')
"""

from .config import StudyConfig, load_config
from .loading import StudyData, load_study
from .settings import configure

__all__ = ['StudyConfig', 'load_config', 'StudyData', 'load_study', 'configure']
