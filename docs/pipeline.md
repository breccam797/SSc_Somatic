# Pipeline

This page lists what each notebook reads and writes. Settings are named by their
location in the configuration file, for example `mutation.snv_table`. Input paths
are relative to `data_root` and generated files are written under `output_root`
(the `outputs` section). The format of each input is described in
[input_specification.md](input_specification.md).

Every notebook starts the same way: it reads the configuration, applies it with
`configure()`, and loads the tables for its analysis with `load_study()`.
`load_study()` reads the passing-sample list for the analysis, removes the samples
on the exclusion lists that are switched on in `exclusions`, reads the metadata for
the remaining samples and assigns each participant to the control group or one of
the two case subtypes.


## Mutation burden

### 01_find_passing_samples

Sample quality control on the ANNOVAR output of the main Mutect2 run.

- Reads: `inputs.metadata`, the exclusion lists in `inputs`, and the ANNOVAR
  multianno tables in `mutation.annovar_dir`.
- Classifies samples as empty, truncated (chrX missing or fewer than 13
  chromosomes with calls) or passing, and counts SNVs per sample.
- Writes: `mutation.passing_sample_ids`, `mutation.snv_table` (tab-separated) and a
  list of truncated samples in `outputs.mutation.sample_qc_dir`.
- The passing samples are the input to SigProfiler extraction and assignment,
  which are run outside the package before `21_signatures`.

### 02_build_variant_table

Builds the tidy variant table used for the burden analyses.

- Reads: the passing samples and gene constraint table through `load_study()`,
  `inputs.blood_sample_ids`, and the annotated VCFs in
  `mutation.annovar_filtered_dir`.
- Keeps samples sequenced from blood, combines their variants, keeps the first
  row for each variant across all samples, requires at least one supporting read
  on each strand, and adds gene pLI scores.
- Writes: `outputs.mutation.combined_variants` and `outputs.mutation.tidy_variants`.

### 03_burden_figures

Mutation burden analyses, enrichment and GWAS convergence.

- Reads: `outputs.mutation.tidy_variants`, the annotated sample table from
  `load_study()`, `reference.gene_annotation_gff`,
  `reference.gwas_summary_statistics` and `reference.magma_gene_locations`.
- Downloads the GO biological process gene-set library named in
  `enrichment.gseapy_library` through gseapy, so it needs internet access.
- Writes: figures to `outputs.mutation.figures_dir`, GO:BP enrichment tables to
  `outputs.mutation.enrichment_dir`, and FUMA and MAGMA input files to
  `outputs.mutation.gwas_validation_dir`.
- MAGMA is then run outside the package. The notebook reads its gene-set results
  from `outputs.mutation.magma_gene_set_results` to draw the MAGMA forest plot.


## CHIP

### 11_chip_tables

Builds the CHIP variant tables from the per-sample whitelist calls of the
CHmasked Mutect2 run.

- Reads: `chip.passing_sample_ids`, `chip.snv_table`, `inputs.metadata`,
  `inputs.blood_sample_ids`, and the per-sample files in `chip.whitelist_calls_dir`
  named by `chip.whitelist_file_pattern`.
- Filters calls on read depth, alternate allele depth and strand bias, removes
  duplicate calls, and annotates each call with participant metadata.
- Writes: `outputs.chip.processed_metadata`, `outputs.chip.combined_whitelist_calls`,
  `outputs.chip.whitelist_tidy_table` and `outputs.chip.whitelist_merged_table`.

### 12_chip_prevalence, 13_chip_genes, 14_chip_clinical

Analyses of the tables written by `11_chip_tables`, loaded through `load_study()`.

- `12_chip_prevalence`: CHIP prevalence by group and age, disease duration by CHIP
  status, and odds ratios adjusted for age. Participants aged `chip.max_age` or
  older are excluded from the age-stratified analyses.
- `13_chip_genes`: variant counts, VAF and age by gene for `chip.genes_of_interest`.
- `14_chip_clinical`: clinical measures and autoantibodies by CHIP status, and
  TET2 and DNMT3A regression analyses. Also reads `inputs.blood_sample_ids`.

These notebooks display their figures without saving them. Pass an
`output_filename` to a plotting function to save a figure.


## Mutational signatures

### 21_signatures

Runs after `01_find_passing_samples` and the SigProfiler extraction and
assignment of the passing samples.

- Reads: the passing samples and SNV table of the main run (written by
  `01_find_passing_samples`), `inputs.metadata`, `signatures.design_file` and the
  SigProfiler files it points to, `signatures.denovo_signature_profiles` and
  `signatures.cosmic_signature_profiles`.
- Compares signature activities across groups, tests associations with clinical
  measures in the full cohort and in the baseline cohort (disease duration up to
  `signatures.baseline_duration` years), and compares de novo with COSMIC
  signatures.
- Writes: figures to `outputs.signatures.figures_dir`. Results for de novo
  signatures are saved with a `denovo` prefix or suffix and results for
  COSMIC-assigned signatures with `cosmic_sbs`.
