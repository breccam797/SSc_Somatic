# Input specification

This page describes every input file the package reads. No data are distributed
with the repository, so these files must be prepared for each study. Each entry
names the configuration setting that points to the file. Paths are relative to
`data_root` unless they are absolute.

Column names in *italics* are set in the `cohort` section of the configuration;
the names shown are those of the SSc example. Column names in `code` are fixed.

Samples are identified by the same ID in every file. In sample lists and
variant tables the ID column is always named `sample_id`.


## Overview of the main inputs

This package analyzes variant calls that have already been made and annotated.
It does not call variants or run ANNOVAR itself. The main inputs are per-sample
output files of ANNOVAR `table_annovar.pl`, run on Mutect2 VCFs aligned to hg38
(GRCh38):

- **ANNOVAR multianno tables** (`.hg38_multianno.txt`), one per sample, read by
  `01_find_passing_samples` for sample QC and SNV counts.
- **ANNOVAR-annotated VCFs** (`.hg38_multianno.vcf`, or the dbSNP and COSMIC
  filtered version), one per sample, read by `02_build_variant_table` to build
  the variant table for the burden analyses.
- **CHIP whitelist tables**, one per sample: multianno tables restricted to CHIP
  whitelist variants, read by `11_chip_tables`.

The other inputs are tables of participant metadata, sample lists, SigProfiler
output and public reference files, described in the sections below.

### Running ANNOVAR

ANNOVAR must be run with `-vcfinput`, so that the original VCF record is kept
alongside the annotation, and with the databases below, so that the columns the
code reads are present. The analysis used the ANNOVAR release and `humandb`
databases available in September 2025.

| ANNOVAR database | Operation | Columns used by the code |
|---|---|---|
| `refGene` | `g` | `Func.refGene`, `Gene.refGene`, `GeneDetail.refGene`, `ExonicFunc.refGene`, `AAChange.refGene` |
| `cytoBand` | `r` | `cytoBand` |
| `exac03` | `f` | `ExAC_ALL` |
| `avsnp147` | `f` | `avsnp147` |
| `dbnsfp30a` | `f` | `SIFT_score`, `SIFT_pred`, `Polyphen2_HDIV_score`, `Polyphen2_HDIV_pred`, `CADD_phred` |
| `gnomad40_exome` | `f` | `gnomad40_exome_AF` and the population frequency columns listed in `ANNOVAR_FIELDS` |
| ClinVar (`clinvar_` database) | `f` | `CLNALLELEID`, `CLNDN`, `CLNDISDB`, `CLNREVSTAT`, `CLNSIG` |
| InterVar (`intervar_` database) | `f` | `InterVar_automated` and the ACMG criteria `PVS1` to `BP7` |

The first five databases are needed for `01_find_passing_samples` and the CHIP
tables. The gnomAD v4, ClinVar and InterVar columns are needed in the annotated
VCFs read by `02_build_variant_table`. An example command for the first five
databases:

```bash
table_annovar.pl {sample_id}_Mutect2_filtered.vcf.gz humandb/ \
    -buildver hg38 -out myanno_{sample_id} -remove \
    -protocol refGene,cytoBand,exac03,avsnp147,dbnsfp30a \
    -operation g,r,f,f,f -nastring . -vcfinput -polish
```

With `-out myanno_{sample_id}` and `-buildver hg38`, ANNOVAR writes
`myanno_{sample_id}.hg38_multianno.txt` and `myanno_{sample_id}.hg38_multianno.vcf`,
which match the default `annovar_file_prefix`, `annovar_file_suffix` and
`annovar_vcf_suffix` settings. Missing values are written as `.` (`-nastring .`).

With `-vcfinput`, the columns `Otherinfo4` to `Otherinfo13` of the multianno table
hold the ten columns of the input VCF record (CHROM to the sample column). The
code reads `Otherinfo11` (INFO), `Otherinfo12` (FORMAT) and `Otherinfo13` (the
sample's values). Other ANNOVAR releases may number these columns differently,
so check the header of one output file before running the notebooks.

The VCFs were produced with GATK Mutect2 and FilterMutectCalls (GATK 4.2.4.0) in
tumor-only mode, with one sample per VCF. The FORMAT field must contain `AD` and
`SB`. `01_find_passing_samples` reads them by position and expects the Mutect2
order `GT:AD:AF:DP:F1R2:F2R1:FAD:SB`, with `AD` second and `SB` eighth.

### Example of the file format

The values below are invented to show the format of each column. They do not
come from any sample, and the gene and transcript names are placeholders.

One row of a multianno table, shown as column and value pairs. Only the columns
the code reads are shown; a real file has more columns, in the order set by the
`-protocol` list.

| Column | Example value |
|---|---|
| `Chr` | `chr1` |
| `Start` | `1234567` |
| `End` | `1234567` |
| `Ref` | `C` |
| `Alt` | `T` |
| `Func.refGene` | `exonic` |
| `Gene.refGene` | `GENE1` |
| `ExonicFunc.refGene` | `nonsynonymous SNV` |
| `AAChange.refGene` | `GENE1:NM_000000:exon5:c.C431T:p.A144V` |
| `cytoBand` | `1p36.33` |
| `ExAC_ALL` | `.` |
| `avsnp147` | `.` |
| `SIFT_score` | `0.01` |
| `SIFT_pred` | `D` |
| `Polyphen2_HDIV_score` | `0.998` |
| `Polyphen2_HDIV_pred` | `D` |
| `CADD_phred` | `24.3` |
| `Otherinfo11` | `DP=89;ECNT=1;MBQ=20,20;MMQ=60,60;MPOS=30;POPAF=7.30;TLOD=25.71` |
| `Otherinfo12` | `GT:AD:AF:DP:F1R2:F2R1:FAD:SB` |
| `Otherinfo13` | `0/1:78,11:0.128:89:38,6:40,5:78,11:40,38,6,5` |

In this row, `DP` is 89, `AD` is 78 reference and 11 alternate reads, and `SB`
gives the reference forward, reference reverse, alternate forward and alternate
reverse read counts (40, 38, 6 and 5).

The same variant in an ANNOVAR-annotated VCF. ANNOVAR adds its annotations to the
INFO field between `ANNOVAR_DATE` and `ALLELE_END`; the INFO field is shortened
here with `...`:

```
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	SAMPLE
chr1	1234567	.	C	T	.	PASS	DP=89;...;ANNOVAR_DATE=...;Func.refGene=exonic;Gene.refGene=GENE1;...;gnomad40_exome_AF=.;...;CLNSIG=.;InterVar_automated=Uncertain_significance;PVS1=0;...;PM2=1;...;BP7=0;ALLELE_END	GT:AD:AF:DP:F1R2:F2R1:FAD:SB	0/1:78,11:0.128:89:38,6:40,5:78,11:40,38,6,5
```

The columns are tab-separated. `02_build_variant_table` reads the INFO keys by
name, so their order does not matter.


## Participant metadata

**`inputs.metadata`**: comma-separated, one row per participant.

| Column | Content |
|---|---|
| *Patient_ID* (`metadata_sample_column`) | Participant ID, matching `sample_id` in the other files |
| *statustype* (`status_column`) | Case or control status; values *Index Case* (`case_label`) and *Control* (`control_label`) |
| *SSc Type* (`subtype_column`) | Case subtype; values listed in `subtypes` (*Limited*, *Diffuse*) |
| *Age* | Age at sampling, in years |
| *Disease Duration* | Years since diagnosis |
| *MRSS* | Modified Rodnan skin score |
| *FVC %predicted*, *DLCO %Predicted* | Pulmonary function, percent of predicted |
| *race*, *Ethnicity* | Used for race and ethnicity categories; see below |

The SSc-specific analyses also use these columns:

- `ana`, `anti-centromere`, `anti-topo` and `anti-RNA polymerase III`:
  autoantibody status, used by the CHIP clinical and signature analyses.
- `gender`: tested in the CHIP clinical analysis.

Race and ethnicity are combined into categories in the mutation burden and
signature analyses: `White` participants are split into `White Hispanic` and
`White Non-Hispanic` using the *Ethnicity* column, `Black` is kept, and other
values are grouped (`Asian` and `Other` in the burden analysis, `Other` in the
signature analysis). The *race* and *Ethnicity* values the code recognizes are
set in `create_analysis_df` in `burden/analysis.py` and `assign_race_ethnicity`
in `signatures/analysis.py`.


## Sample lists

Each file has a `sample_id` column. Other columns are ignored.

| Setting | Content | Separator |
|---|---|---|
| `inputs.panel_of_normals_samples` | Control samples used to build the Mutect2 panel of normals | tab |
| `inputs.multi_antibody_samples` | Samples flagged for multiple antibody positivity | comma |
| `inputs.low_plate_samples` | Low-plate list: samples with fewer than 500 called variants | comma |
| `inputs.blood_sample_ids` | Samples sequenced from blood | comma |
| `chip.passing_sample_ids` | Samples that passed QC in the CHmasked run | comma |
| `mutation.passing_sample_ids` | Samples that passed QC in the main run; written by `01_find_passing_samples` | comma |

Whether a list is applied is set in the `exclusions` section of the configuration.


## SNV count tables

**`chip.snv_table`** and **`mutation.snv_table`**: one row per sample, with a
`sample_id` column and per-sample SNV counts. `01_find_passing_samples` writes the
mutation table with the columns `sample_id`, `synonymous SNV` and
`nonsynonymous SNV`. The separator is set by `snv_table_separator` in the same
configuration section.


## Variant calls

The ANNOVAR settings that produce these files and an example of their format are
given in [Overview of the main inputs](#overview-of-the-main-inputs).

**ANNOVAR multianno tables** (`mutation.annovar_dir`): one tab-separated file per
sample, named `{annovar_file_prefix}{sample_id}{annovar_file_suffix}`. The QC step
uses standard ANNOVAR columns, including `Chr`, `Otherinfo11` (the VCF INFO field,
for total depth) and `Otherinfo13` (the VCF sample field, for allele depths and
strand bias).

**ANNOVAR-annotated VCFs** (`mutation.annovar_filtered_dir`): one file per sample,
named `{annovar_file_prefix}{sample_id}{annovar_vcf_suffix}`, or with
`annovar_filtered_vcf_suffix` for dbSNP and COSMIC filtered files. The INFO field
carries the ANNOVAR annotations listed in `ANNOVAR_FIELDS` in
`burden/vcf_processing.py` (RefSeq gene annotation, gnomAD v4 exome frequencies,
ClinVar and InterVar), and the FORMAT field should include `AD` and `SB`.

**CHIP whitelist calls** (`chip.whitelist_calls_dir`): one tab-separated ANNOVAR
table per sample, restricted to the CHIP whitelist variants of Vlasschaert, Bick
and colleagues (2023, see below) and named by
`chip.whitelist_file_pattern`. Columns used: `Chr`, `Start`, `Gene.refGene`,
`Func.refGene`, `ExonicFunc.refGene`, `AAChange.refGene`, `Otherinfo11` (INFO,
for depth) and `Otherinfo12` and `Otherinfo13` (FORMAT and sample fields, for
allele depths and strand bias). Before restriction to the whitelist, the
multianno tables were filtered for read depth of at least 20, at least 3 reads
supporting the alternate allele, and at least one alternate read in each of the
F1R2 and F2R1 orientations, following Vlasschaert, Bick and colleagues (2023,
[doi:10.1182/blood.2022018825](https://doi.org/10.1182/blood.2022018825)), and restricted to clonal hematopoiesis
genes.


## Mutational signatures

**`signatures.design_file`**: comma-separated, one row per SigProfiler run. The
first row describes the SBS run and the second the indel run.

| Column | Content |
|---|---|
| `sig_id` | Run identifier, for example `SBS_96` |
| `solution_sigs` | Number of signatures in the chosen solution |
| `extraction_path` | SigProfilerExtractor output folder for the run |
| `assignment_path` | SigProfilerAssignment output folder for the run, ending in a path separator |
| `input_matrix` | Mutation matrix given to SigProfiler |

The package reads these files from the paths in the design file:

- Extraction activities:
  `{extraction_path}/All_Solutions/{sig_id without underscores}_{solution_sigs}_Signatures/Activities/{sig_id without underscores}_S{solution_sigs}_NMF_Activities.txt`
- Assignment activities: `{assignment_path}Activities/Decompose_Solution_Activities.txt`

Activity files are tab-separated with a `Samples` column. The sample ID is
recovered by splitting each sample name on `signatures.sample_name_delimiter` and
taking the part at `signatures.sample_name_position`. Mutation matrices are
tab-separated, with one column per sample named with the sample ID followed by
`signatures.matrix_column_suffix`.

**`signatures.denovo_signature_profiles`** and
**`signatures.cosmic_signature_profiles`**: SigProfiler signature profile files,
tab-separated with a `MutationType` column and one column per signature.


## Reference files

| Setting | Content |
|---|---|
| `reference.gene_constraint` | gnomAD gene constraint table (tab-separated) with `gene` and `lof.pLI` columns |
| `reference.gene_annotation_gff` | RefSeq GFF3 gene annotation; CDS features are used to compute coding length per gene |
| `reference.magma_gene_locations` | MAGMA gene location file (`NCBI37.3.gene.loc`), whitespace-separated |
| `reference.gwas_summary_statistics` | GWAS summary statistics, whitespace-separated, with the columns named in the `gwas` section of the configuration |


## Files produced outside the package

**`outputs.mutation.magma_gene_set_results`**: the MAGMA gene-set analysis output
(`.gsa.out`) for the gene-set file written by `03_burden_figures`, with `VARIABLE`,
`BETA`, `SE` and `P` columns.
