"""
VCF processing and tidy table generation functions for SSc mutation burden analysis.
Handles ANNOVAR-annotated VCF parsing, variant filtering, gnomAD cache management,
and VAF threshold calculation.

Functions organized by tidy table notebook workflow in chronological order of use.
"""

import os
from collections import defaultdict

import pandas as pd


def extract_dp(info_str):
    """Extract total depth (DP) from the Otherinfo11 INFO field string."""
    for entry in info_str.split(';'):
        if entry.startswith('DP='):
            try:
                return int(entry.split('=')[1])
            except ValueError:
                return None
    return None


def extract_ad_sb(info_str):
    """
    Extract allele depth and strand bias from the Otherinfo13 FORMAT field string.
    Expects colon-delimited fields with AD at index 1 and SB at index 7.

    Returns:
        Tuple of (ad_list, sb_list) where each is [ref_count, alt_count]
    """
    try:
        values = info_str.split(':')
        ad = list(map(int, values[1].split(',')))
        sb = list(map(int, values[7].split(',')))
        return ad, sb
    except (IndexError, ValueError):
        return [None, None], [None, None]
    
def find_sample_file(sample_id, file_dir, prefix='myanno_', suffix='.hg38_multianno.txt'):
    """
    Locate a file for a sample, checking multiple naming patterns including
    _AMR and _AFR population suffixes and _CHmasked variants.

    Parameters:
        sample_id: str, sample identifier
        file_dir: str, directory containing the files
        prefix: str, filename prefix before sample_id (default: 'myanno_')
        suffix: str, filename suffix after sample_id (default: '.hg38_multianno.txt')

    Returns:
        Tuple of (file_path, pattern_name) if found, (None, None) otherwise
    """
    patterns = [
        f'{prefix}{sample_id}{suffix}',
        f'{prefix}{sample_id}_AMR{suffix}',
        f'{prefix}{sample_id}_AFR{suffix}',
        f'{prefix}{sample_id}_CHmasked{suffix}',
        f'{prefix}{sample_id}_AMR_CHmasked{suffix}',
        f'{prefix}{sample_id}_AFR_CHmasked{suffix}',
        f'{prefix}{sample_id}_AMR_AFR_dbsnp_cosmic_filt{suffix}'
    ]
    for pattern in patterns:
        file_path = os.path.join(file_dir, pattern)
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return file_path, pattern
    return None, None


def process_genomic_data(design_df, annovar_vcf_root, extract_dp_func, extract_ad_sb_func,
                         file_prefix='myanno_', file_suffix='.hg38_multianno.txt'):
    """
    Process ANNOVAR-annotated multianno files to classify samples by QC status
    and extract variant-level metrics for passing samples.

    Samples are classified as:
        - empty: no file found or file unreadable
        - truncated: missing chrX or fewer than 13 chromosomes represented
        - all_chr: all 24 chromosomes (1-22, X, Y) present
        - passing: at least 13 chromosomes and chrX present

    Parameters:
        design_df: DataFrame with 'sample_id' column listing all samples
        annovar_vcf_root: str, path to directory with multianno files
        extract_dp_func: callable, extracts total depth from Otherinfo11 field
        extract_ad_sb_func: callable, extracts allele depth and strand bias from Otherinfo13 field
        file_prefix: str, text before the sample ID in multianno file names (default: 'myanno_')
        file_suffix: str, text after the sample ID in multianno file names (default: '.hg38_multianno.txt')

    Returns:
        dict with keys:
            Sample classification lists:
                empty_samples, truncated_samples, all_chr_samples, passing_samples
            Variant metric lists (passing samples only):
                total_depth_list, ref_allele_depth_list, alt_allele_depth_list,
                strand_bias_list, vaf_list, vaf_sample_list,
                functional_class_list, variant_gene_list, chr_loc_list,
                sift_score_list, sift_pred_list, polyphen_score_list
            Per-sample SNV counts:
                snv_count_list (list of dicts with sample_id, synonymous SNV, nonsynonymous SNV)
            Chromosome coverage data:
                truncated_chr_data, passing_chr_data
            Debugging:
                file_pattern_used (dict mapping sample_id to filename pattern)
    """
    set_chrs = set(['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY'])

    empty_samples = []
    truncated_samples = []
    all_chr_samples = []
    passing_samples = []
    total_depth_list = []
    ref_allele_depth_list = []
    alt_allele_depth_list = []
    strand_bias_list = []
    vaf_list = []
    vaf_sample_list = []
    functional_class_list = []
    variant_gene_list = []
    snv_count_list = []
    chr_loc_list = []
    sift_score_list = []
    sift_pred_list = []
    polyphen_score_list = []
    truncated_chr_data = []
    passing_chr_data = []
    file_pattern_used = {}
    files_processed = 0

    for i, row in design_df.iterrows():
        sample_id = row['sample_id']
        if i % 1000 == 0:
            print(f"Processing sample {i+1}/{len(design_df)}: {sample_id}")
        file_path, pattern_used = find_sample_file(sample_id, annovar_vcf_root, prefix=file_prefix, suffix=file_suffix)
        if file_path is None:
            empty_samples.append(sample_id)
            continue
        file_pattern_used[sample_id] = pattern_used
        files_processed += 1
        if files_processed <= 5:
            print(f"Found file for {sample_id}: {pattern_used}")
        try:
            df = pd.read_csv(file_path, sep='\t')
            if files_processed <= 3:
                print(f"  Shape: {df.shape}, Chromosomes: {sorted(df['Chr'].unique())}")
            df_chrs = set(df['Chr'])
            if set_chrs.issubset(df_chrs):
                all_chr_samples.append(sample_id)
            if 'chrX' not in df_chrs:
                truncated_samples.append(sample_id)
                truncated_chr_data.extend(df['Chr'].tolist())
            elif len(df['Chr'].unique()) >= 13:
                passing_samples.append(sample_id)
                passing_chr_data.extend(df['Chr'].tolist())
                snv_counts = df['ExonicFunc.refGene'].value_counts()
                snv_count_list.append({
                    'sample_id': sample_id,
                    'synonymous SNV': snv_counts.get('synonymous SNV', 0),
                    'nonsynonymous SNV': snv_counts.get('nonsynonymous SNV', 0)
                })
                df['DP'] = df['Otherinfo11'].apply(extract_dp_func)
                df[['AD', 'SB']] = df['Otherinfo13'].apply(
                    lambda x: pd.Series(extract_ad_sb_func(x))
                )
                chr_str = df['Chr'].astype(str).str.replace('chr', '')
                chr_loc = chr_str + '-' + df['Start'].astype(str) + '-' + df['Ref'].astype(str) + '-' + df['Alt'].astype(str)
                sift_score_list.extend(df['SIFT_score'].dropna())
                sift_pred_list.extend(df['SIFT_pred'].dropna())
                polyphen_score_list.extend(df['Polyphen2_HDIV_score'].dropna())
                total_depth_list.extend(df['DP'].dropna())
                functional_class_list.extend(df['ExonicFunc.refGene'].dropna())
                variant_gene_list.extend(df['Gene.refGene'].dropna())
                ref_allele_depth_list.extend([ad[0] for ad in df['AD'] if ad[0] is not None])
                alt_allele_depth_list.extend([ad[1] for ad in df['AD'] if ad[1] is not None])
                computed_vafs = [
                    ad[1] / (ad[0] + ad[1]) if ad[0] is not None and ad[1] is not None else None
                    for ad in df['AD']
                ]
                vaf_list.extend(computed_vafs)
                vaf_sample_list.extend([sample_id] * len(computed_vafs))
                chr_loc_list.extend(chr_loc)
            else:
                truncated_samples.append(sample_id)
                truncated_chr_data.extend(df['Chr'].tolist())
        except Exception as e:
            print(f"Error processing {sample_id}: {str(e)}")
            empty_samples.append(sample_id)

    print(f"\nFile processing summary:")
    print(f"Total files processed: {files_processed}")
    if file_pattern_used:
        pattern_counts = {}
        for pattern in file_pattern_used.values():
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        for pattern, count in pattern_counts.items():
            print(f"  {pattern}: {count} files")
    print(f"Sample classification: {len(empty_samples)} empty, {len(truncated_samples)} truncated, "
          f"{len(all_chr_samples)} all-chr, {len(passing_samples)} passing")

    return {
        'empty_samples': empty_samples,
        'truncated_samples': truncated_samples,
        'all_chr_samples': all_chr_samples,
        'passing_samples': passing_samples,
        'total_depth_list': total_depth_list,
        'ref_allele_depth_list': ref_allele_depth_list,
        'alt_allele_depth_list': alt_allele_depth_list,
        'strand_bias_list': strand_bias_list,
        'vaf_list': vaf_list,
        'vaf_sample_list': vaf_sample_list,
        'functional_class_list': functional_class_list,
        'variant_gene_list': variant_gene_list,
        'snv_count_list': snv_count_list,
        'chr_loc_list': chr_loc_list,
        'sift_score_list': sift_score_list,
        'sift_pred_list': sift_pred_list,
        'polyphen_score_list': polyphen_score_list,
        'truncated_chr_data': truncated_chr_data,
        'passing_chr_data': passing_chr_data,
        'file_pattern_used': file_pattern_used,
    }


def extract_passing_sample_analysis(genomic_results, ssc_ids, normal_ids):
    """
    Analyze passing sample QC results: build SNV count dataframes,
    compute summary statistics, and identify samples needing realignment.

    Parameters:
        genomic_results: dict, output from process_genomic_data()
        ssc_ids: list, SSc case sample IDs
        normal_ids: list, control sample IDs

    Returns:
        dict with keys:
            snv_count_df: DataFrame of SNV counts for all processed samples
            snv_count_passing_df: filtered to passing samples only
            snv_count_passing_ssc_df: passing SSc cases
            snv_count_passing_normal_df: passing controls
            vaf_df: DataFrame with per-variant VAF, depth, functional class, and annotations
            realign_df: DataFrame of empty + truncated samples for realignment
            chr_counts: Series of chromosome frequencies in truncated samples
            chr_count_passing: Series of chromosome frequencies in passing samples
            snv_passing_ids: list of passing sample IDs
    """
    snv_count_df = pd.DataFrame(genomic_results['snv_count_list'])
    chr_counts = pd.Series(genomic_results['truncated_chr_data']).value_counts()
    chr_count_passing = pd.Series(genomic_results['passing_chr_data']).value_counts()
    vaf_df = pd.DataFrame({
        'sample_id': genomic_results['vaf_sample_list'],
        'gene_id': genomic_results['variant_gene_list'],
        'VAF': genomic_results['vaf_list'],
        'total_DP': genomic_results['total_depth_list'],
        'class': genomic_results['functional_class_list'],
        'variant_location': genomic_results['chr_loc_list'],
        'sift_score': genomic_results['sift_score_list'],
        'sift_pred': genomic_results['sift_pred_list'],
        'polyphen_score': genomic_results['polyphen_score_list'],
    })
    passing_samples = genomic_results['passing_samples']
    snv_count_passing_df = snv_count_df.loc[snv_count_df['sample_id'].isin(passing_samples)]
    snv_count_passing_ssc_df = snv_count_passing_df.loc[snv_count_passing_df['sample_id'].isin(ssc_ids)]
    snv_count_passing_normal_df = snv_count_passing_df.loc[snv_count_passing_df['sample_id'].isin(normal_ids)]
    print(f"Empty samples: {len(genomic_results['empty_samples'])}")
    print(f"Truncated samples: {len(genomic_results['truncated_samples'])}")
    print(f"Passing samples: {len(passing_samples)}")
    print(f"All-chromosome samples: {len(genomic_results['all_chr_samples'])}")
    for label, sub_df in [('SSc cases', snv_count_passing_ssc_df), ('Controls', snv_count_passing_normal_df)]:
        print(f"\nSummary statistics for passing {label}:")
        for col in ['synonymous SNV', 'nonsynonymous SNV']:
            print(f"  {col}: mean={sub_df[col].mean():.1f}, "
                  f"median={sub_df[col].median():.1f}, sd={sub_df[col].std():.1f}")
    empty_samples = genomic_results['empty_samples']
    truncated_samples = genomic_results['truncated_samples']
    realign_samples = empty_samples + truncated_samples
    realign_categories = ['empty'] * len(empty_samples) + ['truncated'] * len(truncated_samples)
    realign_df = pd.DataFrame({'sample_id': realign_samples, 'category': realign_categories})
    snv_passing_ids = snv_count_passing_df['sample_id'].tolist()

    return {
        'snv_count_df': snv_count_df,
        'snv_count_passing_df': snv_count_passing_df,
        'snv_count_passing_ssc_df': snv_count_passing_ssc_df,
        'snv_count_passing_normal_df': snv_count_passing_normal_df,
        'vaf_df': vaf_df,
        'realign_df': realign_df,
        'chr_counts': chr_counts,
        'chr_count_passing': chr_count_passing,
        'snv_passing_ids': snv_passing_ids,
    }


def export_truncated_samples(genomic_results, output_dir, date_str):
    """
    Export empty and truncated sample IDs to CSV for realignment.

    Parameters:
        genomic_results: dict, output from process_genomic_data()
        output_dir: str, directory for output file
        date_str: str, date stamp for filename (e.g. '03_05_26')

    Returns:
        str, path to exported CSV file
    """
    truncated_samples = genomic_results['truncated_samples']
    realign_df = pd.DataFrame({
        'sample_id': truncated_samples,
        'category': ['truncated'] * len(truncated_samples),
    })
    filepath = os.path.join(output_dir, f'empty_samples_to_realign_{date_str}.csv')
    realign_df.to_csv(filepath, index=False)
    print(f"Exported {len(truncated_samples)} truncated samples to: {filepath}")
    return filepath


def save_passing_sample_ids(analysis_results, outpath):
    """
    Write passing sample IDs to a text file with a header, one per line.

    Parameters:
        analysis_results: dict, output from extract_passing_sample_analysis()
        outpath: str, output file path

    Returns:
        str, path to created file
    """
    passing_samples = analysis_results['snv_count_passing_df']['sample_id'].tolist()
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    with open(outpath, 'w') as f:
        f.write("sample_id\n")
        for sample_id in passing_samples:
            f.write(f"{sample_id}\n")
    print(f"Saved {len(passing_samples)} passing sample IDs to: {outpath}")
    return outpath


def save_passing_snv_df(analysis_results, outpath):
    """
    Save the SNV count DataFrame for passing samples to a tab-delimited file.

    Parameters:
        analysis_results: dict, output from extract_passing_sample_analysis()
        outpath: str, output file path

    Returns:
        str, path to created file
    """
    snv_df = analysis_results['snv_count_passing_df']
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    snv_df.to_csv(outpath, sep='\t', index=False)
    print(f"Saved SNV data for {len(snv_df)} passing samples to: {outpath}")
    return outpath


# Set annotation fields to expect
ANNOVAR_FIELDS = [
    'Chr', 'Start', 'End', 'Ref', 'Alt',
    'Func.refGene', 'Gene.refGene', 'GeneDetail.refGene',
    'ExonicFunc.refGene', 'AAChange.refGene', 'cytoBand',
    'gnomad40_exome_AF', 'gnomad40_exome_AF_raw',
    'gnomad40_exome_AF_XX', 'gnomad40_exome_AF_XY',
    'gnomad40_exome_AF_grpmax', 'gnomad40_exome_faf95', 'gnomad40_exome_faf99',
    'gnomad40_exome_fafmax_faf95_max', 'gnomad40_exome_fafmax_faf99_max',
    'gnomad40_exome_AF_afr', 'gnomad40_exome_AF_amr', 'gnomad40_exome_AF_asj',
    'gnomad40_exome_AF_eas', 'gnomad40_exome_AF_fin', 'gnomad40_exome_AF_mid',
    'gnomad40_exome_AF_nfe', 'gnomad40_exome_AF_remaining', 'gnomad40_exome_AF_sas',
    'avsnp147',
    'CLNALLELEID', 'CLNDN', 'CLNDISDB', 'CLNREVSTAT', 'CLNSIG',
    'InterVar_automated',
    'PVS1', 'PS1', 'PS2', 'PS3', 'PS4',
    'PM1', 'PM2', 'PM3', 'PM4', 'PM5', 'PM6',
    'PP1', 'PP2', 'PP3', 'PP4', 'PP5',
    'BA1', 'BS1', 'BS2', 'BS3', 'BS4',
    'BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7',
    'Otherinfo1', 'Otherinfo2', 'Otherinfo3', 'Otherinfo4', 'Otherinfo5',
    'Otherinfo6', 'Otherinfo7', 'Otherinfo8', 'Otherinfo9', 'Otherinfo10',
    'Otherinfo11', 'Otherinfo12', 'Otherinfo13',
]

ACMG_FIELDS = [
    'PVS1', 'PS1', 'PS2', 'PS3', 'PS4',
    'PM1', 'PM2', 'PM3', 'PM4', 'PM5', 'PM6',
    'PP1', 'PP2', 'PP3', 'PP4', 'PP5',
    'BA1', 'BS1', 'BS2', 'BS3', 'BS4',
    'BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7',
]

NUMERIC_VARIANT_COLS = [
    'POS', 'exac_freq', 'cadd_phred', 'pLI', 'gnomad_exome_freq', 'vaf',
    'gnomad40_exome_AF_raw', 'gnomad40_exome_AF_XX', 'gnomad40_exome_AF_XY',
    'gnomad40_exome_AF_grpmax', 'gnomad40_exome_faf95', 'gnomad40_exome_faf99',
    'gnomad40_exome_fafmax_faf95_max', 'gnomad40_exome_fafmax_faf99_max',
    'gnomad40_exome_AF_afr', 'gnomad40_exome_AF_amr', 'gnomad40_exome_AF_asj',
    'gnomad40_exome_AF_eas', 'gnomad40_exome_AF_fin', 'gnomad40_exome_AF_mid',
    'gnomad40_exome_AF_nfe', 'gnomad40_exome_AF_remaining', 'gnomad40_exome_AF_sas',
]

TEXT_VARIANT_COLS = [
    'aachange_gene', 'aachange_transcript', 'aachange_exon',
    'aachange_cdna', 'aachange_protein', 'aachange_full',
    'gene', 'func', 'sift', 'polyphen', 'clndn', 'clnsig', 'intervar_automated',
]


def parse_aachange(aachange_str):
    """
    Parse AAChange.refGene string into separate components.
    Takes the first (primary) transcript from comma-separated entries.
    Expected format: Gene:Transcript:Exon:cDNA_change:Protein_change

    Parameters:
        aachange_str: str, raw AAChange.refGene value

    Returns:
        dict with keys: aachange_gene, aachange_transcript, aachange_exon,
        aachange_cdna, aachange_protein
    """
    empty = {
        'aachange_gene': '.', 'aachange_transcript': '.',
        'aachange_exon': '.', 'aachange_cdna': '.', 'aachange_protein': '.',
    }
    if not aachange_str or aachange_str == '.':
        return empty
    components = aachange_str.split(',')[0].strip().split(':')
    keys = ['aachange_gene', 'aachange_transcript', 'aachange_exon',
            'aachange_cdna', 'aachange_protein']
    return {k: (components[i] if i < len(components) else '.') for i, k in enumerate(keys)}


def read_vcf(filename):
    """
    Read an ANNOVAR-annotated VCF file and extract variant-level data including
    INFO annotations, allele depths, VAF, ClinVar fields, InterVar, and ACMG criteria.

    Parameters:
        filename: str, path to .vcf file

    Returns:
        list of dicts, each containing variant fields: CHROM, POS, REF, ALT,
        INFO (dict), VAF, AD, SB, gnomad_exome_freq, ClinVar fields,
        InterVar_automated, and ACMG_criteria (dict)
    """
    variants = []
    with open(filename) as f:
        for line in f:
            if line.startswith('#'):
                continue
            try:
                fields = line.strip().split('\t')
                chrom = fields[0]
                pos = fields[1]
                ref = fields[3]
                alt = fields[4]
                info = fields[7]
                format_field = fields[8] if len(fields) > 8 else None
                sample_field = fields[9] if len(fields) > 9 else None
                info_dict = {}
                gnomad_freq = None
                clnalleleid = None
                clndn = None
                clndisdb = None
                clnrevstat = None
                clnsig = None
                intervar_automated = None
                acmg_criteria = {}
                if info != '.' and isinstance(info, str):
                    for item in info.split(';'):
                        if '=' not in item:
                            continue
                        key, value = item.split('=', 1)
                        info_dict[key] = str(value)
                        if key == 'gnomad40_exome_AF':
                            gnomad_freq = float(value) if value != '.' else None
                        elif key == 'CLNALLELEID':
                            clnalleleid = value
                        elif key == 'CLNDN':
                            clndn = value
                        elif key == 'CLNDISDB':
                            clndisdb = value
                        elif key == 'CLNREVSTAT':
                            clnrevstat = value
                        elif key == 'CLNSIG':
                            clnsig = value
                        elif key == 'InterVar_automated':
                            intervar_automated = value
                        elif key in ACMG_FIELDS:
                            acmg_criteria[key] = value
                vaf = None
                ad_values = None
                sb_values = None
                if format_field and sample_field:
                    format_keys = format_field.split(':')
                    sample_values = sample_field.split(':')
                    format_dict = dict(zip(format_keys, sample_values))
                    if 'AD' in format_dict:
                        try:
                            ad_values = [int(x) for x in format_dict['AD'].split(',')]
                            if len(ad_values) >= 2 and sum(ad_values) > 0:
                                vaf = ad_values[1] / sum(ad_values)
                        except (ValueError, IndexError):
                            ad_values = None
                    if 'SB' in format_dict:
                        try:
                            sb_values = [int(x) for x in format_dict['SB'].split(',')]
                        except (ValueError, IndexError):
                            sb_values = None
                variants.append({
                    'CHROM': chrom,
                    'POS': pos,
                    'REF': ref,
                    'ALT': alt,
                    'INFO': info_dict,
                    'VAF': round(vaf, 5) if vaf is not None else None,
                    'AD': ad_values,
                    'SB': sb_values,
                    'gnomad_exome_freq': gnomad_freq,
                    'CLNALLELEID': clnalleleid,
                    'CLNDN': clndn,
                    'CLNDISDB': clndisdb,
                    'CLNREVSTAT': clnrevstat,
                    'CLNSIG': clnsig,
                    'InterVar_automated': intervar_automated,
                    'ACMG_criteria': acmg_criteria,
                })
            except Exception as e:
                print(f"Error processing line in {filename}: {str(e)}")
                continue
    return variants


def process_vcfs(vcf_dir, output_file, sample_list_input, dbsnp_cosmic_filt=False, constraint_df=None,
                 file_prefix='myanno_', vcf_suffix='.hg38_multianno.vcf',
                 filtered_vcf_suffix='.hg38_multianno_dbsnp_cosmic_filt.vcf'):
    """
    Combine multiple ANNOVAR-annotated VCF files into a single variant dictionary.
    Parses INFO fields, VAF, ClinVar annotations, ACMG criteria, and AAChange components.

    Parameters:
        vcf_dir: str, directory containing annotated .vcf files
        output_file: str, output file path (reserved for future use)
        sample_list_input: list, sample IDs to process
        dbsnp_cosmic_filt: bool, if True look for dbSNP/COSMIC filtered VCFs (default: False)
        constraint_df: DataFrame, optional gnomAD constraint data (reserved for downstream use)
        file_prefix: str, text before the sample ID in VCF file names (default: 'myanno_')
        vcf_suffix: str, text after the sample ID in unfiltered VCF file names
        filtered_vcf_suffix: str, text after the sample ID in dbSNP/COSMIC filtered VCF file names

    Returns:
        Tuple of (variants, samples) where variants is a nested dict
        keyed by chrom_pos -> sample_id -> variant attributes,
        and samples is a set of processed sample IDs
    """
    if dbsnp_cosmic_filt:
        suffix = filtered_vcf_suffix
    else:
        suffix = vcf_suffix

    print(f"Processing VCF files from: {vcf_dir}")
    print(f"Processing {len(sample_list_input)} samples (suffix: {suffix})")
    if constraint_df is not None and hasattr(constraint_df, 'columns'):
        print(f"Constraint columns: {list(constraint_df.columns)}")

    variants = defaultdict(lambda: defaultdict(dict))
    samples = set()
    processed_count = 0
    error_count = 0

    for i, sample_id in enumerate(sample_list_input):
        filepath, pattern_used = find_sample_file(sample_id, vcf_dir, prefix=file_prefix, suffix=suffix)
        if filepath is None:
            continue
        try:
            vcf_variants = read_vcf(filepath)
            samples.add(sample_id)
            processed_count += 1
            if (i + 1) % 500 == 0 or i == 0:
                print(f"Processing {pattern_used}... ({i+1}/{len(sample_list_input)})")
            for variant in vcf_variants:
                info_dict = variant.get('INFO', {})
                if not isinstance(info_dict, dict):
                    info_dict = {}
                pos_key = f"{variant['CHROM']}_{variant['POS']}"
                aachange_str = str(info_dict.get('AAChange.refGene', '.'))
                aachange_components = parse_aachange(aachange_str)
                variant_data = {
                    'ref': str(variant['REF']),
                    'alt': str(variant['ALT']),
                    'gene': str(info_dict.get('Gene.refGene', 'Unknown')),
                    'func': str(info_dict.get('ExonicFunc.refGene', '.')),
                    'exac_freq': str(info_dict.get('ExAC_ALL', '.')),
                    'cadd_phred': str(info_dict.get('CADD_phred', '.')),
                    'sift': str(info_dict.get('SIFT_pred', '.')),
                    'polyphen': str(info_dict.get('Polyphen2_HDIV_pred', '.')),
                    'pLI': '.',
                    'gnomad_exome_freq': str(info_dict.get('gnomad40_exome_AF', '.')),
                    'vaf': str(variant.get('VAF', '.')),
                    'ad': str(variant.get('AD', '.')),
                    'sb': str(variant.get('SB', '.')),
                    'aachange_gene': str(aachange_components['aachange_gene']),
                    'aachange_transcript': str(aachange_components['aachange_transcript']),
                    'aachange_exon': str(aachange_components['aachange_exon']),
                    'aachange_cdna': str(aachange_components['aachange_cdna']),
                    'aachange_protein': str(aachange_components['aachange_protein']),
                    'aachange_full': aachange_str,
                    'clnalleleid': str(variant.get('CLNALLELEID', '.')),
                    'clndn': str(variant.get('CLNDN', '.')),
                    'clndisdb': str(variant.get('CLNDISDB', '.')),
                    'clnrevstat': str(variant.get('CLNREVSTAT', '.')),
                    'clnsig': str(variant.get('CLNSIG', '.')),
                    'intervar_automated': str(variant.get('InterVar_automated', '.')),
                }
                acmg_criteria = variant.get('ACMG_criteria', {})
                for criterion in ACMG_FIELDS:
                    variant_data[criterion.lower()] = str(acmg_criteria.get(criterion, '.'))
                variants[pos_key][sample_id] = variant_data
        except Exception as e:
            print(f"Error processing {pattern_used}: {e}")
            error_count += 1

    print(f"\nProcessing complete: {processed_count} files processed, "
          f"{error_count} errors, {len(samples)} unique samples, "
          f"{len(variants)} unique variants")

    return variants, samples


def variants_to_dataframe(variants, samples):
    """
    Convert variants dictionary to a tidy DataFrame with one row per variant-sample pair.
    Converts numeric columns and preserves text annotation columns as strings.

    Parameters:
        variants: dict, nested dict from process_vcfs() keyed by
            chrom_pos -> sample_id -> variant attributes
        samples: set, sample IDs represented in the variants dict

    Returns:
        DataFrame with columns: CHROM, POS, SAMPLE, plus all variant attribute columns
    """
    rows = []
    for pos_key, samples_dict in variants.items():
        chrom, pos = pos_key.split('_')
        for sample_name, variant_data in samples_dict.items():
            row = {'CHROM': chrom, 'POS': pos, 'SAMPLE': sample_name}
            row.update(variant_data)
            rows.append(row)
    df = pd.DataFrame(rows)
    for col in NUMERIC_VARIANT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in TEXT_VARIANT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

def filter_comb_to_tidy(
    input_df,
    save_filename=None,
    min_vaf=None,
    max_gnomad_freq=None,
    min_cadd=None,
    gene_list=None,
    exclude_synonymous=True,
    exclude_unknown=True,
    exclude_intronic=True,
    exclude_doublets=True,
    min_strand_support=1,
    verbose=True,
    duplicate_columns=None
):
    """
    Filter combined variant DataFrame to a tidy analysis-ready format.

    The first step removes duplicate rows, keeping the first row of each set
    of rows that share the values in duplicate_columns.

    Parameters:
        input_df: DataFrame, combined variant data from variants_to_dataframe
        save_filename: str or None, path to save filtered CSV
        min_vaf: float or None, minimum VAF threshold
        max_gnomad_freq: float or None, maximum gnomAD population frequency
        min_cadd: float or None, minimum CADD phred score
        gene_list: list or None, restrict to these genes
        exclude_synonymous: bool, remove synonymous SNVs (default: True)
        exclude_unknown: bool, remove unknown functional variants (default: True)
        exclude_intronic: bool, remove intronic variants where func=='.' (default: True)
        exclude_doublets: bool, remove variants where REF and ALT are both length 2 (default: True)
        min_strand_support: int, minimum alt reads per strand from SB field (default: 1)
        verbose: bool, print filtering progress (default: True)
        duplicate_columns: list or None, columns that identify duplicate rows. With None
            (default), rows are deduplicated on sample_variant_id when that column exists,
            and no duplicate filter is applied otherwise.

    Returns:
        DataFrame with filtered variants
    """
    filtered_df = input_df.copy()
    initial_size = len(filtered_df)

    if verbose:
        print(f"Starting variant filtering with {initial_size} variants")

    # Remove duplicate rows, keeping the first row of each duplicate set
    if duplicate_columns is None and 'sample_variant_id' in filtered_df.columns:
        duplicate_columns = ['sample_variant_id']
    if duplicate_columns:
        before = len(filtered_df)
        filtered_df = filtered_df.drop_duplicates(subset=list(duplicate_columns), keep='first')
        if verbose:
            print(f"Duplicate filter: removed {before - len(filtered_df)}, {len(filtered_df)} remaining")

    # Coerce numeric columns
    for col in ['vaf', 'gnomad_freq', 'cadd_phred']:
        if col in filtered_df.columns and filtered_df[col].dtype == 'object':
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')

    # VAF filter
    if min_vaf is not None and 'vaf' in filtered_df.columns:
        before = len(filtered_df)
        filtered_df = filtered_df[filtered_df['vaf'] >= min_vaf]
        if verbose:
            print(f"VAF filter (>= {min_vaf}): removed {before - len(filtered_df)}, {len(filtered_df)} remaining")

    # gnomAD frequency filter (keeps NaN values as potentially novel)
    if max_gnomad_freq is not None and 'gnomad_freq' in filtered_df.columns:
        before = len(filtered_df)
        filtered_df = filtered_df[
            (filtered_df['gnomad_freq'].isna()) | (filtered_df['gnomad_freq'] <= max_gnomad_freq)
        ]
        if verbose:
            print(f"gnomAD filter (<= {max_gnomad_freq}): removed {before - len(filtered_df)}, {len(filtered_df)} remaining")

    # CADD filter (only applied to variants that have a score)
    if min_cadd is not None and 'cadd_phred' in filtered_df.columns:
        before = len(filtered_df)
        has_cadd = ~filtered_df['cadd_phred'].isna()
        filtered_df = filtered_df[~has_cadd | (has_cadd & (filtered_df['cadd_phred'] >= min_cadd))]
        if verbose:
            print(f"CADD filter (>= {min_cadd}): removed {before - len(filtered_df)}, {len(filtered_df)} remaining")

    # Gene list filter
    if gene_list is not None:
        gene_col = 'gene' if 'gene' in filtered_df.columns else 'GENE'
        if gene_col in filtered_df.columns:
            before = len(filtered_df)
            filtered_df = filtered_df[filtered_df[gene_col].isin(gene_list)]
            if verbose:
                print(f"Gene list filter: removed {before - len(filtered_df)}, {len(filtered_df)} remaining")

    # Functional annotation filters
    func_col = 'func' if 'func' in filtered_df.columns else ('FUNC' if 'FUNC' in filtered_df.columns else None)
    if func_col:
        if exclude_synonymous:
            before = len(filtered_df)
            filtered_df = filtered_df[~filtered_df[func_col].isin(['synonymous SNV', 'synonymous_SNV'])]
            if verbose:
                print(f"Synonymous filter: removed {before - len(filtered_df)}, {len(filtered_df)} remaining")

        if exclude_unknown or exclude_intronic:
            before = len(filtered_df)
            exclude_values = ['.']
            if exclude_unknown:
                exclude_values.append('unknown')
            filtered_df = filtered_df[~filtered_df[func_col].isin(exclude_values)]
            if verbose:
                print(f"Unknown/intronic filter: removed {before - len(filtered_df)}, {len(filtered_df)} remaining")

    # Doublet variant filter
    if exclude_doublets and 'REF' in filtered_df.columns and 'ALT' in filtered_df.columns:
        before = len(filtered_df)
        doublet_mask = (filtered_df['REF'].str.len() == 2) & (filtered_df['ALT'].str.len() == 2)
        filtered_df = filtered_df[~doublet_mask]
        if verbose:
            print(f"Doublet filter: removed {before - len(filtered_df)}, {len(filtered_df)} remaining")

    # Strand bias filter
    sb_col = 'SB' if 'SB' in filtered_df.columns else ('sb' if 'sb' in filtered_df.columns else None)
    if min_strand_support > 0 and sb_col:
        before = len(filtered_df)
        filtered_df = filtered_df.dropna(subset=[sb_col])
        split_sb = filtered_df[sb_col].astype(str).str.strip('[]').str.split(',', expand=True)
        split_sb = split_sb.apply(pd.to_numeric, errors='coerce')
        if split_sb.shape[1] == 4:
            valid_mask = (split_sb[2] >= min_strand_support) & (split_sb[3] >= min_strand_support)
            filtered_df = filtered_df[valid_mask]
        if verbose:
            print(f"Strand bias filter (min {min_strand_support}/strand): removed {before - len(filtered_df)}, {len(filtered_df)} remaining")

    if save_filename is not None:
        filtered_df.to_csv(save_filename, index=False)
        if verbose:
            print(f"Saved filtered variants to: {save_filename}")

    if verbose:
        removed = initial_size - len(filtered_df)
        pct = (removed / initial_size * 100) if initial_size > 0 else 0
        print(f"Filtering complete: {initial_size} -> {len(filtered_df)} ({removed} removed, {pct:.1f}%)")

    return filtered_df


def add_pli_to_dataframe(tidy_df, constraint_df, gene_column='GENE'):
    """
    Add pLI constraint scores to a variant DataFrame by gene name lookup.

    Parameters:
        tidy_df: DataFrame, variant data with a gene column
        constraint_df: DataFrame, gnomAD constraint metrics with 'gene' and 'lof.pLI' columns
        gene_column: str, column in tidy_df containing gene names (default: 'GENE')

    Returns:
        DataFrame with 'pLI' column added
    """
    df = tidy_df.copy()

    if gene_column not in df.columns:
        available = [c for c in df.columns if 'gene' in c.lower()]
        raise ValueError(f"Column '{gene_column}' not found. Gene-related columns: {available}")

    # Build gene-to-pLI lookup from constraint table
    constraint_clean = constraint_df.copy()
    constraint_clean['gene'] = constraint_clean['gene'].astype(str)
    constraint_clean = constraint_clean[
        constraint_clean['gene'].notna() &
        (constraint_clean['gene'] != 'nan') &
        (constraint_clean['gene'] != '')
    ]
    pli_dict = {}
    for gene, pli in zip(constraint_clean['gene'], constraint_clean['lof.pLI']):
        pli_dict[str(gene)] = str(pli) if pd.notna(pli) else '.'

    # Map pLI values
    df[gene_column] = df[gene_column].astype(str)
    df.loc[df[gene_column].isin(['nan', '']), gene_column] = '.'
    df['pLI'] = df[gene_column].map(pli_dict).fillna('.')

    variants_with_pli = len(df[df['pLI'] != '.'])
    print(f"pLI mapping: {variants_with_pli}/{len(df)} variants annotated")
    return df
