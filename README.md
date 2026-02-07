# filter_fasta

Filter a FASTA protein database to retain only entries with peptide-level evidence from a mass spectrometry proteomics search, as scored by [Percolator](http://percolator.ms/).

## How It Works

1. **Parses one or more Percolator XML results files** and applies two independent q-value filters per file:
   - **PSM-level** — collects peptide sequences from PSMs with `q_value <= --psm-qvalue`.
   - **Peptide-level** — collects peptide sequences from peptides with `q_value <= --peptide-qvalue`.
   - Only peptides passing **both** filters are retained (set intersection).
   - When multiple XML files are provided, passing peptides from all files are **unioned** together.

2. **Filters the input FASTA** — for each protein entry, checks whether its amino acid sequence contains any of the passing peptide sequences as a substring. Matching entries are written to the output; all others are discarded.

Peptide sequences are cleaned before matching: flanking residues (e.g., `K.PEPTIDE.R`) and modification annotations (e.g., `[+57.02]`, `(ox)`) are stripped.

## Requirements

- Python 3.6+
- No external dependencies (standard library only)

## Usage

```bash
python3 filter_fasta.py -f <input.fasta> -x <percolator1.xml> [percolator2.xml ...] [-o <output.fasta>] [--psm-qvalue <cutoff>] [--peptide-qvalue <cutoff>]
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `-f`, `--fasta` | Yes | — | Input FASTA file |
| `-x`, `--xml` | Yes | — | One or more Percolator results XML files |
| `-o`, `--output` | No | stdout | Output FASTA file |
| `--psm-qvalue` | No | `0.01` | PSM-level q-value cutoff |
| `--peptide-qvalue` | No | `0.01` | Peptide-level q-value cutoff |

### Examples

Write filtered FASTA to a file with default q-value cutoffs (0.01):

```bash
python3 filter_fasta.py -f proteins.fasta -x percolator_results.xml -o filtered.fasta
```

Multiple Percolator XML files:

```bash
python3 filter_fasta.py -f proteins.fasta -x results1.xml results2.xml results3.xml -o filtered.fasta
```

Use stricter cutoffs:

```bash
python3 filter_fasta.py -f proteins.fasta -x percolator_results.xml -o filtered.fasta --psm-qvalue 0.005 --peptide-qvalue 0.005
```

Pipe output to another tool:

```bash
python3 filter_fasta.py -f proteins.fasta -x percolator_results.xml | some_other_tool
```

Summary statistics are printed to **stderr** so they don't interfere with piped FASTA output.

## Running Tests

```bash
python3 -m pytest test_filter_fasta.py -v
```
