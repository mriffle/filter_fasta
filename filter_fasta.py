#!/usr/bin/env python3
"""
Filter a FASTA file based on Percolator XML results.

Applies q-value cutoffs at the PSM and peptide levels, then outputs only
FASTA entries whose protein sequence contains at least one peptide that
survived both filters.
"""

import argparse
from collections import deque
import re
import sys
import xml.etree.ElementTree as ET


def get_namespace(element):
    """Extract namespace URI from an XML element's tag."""
    match = re.match(r"\{(.+?)\}", element.tag)
    return match.group(1) if match else ""


def parse_percolator_xml(filepath, psm_qvalue_cutoff, peptide_qvalue_cutoff):
    """
    Parse a Percolator XML file and return the set of peptide sequences
    that pass both PSM-level and peptide-level q-value cutoffs.

    Uses iterparse to stream through the XML, keeping memory usage
    constant regardless of file size.
    """
    # First pass: detect namespace from the root element
    uri = ""
    for event, elem in ET.iterparse(filepath, events=("start",)):
        uri = get_namespace(elem)
        break

    ns_prefix = f"{{{uri}}}" if uri else ""

    def local_tag(tag):
        return f"{ns_prefix}{tag}"

    psm_tag = local_tag("psm")
    peptide_tag = local_tag("peptide")
    q_value_tag = local_tag("q_value")
    peptide_seq_tag = local_tag("peptide_seq")

    psm_passing_peptides = set()
    peptide_passing = set()
    psm_total = 0
    peptide_total = 0

    for event, elem in ET.iterparse(filepath, events=("end",)):
        if elem.tag == psm_tag:
            psm_total += 1
            q_el = elem.find(q_value_tag)
            qval = float(q_el.text) if q_el is not None and q_el.text else None
            if qval is not None and qval <= psm_qvalue_cutoff:
                seq_el = elem.find(peptide_seq_tag)
                seq = seq_el.text if seq_el is not None else None
                if seq:
                    # Strip flanking residues if present (e.g., K.PEPTIDE.R -> PEPTIDE)
                    seq = strip_flanking(seq)
                    psm_passing_peptides.add(seq)
            elem.clear()

        elif elem.tag == peptide_tag:
            peptide_total += 1
            q_el = elem.find(q_value_tag)
            qval = float(q_el.text) if q_el is not None and q_el.text else None
            if qval is not None and qval <= peptide_qvalue_cutoff:
                pep_id = elem.get("peptide_id") or ""
                seq_el = elem.find(peptide_seq_tag)
                seq = (seq_el.text if seq_el is not None else None) or pep_id
                if seq:
                    seq = strip_flanking(seq)
                    peptide_passing.add(seq)
            elem.clear()

    # Peptides must pass BOTH filters
    passing = psm_passing_peptides & peptide_passing

    print(f"Percolator summary:", file=sys.stderr)
    print(f"  PSMs total:                {psm_total}", file=sys.stderr)
    print(f"  PSMs passing (q<={psm_qvalue_cutoff}):   {len(psm_passing_peptides)} unique peptides", file=sys.stderr)
    print(f"  Peptides total:            {peptide_total}", file=sys.stderr)
    print(f"  Peptides passing (q<={peptide_qvalue_cutoff}): {len(peptide_passing)}", file=sys.stderr)
    print(f"  Peptides passing both:     {len(passing)}", file=sys.stderr)

    return passing


def strip_flanking(seq):
    """
    Remove flanking residues and modifications from a peptide sequence.
    E.g., 'K.PEPTIDER.S' -> 'PEPTIDER'
    Also strips common modification notation like brackets.
    """
    # Remove modification annotations like [+57.02] or (ox) first,
    # so that dots inside modifications don't break flanking detection.
    seq = re.sub(r"[\[\(][^)\]]*[\]\)]", "", seq)
    # Remove flanking (e.g., X.SEQ.X)
    parts = seq.split(".")
    if len(parts) == 3 and len(parts[0]) <= 1 and len(parts[2]) <= 1:
        seq = parts[1]
    # Keep only uppercase letters (amino acids)
    seq = re.sub(r"[^A-Z]", "", seq.upper())
    return seq


def parse_fasta(filepath):
    """
    Yield (header, sequence) tuples from a FASTA file.
    """
    header = None
    seq_parts = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line
                seq_parts = []
            else:
                seq_parts.append(line.strip())
    if header is not None:
        yield header, "".join(seq_parts)


class AhoCorasick:
    """Pure Python Aho-Corasick automaton for multi-pattern substring search."""

    def __init__(self, patterns):
        self.goto = [{}]     # goto function: list of dicts (state -> char -> state)
        self.fail = [0]      # failure function
        self.output = [False] # whether a state is an accepting state
        self._build(patterns)

    def _build(self, patterns):
        # Build the trie (goto function)
        for pattern in patterns:
            state = 0
            for ch in pattern:
                if ch not in self.goto[state]:
                    new_state = len(self.goto)
                    self.goto.append({})
                    self.fail.append(0)
                    self.output.append(False)
                    self.goto[state][ch] = new_state
                state = self.goto[state][ch]
            self.output[state] = True

        # Build failure function via BFS
        queue = deque()
        for ch, s in self.goto[0].items():
            self.fail[s] = 0
            queue.append(s)

        while queue:
            r = queue.popleft()
            for ch, s in self.goto[r].items():
                queue.append(s)
                state = self.fail[r]
                while state != 0 and ch not in self.goto[state]:
                    state = self.fail[state]
                self.fail[s] = self.goto[state].get(ch, 0)
                if self.fail[s] == s:
                    self.fail[s] = 0
                if self.output[self.fail[s]]:
                    self.output[s] = True

    def contains_any(self, text):
        """Return True if text contains any of the patterns."""
        state = 0
        for ch in text:
            while state != 0 and ch not in self.goto[state]:
                state = self.fail[state]
            state = self.goto[state].get(ch, 0)
            if self.output[state]:
                return True
        return False


def filter_fasta(fasta_path, peptides, output_path):
    """
    Write FASTA entries to output if their sequence contains at least one
    of the given peptides.
    """
    automaton = AhoCorasick(peptides)
    total = 0
    kept = 0

    out = open(output_path, "w") if output_path != "-" else sys.stdout
    try:
        for header, sequence in parse_fasta(fasta_path):
            total += 1
            seq_upper = sequence.upper()
            if automaton.contains_any(seq_upper):
                kept += 1
                out.write(f"{header}\n")
                # Write sequence in 60-character lines
                for i in range(0, len(sequence), 60):
                    out.write(f"{sequence[i:i+60]}\n")
    finally:
        if out is not sys.stdout:
            out.close()

    print(f"\nFASTA filtering summary:", file=sys.stderr)
    print(f"  Total entries:   {total}", file=sys.stderr)
    print(f"  Entries kept:    {kept}", file=sys.stderr)
    print(f"  Entries removed: {total - kept}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Filter a FASTA file to keep only entries containing "
                    "peptides that pass Percolator q-value cutoffs."
    )
    parser.add_argument(
        "-f", "--fasta", required=True,
        help="Input FASTA file"
    )
    parser.add_argument(
        "-x", "--xml", required=True,
        help="Percolator results XML file"
    )
    parser.add_argument(
        "-o", "--output", default="-",
        help="Output FASTA file (default: stdout)"
    )
    parser.add_argument(
        "--psm-qvalue", type=float, default=0.01,
        help="PSM-level q-value cutoff (default: 0.01)"
    )
    parser.add_argument(
        "--peptide-qvalue", type=float, default=0.01,
        help="Peptide-level q-value cutoff (default: 0.01)"
    )
    args = parser.parse_args()

    # Parse Percolator results and apply q-value filters
    passing_peptides = parse_percolator_xml(args.xml, args.psm_qvalue, args.peptide_qvalue)

    if not passing_peptides:
        print("WARNING: No peptides passed both q-value cutoffs. "
              "Output will be empty.", file=sys.stderr)

    # Filter FASTA
    filter_fasta(args.fasta, passing_peptides, args.output)


if __name__ == "__main__":
    main()