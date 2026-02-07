#!/usr/bin/env python3
"""
Comprehensive tests for filter_fasta.py

Tests cover:
- strip_flanking: flanking residues, modifications, edge cases
- AhoCorasick: single/multiple patterns, overlapping, no match, empty
- parse_fasta: standard, multi-line sequences, single entry, empty file
- parse_percolator_xml: q-value filtering, namespace handling, both filters
- filter_fasta: end-to-end integration with expected output comparison
"""

import os
import tempfile
import textwrap
import unittest

from filter_fasta import (
    AhoCorasick,
    filter_fasta,
    parse_fasta,
    parse_percolator_xml,
    strip_flanking,
)


# ---------------------------------------------------------------------------
# Helper: write a string to a temp file and return its path
# ---------------------------------------------------------------------------
def _tmpfile(content, suffix=""):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

FASTA_BASIC = textwrap.dedent("""\
    >sp|P00001|PROT1 Protein one
    MAAAGKLVTIDERPEPTIDEKWWWWWWWWWW
    GGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
    >sp|P00002|PROT2 Protein two
    MCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    TARGETSEQHEREXXXXXXXXXXXXXXXXXX
    >sp|P00003|PROT3 Protein three
    MDDDDDDDDDDDDDDDDDDDDDDDDDDDD
    DDDDDDDDDDDDDDDDDDDDDDDDDDDDD
    >sp|P00004|PROT4 Protein four
    MEEEEEEEEEEEEEEEEEEEEEEEEEEEABC
    DEFGHIJKLMNOPQRSTUVWXYZANOTHER
""")

# Percolator XML without namespace
PERC_XML_NO_NS = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <percolator_output>
      <psms>
        <psm psm_id="psm1">
          <q_value>0.001</q_value>
          <peptide_seq>PEPTIDE</peptide_seq>
        </psm>
        <psm psm_id="psm2">
          <q_value>0.005</q_value>
          <peptide_seq>TARGETSEQ</peptide_seq>
        </psm>
        <psm psm_id="psm3">
          <q_value>0.05</q_value>
          <peptide_seq>ANOTHER</peptide_seq>
        </psm>
        <psm psm_id="psm4">
          <q_value>0.001</q_value>
          <peptide_seq>NOMATCH</peptide_seq>
        </psm>
      </psms>
      <peptides>
        <peptide peptide_id="pep1">
          <q_value>0.002</q_value>
          <peptide_seq>PEPTIDE</peptide_seq>
        </peptide>
        <peptide peptide_id="pep2">
          <q_value>0.008</q_value>
          <peptide_seq>TARGETSEQ</peptide_seq>
        </peptide>
        <peptide peptide_id="pep3">
          <q_value>0.003</q_value>
          <peptide_seq>ANOTHER</peptide_seq>
        </peptide>
        <peptide peptide_id="pep4">
          <q_value>0.001</q_value>
          <peptide_seq>NOMATCH</peptide_seq>
        </peptide>
      </peptides>
    </percolator_output>
""")

# Percolator XML with namespace
PERC_XML_WITH_NS = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <percolator_output xmlns="http://per-colator.com/percolator_out/15">
      <psms>
        <psm psm_id="psm1">
          <q_value>0.001</q_value>
          <peptide_seq>PEPTIDE</peptide_seq>
        </psm>
        <psm psm_id="psm2">
          <q_value>0.005</q_value>
          <peptide_seq>TARGETSEQ</peptide_seq>
        </psm>
      </psms>
      <peptides>
        <peptide peptide_id="pep1">
          <q_value>0.002</q_value>
          <peptide_seq>PEPTIDE</peptide_seq>
        </peptide>
        <peptide peptide_id="pep2">
          <q_value>0.008</q_value>
          <peptide_seq>TARGETSEQ</peptide_seq>
        </peptide>
      </peptides>
    </percolator_output>
""")

# Second Percolator XML with different peptides (for multi-XML tests)
# ANOTHER passes both filters here; EXTRASEQ passes both
PERC_XML_SECOND = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <percolator_output>
      <psms>
        <psm psm_id="psm1">
          <q_value>0.001</q_value>
          <peptide_seq>ANOTHER</peptide_seq>
        </psm>
        <psm psm_id="psm2">
          <q_value>0.001</q_value>
          <peptide_seq>EXTRASEQ</peptide_seq>
        </psm>
      </psms>
      <peptides>
        <peptide peptide_id="pep1">
          <q_value>0.001</q_value>
          <peptide_seq>ANOTHER</peptide_seq>
        </peptide>
        <peptide peptide_id="pep2">
          <q_value>0.001</q_value>
          <peptide_seq>EXTRASEQ</peptide_seq>
        </peptide>
      </peptides>
    </percolator_output>
""")

# Real-world Percolator XML: seq/n/c attributes on peptide_seq, namespaced psm_id/peptide_id
PERC_XML_REAL_WORLD = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <percolator_output
    xmlns="http://per-colator.com/percolator_out/15"
    xmlns:p="http://per-colator.com/percolator_out/15">
      <psms>
        <psm p:psm_id="psm_001">
          <q_value>2.600277e-02</q_value>
          <peptide_seq n="R" c="G" seq="GSGGGSSGGSIGGR"/>
        </psm>
        <psm p:psm_id="psm_002">
          <q_value>0.001</q_value>
          <peptide_seq n="K" c="R" seq="PEPTIDE"/>
        </psm>
        <psm p:psm_id="psm_003">
          <q_value>0.001</q_value>
          <peptide_seq seq="NOFLANKING"/>
        </psm>
      </psms>
      <peptides>
        <peptide p:peptide_id="pep_001">
          <q_value>0.01</q_value>
          <peptide_seq n="R" c="G" seq="GSGGGSSGGSIGGR"/>
        </peptide>
        <peptide p:peptide_id="pep_002">
          <q_value>0.001</q_value>
          <peptide_seq n="K" c="R" seq="PEPTIDE"/>
        </peptide>
        <peptide p:peptide_id="pep_003">
          <q_value>0.001</q_value>
          <peptide_seq seq="NOFLANKING"/>
        </peptide>
      </peptides>
    </percolator_output>
""")

# Percolator XML with flanking residues and modifications
PERC_XML_FLANKING = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <percolator_output>
      <psms>
        <psm psm_id="psm1">
          <q_value>0.001</q_value>
          <peptide_seq>K.PEPTIDE.R</peptide_seq>
        </psm>
        <psm psm_id="psm2">
          <q_value>0.001</q_value>
          <peptide_seq>R.TARGETSEQ[+57.02].K</peptide_seq>
        </psm>
      </psms>
      <peptides>
        <peptide peptide_id="pep1">
          <q_value>0.001</q_value>
          <peptide_seq>K.PEPTIDE.R</peptide_seq>
        </peptide>
        <peptide peptide_id="pep2">
          <q_value>0.001</q_value>
          <peptide_seq>R.TARGETSEQ[+57.02].K</peptide_seq>
        </peptide>
      </peptides>
    </percolator_output>
""")


# ===================================================================
# Tests for strip_flanking
# ===================================================================
class TestStripFlanking(unittest.TestCase):

    def test_plain_sequence(self):
        self.assertEqual(strip_flanking("PEPTIDE"), "PEPTIDE")

    def test_flanking_residues(self):
        self.assertEqual(strip_flanking("K.PEPTIDER.S"), "PEPTIDER")

    def test_flanking_with_nterm_dash(self):
        self.assertEqual(strip_flanking("-.PEPTIDE.K"), "PEPTIDE")

    def test_flanking_with_cterm_dash(self):
        self.assertEqual(strip_flanking("K.PEPTIDE.-"), "PEPTIDE")

    def test_modification_brackets(self):
        self.assertEqual(strip_flanking("PEPTC[+57.02]IDE"), "PEPTCIDE")

    def test_modification_parens(self):
        self.assertEqual(strip_flanking("M(ox)PEPTIDE"), "MPEPTIDE")

    def test_flanking_plus_modification(self):
        self.assertEqual(strip_flanking("K.PEP[+80]TIDE.R"), "PEPTIDE")

    def test_lowercase_normalized(self):
        self.assertEqual(strip_flanking("peptide"), "PEPTIDE")

    def test_empty_string(self):
        self.assertEqual(strip_flanking(""), "")

    def test_single_residue(self):
        self.assertEqual(strip_flanking("K"), "K")

    def test_multiple_modifications(self):
        self.assertEqual(strip_flanking("K.C[+57.02]M(ox)PEPTIDE.R"), "CMPEPTIDE")


# ===================================================================
# Tests for AhoCorasick
# ===================================================================
class TestAhoCorasick(unittest.TestCase):

    def test_single_pattern_found(self):
        ac = AhoCorasick({"HELLO"})
        self.assertTrue(ac.contains_any("SAYHELLOWORLD"))

    def test_single_pattern_not_found(self):
        ac = AhoCorasick({"HELLO"})
        self.assertFalse(ac.contains_any("SAYGOODBYEWORLD"))

    def test_multiple_patterns(self):
        ac = AhoCorasick({"ABC", "DEF", "GHI"})
        self.assertTrue(ac.contains_any("XYZDEFXYZ"))
        self.assertTrue(ac.contains_any("ABCXYZ"))
        self.assertTrue(ac.contains_any("XYZGHI"))
        self.assertFalse(ac.contains_any("XYZXYZ"))

    def test_pattern_at_start(self):
        ac = AhoCorasick({"ABC"})
        self.assertTrue(ac.contains_any("ABCDEF"))

    def test_pattern_at_end(self):
        ac = AhoCorasick({"DEF"})
        self.assertTrue(ac.contains_any("ABCDEF"))

    def test_exact_match(self):
        ac = AhoCorasick({"ABC"})
        self.assertTrue(ac.contains_any("ABC"))

    def test_empty_text(self):
        ac = AhoCorasick({"ABC"})
        self.assertFalse(ac.contains_any(""))

    def test_empty_patterns(self):
        ac = AhoCorasick(set())
        self.assertFalse(ac.contains_any("ANYTHING"))

    def test_overlapping_patterns(self):
        ac = AhoCorasick({"AB", "BC", "ABC"})
        self.assertTrue(ac.contains_any("ABC"))
        self.assertTrue(ac.contains_any("XBC"))
        self.assertTrue(ac.contains_any("ABX"))

    def test_pattern_is_substring_of_another(self):
        ac = AhoCorasick({"PEPTIDE", "PEP"})
        self.assertTrue(ac.contains_any("XPEPX"))
        self.assertTrue(ac.contains_any("XPEPTIDEX"))

    def test_repeated_characters(self):
        ac = AhoCorasick({"AAA"})
        self.assertTrue(ac.contains_any("BAAAB"))
        self.assertFalse(ac.contains_any("BAAB"))

    def test_consistency_with_python_in(self):
        """AhoCorasick must agree with brute-force 'in' check."""
        patterns = {"PEPTIDE", "TARGETSEQ", "ANOTHER", "NOMATCH"}
        ac = AhoCorasick(patterns)
        texts = [
            "MAAAGKLVTIDERPEPTIDEKWWWWWWWWWW",
            "TARGETSEQHEREXXXXXXXXXXXXXXXXXX",
            "MDDDDDDDDDDDDDDDDDDDDDDDDDDDD",
            "XYZANOTHERTHING",
            "NOTHINGHERE",
            "",
        ]
        for text in texts:
            brute = any(p in text for p in patterns)
            self.assertEqual(ac.contains_any(text), brute, f"Mismatch on: {text!r}")


# ===================================================================
# Tests for parse_fasta
# ===================================================================
class TestParseFasta(unittest.TestCase):

    def test_basic_multi_entry(self):
        path = _tmpfile(FASTA_BASIC, suffix=".fasta")
        try:
            entries = list(parse_fasta(path))
            self.assertEqual(len(entries), 4)
            self.assertEqual(entries[0][0], ">sp|P00001|PROT1 Protein one")
            self.assertIn("PEPTIDE", entries[0][1])
            self.assertEqual(entries[2][0], ">sp|P00003|PROT3 Protein three")
        finally:
            os.unlink(path)

    def test_single_entry(self):
        content = ">single\nACDEFG\n"
        path = _tmpfile(content, suffix=".fasta")
        try:
            entries = list(parse_fasta(path))
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0], (">single", "ACDEFG"))
        finally:
            os.unlink(path)

    def test_multiline_sequence(self):
        content = ">multi\nABC\nDEF\nGHI\n"
        path = _tmpfile(content, suffix=".fasta")
        try:
            entries = list(parse_fasta(path))
            self.assertEqual(entries[0][1], "ABCDEFGHI")
        finally:
            os.unlink(path)

    def test_empty_file(self):
        path = _tmpfile("", suffix=".fasta")
        try:
            entries = list(parse_fasta(path))
            self.assertEqual(entries, [])
        finally:
            os.unlink(path)

    def test_no_trailing_newline(self):
        content = ">entry\nSEQUENCE"
        path = _tmpfile(content, suffix=".fasta")
        try:
            entries = list(parse_fasta(path))
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0][1], "SEQUENCE")
        finally:
            os.unlink(path)


# ===================================================================
# Tests for parse_percolator_xml
# ===================================================================
class TestParsePercolatorXml(unittest.TestCase):

    def test_both_pass_default_cutoff(self):
        """PEPTIDE and TARGETSEQ pass both PSM and peptide q-value at 0.01."""
        path = _tmpfile(PERC_XML_NO_NS, suffix=".xml")
        try:
            result = parse_percolator_xml(path, 0.01, 0.01)
            # PEPTIDE: psm q=0.001, pep q=0.002 -> pass both
            # TARGETSEQ: psm q=0.005, pep q=0.008 -> pass both
            # ANOTHER: psm q=0.05 (fail PSM), pep q=0.003 -> fail
            # NOMATCH: psm q=0.001, pep q=0.001 -> pass both
            self.assertEqual(result, {"PEPTIDE", "TARGETSEQ", "NOMATCH"})
        finally:
            os.unlink(path)

    def test_strict_psm_cutoff(self):
        """With psm cutoff 0.002, only psm q<=0.002 pass."""
        path = _tmpfile(PERC_XML_NO_NS, suffix=".xml")
        try:
            result = parse_percolator_xml(path, 0.002, 0.01)
            # PSM passing: PEPTIDE (0.001), NOMATCH (0.001)
            # Peptide passing: PEPTIDE (0.002), TARGETSEQ (0.008), ANOTHER (0.003), NOMATCH (0.001)
            # Intersection: PEPTIDE, NOMATCH
            self.assertEqual(result, {"PEPTIDE", "NOMATCH"})
        finally:
            os.unlink(path)

    def test_strict_peptide_cutoff(self):
        """With peptide cutoff 0.002, only peptide q<=0.002 pass."""
        path = _tmpfile(PERC_XML_NO_NS, suffix=".xml")
        try:
            result = parse_percolator_xml(path, 0.01, 0.002)
            # PSM passing: PEPTIDE (0.001), TARGETSEQ (0.005), NOMATCH (0.001)
            # Peptide passing: PEPTIDE (0.002), NOMATCH (0.001)
            # Intersection: PEPTIDE, NOMATCH
            self.assertEqual(result, {"PEPTIDE", "NOMATCH"})
        finally:
            os.unlink(path)

    def test_nothing_passes(self):
        """With extremely strict cutoff, nothing passes."""
        path = _tmpfile(PERC_XML_NO_NS, suffix=".xml")
        try:
            result = parse_percolator_xml(path, 0.0001, 0.0001)
            self.assertEqual(result, set())
        finally:
            os.unlink(path)

    def test_namespace_handling(self):
        """XML with namespace should parse identically."""
        path = _tmpfile(PERC_XML_WITH_NS, suffix=".xml")
        try:
            result = parse_percolator_xml(path, 0.01, 0.01)
            self.assertEqual(result, {"PEPTIDE", "TARGETSEQ"})
        finally:
            os.unlink(path)

    def test_real_world_seq_attributes(self):
        """Real-world format: peptide_seq uses seq/n/c attributes, not text."""
        path = _tmpfile(PERC_XML_REAL_WORLD, suffix=".xml")
        try:
            result = parse_percolator_xml(path, 0.05, 0.05)
            # All three PSMs pass at 0.05; all three peptides pass at 0.05
            # Flanking n/c should be stripped: R.GSGGGSSGGSIGGR.G -> GSGGGSSGGSIGGR
            # K.PEPTIDE.R -> PEPTIDE, NOFLANKING -> NOFLANKING
            self.assertIn("GSGGGSSGGSIGGR", result)
            self.assertIn("PEPTIDE", result)
            self.assertIn("NOFLANKING", result)
            self.assertEqual(len(result), 3)
        finally:
            os.unlink(path)

    def test_real_world_strict_cutoff(self):
        """Real-world format with strict cutoff filters out high q-value PSMs."""
        path = _tmpfile(PERC_XML_REAL_WORLD, suffix=".xml")
        try:
            result = parse_percolator_xml(path, 0.01, 0.01)
            # PSM psm_001 has q=0.026 -> fails at 0.01
            # PSM psm_002 has q=0.001 -> passes
            # PSM psm_003 has q=0.001 -> passes
            # Peptide pep_001 has q=0.01 -> passes (<=)
            # Peptide pep_002 has q=0.001 -> passes
            # Peptide pep_003 has q=0.001 -> passes
            # Intersection: PEPTIDE, NOFLANKING (GSGGGSSGGSIGGR fails PSM)
            self.assertIn("PEPTIDE", result)
            self.assertIn("NOFLANKING", result)
            self.assertNotIn("GSGGGSSGGSIGGR", result)
            self.assertEqual(len(result), 2)
        finally:
            os.unlink(path)

    def test_peptide_id_fallback_with_modifications(self):
        """Peptide entries with no peptide_seq element use peptide_id as sequence."""
        xml = textwrap.dedent("""\
            <?xml version="1.0" encoding="UTF-8"?>
            <percolator_output
            xmlns="http://per-colator.com/percolator_out/15"
            xmlns:p="http://per-colator.com/percolator_out/15">
              <psms>
                <psm p:psm_id="psm1">
                  <q_value>0.001</q_value>
                  <peptide_seq n="R" c="G" seq="SSCVGSSMPGGGMGGMGGMGGMPMM"/>
                </psm>
                <psm p:psm_id="psm2">
                  <q_value>0.001</q_value>
                  <peptide_seq n="K" c="R" seq="IVLTLIAPLLKPLFK"/>
                </psm>
              </psms>
              <peptides>
                <peptide p:peptide_id="SSCVGSSM[15.9949]PGGGMGGMGGM[15.9949]GGMPMM">
                  <q_value>0.001</q_value>
                </peptide>
                <peptide p:peptide_id="IVLTLIAPLLKPLFK">
                  <q_value>0.001</q_value>
                </peptide>
              </peptides>
            </percolator_output>
        """)
        path = _tmpfile(xml, suffix=".xml")
        try:
            result = parse_percolator_xml(path, 0.01, 0.01)
            # Modifications stripped from peptide_id: SSCVGSSMPGGGMGGMGGMGGMPMM
            self.assertIn("SSCVGSSMPGGGMGGMGGMGGMPMM", result)
            self.assertIn("IVLTLIAPLLKPLFK", result)
            self.assertEqual(len(result), 2)
        finally:
            os.unlink(path)

    def test_flanking_residues_stripped(self):
        """Flanking residues and modifications are stripped before matching."""
        path = _tmpfile(PERC_XML_FLANKING, suffix=".xml")
        try:
            result = parse_percolator_xml(path, 0.01, 0.01)
            self.assertIn("PEPTIDE", result)
            self.assertIn("TARGETSEQ", result)
            # Should NOT contain the raw annotated form
            self.assertNotIn("K.PEPTIDE.R", result)
        finally:
            os.unlink(path)


# ===================================================================
# Tests for filter_fasta (end-to-end integration)
# ===================================================================
class TestFilterFasta(unittest.TestCase):

    def _run_filter(self, fasta_content, peptides):
        """Helper: run filter_fasta and return the output string."""
        fasta_path = _tmpfile(fasta_content, suffix=".fasta")
        out_path = _tmpfile("", suffix=".fasta")
        try:
            filter_fasta(fasta_path, peptides, out_path)
            with open(out_path) as f:
                return f.read()
        finally:
            os.unlink(fasta_path)
            os.unlink(out_path)

    def _parse_output_headers(self, output):
        """Extract headers from FASTA output."""
        return [line for line in output.splitlines() if line.startswith(">")]

    def test_basic_filtering(self):
        """Only proteins containing a matching peptide are kept."""
        output = self._run_filter(FASTA_BASIC, {"PEPTIDE", "TARGETSEQ"})
        headers = self._parse_output_headers(output)
        # PROT1 contains PEPTIDE, PROT2 contains TARGETSEQ
        self.assertEqual(len(headers), 2)
        self.assertIn(">sp|P00001|PROT1 Protein one", headers)
        self.assertIn(">sp|P00002|PROT2 Protein two", headers)

    def test_no_match(self):
        """No proteins match -> empty output."""
        output = self._run_filter(FASTA_BASIC, {"ZZZZZZZ"})
        self.assertEqual(output.strip(), "")

    def test_all_match(self):
        """If every protein has a match, all are kept."""
        # Each protein has unique content; provide a peptide from each
        peptides = {"PEPTIDE", "TARGETSEQ", "DDDDDDDDDDD", "ANOTHER"}
        output = self._run_filter(FASTA_BASIC, peptides)
        headers = self._parse_output_headers(output)
        self.assertEqual(len(headers), 4)

    def test_empty_peptide_set(self):
        """Empty peptide set -> no output."""
        output = self._run_filter(FASTA_BASIC, set())
        self.assertEqual(output.strip(), "")

    def test_sequence_wrapping_at_60(self):
        """Output sequences should be wrapped at 60 characters per line."""
        fasta = ">test\n" + "A" * 150 + "\n"
        output = self._run_filter(fasta, {"AAAA"})
        lines = output.strip().splitlines()
        # Line 0: header, Lines 1-3: sequence
        self.assertEqual(lines[0], ">test")
        self.assertEqual(len(lines[1]), 60)
        self.assertEqual(len(lines[2]), 60)
        self.assertEqual(len(lines[3]), 30)

    def test_case_insensitive_matching(self):
        """Peptide matching should be case-insensitive on the protein side."""
        fasta = ">test\nabcdef\n"
        output = self._run_filter(fasta, {"ABCDEF"})
        headers = self._parse_output_headers(output)
        self.assertEqual(len(headers), 1)

    def test_exact_expected_output(self):
        """Full end-to-end: compare exact output to expected FASTA."""
        fasta_content = textwrap.dedent("""\
            >protein_A
            AAAAAAAAAAPEPTIDEBBBBBBBBBB
            >protein_B
            CCCCCCCCCCCCCCCCCCCCCCCCCCCC
            >protein_C
            DDDDDDTARGETSEQEEEEEEEEEEE
        """)
        peptides = {"PEPTIDE", "TARGETSEQ"}

        expected = textwrap.dedent("""\
            >protein_A
            AAAAAAAAAAPEPTIDEBBBBBBBBBB
            >protein_C
            DDDDDDTARGETSEQEEEEEEEEEEE
        """)

        output = self._run_filter(fasta_content, peptides)
        self.assertEqual(output, expected)


# ===================================================================
# Full pipeline integration test
# ===================================================================
class TestFullPipeline(unittest.TestCase):

    def test_end_to_end_no_namespace(self):
        """Full pipeline: Percolator XML (no ns) -> filter FASTA -> expected output."""
        fasta_path = _tmpfile(FASTA_BASIC, suffix=".fasta")
        xml_path = _tmpfile(PERC_XML_NO_NS, suffix=".xml")
        out_path = _tmpfile("", suffix=".fasta")
        try:
            peptides = parse_percolator_xml(xml_path, 0.01, 0.01)
            # Expected: PEPTIDE, TARGETSEQ, NOMATCH pass both
            self.assertEqual(peptides, {"PEPTIDE", "TARGETSEQ", "NOMATCH"})

            filter_fasta(fasta_path, peptides, out_path)
            with open(out_path) as f:
                output = f.read()

            headers = [l for l in output.splitlines() if l.startswith(">")]
            # PROT1 has PEPTIDE, PROT2 has TARGETSEQ
            # PROT3 has no match, PROT4 has no match (NOMATCH not in any protein)
            self.assertIn(">sp|P00001|PROT1 Protein one", headers)
            self.assertIn(">sp|P00002|PROT2 Protein two", headers)
            self.assertNotIn(">sp|P00003|PROT3 Protein three", headers)
            self.assertNotIn(">sp|P00004|PROT4 Protein four", headers)
            self.assertEqual(len(headers), 2)
        finally:
            os.unlink(fasta_path)
            os.unlink(xml_path)
            os.unlink(out_path)

    def test_end_to_end_with_namespace(self):
        """Full pipeline: Percolator XML (with ns) -> filter FASTA."""
        fasta_path = _tmpfile(FASTA_BASIC, suffix=".fasta")
        xml_path = _tmpfile(PERC_XML_WITH_NS, suffix=".xml")
        out_path = _tmpfile("", suffix=".fasta")
        try:
            peptides = parse_percolator_xml(xml_path, 0.01, 0.01)
            self.assertEqual(peptides, {"PEPTIDE", "TARGETSEQ"})

            filter_fasta(fasta_path, peptides, out_path)
            with open(out_path) as f:
                output = f.read()

            headers = [l for l in output.splitlines() if l.startswith(">")]
            self.assertEqual(len(headers), 2)
            self.assertIn(">sp|P00001|PROT1 Protein one", headers)
            self.assertIn(">sp|P00002|PROT2 Protein two", headers)
        finally:
            os.unlink(fasta_path)
            os.unlink(xml_path)
            os.unlink(out_path)

    def test_end_to_end_with_flanking(self):
        """Full pipeline: Percolator XML with flanking/mods -> filter FASTA."""
        fasta_path = _tmpfile(FASTA_BASIC, suffix=".fasta")
        xml_path = _tmpfile(PERC_XML_FLANKING, suffix=".xml")
        out_path = _tmpfile("", suffix=".fasta")
        try:
            peptides = parse_percolator_xml(xml_path, 0.01, 0.01)
            self.assertEqual(peptides, {"PEPTIDE", "TARGETSEQ"})

            filter_fasta(fasta_path, peptides, out_path)
            with open(out_path) as f:
                output = f.read()

            headers = [l for l in output.splitlines() if l.startswith(">")]
            self.assertEqual(len(headers), 2)
        finally:
            os.unlink(fasta_path)
            os.unlink(xml_path)
            os.unlink(out_path)

    def test_end_to_end_multiple_xml_union(self):
        """Multiple XML files: passing peptides are unioned across files."""
        fasta_path = _tmpfile(FASTA_BASIC, suffix=".fasta")
        xml_path1 = _tmpfile(PERC_XML_NO_NS, suffix=".xml")
        xml_path2 = _tmpfile(PERC_XML_SECOND, suffix=".xml")
        out_path = _tmpfile("", suffix=".fasta")
        try:
            # XML1 passes: PEPTIDE, TARGETSEQ, NOMATCH (at 0.01)
            # XML2 passes: ANOTHER, EXTRASEQ
            # Union: PEPTIDE, TARGETSEQ, NOMATCH, ANOTHER, EXTRASEQ
            passing = set()
            passing |= parse_percolator_xml(xml_path1, 0.01, 0.01)
            passing |= parse_percolator_xml(xml_path2, 0.01, 0.01)
            self.assertEqual(passing, {"PEPTIDE", "TARGETSEQ", "NOMATCH", "ANOTHER", "EXTRASEQ"})

            filter_fasta(fasta_path, passing, out_path)
            with open(out_path) as f:
                output = f.read()

            headers = [l for l in output.splitlines() if l.startswith(">")]
            # PROT1 has PEPTIDE, PROT2 has TARGETSEQ, PROT4 has ANOTHER
            # PROT3 has no match, NOMATCH/EXTRASEQ not in any protein
            self.assertEqual(len(headers), 3)
            self.assertIn(">sp|P00001|PROT1 Protein one", headers)
            self.assertIn(">sp|P00002|PROT2 Protein two", headers)
            self.assertIn(">sp|P00004|PROT4 Protein four", headers)
            self.assertNotIn(">sp|P00003|PROT3 Protein three", headers)
        finally:
            for p in (fasta_path, xml_path1, xml_path2, out_path):
                os.unlink(p)

    def test_end_to_end_multiple_xml_more_than_single(self):
        """Multiple XML files should keep >= as many proteins as any single file."""
        fasta_path = _tmpfile(FASTA_BASIC, suffix=".fasta")
        xml_path1 = _tmpfile(PERC_XML_NO_NS, suffix=".xml")
        xml_path2 = _tmpfile(PERC_XML_SECOND, suffix=".xml")
        out_single = _tmpfile("", suffix=".fasta")
        out_multi = _tmpfile("", suffix=".fasta")
        try:
            pep_single = parse_percolator_xml(xml_path1, 0.01, 0.01)
            filter_fasta(fasta_path, pep_single, out_single)

            pep_multi = set()
            pep_multi |= parse_percolator_xml(xml_path1, 0.01, 0.01)
            pep_multi |= parse_percolator_xml(xml_path2, 0.01, 0.01)
            filter_fasta(fasta_path, pep_multi, out_multi)

            with open(out_single) as f:
                single_headers = [l for l in f if l.startswith(">")]
            with open(out_multi) as f:
                multi_headers = [l for l in f if l.startswith(">")]

            self.assertGreaterEqual(len(multi_headers), len(single_headers))
        finally:
            for p in (fasta_path, xml_path1, xml_path2, out_single, out_multi):
                os.unlink(p)

    def test_end_to_end_single_xml_still_works(self):
        """Single XML file still works (backward compatibility)."""
        fasta_path = _tmpfile(FASTA_BASIC, suffix=".fasta")
        xml_path = _tmpfile(PERC_XML_NO_NS, suffix=".xml")
        out_path = _tmpfile("", suffix=".fasta")
        try:
            passing = set()
            passing |= parse_percolator_xml(xml_path, 0.01, 0.01)
            self.assertEqual(passing, {"PEPTIDE", "TARGETSEQ", "NOMATCH"})

            filter_fasta(fasta_path, passing, out_path)
            with open(out_path) as f:
                output = f.read()

            headers = [l for l in output.splitlines() if l.startswith(">")]
            self.assertEqual(len(headers), 2)
        finally:
            os.unlink(fasta_path)
            os.unlink(xml_path)
            os.unlink(out_path)

    def test_end_to_end_strict_cutoff_filters_more(self):
        """Stricter cutoffs should result in fewer or equal proteins kept."""
        fasta_path = _tmpfile(FASTA_BASIC, suffix=".fasta")
        xml_path = _tmpfile(PERC_XML_NO_NS, suffix=".xml")
        out_loose = _tmpfile("", suffix=".fasta")
        out_strict = _tmpfile("", suffix=".fasta")
        try:
            pep_loose = parse_percolator_xml(xml_path, 0.01, 0.01)
            filter_fasta(fasta_path, pep_loose, out_loose)

            pep_strict = parse_percolator_xml(xml_path, 0.002, 0.002)
            filter_fasta(fasta_path, pep_strict, out_strict)

            with open(out_loose) as f:
                loose_headers = [l for l in f if l.startswith(">")]
            with open(out_strict) as f:
                strict_headers = [l for l in f if l.startswith(">")]

            self.assertGreaterEqual(len(loose_headers), len(strict_headers))
        finally:
            for p in (fasta_path, xml_path, out_loose, out_strict):
                os.unlink(p)


if __name__ == "__main__":
    unittest.main()
