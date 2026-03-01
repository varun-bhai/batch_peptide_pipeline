#!/usr/bin/env python3
"""
01_parse_input.py

Minimal parser for peptide sequences containing canonical residues and
three-letter-code modifications in parentheses, e.g. APG(5PG)APG.

Outputs (created under input/):
1) FASTA with fully translated canonical amino acid sequence.
2) TXT with one modification per line in this exact format:
   position : modification_code : SMILES
"""

import argparse
import json
import os
from typing import Dict, List, Tuple


def load_modification_index(json_path: str) -> Dict[str, dict]:
    """Load JSON and index entries by 'Three letter code' for fast lookups."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON must be a list of modification records.")

    index: Dict[str, dict] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        code = str(entry.get("Three letter code", "")).strip()
        if code:
            index[code] = entry
    return index


def extract_parent_one_letter(natural_aa_field: str) -> str:
    """
    Extract parent one-letter code from strings like 'Alanine/Ala/A'.
    We use the last slash-delimited token as the canonical 1-letter residue.
    """
    parts = [p.strip() for p in str(natural_aa_field).split("/") if p.strip()]
    if not parts:
        raise ValueError(f"Invalid 'Natural Amino Acid' field: {natural_aa_field!r}")
    one_letter = parts[-1]
    if len(one_letter) != 1 or not one_letter.isalpha():
        raise ValueError(f"Could not extract 1-letter code from: {natural_aa_field!r}")
    return one_letter.upper()


def parse_sequence(
    sequence: str, mod_index: Dict[str, dict]
) -> Tuple[str, List[Tuple[int, str, str]]]:
    """
    Parse input sequence while maintaining 1-based residue position.

    Rules:
    - Alphabetic characters are treated as canonical residues.
    - Parenthesized text '(XXX)' is treated as a 3-letter modification code.
      It is mapped to parent 1-letter code via JSON and inserted into CAA sequence.

    Returns:
    - canonical sequence string
    - list of tuples: (position, modification_code, smiles)
    """
    caa_chars: List[str] = []
    modifications: List[Tuple[int, str, str]] = []

    residue_position = 1  # 1-based position in final translated sequence
    i = 0
    n = len(sequence)

    while i < n:
        ch = sequence[i]

        # Canonical residue case: single alphabetic character.
        if ch.isalpha():
            caa_chars.append(ch.upper())
            residue_position += 1
            i += 1
            continue

        # Modification case: text inside parentheses.
        if ch == "(":
            end_idx = sequence.find(")", i + 1)
            if end_idx == -1:
                raise ValueError(f"Unclosed parenthesis starting at index {i}.")

            mod_code = sequence[i + 1 : end_idx].strip()
            if not mod_code:
                raise ValueError(f"Empty modification code at index {i}.")

            if mod_code not in mod_index:
                raise ValueError(f"Modification code '{mod_code}' not found in JSON.")

            record = mod_index[mod_code]
            parent_one_letter = extract_parent_one_letter(record.get("Natural Amino Acid", ""))
            smiles = str(record.get("SMILES", "")).strip()
            if not smiles:
                raise ValueError(f"Missing SMILES for modification code '{mod_code}'.")

            # Insert mapped parent residue into canonical sequence.
            caa_chars.append(parent_one_letter)

            # Record position for downstream side-chain generation/stitching.
            modifications.append((residue_position, mod_code, smiles))

            residue_position += 1
            i = end_idx + 1
            continue

        # Any other character is invalid for this minimal parser.
        raise ValueError(
            f"Unexpected character '{ch}' at index {i}. "
            "Only letters and '(THREE_LETTER_CODE)' blocks are supported."
        )

    return "".join(caa_chars), modifications


def write_outputs(caa_sequence: str, modifications: List[Tuple[int, str, str]], fasta_path: str, mods_path: str) -> None:
    """Write FASTA and modification mapping text files."""
    fasta_dir = os.path.dirname(fasta_path)
    mods_dir = os.path.dirname(mods_path)

    if fasta_dir:
        os.makedirs(fasta_dir, exist_ok=True)
    if mods_dir:
        os.makedirs(mods_dir, exist_ok=True)

    # Standard FASTA output for AF2/local colabfold backbone generation.
    with open(fasta_path, "w", encoding="utf-8") as f:
        f.write(">parsed_sequence\n")
        f.write(f"{caa_sequence}\n")

    # Exact line format required by downstream ETFlow + stitching logic.
    with open(mods_path, "w", encoding="utf-8") as f:
        for position, code, smiles in modifications:
            f.write(f"{position} : {code} : {smiles}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse peptide sequence with '(Three letter code)' modifications into CAA FASTA + modification map."
    )
    parser.add_argument(
        "--sequence",
        required=True,
        help="Input peptide sequence, e.g. APG(5PG)APG",
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to modifications JSON file",
    )
    parser.add_argument(
        "--fasta_out",
        required=True,
        help="Output FASTA path",
    )
    parser.add_argument(
        "--mods_out",
        required=True,
        help="Output modification map path",
    )

    args = parser.parse_args()

    mod_index = load_modification_index(args.json)
    caa_sequence, modifications = parse_sequence(args.sequence, mod_index)
    write_outputs(caa_sequence, modifications, args.fasta_out, args.mods_out)


if __name__ == "__main__":
    main()
