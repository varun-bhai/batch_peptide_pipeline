#!/usr/bin/env python3
"""
04_stitch.py

Stitch ETFlow-generated modified side chains into an AF2/ESMFold backbone.

Pipeline behavior:
1) Read modification positions/codes from a text file:
      position : modification_code : SMILES
2) Load target backbone structure (output/backbone.pdb).
3) For each modification, load output/mod_{position}_{code}.pdb,
   but first clean it in-memory by removing lines starting with CONECT or END.
4) Apply the same swap logic used in your older dist_batch_2.py:
   - sanitize source residue (remove OXT + all H atoms)
   - align using common anchors N, CA, C, CB
   - replace target residue at requested position
5) Save stitched structure as output/stitched.pdb.
"""

import argparse
import io
import os
import warnings
from typing import List, Tuple

from Bio import PDB
from Bio.PDB.PDBExceptions import PDBConstructionWarning


# Suppress noisy parser warnings (matches style in older script).
warnings.filterwarnings("ignore", category=PDBConstructionWarning)


def parse_modifications_file(mods_path: str) -> List[Tuple[int, str]]:
    """
    Parse modifications text file and return only (position, modification_code).

    Expected line format:
        position : modification_code : SMILES

    We intentionally ignore SMILES here because stitching only needs the residue
    position and the modification code to locate side-chain PDB files.
    """
    if not os.path.exists(mods_path):
        raise FileNotFoundError(f"Modifications file not found: {mods_path}")

    parsed: List[Tuple[int, str]] = []

    with open(mods_path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            # Skip empty lines.
            if not line:
                continue

            # Keep strict 3-field parsing while allowing ':' inside SMILES.
            parts = [p.strip() for p in line.split(":", 2)]
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid line {line_no}: {line!r}. "
                    "Expected format: position : modification_code : SMILES"
                )

            pos_text, mod_code, _smiles = parts

            if not pos_text.isdigit():
                raise ValueError(f"Invalid position on line {line_no}: {pos_text!r}")
            if not mod_code:
                raise ValueError(f"Missing modification code on line {line_no}.")

            parsed.append((int(pos_text), mod_code))

    return parsed


def clean_pdb_text_in_memory(pdb_path: str) -> str:
    """
    Read a PDB file and remove lines starting with CONECT or END in-memory.

    This is equivalent to your clean_et_flow_mod_pdb.py filtering logic,
    but avoids writing temporary files.
    """
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Modification PDB not found: {pdb_path}")

    cleaned_lines: List[str] = []

    with open(pdb_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("CONECT") or line.startswith("END"):
                continue
            cleaned_lines.append(line)

    return "".join(cleaned_lines)


def load_source_structure_from_cleaned_text(parser: PDB.PDBParser, structure_id: str, cleaned_pdb_text: str):
    """
    Load a cleaned PDB string into BioPython using StringIO (in-memory file handle).
    """
    return parser.get_structure(structure_id, io.StringIO(cleaned_pdb_text))


def perform_swap(target_struct, target_res_id: int, source_struct, chain_length: int) -> bool:
    """
    Performs alignment and residue swap using logic adapted from dist_batch_2.py.

    POSITION-AWARE SANITIZATION:
    - Internal NCAA: remove OXT, HXT, H2, H3 (terminal atoms only)
    - N-terminal NCAA: remove OXT, HXT (keep H2, H3 — they belong at chain start)
    - C-terminal NCAA: remove H2, H3 (keep OXT, HXT — they belong at chain end)
    - ALL side-chain hydrogens are always kept.

    Alignment anchors:
    - N, CA, C, CB (requires at least 3 common atoms).
    """
    try:
        target_model = target_struct[0]

        # --- BLIND CHAIN DETECTION START (from older logic) ---
        chains = list(target_model.get_chains())

        if len(chains) == 0:
            print("    [!] Error: Target structure has no chains.")
            return False

        # Assume peptide is in the first chain for this single-sequence pipeline.
        target_chain = chains[0]

        if len(chains) > 1:
            print(
                f"    [!] Warning: Target has multiple chains ({len(chains)}). "
                f"Operating on first one: {target_chain.id}"
            )
        # --- BLIND CHAIN DETECTION END ---

        # Chain membership/indexing accepts int residue IDs in BioPython.
        if target_res_id not in target_chain:
            print(f"    [!] Error: Residue {target_res_id} not found in chain {target_chain.id}.")
            return False

        target_res = target_chain[target_res_id]

        source_residues = list(source_struct.get_residues())
        if not source_residues:
            print("    [!] Error: Source structure has no residues.")
            return False
        source_res = source_residues[0]

    except (KeyError, IndexError) as exc:
        print(f"    [!] Error accessing residues: {exc}")
        return False

    # =========================================================================
    # POSITION-AWARE SANITIZATION
    #
    # ETFlow generates a FREE amino acid (standalone, not in a chain).
    # A free amino acid has extra atoms at both ends:
    #   - Carboxyl end: OXT (oxygen) + HXT (hydrogen on it) = the -OH group
    #   - Amino end: H2, H3 = extra hydrogens on nitrogen (NH2 or NH3+)
    #
    # In a polymer chain, these terminal atoms must be removed — UNLESS
    # the NCAA is actually at the chain terminus where they belong.
    #
    # ALL side-chain hydrogens are ALWAYS kept. They are chemically
    # correct from ETFlow and needed by MACE-OFF for proper minimization.
    # =========================================================================
    is_n_terminal = (target_res_id == 1)
    is_c_terminal = (target_res_id == chain_length)

    # Names of atoms to selectively remove
    # OXT/HXT = the -OH on the carboxyl end (remove unless C-terminal)
    # H2/H3   = extra hydrogens on nitrogen (remove unless N-terminal)
    carboxyl_terminal_atoms = {"OXT", "HXT"}
    amino_terminal_atoms = {"H2", "H3"}

    atoms_to_detach: List[str] = []

    for atom in source_res:
        atom_name = atom.name.strip()

        # Remove carboxyl terminal atoms (OXT, HXT) unless at C-terminus
        if atom_name in carboxyl_terminal_atoms and not is_c_terminal:
            atoms_to_detach.append(atom.name)
            continue

        # Remove amino terminal atoms (H2, H3) unless at N-terminus
        if atom_name in amino_terminal_atoms and not is_n_terminal:
            atoms_to_detach.append(atom.name)
            continue

    for atom_name in atoms_to_detach:
        if atom_name in source_res:
            source_res.detach_child(atom_name)

    # Log what we did
    pos_type = "N-terminal" if is_n_terminal else ("C-terminal" if is_c_terminal else "internal")
    print(f"    Sanitization ({pos_type}): removed {atoms_to_detach if atoms_to_detach else 'nothing'}")
    # =========================================================================

    # --- Alignment logic (same anchors as old script) ---
    potential_anchors = ["N", "CA", "C", "CB"]
    fixed_atoms = []   # Atoms in target residue (reference frame)
    moving_atoms = []  # Corresponding atoms in source residue (to move)

    for atom_name in potential_anchors:
        if atom_name in target_res and atom_name in source_res:
            if not target_res[atom_name].is_disordered() and not source_res[atom_name].is_disordered():
                fixed_atoms.append(target_res[atom_name])
                moving_atoms.append(source_res[atom_name])

    # Need at least 3 atoms to define a stable 3D superimposition.
    if len(fixed_atoms) < 3:
        print(f"    [!] Critical: Fewer than 3 common backbone atoms at pos {target_res_id}. Skipping.")
        return False

    # Apply superimposition to move the source residue into target frame.
    superimposer = PDB.Superimposer()
    superimposer.set_atoms(fixed_atoms, moving_atoms)
    superimposer.apply(source_res.get_atoms())

    # --- Swapping logic ---
    old_id = target_res.id
    target_chain.detach_child(old_id)

    # Reuse target residue ID so sequence numbering is preserved.
    source_res.id = old_id
    target_chain.add(source_res)

    # Keep residues ordered by sequence number.
    target_chain.child_list.sort(key=lambda x: x.id[1])

    print(
        f"    -> Swapped pos {target_res_id} on Chain {target_chain.id} "
        f"with {source_res.get_resname()} (RMSD: {superimposer.rms:.4f} A)"
    )
    return True


def main() -> None:
    """CLI entry point for single-structure stitching."""
    parser = argparse.ArgumentParser(
        description="Stitch ETFlow modification residues into backbone PDB using BioPython superimposition."
    )
    parser.add_argument(
        "--mods",
        required=True,
        help="Path to modifications mapping text file",
    )
    parser.add_argument(
        "--backbone",
        required=True,
        help="Path to backbone PDB to modify",
    )
    parser.add_argument(
        "--mod_dir",
        required=True,
        help="Directory containing side-chain PDB files mod_{position}_{code}.pdb",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output stitched PDB path",
    )

    args = parser.parse_args()

    # Read modification definitions (position + code).
    modifications = parse_modifications_file(args.mods)
    if not modifications:
        print("No modifications found in mapping file; saving backbone unchanged.")

    # Load target backbone structure.
    if not os.path.exists(args.backbone):
        raise FileNotFoundError(f"Backbone PDB not found: {args.backbone}")

    pdb_parser = PDB.PDBParser(QUIET=True)
    target_struct = pdb_parser.get_structure("target", args.backbone)

    # Count residues in first chain so perform_swap knows about terminal positions.
    target_chain = list(target_struct[0].get_chains())[0]
    chain_length = len(list(target_chain.get_residues()))
    print(f"[Stitch] Chain has {chain_length} residues.")

    # Apply each swap in order from the modifications file.
    swaps_applied = 0
    for position, mod_code in modifications:
        mod_filename = f"mod_{position}_{mod_code}.pdb"
        mod_path = os.path.join(args.mod_dir, mod_filename)

        print(f"\n[Swap] position={position}, code={mod_code}")

        if not os.path.exists(mod_path):
            print(f"    [!] Missing modification file: {mod_path}")
            continue

        try:
            # 1) Clean source PDB text in-memory (remove CONECT/END records).
            cleaned_text = clean_pdb_text_in_memory(mod_path)

            # 2) Parse cleaned string directly via StringIO.
            source_struct = load_source_structure_from_cleaned_text(
                pdb_parser,
                structure_id=f"source_{position}_{mod_code}",
                cleaned_pdb_text=cleaned_text,
            )

            # 3) Perform alignment + replacement.
            if perform_swap(target_struct, position, source_struct, chain_length):
                swaps_applied += 1

        except Exception as exc:
            print(f"    [!] Failed to process {mod_filename}: {exc}")

    # Ensure output directory exists.
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Save final stitched structure (or unchanged backbone if no swaps succeeded).
    io_writer = PDB.PDBIO()
    io_writer.set_structure(target_struct)
    io_writer.save(args.out)

    print("\n" + "=" * 60)
    print(f"Stitching complete. Applied {swaps_applied}/{len(modifications)} swaps.")
    print(f"Saved stitched structure: {args.out}")


if __name__ == "__main__":
    main()
