#!/usr/bin/env python3
"""
05_minimize.py (MACE-OFF Version)

Takes the stitched.pdb file, adds missing hydrogens using OpenMM (safely 
handling NCAAs), and relaxes the structure using the MACE-OFF machine 
learning force field via ASE.
"""

import argparse
import os
import sys
import time
import tempfile
import numpy as np
import torch

from ase.io import read, write
from ase.optimize import LBFGS
from mace.calculators import mace_off
from pdbfixer import PDBFixer
from openmm.app import PDBFile

STANDARD_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL',
    'HID', 'HIE', 'HIP', 'CYX', 'ASH', 'GLH',
    'ACE', 'NME', 'NMA',
    'HOH', 'WAT', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN',
}

# ==============================================================================
# 1. CORE FUNCTIONS (From your MACE Notebook)
# ==============================================================================

def preprocess(pdb_path, output_dir):
    def _is_h(line):
        atom_name = line[12:16].strip()
        element   = line[76:78].strip() if len(line) > 76 else ''
        if element == 'H': return True
        if not element:
            return atom_name.startswith('H') or (len(atom_name) > 1 and atom_name[0].isdigit() and atom_name[1] == 'H')
        return False

    def _res_key(line):
        return (line[17:20].strip(), line[21], line[22:26].strip())

    # --- STEP 1: save NCAA H from init, write H-stripped init to temp file ---
    ncaa_h_saved = {}   # res_key -> list of original ATOM lines for H atoms
    stripped_lines = []

    with open(str(pdb_path)) as fh:
        for raw in fh:
            if raw.startswith(('ATOM', 'HETATM')):
                res_name = raw[17:20].strip()
                key = _res_key(raw)
                if res_name not in STANDARD_RESIDUES and _is_h(raw):
                    ncaa_h_saved.setdefault(key, []).append(raw)
                else:
                    stripped_lines.append(raw)
            else:
                stripped_lines.append(raw)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix='_stripped.pdb')
    os.close(tmp_fd)
    with open(tmp_path, 'w') as fh:
        fh.writelines(stripped_lines)

    # --- STEP 2: run PDBFixer exactly as before, on the stripped file ---
    fixer = PDBFixer(filename=tmp_path)
    os.unlink(tmp_path)

    n_initial = sum(1 for _ in fixer.topology.atoms()) + sum(len(v) for v in ncaa_h_saved.values())

    ncaa_list = []
    for res in fixer.topology.residues():
        if res.name not in STANDARD_RESIDUES:
            key = (res.name, '', str(res.id))
            n_h = len(ncaa_h_saved.get(key, []))
            ncaa_list.append((res.name, res.index + 1, n_h))

    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)

    tmp_fd2, tmp_proto = tempfile.mkstemp(suffix='_pdbfixer_out.pdb')
    os.close(tmp_fd2)
    with open(tmp_proto, 'w') as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)

    # --- STEP 3: in PDBFixer's output, remove NCAA H and paste back original H ---
    pdbfixer_lines = open(tmp_proto).readlines()
    os.unlink(tmp_proto)

    last_idx = {}
    for i, raw in enumerate(pdbfixer_lines):
        if raw.startswith(('ATOM', 'HETATM')):
            res_name = raw[17:20].strip()
            if res_name not in STANDARD_RESIDUES:
                last_idx[_res_key(raw)] = i

    out_lines = []
    serial = 1
    inserted = set()
    for i, raw in enumerate(pdbfixer_lines):
        if raw.startswith(('ATOM', 'HETATM')):
            res_name = raw[17:20].strip()
            key = _res_key(raw)
            if res_name not in STANDARD_RESIDUES and _is_h(raw):
                continue
            out_lines.append(f"{raw[:6]}{serial:5d}{raw[11:]}")
            serial += 1
            if res_name not in STANDARD_RESIDUES and i == last_idx.get(key) and key not in inserted:
                for h_raw in ncaa_h_saved.get(key, []):
                    out_lines.append(f"{h_raw[:6]}{serial:5d}{h_raw[11:]}")
                    serial += 1
                inserted.add(key)
        else:
            out_lines.append(raw)

    protonated_path = os.path.join(output_dir, 'protonated.pdb')
    with open(protonated_path, 'w') as fh:
        fh.writelines(out_lines)

    final_atom_lines = [l for l in out_lines if l.startswith(('ATOM', 'HETATM'))]
    n_final   = len(final_atom_lines)
    elements  = sorted(set(l[76:78].strip() for l in final_atom_lines if len(l) > 76 and l[76:78].strip()))

    return protonated_path, n_initial, n_final, ncaa_list, elements

class ExplosionError(Exception):
    pass

class ExplosionDetector:
    def __init__(self, atoms, initial_energy, initial_positions, check_every=50):
        self.atoms = atoms
        self.e0 = initial_energy
        self.pos0 = initial_positions.copy()
        self.check_every = check_every
        self.step_count = 0

    def __call__(self):
        self.step_count += 1
        if self.step_count % self.check_every != 0:
            return

        e_now = self.atoms.get_potential_energy()
        delta_e = e_now - self.e0

        if delta_e < -1000:
            raise ExplosionError(f"Energy dropped {delta_e:.0f} eV. Structure is exploding.")
        if e_now > 0:
            raise ExplosionError(f"Energy is positive ({e_now:.0f} eV). Atoms are overlapping catastrophically.")

        pos_now = self.atoms.get_positions()
        max_disp = np.max(np.linalg.norm(pos_now - self.pos0, axis=1))
        if max_disp > 10.0:
            raise ExplosionError(f"Max atomic displacement is {max_disp:.1f} Å. Structure is disintegrating.")

def minimize(protonated_pdb, output_dir, model_size, fmax, max_steps, device):
    atoms = read(protonated_pdb)
    n_atoms = len(atoms)

    calc = mace_off(model=model_size, device=device, default_dtype="float64")
    atoms.calc = calc

    e0 = atoms.get_potential_energy()
    f0_max = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
    pos0 = atoms.get_positions().copy()

    if e0 > 0:
        print(f"    ⚠ Initial energy is POSITIVE ({e0:.0f} eV) — input structure has severe overlaps")

    logfile = os.path.join(output_dir, 'opt.log')
    trajectory = os.path.join(output_dir, 'opt.traj')
    ase_raw = os.path.join(output_dir, '_ase_raw.pdb')

    opt = LBFGS(atoms, logfile=logfile, trajectory=trajectory)
    detector = ExplosionDetector(atoms, e0, pos0, check_every=50)
    opt.attach(detector)

    t0 = time.time()
    exploded = False
    explosion_msg = ""

    try:
        converged = opt.run(fmax=fmax, steps=max_steps)
    except ExplosionError as ex:
        exploded = True
        explosion_msg = str(ex)
        converged = False

    elapsed = time.time() - t0
    e1 = atoms.get_potential_energy()
    f1_max = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
    write(ase_raw, atoms)

    return {
        'n_atoms': n_atoms,
        'e0': e0, 'e1': e1, 'delta_e': e1 - e0,
        'f0_max': f0_max, 'f1_max': f1_max,
        'steps': opt.nsteps, 'converged': converged,
        'exploded': exploded, 'explosion_msg': explosion_msg,
        'time_sec': elapsed,
        'time_per_step': elapsed / max(opt.nsteps, 1),
        'ase_raw_path': ase_raw,
    }

def reformat_pdb(protonated_pdb, ase_raw_pdb, output_pdb):
    prot_lines = []
    with open(protonated_pdb) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                prot_lines.append(line)
    min_coords = []
    with open(ase_raw_pdb) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                min_coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    if len(prot_lines) != len(min_coords):
        import shutil
        shutil.copy(ase_raw_pdb, output_pdb)
        return False
    with open(output_pdb, 'w') as f:
        for i in range(len(prot_lines)):
            line = prot_lines[i]
            x, y, z = min_coords[i]
            f.write(line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:])
        f.write("TER\n")
        f.write("END\n")
    return True

# ==============================================================================
# 2. THE PIPELINE ORCHESTRATOR 
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Minimize stitched PDB using MACE-OFF.")
    parser.add_argument("--input", required=True, help="Path to input stitched.pdb")
    parser.add_argument("--out", required=True, help="Path to output final_minimized.pdb")
    args = parser.parse_args()

    input_pdb = os.path.abspath(args.input)
    final_pdb = os.path.abspath(args.out)
    work_dir = os.path.dirname(input_pdb)

    if not os.path.exists(input_pdb):
        print(f"[ERROR] Minimizer input not found: {input_pdb}")
        sys.exit(1)

    print(f"[MACE] Starting Machine Learning Minimization...")

    # Determine Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MACE] PyTorch Device: {device.upper()}")

    try:
        # Step 1: Preprocess
        print("[MACE] Preprocessing and protonating structure...")
        protonated_path, n_initial, n_final, ncaa_list, elements = preprocess(input_pdb, work_dir)

        # Step 2: Minimize
        print(f"[MACE] Running MACE-OFF (medium model, float64)...")
        stats = minimize(
            protonated_pdb=protonated_path, 
            output_dir=work_dir, 
            model_size="medium", 
            fmax=0.05, 
            max_steps=500, 
            device=device
        )

        if stats['exploded']:
            print(f"[ERROR] Optimization failed! {stats['explosion_msg']}")
            sys.exit(1)
            
        print(f"[MACE] Converged in {stats['steps']} steps. ΔE = {stats['delta_e']:.2f} eV")

        # Step 3: Reformat
        print("[MACE] Reformatting final coordinates into clean PDB...")
        reformat_pdb(protonated_path, stats['ase_raw_path'], final_pdb)
        
        print(f"[MACE] Minimization complete! Saved to: {final_pdb}")

    except Exception as e:
        print(f"[ERROR] MACE minimization failed: {e}")
        sys.exit(1)
        
    finally:
        # Clean up temporary files
        temp_files = ["temp_stripped.pdb", "temp_pdbfixer_out.pdb", "protonated.pdb", "_ase_raw.pdb"]
        for f in temp_files:
            path = os.path.join(work_dir, f)
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    main()