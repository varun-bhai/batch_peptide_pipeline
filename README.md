# High-Throughput Modified Peptide Prediction Pipeline

An automated, fault-tolerant computational pipeline for predicting the 3D structures of chemically modified peptides containing non-canonical amino acids (NCAAs), D-amino acids, and other synthetic modifications.

This pipeline is designed for **batch processing**: supply a multi-sequence FASTA and the system will generate minimized, energy-relaxed structures at scale. Each job is isolated to prevent overwriting; failures for individual sequences are logged and the batch continues.

---

## Key Features

- **High-throughput batching** — multi-sequence FASTA input; each sequence writes to its own output directory.  
- **Fault-tolerant architecture** — per-sequence error logging and continued processing.  
- **Smart fallbacks** — ESMFold API for rapid backbone generation, automatic fallback to local AlphaFold2/ColabFold when needed.  
- **Exotic chemistry support** — ETFlow conformer generation mapped to reference RCSB geometries (e.g., CSO, MEA, ALY) with sub-angstrom RMSD where available.

---

## Pipeline Architecture

For every input sequence the pipeline runs a 5-step workflow:

1. **Sequence parsing (`01`)**  
   - Read FASTA, extract inline modification annotations (3-letter codes in parentheses, e.g. `(AIB)`), map canonical backbone positions.

2. **Backbone prediction (`02`)**  
   - Predict backbone coordinates (ESMFold by default; AlphaFold2/ColabFold fallback).

3. **Side-chain generation (`03`)**  
   - Generate 3D conformers for NCAAs with the ETFlow Chemical Language Model.

4. **Stitching (`04`)**  
   - Superimpose and snap modified side-chains onto predicted backbone (BioPython-assisted).

5. **Minimization (`05`)**  
   - Relax the stitched structure with OpenBabel to resolve clashes and produce the final minimized PDB.

---

## Input Formats

### 1. Batch FASTA (`--fasta_in`)
- Use standard 1-letter codes for canonical amino acids.
- Inline modifications should be placed in parentheses after the residue using their 3-letter code.

**Example FASTA (copy/paste-ready):**
```fasta
>2rln
KETAAAKFERQH(NLE)D(SET)
>3cmh
LDKE(AIB)VYFCHLDIIW
>3kmz
RLITLADHI(CSO)QIITQDFAR
>3zs2
FVNQHLCGSHLVEALYLVCGERGFY(MEA)TKPT
>4lka
ARTKQTAR(ALY)STGGKAPRKQLA
>6cmh
CSASSLLDKE(AIB)VYFCHLDIIW
