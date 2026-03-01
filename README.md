# Modified Peptide Structure Predictor

A fully automated bioinformatics pipeline for predicting the 3D structures of chemically modified peptides and non-canonical amino acids. 

This tool seamlessly integrates canonical backbone prediction (via AlphaFold2/ESMFold) with side-chain conformer generation (via ETFlow) and structural minimization (via OpenBabel).

## 🚀 Architecture
To prevent dependency conflicts between deep learning frameworks, this pipeline utilizes a master orchestrator (`main.py`) to automatically hand off tasks between two isolated Conda environments.

## 🛠️ Prerequisites & Installation

### 1. Install System Dependencies
You will need Conda (Miniconda/Anaconda) and OpenBabel installed on your system.
```bash
# Example for Ubuntu/Debian
sudo apt-get install openbabel