#!/usr/bin/env python3
"""
05_minimize.py

Relax a stitched peptide PDB using OpenBabel's obminimize CLI.

The script calls obminimize via subprocess, captures minimized PDB text from
stdout, and writes it to the requested output file.
"""

import argparse
import os
import subprocess
import sys


def run_obminimize(input_pdb: str, output_pdb: str, forcefield: str, steps: int, algorithm: str) -> None:
    """
    Run OpenBabel minimization and write the minimized PDB to disk.

    Parameters
    ----------
    input_pdb : str
        Path to input stitched PDB.
    output_pdb : str
        Path where minimized PDB should be written.
    forcefield : str
        Force field name (e.g., UFF or GAFF).
    steps : int
        Number of minimization steps.
    algorithm : str
        Minimization algorithm: 'sd' (steepest descent) or 'cg' (conjugate gradient).
    """
    # Build command exactly as obminimize expects.
    # Example: obminimize -ff UFF -sd -n 500 output/stitched.pdb
    cmd = [
        "obminimize",
        "-ff",
        forcefield,
        f"-{algorithm}",
        "-n",
        str(steps),
        input_pdb,
    ]

    # Run process and capture output text.
    # obminimize prints minimized coordinates to stdout.
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    minimized_pdb_text = result.stdout
    if not minimized_pdb_text.strip():
        raise RuntimeError(
            "OpenBabel completed but returned empty output. "
            "Check input structure validity and obminimize installation."
        )

    # Ensure output folder exists before writing the minimized structure.
    out_dir = os.path.dirname(output_pdb)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_pdb, "w", encoding="utf-8") as handle:
        handle.write(minimized_pdb_text)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Minimize a stitched peptide PDB using OpenBabel obminimize."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input stitched PDB path",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output minimized PDB path",
    )
    parser.add_argument(
        "--ff",
        default="UFF",
        choices=["UFF", "GAFF"],
        help="Force field for minimization (default: UFF)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of minimization steps (default: 500)",
    )
    parser.add_argument(
        "--algo",
        default="sd",
        choices=["sd", "cg"],
        help="Minimization algorithm: sd (steepest descent) or cg (conjugate gradient) (default: sd)",
    )

    args = parser.parse_args()

    # Basic input validation before launching subprocess.
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input PDB not found: {args.input}")

    if args.steps <= 0:
        raise ValueError("--steps must be a positive integer.")

    try:
        run_obminimize(
            input_pdb=args.input,
            output_pdb=args.out,
            forcefield=args.ff,
            steps=args.steps,
            algorithm=args.algo,
        )
        print(f"[✓] Minimized structure saved to: {args.out}")

    except FileNotFoundError:
        # Triggered if 'obminimize' executable is not installed or not on PATH,
        # or if the input file path is missing.
        print(
            "[ERROR] OpenBabel obminimize not found (or input file missing). "
            "Please install OpenBabel and ensure 'obminimize' is on your PATH.",
            file=sys.stderr,
        )
        raise

    except subprocess.CalledProcessError as exc:
        # Triggered when obminimize returns a non-zero exit code.
        stderr_text = (exc.stderr or "").strip()
        print("[ERROR] obminimize failed.", file=sys.stderr)
        if stderr_text:
            print(stderr_text, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
