#!/usr/bin/env python3
"""
main.py

High-throughput orchestrator for the modified-peptide structure pipeline.

This script reads a multi-sequence FASTA file and runs the full 5-step pipeline
for each sequence independently. Each sequence is isolated under:
    output/{job_name}/

If one sequence fails, the pipeline logs the error and continues to the next.
"""

import argparse
import os
import subprocess
from typing import List, Tuple


def print_banner(message: str) -> None:
    """Print a highly visible progress banner."""
    line = "=" * 90
    print(f"\n{line}\n[INFO] {message}\n{line}", flush=True)


def run_step(step_name: str, cmd: List[str], cwd: str) -> None:
    """
    Run a single subprocess step and fail fast for this sequence.

    check=True ensures a non-zero exit code raises CalledProcessError.
    """
    print_banner(step_name)
    print(f"[INFO] Command: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def read_multi_fasta(fasta_path: str) -> List[Tuple[str, str]]:
    """
    Parse a multi-sequence FASTA file.

    Returns:
        List of tuples: (job_name, sequence)

    Rules:
    - job_name is the FASTA header text after '>' (trimmed).
    - sequence is concatenated from all sequence lines under that header.
    """
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"Input FASTA not found: {fasta_path}")

    records: List[Tuple[str, str]] = []
    current_header = None
    current_seq_chunks: List[str] = []

    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Flush previous record before starting a new one.
                if current_header is not None:
                    sequence = "".join(current_seq_chunks).strip()
                    if not sequence:
                        raise ValueError(f"Empty sequence for FASTA header '{current_header}'.")
                    records.append((current_header, sequence))

                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Empty FASTA header at line {line_no}.")

                current_header = header
                current_seq_chunks = []
                continue

            if current_header is None:
                raise ValueError(
                    f"FASTA format error at line {line_no}: sequence data before first header."
                )

            current_seq_chunks.append(line)

    # Flush the final record.
    if current_header is not None:
        sequence = "".join(current_seq_chunks).strip()
        if not sequence:
            raise ValueError(f"Empty sequence for FASTA header '{current_header}'.")
        records.append((current_header, sequence))

    if not records:
        raise ValueError("No FASTA records found in input file.")

    return records


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run the full modified-peptide pipeline for each sequence in a multi-FASTA file."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--fasta_in",
        help="Path to multi-sequence FASTA input file",
    )
    input_group.add_argument(
        "--sequence",
        help='Single raw peptide sequence string, e.g. "APG(5PG)APG"',
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to modifications JSON dictionary",
    )
    parser.add_argument(
        "--job_name",
        default="single_peptide",
        help='Output folder name for --sequence mode (default: "single_peptide")',
    )

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.abspath(args.json)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Modifications JSON file not found: {json_path}")

    # Resolve sub-script paths once.
    step1_script = os.path.join(project_root, "01_parse_input.py")
    step2_script = os.path.join(project_root, "02_run_backbone.py")
    step3_script = os.path.join(project_root, "03_run_sidechains.py")
    step4_script = os.path.join(project_root, "04_stitch.py")
    step5_script = os.path.join(project_root, "05_minimize.py")

    for script_path in [step1_script, step2_script, step3_script, step4_script, step5_script]:
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Required script not found: {script_path}")

    if args.sequence is not None:
        records = [(args.job_name, args.sequence)]
        print_banner("Loaded 1 sequence from command line input")
    else:
        fasta_in = os.path.abspath(args.fasta_in)
        records = read_multi_fasta(fasta_in)
        print_banner(f"Loaded {len(records)} sequence(s) from {fasta_in}")

    successes = 0
    failures = 0

    # Process one peptide at a time; each peptide gets an isolated output folder.
    for idx, (job_name, sequence) in enumerate(records, start=1):
        print_banner(f"Starting job {idx}/{len(records)}: {job_name}")

        # Dedicated directory for this sequence to prevent overwrites.
        job_dir = os.path.join(project_root, "output", job_name)
        os.makedirs(job_dir, exist_ok=True)

        # All paths are explicit and scoped to this job directory.
        parsed_fasta = os.path.join(job_dir, "parsed_sequence.fasta")
        mods_txt = os.path.join(job_dir, "modifications.txt")
        backbone_pdb = os.path.join(job_dir, "backbone.pdb")
        stitched_pdb = os.path.join(job_dir, "stitched.pdb")
        minimized_pdb = os.path.join(job_dir, "final_minimized.pdb")

        # Each job runs the complete 5-step pipeline.
        # If any step fails, we log and continue with the next sequence.
        try:
            # Step 1: Parse input sequence to CAA FASTA + modification map.
            run_step(
                f"{job_name} | Step 1/5: Parsing sequence",
                [
                    "python",
                    step1_script,
                    "--sequence",
                    sequence,
                    "--json",
                    json_path,
                    "--fasta_out",
                    parsed_fasta,
                    "--mods_out",
                    mods_txt,
                ],
                cwd=project_root,
            )

            # Step 2: Backbone prediction in AF2 environment.
            run_step(
                f"{job_name} | Step 2/5: Backbone prediction (af2_env)",
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "af2_env",
                    "python",
                    step2_script,
                    "--fasta",
                    parsed_fasta,
                    "--out_pdb",
                    backbone_pdb,
                ],
                cwd=project_root,
            )

            # Step 3: Side-chain generation in ETFlow environment.
            run_step(
                f"{job_name} | Step 3/5: Side-chain generation (etflow_env)",
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "etflow_env",
                    "python",
                    step3_script,
                    "--mods",
                    mods_txt,
                    "--out_dir",
                    job_dir,
                ],
                cwd=project_root,
            )

            # Step 4: Stitch side-chains into backbone.
            run_step(
                f"{job_name} | Step 4/5: Stitching",
                [
                    "python",
                    step4_script,
                    "--mods",
                    mods_txt,
                    "--backbone",
                    backbone_pdb,
                    "--mod_dir",
                    job_dir,
                    "--out",
                    stitched_pdb,
                ],
                cwd=project_root,
            )

            # Step 5: Minimize final structure.
        #     run_step(
        #         f"{job_name} | Step 5/5: Minimization",
        #         [
        #             "python",
        #             step5_script,
        #             "--input",
        #             stitched_pdb,
        #             "--out",
        #             minimized_pdb,
        #         ],
        #         cwd=project_root,
        #     )

        #     successes += 1
        #     print(f"[INFO] Completed job '{job_name}' successfully.", flush=True)

        # except (subprocess.CalledProcessError, FileNotFoundError, ValueError, RuntimeError) as exc:
        #     failures += 1
        #     print(f"[ERROR] Job '{job_name}' failed: {exc}", flush=True)
        #     print("[INFO] Continuing to next sequence...", flush=True)
        #     continue
            # Step 5: Minimize final structure.
            run_step(
                f"{job_name} | Step 5/5: Minimization",
                [
                    "python",
                    step5_script,
                    "--input",
                    stitched_pdb,
                    "--out",
                    minimized_pdb,
                ],
                cwd=project_root,
            )

            successes += 1
            print(f"[INFO] Completed job '{job_name}' successfully.", flush=True)

            # --- ADD THIS NEW BLOCK ---
            # Immediately back up the completed folder to Google Drive
            drive_dest = f"/content/drive/MyDrive/try_pipeline/{job_name}"
            # Ensure the parent directory exists
            os.makedirs("/content/drive/MyDrive/try_pipeline", exist_ok=True)
            # Copy the folder over
            subprocess.run(["cp", "-r", job_dir, drive_dest], check=False)
            print(f"[INFO] Safely backed up '{job_name}' to Google Drive.", flush=True)
            # --------------------------

        except (subprocess.CalledProcessError, FileNotFoundError, ValueError, RuntimeError) as exc:
            failures += 1
            print(f"[ERROR] Job '{job_name}' failed: {exc}", flush=True)
            print("[INFO] Continuing to next sequence...", flush=True)
            continue

    print_banner("Batch run finished")
    print(f"[INFO] Successful jobs: {successes}", flush=True)
    print(f"[INFO] Failed jobs: {failures}", flush=True)


if __name__ == "__main__":
    main()
