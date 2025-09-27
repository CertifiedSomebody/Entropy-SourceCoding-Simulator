"""
run_comparisons.py

Utility script to compare Entropy, Huffman, and Shannon–Fano coding results
on text or audio files. Produces a neat console table and optional CSV output.

Usage:
------
Text file:
    python run_comparisons.py --mode text --file sample.txt --max-block 8 --csv results.csv

Audio file:
    python run_comparisons.py --mode audio --file sample.wav --csv audio_results.csv

"""

import argparse
import csv
from Entropy_InfoRate_Simulator import (
    run_text_experiment,
    run_audio_experiment,
)


def print_table(results: dict):
    print("\n=== Comparison Results ===")
    print("{:<25} {:>15}".format("Metric", "Value"))
    print("-" * 42)
    for k, v in results.items():
        if isinstance(v, dict):
            continue  # skip nested dicts (e.g. Hn)
        print("{:<25} {:>15.4f}".format(k, v))


def save_csv(results: dict, filename: str):
    flat_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat_results[f"{k}_{kk}"] = vv
        else:
            flat_results[k] = v
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in flat_results.items():
            writer.writerow([k, v])
    print(f"Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Compare entropy and coding algorithms")
    parser.add_argument("--mode", choices=["text", "audio"], required=True)
    parser.add_argument("--file", required=True, help="Input file (txt or wav)")
    parser.add_argument("--max-block", type=int, default=6, help="Max block size for entropy estimation")
    parser.add_argument("--csv", help="Optional CSV output filename")
    args = parser.parse_args()

    if args.mode == "text":
        results = run_text_experiment(args.file, max_block=args.max_block, plot=False)
    else:
        results = run_audio_experiment(args.file, plot=False)

    print_table(results)

    if args.csv:
        save_csv(results, args.csv)


if __name__ == "__main__":
    main()