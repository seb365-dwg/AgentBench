import os
import sys
from math import comb
from pathlib import Path

import pandas as pd


def process_single_file(output_file):
    """Process a single file and return basic stats"""
    df = pd.read_json(output_file, lines=True)
    df = df[df['result'] != "error"]
    df = df[df['run_number'] < 256]
    valid = len(df)
    avg = df["result"].mean()
    std = df.groupby(["run_number"])["result"].mean().std()
    bon = df.groupby(["task_index"])["result"].max().mean()

    df['is_pass'] = df['result'] == 1
    pass_rate_per_run = df.groupby('run_number')['is_pass'].mean()
    mean_pass_rate = pass_rate_per_run.mean()
    std_pass_rate = pass_rate_per_run.std()

    return valid, avg, std, bon, mean_pass_rate, std_pass_rate


output_file = sys.argv[1]

if os.path.isdir(output_file):
    # Process directory
    json_files = list(Path(output_file).glob("*.jsonl"))
    if not json_files:
        print(f"No .jsonl files found in directory: {output_file}")
        sys.exit(1)

    print("File\tValid\tAvg\tStd\tBest_of_n\tPass_Rate_Mean\tPass_Rate_Std")
    for json_file in sorted(json_files):
        try:
            valid, avg, std, bon, mean_pass_rate, std_pass_rate = process_single_file(json_file)
            filename = json_file.name
            print(f"{filename}\t{valid}\t{avg:.3f}\t{std:.3f}\t{bon:.3f}\t{mean_pass_rate:.3f}\t{std_pass_rate:.3f}")
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
else:
    # Original single file processing
    df = pd.read_json(output_file, lines=True)
    df = df[df['result'] != "error"]
    df = df[df['run_number'] < 256]
    valid = len(df)
    avg = df["result"].mean()
    std = df.groupby(["run_number"])["result"].mean().std()
    bon = df.groupby(["task_index"])["result"].max().mean()
    print(f"Valid: {valid} Avg: {avg:.3f} ± {std:.3f} | Best of n: {bon:.3f}")

    df['is_pass'] = df['result'] == 1
    pass_rate_per_run = df.groupby('run_number')['is_pass'].mean()
    mean_pass_rate = pass_rate_per_run.mean()
    std_pass_rate = pass_rate_per_run.std()

    print(f"pass rate mean: {mean_pass_rate:.3f} std: {std_pass_rate:.3f}")

    if len(sys.argv) > 2:
        output_file2 = sys.argv[2]
        df2 = pd.read_json(output_file2, lines=True)
        df = pd.concat([df, df2])
        print("calculating pass@k for two files")

    pass_at_k = {}
    grouped = df.groupby(['task_index'])
    n = grouped.size().max()
    ks = []
    i = 1
    while i < n:
        ks.append(i)
        i *= 2
    if ks[-1] != n:
        ks.append(n)

    for k in ks:
        scores = []
        for task, group in grouped:
            c = (group["result"] == 1).sum()
            score = 1.0 - comb(n - c, k) / comb(n, k)
            scores.append(score)
        pass_at_k[k] = sum(scores) / len(scores)

    print("pass@k:")
    for k, v in pass_at_k.items():
        print(f"pass@{k}: {v:.3f}")
