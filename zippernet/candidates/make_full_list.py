"""Distribute make_list.py jobs and aggregate outputs."""

import argparse
import glob
import os
import sys

import pandas as pd

# Handle command-line arguments.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--data_path", type=str, 
    default="/data/des81.b/data/stronglens/DEEP_FIELDS/SEARCH",
    help="Path to directory with prediction output.")
parser.add_argument(
    "--outdir", type=str, help="Desired loaction for output.")
parser.add_argument(
    "--sequence_length", type=int, default=10, help="Length of sequences used.")
parser.add_argument(
    "--check_progress", action="store_true", help="Print progress and exit.")
parser_args = parser.parse_args()

nodes = [
    "des30", "des31", "des40", "des41", "des50", "des60", "des70", 
    "des71", "des80", "des81", "des90", "des91"][::-1]

job_filename = f"{parser_args.outdir}/status/jobs.txt"
done_nodes = []
if parser_args.check_progress:
    with open(job_filename, 'r') as job_file:
        lines = job_file.readlines()

        candidate_count, done_count, total_count = 0, 0, 0

        for line in lines:
            node, files = line.split(':')

            filenames = [x.strip() for x in files.split(',')]
            cutout_names = [x.split('/')[-1].split('_classifications')[0] for x in filenames]

            # Check overall status.
            done, total = 0, 0
            for cutout_name in cutout_names:
                total += 1
                total_count += 1
                if os.path.exists(f"{parser_args.outdir}/status/{node}_{cutout_name}.DONE"):
                    done += 1
                    done_count += 1

            # Check if finished.
            done_str = ''
            if (os.path.exists(f"{parser_args.outdir}/{node}_candidates.csv") or 
                os.path.exists(f"{parser_args.outdir}/{node}_candidates.EMPTY")):
                done_str = '  -- DONE!'
                done_nodes.append(node)

            # Count candidates.
            try:
                cand_file = glob.glob(f"{parser_args.outdir}/status/*_{node}.CANDS")[0]
                cands = int(cand_file.split('/')[-1].split(f'_{node}')[0])
            except IndexError:
                # progress checked while a particular cand file was being overwritten.
                cands = 0
            candidate_count += cands

            print(node, ':\tDONE: ', done, '\tTODO: ', total - done, '\tTOTAL: ', total, done_str)

        progress = done_count / total_count * 100.0
        print(f"\nProgress: {progress:.2f} %", f"  Found {candidate_count} candidates.")

    # Merge dataframes if all done.
    if len(done_nodes) == len(nodes):
        candidate_files = glob.glob(f"{parser_args.outdir}/*_candidates.csv")
        if len(candidate_files) != 0:
            dfs = [pd.read_csv(filename) for filename in candidate_files]
            out_df = pd.concat(dfs)
            out_df.sort_values(by=['DETECTION_SCORE', 'SCORE'], ascending=False, inplace=True)
            out_df.to_csv(f"{parser_args.outdir}/all_candidates.csv", index=False)
        else:
            print("No candidates detected -- all_candidates.csv not made.")

    sys.exit()


if os.path.exists(parser_args.outdir):
    os.system(f"rm -r {parser_args.outdir}")
os.system(f"mkdir {parser_args.outdir}")  
os.system(f"mkdir {parser_args.outdir}/status")

# Distribute jobs.
filenames = glob.glob(f'{parser_args.data_path}/*classifications_{parser_args.sequence_length}.csv')
jobs = {n: [] for n in nodes}
node_idx = 0
for filename in filenames:
    jobs[nodes[node_idx]].append(filename)
    node_idx += 1
    if node_idx == len(nodes):
        node_idx = 0

with open(job_filename, "w+") as job_file:
    for node, files in jobs.items():

        cmd = (
            f'ssh rmorgan@{node}.fnal.gov ' +
            '"source /data/des81.b/data/stronglens/setup.sh && '
            'cd /data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/zippernet/candidates/ && '
            'python make_list.py '
            f'--data_path {parser_args.data_path} ' +
            f'--outdir {parser_args.outdir} ' + 
            f'--sequence_length {parser_args.sequence_length} ' + 
            f'--node {node} ' + 
            f'--filenames {",".join(files)}" &')
        os.system(cmd)

        job_file.write(f"{node}:{','.join(files)}\n")

