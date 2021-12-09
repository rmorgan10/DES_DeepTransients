"""Evaluate TESTING data with trained network."""

import argparse
import glob
import os
import sys

from data_utils import BASE_DATA_PATH


if __name__ == '__main__':
    # Handle command-line arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network", type=str, help="Name file containing saved network.")
    parser.add_argument(
        "--config_file", type=str, 
        help="Name of config file used in training.")
    parser.add_argument(
        "--outdir", type=str, help="Directtory to save output.")
    parser.add_argument(
        "--check_progress", action="store_true", 
        help="Display progress and exit.")
    parser_args = parser.parse_args()

    if parser_args.check_progress:
        sequence_length = int(parser_args.network.split('_')[-1].split('.')[0])
        f = open("predict_jobs.txt", "r")
        lines = [x.strip() for x in f.readlines()]
        for line in lines:
            node, namestr = line.split(':')
            names = namestr.split(',')
            total = len(names)
            done = sum([os.path.exists(f'{parser_args.outdir}/{x}_predictions_{sequence_length}.csv') for x in names])
            print(f'{node}:\tDONE: {done}\tTODO: {total-done}\tTOTAL: {total}')

        f.close()
        sys.exit()

    # Locate test data.
    sequence_length = int(parser_args.network.split('_')[-1].split('.')[0])
    files = glob.glob(f'{BASE_DATA_PATH}/*testing_ims_{sequence_length}.npy')
    names = [x.split('_testing')[0].split('ZIPPERNET/')[-1] for x in files]

    # Define DES Nodes.
    nodes = ["des30", "des31", "des40", "des41", "des50", "des60", "des70", 
        "des71", "des80", "des81", "des90", "des91"]
    
    # Distribute jobs.
    jobs = {n: [] for n in nodes}
    node_idx = 0
    for name in names:
        jobs[nodes[node_idx]].append(name)
        node_idx += 1
        if node_idx == len(nodes):
            node_idx = 0

    # Write job file.
    f = open("predict_jobs.txt", "w+")
    for node, cutout_names in jobs.items():
        f.write(f'{node}:{",".join(cutout_names)}\n')
    f.close()

    # Start jobs.
    for node in nodes:
        command = (
            f'ssh rmorgan@{node}.fnal.gov ' +
            '"source /data/des81.b/data/stronglens/setup.sh && '
            'conda deactivate && conda activate zippernet && '
            'cd /data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/zippernet/ &&'
            'python predict.py '
            f'--network {parser_args.network} ' +
            f'--config_file {parser_args.config_file} ' + 
            f'--outdir {parser_args.outdir} ' + 
            f'--cutout_names {",".join(jobs[node])}" &')

        os.system(command)


