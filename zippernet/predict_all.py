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
        "--network_dir", type=str, 
        help="Directory containing training output.")
    parser.add_argument(
        "--config_file", type=str, default="main_config.yaml",
        help="Name of config file used in training.")
    parser.add_argument(
        "--outdir", type=str, help="Directtory to save output.")
    parser.add_argument(
        '--sequence_length', type=int, help="Length of processed time series.",
        default=10)
    parser.add_argument(
        '--validate', action='store_true', help="Predict on validation data.")
    parser.add_argument(
        '--validate_on_train', action='store_true', 
        help="Predict on training data.")
    parser.add_argument(
        "--check_progress", action="store_true", 
        help="Display progress and exit.")
    parser_args = parser.parse_args()
    sequence_length = parser_args.sequence_length

    network_files = ','.join(
        glob.glob(f'{parser_args.network_dir}/network_*.pt'))
    
    if parser_args.check_progress:
        f = open("predict_jobs.txt", "r")
        lines = [x.strip() for x in f.readlines()]
        for line in lines:
            node, namestr = line.split(':')
            names = namestr.split(',')
            total = len(names)
            done = sum([os.path.exists(f'{parser_args.outdir}/{x}_classifications_{sequence_length}.csv') for x in names])
            done_str = ' -- DONE!' if done == total else ''
            print(f'{node}:\tDONE: {done}\tTODO: {total-done}\tTOTAL: {total} {done_str}')

        f.close()
        sys.exit()

    # Make outdir if necessary.
    if not os.path.exists(parser_args.outdir):
        os.system(f'mkdir {parser_args.outdir}')

    # Locate data.
    if parser_args.validate:
        # Optionally predict on validation data.
        path = f"/data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/zippernet/data_test_{sequence_length}"
        all_names = [x.split('/')[-1][:-3] for x in glob.glob(f'{path}/data_i*.pt')]
        validate_str = '--validate '
    elif parser_args.validate_on_train:
        path = f"/data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/zippernet/data_train_{sequence_length}"
        all_names = [x.split('/')[-1][:-3] for x in glob.glob(f'{path}/data_i*.pt')]
        validate_str = '--validate '
    
    else:
        # Default to testing data.
        path = f'{BASE_DATA_PATH}'
        files = glob.glob(f'{BASE_DATA_PATH}/*testing_ims_{sequence_length}.npy')
        all_names = [x.split('_testing')[0].split('ZIPPERNET/')[-1] for x in files]
        validate_str = ''

    # Skip data that has already finished.
    names = []
    for name in all_names:
        if not os.path.exists(f'{parser_args.outdir}/{name}_classifications_{sequence_length}.csv'):
            names.append(name)

    # Define DES Nodes.
    nodes = ["des30", "des31", "des40", "des41", "des50", "des60", "des70", 
        "des71", "des80", "des81", "des90", "des91"][::-1]
    
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
            'cd /data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/zippernet/ && '
            'python predict.py '
            f'--networks {network_files} ' +
            f'--config_file {parser_args.config_file} ' + 
            f'--data_path {path} ' +
            f'--outdir {parser_args.outdir} ' + 
            f'--sequence_length {parser_args.sequence_length} ' + 
            validate_str +
            f'--cutout_names {",".join(jobs[node])}" &')

        os.system(command)


