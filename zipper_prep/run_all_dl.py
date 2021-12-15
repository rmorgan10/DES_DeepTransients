"""Distribute deeplenstronomy simulation jobs across the DES grid."""

import argparse
import glob
import os
import sys

BASE_PATH = "/data/des81.b/data/stronglens/DEEP_FIELDS/PROCESSED/TRAINING_A"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--check_progress", action="store_true",
    help="Display the progress and exit.")
parser_args = parser.parse_args()

cutout_dirs = [x for x in glob.glob(f'{BASE_PATH}/*') if os.path.isdir(x)]
des_nodes = ["des30", "des31", "des40", "des41", "des50",
             "des60", "des70", "des71", "des80", "des81", "des90", "des91"]


if parser_args.check_progress:
    f = open("dl_jobs.log", "r")
    lines = [x.strip() for x in f.readlines()]
    f.close()

    progress = {node: {'DONE': 0, 'TOTAL': 0} for node in des_nodes}
    for line in lines:
        node, cutout = line.split(':')
        if (os.path.exists(f"/data/des81.b/data/stronglens/DEEP_FIELDS/SIMULATIONS/{cutout}/CONFIGURATION_2_images.npy") or
            os.path.exists(f"/data/des81.b/data/stronglens/DEEP_FIELDS/SIMULATIONS/{cutout}/EMPTY.SKIP")):
            progress[node]['DONE'] += 1
        elif os.path.exists(f"{cutout}/CONFIGURATION_1_images.npy"):
            progress[node]['DONE'] += 0.5
        progress[node]['TOTAL'] += 1

    for node in progress:
        todo = progress[node]['TOTAL'] - progress[node]['DONE']
        done_str = '- DONE!' if todo == 0 else ''
        print(f"{node}:\tDONE: {progress[node]['DONE']}\tTODO: {todo}\tTOTAL: {progress[node]['TOTAL']} {done_str}")

    sys.exit()

# Distribute cutouts.
jobs = {x: [] for x in des_nodes}
node_idx = 0
for cutout_dir in cutout_dirs:
    cutout_name = cutout_dir.split('/')[-1]
    jobs[des_nodes[node_idx]].append(cutout_name)
    node_idx += 1
    if node_idx == len(des_nodes):
        node_idx = 0

# Save a jobs file.
f = open("dl_jobs.log", "w+")
for node, cutouts in jobs.items():
    for cutout in cutouts:
        f.write(f"{node}:{cutout}\n")
f.close()

# Send off jobs.
for node, cutouts in jobs.items():
    if len(cutouts) == 0:
        continue

    command = (
        f'ssh rmorgan@{node}.fnal.gov ' +
        '"source /data/des81.b/data/stronglens/setup.sh && '
        'conda deactivate && conda activate deeplens && '
        'cd /data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/zipper_prep/ && '
        'python run_dl.py '
        f'--cutout_names {",".join(cutouts)}" &')

    os.system(command)
