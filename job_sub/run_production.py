"""
A script to trigger production for a given season and field
"""

import argparse
import glob
import os
import sys

import pandas as pd


# Get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--test",
                    action="store_true",
                    help="check ssh connection to des nodes")
parser.add_argument("--field", 
                    required=True,
                    type=str,
                    help="Name of DES Deep Field")
parser.add_argument("--season",
                    required=True,
                    type=str,
                    help="Like SV, Y1, Y2, etc.")
parser.add_argument("--maglim",
                    type=float,
                    help="Faintest i-band magniitude to include",
                    default=90.0)
args = parser.parse_args()

# Validate Args
fields = ['C1', 'C2', 'C3', 'E1', 'E2', 'S1', 'S2', 'X1', 'X2', 'X3']
if args.field not in fields:
    raise ValueError("--field argument must be in [" + ', '.join(fields) + ']') 

seasons = ['SV', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5']
if args.season not in seasons:
    raise ValueError("--season argument must be in [" + ','.join(seasons) + ']')

# Read metadata
md = pd.read_csv(f"/data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/metadata/{args.field}_metadata.csv")

# Trim to season
md = md[md['SEASON'].values == args.season].copy().reset_index(drop=True)

# Get list of CCDs
md['CCD'] = md['FILENAME'].str.extract("_c(.*?)_").values.astype(int)
ccds = list(set(md['CCD'].values))

# Establish nodes
des_nodes = ["des30", "des31", "des40", "des41", "des50",
             "des60", "des70", "des71", "des80", "des81", "des90", "des91"]

# Make status directory
status_dir = f"{args.field}_{args.season}_status/"
if not os.path.exists(status_dir):
    os.mkdir(status_dir)

# Distribute jobs
jobs = {n: [] for n in des_nodes}
idx = 0
while ccds:
    jobs[des_nodes[idx]].append(str(ccds.pop()))
    idx += 1
    if idx == len(des_nodes):
        idx = 0

# Make job list
outlines = []
outdir = f"/data/des81.b/data/stronglens/DEEP_FIELDS/CUTOUTS/{args.season}/{args.field}/"
for node, ccds in jobs.items():
    for ccd in ccds:
        outlines.append(f"{node},{outdir}{args.field}_{args.season}_{ccd}.npy,{args.maglim}\n")
f = open(f"{status_dir}{args.field}_{args.season}_jobs.log", "w+")
f.writelines(outlines)
f.close()


# Trigger jobs
for node, ccds in jobs.items():

    if not args.test:
        command = (f'ssh rmorgan@{node}.fnal.gov ' +
                   '"source /data/des81.b/data/stronglens/setup.sh && '
                   'cd /data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/job_sub/ &&'
                   f'python make_cutouts.py {node} {args.season} {args.field} {args.maglim} ' + ' '.join(ccds) + '" &')
    else:
        command = (f'ssh rmorgan@{node}.fnal.gov ' +
                   '"source /data/des81.b/data/stronglens/setup.sh && '
                   'cd /data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/job_sub/ &&'
                   f'python test_script.py {node} {args.season} {args.field} {args.maglim} ' + ' '.join(ccds) + '" &')

    os.system(command)
    
