"""Grab images of all candidates, merge multiple detections, and save."""

import argparse
import sys

import numpy as np
import pandas as pd

IMS_PATH = "/data/des81.b/data/stronglens/DEEP_FIELDS/PROCESSED/TESTING"

# Handle command-line arguments.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--candidate_dir", type=str, help="Path to all_candidates.csv directory.")
parser_args = parser.parse_args()

candidate_df = pd.read_csv(f"{parser_args.candidate_dir}/all_candidates.csv")

candidate_data = {}
num_cutouts = len(np.unique(candidate_df['CUTOUT_NAME'].values))
cutout_counter = 0
for cutout_name, df in candidate_df.groupby('CUTOUT_NAME'):

    # Track progress.
    cutout_counter += 1
    progress = cutout_counter / num_cutouts * 100
    sys.stdout.write(f"\rProgress: {progress:.2f} %        ")
    sys.stdout.flush()

    # Merge instances of multiple observations.
    for coadd_id, coadd_id_df in df.groupby('COADD_OBJECT_ID'):

        # Grab images.
        cutout_ims = np.load(f"{IMS_PATH}/{cutout_name}/images.npy")
        start = coadd_id_df['IDX_MIN'].values[0]
        end = coadd_id_df['IDX_MAX'].values[0] + 1
        ims = cutout_ims[start:end]

        # Store data.
        candidate_data[coadd_id] = {'IMAGES': ims, 'METADATA': coadd_id_df}

np.save(f"{parser_args.candidate_dir}/all_candidates.npy", candidate_data, allow_pickle=True)

print("\nDone!")
