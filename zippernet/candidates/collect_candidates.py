"""Grab images of all candidates, merge multiple detections, and save."""

import argparse
import sys

import numpy as np
import pandas as pd

IMS_PATH = "/data/des81.b/data/stronglens/DEEP_FIELDS/PROCESSED/TESTING"


def diffraction_filter(im_arr):
    """Check if the riz bands are dominated by spikes, then check g band."""
    val = np.sum(np.nanmax(im_arr[:,1:,:,:], axis=(-1, -2)) > 1e5) / len(im_arr)
    if val < 2.5:
        return val
    elif np.sum(np.nanmax(im_arr[:,0,:,:], axis=(-1, -2)) > 1e5) / len(im_arr) < 0.2:
        return 2.4
    return val


# Handle command-line arguments.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--candidate_dir", type=str, help="Path to all_candidates.csv directory.")
parser.add_argument(
    "--trim_spikes", action="store_true", help="Skip candidates with spikes.")
parser.add_argument(
    "--file_limit", type=int, default=200, help="Max candidates per file.")
parser_args = parser.parse_args()

candidate_df = pd.read_csv(f"{parser_args.candidate_dir}/all_candidates.csv")

candidate_data = {}
num_cutouts = len(np.unique(candidate_df['CUTOUT_NAME'].values))
cutout_counter = 0
file_suffix = 1
total_candidates = 0
for cutout_name, df in candidate_df.groupby('CUTOUT_NAME'):
    # Account for multi-year detections in the data structure.
    year = cutout_name.split('_')[1]

    # Track progress.
    cutout_counter += 1
    progress = cutout_counter / num_cutouts * 100
    sys.stdout.write(f"\rProgress: {progress:.2f} %   Found {total_candidates} candidates.       ")
    sys.stdout.flush()

    # Merge instances of multiple observations.
    for coadd_id, coadd_id_df in df.groupby('COADD_OBJECT_ID'):

        # Grab images.
        cutout_ims = np.load(f"{IMS_PATH}/{cutout_name}/images.npy")
        start = coadd_id_df['IDX_MIN'].values[0]
        end = coadd_id_df['IDX_MAX'].values[0] + 1
        ims = cutout_ims[start:end]

        # Diffraction spike filter.
        diffraction_score = diffraction_filter(ims)
        if diffraction_score > 2.5 and parser_args.trim_spikes:
            continue

        # Store data.
        if coadd_id not in candidate_data:
            candidate_data[coadd_id] = {year: {'IMAGES': ims, 'METADATA': coadd_id_df}}
            total_candidates += 1
        else:
            candidate_data[coadd_id][year] = {'IMAGES': ims, 'METADATA': coadd_id_df}
        

    if len(candidate_data) > parser_args.file_limit:
        np.save(f"{parser_args.candidate_dir}/all_candidates_{file_suffix}.npy", candidate_data, allow_pickle=True)
        file_suffix += 1
        del candidate_data
        candidate_data = {}

np.save(f"{parser_args.candidate_dir}/all_candidates_{file_suffix}.npy", candidate_data, allow_pickle=True)

print("\nDone!")
