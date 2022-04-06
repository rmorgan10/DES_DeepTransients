"""For a given list of COADD_OBJECT_IDs, grab all images of object."""

import argparse
import os

import numpy as np
import pandas as pd

from lc_utils import get_lc

BASE_PATH = "/data/des81.b/data/stronglens/DEEP_FIELDS"

# Handle command-line arguments.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--coadd_ids", type=str, 
    help="Comma-delimited list of COADD_OBJECT_IDS to collect.")
parser.add_argument(
    "--outfile", type=str, help="Place to save output.")
parser.add_argument(
    "--skip_images", action="store_true",
    help="Light curves only, do not collect images.")

parser_args = parser.parse_args()

# Determine coadd ids that still need to be run.
if os.path.exists(parser_args.outfile):
    out_data = np.load(parser_args.outfile, allow_pickle=True).item()
    prev_coadd_ids = set([int(x) for x in out_data])
else:
    out_data = {}
    prev_coadd_ids = set()
coadd_ids = list(set([int(x) for x in parser_args.coadd_ids.split(',')]) - prev_coadd_ids)

# Load catalogs.
main_cat = pd.read_csv(f"{BASE_PATH}/PRODUCTION/catalog/deep_catalog.csv")
idx_data = [[i, coadd_id] for i, coadd_id in enumerate(coadd_ids)]
idx_df = pd.DataFrame(data=idx_data, columns=['TEMP_IDX', 'COADD_OBJECT_ID'])
idx_df = idx_df.merge(main_cat, on='COADD_OBJECT_ID', how='inner')
if len(idx_df) != len(coadd_ids):
    print("WARNING: some of the requested COADD_OBJECT_IDs were trimmed during merge.")

missing_counter = 0
years = ['Y1', 'Y2', 'Y3', 'Y4', 'Y5']
for index, row in idx_df.iterrows():

    print(index + 1, "of ", len(coadd_ids), "\tMissing: ", missing_counter)

    out_data[row['COADD_OBJECT_ID']] = {}
    ccd = row['CCD']
    field = row['FIELD'].split('-')[-1]

    for year in years:
        cutout_name = f"{field}_{year}_{ccd}"
        cutout_path = f"{BASE_PATH}/PROCESSED/TESTING/{cutout_name}"

        try:
            cutout_md = pd.read_csv(f"{cutout_path}/metadata.csv")
        except FileNotFoundError:
            out_data[row['COADD_OBJECT_ID']][year] = {
                'IMAGES': None,
                'METADATA': None,
                'MJDS': None,
            }
            missing_counter += 1
            continue
    
        cutout_md['IMG_IDX'] = np.arange(len(cutout_md), dtype=int)
        cutout_cat = pd.read_csv(f"{cutout_path}/catalog.csv")
        obj_cutout_cat = cutout_cat[cutout_cat['COADD_OBJECT_ID'].values == row['COADD_OBJECT_ID']].copy()
        objid = obj_cutout_cat['OBJID'].values[0]
        objid_md = cutout_md[cutout_md['OBJID'].values == objid].copy().reset_index(drop=True)
        objid_md.sort_values('IMG_IDX', inplace=True)
        start = objid_md['IMG_IDX'].values.min()
        end = objid_md['IMG_IDX'].values.max() + 1

        if not parser_args.skip_images:
            im_array = np.load(f"{cutout_path}/images.npy")
            ims = im_array[start:end]
        else:
            ims = None

        mjds = objid_md[['MJD_OBS-g', 'MJD_OBS-r', 'MJD_OBS-i', 'MJD_OBS-z']].values

        out_data[row['COADD_OBJECT_ID']][year] = {
            'IMAGES': ims,
            'METADATA': objid_md,
            'MJDS': mjds,
        }

        # Save RA and DEC each time through loop for database query.
        ra = obj_cutout_cat['RA'].values[0]
        dec = obj_cutout_cat['DEC'].values[0]

    # Query DB for lightcurve.
    out_data[row['COADD_OBJECT_ID']]['LC'] = get_lc(float(ra), float(dec))

    # Save after each lc to handle failures better.
    np.save(parser_args.outfile, out_data, allow_pickle=True)
