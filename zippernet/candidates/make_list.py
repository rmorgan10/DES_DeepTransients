"""Make a list of all candidates in SEARCH."""

import argparse
import os

import numpy as np
import pandas as pd

SCALE_MIN = -7.0
SCALE_MAX = 3.0
CUTOFF = 0.67 
DETECTION_CUTOFF = 0.7 

MD_PATH = "/data/des81.b/data/stronglens/DEEP_FIELDS/ZIPPERNET"
CAT_PATH = "/data/des81.b/data/stronglens/DEEP_FIELDS/PROCESSED/TESTING"

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
    "--filenames", type=str, 
    help="Comma-Delimited list of classificaiton files.")
parser.add_argument(
    "--node", type=str, help="Name of node where this script is running.")
parser_args = parser.parse_args()

total_candidates = 0
os.system(f"touch {parser_args.outdir}/status/{total_candidates}_{parser_args.node}.CANDS")
os.system(f"touch {parser_args.outdir}/status/{DETECTION_CUTOFF}_{parser_args.node}.DETCUTOFF")
os.system(f"touch {parser_args.outdir}/status/{CUTOFF}_{parser_args.node}.CUTOFF")
candidates = []
filenames = parser_args.filenames.split(',')

for filename in filenames:

    cutout_name = filename.split('/')[-1].split('_classifications')[0]
    md_file = f"{MD_PATH}/{cutout_name}_testing_mds_{parser_args.sequence_length}.npy"
    cat_file = f"{CAT_PATH}/{cutout_name}/catalog.csv"
    if not os.path.exists(md_file):
        continue

    try:
        md = np.load(md_file, allow_pickle=True).item()
    except OSError:
        print(f" Skipping {cutout_name}")
        continue
    df = pd.read_csv(filename)
    cat = pd.read_csv(cat_file)
    cat_cols = list(cat.columns)
    
    # Add metadata and catalog to classification df.
    lengths, objids = [], []
    for idx in range(len(df)):
        lengths.append(md[idx]['CADENCE_LENGTH'].values[0])
        objids.append(md[idx]['OBJID'].values[0])
    df['CADENCE_LENGTH'] = lengths
    df['OBJID'] = objids
    df['CUTOUT_NAME'] = cutout_name
    df['IDX'] = np.arange(len(df), dtype=int)
    df = df.merge(cat, how='left', on='OBJID')

    # Add ZipperNet score.
    probs = df['AVERAGE_DIFF'].values
    probs = np.where(probs < SCALE_MIN, SCALE_MIN, probs)
    probs = np.where(probs > SCALE_MAX, SCALE_MAX, probs)
    probs = (probs - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)
    df['SCORE'] = probs

    # Add NumDection / CadenceLength score.
    objid_data = []
    for objid, cand_df in df.groupby('OBJID'):
        objid_data.append([objid, sum(cand_df['SCORE'].values > CUTOFF)])
    objid_df = pd.DataFrame(
        data=objid_data, columns=['OBJID', 'NUM_ABOVE_CUTOFF'])
    df = df.merge(objid_df, how='left', on='OBJID')
    df['DETECTION_SCORE'] = df['NUM_ABOVE_CUTOFF'].values / df['CADENCE_LENGTH'].values

    mask = (df['SCORE'].values > CUTOFF) & (df['DETECTION_SCORE'].values > DETECTION_CUTOFF)
    num_candidates = sum(mask)
    total_candidates += num_candidates

    if num_candidates > 0:
        # Found Candidates!
        candidate_array = df[['IDX', 'CUTOUT_NAME', 'SCORE', 'DETECTION_SCORE', *cat_cols]].values[mask]
        candidates.append(candidate_array)

        os.system(f"rm {parser_args.outdir}/status/*_{parser_args.node}.CANDS")
        os.system(f"touch {parser_args.outdir}/status/{total_candidates}_{parser_args.node}.CANDS")

    os.system(f"touch {parser_args.outdir}/status/{parser_args.node}_{cutout_name}.DONE")

if total_candidates > 0:
    cols = ('IDX', 'CUTOUT_NAME', 'SCORE', 'DETECTION_SCORE', *cat_cols)
    cand_df = pd.DataFrame(data=np.concatenate(candidates), columns=cols)
    cand_df.sort_values(by=['DETECTION_SCORE', 'SCORE'], ascending=False, inplace=True)
    cand_df.to_csv(f"{parser_args.outdir}/{parser_args.node}_candidates.csv", index=False)
else:
    os.system(f"touch {parser_args.outdir}/{parser_args.node}_candidates.EMPTY")

    

