"""Make a list of all candidates in SEARCH."""

import argparse
import glob
import sys

import numpy as np
import pandas as pd

SCALE_MIN = -7.0
SCALE_MAX = 3.0
CUTOFF = 0.875

# Handle command-line arguments.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--data_path", type=str, 
    default="/data/des81.b/data/stronglens/DEEP_FIELDS/SEARCH",
    help="Path to directory with prediction output.")
parser.add_argument(
    "--outfile", type=str, default="candidates.csv",
    help="Name of output file with classifications.")
parser_args = parser.parse_args()

total_candidates = 0
candidates = []
filenames = glob.glob(f'{parser_args.data_path}/*classifications*.csv')
total_filenames = len(filenames)
for count, filename in enumerate(filenames):

    df = pd.read_csv(filename)

    probs = df['AVERAGE_DIFF'].values
    probs = np.where(probs < SCALE_MIN, SCALE_MIN, probs)
    probs = np.where(probs > SCALE_MAX, SCALE_MAX, probs)
    probs = (probs - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)


    mask = probs > CUTOFF
    num_candidates = sum(mask)
    total_candidates += num_candidates

    if num_candidates > 0:
        # Found Candidates!
        df['IDX'] = np.arange(len(df), dtype=int)
        df['SCORE'] = probs
        candidate_array = df[['SCORE', 'IDX', 'Label']].values[mask]

        cutout_name = filename.split('//')[-1].split('_classif')[0]
        sequence_length = int(filename.split('_')[-1][:-4])

        for row in candidate_array:
            candidates.append((cutout_name, sequence_length, *row))

    progress = (count + 1) / total_filenames * 100
    sys.stdout.write(f'\rProgress: {progress:.2f} %  Found {total_candidates} candidates.             ')
    sys.stdout.flush()
    
print('\nDone!')

if total_candidates > 0:
    cols = ('CUTOUT', 'SEQUENCE_LENGTH', 'SCORE', 'IDX', 'LABEL')
    cand_df = pd.DataFrame(data=candidates, columns=cols)
    cand_df.sort_values(by='SCORE', ascending=False, inplace=True)
    cand_df.to_csv(parser_args.outfile, index=False)

    

