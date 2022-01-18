"""Make a list of all candidates in SEARCH."""

import glob
import sys

import numpy as np
import pandas as pd

BASE_DATA_PATH = '/data/des81.b/data/stronglens/DEEP_FIELDS/SEARCH'
SCALE_MIN = -35.516510367393494
SCALE_MAX = 3.8560520893428483
CUTOFF = 0.97978541835632 #0.8991497828484605

total_candidates = 0
candidates = []
filenames = glob.glob(f'{BASE_DATA_PATH}/*classifications*.csv')
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
        candidate_array = df[['SCORE', 'IDX']].values[mask]

        cutout_name = filename.split('/SEARCH/')[1].split('_classif')[0]
        sequence_length = int(filename.split('_')[-1][:-4])

        for row in candidate_array:
            candidates.append((cutout_name, sequence_length, *row))

    progress = (count + 1) / total_filenames * 100
    sys.stdout.write(f'\rProgress: {progress:.2f} %  Found {total_candidates} candidates.             ')
    sys.stdout.flush()
    
print('\nDone!')

if total_candidates > 0:
    cols = ('CUTOUT', 'SEQUENCE_LENGTH', 'SCORE', 'IDX')
    cand_df = pd.DataFrame(data=candidates, columns=cols)
    cand_df.sort_values(by='SCORE', ascending=False, inplace=True)
    cand_df.to_csv('candidates.csv', index=False)

    

