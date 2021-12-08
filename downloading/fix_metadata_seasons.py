"""
Replace floating point seasons with corresponding strings
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

os.chdir('..')

md_files = glob.glob('metadata/*_metadata.csv')

corrections = {0: 'SV', 1: 'Y1', 2: 'Y2', 3: 'Y3', 4: 'Y4', 5: 'Y5'}

for md_file in md_files:
    df = pd.read_csv(md_file)
    season_vals = df['SEASON'].values.round().astype(int)

    corrected_vals = [corrections[x] for x in season_vals]
    df['SEASON'] = corrected_vals

    df.to_csv(md_file, index=False)
