"""Merge training data from different cutouts.

The trainging data for source injection is separated by cutout. This script
builds up a single set of training data from mutliple cutouts.
"""

import glob
import os
from typing import List

from astropy.io import fits
import numpy as np
import pandas as pd

BASE_PATH = "/data/des81.b/data/stronglens/DEEP_FIELDS/PROCESSED"


def get_fits_data(filename: str) -> np.ndarray:
    """Open a file and return the stored array.
    
    Args:
      filename (str): Name of file to open.
    
    Returns:
      The array contained in the fits file.
    """
    hdul = fits.open(filename)
    data = hdul[0].data
    hdul.close()
    return data

def merge_fits(filenames: List[str], output_filename: str):
    """Merge fits files into one file, preserving order.
    
    Args:
      filenames (list): List of files to merge.
      output_filename (str): Filename to store merged data.
    """
    # Load contents into memory.
    output_arrays = []
    for filename in filenames:
        output_arrays.append(get_fits_data(filename))

    # Concatenate arrays.
    output_array = np.concatenate(output_arrays)

    # Save to disk.
    hdu = fits.PrimaryHDU(output_array)
    hdu.writeto(output_filename, clobber=True)


def merge_maps(filenames: List[str], output_filename: str):
    """Merge map.txt files into one file, preserving order.

    Args:
      filenames (list): List of files to merge.
      output_filename (str): Filename to store merged data.
    """
    # Load contents into memory.
    output_dfs = []
    for filename in filenames:
        output_dfs.append(pd.read_csv(filename, delim_whitespace=True))

    # Concatenate.
    output_df = pd.concat(output_dfs)

    # Save.
    output_df.to_csv(output_filename, index=False, sep='\t')

    
if __name__ == "__main__":

    cutout_dirs = [x for x in glob.glob(f'{BASE_PATH}/TRAINING_A/*') if os.path.isdir(x)]

    # Merge fits files.
    for band in "griz":
        filenames = [f"{cutout_dir}/{band}.fits" for cutout_dir in cutout_dirs]
        merge_fits(filenames, f"{BASE_PATH}/TRAINING_A/{band}.fits")

    # Merge map.txt files.
    filenames = [f"{cutout_dir}/map.txt" for cutout_dir in cutout_dirs]
    merge_maps(filenames, f"{BASE_PATH}/TRAINING_A/map.txt")
