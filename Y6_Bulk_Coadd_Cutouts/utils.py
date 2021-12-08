"""Helper functions for cutout.py"""

import glob
import os

BASE_PATH = '/data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION'

def redownload(filename: str):
    """Redownloads an image from NCSA.
    
    Args:
      filename (str): Absolute filename (on FNAL) of file to redownload.
    """

    season, name = filename.split('/')[-2:]

    # Find NCSA address of file to download by searching wget lists.
    wget_files = glob.glob(f'{BASE_PATH}/wget_lists/{season}/*wgetlist.txt')
    found = False
    for wget_file in wget_files:
        f = open(wget_file, 'r')
        for line in f.readlines():
            if line.split('/')[-1].strip() == name:
                address = line.strip()
                found = True
                break

        f.close()
        if found:
            break

    # Trigger redownload.
    os.chdir(f'{BASE_PATH}/images/{season}')
    os.system(f'rm {name}')
    os.system(f'wget --no-check-certificate {address}')

