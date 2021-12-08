"""
Select galaxies from catalog
"""

import glob

import fitsio
import numpy as np
import pandas as pd

def make_catalog():
    files = glob.glob("deep_cal*.csv")
    dfs = []
    
    for file in files:
        # Ingest
        df = pd.read_csv(file)
        
        # Filter
        df = df[(df['CM_MAG_I'].values > -9000) & # quality
                (df['CM_T'].values > 0.05) # star gal sep
               ].copy().reset_index(drop=True)
        
        # Make field, ccd, and run columns
        df['FIELD'] = [x.split('_')[0] for x in df['TILENAME'].values]
        df['CCD'] = [int(x.split('_')[1][1:]) for x in df['TILENAME'].values]
        df['RUN'] = [x.split('_')[-1] for x in df['TILENAME'].values]
        
        # store
        dfs.append(df)
        
    # Collect into one dataframe.
    df = pd.concat(dfs)
    
    # Add redshift and ellipticity info.
    rec = fitsio.read("redshifts_ellipticities.fits").byteswap().newbyteorder()
    info_df = pd.DataFrame.from_records(rec)
    df = df.merge(info_df, how='left', left_on='COADD_OBJECT_ID', right_on='ID')

    # Save output.
    df.to_csv("deep_catalog.csv", index=False)
    
    

make_catalog()
