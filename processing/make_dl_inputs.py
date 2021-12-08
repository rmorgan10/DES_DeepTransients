"""Create deeplenstronomy inputs for a single cutout file.

This script performs several tasks to create inputs for deeplenstronomy source
injection:
    (1) Make a map.txt file from the A examples
    (2) Format A examples for source injection
    (3) Format all images to look like deeplenstronomy outputs
"""

import argparse
import os
from typing import Any, List, Tuple

from astropy.io import fits
import numpy as np
import pandas as pd

CUTOUT_PATH = "/data/des81.b/data/stronglens/DEEP_FIELDS/CUTOUTS"
OUTPUT_PATH = "/data/des81.b/data/stronglens/DEEP_FIELDS/PROCESSED"


def open_cutout(filename: str) -> dict:
    """Load the contents of the cutout file into memory.

    Args:
      filename (str): Name of the file to open.

    Returns:
      Contents of the file, which are stored in a dictionary.
    """
    return np.load(filename, allow_pickle=True).item()


def _recover_pixel_values(nite_data: np.ndarray) -> np.ndarray:
    """Use the IMG_SCALE and IMG_MIN values to get originial pixel values.
    
    Args:
      nite_data (np.ndaray): Images for one nite in the cutout data.

    Returns:
      image array with recovered pixel values.
    """
    return (
        nite_data['IMG'] / 65535 * 
        nite_data['IMG_SCALE'][:,:,np.newaxis,np.newaxis] + 
        nite_data['IMG_MIN'][:,:,np.newaxis,np.newaxis]
    )


def _determine_nites(cutout_data: dict) -> dict:
    """Downselect observations to have the same number of nites in each band.

    Cadence regularity will be important for the RNN, so here we trim the
    observations in each band to produce outputs where each system has the same
    number of nites in each band. To make the downselection, we drop from periods
    of the cadence with higher sampling rates.

    Args:
      cutout_data (dict): The contents of a cutout file.

    Returns:
      A dictionary of which nites to use in each band.
    """

    # Construct output.
    output_nites = {}
    
    # Obtain MJDs and NITEs in each band.
    obs = {
        'g': {'MJDS': [], 'NITES': []}, 
        'r': {'MJDS': [], 'NITES': []}, 
        'i': {'MJDS': [], 'NITES': []}, 
        'z': {'MJDS': [], 'NITES': []},
    }
    for nite in sorted(cutout_data['NITES']):
        for k, v in cutout_data[nite]['METADATA'].items():
            obs[k]['MJDS'].append(v['MJD_OBS'])
            obs[k]['NITES'].append(nite)
       
    # Determine length to truncate sequences.
    min_length = min([len(dates['MJDS']) for dates in obs.values()])
       
    # Preferentially drop observations in periods with dense sampling.
    for band, dates in obs.items():
        mjds = dates['MJDS']
        n = len(mjds) - min_length
        if n == 0:
            output_nites[band] = np.array(dates['NITES'])
            continue
            
        # Get indices of areas with close elements.
        indices = np.argsort(np.diff(mjds))

        # Filter out neighboring elements so we don't drop large groups.
        mask = [indices[0]]  # `mask` collects indices to drop.
        i = 0
        while len(mask) < n:
            i += 1
            next_idx = indices[i]
            if any(abs(next_idx - mask_idx) <= 1 for mask_idx in mask):
                continue
            mask.append(next_idx)

        good_indices = list(set(np.arange(len(mjds))) - set(mask))
        
        output_nites[band] = np.array(dates['NITES'])[good_indices]
        
    return output_nites


def _mkdir(path: str):
    """Create a directory on the file system.
    
    Args:
      path (str): Name of directory to create.
    """
    if not os.path.exists(path):
        os.system(f"mkdir {path}")


def format_for_source_injection(
    cutout_data: dict) -> Tuple[np.ndarray, pd.DataFrame]:
    """Split by band, average nites, and get map.txt info.

    Args:
      cutout_data (dict): Contents of a cutout file.
    
    Returns:
      Image array, map dataframe tuple
    """
    # Recover original pixel values and group images.
    rescaled_ims = []
    for nite in cutout_data['NITES']:
        rescaled_ims.append(_recover_pixel_values(cutout_data[nite]))
    rescaled_ims = np.array(rescaled_ims)

    # Average over nites, ignoring missing nites filled with NaN.
    rescaled_ims = np.nanmean(rescaled_ims, axis=0)

    # Create a map.txt-like DataFrame from the catalog data.
    map_columns_in_catalog = ['Z_PEAK', 'BDF_G_0', 'BDF_G_1']
    map_columns_for_dl = [
        'CONFIGURATION_1-PLANE_1-OBJECT_1-REDSHIFT-g',
        'CONFIGURATION_1-PLANE_1-OBJECT_1-MASS_PROFILE_1-e1-g',
        'CONFIGURATION_1-PLANE_1-OBJECT_1-MASS_PROFILE_1-e2-g',
    ]
    map_data = cutout_data['CATALOG'][map_columns_in_catalog].values
    map_df = pd.DataFrame(data=map_data, columns=map_columns_for_dl)

    # Filter out objects without the redshift and ellipticity info.
    missing_info_mask = (
        np.isnan(map_df['CONFIGURATION_1-PLANE_1-OBJECT_1-REDSHIFT-g'].values) |
        np.isnan(map_df['CONFIGURATION_1-PLANE_1-OBJECT_1-MASS_PROFILE_1-e1-g'].values) |
        np.isnan(map_df['CONFIGURATION_1-PLANE_1-OBJECT_1-MASS_PROFILE_1-e2-g'].values)
    )
    rescaled_ims = rescaled_ims[~missing_info_mask]
    map_df = map_df[~missing_info_mask].copy().reset_index(drop=True)

    return rescaled_ims, map_df


def format_like_deeplenstronomy(
    cutout_data: dict, cutout_size: int = 45
    ) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Organize by stacking nites on top of each other.
    
    Args:
      cutout_data (dict): Contents of a cutout file.
      cutout_size (int): Side length in pixels of a square cutout.

    Returns:
      Image array of all nites, metadata dataframe, catalog dataframe.
    """
    nites = _determine_nites(cutout_data)
    
    # Set up data storage objects.
    num_objects = len(cutout_data[nites['g'][0]]['IMG'])
    num_nites = len(nites['g'])
    img_array = np.empty(
        (num_objects * num_nites, 4, cutout_size, cutout_size), 
        dtype=np.float32)
    metadata = []
    metadata_cols = [
        'MJD_OBS-g', 'SKYBRITE-g', 'EXPTIME-g', 'FWHM-g',
        'MJD_OBS-r', 'SKYBRITE-r', 'EXPTIME-r', 'FWHM-r',
        'MJD_OBS-i', 'SKYBRITE-i', 'EXPTIME-i', 'FWHM-i',
        'MJD_OBS-z', 'SKYBRITE-z', 'EXPTIME-z', 'FWHM-z',
        'OBJID',
    ]
    idx = 0
    
    # Organize images and metadata into dl format.
    for obj_idx in range(num_objects):
        
        for g_nite, r_nite, i_nite, z_nite in zip(
            nites['g'], nites['r'], nites['i'], nites['z']):

            # Recover original pixel values.
            g_ims = _recover_pixel_values(cutout_data[g_nite])
            r_ims = _recover_pixel_values(cutout_data[r_nite])
            i_ims = _recover_pixel_values(cutout_data[i_nite])
            z_ims = _recover_pixel_values(cutout_data[z_nite])
            
            # Organize images.
            img_array[idx,0] = g_ims[obj_idx,0]
            img_array[idx,1] = r_ims[obj_idx,1]
            img_array[idx,2] = i_ims[obj_idx,2]
            img_array[idx,3] = z_ims[obj_idx,3]
        
            # Organize metadata; 1:5 = 'MJD_OBS', 'SKYBRITE', 'EXPTIME', 'FWHM'.
            metadata.append([
                *cutout_data[g_nite]['METADATA']['g'].values[1:5],
                *cutout_data[r_nite]['METADATA']['r'].values[1:5],
                *cutout_data[i_nite]['METADATA']['i'].values[1:5],
                *cutout_data[z_nite]['METADATA']['z'].values[1:5],
                obj_idx,
            ])
            
            idx += 1
            
    # Cast metadata and catalog to dataframes
    metadata = pd.DataFrame(data=metadata, columns=metadata_cols)
    catalog = cutout_data['CATALOG'].copy().reset_index(drop=True)
    catalog['OBJID'] = np.arange(num_objects)
        
    return img_array, metadata, catalog
    

if __name__ == "__main__":

    # Handle command-line arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filenames", type=str, help="comma-separated names of cutout files.")
    parser.add_argument(
        "--for_training_a", action="store_true", 
        help="Set to additionally format images for source injection.")
    parser.add_argument(
        "--for_training_b", action="store_true", 
        help="Set to additionally use images as negatives for training.")
    args = parser.parse_args()

    for filename in args.filenames.split(','):
        # Get field and season from filename.
        field = filename.split('_')[0]
        season = filename.split('_')[1]

        # Load cutout file into memory.
        try:
            cutout_data = open_cutout(f'{CUTOUT_PATH}/{season}/{field}/{filename}')
        except OSError as err:
            print("Detected Error:\n", err, "\nContinuing...")
            os.system(f"rm {CUTOUT_PATH}/{season}/{field}/{filename}")
            continue
        cutout_name = filename.split('.')[0]

        # Make output directories
        if args.for_training_a:
            _mkdir(f"{OUTPUT_PATH}/TRAINING_A/{cutout_name}")
        if args.for_training_b:
            _mkdir(f"{OUTPUT_PATH}/TRAINING_B/{cutout_name}")
        _mkdir(f"{OUTPUT_PATH}/TESTING/{cutout_name}")

        # Optionally prepare for source injection.
        if args.for_training_a:
            rescaled_ims, map_df = format_for_source_injection(cutout_data)

            # Save to disk.
            base_dir = f"{OUTPUT_PATH}/TRAINING_A/{cutout_name}"
            for idx, band in enumerate("griz"):
                hdu = fits.PrimaryHDU(rescaled_ims[:,idx])
                hdu.writeto(f"{base_dir}/{band}.fits")

            map_df.to_csv(f"{base_dir}/map.txt", sep='\t', index=False)

        # Format all cutout data to look like deeplenstronomy outputs.
        img_array, metadata, catalog = format_like_deeplenstronomy(cutout_data)

        # Save to disk.
        base_dir = f"{OUTPUT_PATH}/TESTING/{cutout_name}"
        np.save(f"{base_dir}/images.npy", img_array, allow_pickle=True)
        metadata.to_csv(f"{base_dir}/metadata.csv", index=False)
        catalog.to_csv(f"{base_dir}/catalog.csv", index=False)

        # Create a symlink for TRAINING_B to reduce disk space usage.
        if args.for_training_b:
            training_base_dir = f"{OUTPUT_PATH}/TRAINING_B/{cutout_name}"
            os.system(f'ln -s {base_dir}/images.npy {training_base_dir}/images.npy')
            os.system(f'ln -s {base_dir}/metadata.csv {training_base_dir}/metadata.csv')
            os.system(f'ln -s {base_dir}/catalog.csv {training_base_dir}/catalog.csv')
    
        # Delete cutout file so that it doesn't get re-found.
        os.system(f"rm {CUTOUT_PATH}/{season}/{field}/{filename}")
