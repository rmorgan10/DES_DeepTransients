"""Process training data for ZipperNet.

This script performs several tasks:
  1) Filter out any NaNs from the training data,
  2) Calculates the source isolation and tracks it in the metadata,
  3) Coadd images and extract lightcurves (in 10 epoch sequences),
  4) Scale images and lightcurves to between 0 and 1,
  5) Optionally mirrors and rotates images.

Inputs for ZipperNet are saved in the top-level ZIPPERNET directory.
"""

from typing import Any, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage


def isolation(
    image_source: np.ndarray, image_lens: np.ndarray, cumulative: float = 0.9
    ) -> float:
    """Compute a statistic to measure degree of source image isolation.

    <Written by Keith Bechtol>
   
    Use the simulated image of the lensed source to define an effective weight
    map. Use the weight map to compute the weighted average flux within pixels 
    from the lensed source images and from the foreground lens. Return the 
    effective fraction of the total flux within this weighted aperture that is
    attributed to the source.

    isolation = <source_flux> / (<source_flux> + <lens_flux>)
 
    A weighted aperture is obtained by converting the simulated source image 
    into a PDF, removing the set of faint pixels that contain (1 - cumulative)
    of the total flux, and normalizing the weighted aperture such that integral 
    over all pixels in equal to one.

    Parameters
    ----------
    image_source : `numpy.ndarray` [`float`]
        Image of simulated lensed source; shape (n, n)
    image_lens : `np.ndarray` [`float`]
        Image of lens and other foreground objects; shape (n, n)
    cumulative : `float`
        Fraction of lens total flux to use when defining an aperture. 
        Default = 0.9.
    
    Returns
    -------
    isolation : `float`
        Mean flux fraction attributed to the source images within the weighted
        apertue.
    """
    
    weight = image_source  / np.sum(image_source)
    weight_sorted = np.sort(weight.flatten())
    threshold = weight_sorted[np.cumsum(weight_sorted) > (1. - cumulative)][0]
    aperture = weight > threshold

    weight_aperture = weight * aperture / np.sum(weight * aperture)

    mean_lens = np.sum(weight_aperture * image_lens) 
    mean_sources = np.sum(weight_aperture * image_source)
    isolation = mean_sources / (mean_sources + mean_lens)
    return isolation

def indicize(image_array: np.ndarray, metadata: pd.DataFrame) -> dict:
    """Split image array by objects, use OBJID as key in dict for quick lookup.
    
    Args:
      image_array (np.ndarray): Images.
      metadata (pd.DataFrame): Metadata.
      
    Returns:
      lookup dictionary for image time series based on index.
      
    Raises:
      ValueError if duplicate OBJIDs are detected in the metadata.
    """
    outdata = {}
    current_objid = metadata['OBJID'].values.min()
    prev_idx = 0
    for idx, objid in enumerate(metadata['OBJID'].values):
        
        if objid in outdata:
            raise ValueError("Duplicate OBJIDs detected.")

        if objid != current_objid:
            outdata[current_objid] = image_array[prev_idx:idx]
    
            prev_idx = idx
            current_objid = objid
        
    # Get the last object
    mask = metadata['OBJID'].values == current_objid
    outdata[current_objid] = image_array[mask]
        
    return outdata

def remove_nans(images: np.ndarray) -> np.ndarray:
    """Remove NaN images from time series and re-standardize.
    
    Args:
      images (np.ndarray): A time series image set of images.
    
    Returns:
      An array with shortened first dimension and NaNs removed.
    """
    mask = (np.sum(np.isnan(images), axis=(-1, -2)) > 0)
    min_length_without_nans = np.sum(~mask, axis=0).min()
    output = np.empty((min_length_without_nans, *images.shape[1:]), dtype=float)
    output_idx = {k: 0 for k in range(images.shape[1])}
    
    for i in range(images.shape[0]):  
        for j in range(images.shape[1]):
            
            if output_idx[j] >= min_length_without_nans:
                continue
            
            if not mask[i, j]:
                output[output_idx[j], j] = images[i, j]
                output_idx[j] += 1
     
    return output


def coadd_bands(image_arr: np.ndarray) -> np.ndarray:
    """
    Average an array of images in each band
    
    Args:
        image_arr (np.array): shape (N, <num_bands>, <height>, <width>)
        
    Returns:
        coadded array with shape (<num_bands>, <height>, <width>)
    """
    return np.nanmean(image_arr, axis=0)


def scale_bands(coadded_image_arr: np.ndarray) -> np.ndarray:
    """
    Scale pixel values to 0 to 1 preserving color
    
    Args:
        coadded_image_arr (np.array): shape (<num_bands>, <height>, <width>)
        
    Returns:
        scaled array with shape (<num_bands>, <height>, <width>)
    Raises:
        ValueError if a constant image is detected
    """
    return (
        (coadded_image_arr - coadded_image_arr.min()) / 
        (coadded_image_arr - coadded_image_arr.min()).max())


def extract_lightcurves(images, aperture_rad=15):
    """
    Measure pixel values for each band
    
    Args:
        images (np.array): one time-series example with shape (m, <num_bands>, 
          <height>, <width>).
        aperture_rad (int, default=15): radius in pixels of the aperture to use.
    
    Returns:
        lightcurve array for the example.
    """
    # construct aperature mask
    yy, xx = np.meshgrid(
        range(np.shape(images)[-1]), range(np.shape(images)[-1]))
    center = int(round(np.shape(images)[-1] / 2))
    dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    aperature = (dist <= aperture_rad)
    
    # make time measurements
    images = np.where(np.isnan(images), 0.0, images)
    sum_in_aperature = np.sum(images[:,:,aperature], axis=-1)
    med_outside_aperature = np.median(images[:,:,~aperature], axis=-1)
    res = sum_in_aperature - med_outside_aperature * aperature.sum()
    
    return res


def process(
    image_arr: np.ndarray, metadata: pd.DataFrame, sequence_length: int, bkg_metadata: pd.DataFrame = None,
    planes: np.ndarray = None, cumulative: float = 0.9, band: str = 'g') -> dict:
    """
    Iterate through image_arr and process data.

    Isolations are calculated if planes is not None. Also, TESTING image arrs are
    used in place of dl simulated images, so planes are added to the image_arr if
    they are not None.
    
    Args:
        image_arr (np.ndarray): shape (N, <num_bands>, <height>, <width>).
        metadata (pd.DataFrame): length N dataframe of metadata.
        sequence_length (int): number of epochs in output sequences.
        bkg_metadata (pd.DataFrame, default=None): metadata for the backgrounds.
        planes (np.ndarray, default=None): planes from dl simulations.
        cumulative (float, default=0.9): Isolation aperture argument.
        band (str, default='g'): band to use for metadata.
        
    Returns:
        output data dict.
        - ims: processed_ims with shape (N, <num_bands>, <height>, <width>),
        - lcs: lightcurves,
        - mds: metadata
    """
    # Indicize backgrounds.
    if bkg_metadata is not None:
        indicized_ims = indicize(image_arr, bkg_metadata)

    # Iterate through data and separate by sequence length.
    outdata = {}
    current_objid = metadata[f'OBJID-{band}'].values.min()
    num_skipped = 0
    prev_idx = 0
    for idx, objid in enumerate(metadata[f'OBJID-{band}'].values):

        if objid != current_objid:
            
            # Set a flag for checking length of backgrounds after truncation.
            # Does not apply if planes is None.
            example_usable = planes is None

            # Select the object, metadata.
            example = remove_nans(image_arr[prev_idx:idx])
            example_md = metadata.loc[prev_idx:idx-1].copy().reset_index(drop=True)
            
            # Add the image backgrounds and calculate isolations if planes is given.
            if planes is not None:
                example_planes = planes[prev_idx:idx]

                # Get image backgrounds.
                bkg_idx = int(example_md[f'BACKGROUND_IDX-{band}'].values[0])
                backgrounds = remove_nans(indicized_ims[bkg_idx])

                # Truncate simulations and image backgrounds to be the same size.
                if len(backgrounds) > len(example):
                    backgrounds = backgrounds[0:len(example)]
                elif len(backgrounds) < len(example):
                    example_md = example_md.loc[0:len(backgrounds)-1].copy().reset_index(drop=True)
                    example_planes = example_planes[0:len(backgrounds)]              

                # Perform the addition of simulations to real images.
                example = backgrounds + np.sum(example_planes, axis=1)

                # Calculate isolations if source and lens arrays are given.
                source_arrs = np.sum(example_planes, axis=1)[:,2]
                lens_arrs = backgrounds[:,2]
                isolations = [isolation(source_arrs[i], lens_arrs[i], cumulative=cumulative) for i in range(len(source_arrs))]
                example_md['ISOLATION'] = isolations
                
                # Check the example usability.
                example_usable = len(example) >= sequence_length

            # Determine if the resulting addition is usable.
            if example_usable:
                
                # Determine cadence length
                cadence_length = len(example)
                if cadence_length < sequence_length:
                    #print(f"WARNING: Sequence length ({sequence_length}) must be less that cadence length ({cadence_length}). SKIPPING Example.")
                    num_skipped += 1
                    prev_idx = idx
                    current_objid = objid
                    continue

                if sequence_length not in outdata:
                    outdata[sequence_length] = {"ims": [], 'lcs': [], 'mds': []}

                # Coadd and scale the images, append each sub-sequence to output.
                i = 0
                while sequence_length + i <= cadence_length:
                    indices = list(range(i, sequence_length + i))

                    processed_ims = coadd_bands(example[indices])
                    processed_lcs = extract_lightcurves(example[indices])

                    outdata[sequence_length]["ims"].append(scale_bands(processed_ims))
                    outdata[sequence_length]["lcs"].append(scale_bands(processed_lcs))
                    outdata[sequence_length]["mds"].append(example_md.loc[indices])

                    i += 1

            # Update trackers
            prev_idx = idx
            current_objid = objid

    total_examples = len(outdata[sequence_length]["ims"])
    skip_frac = round(num_skipped / (num_skipped + total_examples) * 100, 2)
    print(f"Skipped {num_skipped} of {total_examples + num_skipped}  ({skip_frac} %).")
    
    return outdata


def mirror_and_rotate(data):
    """
    Apply a complete set of 2D mirrorings and rotations.

    Args:
        data (dict): output of process()
    Returns:
        outdata (dict): Same as data, but has mirrored and rotated copies 
          appended.
    """

    outdata = {}
    for key in data.keys():
        outdata[key] = {'ims': [], 'lcs': [], 'mds': []}
        
        # Rotate and mirror the images, duplicate the metadata and lightcurves.
        for angle in [0.0, 90.0, 180.0, 270.0]:
            rotated_ims = ndimage.rotate(
                data[key]['ims'], axes=(-1,-2), angle=angle, reshape=False)

            # Append rotated images to output.
            outdata[key]["ims"].append(rotated_ims)
            outdata[key]["lcs"].append(data[key]['lcs'])
            outdata[key]["mds"].extend(data[key]['mds'])

            # Mirror images and append to output.
            outdata[key]["ims"].append(rotated_ims[:,:,::-1,:])
            outdata[key]["lcs"].append(data[key]['lcs'])
            outdata[key]["mds"].extend(data[key]['mds'])

        # Stack results
        outdata[key]["ims"] = np.concatenate(outdata[key]["ims"])
        outdata[key]["lcs"] = np.concatenate(outdata[key]["lcs"])
            
    return outdata



def clean_training_a(data: dict) -> dict:
    """Remove dataset examples where brightest SNe obeservation is > mag 30.
    
    Args:
      data (dict): The output of process().
    
    Returns:
      Same type of object with certain examples deleted.
    """
    outdata = {}
    for key in data.keys():
        outdata[key] = {'ims': [], 'lcs': [], 'mds': []}

        for idx, md in enumerate(data[key]['mds']):
            if md['PLANE_2-OBJECT_2-magnitude-i'].values.min() < 30:
                outdata[key]['ims'].append(data[key]['ims'][idx])
                outdata[key]['lcs'].append(data[key]['lcs'][idx])
                outdata[key]['mds'].append(data[key]['mds'][idx])
       
    return outdata


if __name__ == "__main__":

    import argparse
    import glob
    import os
    import time

    BASE_PATH = "/data/des81.b/data/stronglens/DEEP_FIELDS"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mr", action="store_true", 
        help="Optionally mirror and rotate images.")
    parser.add_argument(
        "--training_a", action="store_true", 
        help="Process simulations for training, meaning calculate isolations.")
    parser.add_argument(
        "--training_b", action="store_true", 
        help="Process negative examples for training.")
    parser.add_argument(
        "--testing", action="store_true", help="Process testing data.")
    parser.add_argument(
        "--cumulative", type=float, default=0.9, 
        help="Fraction of lens flux to use when defining isolation aperture.")
    parser.add_argument(
        "--sequence_length", type=int, default=10,
        help="Length of subsequences to extract from cutouts.")
    parser.add_argument(
        "--small_test", action="store_true",
        help="Run on just 2 cutouts for each option selected.")
    
    parser_args = parser.parse_args()

    if parser_args.training_a:
        print("Processing TRAINING_A")
        cutout_paths = []
        all_cutout_paths = [x for x in glob.glob(f'{BASE_PATH}/SIMULATIONS/*') if os.path.isdir(x)]
        for cutout_path in all_cutout_paths:
            cutout_name = cutout_path.split("/")[-1]
            if (len(glob.glob(f"{BASE_PATH}/ZIPPERNET/{cutout_name}_CONFIGURATION_1_training_a_ims_*.npy")) == 0 and
                not os.path.exists(f'{BASE_PATH}/SIMULATIONS/{cutout_name}/EMPTY.SKIP')):
                cutout_paths.append(cutout_path)

        if parser_args.small_test:
            cutout_paths = cutout_paths[:2]

        total_cutouts = len(cutout_paths)
        for cutout_idx, cutout_path in enumerate(cutout_paths):
            cutout_name = cutout_path.split('/')[-1]
            print(f'{cutout_idx + 1} of {total_cutouts}:\t{cutout_name}')

            for configuration in ['CONFIGURATION_1', 'CONFIGURATION_2']:
                print(configuration)
                # Load images, planes, and metadata into memory.
                bkg_metadata = pd.read_csv(f'{BASE_PATH}/PROCESSED/TESTING/{cutout_name}/metadata.csv')
                image_arr = np.load(f'{BASE_PATH}/PROCESSED/TESTING/{cutout_name}/images.npy', allow_pickle=True)
                metadata = pd.read_csv(f'{BASE_PATH}/SIMULATIONS/{cutout_name}/{configuration}_metadata.csv')
                planes = np.load(f'{BASE_PATH}/SIMULATIONS/{cutout_name}/{configuration}_planes.npy', allow_pickle=True)

                # Process training data.
                output = process(
                    image_arr, metadata, parser_args.sequence_length, 
                    bkg_metadata, planes, parser_args.cumulative)

                if parser_args.mr:
                    output = mirror_and_rotate(output)

                # Remove systems where SNe are too faint to detect.
                output = clean_training_a(output)

                # Save processed training data.
                for key in output:
                    out_ims = np.array(output[key]["ims"])
                    out_lcs = np.array(output[key]["lcs"])
                    out_md = {idx: output[key]["mds"][idx] for idx in range(len(output[key]["mds"]))}
                    
                    np.save(f"{BASE_PATH}/ZIPPERNET/{cutout_name}_{configuration}_training_a_ims_{key}.npy", out_ims)
                    np.save(f"{BASE_PATH}/ZIPPERNET/{cutout_name}_{configuration}_training_a_lcs_{key}.npy", out_lcs)
                    np.save(f"{BASE_PATH}/ZIPPERNET/{cutout_name}_{configuration}_training_a_mds_{key}.npy", out_md, allow_pickle=True)

    time.sleep(5)
    if parser_args.training_b:
        print("Processing TRAINING_B")
        # Separate by cutout.
        cutout_names = [x.split('/')[-2] for x in glob.glob(f'{BASE_PATH}/PROCESSED/TRAINING_B/*/images.npy')]

        # Remove cutouts that are already processed.
        cutout_names = [x for x in cutout_names if not os.path.exists(f"{BASE_PATH}/ZIPPERNET/{x}_training_b_ims_{parser_args.sequence_length}.npy")]
        if parser_args.small_test:
            cutout_names = cutout_names[:2]
        total_cutouts = len(cutout_names)

        for cutout_idx, cutout_name in enumerate(cutout_names):
            print(f'{cutout_idx + 1} of {total_cutouts}:\t{cutout_name}')
            # Load images and metadata into memory.
            image_arr = np.load(f'{BASE_PATH}/PROCESSED/TRAINING_B/{cutout_name}/images.npy', allow_pickle=True)
            metadata = pd.read_csv(f'{BASE_PATH}/PROCESSED/TRAINING_B/{cutout_name}/metadata.csv')
            metadata['OBJID-g'] = metadata['OBJID'].values.astype(int)

            # Process data.
            output = process(image_arr, metadata, parser_args.sequence_length)
            if parser_args.mr:
                output = mirror_and_rotate(output)

            # Save output.
            for key in output:
                out_ims = np.array(output[key]["ims"])
                out_lcs = np.array(output[key]["lcs"])
                out_md = {idx: output[key]["mds"][idx] for idx in range(len(output[key]["mds"]))}
                
                np.save(f"{BASE_PATH}/ZIPPERNET/{cutout_name}_training_b_ims_{key}.npy", out_ims)
                np.save(f"{BASE_PATH}/ZIPPERNET/{cutout_name}_training_b_lcs_{key}.npy", out_lcs)
                np.save(f"{BASE_PATH}/ZIPPERNET/{cutout_name}_training_b_mds_{key}.npy", out_md, allow_pickle=True)

    time.sleep(5)
    if parser_args.testing:
        print("Processing TESTING")
        # Separate by cutout.
        cutout_names = [x.split('/')[-2] for x in glob.glob(f'{BASE_PATH}/PROCESSED/TESTING/*/images.npy')]

        # Remove cutouts that are already processed.
        cutout_names = [x for x in cutout_names if not os.path.exists(f"{BASE_PATH}/ZIPPERNET/{x}_testing_ims_{parser_args.sequence_length}.npy")]
        if parser_args.small_test:
            cutout_names = cutout_names[:2]
        total_cutouts = len(cutout_names)

        for cutout_idx, cutout_name in enumerate(cutout_names):
            print(f'{cutout_idx + 1} of {total_cutouts}:\t{cutout_name}')
            # Load images and metadata into memory.
            image_arr = np.load(f'{BASE_PATH}/PROCESSED/TESTING/{cutout_name}/images.npy', allow_pickle=True)
            metadata = pd.read_csv(f'{BASE_PATH}/PROCESSED/TESTING/{cutout_name}/metadata.csv')
            metadata['OBJID-g'] = metadata['OBJID'].values.astype(int)

            # Process data - no optional mirroring / rotation.
            output = process(image_arr, metadata, parser_args.sequence_length)

            # Save output.
            for key in output:
                out_ims = np.array(output[key]["ims"])
                out_lcs = np.array(output[key]["lcs"])
                out_md = {idx: output[key]["mds"][idx] for idx in range(len(output[key]["mds"]))}
                
                np.save(f"{BASE_PATH}/ZIPPERNET/{cutout_name}_testing_ims_{key}.npy", out_ims)
                np.save(f"{BASE_PATH}/ZIPPERNET/{cutout_name}_testing_lcs_{key}.npy", out_lcs)
                np.save(f"{BASE_PATH}/ZIPPERNET/{cutout_name}_testing_mds_{key}.npy", out_md, allow_pickle=True)
