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


def filter_nans(
    images: np.ndarray, metadata: pd.DataFrame, band : str = 'g') -> Tuple[Any]:
    """
    Remove examples where any image in the time series (in 
    any band) contains NaNs. Prints fraction of data removed.
    
    Args:
        images (np.array): shape (N, <num_bands>, <height>, <width>)
        metadata (pd.DataFrame): length N dataframe of metadata
        band (str, default='g'): band to use for metadata
        
    Returns:
        images where a NaN wasn't present, 
        metadata where a NaN wasn't present
    """
    # Find the OBJIDs of the time series examples with NaNs.
    mask = (np.sum(np.isnan(images), axis=(-1, -2, -3)) > 0)
    bad_objids = metadata[f'OBJID-{band}'].values[mask]
    full_mask = np.array(
        [x in bad_objids for x in metadata[f'OBJID-{band}'].values])
    
    # Determine the data loss.
    print("losing", round(sum(full_mask) / len(images) * 100, 2), "% (NaNs)")
    
    # Apply the mask and return.
    return (
        images[~full_mask], 
        metadata[~full_mask].copy().reset_index(drop=True),
    )


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
    image_arr: np.ndarray, metadata: pd.DataFrame, sequence_length: int, 
    source_arr_i: np.ndarray = None, lens_arr_i: np.ndarray = None, 
    cumulative: float = 0.9, band: str = 'g') -> dict:
    """
    Iterate through image_arr and process data.

    Isolations are calculated if source_arr_i and lens_arr_i are not None.
    
    Args:
        image_arr (np.ndarray): shape (N, <num_bands>, <height>, <width>).
        metadata (pd.DataFrame): length N dataframe of metadata.
        sequence_length (int): number of epochs in output sequences.
        source_arr_i (np.ndarray, default=None): i-band source plane shape 
          (N, <height>, <width>).
        lens_arr_i (np.ndarray, default=None): i-band lens plane shape 
          (N, <height>, <width>).
        cumulative (float, default=0.9): Isolation aperture argument.
        band (str, default='g'): band to use for metadata.
        
    Returns:
        output data dict.
        - ims: processed_ims with shape (N, <num_bands>, <height>, <width>),
        - lcs: lightcurves,
        - mds: metadata
    """
    # Clean the data.
    # clean_ims, clean_md = filter_nans(image_arr, metadata)
    clean_ims, clean_md = image_arr, metadata

    # Track the data loss due to errors.
    num_errors = 0
    
    # Iterate through data and separate by sequence length.
    outdata = {}
    current_objid = clean_md[f'OBJID-{band}'].values.min()
    prev_idx = 0
    for idx, objid in enumerate(clean_md[f'OBJID-{band}'].values):
        
        if objid != current_objid:
            
            # Select the object
            example = clean_ims[prev_idx:idx,:,:,:]
            
            # Select the metadata
            example_md = clean_md.loc[prev_idx:idx-1].copy().reset_index(drop=True)

            # Calculate isolations if source and lens arrays are given.
            if source_arr_i is not None and lens_arr_i is not None:
                source_arrs = source_arr_i[prev_idx:idx]
                lens_arrs = lens_arr_i[prev_idx:idx]
                isolations = [isolation(source_arrs[i], lens_arrs[i], cumulative=cumulative) for i in range(len(source_arrs))]
                example_md['ISOLATION'] = isolations

            # Determine cadence length
            cadence_length = len(example)
            if cadence_length < sequence_length:
                raise ValueError(
                    f"Sequence length must be less that cadence length ({cadence_length}).")
            if sequence_length not in outdata:
                outdata[sequence_length] = {"ims": [], 'lcs': [], 'mds': []}
            
            # Coadd and scale the images - skip if error raised
            try:
                # Append each sub-sequence to output.
                i = 0
                while sequence_length + i <= cadence_length:
                    indices = list(range(i, sequence_length + i))

                    processed_ims = coadd_bands(example[indices])
                    processed_lcs = extract_lightcurves(example[indices])

                    outdata[sequence_length]["ims"].append(scale_bands(processed_ims))
                    outdata[sequence_length]["lcs"].append(scale_bands(processed_lcs))
                    outdata[sequence_length]["mds"].append(example_md.loc[indices])

                    i += 1

            except FloatingPointError:
                # Skip example if constant image detected.
                num_errors += 1

            # Update trackers
            prev_idx = idx
            current_objid = objid

    # Report data loss
    #print(
    #    "Losing",  
    #    round(float(num_errors) / float(len(outdata[sequence_length]["ims"])) * 100, 2), 
    #    "% (Constants)",
    #)
    
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
    """Remove dataset examples with NaN time delays.
    
    Args:
      data (dict): The output of process().
    
    Returns:
      Same type of object with certain objects deleted.
    """
    #cols = [
    #    'PLANE_2-OBJECT_2-tdshift_3-i', ]

    #output = {}
    #for key in data:
    #    mask = np.empty(len(data[key]["ims"]), dtype=bool)
    #    for i in range(len(data["ims"])):
    #        if data[key]["mds"][i]
    #        ...
    return data


if __name__ == "__main__":

    import argparse
    import glob
    import os

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
    
    parser_args = parser.parse_args()

    if parser_args.training_a:
        print("Processing TRAINING_A")
        cutout_paths = glob.glob(f'{BASE_PATH}/SIMULATIONS/*')

        total_cutouts = len(cutout_paths)
        for cutout_idx, cutout_path in enumerate(cutout_paths):
            cutout_name = cutout_path.split('/')[-1]
            print(f'{cutout_idx + 1} of {total_cutouts}:\t{cutout_name}')

            for configuration in ['CONFIGURATION_1', 'CONFIGURATION_2']:
                print(configuration)
                # Load images, planes, and metadata into memory.
                image_arr = np.load(f'{BASE_PATH}/SIMULATIONS/{cutout_name}/{configuration}_images.npy', allow_pickle=True)
                metadata = pd.read_csv(f'{BASE_PATH}/SIMULATIONS/{cutout_name}/{configuration}_metadata.csv')
                planes = np.load(f'{BASE_PATH}/SIMULATIONS/{cutout_name}/{configuration}_planes.npy', allow_pickle=True)
                source_planes = planes[:, 1, 2]  # i-band only.
                lens_planes = image_arr[:, 2] - source_planes  # i-band only.

                # Process training data.
                output = process(
                    image_arr, metadata, parser_args.sequence_length, source_planes, 
                    lens_planes, parser_args.cumulative)

                if parser_args.mr:
                    output = mirror_and_rotate(output)

                # Remove systems where time delay was NaN (no lensing).
                output = clean_training_a(output)

                # Save processed training data.
                for key in output:
                    out_ims = np.array(output[key]["ims"])
                    out_lcs = np.array(output[key]["lcs"])
                    out_md = {idx: output[key]["mds"][idx] for idx in range(len(output[key]["mds"]))}
                    
                    np.save(f"{BASE_PATH}/ZIPPERNET/{cutout_name}_{configuration}_training_a_ims_{key}.npy", out_ims)
                    np.save(f"{BASE_PATH}/ZIPPERNET/{cutout_name}_{configuration}_training_a_lcs_{key}.npy", out_lcs)
                    np.save(f"{BASE_PATH}/ZIPPERNET/{cutout_name}_{configuration}_training_a_mds_{key}.npy", out_md, allow_pickle=True)

    if parser_args.training_b:
        print("Processing TRAINING_B")
        # Separate by cutout.
        cutout_names = [x.split('/')[-2] for x in glob.glob(f'{BASE_PATH}/PROCESSED/TRAINING_B/*/images.npy')]

        # Remove cutouts that are already processed.
        cutout_names = [x for x in cutout_names if not os.path.exists(f"{BASE_PATH}/ZIPPERNET/{x}_training_b_ims_{parser_args.sequence_length}.npy")]
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

    if parser_args.testing:
        print("Processing TESTING")
        # Separate by cutout.
        cutout_names = [x.split('/')[-2] for x in glob.glob(f'{BASE_PATH}/PROCESSED/TESTING/*/images.npy')]

        # Remove cutouts that are already processed.
        cutout_names = [x for x in cutout_names if not os.path.exists(f"{BASE_PATH}/ZIPPERNET/{x}_testing_ims_{parser_args.sequence_length}.npy")]
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