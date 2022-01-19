"""
Dataset and Dataloader objects for DeepTransients
"""

from collections import OrderedDict
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

BASE_DATA_PATH = "/data/des81.b/data/stronglens/DEEP_FIELDS/ZIPPERNET"


def read_config(filename: str) -> OrderedDict:
    """Read network config file and return ordered dict.
    
    <https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts>

    Args:
      filename (str): Name of file containing network configuration.
    
    Returns:
      An ordered dictionary containing the network configuration.
    """
    class OrderedLoader(yaml.Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return OrderedDict(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)

    with open(filename, 'r') as stream:
        return yaml.load(stream, OrderedLoader)


class CombinedDataset(Dataset):
    """Dataset of DeepLenstronomy Lightcurves and Images"""

    def __init__(self, images, lightcurves, labels, transform=None):
        """
        docstring
        """
        self.images = images
        self.lightcurves = lightcurves
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        lightcurve = self.lightcurves[idx]
        label = np.array(self.labels[idx])
        
        sample = {'lightcurve': lightcurve, 'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

class ToCombinedTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        lightcurve, image, label = sample['lightcurve'], sample['image'], sample['label']

        return {'lightcurve': torch.from_numpy(lightcurve).float(),
                'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(label)}

    
def load_training_data(
    sequence_length: int, outdir: str, config_dict: dict):
    """Load training data into memory."""
    images, lightcurves, metadata = {0: [], 1: []}, {0: [], 1: []}, {0: [], 1: []}

    # Collect all data sources and attach labels.
    data_sources = []
    name_counter = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    training_b_files = glob.glob(
        f'{BASE_DATA_PATH}/*_training_b_ims_{sequence_length}.npy')
    training_b_names = [x.split('/ZIPPERNET/')[-1].split('_ims')[0] for x in training_b_files]
    for name in training_b_names:
        data_sources.append((name, 0))
        name_counter['0'] += 1

    for config in ('CONFIGURATION_3', 'CONFIGURATION_4', 'CONFIGURATION_5'):
        base_name = config.split('_')[1]
        files = glob.glob(f'{BASE_DATA_PATH}/*_{config}_training_a_ims_{sequence_length}.npy')
        names = [x.split('/ZIPPERNET/')[-1].split('_ims')[0] for x in files]
        for name in names:
            data_sources.append((name, 0))
            name_counter[base_name] += 1

    for config in ('CONFIGURATION_1', 'CONFIGURATION_2'):
        base_name = config.split('_')[1]
        files = glob.glob(f'{BASE_DATA_PATH}/*_{config}_training_a_ims_{sequence_length}.npy')
        names = [x.split('/ZIPPERNET/')[-1].split('_ims')[0] for x in files]
        for name in names:
            data_sources.append((name, 1))
            name_counter[base_name] += 1

    print(f"Loading Data  --  0: {name_counter['0']}  1: {name_counter['1']}   2: {name_counter['2']}  3: {name_counter['3']}  4: {name_counter['4']}  5: {name_counter['5']}")

    # Load data sources into memory and extand labels.
    total = len(data_sources)
    for idx, (data_source, label) in enumerate(data_sources):
        sys.stdout.write(f"\rProgress:  {idx + 1} of {total}")
        sys.stdout.flush()

        # Images.
        im_file = f'{BASE_DATA_PATH}/{data_source}_ims_{sequence_length}.npy'
        im = np.load(im_file, allow_pickle=True)
        
        # Lightcurves.
        lc_file = f'{BASE_DATA_PATH}/{data_source}_lcs_{sequence_length}.npy'
        lc = np.load(lc_file, allow_pickle=True)
        
        # Metadata.
        md_file = f'{BASE_DATA_PATH}/{data_source}_mds_{sequence_length}.npy'
        md = np.load(md_file, allow_pickle=True).item()
        
        # Check lengths.
        if len(set(len(x) for x in [im, lc, md])) != 1:
            raise ValueError(f"{data_source} has inconsistent data lengths: (im, lc, md) = ({len(im)}, {len(lc)}, {len(md)})")

        # Check NaNs.
        im_mask = (np.sum(np.isnan(im), axis=(-1, -2, -3)) > 0)
        lc_mask = (np.sum(np.isnan(lc), axis=(-1, -2, -3)) > 0)
        mask = im_mask | lc_mask
        if sum(mask) == len(im):
            print(f" {data_source} is ALL NaNs.")
            continue
        elif sum(mask) > 0:
            im = im[~mask]
            lc = lc[~mask]
            md_ = [md[i] for i in range(len(mask)) if not mask[i]]
            md = md_
            print(f" {data_source} {round(sum(mask) / len(mask) * 100, 2)} % examples contain NaNs, dropping...")

        # Store data.
        images[label].append(im.reshape((len(im), 4, 45, 45)))
        lightcurves[label].append(lc)
        for i in range(len(im)):
            metadata[label].append(md[i])
        
    images[0] = np.concatenate(images[0])
    lightcurves[0] = np.concatenate(lightcurves[0])
    images[1] = np.concatenate(images[1])
    lightcurves[1] = np.concatenate(lightcurves[1])

    print("\nDone. Making training and validation datasets.")

    # Make sure SNe, SL, and BKG are equally represented in the negative class. 
    neg_configurations = {0: [], 3: [], 4: []}
    for md_idx, md in enumerate(metadata[0]):
        if 'CONFIGURATION_LABEL-g' in md.columns:
            # Group 4 and 5 together.
            configuration = min(int(md['CONFIGURATION_LABEL-g'].values[0][-1]), 4)
        else:
            configuration = 0

        neg_configurations[configuration].append(md_idx)

    min_neg_config_size = min([len(x) for x in neg_configurations.values()])
    all_neg_indices = []
    for k in neg_configurations:
        neg_chosen_indices = list(np.random.choice(
            neg_configurations[k], size=min_neg_config_size, replace=False))
        all_neg_indices += neg_chosen_indices

    all_neg_indices = np.array(all_neg_indices, dtype=int)
    images[0] = images[0][all_neg_indices]
    lightcurves[0] = lightcurves[0][all_neg_indices]
    neg_md = []
    for neg_idx in all_neg_indices:
        neg_md.append(metadata[0][neg_idx])
    metadata[0] = neg_md[:]
    
    # Truncate to have roughly equal class representation.
    for i in (0, 1):
        if np.sum(np.isnan(images[i])) > 0:
            print(f"NaNs detected in Class {i} images before downsample")
        if np.sum(np.isnan(lightcurves[i])) > 0:
            print(f"NaNs detected in Class {i} lightcurves before downsample")

    if len(images[0]) > len(images[1]):
        small_sample, large_sample = 1, 0
    else:
        small_sample, large_sample = 0, 1

    # Downsample.
    indices = np.arange(len(images[large_sample]), dtype=int)
    chosen_indices = np.random.choice(
        indices, size=len(images[small_sample]), replace=False)
    images[large_sample] = images[large_sample][chosen_indices]
    lightcurves[large_sample] = lightcurves[large_sample][chosen_indices]
    out_md = []
    for i in chosen_indices:
        out_md.append(metadata[large_sample][i])
    metadata[large_sample] = out_md

    print(f"Done. Total training set size  --  0: {len(images[0])}, 1: {len(images[1])}")

    for i in (0, 1):
        if np.sum(np.isnan(images[i])) > 0:
            print(f"NaNs detected in Class {i} images after downsample")
        if np.sum(np.isnan(lightcurves[i])) > 0:
            print(f"NaNs detected in Class {i} lightcurves after downsample")

    y = np.array([*[0]*len(images[0]), *[1]*len(images[1])], dtype=int)
    X_im = np.concatenate([images[0], images[1]])
    X_lc = np.concatenate([lightcurves[0], lightcurves[1]])
    all_metadata = [*metadata[0], *metadata[1]]

    # Check for NaNs and drop examples.
    if np.sum(np.isnan(X_im)) > 0 or np.sum(np.isnan(X_lc)) > 0:
        im_mask = (np.sum(np.isnan(X_im), axis=(-1, -2, -3)) > 0)
        lc_mask = (np.sum(np.isnan(X_lc), axis=(-1, -2, -3)) > 0)
        mask = im_mask | lc_mask
        print(f"{round(sum(mask) / len(mask) * 100, 2)} % examples contain NaNs, dropping...")
        X_im = X_im[~mask]
        X_lc = X_lc[~mask]
        y = y[~mask]
        metadata = [all_metadata[i] for i in range(len(mask)) if not mask[i]]
        values, counts = np.unique(y, return_counts=True)
        print(f"Done. Training set size  --  0: {counts[0]}, 1: {counts[1]}")

    # Shuffle and split data
    train_lightcurves, test_lightcurves, train_labels, test_labels = train_test_split(
        X_lc, y, test_size=config_dict["test_size"], random_state=6, stratify=y)
    
    train_images, test_images, garb1, garb2 = train_test_split(
        X_im, y, test_size=config_dict["test_size"], random_state=6, stratify=y)
 
    train_md, test_md, garb1, garb2 = train_test_split(
        all_metadata, y, test_size=config_dict["test_size"], random_state=6, stratify=y)
    
    # Create and save datasets to disk by sharding.
    shard(
        train_images, train_lightcurves, train_md, train_labels, config_dict, 
        "train", outdir, sequence_length)
    shard(
        test_images, test_lightcurves, test_md, test_labels, config_dict, 
        "test", outdir, sequence_length)


def shard(
    images, lightcurves, metadata, labels, config_dict, prefix, outdir, 
    sequence_length):
    """Split the examples and save to disk."""

    data_path = f"data_{prefix}_{sequence_length}" 
    if not os.path.exists(data_path):
        os.system(f"mkdir {data_path}")
    
    num_shards = config_dict["num_shards"]
    examples_per_shard = len(images) // num_shards
    start = 0
    counter = 1
    done_flag = False
    while not done_flag:

        sys.stdout.write(f"\rSharding {prefix} -- {counter} of {num_shards + 1}")
        sys.stdout.flush()

        stop = start + examples_per_shard
        ims = images[start : stop]
        lcs = lightcurves[start: stop]
        mds = metadata[start : stop]
        lbs = labels[start : stop]

        # Create and save sharded dataset to disk.
        dataset = CombinedDataset(ims, lcs, lbs, transform=ToCombinedTensor())
        torch.save(dataset, f"{data_path}/data_i{counter}.pt")

        md = {idx: mds[idx] for idx in range(len(mds))}
        np.save(f"{data_path}/md_i{counter}.npy", md, allow_pickle=True)

        # Increment counter and stop condition.
        start += examples_per_shard
        counter += 1
        done_flag = stop > len(images)

    print("\nDone. Starting training.")
    

def make_dataloader(dataset, config_dict):
    return DataLoader(
        dataset, batch_size=config_dict['batch_size'],
        shuffle=config_dict['shuffle'], num_workers=config_dict['num_workers'])
