"""
Dataset and Dataloader objects for DeepTransients
"""

from collections import OrderedDict
import glob
import os


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
    images, lightcurves, metadata = {0:[], 1:[], 2:[]}, {0:[], 1:[], 2:[]}, {0:[], 1:[], 2:[]}

    # Collect all data sources and attach labels.
    data_sources = []
    training_b_files = glob.glob(
        f'{BASE_DATA_PATH}/*_training_b_ims_{sequence_length}.npy')
    training_b_names = [x.split('/ZIPPERNET/')[-1].split('_ims')[0] for x in training_b_files]
    for name in training_b_names:
        data_sources.append((name, 0))

    training_a_files_conf_1 = glob.glob(f'{BASE_DATA_PATH}/*_CONFIGURATION_1_training_a_ims_{sequence_length}.npy')
    training_a_names_conf_1 = [x.split('/ZIPPERNET/')[-1].split('_ims')[0] for x in training_a_files_conf_1]
    for name in training_a_names_conf_1:
        data_sources.append((name, 1))

    training_a_files_conf_2 = glob.glob(f'{BASE_DATA_PATH}/*_CONFIGURATION_2_training_a_ims_{sequence_length}.npy')
    training_a_names_conf_2 = [x.split('/ZIPPERNET/')[-1].split('_ims')[0] for x in training_a_files_conf_2]
    for name in training_a_names_conf_2:
        data_sources.append((name, 2))


    # Load data sources into memory and extand labels.
    for data_source, label in data_sources:
        # Images.
        im_file = f'{BASE_DATA_PATH}/{data_source}_ims_{sequence_length}.npy'
        im = np.load(im_file, allow_pickle=True)
        images[label].append(im.reshape((len(im), 4, 45, 45)))

        # Lightcurves.
        lc_file = f'{BASE_DATA_PATH}/{data_source}_lcs_{sequence_length}.npy'
        lc = np.load(lc_file, allow_pickle=True)
        lightcurves[label].append(lc)

        # Metadata.
        md_file = f'{BASE_DATA_PATH}/{data_source}_mds_{sequence_length}.npy'
        md = np.load(md_file, allow_pickle=True).item()
        for i in range(len(im)):
            metadata[label].append(md[i])

    # Truncate to have roughly equal class representation.
    images[0] = np.concatenate(images[0])
    lightcurves[0] = np.concatenate(lightcurves[0])
    images[1] = np.concatenate(images[1])
    lightcurves[1] = np.concatenate(lightcurves[1])
    images[2] = np.concatenate(images[2])
    lightcurves[2] = np.concatenate(lightcurves[2])

    if len(images[0]) > 2 * len(images[1]):
        # Downsample negatives.
        indices = np.arange(len(images[0]), dtype=int)
        chosen_indices = np.random.choice(
            indices, size=2*len(images[1]), replace=False)
        images[0] = images[0][chosen_indices]
        lightcurves[0] = lightcurves[0][chosen_indices]
        out_md = []
        for i in chosen_indices:
            out_md.append(metadata[0][i])
        metadata[0] = out_md
    else:
        for config in (1, 2):
            indices = np.arange(len(images[config]), dtype=int)
            chosen_indices = np.random.choice(
                indices, size=len(images[0]) // 2, replace=False)
            images[config] = images[config][chosen_indices]
            lightcurves[config] = lightcurves[config][chosen_indices]
            out_md = []
            for i in chosen_indices:
                out_md.append(metadata[config][i])
            metadata[config] = out_md

    y = np.array([*[0]*len(images[0]), *[2]*len(images[2]), *[2]*len(images[2])], dtype=int)
    X_im = np.concatenate([images[0], images[1], images[2]])
    X_lc = np.concatenate([lightcurves[0], lightcurves[1], lightcurves[2]])
    metadata = [*metadata[0], *metadata[1], *metadata[2]]

    # Shuffle and split data
    train_lightcurves, test_lightcurves, train_labels, test_labels = train_test_split(
        X_lc, y, test_size=0.1, random_state=6, stratify=y)
    
    train_images, test_images, garb1, garb2 = train_test_split(
        X_im, y, test_size=0.1, random_state=6, stratify=y)

    # Split and save metadata  
    train_md, test_md, garb1, garb2 = train_test_split(
        metadata, y, test_size=0.1, random_state=6, stratify=y)
    
    train_md = {idx: train_md[idx] for idx in range(len(train_md))}
    test_md = {idx: test_md[idx] for idx in range(len(test_md))}
    
    np.save(f"{outdir}/train_md_{sequence_length}.npy", train_md, allow_pickle=True)
    np.save(f"{outdir}/validation_md_{sequence_length}.npy", test_md, allow_pickle=True)
    
    # Create PyTorch Datasets and a Training Dataloader
    train_dataset = CombinedDataset(
        train_images, train_lightcurves, train_labels, 
        transform=ToCombinedTensor())
    test_dataset = CombinedDataset(
        test_images, test_lightcurves, test_labels, 
        transform=ToCombinedTensor())
    train_dataloader = DataLoader(
        train_dataset, batch_size=config_dict['batch_size'],
        shuffle=config_dict['shuffle'], num_workers=config_dict['num_workers'])

    return train_dataset, test_dataset, train_dataloader


def save_setup(
    training_data, validation_data, config_file, outdir, sequence_length):
    """Save training data and archive config file."""

    # Archive config file.
    os.system(f"cp {config_file} {outdir}")

    # Save training and validation data.
    torch.save(training_data, f"{outdir}/training_dataset_{sequence_length}.pt")
    torch.save(validation_data, f"{outdir}/validation_dataset_{sequence_length}.pt")