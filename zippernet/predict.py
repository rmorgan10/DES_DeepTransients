"""Evaluate TESTING data with trained network."""

import numpy as np
import pandas as pd
import torch

from data_utils import BASE_DATA_PATH, CombinedDataset, ToCombinedTensor, read_config
from network import ZipperNN

if __name__ == '__main__':
    import argparse
    # Handle command-line arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network", type=str, help="Name file containing saved network.")
    parser.add_argument(
        "--config_file", type=str, 
        help="Name of config file used in training.")
    parser.add_argument(
        "--outdir", type=str, help="Directtory to save output.")
    parser.add_argument(
        '--cutout_names', type=str, help="Comma-delimited list of cutouts.")
    parser_args = parser.parse_args()

    # Locate test data.
    sequence_length = int(parser_args.network.split('_')[-1].split('.')[0])
    names = parser_args.cutout_names.split(',')

    # Load network.
    net = ZipperNN(read_config(parser_args.config_file))
    net.load_state_dict(torch.load(parser_args.network))
    net.eval()

    # Make predictions.
    for name in names:
        im_file = f'{BASE_DATA_PATH}/{name}_testing_ims_{sequence_length}.npy'
        lc_file = f'{BASE_DATA_PATH}/{name}_testing_lcs_{sequence_length}.npy'
        ims = np.load(im_file, allow_pickle=True)
        lcs = np.load(lc_file, allow_pickle=True)

        fake_labels = np.array([3] * len(ims))
        data = CombinedDataset(
            ims, lcs, fake_labels, transform=ToCombinedTensor())

        res = net(data[:]['lightcurve'], data[:]['image']).detach().numpy()
        output = np.hstack((res, fake_labels.reshape(len(fake_labels), 1)))
        df = pd.DataFrame(data=output, columns=["BG", "LSN", "Label"])
        df = df[["BG", "LSN"]].copy().reset_index(drop=True)
        df.to_csv(f'{parser_args.outdir}/{name}_predictions_{sequence_length}.csv', index=False)
