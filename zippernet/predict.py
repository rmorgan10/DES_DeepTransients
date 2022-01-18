"""Evaluate TESTING data with trained network."""

import glob

import numpy as np
import pandas as pd
import torch

from data_utils import CombinedDataset, ToCombinedTensor, read_config
from network import ZipperNN


def get_score_df(data, nets, labels):
    """Compute diagnositic scores for all networks.
    
    Args:
      data: A PyTorch CombinedDataset object.
      nets: A dictionary of net name to net mapping."""

    output = []
    for net_file, net in nets.items():
        res = net(data[:]['lightcurve'], data[:]['image']).detach().numpy()
        res_output = np.hstack((res, labels.reshape(len(labels), 1)))
        df = pd.DataFrame(data=res_output, columns=[f"BKG_{net_file}", f"LSN_{net_file}", "Label"])
        output.append(df)

    df = pd.concat(output, axis=1)
    bkg_columns = [x for x in df.columns if x.startswith('BKG')]
    lsn_columns = [x for x in df.columns if x.startswith('LSN')]

    for net_file in nets:
        df[f"DIFF_{net_file}"] = df[f"LSN_{net_file}"].values - df[f"BKG_{net_file}"].values
    diff_columns = [x for x in df.columns if x.startswith('DIFF')]

    df['BKG_AVERAGE'] = np.mean(df[bkg_columns].values, axis=1)
    df['LSN_AVERAGE'] = np.mean(df[lsn_columns].values, axis=1)
    df['BKG_MEDIANS'] = np.median(df[bkg_columns].values, axis=1)
    df['LSN_MEDIANS'] = np.median(df[lsn_columns].values, axis=1)

    df['AVERAGE_DIFF'] = df['LSN_AVERAGE'].values - df['BKG_AVERAGE'].values
    df['MEDIANS_DIFF'] = df['LSN_MEDIANS'].values - df['BKG_MEDIANS'].values
    df['DIFF_AVERAGE'] = np.mean(df[diff_columns].values, axis=1)
    df['DIFF_MEDIANS'] = np.median(df[diff_columns].values, axis=1)

    return df


if __name__ == '__main__':
    import argparse
    # Handle command-line arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--networks", type=str, help="Comma-delimited list of network files.")
    parser.add_argument(
        "--config_file", type=str, 
        help="Name of config file used in training.")
    parser.add_argument(
        "--outdir", type=str, help="Directtory to save output.")
    parser.add_argument(
        '--cutout_names', type=str, help="Comma-delimited list of cutouts.")
    parser.add_argument(
        '--sequence_length', type=int, help="Length of processed time series.",
        default=10)
    parser.add_argument('--data_path', type=str, help="Path to data directory.")
    parser.add_argument(
        '--validate', action='store_true', help="Predict on validation data.")
    parser_args = parser.parse_args()

    # Locate test data.
    sequence_length = parser_args.sequence_length
    names = parser_args.cutout_names.split(',')

    # Load networks.
    net_files = parser_args.networks.split(',')
    nets = {}
    for net_file in net_files:
        net = ZipperNN(read_config(parser_args.config_file))
        net.load_state_dict(torch.load(net_file))
        net.eval()
        nets[net_file] = net

    # Make predictions.
    for name in names:

        if parser_args.validate:
            # Get validation data.
            data = torch.load(f'{parser_args.data_path}/{name}.pt')
            labels = data[:]['label'].data.numpy()

        else:
            # Build testing data.
            im_file = f'{parser_args.data_path}/{name}_testing_ims_{sequence_length}.npy'
            lc_file = f'{parser_args.data_path}/{name}_testing_lcs_{sequence_length}.npy'
            ims = np.load(im_file, allow_pickle=True)
            lcs = np.load(lc_file, allow_pickle=True)
            labels = np.array([-9] * len(ims))  # Fake labels.

            data = CombinedDataset(
                ims, lcs, labels, transform=ToCombinedTensor())

        # Evaluate data with networks.
        score_df = get_score_df(data, nets, labels)
        score_df.to_csv(f'{parser_args.outdir}/{name}_classifications_{sequence_length}.csv', index=False)