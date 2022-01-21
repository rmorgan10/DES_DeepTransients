"""Train a ZipperNet."""

import os
import sys

import data_utils
import network
import training


if __name__ == "__main__":
    import argparse

    # Handle command-line arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_file", type=str, default="main_config.yaml",
        help="Name of config file.")
    parser.add_argument(
        "--sequence_length", type=int, default=10,
        help="Number of epochs in sequence.")
    parser.add_argument(
        "--outdir", type=str, help="Directory to save output.")
    parser.add_argument(
        "--trim", action='store_true', help="Use 60 percent of data.")
    parser.add_argument(
        "--no_train", action='store_true', help="Exit after making dataset.")
    parser_args = parser.parse_args()
    config_dict = data_utils.read_config(parser_args.config_file)

    if not os.path.exists(parser_args.outdir):
        os.mkdir(parser_args.outdir)
    os.system(f"cp {parser_args.config_file} {parser_args.outdir}") # Archive config file.

    # Use existing training data if it exists.
    training_dir = f"data_train_{parser_args.sequence_length}"
    validation_dir = f"data_test_{parser_args.sequence_length}"

    # Load and shard training data if needed.
    if not all(os.path.exists(x) for x in (training_dir, validation_dir)):
        data_utils.load_training_data(
            parser_args.sequence_length, parser_args.outdir, config_dict, 
            parser_args.trim)

    # Optionally exit early.
    if parser_args.no_train:
        print("Exiting early because of --no_train argument.")
        sys.exit()

    print("Starting training.")

    # Single node training.
    if ('distribution_factor' not in config_dict or 
        config_dict['distribution_factor'] == 1):

        # Instantiate network.
        net = network.ZipperNN(config_dict)

        # Train ZipperNet.
        net = training.train(
            net, training_dir, validation_dir, config_dict, parser_args.outdir, 
            parser_args.sequence_length)

        # Save final performance.
        training.save_performance(
            net, validation_dir, config_dict, parser_args.outdir,
            parser_args.sequence_length)

    # Distributed training.
    else:
        d_factor = int(config_dict['distribution_factor'])
        nodes = [
            'des91', 'des90', 'des81', 'des80', 'des71', 'des70', 'des61',
            'des60', 'des51', 'des50', 'des41', 'des40', 'des31', 'des30'
        ]
        if d_factor > len(nodes):
            raise ValueError(
                f"distribution factor must be less than {len(nodes)}.")

        # Distribute shards evenly.
        shards = {nodes[i]: [] for i in range(d_factor)}
        node_idx = 0
        if 'shard_limit' in config_dict:
            shard_limit = config_dict['shard_limit']
        else:
            shard_limit = config_dict['num_shards']
        for shard_idx in range(min(config_dict['num_shards'], shard_limit)):
            shards[nodes[node_idx]].append(shard_idx + 1)
            node_idx += 1
            if node_idx == d_factor:
                node_idx = 0
                if min(config_dict['num_shards'], shard_limit) - (shard_idx + 1) < d_factor:
                    break

        # Start jobs.
        for node, shard_list in shards.items():
            cmd = (
                f'ssh rmorgan@{node}.fnal.gov ' +
                '"source /data/des81.b/data/stronglens/setup.sh && ' + 
                'conda deactivate && conda activate zippernet && ' + 
                'cd /data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/zippernet/ && ' + 
                'python train_distributed.py ' + 
                f'--config_file {parser_args.config_file} ' + 
                f'--sequence_length {parser_args.sequence_length} ' + 
                f'--outdir {parser_args.outdir} ' + 
                f'--node {node} ' + 
                f'--shard_list {",".join([str(x) for x in shard_list])} ' + 
                f'&> {parser_args.outdir}/{node}_{parser_args.sequence_length}.log" &')

            os.system(cmd)

        
            

