"""Train a ZipperNet."""

import os

import data_utils
import network
import training


if __name__ == "__main__":
    import argparse

    # Handle command-line arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_file", type=str, help="Name of config file.")
    parser.add_argument(
        "--sequence_length", type=int, default=10,
        help="Number of epochs in sequence.")
    parser.add_argument(
        "--outdir", type=str, help="Directtory to save output.")
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
            parser_args.sequence_length, parser_args.outdir, config_dict)

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