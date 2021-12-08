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
        "--config_file", dtype=str, help="Name of config file.")
    parser.add_argument(
        "--sequence_length", dtype=int, help="Number of epochs in sequence.")
    parser.add_argument(
        "--outdir", dtype=int, help="Directtory to save output.")
    parser_args = parser.parse_args()
    config_dict = data_utils.read_config(parser_args.config_file)

    if not os.path.exists(parser_args.outdir):
        os.mkdir(parser_args.outdir)

    # Load training data into memory.
    training_data, validation_data, train_dataloader = data_utils.load_training_data(
        parser_args.sequence_length, parser_args.outdir, config_dict)

    # Save training data and config.
    data_utils.save_setup(
        training_data, validation_data, parser_args.config_file, 
        parser_args.outdir, parser_args.sequence_length)

    # Instantiate network.
    net = network.ZipperNN(config_dict)

    # Train ZipperNet.
    net = training.train(
        net, training_data, validation_data, train_dataloader, config_dict)