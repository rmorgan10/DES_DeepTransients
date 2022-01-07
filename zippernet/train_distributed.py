"""Train a ZipperNet in a distribtued fashion."""

import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_utils import CombinedDataset, ToCombinedTensor, make_dataloader, read_config
from network import ZipperNN

def train(
    net, training_dir, validation_dir, config_dict, outdir, sequence_length,
    shard_list, node):
    """Train the network.
    
    Iterate through the sharded datasets to train the network. The
    config_dict contains parameters for the learning alogrithm. The
    training_data and validation_data are used for monitoring.
    """

    net.train()
    losses, train_acc, validation_acc = [], [], []
    best_val_acc = 0.0

    num_epochs = config_dict['num_epochs']
    learning_rate = config_dict['learning_rate']
    d_factor = config_dict['distribution_factor']
    validation_size = config_dict['validation_size']
    if config_dict['loss'] == 'crossentropy':
        loss_function = nn.CrossEntropyLoss()
    if config_dict['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    elif config_dict['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

    # Begin training.
    for epoch in range(num_epochs):

        for shard_num_idx, shard_num in enumerate(shard_list):

            # Load shard and create dataloader.
            training_data = torch.load(f"{training_dir}/data_i{shard_num}.pt")
            validation_data = torch.load(f"{validation_dir}/data_i{shard_num}.pt")
            train_dataloader = make_dataloader(training_data, config_dict)

            for i_batch, sample_batched in enumerate(train_dataloader):

                #Clear out all existing gradients on the loss surface to reevaluate for this step
                optimizer.zero_grad()

                #Get the CNN's current prediction of the training data
                output = net(sample_batched['lightcurve'], sample_batched['image'])

                #Calculate the loss by comparing the prediction to the truth
                loss = loss_function(output, sample_batched['label']) 

                #Evaluate all gradients along the loss surface using back propagation
                loss.backward()

                #Based on the gradients, take the optimal step in the weight space
                optimizer.step()

                if i_batch % config_dict["monitor_frequency"] == 0:
                    train_output = net(training_data[0:validation_size]['lightcurve'], training_data[0:validation_size]['image'])
                    validation_output = net(validation_data[0:validation_size]['lightcurve'], validation_data[0:validation_size]['image'])

                    train_predictions = torch.max(train_output, 1)[1].data.numpy()
                    validation_predictions = torch.max(validation_output, 1)[1].data.numpy()

                    train_accuracy = np.sum(train_predictions == training_data[0:validation_size]['label'].numpy()) / len(training_data[0:validation_size]['label'].numpy())
                    validation_accuracy = np.sum(validation_predictions == validation_data[0:validation_size]['label'].numpy()) / len(validation_data[0:validation_size]['label'].numpy())

                    print("Epoch: {0} Shard: {5}  Batch: {1} \t| Training Accuracy: {2:.3f} -- Validation Accuracy: {3:.3f} -- Loss: {4:.3f}".format(epoch + 1, i_batch + 1, train_accuracy, validation_accuracy, loss.data.numpy(), shard_num))
                    sys.stdout.flush()

                    losses.append(loss.data.numpy())
                    train_acc.append(train_accuracy)
                    validation_acc.append(validation_accuracy)
                    
                    # Save best network.
                    if validation_accuracy > best_val_acc:
                        torch.save(net.state_dict(), f"{outdir}/network_{node}_{sequence_length}.pt")
                        best_val_acc = validation_accuracy

                        os.system(f'rm {outdir}/valacc_{node}*.EMPTY')
                        os.system(f'touch {outdir}/valacc_{node}_{best_val_acc:.3f}_{sequence_length}.EMPTY')

            # Save the state of the network at the end of the shard.
            torch.save(net.state_dict(), f"{outdir}/end_network_{node}_{sequence_length}.pt")

            # Delete the shard to immediately free up memory.
            del training_data, validation_data, train_dataloader

            # Signal to other networks that we're ready to combine results.
            os.system(f'touch {outdir}/{node}_{shard_num_idx}_{epoch}.READY')

            # Wait for other networks to finish.
            print("Waiting for other networks to finish training current shard.")
            sys.stdout.flush()
            while True:
                ready_files = glob.glob(f'{outdir}/*_{shard_num_idx}_{epoch}.READY')
                if len(ready_files) == d_factor:
                    break
                else:
                    time.sleep(5)

            print("Averaging parameters.")
            sys.stdout.flush()
            # Load other networks, average parameters, and set these parameters.
            net_files = glob.glob(f"{outdir}/end_network_*_{sequence_length}.pt")
            nets = [ZipperNN(config_dict) for i in range(len(net_files))]
            for single_net, net_file in zip(nets, net_files):
                single_net.load_state_dict(torch.load(net_file))
            net_params = [x.named_parameters() for x in nets]

            with torch.no_grad():
                for ((name, param), *q) in zip(net.named_parameters(), *net_params):
                    if not all(name == p[0] for p in q):
                        raise NameError(f"{name} != {', '.join([p[0] for p in q])}")
                    param_vals = np.array([p[1].data.numpy() for p in q])
                    average_param_val = np.mean(param_vals, axis=0)
                    ones = param.data / param.data
                    values = ones * average_param_val
                    param = torch.nn.parameter.Parameter(values)
            
            # Signal to other networks that we're done averaging.
            os.system(f'touch {outdir}/{node}_{shard_num_idx}_{epoch}.DONE')

            print("Waiting for other networks to finish averaging.")
            sys.stdout.flush()
            # Remove READY file and start next shard when other networks finish averaging.
            while True:
                done_files = glob.glob(f'{outdir}/*_{shard_num_idx}_{epoch}.DONE')
                if len(done_files) == d_factor:
                    break
                else:
                    time.sleep(5)
            print("Moving on to next shard.")
            sys.stdout.flush()

    setattr(net, 'losses', losses)
    setattr(net, 'train_acc', train_acc)
    setattr(net, 'validation_acc', validation_acc)

    return net


def save_performance(
    trained_net, validation_dir, config_dict, outdir, sequence_length, 
    shard_list, node):
    """Save the performance of the fully trained network."""

    # Load best network.
    net = ZipperNN(config_dict)
    net.load_state_dict(torch.load(f'{outdir}/network_{node}_{sequence_length}.pt'))
    net.eval()

    # Save network performance.
    out_data = [(a, b, c) for a, b, c in zip(trained_net.losses, trained_net.train_acc, trained_net.validation_acc)]
    df = pd.DataFrame(data=out_data, columns=["Loss", "Train Acc", "Val Acc"])
    df.to_csv(f'{outdir}/monitoring_{node}_{sequence_length}.csv', index=False)

    # Save classifications.
    for shard_num in shard_list:
        
        filename = f'{validation_dir}/data_i{shard_num}.pt'
        test_dataset = torch.load(filename)

        labels = test_dataset[:]['label'].data.numpy()
        res = net(test_dataset[:]['lightcurve'], test_dataset[:]['image']).detach().numpy()
        output = np.hstack((res, labels.reshape(len(labels), 1)))
        df = pd.DataFrame(data=output, columns=["BG", "LSNIa", "LSNCC", "Label"])
        df.to_csv(f'{validation_dir}/classifications_i{shard_num}.csv', index=False)


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
    parser.add_argument(
        "--shard_list", type=str, help="Comma-separated list of shards.")
    parser.add_argument(
        "--node", type=str, help="Name of node we're running on.")
    parser_args = parser.parse_args()

    config_dict = read_config(parser_args.config_file)
    d_factor = config_dict["distribution_factor"]
    config_dict["learning_rate"] *= d_factor  # To account for averaging.
    shard_list = [int(x) for x in parser_args.shard_list.split(',')]

    net = ZipperNN(config_dict)
    training_dir = f"data_train_{parser_args.sequence_length}"
    validation_dir = f"data_test_{parser_args.sequence_length}"

    trained_net = train(
        net, 
        training_dir, 
        validation_dir,
        config_dict, 
        parser_args.outdir, 
        parser_args.sequence_length,
        shard_list,
        parser_args.node,
    )

    save_performance(
        trained_net, 
        validation_dir, 
        config_dict, 
        parser_args.outdir, 
        parser_args.sequence_length, 
        shard_list,
        parser_args.node, 
    )