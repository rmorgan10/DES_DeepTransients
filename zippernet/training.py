"""Functionality to train a network."""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_utils import CombinedDataset, ToCombinedTensor, make_dataloader
from network import ZipperNN


def train(
    net, training_dir, validation_dir, config_dict, outdir, sequence_length):
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
    validation_size = config_dict['validation_size']
    if config_dict['loss'] == 'crossentropy':
        loss_function = nn.CrossEntropyLoss()
    if config_dict['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Begin training.
    for epoch in range(num_epochs):

        sys.stdout.write("\rEpoch {0}\r".format(epoch + 1))
        sys.stdout.flush()

        for shard_idx in range(config_dict['num_shards']):

            # Load shard and create dataloader.
            training_data = torch.load(f"{training_dir}/data_i{shard_idx + 1}.pt")
            validation_data = torch.load(f"{validation_dir}/data_i{shard_idx + 1}.pt")
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

                    print("Epoch: {0} Shard: {5}  Batch: {1} \t| Training Accuracy: {2:.3f} -- Validation Accuracy: {3:.3f} -- Loss: {4:.3f}".format(epoch + 1, i_batch + 1, train_accuracy, validation_accuracy, loss.data.numpy(), shard_idx + 1))

                    losses.append(loss.data.numpy())
                    train_acc.append(train_accuracy)
                    validation_acc.append(validation_accuracy)
                    
                    # Save best network.
                    if validation_accuracy > best_val_acc:
                        torch.save(net.state_dict(), f"{outdir}/network_{sequence_length}.pt")
                        best_val_acc = validation_accuracy

                        os.system(f'rm {outdir}/valacc_*.EMPTY')
                        os.system(f'touch {outdir}/valacc_{best_val_acc:.3f}_{sequence_length}.EMPTY')

            # Delete the shard to immediately free up memory.
            del training_data, validation_data, train_dataloader

    setattr(net, 'losses', losses)
    setattr(net, 'train_acc', train_acc)
    setattr(net, 'validation_acc', validation_acc)

    return net


def save_performance(
    trained_net, validation_dir, config_dict, outdir, sequence_length):
    """Save the performance of the fully trained network."""

    # Load best network.
    net = ZipperNN(config_dict)
    net.load_state_dict(torch.load(f'{outdir}/network_{sequence_length}.pt'))
    net.eval()

    # Save network performance.
    out_data = [(a, b, c) for a, b, c in zip(trained_net.losses, trained_net.train_acc, trained_net.validation_acc)]
    df = pd.DataFrame(data=out_data, columns=["Loss", "Train Acc", "Val Acc"])
    df.to_csv(f'{outdir}/monitoring_{sequence_length}.csv', index=False)

    # Save classifications.
    for filename in glob.glob(f'{validation_dir}/data_i*.pt'):

        shard_num = int(filename.split('data_i')[-1][:-3])
        test_dataset = torch.load(filename)

        labels = test_dataset[:]['label'].data.numpy()
        res = net(test_dataset[:]['lightcurve'], test_dataset[:]['image']).detach().numpy()
        output = np.hstack((res, labels.reshape(len(labels), 1)))
        df = pd.DataFrame(data=output, columns=["BG", "LSNIa", "LSNCC", "Label"])
        df.to_csv(f'{validation_dir}/classifications_i{shard_num}.csv', index=False)