'''
    Filename: utils.py
    Usage: All the helper functions are put here
'''
import numpy as np
from datetime import timedelta
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler


def flat_accuracy(preds, labels, verbose=False):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    if verbose:
        print("> ", pred_flat, labels_flat, pred_flat == labels_flat)
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(timedelta(seconds=elapsed_rounded))


def split_tensor_datasets(dataset, ratio=0.7):
    train_size = int(ratio * len(dataset))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])


def get_training_dataset_loader(ds: TensorDataset, batch_size=4):
    return DataLoader(ds, sampler=RandomSampler(ds), batch_size=batch_size, num_workers=1)


def get_validate_dataset_loader(ds: TensorDataset, batch_size=4):
    return DataLoader(ds, sampler=SequentialSampler(ds), batch_size=batch_size, num_workers=1)
