"""
Adapted from @bguta in https://github.com/ubcspin/emod-eegify/blob/main/src/run_classification.py 
"""

# TODO: refactor to allow for general purpose label generation and dataset splitting (so it works for both EEG and FSR)

import numpy as np
import pandas as pd


def generate_label(feeltrace, split_size=100, k=5, label_type='angle', num_classes=3, overlap=0.5):
    # split into windows (with overlap %)
    dataset = [feeltrace[x: x + split_size]
               for x in range(0, len(feeltrace), int(split_size * (1.0-overlap)))]
    # remove last windows if they are smaller than the rest
    dataset = [x for x in dataset if len(x) == split_size]

    if label_type != 'both':
        labels, raw_label = get_label(
            dataset, n_labels=num_classes, label_type=label_type)  # (N, 1)
    else:
        raise ValueError(f'Unexpected Label Type: {label_type}')

    dataset = generate_eeg_features(dataset)
    dataset = np.vstack([np.expand_dims(x, 0)
                         for x in dataset])  # (N, eeg_feature_size, 64)

    print(f"EEG feature shape (N, freq_bands, channels, channels):  {dataset.shape}")
    print(f"label set shape (N,):  {labels.shape}")

    # split data into train/test indices using kFold validation
    indices = split_dataset(labels, k=k)
    return dataset, labels, indices, raw_label


def get_label(data, n_labels=3, label_type='angle'):
    if label_type == 'angle':
        # angle/slope mapped to [0,1] in a time window
        labels = stress_2_angle(np.vstack([x[:, 1].T for x in data]))
    elif label_type == 'pos':
        # mean value within the time window
        labels = np.vstack([x[:, 1].mean() for x in data])
    else:
        # accumulator mapped to [0,1] in a time window
        labels = stress_2_accumulator(np.vstack([x[:, 1].T for x in data]))

    label_dist = stress_2_label(labels, n_labels=n_labels).squeeze()
    return label_dist, labels.squeeze()


def stress_2_label(mean_stress, n_labels=5):
    # value is in [0,1] so map to [0,labels-1] and discretize
    return np.digitize(mean_stress * n_labels, np.arange(n_labels)) - 1


def stress_2_angle(stress_windows):
    '''
    do a linear least squares fit in the time window
    stress_window: (N_samples, time_window)
    '''
    xvals = np.arange(stress_windows.shape[-1])/1e3/60  # time in (minutes)
    slope = np.polyfit(xvals, stress_windows.T, 1)[
        0]  # take slope linear term # 1/s
    angle = np.arctan(slope) / (np.pi/2) * 0.5 + 0.5  # map to [0,1]
    return angle


def stress_2_accumulator(stress_windows):
    '''
    apply an integral to the time window
    stress_window: (N_samples, time_window)
    '''
    max_area = stress_windows.shape[-1]
    xvals = np.arange(stress_windows.shape[-1])  # time in (ms)
    integral = np.trapz(stress_windows, x=xvals)
    return integral/max_area  # map to [0,1]


def split_dataset(labels, k=5):
    '''
    split the features and labels into k groups for k fold validation
    we use StratifiedKFold to ensure that the class distributions within each sample is the same as the global distribution
    '''

    kf = StratifiedKFold(n_splits=k, shuffle=True)

    # only labels are required for generating the split indices so we ignore it
    temp_features = np.zeros_like(labels)
    indices = [(train_index, test_index)
               for train_index, test_index in kf.split(temp_features, labels)]
    return indices
