import contextlib
import glob
import os

import joblib

import mne
import numpy as np
import pandas as pd
import scipy.io as sp_io
from joblib import Parallel, delayed
from mne.preprocessing import ICA
from tqdm import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    """This is a random helper function"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def create_dataset(src_dir: str, out_dir = 'feeltrace', num_workers=2) -> None:
    """
    :param src_dir: the directory containing all the EEG and Feeltrace data in folders for each subject
    :param out_dir: output directory to write to
    :param num_workers: number of parallel processes to run

    Creates the normalized and cropped dataset in the EEG_FT_DATA directory, throws an error if
    EEG_FT_DATA does not exist
    """
    subject_data_dir = glob.glob(os.path.join(src_dir, 'p*'))

    subject_data = [glob.glob(os.path.join(x, '*')) for x in subject_data_dir]

    all_joystick_data = [ next(filter(lambda item: 'joystick.mat' in item and 'joystick_joystick.mat' not in item, x)) for x in subject_data] # final all joystick.mat

    # the next steps takes a bit of time!!
    ft_data = all_joystick_data

    with tqdm_joblib(tqdm(desc="Dataset Creation", total=len(ft_data))) as progress_bar:
        Parallel(n_jobs=num_workers)(delayed(write_to_csv_dataset_loop)(i, x, out_dir) for i, x in enumerate(ft_data))
    print(f'Created dataset initial dataset csv in {out_dir}')


def write_to_csv_dataset_loop(index: int, x: str, out_dir) -> None:
    """
    Should not be called by the user, for pair at index, create the pandas dataframe and write to a csv file
    :param x: Feeltrace filename
    :param index: index of the pair to write
    """

    ft_column_headers = ['t', 'stress']
    ft = sp_io.loadmat(x)['var']
    normalized_ft = filter_normalize_crop(ft)

    ft_df = pd.DataFrame(data=normalized_ft, columns=ft_column_headers)
    ft_df.to_csv(os.path.join(out_dir, f'feeltrace_{index}.csv'), index=False)


def filter_normalize_crop(ft: np.array) -> np.array:
    """
    Feeltrace -> crop and normalize between [0,1]

    :param eeg:
    :param ft:
    :return:
    """

    # normalize to be between [0,1]
    # min/max determined from data
    min_ft = 0
    max_ft = 225

    ft[:, 1] = (ft[:, 1] - min_ft) / (max_ft - min_ft)
    ft[:,0] /= 1000.0 # convert time to seconds

    return ft


#### TODO: comment the functions below

def load_dataset(dir = 'feeltrace', subject_num = 5, interpolate=False, resample_period='33ms'):
    # choose the subject
    subject_data_files = glob.glob(os.path.join(dir, '*.csv'))
    # sort the files by the index given to them
    file_name_2_index = lambda file : int(file.split('.csv')[0].split('_')[-1])
    subject_data_files.sort() # sort alphabetically
    subject_data_files.sort(key=file_name_2_index) # sort by index
    eeg_ft = subject_data_files[subject_num-1]

    print(f"Chosen subject: {eeg_ft}")
    
    data_signal = pd.read_csv(eeg_ft) # read the Nx2 data for a single subject

    if interpolate:
        data_signal = interpolate_df(data_signal, resample_period=resample_period)

    # return signal
    return data_signal

def interpolate_df(df, timestamps='t', resample_period='33ms'):
    '''
    resample and fill in nan values using zero hold
    '''
    df[timestamps] = pd.to_datetime(df[timestamps], unit='s')
    df_time_indexed = df.set_index(timestamps)
    df_padded = df_time_indexed.resample(resample_period, origin='start') # resample
    df_padded = df_padded.ffill() # fill nan values with the previous valid value
    df_padded.reset_index(inplace=True)
    df_padded[timestamps] = df_padded[timestamps].astype('int64') / 1e9 # nano second to seconds
    return df_padded

def generate_label(feeltrace, split_size=100, k=5, label_type='angle', num_classes=3):
    
    # split into windows (non-overlapping)
    dataset = [feeltrace[x : x + split_size] for x in range(0, len(feeltrace), split_size)]
    if len(dataset[-1]) < split_size:
        dataset.pop() # remove last window if it is smaller than the rest

    if label_type != 'both':
        labels = get_label(dataset, n_labels=num_classes, label_type=label_type).squeeze() # (N, 1)
    else:
        labels = get_combined_label(dataset, n_labels=int(np.sqrt(num_classes))).squeeze() # (N, 1)

    print(f"label set shape (N,):  {labels.shape}")

    indices = split_dataset(labels, k=k) # split data into train/test indices using kFold validation
    return labels, indices


def get_label(data, n_labels=3, label_type='angle'):
    if label_type == 'angle':
        labels = stress_2_angle(np.vstack([x[:,1].T for x in data])) # angle/slope mapped to [0,1] in a time window
    elif label_type == 'pos':
        labels = np.vstack([x[:,1].mean() for x in data]) # mean value within the time window
    else:
        labels = stress_2_accumulator(np.vstack([x[:,1].T for x in data])) # accumulator mapped to [0,1] in a time window
        
    label_dist = stress_2_label(labels, n_labels=n_labels)
    return label_dist

def get_combined_label(data, n_labels=3):
    angle_labels = get_label(data, n_labels=n_labels, label_type='angle').squeeze() # (N, 1)
    pos_labels = get_label(data, n_labels=n_labels, label_type='pos').squeeze() # (N, 1)

    labels = [x for x in range(n_labels)]
    labels_dict =  {(a, b) : n_labels*a+b for a in labels for b in labels} # cartesian product
    combined_labels = [labels_dict[(pos, angle)] for (pos, angle) in zip(pos_labels, angle_labels)]
    return np.array(combined_labels)


def stress_2_label(mean_stress, n_labels=5):
    # value is in [0,1] so map to [0,labels-1] and discretize
    return np.digitize(mean_stress * n_labels, np.arange(n_labels)) - 1

def stress_2_angle(stress_windows):
    '''
    do a linear least squares fit in the time window
    stress_window: (N_samples, time_window)
    '''
    xvals = np.arange(stress_windows.shape[-1])/1e3/60 # time in (minutes)
    slope = np.polyfit(xvals, stress_windows.T, 1)[0] # take slope linear term # 1/s
    angle = np.arctan(slope)/ (np.pi/2) * 0.5 + 0.5 # map to [0,1]
    return angle

def stress_2_accumulator(stress_windows):
    '''
    apply an integral to the time window
    stress_window: (N_samples, time_window)
    '''
    max_area = stress_windows.shape[-1]
    xvals = np.arange(stress_windows.shape[-1]) # time in (ms)
    integral = np.trapz(stress_windows, x=xvals)
    return integral/max_area # map to [0,1]

def split_dataset(labels, k=5):
    '''
    split the features and labels into k groups for k fold validation
    we use StratifiedKFold to ensure that the class distributions within each sample is the same as the global distribution
    '''
    kf = StratifiedKFold(n_splits=k, shuffle=True)

    # only labels are required for generating the split indices so we ignore it
    temp_features = np.zeros_like(labels)
    indices = [(train_index, test_index) for train_index, test_index in kf.split(temp_features, labels)]
    return indices
