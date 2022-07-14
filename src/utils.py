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
    subject_data_dir.sort(key=lambda x:int(x.split("p")[-1])) # sort in non decreasing order

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
