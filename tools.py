"""Shared I/O helpers used by both noise.py and tset.py.

These were previously duplicated verbatim (in some cases under different
names) in both files. Kept here as the single source of truth; noise.py and
tset.py import from here instead.
"""
import numpy as np
import pandas as pd

import os
import glob
import configparser
from pathlib import Path

from moby2.util.mce import MCEFile, MCERunfile


def grab_raw_files(directory_path='.', pattern=''):
    """
    Finds all files in a given directory that do not have an extension
    and contain a given pattern in their filename.

    Args:
        directory_path (str): Directory to scan.
        pattern (str): Substring that must be present in the filename.
                       If empty, no filtering is applied.

    Returns:
        list: Sorted list of filenames (strings).
    """
    path = Path(directory_path)

    return np.sort([
        entry.name for entry in path.iterdir()
        if (
            entry.is_file()
            and not entry.suffix
            and (pattern in entry.name if pattern else True)
        )
    ])


def load_mcefile(filename, return_col=False):
    mcefile = MCEFile(filename)

    if return_col:
        timestamp, tag, rc, col, ndatasets = os.path.basename(filename).split('_')
        col = int(col[1:])
        return mcefile, col
    return mcefile


def load_data(mcefile, col, cycle_index_to_use=None, row_index_to_use=None,
              time_spacing=None, skip=0):

    d = mcefile.Read()
    num_rows_reported = mcefile.header['num_rows_reported']
    row_len = mcefile.header['row_len']

    # Set defaults
    if cycle_index_to_use is None:
        num_cycles = len(d.data[0]) // (row_len * num_rows_reported)
        cycle_index_to_use = np.arange(num_cycles)
    if row_index_to_use is None:
        row_index_to_use = np.arange(num_rows_reported)
    if time_spacing is None:
        time_spacing = row_len

    # Pre-allocate output array
    n_cycles = len(cycle_index_to_use)
    n_rows = len(row_index_to_use)
    data = np.empty((n_cycles, n_rows, time_spacing), dtype=d.data[0].dtype)

    # Vectorized extraction
    raw_data = d.data[0]
    for i_idx, i in enumerate(cycle_index_to_use):
        cycle_start = i * num_rows_reported * row_len
        for j_idx, j in enumerate(row_index_to_use):
            min_t = cycle_start + time_spacing * j + skip
            max_t = cycle_start + time_spacing * (j + 1) + skip
            data[i_idx, j_idx] = raw_data[min_t:max_t]

    return data


# Aliases matching the names noise.py originally used for the same functions,
# kept so existing callers of either name keep working.
load_mcefile_from_raw = load_mcefile
load_raw_data = load_data



def load_config_file(file_path='tune_cfg/slac_cd39.cfg'):
    '''
    returns config file containing SSA, SQ1, and other parameters
    '''
    cfg = configparser.ConfigParser()
    print('Reading: ' + file_path)
    cfg.read(file_path)
    return cfg 


def load_ssa_tune_data(dir_path, suffix='_ssa', run_ext = '.run'):
    '''
    Retrieves the relevant parameters from the SSA tuning data
    '''
    sa_tune = glob.glob(f'{dir_path}/*{suffix}')[0] 
    sa_data = os.path.join(sa_tune)
    print('Reading: ' + str(sa_data))
    mcefile = MCEFile(sa_data)
    sa_data = mcefile.Read(field='error', row_col=True)
    print('Reading: ' + sa_tune+run_ext)
    sa_runfile = MCERunfile(sa_tune+run_ext)

    return sa_data, sa_runfile 


def load_bias_run_data(dir_path, bias_suffix='_sq1servo_sa.bias', run_suffix = '_sq1servo_sa.run'):
    '''
    input: path/to/mce_folder
            grabs the first file that is found with the given suffixes
    output: the requested .run and .bias data
    '''
    bias_file = glob.glob(f'{dir_path}/*{bias_suffix}')[0]
    bias_path = os.path.join(bias_file)
    print('Reading: ' + bias_path)
    try:
        bias_df = pd.read_csv(bias_path, sep=r"\s+",
                    on_bad_lines='warn', index_col=False)
    except TypeError:
        print('Using old version of pandas:')
        bias_df = pd.read_csv(bias_path, sep=r"\s+",
                    error_bad_lines=False, index_col=False)
        
    run_file = glob.glob(f'{dir_path}/*{run_suffix}')[0]

    print('Reading: ' + run_file)
    mce_runfile = MCERunfile(run_file)
    return bias_df, mce_runfile  


def load_csservo_data(dir_path, bias_suffix='_csservo.bias', run_suffix='_csservo.run'):
    '''
    input: path/to/mce_folder
            grabs the first file that is found with the given suffixes
    output: retrieves csservo data
    '''
    bias_suffix = '_csservo_sa.bias'
    run_suffix = '_csservo_sa.run'
    return load_bias_run_data(dir_path, bias_suffix, run_suffix)  


def load_rsservo_data(dir_path, bias_suffix='_rsservo.bias', run_suffix='_rsservo.run'):
    '''
    input: path/to/mce_folder
            grabs the first file that is found with the given suffixes
    output: retrieves rsservo data
    '''
    bias_suffix = '_rsservo_sa.bias'
    run_suffix = '_rsservo_sa.run'
    return load_bias_run_data(dir_path, bias_suffix, run_suffix)  
    

def load_sq1_tune_data(dir_path):
    '''
    input: path/to/mce_folder
            grabs the first file that is found with the given suffixes
    output: retrieves sq1 tuning data
    '''
    bias_suffix = '_sq1servo_sa.bias'
    run_suffix = '_sq1servo_sa.run'
    return load_bias_run_data(dir_path, bias_suffix, run_suffix)  