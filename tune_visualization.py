"""
Plot SSA V-phi curves from MCE sa_ramp data files.

Usage:
    python plot_ssa_ramp.py <ssa_file1> [ssa_file2] [--cols 0 4]

Reads the raw MCE data and runfile, averages over rows, and plots
the V-phi curve for each column. If two files are given, overlays them.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import re
import os


def read_ssa_ramp(filename):
    """
    Read an MCE SA ramp data file and return the feedback vector and
    row-averaged data.

    Returns:
        fb: array of shape (n_fb,) -- SA feedback DAC values
        data_avg: array of shape (n_col, n_fb) -- row-averaged ADC response
        data_raw: array of shape (n_row, n_col, n_fb) -- raw per-row data
    """
    runfile = filename + '.run'
    with open(runfile) as f:
        text = f.read()

    nr = int(re.search(r'<RB cc num_rows_reported>\s+(\d+)', text).group(1))

    # Parse ramp parameters from runfile
    m = re.search(r'<par_step loop1 par1>\s+(-?\d+)\s+(-?\d+)\s+(\d+)', text)
    fb0, d_fb, n_fb = int(m.group(1)), int(m.group(2)), int(m.group(3))
    fb = fb0 + np.arange(n_fb) * d_fb

    # Read binary data and auto-detect nc from file size
    header_size = 43  # uint32 words
    footer_size = 1
    raw = np.fromfile(filename, dtype='<u4')
    # total_words = n_fb * (header_size + nr*nc + footer_size), solve for nc
    nc = (len(raw) // n_fb - header_size - footer_size) // nr

    frame_size = header_size + nr * nc + footer_size
    n_frames = len(raw) // frame_size
    raw = raw[:n_frames * frame_size].reshape(n_frames, frame_size)
    payload = raw[:, header_size:header_size + nr * nc].reshape(n_frames, nr, nc)

    # Sign-extend from 25 bits (data_mode 0)
    data = payload.astype('int32')
    data[data >= 2**24] -= 2**25

    # data shape: (n_fb, n_row, n_col) -> transpose to (n_row, n_col, n_fb)
    data = data.transpose(1, 2, 0)

    # Average over rows
    data_med = np.median(data, axis=0)  # (n_col, n_fb)

    return fb, data_med, data


def load_sa_fb(tune_dir):
    """
    Load the tuned sa_fb values from experiment.cfg in the given tune directory.

    Args:
        tune_dir: path to the tuning directory containing experiment.cfg

    Returns:
        dict of {col_index: sa_fb_value} for columns with non-default values
    """
    cfg_path = os.path.join(tune_dir, 'experiment.cfg')
    with open(cfg_path) as f:
        text = f.read()

    m = re.search(r'sa_fb\s*=\s*\[([^\]]+)\]', text)
    if m is None:
        raise ValueError('sa_fb not found in %s' % cfg_path)

    values = [int(x) for x in m.group(1).split(',')]

    # Return as dict, keyed by column index
    # The array has 32 entries (all RCs), but only the first nc are relevant
    sa_fb = {}
    for i, v in enumerate(values):
        sa_fb[i] = v
    return sa_fb


def read_sq1_rampc(filename):
    """
    Read an MCE SQ1 rampc data file and return the feedback vector and data.

    Returns:
        fb: array of shape (n_fb,) -- SQ1 feedback DAC values
        data: array of shape (n_row, n_col, n_fb) -- SA feedback per row/col
    """
    runfile = filename + '.run'
    with open(runfile) as f:
        text = f.read()

    nr = int(re.search(r'<RB cc num_rows_reported>\s+(\d+)', text).group(1))

    # There can be multiple readout cards (rc1, rc2, ...), each reporting
    # its own block of columns. <RB cc num_cols_reported> / <RB rcN
    # num_cols_reported> only give the per-card column count -- the total
    # number of columns in the frame is the sum over every RC block that is
    # actually present in the .run file (in ascending card order, which is
    # also the order the columns appear in the data payload).
    rc_ids = sorted(set(re.findall(r'<RB (rc\w+) num_cols_reported>', text)))
    if not rc_ids:
        # single/legacy RC naming
        rc_ids = sorted(set(re.findall(r'<RB (rca) num_cols_reported>', text)))

    nc_per_card = []
    for rc in rc_ids:
        m = re.search(rf'<RB {rc} num_cols_reported>\s+(\d+)', text)
        nc_per_card.append(int(m.group(1)))
    nc = sum(nc_per_card)

    # Parse ramp parameters
    m = re.search(r'<par_step loop1 par1>\s+(-?\d+)\s+(-?\d+)\s+(\d+)', text)
    fb0, d_fb, n_fb = int(m.group(1)), int(m.group(2)), int(m.group(3))
    fb = fb0 + np.arange(n_fb) * d_fb

    # Read binary data
    header_size = 43
    footer_size = 1
    frame_size = header_size + nr * nc + footer_size

    raw = np.fromfile(filename, dtype='<u4')
    n_frames = len(raw) // frame_size
    raw = raw[:n_frames * frame_size].reshape(n_frames, frame_size)
    payload = raw[:, header_size:header_size + nr * nc].reshape(n_frames, nr, nc)

    # servo_mode=1, data_mode=0: data is signed 32-bit SA feedback
    data = payload.astype('int32')

    # data shape: (n_fb, n_row, n_col) -> transpose to (n_row, n_col, n_fb)
    data = data.transpose(1, 2, 0)

    return fb, data
