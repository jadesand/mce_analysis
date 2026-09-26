import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, savgol_filter
from scipy.interpolate import interp1d

import os
import sys
import glob

from moby2.util.mce import MCEFile

from .tools import grab_raw_files, load_mcefile_from_raw, load_raw_data

###########
### I/O ###
###########
def read_mce_normal_or_fast(filename, unfilter='DC'):
    """
    Read MCE data file and return data object.
    
    Parameters
    ----------
    filename : str
        Path to MCE data file (without .run extension)
    
    Returns
    -------
    data : MCEData object with .data attribute containing the signal
    """
    if not os.path.exists(filename):
        print("File not found: {}".format(filename))
        return None
    
    try:
        f = MCEFile(filename)
        if 'all_rcs' in filename:
            d = f.Read(row_col=False, unfilter=unfilter, count=None, start=0)
        else:
            d = f.Read()
        return d, f
    except Exception as e:
        print("Error reading {}: {}".format(filename, e))
        return None, None

def read_mce_no_runfile(filename, num_frames=None, data_mode=0):
    """
    Read MCE data file without a runfile.
    
    This uses SmallMCEFile with runfile=False.
    
    Parameters
    ----------
    filename : str
        Path to MCE data file
    num_frames : int, optional
        Number of frames to read (default: all)
    data_mode : int
        Data mode (0=error signal 32-bit, 1=DAC values 16-bit)
    
    Returns
    -------
    data : ndarray
        Raw data values (signed 16-bit integers) 
    fsamp : float
        Sampling frequency in Hz
    header_info : dict
        Header information
    """
    if not os.path.exists(filename):
        print("File not found: {}".format(filename))
        return None, None
    
    try:
        f = MCEFile(filename, runfile=False)
        if num_frames is not None:
            raw_data = f.ReadRaw(count=num_frames)
        else:
            raw_data = f.ReadRaw()
        
        if data_mode == 0:
            # For data_mode=0, data is 32-bit signed error signal
            data = raw_data.flatten().astype('int32')
        elif data_mode == 1:
            # For data_mode=1, data is 16-bit signed DAC values
            data = raw_data.flatten().astype('int16')
        
        return data, f
        
    except Exception as e:
        print("Error reading {}: {}".format(filename, e))
        import traceback
        traceback.print_exc()
        return None, None

def read_mce_calib(filename, col, numframes=None, data_mode=0):
    """
    Read MCE calibration data file without a runfile.

    Calibration data is taken at fast rate (~10 kHz) with multiple columns per frame.
    This function extracts just the requested column.

    Parameters
    ----------
    filename : str
        Path to MCE calibration data file
    col : int
        Column index to extract (0-7 for single RC)
    numframes : int, optional
        Number of frames to read (default: all)
    data_mode : int
        Data mode (0=error signal 32-bit)

    Returns
    -------
    data : ndarray
        Data for the requested column (signed 32-bit integers)
    f : SmallMCEFile
        File object with header info and calculated sampling frequency
    """
    if not os.path.exists(filename):
        print("File not found: {}".format(filename))
        return None, None

    try:
        # Read raw data
        data, f = read_mce_no_runfile(filename, num_frames=numframes, data_mode=data_mode)

        # Get number of columns from header status bits
        status = f.header['status']
        cc_ncols_per_rc = (status >> 16) & 0xF
        if cc_ncols_per_rc == 0:
            cc_ncols_per_rc = 8
        n_cols = cc_ncols_per_rc * f.n_rc

        # Reshape data to (n_frames, n_cols) and extract the requested column
        n_frames = len(data) // n_cols
        data_reshaped = data[:n_frames * n_cols].reshape(n_frames, n_cols)
        data_col = data_reshaped[:, col % n_cols]

        # Calculate sampling frequency
        row_len = f.header['row_len']
        num_rows = f.header['num_rows']
        data_rate = f.header['data_rate']
        cc_nrows = f.header['num_rows_reported']

        # For fast data: fsamp = 50e6/(row_len*num_rows*data_rate) * cc_nrows
        # (one row per frame, all columns)
        f.freq = (50e6 / (row_len * num_rows * data_rate)) * cc_nrows
        f.n_cols = n_cols

        return data_col, f

    except Exception as e:
        print("Error reading {}: {}".format(filename, e))
        import traceback
        traceback.print_exc()
        return None, None

def read_mce_superfast(filename, numframes=None, data_mode=0, raw=True, n_rows=1, n_cols=1):
    """
    Read MCE superfast/raw data file without a runfile.
    
    This uses SmallMCEFile with runfile=False and manually sets the parameters
    needed for raw/rectangle mode data (data_mode=0).
    
    Based on how MATLAB's read_mce_datfile.m handles files without runfiles.
    
    Parameters
    ----------
    filename : str
        Path to MCE data file
    numframes : int, optional
        Number of frames to read (default: all)
    
    Returns
    -------
    data : ndarray
        Raw data values (signed 32-bit integers) 
    fsamp : float
        Sampling frequency in Hz
    header_info : dict
        Header information
    """
    if not os.path.exists(filename):
        print("File not found: {}".format(filename))
        return None, None
    
    try:
        # Create file object without runfile - this still reads the binary header
        data, f = read_mce_no_runfile(filename, num_frames=numframes, data_mode=data_mode)
        
        # At this point, f.header contains info from the binary header
        # and f._GetPayloadInfo() has been called, setting:
        #   f.n_ro, f.size_ro, f.n_rc, f.rc_step, f.frame_bytes
        
        # For superfast/raw mode, manually set the content parameters
        # These would normally come from the runfile
        f.data_mode = data_mode  # Error signal mode (32-bit signed)
        f.raw_data = raw  # Raw mode data
        f.n_rows = n_rows  # Single row in rectangle mode
        f.n_cols = n_cols  # Single column in rectangle mode  
        f.divid = f.header['data_rate']  # data_rate from header
        
        # Calculate number of frames
        # For raw mode: n_frames = n_ro * size_ro / n_cols
        f.n_frames = f.n_ro * f.size_ro // f.n_cols
        
        # Calculate sampling frequency
        # From header: row_len, num_rows (for muxing), data_rate
        # For rectangle mode: fsamp = 50e6/(row_len*num_rows*data_rate) * (cc_nrows*cc_ncols)/(rc_nrows*rc_ncols)
        row_len = f.header['row_len']
        num_rows = f.header['num_rows']  # actual mux rows
        data_rate = f.header['data_rate']
        cc_nrows = f.header['num_rows_reported']
        
        # Get cc_ncols from status bits
        status = f.header['status']
        cc_ncols_per_rc = (status >> 16) & 0xF
        if cc_ncols_per_rc == 0:
            cc_ncols_per_rc = 8
        cc_ncols = cc_ncols_per_rc * f.n_rc
        
        # For rectangle mode with rc_nrows=1, rc_ncols=1
        f.freq = (50e6 / (row_len * num_rows * data_rate)) * (cc_nrows * cc_ncols)
        
        return data, f
        
    except Exception as e:
        print("Error reading {}: {}".format(filename, e))
        import traceback
        traceback.print_exc()
        return None, None


### Raw data
# grab_raw_files, load_mcefile_from_raw, load_raw_data moved to tools.py
# (they were byte-identical to functions in tset.py under different names).

###########
### PSD ###
###########

def compute_psd_welch(data, fs, window='boxcar', nperseg=100000, noverlap=0):
    """
    Compute power spectral density.
    
    Parameters
    ----------
    data : 1D array
        Time series data
    fs : float
        Sampling frequency in Hz
    window : str
        Window type for PSD computation (default: 'boxcar')
    
    Returns
    -------
    freq : array
        Frequency array
    psd : array
        Power spectral density
    """
    from scipy import signal
    
    freq, psd = signal.welch(data, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return freq[1:], psd[1:]


def estimate_f3dB(freqs, power, smooth_window_frac=0.02):
    """
    freqs        : frequency axis (Hz)
    power        : linear PSD from welch (not dB)
    smooth_window_frac : Savitzky-Golay window as fraction of total points
                         increase if still noisy
    """
    freqs = np.asarray(freqs)
    power = np.asarray(power)

    # --- 1. Smooth on log-frequency axis (matches your log-spaced plot) ---
    log_f = np.log10(freqs)
    n = len(power)
    win = int(smooth_window_frac * n) | 1   # must be odd
    poly = 3
    p_smooth = savgol_filter(power, window_length=win, polyorder=poly)

    # --- 2. Peak of smoothed spectrum ---
    peak_idx = np.argmax(p_smooth)
    half_max = p_smooth[peak_idx] / 2.0     # P_peak / 2  ↔  −3 dB

    # --- 3. Interpolated crossings ---
    def first_crossing(f, p, thresh, from_left=True):
        """Find where p crosses thresh, searching inward from an edge."""
        if from_left:
            idx = np.where(p >= thresh)[0]
            if len(idx) == 0: return f[0]
            i = idx[0]
            if i == 0: return f[0]
        else:
            idx = np.where(p >= thresh)[0]
            if len(idx) == 0: return f[-1]
            i = idx[-1]
            if i >= len(p) - 1: return f[-1]
            i = i + 1   # step to the falling edge

        # linear interp between (f[i-1], p[i-1]) and (f[i], p[i])
        f_cross = f[i-1] + (thresh - p[i-1]) * (f[i] - f[i-1]) / (p[i] - p[i-1])
        return f_cross

    f_low  = first_crossing(freqs,              p_smooth,              half_max, from_left=True)
    f_high = first_crossing(freqs[peak_idx:],   p_smooth[peak_idx:],   half_max, from_left=False)

    bw = f_high - f_low
    return f_low, f_high, bw, p_smooth   # return smooth for sanity-check plot


###################
### Calibration ###
###################

def get_calib_step(script_path, col):
    """
    Parse a noise_superfast.scr.rowN script and return the fb_const step size
    (high - low) for the calib block corresponding to the given absolute column.
    
    Parameters
    ----------
    script_path : str
        Path to noise_superfast.scr.rowN
    col : int
        Absolute column index (e.g. 12)
    
    Returns
    -------
    step : int
        fb_const step size in DAC units (high - low)
    """
    target = f'col{col}'
    in_block = False
    fb_vals = []

    with open(script_path) as f:
        for line in f:
            line = line.strip()

            if line.startswith('acq_config'):
                tokens = line.split()
                filename = tokens[1]
                if 'calib' in filename and target in filename:
                    in_block = True
                    fb_vals = []
                elif in_block:
                    # next acq_config closes the block
                    break
                continue

            if in_block and line.startswith('wb rca fb_const'):
                fb_vals.append(int(line.split()[3]))  # all 8 identical, take first

    if not fb_vals:
        raise ValueError(f'No fb_const found for col {col} in {script_path}')

    return (max(fb_vals) - min(fb_vals))/2
    

def compute_calibration(data_dir, row, col, bias=0, thres=3, calib_only=True, plot=False, save_dir=None):
    """
    Compute calibration values from MCE calibration data files.

    Parameters
    ----------
    data_dir : str
        Directory containing calibration data subdirectories
    row : int
        Row index
    col : int
        Column index
    thres : float
        Threshold for med/err warning
    calib_only : bool
        If True, only return calibration values without plotting

    Returns
    -------
    calib : list of tuples
        List of calibration values [(med, err, snr)] or [med] if calib_only is True
    """
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 2), constrained_layout=True, facecolor='white')
        if bias is not None:
            ax[0].set_title('Timestreams for bias {}, Row {}, Col {}'.format(bias, row, col))
        else:
            # default to zero
            ax[0].set_title('Timestreams for bias 0, Row {}, Col {}'.format(row, col))
        ax[0].set_xlabel('Sample Index')
        ax[0].set_ylabel('Signal (DAC units)')
        ax[1].set_title('Normalized Timestreams')
        ax[1].set_xlabel('Sample Index')
        ax[1].set_ylabel('Signal / Median')

    if bias is not None:
        calib_file = os.path.join(data_dir, 'bias{}/calib_row{}_col{}'.format(bias, row, col))
        script_file = os.path.join(data_dir, 'bias{}/noise_superfast.scr.row{}'.format(bias, row))
    else:
        calib_file = os.path.join(data_dir, 'calib_row{}_col{}'.format(row, col))
        script_file = os.path.join(data_dir, 'noise_superfast.scr.row{}'.format(row))
        
    sq1fb_stepsize = get_calib_step(script_file, col)

    d, f = read_mce_calib(calib_file, col=col, data_mode=0)
    d_proc = np.abs(d - np.mean(d))

    med = np.median(d_proc)
    err = np.diff(np.quantile(d_proc, [0.16, 0.84]))[0]/2

    if med/err<thres:
        print("Calibration may not be solid for r{}c{}: med/err={:.2f}".format(row, col, med/err))

    if not calib_only:
        calib = (med, err, med/err)
    else:
        calib = med

    if plot:
        d_remove_mean = d - np.mean(d)
        ax[0].plot(d_remove_mean[:500])
        ax[1].plot(d_remove_mean[:500]/med)
        
    if plot:
        if bias is not None:
            fname = f'calibration_bias{bias}_row{row}_col{col}.png'
        else:
            fname = f'calibration_bias{0}_row{row}_col{col}.png'
        # ax[1].legend(ncol=3)
        ax[1].axhline(1, color='k', ls='--', alpha=0.5)
        ax[1].set_ylim(-3, 3)
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            fig.savefig(os.path.join(save_dir, fname))

    return np.array(calib) / sq1fb_stepsize


def compute_calibration_2level(data_dir, row, col, thres=3, calib_only=True, plot=False, save_dir=None):
    """
    Compute calibration values from MCE calibration data files.

    Parameters
    ----------
    data_dir : str
        Directory containing calibration data subdirectories
    row : int
        Row index
    col : int
        Column index
    thres : float
        Threshold for med/err warning
    calib_only : bool
        If True, only return calibration values without plotting

    Returns
    -------
    calib : list of tuples
        List of calibration values [(med, err, snr)] or [med] if calib_only is True
    """
    cs = int(os.path.basename(data_dir)[2:])
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 2), constrained_layout=True, facecolor='white')
        ax[0].set_title('Timestreams for CS {}, Row {}, Col {}'.format(cs, row, col))
        ax[0].set_xlabel('Sample Index')
        ax[0].set_ylabel('Signal (DAC units)')
        ax[1].set_title('Normalized Timestreams')
        ax[1].set_xlabel('Sample Index')
        ax[1].set_ylabel('Signal / Median')

    calib_file = os.path.join(data_dir, 'calib_row{}_col{}'.format(row, col))
    script_file = os.path.join(data_dir, 'noise_superfast.scr.row{}'.format(row))
    sq1fb_stepsize = get_calib_step(script_file, col)

    d, f = read_mce_calib(calib_file, col=col, data_mode=0)
    d_proc = np.abs(d - np.mean(d))

    med = np.median(d_proc)
    err = np.diff(np.quantile(d_proc, [0.16, 0.84]))[0]/2

    if med/err<thres:
        print("Calibration may not be solid for r{}c{}: med/err={:.2f}".format(row, col, med/err))

    if not calib_only:
        calib = (med, err, med/err)
    else:
        calib = med

    if plot:
        d_remove_mean = d - np.mean(d)
        ax[0].plot(d_remove_mean[:500])
        ax[1].plot(d_remove_mean[:500]/med)
        
    if plot:
        # ax[1].legend(ncol=3)
        ax[1].axhline(1, color='k', ls='--', alpha=0.5)
        ax[1].set_ylim(-3, 3)
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fig.savefig(os.path.join(save_dir, 'calibration_cs{}_row{}_col{}.png'.format(cs, row, col)))

    return np.array(calib) / sq1fb_stepsize


############
### Plot ###
############
def plot_psd(freq, psd, title='', ax=None, label=None, ls='-', c=None, lw=1, zorder=None, yunit='DAC/rtHz'):
    """
    Plot power spectral density.
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    if c is not None:
        ax.loglog(freq[1:], np.sqrt(psd[1:]), label=label, ls=ls, zorder=zorder, lw=lw, c=c)
    else:
        ax.loglog(freq[1:], np.sqrt(psd[1:]), label=label, ls=ls, zorder=zorder, lw=lw)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(f'PSD [{yunit}]')
    ax.set_title(title)
    ax.grid(True, alpha=0.5, which='major', lw=0.5, zorder=-1)
    
    if label is not None:
        ax.legend()
    
    return ax

def plot_fit_mask(ax, freq, fit_mask, color='gray', alpha=0.3, linewidth=0):
    excluded = ~fit_mask
    # find edges of contiguous excluded regions
    diff = np.diff(excluded.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0] + 1
    # handle if starts/ends at boundary
    if excluded[0]:
        starts = np.concatenate([[0], starts])
    if excluded[-1]:
        ends = np.concatenate([ends, [len(freq)]])
    for s, e in zip(starts, ends):
        ax.axvspan(freq[s], freq[e-1], color=color, alpha=alpha, linewidth=linewidth)



###################
### 50 MHz data ###
###################

def downsample_raw(raw, row_len=62, sample_num=10, sample_dly=None):
    """
    Segment `raw` into blocks of `row_len` samples, then average a
    `sample_num`-sample sub-window of each block (offset by `sample_dly`
    samples to skip the row-switch settling transient).

    Parameters
    ----------
    raw : 1D array
        Raw 50 MHz ADC samples (frozen/open-loop, single channel).
    row_len : int
        Row period in raw samples (e.g. 62 for the superfast config,
        119 for the default config -- confirm against the config that
        was actually active during acquisition).
    sample_num : int
        Number of samples to average within each row period.
    sample_dly : int or None
        Offset (in samples) into each row period before the averaging
        window starts. Defaults to row_len - sample_num, matching the
        convention used in noise_superfast_ba_h5.sh (sampledly =
        row_len - samplenum).

    Returns
    -------
    1D array of length (len(raw) // row_len), one averaged value per
    row period. Effective sample rate is 50e6 / row_len.
    """
    raw = np.asarray(raw)

    if sample_dly is None:
        sample_dly = row_len - sample_num

    if sample_dly < 0 or sample_dly + sample_num > row_len:
        raise ValueError(
            f"sample_dly ({sample_dly}) + sample_num ({sample_num}) "
            f"must fit within row_len ({row_len})"
        )

    n_blocks = len(raw) // row_len
    trimmed = raw[: n_blocks * row_len]
    blocks = trimmed.reshape(n_blocks, row_len)

    window = blocks[:, sample_dly : sample_dly + sample_num]
    return window.mean(axis=1)



def raw2superfast(raw, sample_num=10):
    """
    Segment `raw` into blocks of `row_len` samples, then average a
    `sample_num`-sample sub-window of each block (offset by `sample_dly`
    samples to skip the row-switch settling transient).

    Parameters
    ----------
    raw : 1D array
        Raw 50 MHz ADC samples (frozen/open-loop, single channel).
    row_len : int
        Row period in raw samples (e.g. 62 for the superfast config,
        119 for the default config -- confirm against the config that
        was actually active during acquisition).
    sample_num : int
        Number of samples to average within each row period.
    sample_dly : int or None
        Offset (in samples) into each row period before the averaging
        window starts. Defaults to row_len - sample_num, matching the
        convention used in noise_superfast_ba_h5.sh (sampledly =
        row_len - samplenum).

    Returns
    -------
    1D array of length (len(raw) // row_len), one averaged value per
    row period. Effective sample rate is 50e6 / row_len.
    """
    raw = np.asarray(raw)
    
    n_blocks = len(raw) // sample_num
    trimmed = raw[: n_blocks * sample_num]
    blocks = trimmed.reshape(n_blocks, -1)
    
    return blocks.sum(axis=1)
