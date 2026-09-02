# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal as sig
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import re
import sys
from itertools import groupby
# sys.path.append('/home/cryo/rshi/local/mce_script/python')
# from mce_data import MCEFile

from .tools import grab_raw_files, load_mcefile, load_data


#######
# I/O #
#######

# grab_raw_files moved to tools.py (was byte-identical to noise.py's copy).


def group_filenames(filenames):
    """
    Group filenames by (rc, c) pairs, sorted by index within each group.
    Pattern: {timestamp}_raw_{rc}_{c}_{index}
    """
    def parse(name):
        m = re.search(r'_raw_(rc\d+)_(c\d+)_(\d+)$', name)
        return m.group(1), m.group(2), int(m.group(3))

    # Sort by (rc, c, index) so groupby sees contiguous (rc, c) runs
    sorted_names = sorted(filenames, key=lambda n: parse(n))

    return [
        list(group)
        for (rc, c), group in groupby(sorted_names, key=lambda n: parse(n)[:2])
    ]


# load_mcefile, load_data moved to tools.py (were byte-identical to
# noise.py's load_mcefile_from_raw / load_raw_data).


def segment_data(data, row_len=119, num_rows_reported=41, cycle_index_to_use=None, row_index_to_use=None, time_spacing=None, skip=0):
    # Set defaults
    if cycle_index_to_use is None:
        num_cycles = len(data[0]) // (row_len * num_rows_reported)
        cycle_index_to_use = np.arange(num_cycles)
    if row_index_to_use is None:
        row_index_to_use = np.arange(num_rows_reported)
    if time_spacing is None:
        time_spacing = row_len
    
    # Pre-allocate output array
    n_cols = len(data)
    n_cycles = len(cycle_index_to_use)
    n_rows = len(row_index_to_use)
    seg_data = np.empty((n_cols, n_cycles, n_rows, time_spacing), dtype=data.dtype)
    
    # Vectorized extraction
    for i_idx, i in enumerate(cycle_index_to_use):
        cycle_start = i * num_rows_reported * row_len
        for j_idx, j in enumerate(row_index_to_use):
            min_t = cycle_start + time_spacing * j + skip
            max_t = cycle_start + time_spacing * (j + 1) + skip
            seg_data[:, i_idx, j_idx] = data[:, min_t:max_t]
    
    return seg_data

def segment_data_two_level(data, row_len=119, num_rows_reported=41, cycle_index_to_use=None, row_index_to_use=None, time_spacing=None, skip=0):
    # Set defaults
    if cycle_index_to_use is None:
        num_cycles = data.shape[-1] // (row_len * num_rows_reported)
        cycle_index_to_use = np.arange(num_cycles)
    if row_index_to_use is None:
        row_index_to_use = np.arange(num_rows_reported)
    if time_spacing is None:
        time_spacing = row_len
    
    # Pre-allocate output array
    n_chips = len(data)
    n_cycles = len(cycle_index_to_use)
    n_rows = len(row_index_to_use)
    seg_data = np.empty((n_chips, n_cycles, n_rows, time_spacing), dtype=data.dtype)
    
    # Vectorized extraction
    for i_idx, i in enumerate(cycle_index_to_use):
        cycle_start = i * num_rows_reported * row_len
        for j_idx, j in enumerate(row_index_to_use):
            min_t = cycle_start + time_spacing * j + skip
            max_t = cycle_start + time_spacing * (j + 1) + skip
            seg_data[:, i_idx, j_idx] = data[:, min_t:max_t]
    
    return seg_data

def calib_data(data, adu_to_mV, sa_offset, sa_offset_dac_to_mV, in_place=True):
    if in_place:
        data[:] = data / adu_to_mV + sa_offset * sa_offset_dac_to_mV
    else:
        return data / adu_to_mV + sa_offset * sa_offset_dac_to_mV


def load_cfg_array(filepath, array_name):
    with open(filepath, 'r') as f:
        content = f.read()
    
    pattern = rf'{array_name}\s*=\s*\[([\d\s,\-]+)\];'
    match = re.search(pattern, content)
    
    if match:
        return np.array([int(x) for x in match.group(1).split(',')])
    raise ValueError(f"Could not find {array_name} in file")
    


###########
# Filters #
###########
def lowpass_data(data, order=2, Wn=0.1):
    """
    Apply lowpass filter to 3D data array.
    
    Time complexity: O(n_cycles × n_rows × n_samples)
    Space complexity: O(n_cycles × n_rows × n_samples)
    """
    sos = sig.butter(order, Wn=Wn, btype='lowpass', analog=False, output='sos')
    
    # Apply filter along the last axis (time samples) for all cycles and rows at once
    # This is much faster than nested loops
    lp_data = sig.sosfiltfilt(sos, data, axis=-1)
    
    return lp_data


    




###################
# Multi-component #
###################

def single_response(t, t0, amplitude, tau):
    """
    Generate a single exponential response
    
    y(t) = A * (1 - exp(-(t-t0)/τ)) for t >= t0
    
    Parameters:
    -----------
    t : array
        Time points
    t0 : float
        Switch time (when component turns on)
    amplitude : float
        Response amplitude (can be negative)
    tau : float
        Time constant (larger = slower response)
        
    Returns:
    --------
    response : array
        Component response at each time point
    """
    response = np.zeros_like(t, dtype=float)
    mask = t >= t0
    response[mask] = amplitude * (1 - np.exp(-(t[mask] - t0) / tau))
    return response


def multi_response(t, baseline, *params):
    """
    Generate multi-component response (sum of exponentials)
    
    Parameters:
    -----------
    t : array
        Time points
    baseline : float
        Baseline offset
    *params : float
        Triplets of (t0, amplitude, tau) for each component
        Example: t0_1, A_1, tau_1, t0_2, A_2, tau_2, ...
        
    Returns:
    --------
    signal : array
        Total signal = baseline + sum of all components
    """
    signal = np.full_like(t, baseline, dtype=float)
    
    # params should be groups of 3: (t0, amplitude, tau)
    n_components = len(params) // 3
    
    for i in range(n_components):
        t0 = params[3*i]
        amplitude = params[3*i + 1]
        tau = params[3*i + 2]
        signal += single_response(t, t0, amplitude, tau)
    
    return signal


def fit_multi_exponential(y, n_components,
                          initial_guess=None,
                          bounds=None, 
                          absolute_sigma=None,
                          sigma=None,
                          kwargs={}):
    """
    Fit multi-exponential model to data
    
    Parameters:
    -----------
    y : array
        Observed signal
    n_components : int
        Number of exponential components
    initial_guess : list, optional
        Initial parameter guess [baseline, t0_1, A_1, tau_1, t0_2, A_2, tau_2, ...]
        If None, will use automatic guess
    bounds : tuple of lists, optional
        (lower_bounds, upper_bounds) for parameters
        
    Returns:
    --------
    params : array
        Fitted parameters [baseline, t0_1, A_1, tau_1, ...]
    covariance : array
        Covariance matrix of fitted parameters
    """
    if len(np.unique(y))==1:
        return np.array([])

    t = np.arange(y.shape[-1])
    # Auto-generate initial guess if not provided
    if initial_guess is None:
        # baseline_guess = np.mean(y[:min(100, len(y)//4)])  # Use first 25% as baseline
        baseline_guess = np.mean(y[:20])  # Use first 20 samples as baseline
        
        # Find when major changes occur
        dy = np.abs(np.diff(y))
        peaks = np.argsort(dy)[-n_components:]  # n_components largest changes
        switch_times = t[peaks]
        
        initial_guess = [baseline_guess]
        for i in range(n_components):
            t0_guess = switch_times[i] if i < len(switch_times) else t[len(t)//4]
            amp_guess = (np.max(y) - np.min(y)) / n_components * ((-1)**i)  # Alternate signs
            tau_guess = (t[-1] - t[0]) / (5 * n_components)  # Reasonable time scale
            initial_guess.extend([t0_guess, amp_guess, tau_guess])
    
    # Auto-generate bounds if not provided
    if bounds is None:
        t_min, t_max = t[0], t[-1]
        y_min, y_max = np.min(y), np.max(y)
        y_range = y_max - y_min
        
        lower = [y_min - y_range]  # baseline
        upper = [y_max + y_range]  # baseline
        
        for i in range(n_components):
            lower.extend([t_min, -2*y_range, 0.1])  # t0, amplitude, tau
            upper.extend([t_max, 2*y_range, t_max - t_min])  # t0, amplitude, tau
        
        bounds = (lower, upper)
    
    # Perform fitting
    try:
        params, _ = curve_fit(multi_response, t, y, 
                              p0=initial_guess, 
                              absolute_sigma=absolute_sigma,
                              sigma=sigma,
                              bounds=bounds,
                              check_finite=False, 
                              **kwargs)
    except RuntimeError:
        params = np.array([])
    
    return params


def plot_fit_results(t, y_data, params, axes=None):
    """
    Plot data, fit, residuals, and individual components
    """
    y_fit = multi_response(t, *params)
    residuals = y_data - y_fit
    
    n_components = (len(params) - 1) // 3
    
    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Data and fit
    axes[0].plot(t, y_data, 'o', markersize=3, c='gray', label='Data')
    axes[0].plot(t, y_fit, 'deeppink', linewidth=1, label='Fit')
    axes[0].axhline(params[0], color='k', lw=0.75, linestyle='--', label='Baseline')
    axes[0].set_ylabel('Signal')
    axes[0].set_title('Data and Fitted Model')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Individual components
    axes[1].axhline(0, color='k', linestyle=':', alpha=0.5)
    
    for i in range(n_components):
        idx = 1 + 3*i
        t0, amp, tau = params[idx], params[idx+1], params[idx+2]
        component = single_response(t, t0, amp, tau)
        axes[1].plot(t, component, color=plt.cm.tab10(i), linewidth=2,
                    label='Comp {:d}: τ={:.1f}'.format(i+1, tau))
        axes[1].axvline(t0, color=plt.cm.tab10(i), linestyle='--', alpha=0.3)
    
    axes[1].set_ylabel('Component Contribution')
    axes[1].set_title('Individual Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    axes[2].plot(t, residuals, 'deeppink', markersize=2, alpha=0.5)
    axes[2].axhline(0, color='k', linestyle='-', alpha=0.2)
    axes[2].fill_between(t, -2*np.std(residuals), 2*np.std(residuals), 
                         alpha=0.2, color='silver', label='±2σ')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Residuals')
    axes[2].set_title('Residuals (σ = {:.4f})'.format(np.std(residuals)))
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)



####################
# Find settle time #
####################

# def find_settle_time(timestream, threshold=0.05):
#     """
#     Time complexity: O(n_cycles × n_rows × n_samples)
#     Best for: Large arrays where memory isn't a constraint
#     """
#     n_cycles, n_rows, n_samples = timestream.shape
#     t = np.arange(n_samples)
    
#     # Final values for each trace
#     final = timestream[..., -1, np.newaxis]  # Shape: (n_cycles, n_rows, 1)
    
#     # Boolean: where NOT within threshold
#     violations = np.abs(timestream - final) >= threshold
    
#     # For each (cycle, row), find index of last True in violations
#     # Trick: Use argmax on flipped array
#     violations_flipped = np.flip(violations, axis=-1)
    
#     # Find first True in flipped array (= last True in original)
#     has_violation = np.any(violations, axis=-1, keepdims=True)
#     last_violation_from_end = np.argmax(violations_flipped, axis=-1)
#     last_violation = n_samples - 1 - last_violation_from_end
    
#     # Settling time is one sample after last violation
#     settle_indices = last_violation + 1
#     settle_indices[settle_indices>=n_samples] = n_samples-1
    
#     # Handle edge cases
#     never_settles = violations[..., -1]  # Violation at last sample
#     always_settled = ~np.any(violations, axis=-1)  # No violations
    
#     settle_times = t[settle_indices]
#     settle_times = np.where(never_settles, np.inf, settle_times)
#     settle_times = np.where(always_settled, t[0], settle_times)
    
#     return settle_times


def find_settle_time(timestream, adc_offset, threshold=50):
    """
    Find the last sample time where |y - adc_offset| >= threshold
    
    Time complexity: O(n_cycles × n_rows × n_samples)
    
    Parameters:
    -----------
    timestream : ndarray, shape (n_cycles, n_rows, n_samples)
        The time series data
    adc_offset : float or ndarray
        Expected y-value (can be scalar or broadcast-compatible array)
    threshold : float
        Tolerance band around adc_offset
    
    Returns:
    --------
    settle_times : ndarray, shape (n_cycles, n_rows)
        Time of last violation, or -inf if always settled, or inf if never settles
    """
    n_cycles, n_rows, n_samples = timestream.shape
    t = np.arange(n_samples)
    
    # Boolean: where value is outside the band [adc_offset - threshold, adc_offset + threshold]
    violations = np.abs(timestream - adc_offset) >= threshold
    
    # Find last True in violations along time axis
    violations_flipped = np.flip(violations, axis=-1)
    last_violation_from_end = np.argmax(violations_flipped, axis=-1)
    last_violation = n_samples - 1 - last_violation_from_end
    
    # Edge cases
    never_settles = violations[..., -1]  # Still violating at last sample
    always_settled = ~np.any(violations, axis=-1)  # No violations anywhere
    
    settle_times = t[last_violation]
    settle_times = np.where(never_settles, np.nan, settle_times)
    settle_times = np.where(always_settled, np.nan, settle_times)
    
    return settle_times


def find_settle_time_per_segment(timestream, adc_offset, threshold=50):
    """
    Find the last sample time where |y - adc_offset| >= threshold
    
    Time complexity: O(n_cycles × n_rows × n_samples)
    
    Parameters:
    -----------
    timestream : ndarray, shape (n_cycles, n_rows, n_samples)
        The time series data
    adc_offset : float or ndarray
        Expected y-value (can be scalar or broadcast-compatible array)
    threshold : float
        Tolerance band around adc_offset
    
    Returns:
    --------
    settle_times : ndarray, shape (n_cycles, n_rows)
        Time of last violation, or -inf if always settled, or inf if never settles
    """
    n_samples = timestream.shape[0]
    t = np.arange(n_samples)
    
    # Boolean: where value is outside the band [adc_offset - threshold, adc_offset + threshold]
    violations = np.abs(timestream - adc_offset) >= threshold
    
    # Find last True in violations along time axis
    violations_flipped = np.flip(violations, axis=-1)
    last_violation_from_end = np.argmax(violations_flipped, axis=-1)
    last_violation = n_samples - 1 - last_violation_from_end
    
    # Edge cases
    never_settles = violations[-1]  # Still violating at last sample
    always_settled = ~np.any(violations, axis=-1)  # No violations anywhere
    
    settle_times = t[last_violation]
    settle_times = np.where(never_settles, np.nan, settle_times)
    settle_times = np.where(always_settled, np.nan, settle_times)
    
    return settle_times

# def find_settle_time_per_segment(timestream,
#                            t_range,
#                            threshold = 0.1):
#     """
#     Calculate when the signal settles within threshold of its final value.
    
#     The signal is considered settled when |y(t) - y_final| < threshold
#     for all subsequent times.
    
#     Time complexity: O(n) instead of O(n²)
    
#     Parameters:
#     -----------
#     timestream : array
#         Signal values
#     t_range : array
#         Time array corresponding to timestream
#     threshold : float
#         Absolute threshold for settling
        
#     Returns:
#     --------
#     settling_time : float
#         Time when signal settles within threshold
#         Returns np.inf if not settled within t_range
#     """
#     final = timestream[-1]
    
#     # Boolean array: True where within threshold
#     within_threshold = np.abs(timestream - final) < threshold
    
#     # Find all violations (where signal leaves threshold band)
#     # Working backwards is key for O(n) complexity
#     violations = np.where(~within_threshold)[0]
    
#     if len(violations) == 0:
#         # Always within threshold
#         return t_range[0]
    
#     # Settling time is right after the last violation
#     last_violation_idx = violations[-1]
    
#     if last_violation_idx == len(t_range) - 1:
#         # Never settles (violation at last point)
#         return np.inf
    
#     return t_range[last_violation_idx + 1]


### Build masks

def build_mask(fname):
    df = pd.read_csv(fname, header=None, skiprows=2)
    
    # Forward-fill CS column, extract indices
    df[0] = df[0].ffill()
    df['cs']  = df[0].str.extract(r'CS(\d+)').astype(int)
    df['row'] = df[1].str.extract(r'RS(\d+)').astype(int)
    
    # Cols 3–6 in CSV → pixel cols 0–3
    col_map = {3: 0, 4: 1, 5: 2, 6: 3}
    
    def normalize(val):
        if pd.isna(val): return ''
        return re.sub(r'\s+', ' ', str(val)).strip().lower()
    
    records = []
    for _, row in df.iterrows():
        for raw_col, pixel_col in col_map.items():
            records.append({'cs': row['cs'], 'row': row['row'],
                            'col': pixel_col, 'label': normalize(row[raw_col])})
    
    pixels = pd.DataFrame(records)
    
    mask_rs_bypassed = pixels[pixels['label'] == 'rs bypassed'][['cs','row','col']]
    mask_cs_bypassed = pixels[pixels['label'] == 'cs bypassed'][['cs','row','col']]
    mask_shorted     = pixels[pixels['label'].str.startswith('shorted')][['cs','row','col']]
    mask_megatest    = pixels[pixels['label'].str.startswith('megatest')][['cs','row','col']]
    return mask_rs_bypassed, mask_cs_bypassed, mask_shorted, mask_megatest

def make_bool_mask(df_mask, n_cs=8, n_rows=10, n_cols=4):
    # Shape: (n_rows, n_cs * n_cols) — same layout as vis_tset
    mask = np.zeros((n_rows, n_cs * n_cols), dtype=bool)
    for _, r in df_mask.iterrows():
        flat_col = r['cs'] * n_cols + r['col']  # CS order matches your reshape
        mask[int(r['row']), flat_col] = True
    return mask

