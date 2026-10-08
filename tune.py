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


def read_sqtune(filename, section='SQUID_SQ1_RAMPC'):
    """
    Read lock-point characterization (per row, per column) from an MCE
    .sqtune file.

    The <SQUID_SQ1_RAMP> and <SQUID_SQ1_RAMPC> sections each hold, for
    every column, one line per metric keyed as '<metric>_C##' with one
    value per row -- this is the per-channel V-phi/lock-point info (peak-
    to-peak, phi0, lock range/slope/count). <SQUID_SQ1_RAMPC> is the final
    post-lock characterization; <SQUID_SQ1_RAMP> is the initial scan.

    Args:
        filename: path to the .sqtune file
        section: 'SQUID_SQ1_RAMPC' (default) or 'SQUID_SQ1_RAMP'

    Returns:
        dict of {metric_name: array of shape (n_row, n_col)}, with metrics
        such as 'vphi_p2p', 'vphi_phi0', 'lock_range', 'lock_slope_up',
        'lock_slope_down', 'lock_count_up', 'lock_count_dn'.
    """
    with open(filename) as f:
        text = f.read()

    m = re.search(rf'<{section}>(.*?)</{section}>', text, re.S)
    if m is None:
        raise ValueError(f'section {section} not found in {filename}')
    body = m.group(1)

    per_col = {}
    for line in body.strip().splitlines():
        line = line.strip()
        tag_m = re.match(r'<(\w+)_C(\d+)>\s+(.*)', line)
        if tag_m is None:
            continue
        metric, col, values = tag_m.group(1), int(tag_m.group(2)), tag_m.group(3)
        per_col.setdefault(metric, {})[col] = np.array(values.split(), dtype=float)

    result = {}
    for metric, cols in per_col.items():
        n_col = max(cols) + 1
        n_row = len(next(iter(cols.values())))
        arr = np.full((n_row, n_col), np.nan)
        for col, values in cols.items():
            arr[:, col] = values
        result[metric] = arr

    return result


def find_sq1_lock_points(flux, error, safb):
    """
    Locate the SQ1 servo lock point (flux, SA feedback) for every row/column
    from raw sq1servo_sa data, as the first "down" zero-crossing of the
    error signal (error going from positive to negative) -- the same
    crossing direction as the <lock_slope_down> metric in .sqtune files.
    Takes the earliest such crossing since the V-phi curve repeats every
    flux quantum and there is no other stored reference for which period
    was actually used.

    Args:
        flux: array of shape (n_flux,) -- flux axis, as returned by
            read_sq1servo_sa
        error: array of shape (n_row, n_bias, n_flux, n_col) -- error signal,
            as returned by read_sq1servo_sa
        safb: array of shape (n_row, n_bias, n_flux, n_col) -- SA feedback,
            as returned by read_sq1servo_sa

    Returns:
        lock_flux: array of shape (n_row, n_bias, n_col) -- flux value at
            the lock point (linearly interpolated to the zero crossing)
        lock_safb: array of shape (n_row, n_bias, n_col) -- SA feedback
            value at the lock point (linearly interpolated to match)
    """
    n_row, n_bias, n_flux, n_col = error.shape
    lock_flux = np.full((n_row, n_bias, n_col), np.nan)
    lock_safb = np.full((n_row, n_bias, n_col), np.nan)

    for row in range(n_row):
        for b in range(n_bias):
            for col in range(n_col):
                err_curve = error[row, b, :, col]
                safb_curve = safb[row, b, :, col]
                down_idx = np.where(
                    (err_curve[:-1] > 0) & (err_curve[1:] < 0)
                )[0]
                if len(down_idx) == 0:
                    continue
                i = down_idx[0]
                # linear interpolation to the exact zero crossing
                frac = err_curve[i] / (err_curve[i] - err_curve[i + 1])
                lock_flux[row, b, col] = flux[i] + frac * (flux[i + 1] - flux[i])
                lock_safb[row, b, col] = safb_curve[i] + frac * (safb_curve[i + 1] - safb_curve[i])

    return lock_flux, lock_safb


def read_sq1servo_sa(filename):
    """
    Read an MCE RCs_sq1servo_sa data file (ASCII table with a
    '<bias> <flux> <row> <error00..31> <safb00..31>' header) and return the
    bias/flux axes and per-row error/safb data.

    Returns:
        bias: array of shape (n_bias,) -- unique SA bias values swept
        flux: array of shape (n_flux,) -- unique flux values swept (per bias)
        error: array of shape (n_row, n_bias, n_flux, n_col) -- error signal
        safb: array of shape (n_row, n_bias, n_flux, n_col) -- SA feedback
    """
    data = np.loadtxt(filename, skiprows=1)

    bias_col = data[:, 0]
    flux_col = data[:, 1]
    row_col = data[:, 2].astype(int)
    error_cols = data[:, 3:35]
    safb_cols = data[:, 35:67]

    bias = np.unique(bias_col)
    flux = np.unique(flux_col)
    rows = np.unique(row_col)

    n_bias, n_flux, n_row, n_col = len(bias), len(flux), len(rows), error_cols.shape[1]

    error = np.full((n_row, n_bias, n_flux, n_col), np.nan)
    safb = np.full((n_row, n_bias, n_flux, n_col), np.nan)

    bias_idx = np.searchsorted(bias, bias_col)
    flux_idx = np.searchsorted(flux, flux_col)
    row_idx = np.searchsorted(rows, row_col)

    error[row_idx, bias_idx, flux_idx] = error_cols
    safb[row_idx, bias_idx, flux_idx] = safb_cols

    return bias, flux, error, safb


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


def _get_curve_regions(y, extremality=0.9):
    """
    Port of auto_setup.servo.get_curve_regions (pairs=False), used only
    for its side calculation in get_lock_points. Not needed standalone
    here; kept out for simplicity -- see _get_lock_points below, which
    inlines exactly what get_lock_points needs.
    """
    raise NotImplementedError


def _get_slopes(data, index, n_points=5, min_index=None, max_index=None,
                intercept=None):
    """
    Direct port of auto_setup.servo.get_slopes.
    Fit straight line to each row of `data` (2-d array) in the vicinity
    of the corresponding entry of `index` (1-d array).
    """
    n = data.shape[0]
    if min_index is None:
        min_index = np.zeros(n)
    if max_index is None:
        max_index = np.array([d.shape[-1] for d in data])

    fits = []
    for d, i, lo, hi in zip(data, index, min_index, max_index):
        sl_idx = np.arange(max(lo, i - n_points//2),
                            min(hi, i + (n_points+1)//2))
        if len(sl_idx) < 2:
            fits.append([0., 0.])
        else:
            fits.append(np.polyfit(sl_idx - i, d[sl_idx], 1))
    fits = np.array(fits).transpose()

    if intercept is None:
        return fits[0]
    if intercept == 'y':
        return fits[0], fits[1]
    if intercept == 'x':
        x0 = -fits[1] / fits[0]
        x0[fits[0] == 0] = 0.
        return fits[0], x0
    raise ValueError('Invalid intercept request "%s"' % intercept)


def _get_lock_points(y, scale=5, lock_amp=True, slope=1., extremality=0.9):
    """
    Direct port of auto_setup.servo.get_lock_points (start/stop/x_adjust
    dropped since we don't need them here).

    y : (n_curves, nflux) array
    slope : scalar or (n_curves,) array of +1/-1, selects which
            zero-crossing direction to lock to (matches the .sqtune
            "slope" convention: after sign-correcting the curve by
            `slope`, the lock point is found on what is now a rising
            crossing).
    """
    n_curves, nflux = y.shape
    y = y.astype(float)

    y1 = y.max(axis=1)
    y0 = y.min(axis=1)
    mids = ((y1 + y0) / 2).reshape(-1, 1)
    amps = ((y1 - y0) / 2).reshape(-1, 1)
    amps[amps == 0] = 1.

    slope = np.sign(np.asarray(slope, dtype=float))
    if slope.ndim == 0:
        slope = np.full(n_curves, slope)
    slope = slope.reshape(-1, 1)

    y2 = slope * (y - mids) / amps

    ranges = []
    oks = []
    for yy in y2:
        ok = True
        right_idx = (yy >= extremality).nonzero()[0]
        if len(right_idx) == 0:
            right_idx = np.array([len(yy) - 1])
            ok = False
        left_idx = (yy[:right_idx[-1]] <= -extremality).nonzero()[0]
        if len(left_idx) > 0:
            left_idx = left_idx[-1]
            right_idx = min(right_idx[right_idx >= left_idx])
        else:
            ok = False
            left_idx = right_idx[-1]
            right_idx = np.hstack(((yy <= -extremality).nonzero()[0], 0))[0]
        if left_idx >= right_idx:
            left_idx, right_idx = 0, 0
        ranges.append((left_idx, right_idx))
        oks.append(ok)
    ranges = np.array(ranges)
    i_left, i_right = ranges[:, 0], ranges[:, 1]

    if lock_amp:
        target = np.array([yy[a] + yy[b] for yy, a, b in
                            zip(y, i_left, i_right)]) / 2
        lock_idx = np.array([a + np.argmin(np.abs(yy[a:b+1] - t)) for
                              a, b, t, yy in zip(i_left, i_right, target, y)]
                             ).astype(int)
        lock_slope, dx = _get_slopes(y - target.reshape(-1, 1), lock_idx,
                                      intercept='x', n_points=scale,
                                      min_index=i_left, max_index=i_right)
        lock_y = np.array([yy[i] for i, yy in zip(lock_idx, y)])
    else:
        lock_idx = ((i_left + i_right) / 2).astype(int)
        lock_slope, lock_y = _get_slopes(y, lock_idx, intercept='y',
                                          n_points=scale,
                                          min_index=i_left, max_index=i_right)
        dx = np.zeros(n_curves)

    return {
        'lock_idx': lock_idx,
        'lock_didx': dx,
        'lock_y': lock_y,
        'lock_slope': lock_slope,
        'left_idx': i_left,
        'right_idx': i_right,
        'ok': np.array(oks),
    }


def sq1_lock_gain(sq1fb, safb, slope=-1., lock_amp=True, scale=None):
    """
    Compute the SQ1 lock point and local SQ1-stage gain (dSAFB/dSQ1FB)
    for every (row, col) curve, following the same algorithm as
    auto_setup.servo.get_lock_points / SQ1ServoSA.reduce (the routine
    that produces "lock_slope" for sq1servo_sa data).

    Parameters
    ----------
    sq1fb : (nflux,) array
        SQ1 feedback values swept during the servo.
    safb : (nrow, nflux, ncol) array
        SA feedback response curves.
    slope : scalar or (nrow, ncol) array of +1/-1
        Which zero-crossing direction to lock to. This matches the
        sign convention used throughout auto_setup (e.g.
        SQ1Servo.reduce's default of -sign(default_servo_i * sq1_servo_gain)).
        Flip the sign if you're consistently getting the wrong branch.
    lock_amp : bool
        Lock mid-way in y (SAFB amplitude), matching the auto_setup
        default for sq1servo_sa.
    scale : int, optional
        Number of points used in the local slope fit; defaults to
        max(nflux/40, 1), same as auto_setup.

    Returns
    -------
    lock_x : (nrow, ncol) array
        SQ1 feedback value at the lock point.
    lock_slope : (nrow, ncol) array
        Local slope dSAFB/dSQ1FB at the lock point (the SQ1-stage gain).
    ok : (nrow, ncol) bool array
        Whether a good lock point (real hi/lo extrema pair) was found.
    """
    sq1fb = np.asarray(sq1fb, dtype=float)
    nrow, nflux, ncol = safb.shape
    d_fb = sq1fb[1] - sq1fb[0]

    y = safb.transpose(0, 2, 1).reshape(nrow * ncol, nflux)

    if scale is None:
        scale = max(nflux // 40, 1)

    slope_arr = np.asarray(slope, dtype=float)
    if slope_arr.ndim == 0:
        slope_arr = np.full(nrow * ncol, slope_arr)
    else:
        slope_arr = slope_arr.reshape(nrow * ncol)

    an = _get_lock_points(y, scale=scale, lock_amp=lock_amp, slope=slope_arr)

    lock_idx = an['lock_idx'] + (an['lock_didx']).astype(int)
    lock_x = sq1fb[0] + lock_idx * d_fb
    lock_slope = an['lock_slope'] / d_fb

    lock_x = lock_x.reshape(nrow, ncol)
    lock_slope = lock_slope.reshape(nrow, ncol)
    ok = an['ok'].reshape(nrow, ncol)

    return lock_x, lock_slope, ok