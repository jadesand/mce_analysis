'''
Written by Tom Liu, adapting code from David Goldfinger
2023 May
'''


import numpy as np
import scipy
import scipy.constants
import scipy.stats as sst
from scipy.signal import savgol_filter

import os
import argparse
import time

from .. import tools as tls
from . import plot as pd


# we should figure out where these warnings are coming from some day
import warnings
warnings.filterwarnings("ignore")


PHI0_WB = scipy.constants.value(u'mag. flux quantum')  # magnetic flux quantum, in Wb


####################
# Autocorrelation  #
####################

def serial_corr(wave, lag=1):
    n = len(wave)
    y1 = wave[lag:]
    y2 = wave[:n - lag]
    return np.corrcoef(y1, y2)[0, 1]


def autocorr(wave):
    lags = range(len(wave))
    corrs = np.array([serial_corr(wave, lag) for lag in lags])
    return lags, corrs


def estimate_phi0(x, y, min_acorr_dist_from_zero_frac=0.1, debug=False):
    """
    Estimate a SQUID curve's phi0 from the period of the autocorrelation of y.

    Parameters
    ----------
    x : ndarray
        Flux bias.
    y : ndarray
        SQUID response (voltage or current), one value per x.
    min_acorr_dist_from_zero_frac : float
        Minimum lag (as a fraction of len(x)) to consider as a candidate period.

    Returns
    -------
    phi0 : float
        Estimated phi0, in the units of x. Raises ValueError if no peak is found.
    """
    min_lag = len(x) * min_acorr_dist_from_zero_frac
    if debug:
        print(f'-> min_acorr_dist_from_zero_frac={min_acorr_dist_from_zero_frac}')
        print(f'-> min_lag={min_lag}')

    _, corrs = autocorr(y)

    from scipy.signal import find_peaks
    peaks, _ = find_peaks(corrs, height=0)
    sorted_peaks = sorted(peaks)
    if debug:
        print(f'-> sorted_peaks[:4]={sorted_peaks[:4]}')

    try:
        phi0_idx = next(pk for pk in sorted_peaks if pk > min_lag)
    except StopIteration:
        raise ValueError('No peaks found')

    phi0 = np.abs(x[phi0_idx] - x[0])
    if debug:
        print(f'-> phi0={phi0}')
    return phi0


def get_midpoints(x):
    return (x[1:] + x[:-1]) / 2.


def mutual_inductance_pH(phi0_ua):
    """Mutual inductance (pH) implied by a phi0 measured in uA."""
    return 1.e12 * PHI0_WB / (phi0_ua * 1.e-6)


#################################
# Slope fitting (SSA & SQ1)     #
#################################

def _extrema_in_first_phi0(x, dy, phi0, x0=None):
    """Index (into dy / midpoints) of the steepest up- and down-slope within
    one phi0 of the start of x (or of x0, if given)."""
    x0 = x[0] if x0 is None else x0
    in_first_phi0 = np.where(x < x0 + phi0)
    up_idx = np.argmax(dy[in_first_phi0])
    down_idx = np.argmin(dy[in_first_phi0])
    return up_idx, down_idx


def fit_ssa_slopes(fb, adu, phi0):
    """
    Find the steepest up- and down-slope of the SSA flux-modulation curve
    within its first phi0 period, fit each with a line, and convert to
    physical transimpedance units.

    Returns
    -------
    up, down : dict
        idx, m, b (ADU vs uA line fit y = m*x + b), plus:
        dV_dFB   : d(ADU)/d(SAFB DAC)
        dV_dPhi0 : d(SSA output, nV) / d(SAFB, u-Phi0)
        dI_dI    : d(SSA output, nV) / d(SSA input, pA)
    max_idx : index of the curve's first-phi0 maximum (for plotting).
    """
    fb_mid, adu_mid = get_midpoints(fb), get_midpoints(adu)
    dy = np.diff(adu)
    # NB: the slope search window is fb < phi0 (unshifted), unlike max_idx's
    # fb < min(fb) + phi0 below -- kept as in the original implementation.
    up_idx, down_idx = _extrema_in_first_phi0(fb, dy, phi0, x0=0)
    max_idx = np.argmax(adu[np.where(fb < np.min(fb) + phi0)])

    d_fb = fb[1] - fb[0]
    fits = {}
    for name, idx in (('up', up_idx), ('down', down_idx)):
        m = dy[idx] / d_fb
        b = adu_mid[idx] - m * fb_mid[idx]
        fits[name] = {'idx': idx, 'm': m, 'b': b}
    return fits['up'], fits['down'], max_idx, fb_mid, adu_mid


def _ssa_slope_to_physical(cfg, m, phi0):
    """Convert an SSA ADU/uA slope to (dV_dPhi0 [nV/uPhi0], dI_dI [nV/pA])."""
    adu_to_v = float(cfg['PREAMPADC']['ADU_TO_VOLTS_AT_PREAMP_INPUT'])
    dv_dphi0 = m * adu_to_v * phi0 * 1.e6 / 1000.  # V/uA -> V/phi0 -> nV/uPhi0
    m_in_pA = 1.e6 * PHI0_WB / (float(cfg['SSA']['SSA_M_IN_PICOHENRY']) * 1.e-12)
    di_di = dv_dphi0 / m_in_pA
    return dv_dphi0, di_di


def calculate_ssa_parameters(sa_data, sa_runfile, cfg, col, show_plot=False):
    '''
    Compute and plot the SSA flux-modulation parameters for one column.
    Adapted from David Goldfinger's script.
    '''
    sa_bias = sa_runfile.Item('HEADER', 'RB sa bias', type='int')

    fb0, d_fb, n_fb = tuple(int(i) for i in sa_runfile.Item('par_ramp', 'par_step loop1 par1'))
    fb_dac_to_uA = 1.e6 * (float(cfg['SSA']['SSA_FB_DAC_MAX_VOLTAGE_VOLTS']) / (
        (np.power(2, int(cfg['SSA']['SSA_FB_DAC_NBITS']))) * (
            float(cfg['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_RESISTANCE_OHMS']) +
            float(cfg['SSA']['SSA_FB_BACKPLANE_RESISTANCE_OHMS']) +
            float(cfg['SSA']['SSA_FB_BC_RESISTANCE_OHM']))))
    fb = (fb0 + d_fb * np.arange(n_fb)) * fb_dac_to_uA

    nrow, ncol, n_pt = sa_data.data.shape
    coadd_adu = np.average([sa_data.data[row][col] for row in range(nrow)], axis=0)
    rc = int(col / 8) + 1
    sample_num = sa_runfile.Item('HEADER', f'RB rc{rc} sample_num', type='int')[0]
    adu = coadd_adu / sample_num

    bias_dac = sa_bias[col]
    bias_ua = 1.e6 * bias_dac * float(cfg['SSA']['SSA_BIAS_DAC_MAX_VOLTAGE_VOLTS']) / (
        (2 ** (float(cfg['SSA']['SSA_BIAS_DAC_NBITS']))) *
        (float(cfg['SSA']['SSA_BIAS_RC_RESISTANCE_OHMS']) +
         float(cfg['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_RESISTANCE_OHMS'])))

    try:
        phi0 = estimate_phi0(fb, adu, debug=False)
    except ValueError:
        pd.plot_ssa_debug(fb, adu)
        return

    up, down, max_idx, fb_mid, adu_mid = fit_ssa_slopes(fb, adu, phi0)

    adu_to_v = float(cfg['PREAMPADC']['ADU_TO_VOLTS_AT_PREAMP_INPUT'])
    Vmod_mV = 1.e3 * (np.max(adu) - np.min(adu)) * adu_to_v
    M_fb = mutual_inductance_pH(phi0)

    for slope in (up, down):
        slope['dV_dFB'] = slope['m'] * fb_dac_to_uA
        slope['dV_dPhi0'], slope['dI_dI'] = _ssa_slope_to_physical(cfg, slope['m'], phi0)

    ax = pd.plot_ssa(cfg, fb, adu, phi0, max_idx, fb_mid, adu_mid,
                      up['idx'], down['idx'], fb_dac_to_uA, up, down,
                      col, bias_ua, Vmod_mV, M_fb, show_plot=show_plot)

    return fb_dac_to_uA, M_fb, ax


#################################
# SQ1 servo + crosstalk fitting #
#################################

def sq1_fb_dac_to_nA(cfg):
    """DAC-code-to-nA conversion factor for the SQ1 feedback line."""
    n_bits = int(cfg['SQ1']['SQ1_FB_DAC_NBITS'])
    i_max = float(cfg['SQ1']['SQ1_FB_DAC_MAX_CURRENT_A'])
    r33 = float(cfg['SQ1']['SQ1_FB_RC_R33_OHM'])
    r28 = float(cfg['SQ1']['SQ1_FB_RC_R28_OHM'])
    r_cable = float(cfg['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_RESISTANCE_OHMS'])
    r_backplane = float(cfg['SQ1']['SQ1_FB_BACKPLANE_RESISTANCE_OHMS'])
    return 1.e9 * i_max / (2 ** n_bits) * (r33 / (r33 + r28 + r_cable + r_backplane))


def _local_window(x, y, center_idx, n_pts):
    '''Slice out n_pts points of (x, y) centered on center_idx.'''
    lo = center_idx - int(n_pts / 2 - 1)
    hi = center_idx + int(n_pts / 2 + 1)
    return x[lo:hi], y[lo:hi]


def fit_sq1_slopes(fb, servo, phi0, n_fit):
    """
    Find the steepest up- and down-slope of the SQ1 servo curve (max SSA FB
    vs SQ1 FB) within its first phi0 period. SQ1 curves are noisier than SSA,
    so each slope is fit to a small window of neighboring points rather than
    a single adjacent pair.

    Returns
    -------
    up, down : dict
        idx, m, b, fit_x, fit_y (the window used for the fit).
        down['m']/['b'] fall back to 0 if the fit failed.
    max_idx : index of the curve's first-phi0 maximum (for plotting).
    fb_mid, servo_mid : midpoint arrays (for plotting).
    """
    fb_mid, servo_mid = get_midpoints(fb), get_midpoints(servo)
    dy = np.diff(servo)
    up_idx, down_idx = _extrema_in_first_phi0(fb, dy, phi0)
    max_idx = np.argmax(servo[np.where(fb < np.min(fb) + phi0)])

    down = {'idx': down_idx}
    down['fit_x'], down['fit_y'] = _local_window(fb, servo, down_idx, n_fit)
    try:
        down['m'], down['b'] = np.polyfit(down['fit_x'], down['fit_y'], 1)
    except Exception:
        print("Failed to fit SQ1 downslope")
        down['m'], down['b'] = 0, 0

    up = {'idx': up_idx}
    up['fit_x'], up['fit_y'] = _local_window(fb, servo, up_idx, n_fit)
    up['m'], up['b'] = np.polyfit(up['fit_x'], up['fit_y'], 1)

    return up, down, max_idx, fb_mid, servo_mid


def calculate_sq1_slopes(cfg, sq1df, sq1_fb, sq1_b, bias_at_max_span, row, col,
                          fb_dac_to_uA, n_fit, M_ssa_fb, sa_ax,
                          filter_sq1, sgfilter_window=5, sgfilter_poly=2):
    '''
    Fit and plot the SQ1 up/downslope response, and the resulting input-current
    crosstalk ratio dI_SSA_IN/dI_SQ1_IN.
    '''
    bias_idx = np.where(sq1_b == bias_at_max_span)[0][0]
    servo_dac = sq1df[(sq1df['<row>'] == row) & (sq1df['<bias>'] == bias_idx)][f'<safb{col:02}>'].values

    fb_nA = sq1_fb_dac_to_nA(cfg)
    fb = fb_nA * sq1_fb / 1000.
    servo_unfilt = servo_dac * fb_dac_to_uA
    servo = savgol_filter(servo_unfilt, sgfilter_window, sgfilter_poly) if filter_sq1 else servo_unfilt

    phi0 = estimate_phi0(fb, servo, debug=False)
    M_fb = mutual_inductance_pH(phi0)

    up, down, max_idx, fb_mid, servo_mid = fit_sq1_slopes(fb, servo, phi0, n_fit)

    def crosstalk_ratio(m):
        # d(SSA_FB) -> d(SSA_IN) -> divided by d(SQ1_FB) -> d(SQ1_IN)
        ssa_in_per_fb = 1.e-6 * 1.e12 * (M_ssa_fb / float(cfg['SSA']['SSA_M_IN_PICOHENRY']))
        sq1_in_per_fb = 1.e-6 * 1.e12 * (M_fb / float(cfg['SQ1']['SQ1_M_IN_PICOHENRY']))
        return m * ssa_in_per_fb / sq1_in_per_fb

    down['dI_dI'] = crosstalk_ratio(down['m'])
    up['dI_dI'] = crosstalk_ratio(up['m'])

    pd.plot_sq1(col, row, fb, servo, filter_sq1, servo_unfilt,
                max_idx, fb_mid, servo_mid,
                down['idx'], down['fit_x'], down['fit_y'],
                up['idx'], up['fit_x'], up['fit_y'],
                phi0, M_fb, fb_dac_to_uA, sa_ax, cfg, up, down)


def _group_by_bias(sq1df_row, colname):
    '''
    Split one row's servo column into a list of per-bias-index arrays, in
    ascending bias-index order. Equivalent to
    [grp[colname].values for _, grp in sq1df_row.groupby('<bias>')] but
    ~10x faster: a plain numpy sort instead of pandas groupby machinery.
    '''
    bias_arr = sq1df_row['<bias>'].to_numpy()
    val_arr = sq1df_row[colname].to_numpy()
    order = np.argsort(bias_arr, kind='stable')
    bias_sorted = bias_arr[order]
    val_sorted = val_arr[order]
    uniq_bias, starts = np.unique(bias_sorted, return_index=True)
    starts = np.append(starts, len(bias_sorted))
    curves = [val_sorted[starts[i]:starts[i + 1]] for i in range(len(uniq_bias))]
    return uniq_bias, curves


def calculate_sq1_parameters(sq1df, sq1_runfile, cfg, col, row, ssa_params,
                              filter_sq1=True, sgfilter_window=5, calc_slopes=False,
                              sgfilter_poly=2, n_fit=6):
    '''
    Take in the sq1 servo data and find the bias giving the largest SQ1
    modulation span for one row. Adapted from David Goldfinger's script.
    '''
    fb_dac_to_uA, M_ssa_fb, sa_ax = ssa_params
    b0, d_b, n_b = tuple(int(i) for i in sq1_runfile.Item('par_ramp', 'par_step loop1 par1'))
    sq1_b = b0 + d_b * np.arange(n_b)
    fb0, d_fb, n_fb = tuple(int(i) for i in sq1_runfile.Item('par_ramp', 'par_step loop2 par1'))
    sq1_fb = fb0 + d_fb * np.arange(n_fb)

    sq1df_row = sq1df[(sq1df['<row>'] == row)]
    bias_indices, curves_dac = _group_by_bias(sq1df_row, f'<safb{col:02}>')
    biases_dac = [sq1_b[b_index] for b_index in bias_indices]
    spans = [np.max(curve) - np.min(curve) for curve in curves_dac]

    max_span = 0
    bias_at_max_span = None
    if spans:
        max_i = int(np.argmax(spans))
        if spans[max_i] > max_span:
            max_span = spans[max_i]
            bias_at_max_span = biases_dac[max_i]

    if calc_slopes:
        print("Calculating Slopes")
        calculate_sq1_slopes(cfg, sq1df, sq1_fb, sq1_b, bias_at_max_span, row, col,
                              fb_dac_to_uA, n_fit, M_ssa_fb, sa_ax,
                              filter_sq1, sgfilter_window=sgfilter_window, sgfilter_poly=sgfilter_poly)

    return curves_dac, biases_dac, bias_at_max_span, max_span, sq1_fb


########################
# Ic min/max & drivers #
########################

def calculate_icminmax(cfg, filter_sq1, row, col, sq1_params, ssa_params,
                        sgfilter_window, sgfilter_poly, mod_thresh=100, convert_units=False):
    '''
    From the SQ1 servo curves at each bias, compute the Ic min/max envelope
    vs. SQ1 bias, the bias of maximum modulation, and the bias where
    modulation first exceeds mod_thresh.
    '''
    curves_dac, biases_dac, bias_at_max_span, max_span, sq1_fb = sq1_params
    fb_dac_to_uA, M_ssa_fb, sa_ax = ssa_params
    assert len(biases_dac) > 1, "Must have more than 1 bias point: bias is currently" + str(biases_dac)

    # np.max/np.min reversed because of SQUID coil polarity; must flip to get physical current
    mins_dac = np.array([np.max(curve) for curve in curves_dac])
    maxs_dac = np.array([np.min(curve) for curve in curves_dac])

    if convert_units:
        n_bits = int(cfg['SQ1']['SQ1_BIAS_DAC_NBITS'])
        r_total = (float(cfg['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_RESISTANCE_OHMS']) +
                   float(cfg['SQ1']['SQ1_BIAS_BACKPLANE_RESISTANCE_OHMS']) +
                   float(cfg['SQ1']['SQ1_BIAS_BC_RESISTANCE_OHM']))
        bias_dac_to_uA = 1.e6 * float(cfg['SQ1']['SQ1_BIAS_DAC_MAX_VOLTAGE_VOLTS']) / (2 ** n_bits * r_total)
    else:
        bias_dac_to_uA = 1
    bias_ua = np.array(biases_dac) * bias_dac_to_uA

    fb_dac_to_sa_in_uA = (fb_dac_to_uA * M_ssa_fb / float(cfg['SSA']['SSA_M_IN_PICOHENRY'])) if convert_units else 1

    ic_min = fb_dac_to_sa_in_uA * (np.mean(curves_dac[0]) - mins_dac)
    ic_max = fb_dac_to_sa_in_uA * (np.mean(curves_dac[0]) - maxs_dac)
    if filter_sq1:
        ic_min = savgol_filter(ic_min, sgfilter_window, sgfilter_poly)
        ic_max = savgol_filter(ic_max, sgfilter_window, sgfilter_poly)

    mod = ic_max - ic_min
    max_idx = np.argmax(mod)
    max_mod = mod[max_idx]

    threshold = mod_thresh * fb_dac_to_sa_in_uA
    start_idx = np.argmax(mod > threshold)
    if start_idx == 0:
        start_idx = -1
    start_mod = ic_max[start_idx]

    return bias_ua, ic_min, ic_max, max_idx, max_mod, start_idx, start_mod


def fill_grid_data(value, row, col, grid=None, max_rows=41, max_cols=32):
    if grid is None:
        grid = np.zeros((max_rows, max_cols))
    grid[row, col] = value
    return grid


def get_rms_noise(sq1df, row, col):
    '''RMS noise of the SQ1 servo at zero bias.'''
    mask = (sq1df['<row>'].to_numpy() == row) & (sq1df['<bias>'].to_numpy() == 0)
    servo = sq1df[f'<safb{col:02}>'].to_numpy()[mask]
    return np.sqrt(np.mean(np.square(servo - np.mean(servo))))


def get_icmaxcolmod(conditions, manual_bias=None):
    '''
    From a list of (cond, ic_params) pairs for one row, extract Ic,min,
    Ic,max, modulation, and the optimal bias -- all from the primary
    (first) condition -- plus, for every other condition, its Ic,col and
    crosstalk-limit bias relative to the primary condition.

    Returns
    -------
    ic_min, ic_max, mod, optimal_bias, manual_mod : from the primary condition.
    ic_col_by_label, crosstalk_bias_by_label : dict, keyed by
        plot.condition_label(cs_state, rs_state) of each non-primary
        condition, mapping to (ic_col, crosstalk_bias).
    '''
    _, (bias_ua, ic_min, ic_max, max_idx, max_mod, start_idx, start_mod) = conditions[0]

    ic_min_at_start = ic_min[start_idx]
    ic_max_at_max = ic_max[max_idx]
    mod = -(ic_min[max_idx] - ic_max[max_idx])
    manual_mod = -(ic_min[manual_bias] - ic_max[manual_bias]) if manual_bias is not None else None
    optimal_bias = bias_ua[max_idx]

    ic_col_by_label = {}
    crosstalk_bias_by_label = {}
    for cond, ic_params in conditions[1:]:
        bias_ua2, ic_min2, ic_max2, max_idx2, max_mod2, start_idx2, start_mod2 = ic_params
        label = pd.condition_label(cond['cs_state'], cond['rs_state'])
        ic_col_by_label[label] = ic_max2[start_idx2]
        crosstalk_bias_by_label[label] = bias_ua2[start_idx2]

    return (ic_min_at_start, ic_max_at_max, mod, optimal_bias, manual_mod,
            ic_col_by_label, crosstalk_bias_by_label)


def make_grids(rows, cols, ctime, show_plot, savedir, convert_units,
                ic_max_grid, mod_grid, optimal_bias_grid, manual_mod,
                pairwise_grids):
    '''
    pairwise_grids : dict, keyed by non-primary condition label, of dicts
        {'ic_col', 'ic_maxcoldiff', 'crosstalk_bias', 'bias_crosstalk_diff'}
        (see ic_driver / get_icmaxcolmod).
    '''
    uname = 'uA' if convert_units else 'DAC'

    print('plotting grids...')
    vmin, vmax = (5, 15) if convert_units else (2000, 6000)
    pd.tile_plot(rows, cols, ic_max_grid, 'Ic,max (' + uname + ')', 'Ic_max_units' + uname,
                 show_plot=show_plot, savedir=savedir, vmin=vmin, vmax=vmax, cmap='magma')

    vmin, vmax = (0, 5) if convert_units else (0, 2000)
    pd.tile_plot(rows, cols, mod_grid, 'Optimal Modulation (' + uname + ')', 'optmod_units' + uname,
                 show_plot=show_plot, savedir=savedir, vmin=vmin, vmax=vmax, cmap='cividis')
    pd.tile_plot(rows, cols, manual_mod, 'Manullay Picked Modulation (' + uname + ')', 'manualmod_units' + uname,
                 show_plot=show_plot, savedir=savedir, vmin=vmin, vmax=vmax, cmap='RdPu')

    vmin, vmax = (1000, 3000) if convert_units else (5000, 15000)
    pd.tile_plot(rows, cols, optimal_bias_grid, 'Optimal Bias (' + uname + ')', 'optbias_units' + uname,
                 show_plot=show_plot, savedir=savedir, vmin=vmin, vmax=vmax, cmap='viridis')

    for label, grids in pairwise_grids.items():
        suffix = '_' + label.replace(' ', '_')

        vmin, vmax = (5, 15) if convert_units else (2000, 6000)
        pd.tile_plot(rows, cols, grids['ic_col'], f'Ic,col, {label} (' + uname + ')',
                     'Ic_col_units' + uname + suffix, display_title=f'Ic,col ({uname})\n{label}',
                     show_plot=show_plot, savedir=savedir, vmin=vmin, vmax=vmax, cmap='magma')

        vmin, vmax = (1000, 3000) if convert_units else (5000, 15000)
        pd.tile_plot(rows, cols, grids['crosstalk_bias'], f'Crosstalk Bias Limit, {label} (' + uname + ')',
                     'crosstalk_units' + uname + suffix, display_title=f'Crosstalk Bias Limit ({uname})\n{label}',
                     show_plot=show_plot, savedir=savedir, vmin=vmin, vmax=vmax, cmap='viridis')

        vmin, vmax = (-1000, 1000) if convert_units else (-5000, 5000)
        pd.tile_plot(rows, cols, grids['bias_crosstalk_diff'],
                     f'Optimal Bias - Crosstalk Bias Limit, {label} (' + uname + ')',
                     'optbias_crosstalk_diff_units' + uname + suffix,
                     display_title=f'Optimal Bias - Crosstalk Limit ({uname})\n{label}',
                     show_plot=show_plot, savedir=savedir, vmin=vmin, vmax=vmax)

        vmin, vmax = (-5, 5) if convert_units else (-2000, 2000)
        pd.tile_plot(rows, cols, grids['ic_maxcoldiff'], f'Ic,max - Ic,col, {label} (' + uname + ')',
                     'Ic_maxcol_diff_units' + uname + suffix, display_title=f'Ic,max - Ic,col ({uname})\n{label}',
                     show_plot=show_plot, savedir=savedir, vmin=vmin, vmax=vmax)


def _ic_params_for_condition(cond, col, row, cfg, ssa_params, filter_sq1,
                              sgfilter_window, sgfilter_poly, convert_units,
                              mod_thresh=100):
    '''
    Compute sq1_params/ic_params for one condition, one row/column. `cond`
    must have 'sq1df' pre-filtered to this column (see ic_driver).
    '''
    sq1df_row = cond['sq1df'][cond['sq1df']['<row>'] == row]
    sq1_params = calculate_sq1_parameters(
        sq1df_row, cond['sq1_runfile'], cfg, col, row, ssa_params,
        filter_sq1=filter_sq1, calc_slopes=False,
        sgfilter_window=sgfilter_window, sgfilter_poly=sgfilter_poly)
    ic_params = calculate_icminmax(
        cfg, filter_sq1, row, col, sq1_params, ssa_params,
        sgfilter_window, sgfilter_poly, convert_units=convert_units,
        mod_thresh=mod_thresh)
    return ic_params


def _chip_num_for_row(row, chip_starts):
    '''
    Index of the chip containing `row`, given chip_starts boundaries
    (e.g. [0,10,20,30,40,50] for 5 chips of 10 rows). None if out of range.
    chip_starts=None means "no chip grouping" -- every row maps to chip 0,
    so all rows stack onto a single summary plot.
    '''
    if chip_starts is None:
        return 0
    for i in range(len(chip_starts) - 1):
        if chip_starts[i] <= row < chip_starts[i + 1]:
            return i
    return None


def ic_driver(cfg, sa_data, sa_runfile, conditions, ctime=None, filter_sq1=True,
              plot_grid_only=False,
              manually_picked_biases=np.array([
                  7000, 7000, 7000, 7000,
                  7000, 7000, 7000, 7000,
                  7000, 7000, 7000, 7000,
                  7000, 7000, 7000, 7000,
                  0, 0, 0, 0,
                  0, 0, 0, 0,
                  0, 0, 0, 0,
                  0, 0, 0, 0,
              ]),
              cols=range(0, 16), rows=range(0, 40), convert_units=False, plot_all_rows=False,
              savedir='../output_data', chip_starts=None):
    '''
    Sweep SQ1 bias for every row/column under each condition, and grid up
    the resulting Ic/crosstalk parameters.

    conditions : list of dicts, 1 to 4 entries, each
        {'cs_state': True|False|None, 'rs_state': True|False,
         'sq1df': DataFrame, 'sq1_runfile': MCERunfile}.
        The first entry is the "primary" condition: its Ic,min/Ic,max/
        modulation/optimal-bias drive the single-valued grids, and every
        later condition is compared against it (Ic,col, crosstalk bias)
        using its own RMS noise as the modulation threshold.
    chip_starts : row boundaries for per-chip summary plots (plot_grid_only
        and plot_all_rows=True are unaffected). Default None stacks every
        row of a column onto a single summary plot, as before; pass e.g.
        (0, 10, 20, 30, 40, 50) for 5 separate per-chip summary plots.
    '''
    try:
        os.mkdir(savedir)
    except FileExistsError:
        pass

    rms_multiplier = 20
    sgfilter_window, sgfilter_poly = 5, 2
    show_plot = False
    sq1_runfile = conditions[0]['sq1_runfile']
    max_rows = max(rows) + 1

    ic_max_grid = mod_grid = optimal_bias_grid = manual_mod_grid = None
    pairwise_grids = {}  # label -> {'ic_col', 'ic_maxcoldiff', 'crosstalk_bias', 'bias_crosstalk_diff'}
    fig, ax = None, None
    chip_figs = {}  # chip_num -> (fig, ax), for the per-chip summary plots

    for col in cols:
        colname = f'<safb{col:02}>'
        print("Analyzing Column: " + str(col))
        try:
            ssa_params = calculate_ssa_parameters(sa_data, sa_runfile, cfg, col, show_plot=show_plot)
        except TypeError:
            print('Skipping Column: ' + str(col))
            continue
        if ssa_params is None:
            print('Skipping Column: ' + str(col))
            continue

        b0, d_b, n_b = tuple(int(i) for i in sq1_runfile.Item('par_ramp', 'par_step loop1 par1'))
        sq1_b = b0 + d_b * np.arange(n_b)
        manual_bias = manually_picked_biases[col]
        manual_bias_idx = (manual_bias <= sq1_b).argmax()

        cols_cond = [dict(cond, sq1df=cond['sq1df'].filter(['<bias>', '<flux>', '<row>', colname], axis=1))
                     for cond in conditions]

        # Last swept row for each chip, so plot_icminmax_col knows when to
        # finalize each chip's summary plot.
        chip_of_row = {row: _chip_num_for_row(row, chip_starts) for row in rows}
        last_row_of_chip = {}
        for row in rows:
            last_row_of_chip[chip_of_row[row]] = row

        for row in rows:
            chip_num = chip_of_row[row]
            last_fig = (row == last_row_of_chip[chip_num])

            # RMS noise for the non-primary conditions' mod_thresh is always
            # taken from the primary condition's zero-bias data, matching
            # the original implementation.
            primary_rms_noise = get_rms_noise(cols_cond[0]['sq1df'], row, col)

            row_results = []
            for i, cond in enumerate(cols_cond):
                mod_thresh = 100 if i == 0 else primary_rms_noise * rms_multiplier
                ic_params = _ic_params_for_condition(
                    cond, col, row, cfg, ssa_params, filter_sq1,
                    sgfilter_window, sgfilter_poly, convert_units, mod_thresh=mod_thresh)
                row_results.append((cond, ic_params))

            if not plot_grid_only:
                if plot_all_rows:
                    fig, ax = pd.plot_icminmax(
                        col, row, row_results, ctime=ctime, convert_units=convert_units,
                        s1b_minmax_ax=ax, s1b_minmax_fig=fig, savedir=savedir, show_plot=show_plot,
                        manual_bias_idx=manual_bias_idx)
                else:
                    chip_fig, chip_ax = chip_figs.get(chip_num, (None, None))
                    chip_fig, chip_ax = pd.plot_icminmax_col(
                        last_fig, col, row_results, ctime=ctime,
                        s1b_minmax_ax=chip_ax, s1b_minmax_fig=chip_fig, convert_units=convert_units,
                        show_plot=show_plot, savedir=savedir, manual_bias_idx=manual_bias_idx,
                        chip_num=None if chip_starts is None else chip_num)
                    if last_fig:
                        chip_figs.pop(chip_num, None)
                    else:
                        chip_figs[chip_num] = (chip_fig, chip_ax)

            ic_min, ic_max, mod, optimal_bias, manual_mod, ic_col_by_label, crosstalk_bias_by_label = \
                get_icmaxcolmod(row_results, manual_bias=manual_bias_idx)

            ic_max_grid = fill_grid_data(ic_max, row, col, grid=ic_max_grid, max_rows=max_rows)
            mod_grid = fill_grid_data(mod, row, col, grid=mod_grid, max_rows=max_rows)
            optimal_bias_grid = fill_grid_data(optimal_bias, row, col, grid=optimal_bias_grid, max_rows=max_rows)
            manual_mod_grid = fill_grid_data(manual_mod, row, col, grid=manual_mod_grid, max_rows=max_rows)

            for label, ic_col in ic_col_by_label.items():
                crosstalk_bias = crosstalk_bias_by_label[label]
                grids = pairwise_grids.setdefault(label, {
                    'ic_col': None, 'ic_maxcoldiff': None,
                    'crosstalk_bias': None, 'bias_crosstalk_diff': None})
                grids['ic_col'] = fill_grid_data(ic_col, row, col, grid=grids['ic_col'], max_rows=max_rows)
                grids['ic_maxcoldiff'] = fill_grid_data(
                    ic_max - ic_col, row, col, grid=grids['ic_maxcoldiff'], max_rows=max_rows)
                grids['crosstalk_bias'] = fill_grid_data(
                    crosstalk_bias, row, col, grid=grids['crosstalk_bias'], max_rows=max_rows)
                grids['bias_crosstalk_diff'] = fill_grid_data(
                    optimal_bias - crosstalk_bias, row, col, grid=grids['bias_crosstalk_diff'], max_rows=max_rows)

    make_grids(rows, cols, ctime, show_plot, savedir, convert_units,
               ic_max_grid, mod_grid, optimal_bias_grid, manual_mod_grid, pairwise_grids)


def rs_driver(cfg, sa_data, sa_runfile, rsdf, rs_runfile, ctime=None,
              rsdf_off=None, rs_runfile_off=None, filter_sq1=True,
              cols=range(0, 16), rows=range(0, 40), savedir='../output_data/'):
    '''Plot the row-select servo curve for every row/column, grouped by chip.'''
    chip_starts = [0, 10, 20, 30, 41]
    sgfilter_window, sgfilter_poly = 5, 2
    show_plot = False

    for col in cols:
        print("Analyzing Column: " + str(col))
        try:
            ssa_params = calculate_ssa_parameters(sa_data, sa_runfile, cfg, col, show_plot=show_plot)
        except TypeError:
            print('Skipping Column: ' + str(col))
            continue
        if ssa_params is None:
            print('Skipping Column: ' + str(col))
            continue

        fig, ax = None, None
        for row in rows:
            chip_num = next((i for i in range(4) if chip_starts[i] <= row < chip_starts[i + 1]), None)
            if chip_num is None:
                raise ValueError("Row does not correspond to chip: " + str(row))
            last_fig = (row == rows[-1])

            sq1_params = calculate_sq1_parameters(
                rsdf, rs_runfile, cfg, col, row, ssa_params,
                filter_sq1=filter_sq1, calc_slopes=False,
                sgfilter_window=sgfilter_window, sgfilter_poly=sgfilter_poly)

            sq1_params2 = None
            if rsdf_off is not None:
                sq1_params2 = calculate_sq1_parameters(
                    rsdf_off, rs_runfile_off, cfg, col, row, ssa_params, filter_sq1=filter_sq1)

            fig, ax = pd.plot_rsservo_col(
                last_fig, col, chip_num, sq1_params, sq1_params2=sq1_params2, ctime=ctime,
                s1b_minmax_ax=ax, s1b_minmax_fig=fig, savedir=savedir)


def save_subset(df, savename, rows=(28, 29, 30, 31, 32, 33), cols=(8, 9, 10, 11, 12)):
    '''Save a small subset of the dataframe (for building test fixtures).'''
    all_cols = ['<bias>', '<flux>', '<row>'] + [f'<safb{c:02}>' for c in cols]
    subset = df.filter(all_cols, axis=1)
    subset[subset['<row>'].isin(rows)].to_csv(savename)


#######
# CLI #
#######

def _build_conditions(cs_on_rs_on, cs_on_rs_off, cs_off_rs_on, cs_off_rs_off):
    '''
    Build the ic_driver `conditions` list from the four --cs-*-rs-* directory
    flags. If no --cs-off-* flag was given, this is 1-level muxing: the
    cs-on directories become cs_state=None conditions (no "cs" axis). The
    primary condition is always cs-on-rs-on, so it must be given whenever
    any other condition is.
    '''
    two_level = (cs_off_rs_on is not None) or (cs_off_rs_off is not None)

    specs = [
        (True if two_level else None, True, cs_on_rs_on),
        (True if two_level else None, False, cs_on_rs_off),
        (False, True, cs_off_rs_on),
        (False, False, cs_off_rs_off),
    ]
    conditions = []
    for cs_state, rs_state, ctime_dir in specs:
        if ctime_dir is None:
            continue
        sq1df, sq1_runfile = tls.load_sq1_tune_data(ctime_dir)
        conditions.append({'cs_state': cs_state, 'rs_state': rs_state,
                            'sq1df': sq1df, 'sq1_runfile': sq1_runfile})
    if not conditions:
        raise ValueError('At least --cs-on-rs-on must be given for --dev_cur')
    return conditions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--ctime', help='/path/to/ctime/directory (--rsservo only)')
    parser.add_argument('-u', '--ctime_off',
                        help='/path/to/ctime/directory when the row selects are turned off (--rsservo only)')
    parser.add_argument('--cs-on-rs-on', help='/path/to/ctime/directory, cs on rs on (or the only dir for 1-level)')
    parser.add_argument('--cs-on-rs-off', help='/path/to/ctime/directory, cs on rs off (or 1-level rs off)')
    parser.add_argument('--cs-off-rs-on', help='/path/to/ctime/directory, cs off rs on (2-level only)')
    parser.add_argument('--cs-off-rs-off', help='/path/to/ctime/directory, cs off rs off (2-level only)')
    parser.add_argument('-c', '--config', help='/path/to/config/file')
    parser.add_argument('-i', '--dev_cur', action='store_true', help='whether to perform device current analysis')
    parser.add_argument('-r', '--rsservo', action='store_true', help='whether to perform rsservo analysis')
    parser.add_argument('--units', choices=['dac', 'ua', 'both'], default='dac',
                        help='unit for --dev_cur plots/grids: dac (default), ua, or both')
    args = parser.parse_args()

    cols = range(0, 32)
    cfg = tls.load_config_file()

    if args.dev_cur:
        t0 = time.time()
        primary_dir = args.cs_on_rs_on
        ctime = os.path.basename(os.path.dirname(primary_dir))
        print('ctime: ' + str(ctime))

        sa_data, sa_runfile = tls.load_ssa_tune_data(primary_dir)
        conditions = _build_conditions(
            args.cs_on_rs_on, args.cs_on_rs_off, args.cs_off_rs_on, args.cs_off_rs_off)
        rows = range(0, max(conditions[0]['sq1df']['<row>'].astype(int)) + 1)

        print('Done reading files, time elapsed (s):' + str(time.time() - t0))
        unit_variants = {'dac': [False], 'ua': [True], 'both': [True, False]}[args.units]
        for convert_units in unit_variants:
            ic_driver(cfg, sa_data, sa_runfile, conditions,
                      filter_sq1=True, ctime=ctime,
                      cols=cols, rows=rows, convert_units=convert_units, plot_all_rows=False)

    if args.rsservo:
        t0 = time.time()
        print('Reading in files:' + str(args.ctime))
        ctime = os.path.basename(os.path.dirname(args.ctime))
        print('ctime: ' + str(ctime))

        sa_data, sa_runfile = tls.load_ssa_tune_data(args.ctime)
        rsservo_df, rsservo_runfile = tls.load_rsservo_data(args.ctime)
        rows = range(0, max(rsservo_df['<row>'].astype(int)) + 1)

        rsservo_df_off, rsservo_runfile_off = None, None
        if args.ctime_off is not None:
            rsservo_df_off, rsservo_runfile_off = tls.load_rsservo_data(args.ctime_off)

        print('Done reading files, time elapsed (s):' + str(time.time() - t0))
        rs_driver(cfg, sa_data, sa_runfile, rsservo_df, rsservo_runfile,
                  rsdf_off=rsservo_df_off, rs_runfile_off=rsservo_runfile_off,
                  filter_sq1=True, ctime=ctime + '_rsservo', cols=cols, rows=rows)


if __name__ == '__main__':
    t0 = time.time()
    main()
    print("Analysis complete, time elapsed (s): " + str(time.time() - t0))
