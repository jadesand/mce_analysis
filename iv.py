# Merged from ~/brianna/get_iv_info.py and iv_tools.py
import numpy as np

from moby2.util.mce import MCEFile


def find_outlier(data, sigma=5, return_keep=True):
    """
    Find outliers in the data based on a specified number of standard deviations.

    Parameters
    ----------
    data : array
        The data to analyze.
    sigma : float, optional
        The number of standard deviations to use for outlier detection. Default is 5.
    return_keep : bool, optional
        If True, return the indices of the non-outliers. If False, return the indices of the outliers. Default is True.

    Returns
    -------
    array
        The indices of the non-outliers or outliers, depending on the value of `return_keep`.
    """
    med = np.median(data)
    err = np.diff(np.quantile(data, [0.16, 0.84]))/2
    indx = np.argwhere(np.abs(data - med) > sigma*err).ravel()
    if return_keep:
        return np.delete(np.arange(len(data)), indx)
    else:
        return indx


def load_iv(fn, unfilter='DC', unit='uA', bias_dac_to_bias_amp=None, dac_sq1fb_to_tes_amp=None):
    """
    Load IV data from MCE file.

    Parameters
    ----------
    fn : str
        Path to the MCE file.
    bias_dac_to_bias_amp : float
        Conversion factor from DAC to amperes for bias current.
    dac_sq1fb_to_tes_amp : float
        Conversion factor from SQ feedback DAC to TES amperes.
    unit : str, optional
        Unit of the output currents. Default is 'uA'.
    unfilter : str, optional
        Unfiltering method for the data. Default is 'DC'.

    Returns
    -------
    Ibias : array
        Bias current on TES circuit.
    Isq1fb or i_tes : array
        SQ1 feedback current (DAC) or TES current (amp), depending on the unit specified.
    """
    assert unit.upper() in ['UA', 'DAC'], f"Unsupported unit: {unit}. Use 'uA', 'dac', or 'DAC'."

    fn_bias = fn + '.bias'
    Ibias_dac = np.loadtxt(fn_bias, skiprows=1)[::-1]

    f = MCEFile(fn)
    Isq1fb_dac = -1.0*f.Read(row_col=True,unfilter=unfilter).data[..., ::-1]

    if unit.upper() == 'DAC':
        return Ibias_dac, Isq1fb_dac

    assert bias_dac_to_bias_amp is not None, "bias_dac_to_bias_amp must be provided for unit conversion."
    assert dac_sq1fb_to_tes_amp is not None, "dac_sq1fb_to_tes_amp must be provided for unit conversion."

    return 1e6 * Ibias_dac * bias_dac_to_bias_amp, 1e6 * Isq1fb_dac * dac_sq1fb_to_tes_amp


def fit_Rpar_mOhm(Ibias_uA, Ites_uA, err_thresh=0.1, Rsh_mOhm=3.0, num_search=15, num_fit=10):
    """
    Fit the parallel resistance from the IV data.

    Parameters
    ----------
    Ibias_uA : array
        Bias current (uA) on TES circuit.
    Ites_uA : array
        TES current (uA).
    err_thresh : float, optional
        Error threshold for the slope calculation. Default is 0.1.
    Rsh_mOhm : float, optional
        Shunt resistance (mOhms). Default is 3.0.
    num_search : int, optional
        Number of data points to search for the slope. Default is 15.
    num_fit : int, optional
        Minimum number of data points required for fitting. Default is 10.

    Returns
    -------
    Rpar_mOhm : float
        Fitted parallel resistance (mOhms).
    """
    order = np.argsort(Ibias_uA)
    Ibias_A = 1e-6 * Ibias_uA[order]
    Ites_A = 1e-6 * Ites_uA[order]
    n = len(Ibias_A)

    if n < num_search:
        print('not enough data point to analyze')
        return np.nan

    Ibias_A = Ibias_A[:num_search]
    Ites_A = Ites_A[:num_search]

    dItes_dIbias = np.diff(Ites_A) / np.diff(Ibias_A)
    err = np.diff(np.quantile(dItes_dIbias, [0.16, 0.84]))/2

    if not err < err_thresh:
        print('too much variation in dItes')
        return np.nan

    idx = find_outlier(dItes_dIbias, sigma=5, return_keep=True)
    if len(idx) < num_fit:
        print('not enough data point to fit')
        return np.nan

    Ites_A = Ites_A[idx]
    Ibias_A = Ibias_A[idx]

    Vsh_V = (Ibias_A - Ites_A) * Rsh_mOhm * 1e-3
    Rpar_Ohm, _ = np.polyfit(Ites_A, Vsh_V, 1)

    return Rpar_Ohm * 1e3


def fit_iv_linear(Ibias_uA, Ites_uA, 
                  num_search=15, err_thresh=0.1, num_fit=10,
                  branch='normal'
                  ):
    """
    Fit the linear portion of the IV curve to determine the SQ feedback response.

    Parameters
    ----------
    Ibias_uA : array
        Bias current (uA).
    Ites_uA : array
        TES current (uA).
    num_search : int, optional
        Number of data points to search for the slope. Default is 15.
    err_thresh : float, optional
        Error threshold for the slope calculation. Default is 0.1.
    num_fit : int, optional
        Minimum number of data points required for fitting. Default is 10.

    Returns
    -------
    k : float
        Slope of the linear fit, representing the SQ feedback response (uA/uA).
    b : float
        Intercept of the linear fit, representing the SQ feedback offset (uA).
    """
    assert branch in ['normal', 'sc'], f"Unsupported branch: {branch}. Use 'normal' or 'sc'."
    order = np.argsort(Ibias_uA)
    x = Ibias_uA[order].copy()
    y = Ites_uA[order].copy()
    n = len(x)

    if n < num_search:
        print('not enough data point to analyze')
        return np.nan, np.nan

    if branch == 'normal':
        x = x[-num_search:]
        y = y[-num_search:]
    else:
        x = x[:num_search]
        y = y[:num_search]

    if np.sum(np.abs(np.diff(y))) == 0:
        print('constant data')
        return np.nan, np.nan

    dy_dx = np.diff(y) / np.diff(x)
    err = np.diff(np.quantile(dy_dx, [0.16, 0.84]))/2
    if not err < err_thresh:
        print('too much variation in dItes/dIbias')
        return np.nan, np.nan
    
    idx = find_outlier(dy_dx, sigma=5, return_keep=True)
    if len(idx) < num_fit:
        print('not enough data point to fit')
        return np.nan, np.nan

    x_fit = x[idx]
    y_fit = y[idx]

    k, b = np.polyfit(x_fit, y_fit, 1)

    return k, b


def correct_iv_uA(Ibias_uA, Ites_uA, jump_thresh=5, num_search=15, err_thresh=1., num_fit=10):
    """
    Correct the IV data by removing the SQ feedback offset.

    Parameters
    ----------
    Ibias_uA : array
        Bias current (uA).
    Ites_uA : array
        TES current (uA).
    num_search : int, optional
        Number of data points to search for the slope. Default is 15.
    err_thresh : float, optional
        Error threshold for the slope calculation. Default is 0.1.
    num_fit : int, optional
        Minimum number of data points required for fitting. Default is 5.

    Returns
    -------
    Ibias_uA_new : array
        0 prepended bias current (uA).
    Ites_uA_new : array
        0 prepended, corrected TES current (uA).
    """

    kn, bn = fit_iv_linear(Ibias_uA, Ites_uA,
                           num_search=num_search,
                           err_thresh=err_thresh,
                           num_fit=num_fit, 
                           branch='normal')

    ks, bs = fit_iv_linear(Ibias_uA, Ites_uA,
                           num_search=num_search,
                           err_thresh=err_thresh,
                           num_fit=num_fit,
                           branch='sc')

    if np.isnan(kn) or np.isnan(ks):
        print('could not fit linear portion of IV curve')
        return np.nan*np.ones(len(Ibias_uA)+1), np.nan*np.ones(len(Ites_uA)+1)
    
    idx = np.argwhere(np.abs(np.diff(Ites_uA))>jump_thresh).ravel()
    if len(idx)==0:
        print('no jump found in IV curve')
        return np.nan*np.ones(len(Ibias_uA)+1), np.nan*np.ones(len(Ites_uA)+1)
    
    idx_sc_end = idx.min() - 3
    idx_tr_start = idx.max() + 3

    
    Ites_uA_new = Ites_uA.copy()
    Ites_uA_new[:idx_sc_end] -= bs
    Ites_uA_new[idx_tr_start:] -= bn
    Ites_uA_new[idx_sc_end:idx_tr_start] = np.nan

    if Ibias_uA[0] != 0:
        Ibias_uA_new = np.concatenate([[0], Ibias_uA])
        Ites_uA_new = np.concatenate([[0], Ites_uA_new])
    return Ibias_uA_new, Ites_uA_new



def get_Rtes_Ptes_Rpar(Ibias_uA, Ites_uA, Rsh_mOhm=3., fit_Rpar=True):
    """
    From IV data, grab resistance v. power

    Parameters
    ----------
    Ibias_uA : array
        Bias current (µA) on TES circuit.
    Ites_uA : array
        TES current (µA).
    Rsh_mOhm : float, optional
        Shunt resistance (mOhms). Default is 3.0 mOhms.
    fit_Rpar : bool, optional
        Whether to fit for the parallel resistance. Default is True.

    Returns
    -------
    Rtes : array
        TES resistance (mOhm)
    Ptes : array
        TES power (pW)
    Rpar_mOhm : float
        Fitted parallel resistance (mOhm)
    """
    Rsh_Ohm = Rsh_mOhm * 1e-3
    Ibias_A = 1e-6 * Ibias_uA
    Ites_A = 1e-6 * Ites_uA

    if fit_Rpar:
        Rpar_mOhm = fit_Rpar_mOhm(Ibias_uA, Ites_uA, Rsh_mOhm=Rsh_mOhm)
    else:
        Rpar_mOhm = 0.0
    Rpar_Ohm = Rpar_mOhm * 1e-3


    Vsh_V = (Ibias_A - Ites_A) * Rsh_Ohm
    Rtot_Ohm = Vsh_V / Ites_A
    Rtes_Ohm = Rtot_Ohm - Rpar_Ohm

    Vtes_V = Vsh_V * (Rtes_Ohm / Rtot_Ohm)
    Ptes_W = Vtes_V * Ites_A

    return Rtes_Ohm * 1e3, Ptes_W * 1e12, Rpar_mOhm