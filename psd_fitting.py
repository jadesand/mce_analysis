import numpy as np
import scipy.optimize as sop
import scipy.signal as sig
from tqdm import tqdm

##########################################################
### Log 10 Power Spectral Density (PSD) Fitting Models ###
##########################################################
def log10_superfast_noise(log10_params, freq):
    '''
    Superfast noise model: red+white noise convolved with a single-pole low-pass filter,
    plus an additional white noise floor.

    The model in linear space is:
        P(f) = wnl1 * [1 + (f/fknee)^alpha] / [1 + (f/f3dB)^2] + wnl2

    Parameters
    ----------
    log10_params : array_like
        Parameters [alpha, log10_fknee, log10_wnl1, log10_f3dB, log10_wnl2], where
        alpha       : red noise spectral index (negative for red noise)
        log10_fknee : log10 of the knee frequency (Hz)
        log10_wnl1  : log10 of the white noise level for the red+white component (PSD units)
        log10_f3dB  : log10 of the single-pole low-pass filter 3dB frequency (Hz)
        log10_wnl2  : log10 of the additional white noise floor (PSD units)
    freq : array_like
        Frequencies at which to evaluate the model (Hz).

    Returns
    -------
    array_like
        log10 of the model PSD evaluated at the specified frequencies.
    '''
    _params = np.atleast_1d(log10_params)
    alpha, log10_fknee, log10_wnl1, log10_f3dB, log10_wnl2 = _params
    fknee = 10**log10_fknee
    f3dB  = 10**log10_f3dB
    wnl1  = 10**log10_wnl1
    wnl2  = 10**log10_wnl2
    P = wnl1**2 * (1 + (freq / fknee)**alpha) / (1 + (freq / f3dB)**2) + wnl2**2
    return np.log10(P)


def log10_superfast_noise_nored(log10_params, freq):
    '''
    Superfast noise model without red noise: white noise convolved with a single-pole
    low-pass filter, plus an additional white noise floor.

    The model in linear space is:
        P(f) = wnl1 / [1 + (f/f3dB)^2] + wnl2

    Parameters
    ----------
    log10_params : array_like
        Parameters [log10_wnl1, log10_f3dB, log10_wnl2], where
        log10_wnl1 : log10 of the white noise level for the LP-filtered component (PSD units)
        log10_f3dB : log10 of the single-pole low-pass filter 3dB frequency (Hz)
        log10_wnl2 : log10 of the additional white noise floor (PSD units)
    freq : array_like
        Frequencies at which to evaluate the model (Hz).

    Returns
    -------
    array_like
        log10 of the model PSD evaluated at the specified frequencies.
    '''
    _params = np.atleast_1d(log10_params)
    log10_wnl1, log10_f3dB, log10_wnl2 = _params
    f3dB = 10**log10_f3dB
    wnl1 = 10**log10_wnl1
    wnl2 = 10**log10_wnl2
    P = wnl1**2 / (1 + (freq / f3dB)**2) + wnl2**2
    return np.log10(P)

def log10_superfast_noise_red_and_white(log10_params, freq):
    '''
    Superfast noise model without red noise: white noise convolved with a single-pole
    low-pass filter, plus an additional white noise floor.

    The model in linear space is:
        P(f) = wnl1 / [1 + (f/f3dB)^2] + wnl2

    Parameters
    ----------
    log10_params : array_like
        Parameters [log10_wnl1, log10_f3dB, log10_wnl2], where
        alpha : spectral index
        log10_fknee : log10 of the fknee (Hz)
        log10_wnl : log10 of the white noise floor (PSD units)
    freq : array_like
        Frequencies at which to evaluate the model (Hz).

    Returns
    -------
    array_like
        log10 of the model PSD evaluated at the specified frequencies.
    '''
    _params = np.atleast_1d(log10_params)
    alpha, log10_fknee, log10_wnl = _params
    fknee = 10**log10_fknee
    wnl = 10**log10_wnl
    P = wnl**2 * (1 + (freq/fknee)**alpha)
    return np.log10(P)


def calc_chi2_superfast(log10_params, fb, log10_Pb, Mb=None):
    '''
    Chi-squared statistic for the superfast noise model fit in log10 space.

    Parameters
    ----------
    log10_params : array_like
        Parameters [alpha, log10_fknee, log10_wnl1, log10_f3dB, log10_wnl2].
    fb : array_like
        Binned frequencies (Hz).
    log10_Pb : array_like
        log10 of the binned PSD values.
    Mb : array_like, optional
        Number of points per bin, used as weights. If None, uniform weights are used.

    Returns
    -------
    float
        Weighted sum of squared residuals in log10 space.
    '''
    log10_model = log10_superfast_noise(log10_params, fb)
    if Mb is None:
        weight = np.ones_like(log10_Pb)
    else:
        weight = np.array(Mb, dtype=float)
    weight /= np.sum(weight)
    return np.sum(weight * (log10_Pb - log10_model) ** 2)


def calc_chi2_superfast_nored(log10_params, fb, log10_Pb, Mb=None):
    '''
    Chi-squared statistic for the no-red superfast noise model fit in log10 space.

    Parameters
    ----------
    log10_params : array_like
        Parameters [log10_wnl1, log10_f3dB, log10_wnl2].
    fb : array_like
        Binned frequencies (Hz).
    log10_Pb : array_like
        log10 of the binned PSD values.
    Mb : array_like, optional
        Number of points per bin, used as weights. If None, uniform weights are used.

    Returns
    -------
    float
        Weighted sum of squared residuals in log10 space.
    '''
    log10_model = log10_superfast_noise_nored(log10_params, fb)
    if Mb is None:
        weight = np.ones_like(log10_Pb)
    else:
        weight = np.array(Mb, dtype=float)
    weight /= np.sum(weight)
    return np.sum(weight * (log10_Pb - log10_model) ** 2)

def calc_chi2_superfast_red_and_white(log10_params, fb, log10_Pb, Mb=None):
    '''
    Chi-squared statistic for the no-red superfast noise model fit in log10 space.

    Parameters
    ----------
    log10_params : array_like
        Parameters [log10_wnl1, log10_f3dB, log10_wnl2].
    fb : array_like
        Binned frequencies (Hz).
    log10_Pb : array_like
        log10 of the binned PSD values.
    Mb : array_like, optional
        Number of points per bin, used as weights. If None, uniform weights are used.

    Returns
    -------
    float
        Weighted sum of squared residuals in log10 space.
    '''
    log10_model = log10_superfast_noise_red_and_white(log10_params, fb)
    if Mb is None:
        weight = np.ones_like(log10_Pb)
    else:
        weight = np.array(Mb, dtype=float)
    weight /= np.sum(weight)
    return np.sum(weight * (log10_Pb - log10_model) ** 2)


##################################
### Simulation and Verification ###
##################################

def simulate_superfast_timestream(N, fs, alpha, log10_fknee, log10_wnl1, log10_f3dB, log10_wnl2, seed=None):
    '''
    Generate a simulated time stream with superfast noise PSD.

    Parameters
    ----------
    N : int
        Number of time samples.
    fs : float
        Sampling frequency (Hz).
    alpha : float
        Red noise spectral index (negative for red noise).
    log10_fknee : float
        log10 of the knee frequency (Hz).
    log10_wnl1 : float
        log10 of the white noise level for the red+white component (PSD units).
    log10_f3dB : float
        log10 of the single-pole low-pass filter 3dB frequency (Hz).
    log10_wnl2 : float
        log10 of the additional white noise floor (PSD units).
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    t : np.ndarray
        Time array (s).
    d : np.ndarray
        Simulated time stream.
    freq : np.ndarray
        Frequencies (Hz) corresponding to the rfft of d.
    psd_true : np.ndarray
        True PSD used to generate the timestream.
    '''
    if seed is not None:
        np.random.seed(seed)

    freq = np.fft.rfftfreq(N, d=1.0/fs)
    log10_params = [alpha, log10_fknee, log10_wnl1, log10_f3dB, log10_wnl2]

    psd_true = np.zeros_like(freq)
    nz = freq > 0
    psd_true[nz] = 10**log10_superfast_noise(log10_params, freq[nz])

    amplitudes = np.sqrt(psd_true * N * fs / 2)
    phases = np.random.uniform(0, 2*np.pi, len(freq))
    ft = amplitudes * np.exp(1j * phases)
    d = np.fft.irfft(ft, n=N)
    t = np.arange(N) / fs

    return t, d, freq, psd_true


def simulate_superfast_timestream_nored(N, fs, log10_wnl1, log10_f3dB, log10_wnl2, seed=None):
    '''
    Generate a simulated time stream with superfast noise PSD.

    Parameters
    ----------
    N : int
        Number of time samples.
    fs : float
        Sampling frequency (Hz).
    alpha : float
        Red noise spectral index (negative for red noise).
    log10_fknee : float
        log10 of the knee frequency (Hz).
    log10_wnl1 : float
        log10 of the white noise level for the red+white component (PSD units).
    log10_f3dB : float
        log10 of the single-pole low-pass filter 3dB frequency (Hz).
    log10_wnl2 : float
        log10 of the additional white noise floor (PSD units).
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    t : np.ndarray
        Time array (s).
    d : np.ndarray
        Simulated time stream.
    freq : np.ndarray
        Frequencies (Hz) corresponding to the rfft of d.
    psd_true : np.ndarray
        True PSD used to generate the timestream.
    '''
    if seed is not None:
        np.random.seed(seed)

    freq = np.fft.rfftfreq(N, d=1.0/fs)
    log10_params = [log10_wnl1, log10_f3dB, log10_wnl2]

    psd_true = np.zeros_like(freq)
    nz = freq > 0
    psd_true[nz] = 10**log10_superfast_noise_nored(log10_params, freq[nz])

    amplitudes = np.sqrt(psd_true * N * fs / 2)
    phases = np.random.uniform(0, 2*np.pi, len(freq))
    ft = amplitudes * np.exp(1j * phases)
    d = np.fft.irfft(ft, n=N)
    t = np.arange(N) / fs

    return t, d, freq, psd_true

###############################
### Masking and Binning PSD ###
###############################

def gen_fit_mask(freq, l_cut=0., h_cut=2e5, f_peak=None, f_harm=None, manual=[]):
    '''
    Generate a mask for fitting raw PSD data based on frequency ranges.

    Parameters
    ----------
    freq : array_like
        Frequencies at which the PSD is evaluated.
    l_cut : float, optional
        Lower frequency cut. Default is 0.
    h_cut : float, optional
        Upper frequency cut. Default is 2e5.

    Returns
    -------
    fit_mask : array_like
        A boolean mask indicating which frequencies are suitable for fitting.
    '''
    
    #below ranges might need to be tweaked
    df = 1000
    fit_cuts = []

    for c in manual:
        fit_cuts.append(c)
    # fit_cuts.append([30000-df, 30000+df])
    # fit_cuts.append([37800-df, 37800+df])
    # fit_cuts.append([45800-df, 45800+df])
    # fit_cuts.append([49000-df, 49000+df])
    # fit_cuts.append([53800-df, 53800+df])
    
    # fit_cuts.append([61500-df, 61500+df])
    # fit_cuts.append([69000-df, 69000+df])
    # fit_cuts.append([76200-df, 76200+df])
    # fit_cuts.append([84400-df, 84400+df])
    # fit_cuts.append([92300-df, 92300+df])
    
    # fit_cuts.append([100000-df, 100000+df])
    # fit_cuts.append([108000-df, 108000+df])
    # fit_cuts.append([115800-df, 115800+df])
    # fit_cuts.append([121600-df, 121600+df])
    # fit_cuts.append([129400-df, 129400+df])

    # fit_cuts.append([138000-df, 138000+df])
    # fit_cuts.append([146000-df, 146000+df])
    # fit_cuts.append([154000-df, 154000+df])
    # fit_cuts.append([161600-df, 161600+df])
    # fit_cuts.append([169400-df, 169400+df])

    # fit_cuts.append([177200-df, 177200+df])
    # fit_cuts.append([185300-df, 185300+df])
    # fit_cuts.append([193800-df, 193800+df])

    if f_peak is not None:
        for fp in f_peak:
            fit_cuts.append([fp-df,fp+df])
    

    # if f_harm is not None:
    #     assert isinstance(f_harm, list)
    #     f0, fh = f_harm
    #     for i in range(int((freq.max()-f0)//fh) + 1):
    #         fit_cuts.append([f0+fh*i-2000,f0+fh*i+2000])
    
    fit_mask = [np.any([freq > fcuts[1], freq < fcuts[0]], axis=0) for fcuts in fit_cuts]
    fit_mask = np.all( [freq > l_cut, freq < h_cut, *fit_mask], axis=0)
    return fit_mask


def bin_loglog(freq, psd, fit_mask=None, nbins=20, bias=1.):
    """
    Bin both frequency and periodogram in log-frequency space.
    Returns the geometric mean of frequency and PSD in each bin, and the number of points per bin.

    Parameters
    ----------
    freq : array_like
        Frequencies at which the PSD is evaluated.
    psd : array_like
        Power spectral density values.
    fit_mask : array_like, optional
        Boolean mask to apply to the frequencies and PSD values before binning. Default is None.
        If None, no masking is applied.
    nbins : int, optional
        Number of bins to use for binning. Default is 20.
    bias : float, optional
        Bias to apply to the PSD values before taking the logarithm. Default is 1.501.
        See details in /home/shiur/tutorials/25-SU/week-03/psd_fitting.ipynb

    Returns
    -------
    fb : np.ndarray
        Binned frequencies (geometric mean).
    Pb : np.ndarray
        Binned power spectral density values (geometric mean, adjusted by bias).
    Mb : np.ndarray
        Number of points in each bin.
    """
    if fit_mask is not None:
        f, P = freq[fit_mask], psd[fit_mask]
    else:
        mask = (freq > 0) & np.isfinite(psd) & (psd > 0)
        f, P = freq[mask], psd[mask]
    logf = np.log10(f)
    logP = np.log10(P)
    bins = np.linspace(logf.min(), logf.max(), nbins + 1)
    digitized = np.digitize(logf, bins)
    fb, Pb, Mb = [], [], []
    for i in range(1, len(bins)):
        idx = digitized == i
        if np.any(idx):
            # Geometric mean for both frequency and PSD
            fb.append(10 ** np.mean(logf[idx]))
            Pb.append(10 ** np.mean(logP[idx]))
            Mb.append(np.sum(idx))
    fb = np.array(fb)
    Pb = np.array(Pb)
    Mb = np.array(Mb)

    ma = Mb==0
    return fb[~ma], bias*Pb[~ma], Mb[~ma]


###############
### Fitting ###
###############

def fit_superfast_psd(fb, Pb, Mb, ini_guess,
                      method='Nelder-Mead',
                      bounds=[(-1, 0), (0, 3), (0, 5), (1, 4), (-2, 3)],
                      minimize_dict={'tol': 1e-3}):
    '''
    Fit the superfast noise model to a single binned PSD.

    Parameters
    ----------
    fb : array_like
        Binned frequencies (Hz).
    Pb : array_like
        Binned PSD values (linear, not log10).
    Mb : array_like
        Number of points per bin, used as weights.
    ini_guess : list of array_like
        Initial guess as [alpha, log10_fknee, log10_wnl1, log10_f3dB, log10_wnl2].
        Each element may be a scalar or array; all combinations are tried.
    method : str, optional
        Optimization method passed to scipy.optimize.minimize. Default is 'Nelder-Mead'.
    bounds : list of tuple, optional
        Bounds for [alpha, log10_fknee, log10_wnl1, log10_f3dB, log10_wnl2].
        Default is [(-3, -0.01), (0, 3), (0, 5), (2, 4), (-2, 3)].
    minimize_dict : dict, optional
        Extra keyword arguments for scipy.optimize.minimize. Default is {'tol': 1e-3}.

    Returns
    -------
    best_params : np.ndarray
        Best-fit parameters [alpha, log10_fknee, log10_wnl1, log10_f3dB, log10_wnl2],
        or array of NaNs if all fits failed.
    ret : OptimizeResult or None
        The scipy result object for the best fit, or None if all fits failed.
    '''
    ma = np.isfinite(Pb) & (Pb > 0)
    fb_fit = fb[ma]
    log10_Pb_fit = np.log10(Pb[ma])
    Mb_fit = Mb[ma]

    ini = [np.atleast_1d(p) for p in ini_guess]

    all_ini = np.array([[a, fk, w1, f3, w2]
                        for a  in ini[0]
                        for fk in ini[1]
                        for w1 in ini[2]
                        for f3 in ini[3]
                        for w2 in ini[4]])

    best_ret = None
    best_chi2 = np.inf
    for p0 in all_ini:
        ret = sop.minimize(calc_chi2_superfast, p0,
                           args=(fb_fit, log10_Pb_fit, Mb_fit),
                           method=method,
                           bounds=bounds,
                           **minimize_dict)
        if ret.success and ret.fun < best_chi2:
            best_chi2 = ret.fun
            best_ret = ret

    if best_ret is None:
        return np.full(5, np.nan), None
    return best_ret.x, best_ret


def fit_superfast_psd_nored(fb, Pb, Mb, ini_guess,
                             method='Nelder-Mead',
                             bounds=[(0, 5), (1, 3), (-2, 3)],
                             minimize_dict={'tol': 1e-3}):
    '''
    Fit the no-red superfast noise model to a single binned PSD.

    Parameters
    ----------
    fb : array_like
        Binned frequencies (Hz).
    Pb : array_like
        Binned PSD values (linear, not log10).
    Mb : array_like
        Number of points per bin, used as weights.
    ini_guess : list of array_like
        Initial guess as [log10_wnl1, log10_f3dB, log10_wnl2].
        Each element may be a scalar or array; all combinations are tried.
    method : str, optional
        Optimization method passed to scipy.optimize.minimize. Default is 'Nelder-Mead'.
    bounds : list of tuple, optional
        Bounds for [log10_wnl1, log10_f3dB, log10_wnl2].
        Default is [(0, 5), (2, 4), (-2, 3)].
    minimize_dict : dict, optional
        Extra keyword arguments for scipy.optimize.minimize. Default is {'tol': 1e-3}.

    Returns
    -------
    best_params : np.ndarray
        Best-fit parameters [log10_wnl1, log10_f3dB, log10_wnl2],
        or array of NaNs if all fits failed.
    ret : OptimizeResult or None
        The scipy result object for the best fit, or None if all fits failed.
    '''
    ma = np.isfinite(Pb) & (Pb > 0)
    fb_fit = fb[ma]
    log10_Pb_fit = np.log10(Pb[ma])
    Mb_fit = Mb[ma]

    ini = [np.atleast_1d(p) for p in ini_guess]

    all_ini = np.array([[w1, f3, w2]
                        for w1 in ini[0]
                        for f3 in ini[1]
                        for w2 in ini[2]])

    best_ret = None
    best_chi2 = np.inf
    for p0 in all_ini:
        ret = sop.minimize(calc_chi2_superfast_nored, p0,
                           args=(fb_fit, log10_Pb_fit, Mb_fit),
                           method=method,
                           bounds=bounds,
                           **minimize_dict)
        if ret.success and ret.fun < best_chi2:
            best_chi2 = ret.fun
            best_ret = ret

    if best_ret is None:
        return np.full(3, np.nan), None
    return best_ret.x, best_ret


def fit_superfast_psd_red_and_white(fb, Pb, Mb, ini_guess,
                             method='Nelder-Mead',
                             bounds=[(-3, 0), (1, 4), (-2, 3)],
                             minimize_dict={'tol': 1e-6}):
    '''
    Fit the no-red superfast noise model to a single binned PSD.

    Parameters
    ----------
    fb : array_like
        Binned frequencies (Hz).
    Pb : array_like
        Binned PSD values (linear, not log10).
    Mb : array_like
        Number of points per bin, used as weights.
    ini_guess : list of array_like
        Initial guess as [log10_wnl1, log10_f3dB, log10_wnl2].
        Each element may be a scalar or array; all combinations are tried.
    method : str, optional
        Optimization method passed to scipy.optimize.minimize. Default is 'Nelder-Mead'.
    bounds : list of tuple, optional
        Bounds for [alpha, log10_fknee, log10_wnl].
        Default is [(0, 5), (2, 4), (-2, 3)].
    minimize_dict : dict, optional
        Extra keyword arguments for scipy.optimize.minimize. Default is {'tol': 1e-3}.

    Returns
    -------
    best_params : np.ndarray
        Best-fit parameters [alpha, log10_fknee, log10_wnl],
        or array of NaNs if all fits failed.
    ret : OptimizeResult or None
        The scipy result object for the best fit, or None if all fits failed.
    '''
    ma = np.isfinite(Pb) & (Pb > 0)
    fb_fit = fb[ma]
    log10_Pb_fit = np.log10(Pb[ma])
    Mb_fit = Mb[ma]

    ini = [np.atleast_1d(p) for p in ini_guess]

    all_ini = np.array([[w1, f3, w2]
                        for w1 in ini[0]
                        for f3 in ini[1]
                        for w2 in ini[2]])

    best_ret = None
    best_chi2 = np.inf
    for p0 in all_ini:
        ret = sop.minimize(calc_chi2_superfast_red_and_white, p0,
                           args=(fb_fit, log10_Pb_fit, Mb_fit),
                           method=method,
                           bounds=bounds,
                           **minimize_dict)
        if ret.success and ret.fun < best_chi2:
            best_chi2 = ret.fun
            best_ret = ret

    if best_ret is None:
        return np.full(3, np.nan), None
    return best_ret.x, best_ret


def fit_superfast_data(freq, psd, ini_guess, nbins=50, manual=[], h_cut=2e6, **fit_kwargs):
    '''
    Full pipeline: mask, bin, and fit a superfast noise PSD.

    Parameters
    ----------
    freq : array_like
        Frequencies (Hz).
    psd : array_like
        Power spectral density values (linear).
    ini_guess : list of array_like
        Initial guess as [alpha, log10_fknee, log10_wnl1, log10_f3dB, log10_wnl2].
        Each element may be a scalar or array; all combinations are tried.
    nbins : int, optional
        Number of log-spaced bins for bin_loglog. Default is 50.
    **fit_kwargs
        Extra keyword arguments passed to fit_superfast_psd.

    Returns
    -------
    params : np.ndarray
        Best-fit parameters [alpha, log10_fknee, log10_wnl1, log10_f3dB, log10_wnl2],
        or array of NaNs if fit failed.
    ret : OptimizeResult or None
        The scipy result object for the best fit, or None if fit failed.
    fb : np.ndarray
        Binned frequencies (Hz).
    Pb : np.ndarray
        Binned PSD values (linear).
    Mb : np.ndarray
        Number of points per bin.
    '''
    mask = gen_fit_mask(freq, manual=manual, h_cut=h_cut)
    fb, Pb, Mb = bin_loglog(freq, psd, fit_mask=mask, nbins=nbins)

    params, ret = fit_superfast_psd(fb, Pb, Mb, ini_guess, **fit_kwargs)
    return params, ret, fb, Pb, Mb


def fit_superfast_data_nored(freq, psd, ini_guess, nbins=50, manual=[], **fit_kwargs):
    '''
    Full pipeline: mask, bin, and fit a superfast noise PSD.

    Parameters
    ----------
    freq : array_like
        Frequencies (Hz).
    psd : array_like
        Power spectral density values (linear).
    ini_guess : list of array_like
        Initial guess as [alpha, log10_fknee, log10_wnl1, log10_f3dB, log10_wnl2].
        Each element may be a scalar or array; all combinations are tried.
    nbins : int, optional
        Number of log-spaced bins for bin_loglog. Default is 50.
    **fit_kwargs
        Extra keyword arguments passed to fit_superfast_psd.

    Returns
    -------
    params : np.ndarray
        Best-fit parameters [alpha, log10_fknee, log10_wnl1, log10_f3dB, log10_wnl2],
        or array of NaNs if fit failed.
    ret : OptimizeResult or None
        The scipy result object for the best fit, or None if fit failed.
    fb : np.ndarray
        Binned frequencies (Hz).
    Pb : np.ndarray
        Binned PSD values (linear).
    Mb : np.ndarray
        Number of points per bin.
    '''
    ma = freq>25000
    spread = np.diff(np.quantile(np.log10(psd[ma]), [0.16, 0.84]))[0]/2
    loc, _ = sig.find_peaks(np.log10(psd[ma])-np.mean(np.log10(psd[ma])), height=7*spread)
    fit_mask = gen_fit_mask(freq, f_peak=freq[ma][loc], manual=manual)
    fb, Pb, Mb = bin_loglog(freq, psd, fit_mask=fit_mask, nbins=nbins)

    params, ret = fit_superfast_psd_nored(fb, Pb, Mb, ini_guess, **fit_kwargs)
    return params, ret, fb, Pb, Mb


def fit_superfast_data_red_and_white(freq, psd, ini_guess, nbins=50, manual=[], h_cut=2e6, **fit_kwargs):
    '''
    Full pipeline: mask, bin, and fit a superfast noise PSD.

    Parameters
    ----------
    freq : array_like
        Frequencies (Hz).
    psd : array_like
        Power spectral density values (linear).
    ini_guess : list of array_like
        Initial guess as [alpha, log10_fknee, log10_wnl1, log10_f3dB, log10_wnl2].
        Each element may be a scalar or array; all combinations are tried.
    nbins : int, optional
        Number of log-spaced bins for bin_loglog. Default is 50.
    **fit_kwargs
        Extra keyword arguments passed to fit_superfast_psd.

    Returns
    -------
    params : np.ndarray
        Best-fit parameters [alpha, log10_fknee, log10_wnl1, log10_f3dB, log10_wnl2],
        or array of NaNs if fit failed.
    ret : OptimizeResult or None
        The scipy result object for the best fit, or None if fit failed.
    fb : np.ndarray
        Binned frequencies (Hz).
    Pb : np.ndarray
        Binned PSD values (linear).
    Mb : np.ndarray
        Number of points per bin.
    '''
    ma = freq>10000
    spread = np.diff(np.quantile(np.log10(psd[ma]), [0.16, 0.84]))[0]/2
    loc, _ = sig.find_peaks(np.log10(psd[ma])-np.mean(np.log10(psd[ma])), height=7*spread)
    fit_mask = gen_fit_mask(freq, f_peak=freq[ma][loc], manual=manual, h_cut=h_cut)
    fb, Pb, Mb = bin_loglog(freq, psd, fit_mask=fit_mask, nbins=nbins)

    params, ret = fit_superfast_psd_red_and_white(fb, Pb, Mb, ini_guess, **fit_kwargs)
    return params, ret, fb, Pb, Mb


def check_fit_bias(true_params, ini_guess, 
                   N=4000000, fs=4e5, n_trials=3,
                   fit_mask_func=None, suffix='_nored',
                   nbins=50, seed=0, **fit_kwargs):
    '''
    Run multiple simulations to check ensemble bias of fit_superfast_psd.

    Parameters
    ----------
    true_params : array_like
        True parameters [alpha, log10_fknee, log10_wnl1, log10_f3dB, log10_wnl2].
    ini_guess : list of array_like
        Initial guess as [alpha, log10_fknee, log10_wnl1, log10_f3dB, log10_wnl2].
        Each element may be a scalar or array; all combinations are tried.
    N : int, optional
        Number of time samples for each simulation. Default is 4 million.
    fs : float, optional
        Sampling frequency (Hz) for each simulation. Default is 400 kHz.
    n_trials : int, optional
        Number of independent simulations to run. Default is 3.
    fit_mask_func : callable or None, optional
        Function with signature f(freq) -> bool mask. If None, gen_fit_mask is used.
    suffix : str, optional
        Suffix for the simulation and fitting functions. Default is '_nored'.
    nbins : int, optional
        Number of log-spaced bins for bin_loglog. Default is 50.
    seed : int, optional
        Base random seed; each trial uses seed+i. Default is 0.
    **fit_kwargs
        Extra keyword arguments passed to fit_superfast_psd.

    Returns
    -------
    all_params : np.ndarray, shape (n_trials, n_params)
        Fit parameters for each trial. Rows with failed fits contain NaNs.
    bias : np.ndarray, shape (n_params,)
        Mean of (fit - true) across successful trials.
    std : np.ndarray, shape (n_params,)
        Std of (fit - true) across successful trials.
    '''
    if suffix not in ['_nored']:
        raise ValueError(f'Unsupported suffix: {suffix}')
    
    if suffix != '_nored':
        param_names = ['alpha', 'log10_fknee', 'log10_wnl1', 'log10_f3dB', 'log10_wnl2']
    else:
        param_names = ['log10_wnl1', 'log10_f3dB', 'log10_wnl2']
    
    n_params = len(param_names)
    true_params = np.asarray(true_params)
    if fit_mask_func is None:
        fit_mask_func = gen_fit_mask

    all_params = np.full((n_trials, n_params), np.nan)
    for i in tqdm(range(n_trials)):
        _, d, freq, _ = eval(f'simulate_superfast_timestream{suffix}')(N, fs, *true_params, seed=seed+i)
        # psd = np.abs(np.fft.rfft(d))**2 / (N * fs / 2)
        freq, psd = sig.welch(d, fs=fs, window='boxcar', nperseg=65536*2, noverlap=0)
        mask = fit_mask_func(freq)
        fb, Pb, Mb = bin_loglog(freq, psd, fit_mask=mask, nbins=nbins)
        params, _ = eval(f'fit_superfast_psd{suffix}')(fb, Pb, Mb, ini_guess, **fit_kwargs)
        all_params[i] = params

    ok = np.all(np.isfinite(all_params), axis=1)
    n_failed = np.sum(~ok)
    bias = np.mean(all_params[ok] - true_params, axis=0)
    std  = np.std( all_params[ok] - true_params, axis=0)

    print(f'Trials: {n_trials}  |  Failed: {n_failed}')
    print(f'{"param":<15} {"true":>8} {"bias":>10} {"std":>10}')
    for name, tv, b, s in zip(param_names, true_params, bias, std):
        print(f'{name:<15} {tv:>8.3f} {b:>10.4f} {s:>10.4f}')

    return all_params, bias, std