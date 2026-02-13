import numpy as np


def _compute_max_gain(b11, b12, b21, b22, k1, k2):
    """Compute max normalized gain for given quantized params."""
    K = 1.0 / 2**14
    n = 2000
    omega = np.pi * np.arange(1, n) / n  # skip DC, go to just below Nyquist
    z = np.exp(-1j * omega)

    b11s, b12s, b21s, b22s = b11 * K, b12 * K, b21 * K, b22 * K
    denom1 = 1.0 - b11s*z + b12s*z**2
    denom2 = 1.0 - b21s*z + b22s*z**2

    H = (1.0 + z)**4 / denom1 / denom2
    H = H / 2**(k1 + k2)

    # DC gain for normalization
    H_dc = 16.0 / (1.0 - b11s + b12s) / (1.0 - b21s + b22s) / 2**(k1 + k2)
    Y = np.abs(H) / np.abs(H_dc)
    return np.max(Y)


def _params_for_Wn(Wn):
    """Compute quantized params for a given normalized cutoff Wn (no search)."""
    wc_a = 2.0 * np.tan(np.pi * Wn / 2.0)
    angles = np.pi * (2 * np.arange(4) + 5) / 8.0
    s_poles = wc_a * np.exp(1j * angles)
    z_poles = (2.0 + s_poles) / (2.0 - s_poles)

    idx = np.argsort(np.abs(np.angle(z_poles)))
    pair2 = z_poles[idx[0]], z_poles[idx[1]]
    pair1 = z_poles[idx[2]], z_poles[idx[3]]

    a1_0 = -(pair1[0] + pair1[1]).real
    a2_0 = (pair1[0] * pair1[1]).real
    a1_1 = -(pair2[0] + pair2[1]).real
    a2_1 = (pair2[0] * pair2[1]).real

    b11 = int(np.round(abs(a1_0) * 2**14))
    b12 = int(np.round(a2_0 * 2**14))
    b21 = int(np.round(abs(a1_1) * 2**14))
    b22 = int(np.round(a2_1 * 2**14))

    G0 = 4.0 / (1 + a1_0 + a2_0)
    G1 = 4.0 / (1 + a1_1 + a2_1)
    k1 = int(np.floor(np.log2(G0))) - 10
    k2 = int(np.floor(np.log2(G1)))

    return [b11, b12, b21, b22, k1, k2]


def mce_butter_params(nrow, rowlen, f_cutoff, find_friendly=True):
    """
    Compute the 6 MCE Butterworth filter parameters (b11, b12, b21, b22, k1, k2)
    for a given nrow, rowlen, and target cutoff frequency, following the MCE wiki
    procedure.

    The filter is a 4th-order Butterworth low-pass implemented as two cascaded
    biquad sections. The sampling frequency is:
        f_samp = 50 MHz / (nrow * rowlen)
    and the normalized cutoff is:
        Wn = 2 * f_cutoff / f_samp

    Parameters
    ----------
    nrow : int
        Number of rows.
    rowlen : int
        Row length.
    f_cutoff : float
        Desired -3dB cutoff frequency in Hz (maximum cutoff).
    find_friendly : bool, optional
        If True (default), search for the largest quantization-friendly cutoff
        frequency <= f_cutoff that produces no gain peaking.

    Returns
    -------
    params : list of int
        [b11, b12, b21, b22, k1, k2] as integer values for the MCE.
    actual_f_cutoff : float (only if find_friendly=True)
        The actual cutoff frequency used (may be slightly less than f_cutoff).
    """
    f_samp = 50e6 / nrow / rowlen
    Wn_max = 2.0 * f_cutoff / f_samp
    if Wn_max >= 1.0:
        raise ValueError(
            "f_cutoff={} Hz gives Wn={} >= 1 (cutoff at or above Nyquist)".format(f_cutoff, Wn_max)
        )

    if not find_friendly:
        # Just return params for the exact cutoff (may have peaking)
        return _params_for_Wn(Wn_max)

    # Search for largest Wn <= Wn_max where quantized filter has no peaking.
    # We search downward in small steps until we find one with peak <= 1.0
    # Step size chosen so we don't miss good candidates (relative step ~0.1%)
    step = Wn_max * 0.001
    min_Wn = Wn_max * 0.5  # Don't go below 50% of requested cutoff

    best_Wn = None
    best_params = None

    Wn = Wn_max
    while Wn >= min_Wn:
        params = _params_for_Wn(Wn)
        peak = _compute_max_gain(params[0], params[1], params[2], params[3],
                                  params[4], params[5])
        if peak <= 1.0:
            best_Wn = Wn
            best_params = params
            break
        Wn -= step

    if best_params is None:
        # Fallback: just use the requested cutoff even if it has peaking
        best_Wn = Wn_max
        best_params = _params_for_Wn(Wn_max)

    actual_f_cutoff = best_Wn * f_samp / 2.0
    return best_params, actual_f_cutoff


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute MCE Butterworth filter parameters for a given "
                    "nrow, rowlen, and cutoff frequency."
    )
    parser.add_argument("nrow", type=int, help="Number of rows")
    parser.add_argument("rowlen", type=int, help="Row length")
    parser.add_argument("f_cutoff", type=float, help="Maximum cutoff frequency in Hz")
    parser.add_argument("--no-search", default=False, help="Disable search for quantization-friendly cutoff")
    parser.add_argument("--verbose", type=int, help="Verbose output (0 or 1)", default=0)
    args = parser.parse_args()

    f_samp = 50e6 / args.nrow / args.rowlen

    if args.no_search:
        params = mce_butter_params(args.nrow, args.rowlen, args.f_cutoff,
                                   find_friendly=False)
        actual_f_cutoff = args.f_cutoff
    else:
        params, actual_f_cutoff = mce_butter_params(args.nrow, args.rowlen,
                                                     args.f_cutoff)

    if args.verbose:
        print("f_samp = {:.2f} Hz".format(f_samp))
        print("f_cutoff_requested = {} Hz".format(args.f_cutoff))
        print("f_cutoff_actual = {:.2f} Hz".format(actual_f_cutoff))
        print("Wn = {:.6f}".format(2*actual_f_cutoff/f_samp))
        print("b11, b12, b21, b22, k1, k2 = {}".format(params))
    else:
        print(params)