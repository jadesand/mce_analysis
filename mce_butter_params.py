import numpy as np


def mce_butter_params(nrow, rowlen, f_cutoff):
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
        Desired -3dB cutoff frequency in Hz.

    Returns
    -------
    params : list of int
        [b11, b12, b21, b22, k1, k2] as integer values for the MCE.
    """
    f_samp = 50e6 / nrow / rowlen
    Wn = 2.0 * f_cutoff / f_samp
    if Wn >= 1.0:
        raise ValueError(
            "f_cutoff={} Hz gives Wn={} >= 1 (cutoff at or above Nyquist)".format(f_cutoff, Wn)
        )

    # Design 4th-order Butterworth via bilinear transform (no scipy needed).
    # Pre-warp the digital cutoff to analog frequency
    wc_a = 2.0 * np.tan(np.pi * Wn / 2.0)

    # 4th-order Butterworth poles in the s-plane at angles pi*(2k+5)/8, k=0..3
    angles = np.pi * (2 * np.arange(4) + 5) / 8.0
    s_poles = wc_a * np.exp(1j * angles)

    # Bilinear transform to z-plane: z = (2 + s) / (2 - s)
    z_poles = (2.0 + s_poles) / (2.0 - s_poles)

    # Sort poles by angle to form conjugate pairs
    # Pair with smaller angle -> section 1 (b11, b12), further from unit circle
    # Pair with larger angle  -> section 2 (b21, b22), closer to unit circle
    idx = np.argsort(np.abs(np.angle(z_poles)))
    pair1 = z_poles[idx[0]], z_poles[idx[1]]  # smaller angle
    pair2 = z_poles[idx[2]], z_poles[idx[3]]  # larger angle

    # Biquad denominator coefficients: 1 + a1*z^-1 + a2*z^-2
    # from (1 - p1*z^-1)(1 - p2*z^-1) => a1 = -(p1+p2), a2 = p1*p2
    a1_0 = -(pair1[0] + pair1[1]).real
    a2_0 = (pair1[0] * pair1[1]).real
    a1_1 = -(pair2[0] + pair2[1]).real
    a2_1 = (pair2[0] * pair2[1]).real

    # Quantize to MCE 1.14 fixed-point format: b = floor(|a| * 2^14)
    b11 = int(np.floor(abs(a1_0) * 2**14))
    b12 = int(np.floor(a2_0 * 2**14))
    b21 = int(np.floor(abs(a1_1) * 2**14))
    b22 = int(np.floor(a2_1 * 2**14))

    # Compute section DC gains (with unnormalized (1+z^-1)^2 numerator per section)
    # G_i = 4 / (1 + a1_i + a2_i)
    G0 = 4.0 / (1 + a1_0 + a2_0)
    G1 = 4.0 / (1 + a1_1 + a2_1)

    # Truncation parameters from MCE wiki:
    # k1 = floor(log2(G1)) - 10  (output truncation, from section 2 gain)
    # k2 = floor(log2(G0))       (inter-stage truncation, from section 1 gain)
    k1 = int(np.floor(np.log2(G1))) - 10
    k2 = int(np.floor(np.log2(G0)))

    return [b11, b12, b21, b22, k1, k2]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute MCE Butterworth filter parameters for a given "
                    "nrow, rowlen, and cutoff frequency."
    )
    parser.add_argument("nrow", type=int, help="Number of rows")
    parser.add_argument("rowlen", type=int, help="Row length")
    parser.add_argument("f_cutoff", type=float, help="Cutoff frequency in Hz")
    parser.add_argument("--verbose", type=int, help="Verbose output (0 or 1)", default=0)
    args = parser.parse_args()

    f_samp = 50e6 / args.nrow / args.rowlen
    params = mce_butter_params(args.nrow, args.rowlen, args.f_cutoff)

    if args.verbose:
        print("f_samp = {:.2f} Hz".format(f_samp))
        print("f_cutoff = {} Hz".format(args.f_cutoff))
        print("Wn = {:.6f}".format(2*args.f_cutoff/f_samp))
        print("b11, b12, b21, b22, k1, k2 = {}".format(params))
    else:
        print(params)