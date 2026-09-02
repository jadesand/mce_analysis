import numpy as np


def find_spike_simple_sigma(d, sigma=8):
    sig = np.diff(np.quantile(np.diff(d), [0.16, 0.84]))/2
    return np.argwhere(np.abs(np.diff(d)) > sigma*sig).ravel()


def fix_baseline(d, loc=[], win=1000, delta=50):

    # raise 
    
    nmax = len(d)
    thresh = np.diff(np.quantile(d, [0.16, 0.84])) #/np.sqrt(win)

    out = d.copy()
    for l in loc:
        # print(l)
        l0 = max(l-delta-win, 0)
        l1 = max(l-delta, 0)

        r0 = min(l+delta, nmax)
        r1 = min(l+delta+win, nmax)
        
        if (l1-l0)<10 or (r1-r0)<10:
            
            continue

        lbase = np.median(d[l0:l1])
        rbase = np.median(d[r0:r1])

        if np.abs(rbase-lbase) < thresh:
            continue

        # print(rbase - lbase)
        out[l:] -= rbase-lbase
    return out