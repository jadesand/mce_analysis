"""
Model MCE 4 pole butterworth filters.
"""
# from https://e-mode.phas.ubc.ca/mcewiki/images/1/10/Mce_filt_py.txt

from pylab import *

class MCEButterworth:
    def __init__(self, params):
        self.params = params
        self.accum = zeros(4)
    
    def spectrum(self, f):
        K = 1./2**14
        scalars = [K, K, K, K, 1., 1.]
        b11, b12, b21, b22, k1, k2 = [s*p for s,p in zip(scalars, self.params)]
        z = exp(-1j*f)
        H = (1. + z)**4 / (1. - b11*z + b12*z**2) / (1. - b21*z + b22*z**2)
        return H  / 2**(k1+k2)

    def gain(self):
        return self.spectrum(0)

