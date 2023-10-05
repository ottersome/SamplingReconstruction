import numpy as np

def intra_zeropad(og_sig, upsamp_factor):
    zeros = np.zeros(upsamp_factor*len(og_sig) - (upsamp_factor-1))
    zeros[::upsamp_factor] = og_sig
    return 

def sinc_interpolation(padded_signal,upsamp_factor,fc=None):

def perfect_upsampling(signal,upsamp_factor):
    padded = intra_zeropad(sigal,upsamp_factor)

