import numpy as np
from numpy import pi, sin


def intra_zeropad(og_sig, upsamp_factor):
    padded = np.zeros(upsamp_factor * len(og_sig) - (upsamp_factor - 1))
    padded[::upsamp_factor] = og_sig
    return padded


"""
    Parameters
    ----------
    padded_signal : sigal with 0 between samples
    upsamp_factor : number of zeros to insert between samples
    fc : cutoff frequency
"""


def sinc_interpolation(padded_signal, upsamp_factor, fc=None, num_lobes=10):
    # Generaly speaking we have normalized frequency
    if fc is None:
        # fc = 1.0/upsamp_factor
        fc = 1.0 / 2  # Assumign sampling rate is 1

    print("Using cutoff frequency: " + str(fc))

    # def sinc(t,wc,T):
    #    return np.where(t==0,1.0e-20, (wc*T*np.sin(wc*t))/(np.pi*t*wc))
    def sinc(t):
        return np.where(t == 0, 1.0, np.sin(np.pi * t) / (np.pi * t))

    N = len(padded_signal)
    # result = np.zeros_like(padded_signal)
    extended_range = N + 2 * num_lobes * upsamp_factor
    # t = np.arange(-N//2,N//2)/upsamp_factor# To fill the kernel.
    lint = np.arange(-upsamp_factor * N // 2, upsamp_factor * N // 2) / float(
        upsamp_factor
    )  # To fill the kernel.
    interpolating_kernel = sinc(lint / upsamp_factor)

    # for i in range(upsamp_factor):
    #    # Set up Kernel
    #    t = np.arange(-i/upsamp_factor, (N-i)/upsamp_factor,
    #    1/upsamp_factor)# t-nT
    #    interpolating_kernel = sinc(t*fc)

    #    # Add Convoutin to result
    #    result += np.convolve(padded_signal,interpolating_kernel,mode='same')
    interpolated = np.convolve(padded_signal, interpolating_kernel, mode="same")
    return interpolated
    # return my_conv(padded_signal,interpolating_kernel,upsamp_factor,fc,1)


def perfect_upsampling(signal, upsamp_factor):
    padded = intra_zeropad(signal, upsamp_factor)
    return sinc_interpolation(padded, upsamp_factor)


def my_conv(padded_signal, kernel, upsamp_rate, wc, T):
    new_signal = np.zeros_like(padded_signal)
    for t in range(len(padded_signal)):
        xnt = padded_signal[::upsamp_rate]
        nT = np.arange(0, len(padded_signal) // upsamp_rate + 1)
        c_0 = (wc * T) / np.pi
        sincy = np.sin(wc * (t - nT)) / (wc * (t - nT))
        new_signal[t] = np.sum(xnt * c_0 * sincy)
    return new_signal


# Assyume Sampling Rate is Equal to 1
def sinc_interp(x, S, T):
    """
    Parameters
    ----------
        x : decimated Signal
        S : higher resoluition interval
        T : Original Sampling interval
    """
    finer_timescale = np.arange(0, (len(x)-1)*T + S, S)

    conv_idx = np.outer(finer_timescale, np.ones_like(x,dtype=np.float32))

    indices = T * np.outer(
        np.ones_like(finer_timescale), np.arange(len(x))
    )  # Broadcast horizontally(Displacement)
    displacement = (conv_idx - indices) / T
    # Convolve
    res = np.dot(np.sinc(displacement), x)
    return res
