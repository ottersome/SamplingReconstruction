import numpy as np
from enum import Enum, auto


class GenMethods(Enum):
    NORMAL = auto()
    UNIFORM = auto()


class DatasetGenerator():
    def __init__(self, samples=1000, num_coeffs, gen_method=GenMethods.NORMAL):
        self.samples = samples
        self.num_coeffs = num_coeffs
        self.samples = []

    def get_time_signal():
        # Generate Positive Coefficietns
        pos_freq_coeff = np.random.randn(N,coeffs) + 1j*np.random.randn(N,coeffs)
        # Do conjugate symmetric
        print(pos_freq_coeff[0,::-1])

        all_coefs = np.concatenate((np.array([0]), pos_freq_coeff[0,:],
            np.conj(pos_freq_coeff[0,::-1])))

        # Do Inverse Fourier Transform
        time_signal = np.fft.ifft(all_coefs)
        return time_signal
