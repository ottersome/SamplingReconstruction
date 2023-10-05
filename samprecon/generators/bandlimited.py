import numpy as np
from enum import Enum, auto
import numpy.typing as npt

class GenMethods(Enum):
    NORMAL = auto()
    UNIFORM = auto()


"""
Simple in the sense that sampling frequency is equal across
"""
class BandlimitedSimple():
    def __init__(self,
        length_of_signals = 1e3
    ):
        self.length_of_signals = length_of_signals
    
    def generate_signals(self,amount_of_signals):
        num_coeffs = self.length_of_signals//2
        M = amount_of_signals
        N = self.length_of_signals

        spectrums = np.zeros((M,N),dtype=complex)
        signals = np.zeros_like(spectrums)

        # TODO: be done with loop, use numpy vectorization only
        for i in range(M):
            # Coeffs
            #a = generated_random((1,num_coeffs),GenMethods.NORMAL,rang=[0,1])
            #b = generated_random((1,num_coeffs),GenMethods.NORMAL,rang=[0,1])
            a = np.random.randn(1,num_coeffs)*10
            b = np.random.randn(1,num_coeffs)*10
            coeffs = a+1j*b
            #coeffs = np.linspace(0,1,num_coeffs)

            # Assign Positive and Negative, respectively.
            spectrums[i,0:num_coeffs] = np.conj(coeffs[0,::-1])
            spectrums[i,num_coeffs] = 0
            spectrums[i,num_coeffs+1:] = coeffs[0,:]

            spectrum_as_input = np.zeros((1,N),dtype=complex)
            spectrum_as_input[0,0] = 0
            spectrum_as_input[0,1:num_coeffs+1] = coeffs[0,:]
            spectrum_as_input[0,num_coeffs+1:] = np.conj(coeffs[0,::-1])

            signal =  np.fft.ifft(spectrum_as_input[0,:])
            signals[i,:] = signal
            

        return spectrums, signals

    def old_generate_signals(self, amount_of_signals):
        # Generate Random Number
        M = amount_of_signals
        N = self.length_of_signals
        pos_coeffs = generated_random((M,N//2),GenMethods.UNIFORM,rang=[0,1])

        # DC Value + Pos Coeff  + Neg Coeff
        spectra = np.concatenate((np.full((M,1),0), pos_coeffs, np.conj(pos_coeffs[::-1])),axis=1,dtype=complex)

        signals = np.zeros_like(spectra)
        for i in range(spectra.shape[0]):
            signals[i,:] = np.real(np.fft.ifft(spectra[i,:]))
        return spectra,signals

class BandlimitedGenerator():
    def __init__(self, 
            samples,
            length_of_signals=1e3,
            method=GenMethods.NORMAL,
            lim_gen_method=GenMethods.UNIFORM):

        self.N  = length_of_signals
        self.M = samples
        self.num_coeffs = num_coeffs
        self.sampling_rates = []
        self.signals = np.zeros((self.M,self.N))

    # New Version
    def generate_signals(self):
        self.sampling_rates = generated_random((self.N, 1), GenMethods.UNIFORM, rang=[0,1])
        max_frequencies = 0.5*self.sampling_rates # TODO check
        num_coeffs = max_frequencies

        spectrums = np.zeros((self.M, self.N),dtype=complex)

        # TODO: be done with loop, use numpy vectorization only
        for i in range(self.M):
            # Coeffs
            coeffs_num = max_frequencies[i,1]
            coeffs = generated_random((1,coeffs_num),GenMethods.UNIFORM,rang=[0,1])

            spectrum = 2*max_frequencies[i,1]+1
            
            # Assign Positive and Negative, respectively.
            spectrums[i,1:coeffs_num+1] = coeffs
            spectrums[i,-coeffs_num,:] = np.conj(coeffs[::-1])

            signal = np.real(np.fft.ifft(spectrums[i,:]))
            self.signals[i,:] = signal
            
            coeffs = (np.random.rand(self.N, num_coeffs)-0.5)

        return self.signals

    def get_subsampled_signals(self,decimation_rates):
        return self.signals[:,::decimation_rates]

def generated_random(dimen: npt.ArrayLike,method: GenMethods, **kwargs):
    # TODO add lower bounding if necessary
    if method == GenMethods.NORMAL:
        return np.random.randn(*dimen)*kwargs['rang'][-1]
    if method == GenMethods.UNIFORM:
        return np.random.randn(*dimen)*kwargs['rang'][-1]


    # Old Version
    def get_time_signal(self):
        # Generate Positive Coefficietns
        pos_freq_coeff = np.random.randn(N,coeffs) + 1j*np.random.randn(N,coeffs)
        # Do conjugate symmetric
        print(pos_freq_coeff[0,::-1])

        all_coefs = np.concatenate((np.array([0]), pos_freq_coeff[0,:],
            np.conj(pos_freq_coeff[0,::-1])))

        # Do Inverse Fourier Transform
        time_signal = np.fft.ifft(all_coefs)
        return time_signal

