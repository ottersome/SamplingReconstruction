from scipy.interpolate import interp1d
from samprecon.generators import BandlimitedGenerator

class SignalEnvironment:

    def __init__(self, signal_generator: BandlimitedGenerator):
        self.signal_generator = signal_generator
        self.original_signal = self.signal_generator.generate_signals()
        # TODO define original signal

    def step()
