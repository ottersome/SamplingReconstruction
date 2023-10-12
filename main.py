import torch
import numpy as np
import matplotlib.pyplot as plt
from  samprecon.environments.bandlimited import BandlimitedGenerator, GenMethods
from samprecon.learners.actors import SimpleActor
from samprecon.learners.qestimators import SimpleQEstimator
import argparse
from tqdm import tqdm

def argsFunc():
    argsParsr = argparse.ArgumentParser()
    argsParsr.add_argument('--dataset_size',default=1000)
    argsParsr.add_argument('--data_points',default=1000)
    argsParsr.add_argument('--num_coefficients',default=10000)
    argsParsr.add_argument('--time_period',default=1)
    argsParsr.add_argument('--training_epochs',default=1)
    argsParsr.add_argument('--sample_budget',default=5)
    argsParsr.add_argument('--band_a',default=2000)
    argsParsr.add_argument('--band_b',default=1)

    argsParsr.add_argument('-v',action="store_true")
    argsParsr.add_argument('-vv',action="store_true")


    return argsParsr.parse_args()


args = argsFunc()

def plot_generated_coeffs(coeffs):
    magnitudes = np.abs(coeffs)
    #freqs = np.linspace(band_a[],args.band_a,2*args.num_coefficients+1)
    freqs = np.fft.fftfreq(len(coeffs))
    num_coeffs = len(coeffs)
    phases = np.angle(coeffs)
    time_signal = np.fft.ifft(coeffs)
    t = np.linspace(0,1,num_coeffs)

    fig,axs =plt.subplots(1,3,figsize=(15,5))

    axs[0].set_title('Fourier Coefficient Magnitude')
    axs[0].plot(freqs,magnitudes)
    axs[0].set_xlabel('Frequencies')
    axs[0].set_ylabel('Magnitudes')

    axs[1].set_title('Phase')
    axs[1].plot(freqs,phases)
    axs[1].set_xlabel('Frequency index')
    axs[1].set_ylabel('Phase (radians)')

    axs[2].plot(t, np.real(time_signal))
    axs[2].set_title('Band-limted, Non-periodic Signal')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Amplitude')
    plt.show()

N = args.dataset_size
T = args.time_period
epochs = args.training_epochs
samp_budget = args.sample_budget
dataPoints = args.data_points
band = [args.band_a,args.band_b]
coeffs = args.num_coefficients

x = np.linspace(0,T*dataPoints, dataPoints)

def main():

    # Chose Frequency
    # f_max would be f_nyquist
    fmax = 1

    # Subsampling rate
    ssr = 

    # Create Actors and Critics
    # Actors chose sampling rate
    actor = SimpleActor(samp_budget,128)

    ## Training Loop
    epoch_meter = tqdm(total=epochs,desc='Training Epoch')
    for epoch in epochs
    #Once actor chose sampling rate we regenerate the environment and collect payoff
        # Get State
        state = 

        epoch_meter.update(1)

    # Basically N Environments
    dataset_gen = DatasetGenerator(N,coeffs,GenMethods.NORMAL)
    time_signal = dataset_gen.get_time_signal()

    # GV

    # TODO 


    

## Create all Coefficients

## Sample from Signals
# We build a sample bank with random sampling times for offline learning.
decimation_rate = np.random.rand(N)*10 + 1e-5




## Reconstruct Signals
