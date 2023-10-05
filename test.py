import numpy as np
import matplotlib.pyplot as plt

# Number of points
N = 500

# Time array
t = np.linspace(0, 1, N, endpoint=False) # 1-second duration

# Frequency components and their coefficients
frequencies = np.array([1, 2]) # 1 Hz and 2 Hz
coefficients = np.array([2 + 3j, 1 + 4j]) # Chosen coefficients

# Construct the complete Fourier coefficients array
full_coefficients = np.zeros(N, dtype=complex)
full_coefficients[frequencies] = coefficients
full_coefficients[-frequencies] = np.conj(coefficients)

# Adding a DC component
DC_value = 5
full_coefficients[0] = DC_value

# Generate the signal using inverse Fourier transform
signal = np.fft.ifft(full_coefficients)
print(signal)

# Plot the real part of the generated signal
plt.plot(t, signal.real)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Generated Real Signal with DC Component')
plt.grid(True)
plt.show()