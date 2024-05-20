import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Part 1: Calculate DFT for N2 = N1 and N2 = 2N1
def calculate_dft(signal, N1):
    dft_N1 = fft(signal, N1)
    dft_2N1 = fft(signal, 2*N1)
    return dft_N1, dft_2N1

# Part 2: Generate smoothed signal and calculate DFT
def generate_smoothed_signal(signal):
    N1 = len(signal)
    smoothed_signal = np.zeros_like(signal)
    smoothed_signal[0] = signal[0]
    smoothed_signal[1] = (signal[0] + signal[1]) / 2
    smoothed_signal[N1 - 1] = signal[N1 - 1]

    for n in range(2, N1 - 2):
        smoothed_signal[n] = np.sum(signal[n - 2: n + 3]) / 5

    dft_smoothed_N1, dft_smoothed_2N1 = calculate_dft(smoothed_signal, N1)
    return smoothed_signal, dft_smoothed_N1, dft_smoothed_2N1

# Part 3: Plot DFT components and analyze results
def plot_dft_components(signal, dft_N1, dft_2N1, smoothed_signal, dft_smoothed_N1, dft_smoothed_2N1):
    N1 = len(signal)
    
    # Plot real and imaginary components of DFT
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(np.real(dft_N1), label='Real')
    plt.plot(np.imag(dft_N1), label='Imaginary')
    plt.title('DFT Components (N2 = N1)')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(np.real(dft_2N1), label='Real')
    plt.plot(np.imag(dft_2N1), label='Imaginary')
    plt.title('DFT Components (N2 = 2N1)')
    plt.legend()

    # Plot magnitude and phase of DFT
    plt.subplot(2, 2, 3)
    plt.plot(np.abs(dft_N1), label='Magnitude')
    plt.title('Magnitude (N2 = N1)')

    plt.subplot(2, 2, 4)
    plt.plot(np.abs(dft_2N1), label='Magnitude')
    plt.title('Magnitude (N2 = 2N1)')

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(np.angle(dft_N1), label='Phase')
    plt.title('Phase (N2 = N1)')

    plt.subplot(2, 2, 2)
    plt.plot(np.angle(dft_2N1), label='Phase')
    plt.title('Phase (N2 = 2N1)')

    # Plot smoothed signal and its DFT
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(smoothed_signal)
    plt.title('Smoothed Signal')

    plt.subplot(2, 2, 2)
    plt.plot(np.abs(dft_smoothed_N1), label='Magnitude')
    plt.title('Magnitude of Smoothed Signal (N2 = N1)')

    plt.subplot(2, 2, 3)
    plt.plot(np.abs(dft_smoothed_2N1), label='Magnitude')
    plt.title('Magnitude of Smoothed Signal (N2 = 2N1)')

    plt.subplot(2, 2, 4)
    plt.plot(np.angle(dft_smoothed_2N1), label='Phase')
    plt.title('Phase of Smoothed Signal (N2 = 2N1)')

    plt.show()

def main():
    # Read data from the text file
    data = np.loadtxt('Signals.txt', skiprows=1)  # Skip the first row
    N1 = len(data)  # Assuming each column has the same length
    signals = [data[:, i] for i in range(1, 5)]  # Extracting individual signals
    print("signals", signals)

    for i, signal in enumerate(signals):
        print(f"Processing Signal {i+1}:")
        dft_N1, dft_2N1 = calculate_dft(signal, N1)
        smoothed_signal, dft_smoothed_N1, dft_smoothed_2N1 = generate_smoothed_signal(signal)
        plot_dft_components(signal, dft_N1, dft_2N1, smoothed_signal, dft_smoothed_N1, dft_smoothed_2N1)

if __name__ == "__main__":
    main()
