import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

v_ref = 3.3

resolution = 2**12

hpCutoff_freq = 100
lpCutoff_freq = 10000


def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.
    Returns sample period and ndarray with one column per channel.
    Sampled data for each channel, in dimensions NUM_SAMPLES x NUM_CHANNELS.
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype=np.uint16)
        data = data.reshape((-1, channels))
        data = data.T
    return sample_period, data


def butter_coeff(cutoff, fs, fType, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return signal.butter(order, normal_cutoff, btype=fType, analog=False)


def butter_filter(dataPoints, cutoff, fs, fType, order=6):
    b, a = butter_coeff(cutoff, fs, fType, order=order)
    return signal.filtfilt(b, a, dataPoints)



# Import data from bin file
sample_period, data = raspi_import('./adcData.bin')

sample_freq = 1 / (sample_period * 1e-6)

#print(sample_freq)

#data = signal.detrend(data, axis=0)  # removes DC component for each channel
sample_period *= 1e-6  # change unit to micro seconds
data = data * (v_ref / resolution) # Change to volts from mV

#print(data)

data = butter_filter(data, hpCutoff_freq, sample_freq, 'high', 6)
data = butter_filter(data, lpCutoff_freq, sample_freq, 'low', 6)

# Generate time axis
num_of_samples = data.shape[1]  # returns shape of matrix
t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

# Generate frequency axis and take FFT
freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
spectrum = np.fft.fft(data)[:3] # takes FFT of all channels

# Lag from mic 2 to mic 1
xcorr21 = np.correlate(data[1][int(len(data[1])/2):], data[0][int(len(data[0])/2):], mode='full')
# Lag from mic 3 to mic 1
xcorr31 = np.correlate(data[2][int(len(data[2])/2):], data[0][int(len(data[0])/2):], mode='full')
# Lag from mic 3 to mic 2
xcorr32 = np.correlate(data[2][int(len(data[2])/2):], data[1][int(len(data[1])/2):], mode='full')


lagAxis = np.linspace(-len(xcorr21) / 2, len(xcorr21) / 2, num=len(xcorr21))

print(xcorr21)

l21 = int(lagAxis[np.where(xcorr21 == max(xcorr21))[0]])
l31 = int(lagAxis[np.where(xcorr31 == max(xcorr31))[0]])
l32 = int(lagAxis[np.where(xcorr32 == max(xcorr32))[0]])

print(f"{l21}, {l31}, {l32}")

plt.figure("Correlation")
plt.subplot(3,1,1)
plt.stem(lagAxis, xcorr21)
plt.subplot(3,1,2)
plt.stem(lagAxis, xcorr31) 
plt.subplot(3,1,3)
plt.stem(lagAxis, xcorr32)

#plt.show()


# Plot the results in two subplots
# NOTICE: This lazily plots the entire matrixes. All the channels will be put into the same plots.
# If you want a single channel, use data[n-1] to get channel n
plt.figure("Data")
plt.subplot(2, 1, 1)
plt.title("Time domain signal")
plt.xlabel("Time [us]")
plt.ylabel("Voltage")
plt.plot(t, data[0], label="1")
plt.plot(t, data[1], label="2")
plt.plot(t, data[2], label="3")
plt.legend(loc="upper left")

plt.subplot(2, 1, 2)
plt.title("Power spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.plot(freq, 20*np.log(np.abs(spectrum.T))) # get the power spectrum



plt.show()
