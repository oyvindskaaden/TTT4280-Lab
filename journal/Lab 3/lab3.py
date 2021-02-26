#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import math as m

v_ref = 3.3

resolution = 2**12

maxSpeed = 10

lpCutoff_freq = 160*maxSpeed


fs = 31250

c = 343


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
        # Remove garbage signals
        data = data[int(len(data) / 2):]
        # Transpose so each channel has its own array
        data = data.T
        # Scale sample period
        sample_period *= 1e-6
        print(data)
    return sample_period, np.array(data)


def butter_coeff(cutoffFreq, sampleFreq, filterType='high', order=6):
    """
    Find butterworth filter coefficients for a given cutoff frequency in Hz
    with a given sample rate 'sampleFreq'. Highpass with fitlerType='high' and low with filterType='low'.
    Order of the filter is given by order=, default is 6
    """
    # Find the nyquist frequency
    nyquistFreq = 0.5 * sampleFreq
    # Normalized frequency for cutoff
    normal_cutoff = cutoffFreq / nyquistFreq
    return signal.butter(order, normal_cutoff, btype=filterType, analog=False)


def butter_filter(dataPoints, cutoffFreq, sampleFreq, filterType='high', order=6):
    """ 
    Filter dataPoints using butterworth filter for a given cutoff frequency in Hz
    with a given sample rate 'samplerate'. Highpass with fitlerType='high' and low with filterType='low'.
    Order of the filter is given by order=, default is 6
    """
    b, a = butter_coeff(cutoffFreq, sampleFreq, filterType, order=order)
    return signal.filtfilt(b, a, dataPoints)




def Calculate(filename):
    # Import data from bin file
    sample_period, data = raspi_import(filename)

    sample_freq = 1 / sample_period

    data = data[:2]
    #sample_period *= 1e-6  # change unit to micro seconds
    data = data * (v_ref / resolution) # Change to volts from mV


    data = signal.detrend(data)
    #data = butter_filter(data, hpCutoff_freq, sample_freq)
    data = butter_filter(data, lpCutoff_freq, sample_freq, 'low')

    
    # Generate time axis
    num_of_samples = data.shape[1]  # returns shape of matrix
    t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

    # Generate frequency axis and take FFT
    freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
    spectrum = np.fft.fft(data) # takes FFT channels 1 and 2

    # Plot the results in two subplots
    # NOTICE: This lazily plots the entire matrixes. All the channels will be put into the same plots.
    # If you want a single channel, use data[n-1] to get channel n
    plt.figure("Data")
    plt.subplot(2, 1, 1)
    plt.title("Time domain signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage")
    plt.plot(t, data[0], label="1")
    plt.plot(t, data[1], label="2")
    plt.legend(loc="upper left")

    plt.subplot(2, 1, 2)
    plt.title("Power spectrum of signal")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power [dB]")
    plt.plot(freq, 20*np.log(np.abs(spectrum.T))) # get the power spectrum

    plt.show()

    return


Calculate("./dopplerData/adcData.bin")

