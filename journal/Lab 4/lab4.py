#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import math as m
import sys

sample_freq = 60

lowestBPM = 40
highestBPM = 240

hpCutoff_freq = lowestBPM/60
lpCutoff_freq = highestBPM/60

def raspi_import(path):

    data = np.loadtxt(path, delimiter=" ").T

    sample_period = 1 / sample_freq

    return sample_period, data


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

def AddWindow(data, window="hamming"):
    return data * signal.get_window(window, data.shape[1])
    


def Calculate(filename):
    # Import data from bin file
    sample_period, data = raspi_import(filename)

    #data = data[:2]
    #sample_period *= 1e-6  # change unit to micro seconds
    #data = data * (v_ref / resolution) # Change to volts from mV


    data = signal.detrend(data)
    data = butter_filter(data, hpCutoff_freq, sample_freq)
    data = butter_filter(data, lpCutoff_freq, sample_freq, 'low')

    data = AddWindow(data)
    
    # Generate time axis
    num_of_samples = data.shape[1]  # returns shape of matrix
    t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

    # Generate frequency axis and take FFT
    freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
    spectrum = np.fft.fft(data) # takes FFT

    #dopplershift = freq[np.abs(spectrum).argmax()] #int(freq[np.where(spectrum == max(spectrum))])
    #print(f"Dopplershift is : {dopplershift} and the velocity is {dopplershift/160}")

    # ... ............,,. .,.  *(((((*/////******,,/((######((((((((((((((((((((((((((((***//*,,,,        
    # ..  .,  .,....,   ..  ./(/((((//////******,,*(##((((/((((/****////////(((((((/*,,*//*,,,.        
    # ..............,.   ,   ., *(#((((//////******,,*,//****//////////////(((((((((((((**///*,,,         
    # .,  .,,...,...,,.........,/(((((((////******,*****     ,,,*****///////((((((((((((**///*,,,         
    # . .,....,...,.. .,   ..  ,/(/((((((////**********/*  ..     .**//////(((((((((((((**///*,,,         
    # .,,..,,...,,..,.,.,  ,,*/////(((((/**,,********,,**.,.   ,**////////((((((((((((**///*,,,         
    # .,...,,...,....,.....  ..,//(//,,,**/*,,,..,,,***,*/*,     .**////////(/(((((((((/**///*,,,       ..
    # ................. .....,/***,*.,/*/(*******///****(/*    .,**********////////(((/,*///*,,,       ..
    #                     ....*///**,//(/***////((/**/****    ,***/////////(((((((((((**///**,,.  ..    
    # ..........................,//(###((##(//*/((//***/***,,,, **/////(((((((((((((((((**///***,.  ..    
    # .  ..............        .(//((((/*//,,*///******/.*******//////(((((((((((((((((**///*,,,.  ..    
    #     ..............        .((**//(/*(////**,,*****/.***//////////(((((((((((((((((**///*,,,.  ..    
    # .................         .(((/*,*,,*/**/***,,*****.*///////(/(/((((((((((((((((((**///*,,,.  ..    
    # ..................       .,((((/***,,*,,,**********.*////////////////////////////*,,*//*,,,.  ..    
    # ...............,..      ..,/(((//****/////*****,,**,(,**//////////////((((((((((((***//*,,,.   .    
    #     .....,,..,,,.      .../////*%*,,,,,...,*******,@& .*//////((((((/((((((((((((***//*,,,.   . .  
    # ..........,,..,,..     ...,(((/*/@@//*************.@@@#.....*/((/((((/((((((((((((/**//*,,,.   .    
    # .........,,,..,,..     ...,/*,,,,@@@@&**********,&@@@@%.............,/((((((((((((***//*,,,.   .    
    # ..........,,..,,..     .*,,,,,,,,,@@@@@@&//****@@@@@@@/...................,*((((((/**//*,,,.   .    
    # ..........,,..,,,. ,,,,,,,,,,,,,,,,@@@@@@@@&&@@@@@@@@@/,........................,****//*,,,.   . .. 
    # ...........,..,*,,,,,,,,,,,,,,,,,,,,&@@@@@/***%@@@@@@@/,............................*//*,,,.   . .  
    # ........,,,,,,,,,,,,,,,,//,,,,,,,,,*@@@@&*/***,@@@@@&#..............................*/*,,,.   . .  
    # .....,,,,,,,,,,,,......,*,*,,,,,,,,,,/@@@#/*.,/*&(@@@#,...............,...............*/,,,.   . .. 
    # Modern problems require modern solutions

    bpm = []
    for spec in spectrum:
        bpm.append(abs(freq[np.abs(spec).argmax()])*60)

    return [t, data, freq, spectrum, bpm]


def Plot(t, data, freq, spectrum):
    # Plot the results in two subplots
    # NOTICE: This lazily plots the entire matrixes. All the channels will be put into the same plots.
    # If you want a single channel, use data[n-1] to get channel n
    plt.figure("Data")
    plt.subplot(2, 1, 1)
    plt.title("Time domain signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage")
    plt.plot(t, data[0], "r", label="R")
    plt.plot(t, data[1], "g", label="G")
    plt.plot(t, data[2], "b", label="B")
    plt.legend(loc="upper right")

    plt.subplot(2, 1, 2)
    plt.title("Power spectrum of signal")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Relative Power [dB]")
    plt.plot(freq[:int(len(freq)/8)], 20*np.log(np.abs(spectrum[0].T))[:int(len(freq)/8)], "r", label="R") # get the power spectrum
    plt.plot(freq[:int(len(freq)/8)], 20*np.log(np.abs(spectrum[1].T))[:int(len(freq)/8)], "g", label="G") # get the power spectrum
    plt.plot(freq[:int(len(freq)/8)], 20*np.log(np.abs(spectrum[2].T))[:int(len(freq)/8)], "b", label="B") # get the power spectrum
    plt.legend(loc="upper right")

    plt.show()



bpm = []
for i in range(1,6):
    bpm.append(Calculate(f"./data/direkte{i}")[4])
    print(f"direkte{i} is {bpm[-1]}")

bpm = np.array(bpm).T

print(f"Mean for 'direkte' is:")
for i, c in enumerate(["Red", "Green", "Blue"]):
    print(f"{c}: \t Mean: {np.mean(bpm[i])} BPM, and the std is: {np.std(bpm[i])} BPM")


pD = Calculate(f"./data/{sys.argv[1]}")
Plot(pD[0], pD[1], pD[2], pD[3])