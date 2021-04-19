#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import math as m

hpCutoff_freq = 100
lpCutoff_freq = 10000

upsampleFac = 4
corrLen = 10
c = 343

d = 0.06

mic1 = d * np.array([1/2, np.sin(m.pi / 3)])
mic2 = d * np.array([0,0])
mic3 = d * np.array([1,0])


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
        data = data.T
        # Resample signals
        data = [signal.resample(d, len(d) * upsampleFac) for d in data]
        sample_period /= upsampleFac
        sample_period *= 1e-6
    return sample_period, np.array(data)


def butter_coeff(cutoff, fs, fType, order=6):
    """
    Finds the coeffitients to a butterworth filter
    with order `order`, a cutoff frequency `cutoff`,
    samplerate `fs` and what `type` it should be.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return signal.butter(order, normal_cutoff, btype=fType, analog=False)


def butter_filter(dataPoints, cutoff, fs, fType, order=6):
    """ 
    Filters `dataPoints` with a butterfilter with a cutoff frequency `cutoff`
    with samplerate `fs` and `type` with order `order`.
    """ 
    b, a = butter_coeff(cutoff, fs, fType, order=order)
    return signal.filtfilt(b, a, dataPoints)


def FindLagAndCorr(data1, data2):
    """ 
    Returns the result of a correlation between `data1` and `data2`.
    """
    corr = np.correlate(data1[corrLen * upsampleFac:-corrLen * upsampleFac], data2, mode="valid")
    lagAxis = np.linspace(int(-len(corr) / 2), int(len(corr) / 2), num=len(corr))
    l = int(lagAxis[np.where(corr == max(corr))])
    return corr, lagAxis, l


def FindAngle(tau):
    """
    Given the time delays `tau`, it calculates the angle for the incomming signal.
    """
    micMatrix = np.array([mic2 - mic1, mic3 - mic1, mic3 - mic2])

    x_vec = (-c * (np.linalg.pinv(micMatrix) @ tau))

    return np.angle(x_vec[0] + x_vec[1]*1j, deg=True)


def CalculateAngle(filename):
    """
    Caculates an angle from datapoints in file `filename`.
    """
    # Import data from bin file
    sample_period, data = raspi_import(filename)

    # Find sample freq from file
    sample_freq = 1 / sample_period

    data = butter_filter(data, hpCutoff_freq, sample_freq, 'high', 6)


    _xcorr21, _axis21, l21  = FindLagAndCorr(data[1], data[0])
    _xcorr31, _axis31, l31  = FindLagAndCorr(data[2], data[0])
    _xcorr32, _axis32, l32  = FindLagAndCorr(data[2], data[1])
    acorr, aaxis, la        = FindLagAndCorr(data[0], data[0])

    return FindAngle([l21, l31, l32])


fourAngles = []
for d in [10, 90, 170, 270]:
    results = []
    for n in range(10):
        results.append(CalculateAngle(f"./{d}/{n}.bin"))
    
    mean = np.mean(results)
    s = np.std(results)

    if mean < 0:
        mean += 360

    fourAngles.append([d, results, mean, s])
    print(f"Angle: {d}, mean of measures: {mean}, std of measures: {s}")

tenAngles = []
for file in ["000","036", "072","108","144","180","216","252","288","324"]:
    results = []
    for n in range(10):
        results.append(CalculateAngle(f"./tenAngles/{file}/{n}.bin"))

    mean = np.mean(results)
    s = np.std(results)

    if mean < 0:
        mean += 360

    tenAngles.append([file, results, mean, s])

    print(f"Angle: {file}, mean of measures: {mean}, std of measures: {s}")
