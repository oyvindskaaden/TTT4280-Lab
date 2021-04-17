import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import math as m

v_ref = 3.3

resolution = 2**12

hpCutoff_freq = 100
lpCutoff_freq = 10000

corrLen = 10
upsampleFac = 4

a = 0.035

mic1 = np.array([0, 1]) * a
mic2 = np.array([-m.sqrt(3)/2, -0.5]) * a
mic3 = np.array([m.sqrt(3)/2, -0.5]) * a

upsampleFac = 4

fs = 31250 * upsampleFac

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
    data = data.T
    data = [signal.resample(d, len(d) * upsampleFac) for d in data]
    sample_period /= upsampleFac
    sample_period *= 1e-6
  return sample_period, np.array(data)


def butter_coeff(cutoff, fs, fType, order=6):
  nyq = 0.5 * fs
  normal_cutoff = cutoff / nyq
  return signal.butter(order, normal_cutoff, btype=fType, analog=False)


def butter_filter(dataPoints, cutoff, fs, fType, order=6):
  b, a = butter_coeff(cutoff, fs, fType, order=order)
  return signal.filtfilt(b, a, dataPoints)


def FindLagAndCorr(data1, data2):
  corr = np.correlate(data1[corrLen * upsampleFac:-corrLen * upsampleFac], data2, mode="valid")
  lagAxis = np.linspace(int(-len(corr) / 2), int(len(corr) / 2), num=len(corr))
  l = int(lagAxis[np.where(corr == max(corr))])
  return corr, lagAxis, l


def micVector(fromMic, toMic):
  return fromMic - toMic


def FindAngle(tau):
  micMatrix = np.array([mic2 - mic1, mic3 - mic1, mic3 - mic2])

  x_vec = (-c * (np.linalg.pinv(micMatrix) @ tau))

  return np.angle(x_vec[0] + x_vec[1]*1j, deg=True)  #(np.arctan(Xvec[1]/Xvec[0]) * 180 / m.pi ) + correction


def Calculate(filename):
  # Import data from bin file
  sample_period, data = raspi_import(filename)

  sample_freq = 1 / sample_period


  #sample_period *= 1e-6  # change unit to micro seconds
  data = data * (v_ref / resolution) # Change to volts from mV


  data = butter_filter(data, hpCutoff_freq, sample_freq, 'high', 6)
  #data = butter_filter(data, lpCutoff_freq, sample_freq, 'low', 6)


  xcorr21, axis21, l21 = FindLagAndCorr(data[1], data[0])
  xcorr31, axis31, l31 = FindLagAndCorr(data[2], data[0])
  xcorr32, axis32, l32 = FindLagAndCorr(data[2], data[1])
  acorr, aaxis, la = FindLagAndCorr(data[0], data[0])



  #print(f"{l21}, {l31}, {l32}, auto: {la}")

  return FindAngle([l21, l31, l32])


for d in [10, 90, 170, 270]:
  results = []
  for n in range(10):
    #print(f"./180/{n}.bin")
    results.append(Calculate(f"./{d}/{n}.bin"))
  
  mean = np.mean(results)
  s = np.std(results)

  print(f"Angle: {d}, mean of measures: {mean}, std of measures: {s}")

for file in ["000","036", "072","108","144","180","216","252","288","324"]:
  results = []
  for n in range(10):
    results.append(Calculate(f"./tenAngles/{file}/{n}.bin"))
    
    #print(f"Angle {n}: {result}")
  mean = np.mean(results)
  s = np.std(results)

  if mean < 0:
    mean += 360

  print(f"Angle: {file}, mean of measures: {mean}, std of measures: {s}")
  

"""
# Generate time axis
num_of_samples = data.shape[1]  # returns shape of matrix
t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

# Generate frequency axis and take FFT
freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
spectrum = np.fft.fft(data)[:3] # takes FFT of all channels

"""

"""
plt.figure("Correlation")
plt.subplot(4,1,1)
plt.stem(axis21, xcorr21)
plt.subplot(4,1,2)
plt.stem(axis31, xcorr31) 
plt.subplot(4,1,3)
plt.stem(axis32, xcorr32)
plt.subplot(4,1,4)
plt.stem(aaxis, acorr)

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



plt.show()"""
