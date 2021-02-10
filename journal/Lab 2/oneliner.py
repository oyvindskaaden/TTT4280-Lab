from matplotlib.pyplot import xcorr
import numpy as np
import math as m

from numpy.core.numeric import correlate

soundVelocity = 343

a = 0.035

mic1 = np.array([0, 1]) * a
mic2 = np.array([-m.sqrt(3)/2, -0.5]) * a
mic3 = np.array([m.sqrt(3)/2, -0.5]) * a

upsampleFac = 4

fs = 31250 * upsampleFac
c = 343

#3,-2,-5 20degree
# -5,0,1 230 degree
tau = np.array([14, -9, -22]) / fs

print(tau)
#tau = 1e-3*np.array([0.386367 , -0.087589 , -0.473957])

print(np.shape(tau))

def micVector(fromMic, toMic):
  return [fromMic[0] - toMic[0], fromMic[1] - toMic[1]]

micMatrix = np.array([micVector(mic2, mic1), micVector(mic3, mic1), micVector(mic3, mic2)])

print(np.transpose(micMatrix))

inverseMicMatrix = np.linalg.pinv(micMatrix)

Xvec = (-soundVelocity * (inverseMicMatrix @ tau))

print(Xvec)
correction = 0
if Xvec[0] < 0:
  correction = 180

angle = (np.arctan(Xvec[1]/Xvec[0]) * 180 / m.pi ) + correction

#angle = np.angle(-soundVelocity * np.linalg.pinv(np.array([micVector(mic2, mic1), micVector(mic3, mic1), micVector(mic3, mic2)]) @ tau))

print(angle)