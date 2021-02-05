import numpy as np
import math as m

soundVelocity = 343

a = 0.035

mic1 = np.array([0, 1]) * a
mic2 = np.array([-m.sqrt(3)/2, -0.5]) * a
mic3 = np.array([m.sqrt(3)/2, -0.5]) * a

fs = 31250
c = 343

#3,-2,-5
tau = np.array([-5,0,1]) / fs

print(tau)
#tau = 1e-3*np.array([0.386367 , -0.087589 , -0.473957])

print(np.shape(tau))

def micVector(fromMic, toMic):
  return [fromMic[0] - toMic[0], fromMic[1] - toMic[1]]

micMatrix = np.array([micVector(mic2, mic1), micVector(mic3, mic1), micVector(mic3, mic2)])

print(np.transpose(micMatrix))

inverseMicMatrix = np.linalg.pinv(micMatrix)

print(-soundVelocity * (inverseMicMatrix @ tau))
Xvec = (-soundVelocity * (inverseMicMatrix @ tau))

angle = np.arctan(Xvec[1]/Xvec[0]) * 180 / m.pi

#angle = np.angle(-soundVelocity * np.linalg.pinv(np.array([micVector(mic2, mic1), micVector(mic3, mic1), micVector(mic3, mic2)]) @ tau))

print(angle)