import numpy as np
import math as m

soundVelocity = 343

mic1 = [0, 1]
mic2 = [-m.sqrt(3)/2, -0.5]
mic3 = [m.sqrt(3)/2, -0.5]

tau = 1e-3*np.array([0.386367 , -0.087589 , -0.473957])

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