import numpy as np
import math as math

"""
f_s = 1000

vec12 = [a, a]
vec13 = [a, a]
vec23 = [a, a]


inp1 = []
inp2 = []
inp3 = []

L_12 = np.correlate(inp1, inp2)
L_13 = np.correlate(inp1, inp3)
L_23 = np.correlate(inp2, inp3)

fors_1 = np.where(L_12 == max(L_12))/f_s #kan ta delt på f_s
fors_2 = np.where(L_13 == max(L_13))/f_s #kan ta delt på f_s
fors_3 = np.where(L_23 == max(L_23))/f_s #kan ta delt på f_s

vec12.append(fors_1)
vec13.append(fors_2)
vec23.append(fors_3)
arr = [vec12, vec13, vec23]
"""

def autocorr(inp):
    arr = np.correlate(inp, inp)
    return np.where(arr == max(arr))

arr = [[-math.sqrt(3)/2, -3/2, 0], [math.sqrt(3)/2, -3/2, 0], [math.sqrt(3), 0, 1]]


def a(arr): #arr på formen [x, y, ct]
    sum = 0
    for element in arr:
        sum += element[0]**2
    return 2*sum
def b(arr):
    sum = 0
    for element in arr:
        sum += element[0]*element[1]
    return 2 * sum
def c(arr):
    sum = 0
    for element in arr:
        sum += element[0]*element[2]
    return 2 * sum
def d(arr):
    sum = 0
    for element in arr:
        sum += element[1]**2
    return 2 * sum
def e(arr):
    sum = 0
    for element in arr:
        sum += element[1]*element[2]
    return 2 * sum

def let(arr):
    return a(arr), b(arr), c(arr), d(arr), e(arr)

A, B, C, D, E = let(arr)

angle = 0
print(C)
print(D)
print(E)
if (C/A != 0):
    angle = np.arctan(-C*D/(A*E))
else:
    angle = 0
if(C/A < 0):
    angle += np.pi



print(angle/np.pi)

