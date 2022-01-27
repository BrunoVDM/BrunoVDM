# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import math
import time
import nevergrad as ng


# +
#Define a sigmoid function
def sigmoid(x):
    s = 1 / (1 + np.exp(-np.array(x)))
    return s

#Define a weights function
def weights(x):
    x=list(np.round(sigmoid(x)) * sigmoid(x))
    if np.sum(x) > 0:
        return x/np.sum(x)
    else:
        return x
#    return result


# +
OIL_WELLS=[10,20,30,40,50,60,70,80,90,100]
GOR=OIL_WELLS
GAZ_MAX=10000

def GAZ_TOTAL(x):
    return np.sum(np.array(OIL_WELLS) * np.array(x)*np.array(GOR))

def function(x):
    y=sigmoid(x)
    #x=weights(x)
    GAZ_TOTAL=np.sum(np.array(OIL_WELLS) * np.array(y)*np.array(GOR))
    OIL_TOTAL=np.sum(np.array(OIL_WELLS) * np.array(y))*int(GAZ_TOTAL<GAZ_MAX)
    return -OIL_TOTAL

def print_candidate_and_value(optimizer, candidate, value):
    print(candidate, value)

budgets = [10,30,100,300,1000,3000,10_000]
for budget in  budgets:
    for j in range(1):
        tic=time.clock()
        optimizer = ng.optimizers.OnePlusOne(parametrization=10, budget=budget)
        recommendation = optimizer.minimize(function)
        toc=time.clock()
        print(-function(recommendation.value),budget,toc-tic)


# +
def function(x):
    x = np.tanh(x)
    #return sum(math.floor((x)**2))
    PORV = x[0]
    KY34F19 = x[1]
    KX34F15 = x[2]
    KVKH3 = x[3]
    TM1 = x[4]
    TM23 = x[5]
    TM45 = x[6]
    NW4 = x[7]
    KRW4 = x[8]
    return abs(0.58-(0.523281528+PORV*-0.06059188+KY34F19*0.307888575+KX34F15*-0.039980579+KVKH3*0.001237663+TM1*0.003743428+TM23*0.003400026+TM45*-0.033788989+NW4*-2.54E-02+KRW4*5.89E-02+PORV*KY34F19*1.75E-02+PORV*KX34F15*-0.003934512+PORV*KVKH3*-0.001946354+PORV*TM1*-0.001727694+PORV*TM23*-3.57E-03+PORV*TM45*-0.00273258+PORV*NW4*0.00496501+PORV*KRW4*0.00287984+KY34F19*KX34F15*0.009306995+KY34F19*KVKH3*-2.16E-03+KY34F19*TM1*-0.001265603+KY34F19*TM23*0.00744823+KY34F19*TM45*6.77E-03+KY34F19*NW4*0.010566569+KY34F19*KRW4*0.013717374+KX34F15*KVKH3*0.00770155+KX34F15*TM1*-5.60E-03+KX34F15*TM23*4.83E-03+KX34F15*TM45*-0.017304812+KX34F15*NW4*3.18E-03+KX34F15*KRW4*-0.008828259+KVKH3*TM1*0.002897359+KVKH3*TM23*-0.009431151+KVKH3*TM45*0.005215089+KVKH3*NW4*0.000937631+KVKH3*KRW4*-0.005352293+TM1*TM23*-0.004604821+TM1*TM45*-0.000909857+TM1*NW4*-0.006206492+TM1*KRW4*0.000776944+TM23*TM45*-0.011999925+TM23*NW4*0.001232013+TM23*KRW4*0.003567179+TM45*NW4*0.005053602+TM45*KRW4*-0.00465981+NW4*KRW4*0.00052099+PORV**2*0.004043209+KY34F19**2*-0.199821369+KX34F15**2*-3.88E-04+KVKH3**2*-1.15E-02+TM1**2*0.00342974+TM23**2*-0.022645022+TM45**2*0.001174215+NW4**2*-3.57E-03+KRW4**2*-0.004359981))
    #return sum((x - .5)**2)/len(x)

def print_candidate_and_value(optimizer, candidate, value):
    print(candidate, value)

#size = [30,100,300]
for i in  range(1):
    for j in range(1):
        tic=time.clock()
        optimizer = ng.optimizers.OnePlusOne(parametrization=9, budget=10**3)
        # define a constraint on all variables of x:
        #optimizer.parametrization.register_cheap_constraint(lambda x: np.amax(x) <= 0.2)
        #optimizer.register_callback("tell", print_candidate_and_value)
        recommendation = optimizer.minimize(function)
        toc=time.clock()
        print(function(recommendation.value),toc-tic,i,j)


# +
def function(x):
    #return sum(math.floor((x)**2))
    return sum((sigmoid(x) - .5)**2)/len(np.array(x))

def print_candidate_and_value(optimizer, candidate, value):
    print(candidate, value)

size = [30,100,300]
for i in size:
    for j in range(4):
        tic=time.clock()
        optimizer = ng.optimizers.OnePlusOne(parametrization=i**2, budget=10**j)
        recommendation = optimizer.minimize(function)
        toc=time.clock()
        print(function(recommendation.value),toc-tic,i,j)
# -


