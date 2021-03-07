# Module
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import scipy.stats as si


# Q2
def Box_Muller():
    """
    :return: Box-Muller algorithm generate normal variables
    """
    u1 = rd.random()
    u2 = rd.random()
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    return z1, z2


# Q5
def Modified_Box_Muller():
    """
    :return: Box-Muller algorithm generate normal variables
    """
    z = 10
    while z > 1:
        u1 = (rd.random.uniform(-1, 1))
        u2 = (rd.random.uniform(-1, 1))
        z = u1 * u1 + u2 * u2
    r = -2 * np.log(z)
    x1 = np.sqrt(r / z) * u1
    x2 = np.sqrt(r / z) * u2
    return x1, x2


