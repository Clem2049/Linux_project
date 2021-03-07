# Module
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import scipy.stats as si


# Q1
def call_antithetic_variate(S0, K, T, r, sigma, n_simu):
    """
    :param S0: prix actuel
    :param K: prix strike
    :param T: Maturité
    :param r: taux sans risque
    :param sigma: volatilité
    :param n_simu: nombre de simulation
    :return: prix du call
    """

    n_steps = int(T * 252)
    dt = T / n_steps

    z1 = np.random.normal(0, 1, (n_simu, n_steps))
    z2 = - z1

    var1 = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z1
    var2 = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z2

    st1 = S0 * np.exp(np.cumsum(var1, axis=1))
    st2 = S0 * np.exp(np.cumsum(var2, axis=1))

    # On divise par 2 car il y a 2 fois plus de payoffs
    call_price = np.exp(-r * T) * np.mean(np.maximum(st1[:, -1] - K, 0) + np.maximum(st2[:, -1] - K, 0)) / 2

    return call_price


# Q2
def blackscholes_put(S0, K, T, r, sigma):
    """
    :param S0: prix actuel
    :param K: prix strike
    :param T: maturité
    :param r: taux sans risque
    :param sigma: volatilité
    :return: prix du call
    """

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S0 * si.norm.cdf(-d1, 0.0, 1.0))

    return put


def call_control_variates(S0, K, T, r, sigma, n_simu):
    """
    :param S0: prix actuel
    :param K: prix strike
    :param T: Maturité
    :param r: taux sans risque
    :param sigma: volatilité
    :param n_simu: nombre de simulation
    :return: prix du call
    """

    n_steps = int(T * 252)
    dt = T / n_steps

    # Prix d'un put ATM avec la méthode de Black Scholes
    p_bs = blackscholes_put(S0, K, T, r, sigma)

    z = np.random.normal(0, 1, (n_simu, n_steps))
    var = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z

    st = S0 * np.exp(np.cumsum(var, axis=1))

    call_price = np.exp(-r * T) * np.mean(np.maximum(st[:, -1] - K, 0))

    return call_price, st


print(call_antithetic_variate(100, 100, 0.5, 0.01, 0.2, 10000))