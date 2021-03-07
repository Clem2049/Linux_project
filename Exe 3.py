# Module
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import scipy.stats as si


# Q3
def call_montecarlo(S0, K, T, r, sigma, n_simu):
    """
    :param S0: prix actuel
    :param K: prix strike
    :param T: Maturité
    :param r: taux sans risque
    :param sigma: volatilité
    :param n_simu: nombre de simulation
    :return: prix du call + matrice des prix à chaque instant t
    """

    n_steps = int(T * 252)
    dt = T / n_steps

    z = np.random.normal(0, 1, (n_simu, n_steps))
    var = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z

    st = S0 * np.exp(np.cumsum(var, axis=1))

    call_price = np.exp(-r * T) * np.mean(np.maximum(st[:, -1] - K, 0))

    return call_price, st


# Q4
def confidence_interval(S0, K, T, r, sigma, n_simu):
    """
    :param S0: prix actuel
    :param K: prix strike
    :param T: Maturité
    :param r: taux sans risque
    :param sigma: volatilité
    :param n_simu: nombre de simulation
    :return: intervalle de confiance à 99 pourcents du prix du call
    """
    call, st = call_montecarlo(S0, K, T, r, sigma, n_simu)[0], call_montecarlo(S0, K, T, r, sigma, n_simu)[1]
    error = si.norm.ppf(0.99) * np.std(st) / n_simu

    interval = "[{price} +- {erreur}]".format(price=round(call, 2), erreur=round(error, 4))

    return interval


# Q5
def blackscholes_call(S0, K, T, r, sigma):
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

    call = (S0 * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))

    return call


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

print(confidence_interval(100, 100, 0.5, 0.01, 0.2, 10000))