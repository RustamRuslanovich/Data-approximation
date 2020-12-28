#!/usr/bin/env python
# coding: utf-8
import sys, os
import matplotlib.pyplot as plt
import numpy as np
import emcee
import json
from scipy.optimize import minimize
from warnings import warn


def open_fl1(path):
    with open(os.path.join(path, 'flux for2.txt')) as fr:
        fl1 = np.array(json.load(fr))
    return fl1


def open_fl2(path):
    with open(os.path.join(path, 'flux for8.txt')) as fr:
        fl2 = np.array(json.load(fr))
    return fl2


def open_time2(path):
    with open(os.path.join(path, 'time2.txt')) as fr:
        time2 = np.array(json.load(fr))
    return time2


def open_time8(path):
    with open(os.path.join(path, 'time8.txt')) as fr:
        time8 = np.array(json.load(fr))
    return time8


def open_position1(path):
    with open(os.path.join(path, 'position of core for2.txt')) as fr:
        position1 = np.array(json.load(fr))
    return position1


def open_position2(path):
    with open(os.path.join(path, 'position of core for8.txt')) as fr:
        position2 = np.array(json.load(fr))
    return position2


def log_likelihood(theta, path='../../'):
    """Returning logarithm of likelihood function
    path - path to your model data"""
    #files = ['fl1', 'fl2', 'position1', 'position2', 'time2', 'time8']
    a, b1, b2, k = theta
    fl1 = open_fl1(path=path)
    fl2 = open_fl2(path=path)
    position1 = open_position1(path=path)
    position2 = open_position2(path=path)
    crsh = position1 - position2
    inaccuracy_crshf = crsh * 0.1

    model = a + b1*fl1**k + b2*fl2**k
    sigma2 = (inaccuracy_crshf)**2
    return -0.5 * np.sum((crsh - model) ** 2 / sigma2 + np.log(sigma2))


def mle(a_true=0, b1_true=1, b2_true=1, k_true=0.28, first_freq='2', second_freq='8', path='../../'):
    """Finding numerical optimum of likelihood function"""
    np.random.seed(42)
    nll = lambda *args: -log_likelihood(*args, path=path)
    initial = np.array([a_true, b1_true, b2_true, k_true]) + 0.1 * np.random.randn(4)
    soln = minimize(nll, initial)

    if not soln.success:
        warn('Отимизация не удалась :(')
    return soln


def plotting(path='../../'):
    a_ml, b1_ml, b2_ml, k_ml = mle(path=path).x
    print("Maximum likelihood estimates:")
    print("a = {0:.3f}".format(a_ml))
    print("b1 = {0:.3f}".format(b1_ml))
    print("b2 = {0:.3f}".format(b2_ml))
    print("k = {0:.3f}".format(k_ml))
    fl1 = open_fl1(path=path)
    fl2 = open_fl2(path=path)
    position1 = open_position1(path=path)
    position2 = open_position2(path=path)
    crsh = position1 - position2
    inaccuracy_crshf = crsh * 0.1
    freqs = {'8': fl2, '2': fl1}
    for key in freqs:
        plt.errorbar(freqs[key], crsh, yerr=inaccuracy_crshf, fmt=".k", capsize=0)
        plt.plot(freqs[key], a_ml + b1_ml*fl1**k_ml + b2_ml*fl2**k_ml, ":k", label="ML")
        plt.legend(fontsize=14)
        plt.xlabel(f'flux for {key} GHz')
        plt.ylabel('coreshift')
        plt.show()


def log_prior(theta, path='../../'):
    """This function encodes any previous knowledge that we have
     about the parameters: results from other experiments,
      physically acceptable ranges, etc."""
    a, b1, b2, k = theta
    if -1.0 < a < 1. and -10. < b1 < 10.0 and -10. < b2 < 10.0 and 0.23 < k < 0.3:
        return 0.0
    return -np.inf


def log_probability(theta, fl1, fl2, crsh, inaccuracy_crshf, path='../../'):
    """Combining log_prior with the definition of log_likelihood from above.
     Returns full log-probability function"""
    lp = log_prior(theta, path=path)
    fl1 = open_fl1(path=path)
    fl2 = open_fl2(path=path)
    position1 = open_position1(path=path)
    position2 = open_position2(path=path)
    crsh = position1 - position2
    inaccuracy_crshf = crsh * 0.1
    return lp + log_likelihood(theta, path=path)


def final_fitting(path='../../'):
    """Makes all procedures and fits data"""
    #mle()
    fl1 = open_fl1(path=path)
    fl2 = open_fl2(path=path)
    position1 = open_position1(path=path)
    position2 = open_position2(path=path)
    crsh = position1 - position2
    inaccuracy_crshf = crsh * 0.1
    pos = mle(path=path).x + 1e-4 * np.random.randn(32, 4)
    nwalkers, ndim = pos.shape
    #print(mle())
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(fl1, fl2, crsh, inaccuracy_crshf, path))
    sampler.run_mcmc(pos, 500, progress=True)
    tau = sampler.get_autocorr_time()
    print(tau)
    return tau


def main():
    path = sys.argv[1]
    final_fitting(path=path)
    plotting(path=path)
    
if __name__ == '__main__':
    main()

