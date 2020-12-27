#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import emcee
from scipy.optimize import minimize


def open_files(path='../../'):
    '''Opening files of jet's model.
    Returning fluxes, epoches, coreshift
    path - write your directory'''
    files = {'fl1':'flux for2', 'fl2':'flux for8', 'time2':'time2', 'time8':'time8',
             'position1':'position of core for2', 'position2':'position of core for8'}

    for key in files:
        with open(path + globals()[files[key]] + '.txt') as fr:
            global globals()[files[key]]
            globals()[files[key]] = np.array(fr.readlines())

    global crsh
    crsh = position1 - position2
    global inaccuracy_crshf
    inaccuracy_crshf = crsh*0.1
    return crsh, inaccuracy_crshf, fl1, fl2, time2, time8


def log_likelihood(theta, path='../../'):
    """Returning logarithm of likelihood function
    path - path to your model data"""
    open_files(path=path)
    a, b1, b2, k = theta
    model = a + b1*fl1**k + b2*fl2**k
    sigma2 = inaccuracy_crshf**2
    return -0.5 * np.sum((crsh - model) ** 2 / sigma2 + np.log(sigma2))


def mle(a_true=0, b1_true=1, b2_true=1, k_true=0.28, first_freq='2', second_freq='8'):
    """Finding numerical optimum of likelihood function"""
    np.random.seed(42)
    nll = lambda *args: -log_likelihood(*args)
    initial = np.array([a_true, b1_true, b2_true, k_true]) + 0.1 * np.random.randn(4)
    global soln
    soln = minimize(nll, initial)
    a_ml, b1_ml, b2_ml, k_ml = soln.x
    print("Maximum likelihood estimates:")
    print("a = {0:.3f}".format(a_ml))
    print("b1 = {0:.3f}".format(b1_ml))
    print("b2 = {0:.3f}".format(b2_ml))
    print("k = {0:.3f}".format(k_ml))
    freqs = {'8': fl2, '2': fl1}
    for key in freqs:
        plt.errorbar(freqs[key], crsh, yerr=inaccuracy_crshf, fmt=".k", capsize=0)
        plt.plot(freqs[key], a_ml + b1_ml*fl1**k_ml + b2_ml*fl2**k_ml, ":k", label="ML")
        plt.legend(fontsize=14)
        plt.xlabel('flux for '+str(key) + 'GHz')
        plt.ylabel('coreshift')
        plt.show()
    return soln


def log_prior(theta):
    """This function encodes any previous knowledge that we have
     about the parameters: results from other experiments,
      physically acceptable ranges, etc."""
    a, b1, b2, k = theta
    if -1.0 < a < 1. and -10. < b1 < 10.0 and -10. < b2 < 10.0 and 0.23 < k < 0.3:
        return 0.0
    return -np.inf


def log_probability(theta, fl1, fl2, crsh, inaccuracy_crshf):
    """Combining log_prior with the definition of log_likelihood from above.
     Returns full log-probability function is:"""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, fl1, fl2, crsh, inaccuracy_crshf)


def final_fitting():
    """Makes all procedures and fits data"""
    mle()
    pos = soln.x + 1e-4 * np.random.randn(32, 4)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(fl1, fl2, crsh, inaccuracy_crshf))
    sampler.run_mcmc(pos, 5000, progress=True)
    tau = sampler.get_autocorr_time()
    print(tau)
    return tau

def main():
    final_fitting()
    
    
if __name__ == '__main__':
    main()
