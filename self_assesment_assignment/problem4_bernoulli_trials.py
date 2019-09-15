#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import matplotlib
import math
matplotlib.use("TkAgg") # without this nothing appears?!
import matplotlib.pyplot as plt
import numpy as np
import sys

matplotlib.rcParams.update({'font.size':30})

def get_bernoulli_sample_mean(n, p):
    """
        Compute and return the mean of 'n' Bernoulli samples with 'p' probability.
    """
    return np.mean(np.random.binomial(1, p, size=n))

def get_epsilon(n, alpha):
    """Compute the epsilon value"""
    return math.sqrt((1/(2*n))*math.log(2/alpha))

def simulate():
    p = 0.4
    alpha = 0.05
    coverages = []
    interval_lengths = []
    sample_sizes = []
    simulations = 1000
    N = 10000
    for n in range(1, N, 100): 
        # print ("n = %d"%(n))
        samples = [get_bernoulli_sample_mean(n, p) for x in range(simulations)]
        epsilon = get_epsilon(n, alpha)
        interval_length = 2*epsilon
        coverage = 0
        for mean in samples:
            # print(sample)
            interval_lower_bound = mean - epsilon
            interval_upper_bound = mean + epsilon
            if interval_lower_bound < p and p < interval_upper_bound:
                coverage = coverage + 1
        coverage = coverage/len(samples)
        # interval_lengths.append(interval_length)
        # if (interval_length < 0.051 and interval_length > 0.0495):
        #     print ("n = %d, interval: %f"%(n, interval_length))
        coverages.append(coverage)
        sample_sizes.append(n)
    plt.plot(sample_sizes, coverages)
    plt.xlabel("N")
    plt.ylabel("Prob.")
    plt.title("Prob. vs N (Coverage)")
    plt.show()

simulate()