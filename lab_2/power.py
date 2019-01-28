# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:40:55 2019

@author: oa18525
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np

def valor_p(sample1, sample2, reps, size, tobs):
    
    c=0
    sample = np.concatenate((sample1, sample2))
    
    for it in range(reps):
        s = np.random.choice(sample, size=2*size, replace=False)
       
        s1 = s[:size]
        s2 = s[size:]
        
        tperm = s2.mean - s1.mean
        
        if tperm > tobs:
            c += 1
    
    return c/reps

def power(sample1, sample2, reps, size, alpha):
    tobs = sample2.mean() - sample1.mean()
    k = 0
    
    for it in range(reps):
        s1 = np.random.choice(sample1, size=size, replace=True)
        s2 = np.random.choice(sample2, size=size, replace=True)
        p = valor_p(s1, s2, reps, size, tobs)
        
        if p < (1-alpha):
            k += 1
            
    return k/reps


