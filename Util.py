import pandas as pd
import numpy as np
import os
from os.path import join
import scipy.io
from multiprocessing import Pool
from subprocess import Popen, PIPE

def save_past_returns(close, T, today):
    pastPeriod = range(today-T, today)
    pastReturns = close.iloc[pastPeriod,1:].pct_change(1)
    return pastReturns[1:]

def save_oos_returns(close, T, today):
    oosPeriod = range(today, today+T)
    oosReturns = close.iloc[oosPeriod,1:].pct_change(1)
    return oosReturns[1:]
	
def retConstShare(retMat, w):
    n, p = retMat.shape
    if len(w.shape) == 1:
        w = np.expand_dims(w, 1)
    wSum1 = w/np.sum(w)
    totalRetMat = 1 + retMat

    cummProdd = np.cumprod(totalRetMat, axis = 0)
    marketValue = np.matmul(cummProdd, wSum1)

    marketValue_m1 = np.concatenate((np.ones((1,1)), marketValue[:(n-1),]))
    PeridoRet = np.divide(marketValue, marketValue_m1)-1
    PeridoRet = PeridoRet * np.sum(w)

    return np.sum(PeridoRet)

def optimal_weights(cov):
	n = cov.shape[0]
	prec = np.linalg.inv(cov)
	denom = np.matmul(np.matmul(np.ones(n), prec), np.ones(n))
	return np.matmul(prec, np.ones(n)) / denom

def annual_analysis(rets):   
	avg = 100 * 12 * np.mean(rets)
	std = 100 * np.sqrt(12)*float(np.std(rets))
	return avg, std, avg/std