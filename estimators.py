import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.covariance
import scipy
import os, pickle
from subprocess import Popen, PIPE
import Util
import uuid

def get_uuid():
    return uuid.UUID(bytes=os.urandom(16), version=4)

def equiweight_cov(X):
    n, p = X.shape
    return np.eye(p)

def glasso_wrapper(X):
    try:
        glasso_method = sklearn.covariance.GraphicalLassoCV(cv=3)
        glasso_method.fit(X)
        return np.linalg.inv(glasso_method.get_precision())
    except:
        return None

def POET_5_wrapper(X):
    uid = get_uuid()
    X = X - np.mean(X, axis = 0)
    in_name = os.path.join(os.getcwd(), "rscripts", "POET_in_{}.npy".format(uid))
    out_name = os.path.join(os.getcwd(), "rscripts", "POET_out_{}.npy".format(uid))
    np.save(in_name, X)
    args = ['Rscript', 'rscripts/POET_script_5facs.R', str(uid)]
    # args = ['rscripts/POET_script_5facs.R']
    p = Popen(args, stdout=PIPE)
    while p.poll() is None:
        # print(p.stdout.readline())
        pass
    cov = np.load(out_name)
    os.remove(in_name)
    os.remove(out_name)
    return cov

def NLS_wrapper(X):
    uid = get_uuid()
    X = X - np.mean(X, axis = 0)
    in_name = os.path.join(os.getcwd(), "rscripts", "NLS_in_{}.npy".format(uid))
    out_name = os.path.join(os.getcwd(), "rscripts", "NLS_out_{}.npy".format(uid))
    np.save(in_name, X)
    args = ['Rscript', 'rscripts/NLS.R', str(uid)]
    p = Popen(args, stdout=PIPE)
    while p.poll() is None:
        # print(p.stdout.readline())
        pass
    NLS = np.load(out_name)
    os.remove(in_name)
    os.remove(out_name)
    return NLS

def sample_cov_wrapper(X):
    try:
        sample_cov_method = sklearn.covariance.EmpiricalCovariance()
        sample_cov_method.fit(X)
        return sample_cov_method.covariance_
    except:
        return None