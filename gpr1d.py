##### gpr1d.py #####

import numpy as np
import pandas as pd


##### EXPONENTIAL METHOD #####
def kexp(xj_diff, theta):
    return np.exp(-0.5*(xj_diff**2)/(theta**2))


##### COVARIANCE MATRICES #####
def K_deriv_construct(x, theta, chi, sigma_noise):

    ### Construct zero matrix K
    K = np.zeros((x.shape[0], x.shape[0]))

    ### Populating K
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            K[i,j] = (chi**2.0)*kexp(x[i]-x[j], theta)*(((theta**2.0)-((x[i]-x[j])**2.0))/(theta**4.0))

            ## If i == j, add in the noise
            if i == j:
                K[i,j] += sigma_noise**2.0

    return K


def K_construct(x, theta, chi, sigma_noise):

    ### Construct zero matrix K
    K = np.zeros((x.shape[0], x.shape[0]))

    ### Populating K
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            K[i,j] = (chi**2.0)*kexp(x[i]-x[j], theta)

            ## If i == j, add in the noise
            if i == j:
                K[i,j] += sigma_noise**2.0

    return K


##### TEST VECTORS #####
def k_deriv_test(x_star, x, theta, chi):

    ### Construct zero vector k
    k = np.zeros(x.shape[0])

    ### Populating k
    for i in range(x.shape[0]):
        k[i] = (chi**2.0)*kexp(x_star-x[i], theta)*((x_star-x[i])/(theta**2.0))

    return k


def k_test(x_star, x, theta, chi):

    ### Construct zero vector k
    k = np.zeros(x.shape[0])

    ### Populating k
    for i in range(x.shape[0]):
        k[i] = (chi**2.0)*kexp(x_star-x[i], theta)

    return k


##### INVERTOR #####
def solver(K, y):

    #### Solve L(y_dummy) = y #####
    L = np.linalg.cholesky(K)
    y_dummy = np.linalg.solve(L, y)

    v = np.linalg.solve(L.T, y_dummy)

    return v


##### FITTER #####
def fitter(v, k_test):
    return np.dot(v, k_test)
