import numpy as np
import pandas as pd

#Generate covariance matrix
def gen_cov_matrix(p):

    P = p # number of features/variables in the data set (complexity)
    a = np.random.rand(P, P)
    m = np.tril(a) + np.tril(a, -1).T
    B = np.dot(m, m.transpose())

    cov_matrix = pd.DataFrame(B).cov()

    return cov_matrix


#Check whether the covariance matrix is positive definite
def positive_definite(matrix):
    sign, determinant = np.linalg.slogdet(matrix)
    
    if sign <= 0:
        print('Matrix is not positive definite')
    else:
        print('Matrix is positive definite')



# Generate covariates from a normal distribution using the covariance matrix
def gen_covariates(p, cov_matrix, N):
       
    mean=np.zeros(p, dtype=int)
    X = np.random.default_rng().multivariate_normal(mean, cov_matrix, N)
    covariates = pd.DataFrame(X)
    covariates.columns=["X_"+str(i) for i in range(p)]

    return covariates

