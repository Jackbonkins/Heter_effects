import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures 


class SimulationStudy:
    '''

    Class creates simulation study object with specified simulation parameters


    Attributes
    ----------
    p : int
        Number of covariates/complexity for the data set
    mean_correlation : float
        Average feature correlation on the data set
    n : int
        Observations in the data set    

    '''

    def __init__(self, p: int, mean_correlation: float, n: int):
        self.p = p
        self.mean_correlation = mean_correlation
        self.n = n



    def get_covariance_matrix(self) -> np.ndarray:
        ''' 
        
        Get n*n covariance matrix 
        
        '''
        mean = np.zeros(self.p)
        variance_diagonal = np.ones(self.p) # in principle, n could be n draws from some meta-distribution
        cov_matrix = np.zeros((self.p, self.p))       
        # assign variance to diagonal
    
        while True:
            try:
                for i in range(self.p): #columns
                    cov_matrix[i, i] = variance_diagonal[i]

                    for j in range(i+1, self.p):
                        correlation = np.clip(np.random.normal(self.mean_correlation, 0.01), -1, 1)
                        #print(f"Correlation: {correlation}")
                        cov_matrix[i, j] = cov_matrix[j, i] = correlation *  np.sqrt(variance_diagonal[i] * variance_diagonal[j])

                pos_def_test = np.linalg.cholesky(cov_matrix)    

            except np.linalg.LinAlgError:
                continue
    
            break

        return cov_matrix, mean
       


    def get_dataframe(self, cov_matrix: np.ndarray, mean: np.ndarray) -> pd.DataFrame:
        rng = np.random.default_rng()
        multivariate_samples = rng.multivariate_normal(mean, cov_matrix, self.n)
        df = pd.DataFrame(multivariate_samples, columns=[f"X{i}" for i in range(self.p)])
        return df



    def gen_mu_x(self, df: pd.DataFrame) -> pd.DataFrame:

        feat_no = int(self.p/2)

        columns = [f"X{i}" for i in range(feat_no)]
        poly = PolynomialFeatures(interaction_only=True)

        poly_features = poly.fit_transform(df[columns])
        sum_poly_features = poly_features.sum(axis=1)

        df['mu_x'] = sum_poly_features + np.random.normal(0, 1, self.n)

        return df
    


    def gen_cate(self, degree: int, df: pd.DataFrame) -> pd.DataFrame: 
    
        feat_no = int(self.p/2)

        columns = [f"X{i}" for i in range(feat_no)]
        poly = PolynomialFeatures(degree)
        poly_features = poly.fit_transform(df[columns])

        # Sum the polynomial features along axis 1
        sum_poly_features = poly_features.sum(axis=1)

        # Add the new variable to DataFrame
        df['T'] = np.random.binomial(1, 0.5, len(df))
        df['CATE'] = (sum_poly_features + np.random.normal(0, 1, self.n))*df['T']

        return df
