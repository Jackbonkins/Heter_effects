import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures 


class SimulationStudy:

    '''

    SimulationStudy class creates simulation study object with specified parameters by performing the following steps:

    1) Generate a p*p flexible positive-definite covariance matrix
    2) Use the generated covariance matrix to draw p covariates from a multivariate normal distribution with n observations
    3) Create Mu(x) function
    4) Create CATE
    5) Generate outcome variable y
    6) Create final data set


    Attributes
    ----------
    p : int
        Number of covariates/complexity for the data set
    mean_correlation : float
        Average feature correlation on the data set
    n : int
        Observations in the data set    
    poly_degree: int
        Sets the polynomial degree for the Polynomial Feature function 
    

    '''


    def __init__(self, p: int, mean_correlation: float, n: int, poly_degree: int):
        self.p = p
        self.mean_correlation = mean_correlation
        self.n = n
        self.poly_degree = poly_degree



    def get_covariance_matrix(self) -> np.ndarray:
        ''' 
        
        Get p*p covariance matrix. Takes parameter self and returns covariance and mean matrix.

        If covariance matrix is not positive-definite, cholesky decomposition is not possible. In which case, LinAlgError occurs. While loop avoids error
        due to random value choices by choosing a different set of numbers.
        
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

                pos_def_test = np.linalg.cholesky(cov_matrix) #Test covariance matrix for symmetry and positive-definiteness

            except np.linalg.LinAlgError:
                continue
    
            break

        return cov_matrix, mean
       


    def get_features(self, cov_matrix: np.ndarray, mean: np.ndarray) -> pd.DataFrame:

        '''
        Uses self parameter, the covariance matrix, and the mean matrix to draw p features from a multivariate normal distribution with n observations. 

        Method returns a pandas dataframe.
       
        '''
        rng = np.random.default_rng()
        multivariate_samples = rng.multivariate_normal(mean, cov_matrix, self.n)
        df_original = pd.DataFrame(multivariate_samples, columns=[f"X{i}" for i in range(self.p)])
        return df_original



    def gen_mu_x(self, df: pd.DataFrame) -> pd.DataFrame:

        feat_no = int(self.p/2)

        columns = [f"X{i}" for i in range(feat_no)]
        poly = PolynomialFeatures(interaction_only=True)

        poly_features = poly.fit_transform(df[columns])
        interaction_sum = pd.DataFrame(np.sum(poly_features, axis=1) - np.sum(df[columns].values, axis=1), columns=['sum'])

        df['mu_x'] =  interaction_sum

        return df
    


    def gen_cate(self, poly_degree: int, df: pd.DataFrame, geom: bool = False) -> pd.DataFrame: 
    
        feat_no = int(self.p/2)
        columns = [f"X{i}" for i in range(feat_no)]



        poly = PolynomialFeatures(poly_degree, include_bias=False)
        interactions = PolynomialFeatures(interaction_only=True, include_bias=False)

        poly_features = poly.fit_transform(df[columns])
        interaction_features = interactions.fit_transform(df[columns])

        # Sum polynomial features and subtract redundant values      
        sum_poly_features = pd.DataFrame(np.sum(poly_features, axis=1) - np.sum(df[columns].values, axis=1) - np.sum(df[columns].values, axis=1), columns=['sum'])

        '''
        To do: Random feature weights 
        
        values between 0 and 1 (Normally distributed? Uniformly distributed?)


        '''

        #if geom == True:
            
        
        # Add the new variable to DataFrame
        df['T'] = np.random.binomial(1, 0.5, len(df))
        df['CATE'] = (sum_poly_features['sum'] + np.random.normal(0, 1, self.n))*df['T']

        return df

    #def gen_random_feat_weight(self, df: pd.DataFrame) -> pd.DataFrame:
     #   df_columns = df.columns
       # weights = (np.random.randint(0, 100, ))/100


        
    def gen_outcome(self, df: pd.DataFrame) -> pd.DataFrame:

        df['y'] = df['CATE'] + df['mu_x'] + np.random.normal(0, 1, self.n)
        return df



    def create_dataset(self):

        cov_matrix, mean = self.get_covariance_matrix()
        df = self.get_features(cov_matrix=cov_matrix, mean=mean)
        df_mu_x = self.gen_mu_x(df=df)
        df_cate = self.gen_cate(df=df_mu_x, poly_degree=self.poly_degree)
        final_df = self.gen_outcome(df_cate)

        return final_df

 

#sim_1: SimulationStudy = SimulationStudy(p=200, mean_correlation=0.8, n=2000, poly_degree=2)
#simulation_1 = sim_1.create_dataset()