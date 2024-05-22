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
    p: int
        Number of covariates/complexity for the data set
    mean_correlation : float
        Average feature correlation on the data set
    cor_variance: float
        Adjusts the variance of the correlation distribution from which the correlation values are drawn
    n: int
        Observations in the data set
    include_cat_var:
        Generates categorical features for half of the total number of features. Default is false.
    poly_degree: int
        Sets the polynomial degree for the Polynomial Feature function. Default is 2.
    geom: bool
        Default is false. If true, CATE is computed as a sinus function.
    
    Returns
    -------
    SimulationStudy object
    '''




    def __init__(self, p: int, mean_correlation: float, cor_variance: float, n: int, include_cat_var: bool = False, poly_degree: int = 2, geom: bool = False) -> pd.DataFrame:
        self.p = p
        self.mean_correlation = mean_correlation
        self.cor_variance = cor_variance
        self.n = n
        self.include_cat_var = include_cat_var
        self.poly_degree = poly_degree
        self.geom = geom




    def get_covariance_matrix(self) -> np.ndarray:
        ''' 
        
        Get p*p covariance matrix. Takes parameter self and returns covariance and mean matrix.

        If covariance matrix is not positive-definite, cholesky decomposition is not possible. In which case, LinAlgError occurs. 
        Either the mean of the correlation distribution or the variance must be adjusted.
        
        '''
        mean = np.zeros(self.p)
        variance_diagonal = np.ones(self.p) # in principle, n could be n draws from some meta-distribution
        cov_matrix = np.zeros((self.p, self.p))

        # assign variance to diagonal
        try:
            for i in range(self.p): #columns
                cov_matrix[i, i] = variance_diagonal[i]

                for j in range(i+1, self.p):
                    correlation = np.clip(np.random.normal(self.mean_correlation, self.cor_variance), -0.8, 0.8) 
                    cov_matrix[i, j] = cov_matrix[j, i] = correlation *  np.sqrt(variance_diagonal[i] * variance_diagonal[j]) 
            
            #Test covariance matrix for symmetry and positive-definiteness
            pos_def_test = np.linalg.cholesky(cov_matrix) 

        except np.linalg.LinAlgError:
            print('Correlation structure has to be adjusted')

        return cov_matrix, mean
        
      


    def get_features(self, cov_matrix: np.ndarray, mean: np.ndarray) -> pd.DataFrame:

        '''
        Uses self parameter, the covariance matrix, and the mean matrix to draw p features from a multivariate normal distribution with n observations. 

        Method returns a pandas dataframe.
       
        '''
        rng = np.random.default_rng()
        multivariate_samples = rng.multivariate_normal(mean, cov_matrix, self.n)

        if self.include_cat_var is True:
            #cat_var = np.random.randint(1, 5, size=self.n)
            #df_original = pd.DataFrame(cat_var, columns=[f"X{i}" for i in range(int((self.p)/2))])
            #df_original = pd.DataFrame(multivariate_samples, columns=[f"X{i}" for i in range(int((self.p)/2), (self.p))])
            #df_original = df_original[np.random.permutation(df_original.columns)]
            pass
        else:
            df_original = pd.DataFrame(multivariate_samples, columns=[f"X{i}" for i in range(self.p)])
            
        return df_original





    def gen_mu_x(self, df: pd.DataFrame) -> pd.DataFrame:

        feat_no = int(self.p/2)

        columns = [f"X{i}" for i in range(feat_no)]
        poly = PolynomialFeatures(interaction_only=True)

        poly_features = poly.fit_transform(df[columns])
        interaction_sum = np.sum(poly_features, axis=1) - np.sum(df[columns].values, axis=1)

        df['mu_x'] =  interaction_sum

        return df
    



    def gen_cate(self, poly_degree: int, df: pd.DataFrame, geom: bool = False) -> pd.DataFrame: 

        #Choose number of features that will be used for the CATE function    
        feat_no = int(self.p/2)
        columns = [f"X{i}" for i in range(feat_no)]

        if geom==True:
            #weights = (np.random.randint(0, 100, size=(feat_no)))/100
            #mult_col_weight = np.outer(weights, df[columns])
            #sum_weighted = np.sum(mult_col_weight)
            #cate_sin = np.sin(sum_weighted)
            feature_sum = np.sum(df[columns], axis=1)
            cate_sin = np.sin(feature_sum)
            df['CATE'] = cate_sin #+ np.random.normal(0, 1, self.n)

        else:
            poly = PolynomialFeatures(poly_degree, include_bias=False)
            interactions = PolynomialFeatures(interaction_only=True, include_bias=False)

            poly_features = poly.fit_transform(df[columns])
            interaction_features = interactions.fit_transform(df[columns])

            # Sum polynomial features and subtract redundant values      
            sum_poly_features = np.sum(poly_features, axis=1) - np.sum(interaction_features, axis=1) - np.sum(df[columns].values, axis=1)

            df['CATE'] = (sum_poly_features + np.random.normal(0, 1, self.n))

        return df




    def gen_model(self, df: pd.DataFrame) -> pd.DataFrame:
        
        #Generate treatment assignment
        df['T'] = np.random.binomial(1, 0.5, len(df)).astype(int)
      
        #Generate outcome y
        df['y'] = df['CATE']*df['T'] + df['mu_x'] + np.random.normal(0, 1, self.n)
        return df




    def create_dataset(self):

        cov_matrix, mean = self.get_covariance_matrix()
        df = self.get_features(cov_matrix=cov_matrix, mean=mean)
        df_mu_x = self.gen_mu_x(df=df)
        df_cate = self.gen_cate(df=df_mu_x, poly_degree=self.poly_degree, geom=self.geom)
        final_df = self.gen_model(df_cate)

        return final_df

 

#sim_1: SimulationStudy = SimulationStudy(p=200, mean_correlation=0.8, cor_variance=0.01, n=2000, poly_degree=2)
#simulation_1 = sim_1.create_dataset()

