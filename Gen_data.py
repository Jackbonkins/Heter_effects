import pandas as pd
import numpy as np
import scipy.stats as ss
from sklearn.preprocessing import PolynomialFeatures 
import math
import warnings

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
    no_feat_cate: int
        Defines the number of features that define the CATE function
    non_linear: bool
        Determines whether the CATE function is non-linear. If true, CATE is the sum of sinusoid functions and a linear component. Default is that the CATE is linear.
    
    
    Returns
    -------
    SimulationStudy object

    '''




    #def __init__(self, p: int, mean_correlation: float, cor_variance: float, n: int, geom: bool = False) -> pd.DataFrame:
    def __init__(self, p: int, mean_correlation: float, cor_variance: float, 
                 n: int, no_feat_cate: int, non_linear: bool = False) -> pd.DataFrame:
        
        self.p = p
        self.mean_correlation = mean_correlation
        self.cor_variance = cor_variance
        self.n = n
        self.no_feat_cate = no_feat_cate
        #self.poly_degree = poly_degree
        self.non_linear = non_linear




    def get_covariance_matrix(self) -> np.ndarray:
        ''' 
        
        Get p*p covariance matrix. Takes parameter self and returns covariance and mean matrix.

        If covariance matrix is not positive-definite, cholesky decomposition is not possible. In which case, LinAlgError occurs. 
        Either the mean of the correlation distribution or the variance must be adjusted.
        
        '''
        mean = np.zeros(self.p)
        variance_diagonal = np.ones(self.p) 
        cov_matrix = np.zeros((self.p, self.p))

        # assign variance to diagonal
        try:
            for i in range(self.p): #columns
                cov_matrix[i, i] = variance_diagonal[i]

                for j in range(i+1, self.p):

                    #corr_value = ss.pearson3.rvs(loc = self.mean_correlation, scale = self.cor_variance, skew = -1, size = 1)
                    #correlation = np.clip(corr_value[0], -0.95, 0.95)

                    correlation = np.clip(np.random.normal(self.mean_correlation, self.cor_variance), -0.95, 0.95) 
                    cov_matrix[i, j] = cov_matrix[j, i] = correlation *  np.sqrt(variance_diagonal[i] * variance_diagonal[j])
                            
            #Test covariance matrix for symmetry and positive-definiteness
            pos_def_test = np.linalg.cholesky(cov_matrix) 

        except np.linalg.LinAlgError:
            pass

        return cov_matrix, mean
        
      


    def get_features(self, cov_matrix: np.ndarray, mean: np.ndarray) -> pd.DataFrame:

        '''
        Uses self parameter, the covariance matrix, and the mean matrix to draw p features from a multivariate normal distribution with n observations. 

        Method returns a pandas dataframe.
       
        '''
        warnings.filterwarnings("ignore", category=RuntimeWarning) 
        rng = np.random.default_rng()
        multivariate_samples = rng.multivariate_normal(mean, cov_matrix, self.n, tol=1e-6)

        df_original = pd.DataFrame(multivariate_samples, columns=[f"X{i}" for i in range(self.p)])
            
        return df_original





    def gen_mu_x(self, df: pd.DataFrame) -> pd.DataFrame:

        feat_no = int(self.p/2)

        columns = [f"X{i}" for i in range(feat_no)]
        poly = PolynomialFeatures(interaction_only=True)

        poly_features = poly.fit_transform(df[columns].to_numpy())
        interaction_sum = np.sum(poly_features, axis=1) - np.sum(df[columns].to_numpy(), axis=1)

        df['mu_x'] =  interaction_sum

        return df
    



    def gen_cate(self, df: pd.DataFrame, non_linear=False) -> pd.DataFrame:
        columns = [f"X{i}" for i in range(self.no_feat_cate)]

        feat_cate = df[columns]
        np.random.seed(42) #Set seed to make sure that the weights are reproducible for consistent results
        weights = np.random.choice(range(1, self.no_feat_cate + 1), size=self.no_feat_cate, replace=False)
        weighted_feat = feat_cate*weights

        if non_linear == True:

            if self.no_feat_cate == 1:
                non_lin_cate = np.sin(feat_cate.to_numpy())
                

            elif self.no_feat_cate > 1:
                
                feature_sin = weighted_feat.iloc[:, :-1].to_numpy()
                feature_sin = np.sum(np.sin(feature_sin), axis=1).reshape(-1,1)
                feature_lin = feat_cate.iloc[:, -1:].to_numpy()
                non_lin_cate = feature_sin + feature_lin

            else:
                raise Exception('Cate features cannot be 0 or negative')
            
            df['CATE'] = non_lin_cate

        else:
            lin_cate = np.sum(weighted_feat.to_numpy(), axis=1)
            df['CATE'] = lin_cate + np.random.normal(0, 1, self.n)

        
        return df


    #def gen_cate(self, df: pd.DataFrame, geom: bool = False) -> pd.DataFrame: 

        #Choose number of features that will be used for the CATE function    
     #   feat_no = math.ceil(self.p/3)
      #  columns = [f"X{i}" for i in range(feat_no)]

       # if geom==True:
            
        #    middle = math.ceil(len(columns)/2)
         #   first_half = columns[:middle]
          #  second_half = columns[middle:]
            
            #weights = (np.random.randint(0, 4, size=(len(first_half))))
            #weighted_feat = np.outer(weights, df[first_half].to_numpy())


           # feature_sum = np.sum(df[second_half].to_numpy(), axis=1)
            #weighted_feat_sum = np.sum(weighted_feat, axis=1)
            #cate_sin = np.sin(weighted_feat_sum)
            #cate_sin = np.sin(np.sum(df[first_half].to_numpy(), axis=1))
            #df['CATE'] = cate_sin + feature_sum

        #else:
            #poly = PolynomialFeatures(poly_degree, include_bias=False)
            #interactions = PolynomialFeatures(interaction_only=True, include_bias=False)

            #poly_features = poly.fit_transform(df[columns].to_numpy())
            #interaction_features = interactions.fit_transform(df[columns].to_numpy())

            # Sum polynomial features and subtract redundant values      
            #sum_poly_features = np.sum(poly_features, axis=1) - np.sum(interaction_features, axis=1) - np.sum(df[columns].to_numpy(), axis=1)

            #df['CATE'] = sum_poly_features + np.random.normal(0, 1, self.n)

        #else:
         #   pass
            #df['CATE'] = 
            

        #return df




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
        df_cate = self.gen_cate(df=df_mu_x, non_linear=self.non_linear)
        final_df = self.gen_model(df_cate)

        return final_df

 
