import pandas as pd
import numpy as np
import scipy.stats as ss
from sklearn.preprocessing import PolynomialFeatures 
import math
import warnings
from sklearn.preprocessing import StandardScaler

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
    non_linear: None
        Determines whether the CATE function is non-linear. If 'quadratic' is inputed, then the CATE is the sum of the square of features. 
        Default is a linear CATE.

    
    
    Returns
    -------
    SimulationStudy object

    '''




    def __init__(self, p: int, mean_correlation: float, cor_variance: float, 
                 n: int, no_feat_cate: int, non_linear=None, seed=None) -> pd.DataFrame:
        
        self.p = p
        self.mean_correlation = mean_correlation
        self.cor_variance = cor_variance
        self.n = n
        self.no_feat_cate = no_feat_cate
        self.non_linear = non_linear
        self.seed = seed




    def get_covariance_matrix(self) -> np.ndarray:

        ''' 
        
        Get p*p covariance matrix. Takes parameter self and returns covariance and mean matrix.

        If covariance matrix is not positive-definite, cholesky decomposition is not possible. In which case, LinAlgError occurs. 

        Either the mean of the correlation distribution or the variance must be adjusted.

        '''

        mean = np.zeros(self.p)
        variance_diagonal = np.ones(self.p) 
        cov_matrix = np.zeros((self.p, self.p))


        try:
            for i in range(self.p):
                cov_matrix[i, i] = variance_diagonal[i]

                for j in range(i+1, self.p):
                    correlation = np.clip(np.random.normal(self.mean_correlation, self.cor_variance), -1, 1)

                    while True: 
                        if math.isclose(correlation, 1, rel_tol=1e-20) or math.isclose(correlation, -1, rel_tol=1e-20):
                            correlation = np.clip(np.random.normal(self.mean_correlation, self.cor_variance), -1, 1)
                            continue
                        else:
                            break
                    

                    cov_matrix[i, j] = cov_matrix[j, i] = correlation 

            #Test covariance matrix for symmetry and positive-definiteness
            pos_def_test = np.linalg.cholesky(cov_matrix) 

        except np.linalg.LinAlgError:
            pass


        return cov_matrix, mean
        
      


    def get_features(self, cov_matrix: np.ndarray, mean: np.ndarray) -> pd.DataFrame:

        '''

        Uses self parameter, the covariance matrix, and the mean matrix to draw p features from a multivariate normal distribution with n observations. 

        Method returns a pandas dataframe.

        Note that the StandardScaler modified the correlation matrix by scaling it down. A post-standardization correction is then made to ensure that the original covariance-variance matrix is valid.

        '''
        
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        rng = np.random.default_rng(self.seed)
        multivariate_samples = rng.multivariate_normal(mean, cov_matrix, self.n, tol=1e-6)
        df_original = pd.DataFrame(multivariate_samples, columns=[f"X{i}" for i in range(self.p)]) 

        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(multivariate_samples), columns=[f"X{i}" for i in range(self.p)])

        original_corr = df_original.corr()
        chol_decomp = np.linalg.cholesky(original_corr)
        adjusted_samples = df_scaled.values @ chol_decomp.T
       
        final_scaler = StandardScaler(with_mean=False)
        df_final = pd.DataFrame(final_scaler.fit_transform(adjusted_samples), columns=[f"X{i}" for i in range(self.p)])
        return df_final


    
    def gen_mu_x(self, df: pd.DataFrame) -> pd.DataFrame:

        '''

        Method creates the mean effect of the covariates on the outcome. It contains two times the number of features from the CATE function.

        Takes self and the generated data frame with the features as arguments. 

        Method returns the updated pandas data frame with a column containing the mu function.

        '''

        feat_no = (self.no_feat_cate)*2  
        columns = [f"X{i}" for i in range(feat_no)]
        
        poly = PolynomialFeatures(interaction_only=True)
        poly_features = poly.fit_transform(df[columns].to_numpy())
        interaction_sum = np.sum(poly_features, axis=1)
        
        df['mu_x'] = interaction_sum

        return df




    def gen_cate(self, df: pd.DataFrame, non_linear=None) -> pd.DataFrame:

        '''

        Method generates the CATE function. It takes the self parameter, the pandas data frame with the generated features, and a non-linear argument.

        non_linear is None and returns a linear CATE. 

        If 'quadratic', then returns a non-linear CATE function.

        '''

        columns = [f"X{i}" for i in range(self.no_feat_cate)]

        feat_cate = df[columns]
        np.random.seed(220924) #Set seed to make sure that the weights are reproducible for consistent results
        weights = np.random.choice(range(1, self.no_feat_cate + 1), size=self.no_feat_cate, replace=False)
       
        if non_linear == 'quadratic':

            sq_feat = np.square(feat_cate.to_numpy())
            quad_sum = np.sum(sq_feat*weights, axis=1).reshape(-1,1)
            df['CATE'] = quad_sum

        else:
            lin_cate = np.sum(feat_cate.to_numpy()*weights, axis=1)
            df['CATE'] = lin_cate

        return df




    def gen_outcome(self, df: pd.DataFrame) -> pd.DataFrame:

        '''

        Method generates the outcome y. Takes the parameter self and a pandas data frame containing the generated features.

        Returns a pandas data frame containing a column with the outcome y.

        '''
        
        #Generate treatment assignment
        df['T'] = np.random.binomial(1, 0.5, len(df)).astype(int)
      
        #Generate outcome y
        df['y'] = df['CATE']*df['T'] + df['mu_x'] + np.random.normal(0, 1, self.n)
        return df




    def create_dataset(self):

        '''

        Takes the self parameter. 
        
        Combines all previous methods in the SimulationStudy class sequentially and returns the total data set object.

        '''

        cov_matrix, mean = self.get_covariance_matrix()
        df = self.get_features(cov_matrix=cov_matrix, mean=mean)
        df_mu_x = self.gen_mu_x(df=df)
        df_cate = self.gen_cate(df=df_mu_x, non_linear=self.non_linear)
        final_df = self.gen_outcome(df_cate)

        return final_df