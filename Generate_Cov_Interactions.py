import numpy as np
import pandas as pd
from Generate_Covariates import gen_cov_matrix, gen_covariates


#Define new function to interact covariates with each other and create Mu_0(x)
def gen_cov_interactions(covariates, no_interactions):

    first_x = covariates.columns[ :no_interactions]

    i = 0
    interactions_df = pd.DataFrame()

    #Loop over the first x variables in the covariate data frame
    for variable in first_x:
        if i % 2 == 0:
            temp_even = covariates[variable] 
        else:
            temp_uneven = covariates[variable]
            interactions_df.insert(0, f'X{i-1}_X{i}', temp_even*temp_uneven)    
        
        i += 1

    return interactions_df

