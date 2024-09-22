
''' 

Script defines functions that are used to calculate the RMSE values and creates the pandas dfs.

'''

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import Methods_all_sample as method
from Gen_data import SimulationStudy
from tqdm import tqdm



#Split data set into train and test sets
def get_split(simulation: pd.DataFrame) -> tuple[pd.DataFrame]:
    
    train_df, test_df = train_test_split(simulation, test_size=0.5)
    # Extract features and target variables for training
    X_train = train_df[[col for col in simulation.columns if col.startswith('X')]]
    T_train = train_df['T']
    Y_train = train_df['y']

    # Extract features and true CATE for testing
    X_test = test_df[[col for col in simulation.columns if col.startswith('X')]]
    T_test = test_df['T']
    y_test = test_df['y']

    true_cate_train = train_df[['CATE', 'T']]
    true_cate_test = test_df[['CATE', 'T']]

    return train_df, test_df, X_train, Y_train, T_train, X_test, T_test, y_test, true_cate_train, true_cate_test



# Creates the RMSE df by averaging out the RMSE from each simulation
def create_rmse_df(dict_test: dict, dict_train: dict, li_train: list, li_test: list, set_size: int) -> pd.DataFrame:
    rmse_train_mean = np.mean(np.array(li_train))
    rmse_test_mean = np.mean(np.array(li_test))
                       
    key = f'{set_size}'
    dict_test[key] = rmse_test_mean
    dict_train[key] = rmse_train_mean

    rmse_df = pd.DataFrame()
    rmse_df['n'] = dict_test.keys()
    rmse_df['RMSE Test'] = dict_test.values()
    rmse_df['RMSE Train'] = dict_train.values()

    return rmse_df



#Start RMSE analysis for each estimator
def rmse_analysis(p: int, mean_correlation: float, n_list: list, no_feat_cate:int, estimator = None, function = None) -> pd.DataFrame:
    dict_test = {}
    dict_train = {}

    for n in n_list:
           
        li_train = []
        li_test = []

        #Generate 5 runs of data sets that compute the esimated RMSEs for each estimator
        for i in range(5):

            sim: SimulationStudy = SimulationStudy(p=p, mean_correlation=mean_correlation, cor_variance=0.2, n=n, no_feat_cate=no_feat_cate, non_linear=function)
            simulation = sim.create_dataset()
            train_df, test_df, X_train, Y_train, T_train, X_test, T_test, Y_test, true_cate_train, true_cate_test = get_split(simulation)
            
            
            if estimator=='OLS':
                coeff, estimated_cate_train, estimated_cate_test, RMSE_train, RMSE_test = method.ols_estimator(X_train, X_test, Y_train, T_train, T_test, true_cate_train, true_cate_test)
            elif estimator=='T-Learner':
                est_t, estimated_cate_train, estimated_cate_test, RMSE_test, RMSE_train = method.TLearner_estimator(Y_train, T_train, X_train, X_test, T_test, true_cate_train, true_cate_test)
            elif estimator=='CF DML':
                est_cfdml, feat_importance, estimated_cate_train, estimated_cate_test, RMSE_test, RMSE_train = method.CF_DML(Y_train, T_train, X_train, X_test, T_test, true_cate_train, true_cate_test)
            elif estimator=='GRF':
                est_grf, feat_importance, estimated_cate_train, estimated_cate_test, RMSE_test, RMSE_train = method.GRF_estimator(Y_train, T_train, X_train, X_test, T_test, true_cate_train, true_cate_test)
            else:
                print('Choose either OLS, T-Learner, CF DML, or GRF')

           
            li_train.append(RMSE_train)
            li_test.append(RMSE_test)

            set_size = train_df.shape[0]

        print(f'Set size is equal to: {set_size}')
        rmse_df = create_rmse_df(dict_test, dict_train, li_train, li_test, set_size)
        

                    
    return rmse_df
       



#Function creates the final RMSE dictionary including all specified settings for all specified estimators
def get_rmse(p_list: list, n_list: list, no_feat_cate: int, mean_correlation: float, estimators: list, function=None) -> dict:
    p_dict_rmse= {}
    estimator_list = estimators

    for p in tqdm(p_list):
        
        estimator_dict_rmse = {}

        for estimator in estimator_list:
                print(f'Current p is {p} and current estimator is {estimator}')
                key_est = f'{estimator}'

                rmse_simulation = rmse_analysis(p=p, mean_correlation=mean_correlation, n_list=n_list, no_feat_cate=no_feat_cate, estimator=estimator, function=function)

                estimator_dict_rmse[key_est] = rmse_simulation

        key_p = f'{p}'
        p_dict_rmse[key_p] = estimator_dict_rmse

    return p_dict_rmse