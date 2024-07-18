from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import Methods_new as method
from Gen_data import SimulationStudy
from tqdm import tqdm




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




def coverage(ci_bounds):
    column_names = ['Lower Bound', 'Upper Bound', 'True CATE']
    ci_bounds = pd.DataFrame(ci_bounds, columns = column_names)
    ci_bounds['coverage'] = np.where((ci_bounds['Lower Bound'] <= ci_bounds['True CATE']) & (ci_bounds['True CATE'] <= ci_bounds['Upper Bound']), 1, 0)
    #print(ci_bounds)
    coverage = np.sum(ci_bounds['coverage'])/ci_bounds['coverage'].count()

    return coverage




def create_mse_df(dict_test: dict, dict_train: dict, li_train: list, li_test: list, n: int) -> pd.DataFrame:
    mse_train_mean = np.mean(np.array(li_train))
    mse_test_mean = np.mean(np.array(li_test))
                       
    key = f'{n}'
    dict_test[key] = mse_test_mean
    dict_train[key] = mse_train_mean

    mse_df = pd.DataFrame()
    mse_df['n'] = dict_test.keys()
    mse_df['MSE Test'] = dict_test.values()
    mse_df['MSE Train'] = dict_train.values()

    return mse_df




def create_coverage_df(p_dict_coverage: dict) -> pd.DataFrame:
    rows = []
    for p_key, est_dict in p_dict_coverage.items():
        for est_key, n_dict in est_dict.items():
            for n_key, coverage_rate in n_dict.items():
                rows.append({'p': p_key, 'est': est_key, 'n': n_key, 'coverage_rates': coverage_rate})

    # Creating the DataFrame
    coverage_df = pd.DataFrame(rows, columns=['p', 'est', 'n', 'coverage_rates'])
    return coverage_df




def mse_ci_analysis(p: int, mean_correlation: float, n_list: list, no_feat_cate:int, estimator = None, function = None) -> tuple[pd.DataFrame, dict]:
    dict_test = {}
    dict_train = {}
    coverage_dict = {}

    for n in n_list:
           
        li_train = []
        li_test = []

        coverage_li = []

        for i in range(5):

            sim: SimulationStudy = SimulationStudy(p=p, mean_correlation=mean_correlation, cor_variance=0.2, n=n, no_feat_cate=no_feat_cate, non_linear=function)
            simulation = sim.create_dataset()
            train_df, test_df, X_train, Y_train, T_train, X_test, T_test, Y_test, true_cate_train, true_cate_test = get_split(simulation)
            
            
            if estimator=='OLS':
                estimated_cate_train, estimated_cate_test, MSE_train, MSE_test = method.ols_estimator(X_train, X_test, Y_train, T_train, T_test, true_cate_train, true_cate_test)
            elif estimator=='T-Learner':
                estimated_cate_train, estimated_cate_test, MSE_test, MSE_train = method.TLearner_estimator(Y_train, T_train, X_train, X_test, true_cate_train, true_cate_test)
            elif estimator=='HRF':
                estimated_cate_train, estimated_cate_test, ci_bounds, MSE_test, MSE_train = method.HRF(Y_train, T_train, X_train, X_test, true_cate_train, true_cate_test)
            elif estimator=='CF DML':
                estimated_cate_train, estimated_cate_test, ci_bounds, MSE_test, MSE_train = method.CF_DML(Y_train, T_train, X_train, X_test, true_cate_train, true_cate_test)
            elif estimator=='GRF':
                estimated_cate_train, estimated_cate_test, ci_bounds, MSE_test, MSE_train = method.GRF_estimator(Y_train, T_train, X_train, X_test, true_cate_train, true_cate_test)
            else:
                print('Choose either OLS, T-Learner, HRF, CF DML, or GRF')

           
            li_train.append(MSE_train)
            li_test.append(MSE_test)

            if estimator == 'OLS' or estimator == 'T-Learner':
                pass
            else:
                coverage_rate = coverage(ci_bounds)
                coverage_li.append(coverage_rate)
            
               
        mse_df = create_mse_df(dict_test, dict_train, li_train, li_test, n)

        if estimator == 'OLS' or estimator == 'T-Learner':
            pass
        else:
            coverage_mean = np.mean(np.array(coverage_li))
            coverage_dict[f'{n}'] = coverage_mean
            

       
    if estimator=='OLS' or estimator=='T-Learner':
        return mse_df
    else:
        return mse_df, coverage_dict





def get_mse_coverage(p_list: list, n_list: list, mean_correlation: float, function=None) -> tuple[dict]:
    p_dict_mse= {}
    p_dict_coverage = {}
    estimator_list = ['OLS', 'T-Learner', 'HRF', 'CF DML', 'GRF']

    for p in tqdm(p_list):
        
        estimator_dict_mse = {}
        estimator_dict_coverage = {}

        for estimator in estimator_list:
                print(f'Current p is {p} and current estimator is {estimator}')
                key_est = f'{estimator}'

                if estimator=='OLS' or estimator == 'T-Learner':
                    mse_simulation = mse_ci_analysis(p=p, mean_correlation=mean_correlation, n_list=n_list, no_feat_cate=2, estimator=estimator, function=function)

                else:
                    mse_simulation, coverage_dict = mse_ci_analysis(p=p, mean_correlation=mean_correlation, n_list=n_list, no_feat_cate=2, estimator=estimator, function=function)
                    estimator_dict_coverage[key_est] = coverage_dict
                    print(estimator_dict_coverage)

                estimator_dict_mse[key_est] = mse_simulation

        key_p = f'{p}'
        p_dict_mse[key_p] = estimator_dict_mse
        p_dict_coverage[key_p] = estimator_dict_coverage

    return p_dict_mse, p_dict_coverage