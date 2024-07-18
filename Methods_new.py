from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from econml.metalearners import TLearner
from econml.dml import CausalForestDML
from econml.dml import DML
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from econml.grf import CausalForest, RegressionForest
 




def ols_estimator(X_train: pd.DataFrame, X_test: pd.DataFrame, Y_train: pd.DataFrame, T_train: pd.DataFrame, T_test: pd.DataFrame, true_cate_train: pd.DataFrame, true_cate_test: pd.DataFrame) -> tuple[pd.DataFrame, int]:

    interactions_train = pd.DataFrame()
    interactions_test = pd.DataFrame()


    for col in X_train.columns:
        interactions_train[f'{col}_T'] = X_train[col]*T_train
        interactions_test[f'{col}_T'] = X_test[col]*T_test


    X_train_ols = X_train.join([interactions_train, pd.DataFrame(T_train, columns=['T'])])
    X_test_ols = X_test.join([interactions_test, pd.DataFrame(T_test, columns=['T'])])

    #Create OLS object
    ols = LinearRegression()

    #Fit OLS
    X_train_regression = X_train_ols[[col for col in X_train_ols.columns if col.startswith('X') or (col.startswith('X') and col.endswith('T'))]]
    ols.fit(X_train_regression, Y_train)

    #Restrict training and test sets only to the treated group
    X_train_ols_treated = X_train_ols[X_train_ols['T']==1]
    X_train_ols_treated = X_train_ols_treated[[col for col in X_train_ols_treated if col.startswith('X') and col.endswith('T')]]

    X_test_ols_treated = X_test_ols[X_test_ols['T']==1]
    X_test_ols_treated = X_test_ols_treated[[col for col in X_test_ols_treated.columns if col.startswith('X') and col.endswith('T')]]
        
    #Get regression coefficients
    coeff = pd.DataFrame(ols.coef_, X_train_regression.columns).T
    coeff = coeff[[col for col in coeff.columns if col.startswith('X') and col.endswith('T')]].T

    #Estimate CATE train
    estimated_cate_ols_train = X_train_ols_treated.dot(coeff)

    #Estimate CATE test
    estimated_cate_ols_test = X_test_ols_treated.dot(coeff)

    #Calculate MSE train set
    true_cate_train= true_cate_train[true_cate_train['T'] == 1]
    true_cate_train = true_cate_train['CATE'] 
    OLS_MSE_train = mean_squared_error(true_cate_train, estimated_cate_ols_train)

    #Calculate MSE test set
    true_cate_test = true_cate_test[true_cate_test['T'] == 1]
    true_cate_test = true_cate_test['CATE']      
    OLS_MSE_test = mean_squared_error(true_cate_test, estimated_cate_ols_test)

    return estimated_cate_ols_train, estimated_cate_ols_test, OLS_MSE_train, OLS_MSE_test




def TLearner_estimator(Y_train: pd.DataFrame, T_train: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame, true_cate_train: pd.DataFrame, true_cate_test: pd.DataFrame) -> tuple[np.ndarray, float]:

    Y_train = Y_train.to_numpy()
    T_train = T_train.to_numpy()
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    true_cate_test = true_cate_test['CATE'].to_numpy()
    true_cate_train = true_cate_train['CATE'].to_numpy()

    est_t = TLearner(models=RandomForestRegressor(n_estimators=1000))
    est_t.fit(Y=Y_train, T=T_train, X=X_train)

    estimated_cate_t_train = est_t.effect(X_train)
    estimated_cate_t_test = est_t.effect(X_test)

    T_MSE_train = mean_squared_error(true_cate_train, estimated_cate_t_train)
    T_MSE_test = mean_squared_error(true_cate_test, estimated_cate_t_test)

    return estimated_cate_t_train, estimated_cate_t_test, T_MSE_train, T_MSE_test




def HRF(Y_train: pd.DataFrame, T_train: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame, true_cate_train:pd.DataFrame, true_cate_test: pd.DataFrame) -> tuple[np.ndarray, float]:
    Y_train = Y_train.to_numpy()
    T_train = T_train.to_numpy()
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    true_cate_test = true_cate_test['CATE'].to_numpy()
    true_cate_train = true_cate_train['CATE'].to_numpy()

    est_hrf = RegressionForest(n_estimators=1000)
    est_hrf.fit(y=Y_train, X=X_train)
    lb, ub = est_hrf.predict_interval(X_test, alpha=0.05)
    ci_bounds = np.column_stack((lb, ub, true_cate_test))

    estimated_cate_hrf_train = est_hrf.predict(X_train)
    estimated_cate_hrf_test = est_hrf.predict(X_test)

    HRF_MSE_train = mean_squared_error(true_cate_train, estimated_cate_hrf_train)
    HRF_MSE_test= mean_squared_error(true_cate_test, estimated_cate_hrf_test)

    return estimated_cate_hrf_train, estimated_cate_hrf_test, ci_bounds, HRF_MSE_train, HRF_MSE_test






def CF_DML(Y_train: pd.DataFrame, T_train: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame, true_cate_train:pd.DataFrame, true_cate_test: pd.DataFrame) -> tuple[np.ndarray, float]:
 
    Y_train = Y_train.to_numpy()
    T_train = T_train.to_numpy()
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    true_cate_test = true_cate_test['CATE'].to_numpy()
    true_cate_train = true_cate_train['CATE'].to_numpy()

#Estimate the causal forest model
    est_cfdml = CausalForestDML(model_y='auto',
                        model_t=DummyClassifier(),
                        discrete_treatment=True,
                        n_estimators=1000,
                        cv=5,
                        #random_state=42,
                        criterion='mse',
                        honest=True)

    est_cfdml.fit(Y=Y_train, T=T_train, X=X_train, W=None, cache_values=True)

    estimated_cate_cfdml_train = est_cfdml.effect(X_train)
    estimated_cate_cfdml_test = est_cfdml.effect(X_test)

    lb, ub = est_cfdml.effect_interval(X_test, alpha=0.05)
    ci_bounds = np.column_stack((lb, ub, true_cate_test))

    CFDML_MSE_train = mean_squared_error(true_cate_train, estimated_cate_cfdml_train)
    CFDML_MSE_test = mean_squared_error(true_cate_test, estimated_cate_cfdml_test)
    
    return estimated_cate_cfdml_train, estimated_cate_cfdml_test, ci_bounds, CFDML_MSE_train, CFDML_MSE_test




def GRF_estimator(Y_train: pd.DataFrame, T_train: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame, true_cate_train: pd.DataFrame, true_cate_test: pd.DataFrame) -> tuple[np.ndarray, float]:

    Y_train = Y_train.to_numpy()
    T_train = T_train.to_numpy()
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    true_cate_test = true_cate_test['CATE'].to_numpy()
    true_cate_train = true_cate_train['CATE'].to_numpy()


    est_grf = CausalForest(n_estimators=1000)
    
    est_grf.fit(y=Y_train, T=T_train, X=X_train)

    lb, ub = est_grf.predict_interval(X_test, alpha=0.05)
    ci_bounds = np.column_stack((lb, ub, true_cate_test))

    estimated_cate_grf_train = est_grf.predict(X_train)
    estimated_cate_grf_test = est_grf.predict(X_test)

    GRF_MSE_train = mean_squared_error(true_cate_train, estimated_cate_grf_train)
    GRF_MSE_test= mean_squared_error(true_cate_test, estimated_cate_grf_test)

    return estimated_cate_grf_train, estimated_cate_grf_test, ci_bounds, GRF_MSE_train, GRF_MSE_test



def coverage(ci_bounds):
    column_names = ['Lower Bound', 'Upper Bound', 'True CATE']
    ci_bounds = pd.DataFrame(ci_bounds, columns = column_names)
    ci_bounds['coverage'] = np.where((ci_bounds['Lower Bound'] <= ci_bounds['True CATE']) & (ci_bounds['True CATE'] <= ci_bounds['Upper Bound']), 1, 0)
    coverage = np.sum(ci_bounds['coverage'])/ci_bounds['coverage'].count()

    return coverage