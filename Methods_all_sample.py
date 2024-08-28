from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from econml.metalearners import TLearner
from econml.dml import CausalForestDML
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from econml.grf import CausalForest
 



def ols_estimator(X_train: pd.DataFrame, X_test: pd.DataFrame, Y_train: pd.DataFrame, T_train: pd.DataFrame, T_test: pd.DataFrame, true_cate_train: pd.DataFrame, true_cate_test: pd.DataFrame) -> tuple[pd.DataFrame, int]:

    interactions_train = pd.DataFrame()
    interactions_test = pd.DataFrame()


    for col in X_train.columns:
        interactions_train[f'{col}_T'] = X_train[col]*T_train
        interactions_test[f'{col}_T'] = X_test[col]*T_test


    X_train_ols = X_train.join(interactions_train)
    X_test_ols = X_test.join(interactions_test)

    #Create OLS object
    ols = LinearRegression()

    #Fit OLS
    X_train_regression = X_train_ols[[col for col in X_train_ols.columns if col.startswith('X') or (col.startswith('X') and col.endswith('T'))]]
    ols.fit(X_train_regression, Y_train)


    coeff = pd.DataFrame(ols.coef_, X_train_ols.columns).T
    coeff = coeff[[col for col in coeff.columns if col.startswith('X') and col.endswith('T')]].T
    
    #Estimate CATE train
    X_train_ols = X_train_ols[[col for col in X_train_ols.columns if (col.startswith('X') and not col.endswith('T'))]]
    estimated_cate_ols_train = X_train_ols.dot(coeff.to_numpy())

    #Estimate CATE test
    X_test_ols = X_test_ols[[col for col in X_test_ols.columns if (col.startswith('X') and not col.endswith('T'))]]
    estimated_cate_ols_test = X_test_ols.dot(coeff.to_numpy())        
  
  
    #Calculate MSE train set
    true_cate_train = true_cate_train['CATE'] 
    OLS_RMSE_train = root_mean_squared_error(true_cate_train, estimated_cate_ols_train)

    #Calculate MSE test set
    true_cate_test = true_cate_test['CATE']      
    OLS_RMSE_test = root_mean_squared_error(true_cate_test, estimated_cate_ols_test)

    return estimated_cate_ols_train, estimated_cate_ols_test, OLS_RMSE_train, OLS_RMSE_test




def TLearner_estimator(Y_train: pd.DataFrame, T_train: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame, T_test:pd.DataFrame, true_cate_train: pd.DataFrame, true_cate_test: pd.DataFrame) -> tuple[np.ndarray, float]:
   
    Y_train = Y_train.to_numpy()
    X_train = X_train.to_numpy()
    T_train = T_train.to_numpy()
    X_test = X_test.to_numpy()
    true_cate_test = true_cate_test['CATE'].to_numpy()
    true_cate_train = true_cate_train['CATE'].to_numpy()

    est_t = TLearner(models=RandomForestRegressor(n_estimators=1000, 
                                                  random_state=220924, 
                                                  n_jobs=-1))
    
    est_t.fit(Y=Y_train, T=T_train, X=X_train)

    estimated_cate_t_train = est_t.effect(X_train)
    estimated_cate_t_test = est_t.effect(X_test)

    T_RMSE_train = root_mean_squared_error(true_cate_train, estimated_cate_t_train)
    T_RMSE_test = root_mean_squared_error(true_cate_test, estimated_cate_t_test)

    return estimated_cate_t_train, estimated_cate_t_test, T_RMSE_train, T_RMSE_test




def GRF_estimator(Y_train: pd.DataFrame, T_train: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame, T_test: pd.DataFrame, true_cate_train: pd.DataFrame, true_cate_test: pd.DataFrame) -> tuple[np.ndarray, float]:

    Y_train = Y_train.to_numpy()
    X_train = X_train.to_numpy()
    T_train = T_train.to_numpy()
    X_test = X_test.to_numpy()
    true_cate_test = true_cate_test['CATE'].to_numpy()
    true_cate_train = true_cate_train['CATE'].to_numpy()


    est_grf = CausalForest(n_estimators=1000, 
                           criterion='het', 
                           random_state=220924, 
                           n_jobs=-1)
    
    est_grf.fit(y=Y_train, T=T_train, X=X_train)

    estimated_cate_grf_train = est_grf.predict(X_train)
    estimated_cate_grf_test = est_grf.predict(X_test)

    GRF_RMSE_train = root_mean_squared_error(true_cate_train, estimated_cate_grf_train)
    GRF_RMSE_test= root_mean_squared_error(true_cate_test, estimated_cate_grf_test)

    return estimated_cate_grf_train, estimated_cate_grf_test, GRF_RMSE_train, GRF_RMSE_test




def CF_DML(Y_train: pd.DataFrame, T_train: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame, T_test: pd.DataFrame, true_cate_train:pd.DataFrame, true_cate_test: pd.DataFrame) -> tuple[np.ndarray, float]:
 
    Y_train = Y_train.to_numpy()
    X_train = X_train.to_numpy()
    T_train = T_train.to_numpy()
    X_test = X_test.to_numpy()
    true_cate_test = true_cate_test['CATE'].to_numpy()
    true_cate_train = true_cate_train['CATE'].to_numpy()

#Estimate the causal forest model
    est_cfdml = CausalForestDML(
                        model_y=RandomForestRegressor(n_estimators=1000, random_state=220924, n_jobs=-1),
                        model_t=DummyClassifier(random_state=220924),
                        discrete_treatment=True,
                        n_estimators=1000,
                        cv=5,
                        random_state=220924,
                        n_jobs=-1,
                        criterion='het',
                        honest=True)

    est_cfdml.fit(Y=Y_train, T=T_train, X=X_train, W=None, cache_values=True)
    #print(est_cfdml.models_y)

    estimated_cate_cfdml_train = est_cfdml.effect(X_train)
    estimated_cate_cfdml_test = est_cfdml.effect(X_test)

    CFDML_RMSE_train = root_mean_squared_error(true_cate_train, estimated_cate_cfdml_train)
    CFDML_RMSE_test = root_mean_squared_error(true_cate_test, estimated_cate_cfdml_test)
    
    return estimated_cate_cfdml_train, estimated_cate_cfdml_test, CFDML_RMSE_train, CFDML_RMSE_test