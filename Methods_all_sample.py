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
from scipy import stats
 


#Compute OLS confidence intervals
def get_confidence_intervals_ols(ols, x_train_ols, y_train, coeff, intercept):
    alpha = 0.05
    coefs = np.r_[[intercept], coeff]
    # build an auxiliary dataframe with the constant term in it
    X_aux = x_train_ols.copy()
    X_aux.insert(0, 'const', 1)
    # degrees of freedom
    dof = -np.diff(X_aux.shape)[0]
    # Student's t-distribution table lookup
    t_val = stats.t.isf(alpha/2, dof)
    # MSE of the residuals
    mse = np.sum((y_train - ols.predict(x_train_ols)) ** 2) / dof
    # inverse of the variance of the parameters
    var_params = np.diag(np.linalg.inv(X_aux.T.dot(X_aux)))
    # distance between lower and upper bound of CI
    gap = t_val * np.sqrt(mse * var_params)

    conf_int = pd.DataFrame({'lower': coefs - gap, 'upper': coefs + gap}, index=X_aux.columns)

    return conf_int



#Define interactive OLS
def ols_estimator(X_train: pd.DataFrame, X_test: pd.DataFrame, Y_train: pd.DataFrame, T_train: pd.DataFrame, T_test: pd.DataFrame, true_cate_train: pd.DataFrame, true_cate_test: pd.DataFrame, ci = None) -> tuple[pd.DataFrame, int]:

    interactions_train = pd.DataFrame()
    interactions_test = pd.DataFrame()

    #Get interactions between features and the treatment indicator
    for col in X_train.columns:
        interactions_train[f'{col}_T'] = X_train[col]*T_train
        interactions_test[f'{col}_T'] = X_test[col]*T_test

    #Join interactions with the data set
    X_train_ols = X_train.join(interactions_train)
    X_test_ols = X_test.join(interactions_test)

    #Fit OLS
    X_train_regression = X_train_ols[[col for col in X_train_ols.columns if col.startswith('X') or (col.startswith('X') and col.endswith('T'))]]
    ols = LinearRegression().fit(X_train_regression, Y_train)

    #Get coefficients for the interaction terms
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

    if ci is None:
        return coeff, estimated_cate_ols_train, estimated_cate_ols_test, OLS_RMSE_train, OLS_RMSE_test
    else:
        conf_int = get_confidence_intervals_ols(ols, X_train_regression, Y_train, ols.coef_, ols.intercept_)
        return conf_int, coeff, estimated_cate_ols_train, estimated_cate_ols_test, OLS_RMSE_train, OLS_RMSE_test




#Estimate the T-learner model
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

    return est_t, estimated_cate_t_train, estimated_cate_t_test, T_RMSE_train, T_RMSE_test



#Estimate the GRF model
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
    feat_importance = est_grf.feature_importances_

    estimated_cate_grf_train = est_grf.predict(X_train)
    estimated_cate_grf_test = est_grf.predict(X_test)

    GRF_RMSE_train = root_mean_squared_error(true_cate_train, estimated_cate_grf_train)
    GRF_RMSE_test= root_mean_squared_error(true_cate_test, estimated_cate_grf_test)

    return est_grf, feat_importance, estimated_cate_grf_train, estimated_cate_grf_test, GRF_RMSE_train, GRF_RMSE_test



#Estimate the causal forest model
def CF_DML(Y_train: pd.DataFrame, T_train: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame, T_test: pd.DataFrame, true_cate_train:pd.DataFrame, true_cate_test: pd.DataFrame) -> tuple[np.ndarray, float]:
 
    Y_train = Y_train.to_numpy()
    X_train = X_train.to_numpy()
    T_train = T_train.to_numpy()
    X_test = X_test.to_numpy()
    true_cate_test = true_cate_test['CATE'].to_numpy()
    true_cate_train = true_cate_train['CATE'].to_numpy()

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
    feat_importance = est_cfdml.feature_importances_

    estimated_cate_cfdml_train = est_cfdml.effect(X_train)
    estimated_cate_cfdml_test = est_cfdml.effect(X_test)

    CFDML_RMSE_train = root_mean_squared_error(true_cate_train, estimated_cate_cfdml_train)
    CFDML_RMSE_test = root_mean_squared_error(true_cate_test, estimated_cate_cfdml_test)
    
    return est_cfdml, feat_importance, estimated_cate_cfdml_train, estimated_cate_cfdml_test, CFDML_RMSE_train, CFDML_RMSE_test