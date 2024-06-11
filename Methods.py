from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from econml.metalearners import TLearner
from econml.dml import CausalForestDML, NonParamDML
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error 



class EstimationMethods:

    '''

    Attributes
    ----------
    X_train: pd.DataFrame
        Features from the training data set
    T_train: pd.DataFrame
        Treatment variable for the training set
    Y_train: pd.DataFrame
        Outcome variable for the training set
    true_cate_train: pd.DataFrame
        True cate function of the training set
    X_test: pd.DataFrame
        Features from the test data set
    T_test: pd.DataFrame
        Treatment variable for the test set
    Y_test: pd.DataFrame
        Outcome variable for the test set
    true_cate_test: pd.DataFrame
        True cate function of the test set
    '''

    
    def __init__(self, X_train: pd.DataFrame, T_train: pd.DataFrame, Y_train: pd.DataFrame, X_test: pd.DataFrame, 
                 T_test: pd.DataFrame, Y_test: pd.DataFrame, true_cate_train: pd.DataFrame, true_cate_test: pd.DataFrame) -> tuple[pd.DataFrame | np.array, float]:

        self.X_train = X_train
        self.T_train = T_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.T_test = T_test
        self.Y_test = Y_test
        self.true_cate_train = true_cate_train
        self.true_cate_test = true_cate_test
    
        


    def ols_estimator(self) -> tuple[pd.DataFrame, float]:

        interactions_train = pd.DataFrame()
        interactions_test = pd.DataFrame()


        for col in self.X_train.columns:
            interactions_train[f'{col}_T'] = self.X_train[col]*self.T_train
            interactions_test[f'{col}_T'] = self.X_test[col]*self.T_test


        X_train_ols = self.X_train.join([interactions_train, pd.DataFrame(self.T_train, columns=['T'])])
        X_test_ols = self.X_test.join([interactions_test, pd.DataFrame(self.T_test, columns=['T'])])

        #Create OLS object
        ols = LinearRegression()

        #Fit OLS
        X_train_regression = X_train_ols[[col for col in X_train_ols.columns if col.startswith('X') or (col.startswith('X') and col.endswith('T'))]]
        ols.fit(X_train_regression, self.Y_train)

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
        estimated_cate_ols_test =X_test_ols_treated.dot(coeff)

        #Calculate MSE train set
        true_cate_train= self.true_cate_train[self.true_cate_train['T'] == 1]
        true_cate_train = true_cate_train['CATE'] 
        OLS_MSE_train = mean_squared_error(true_cate_train, estimated_cate_ols_train)

        #Calculate MSE test set
        true_cate_test = self.true_cate_test[self.true_cate_test['T'] == 1]
        true_cate_test = true_cate_test['CATE']      
        OLS_MSE_test = mean_squared_error(true_cate_test, estimated_cate_ols_test)

        return estimated_cate_ols_train, estimated_cate_ols_test, OLS_MSE_train, OLS_MSE_test
    



    def TLearner_estimator(self) -> tuple[np.ndarray, float]:

        Y_train = self.Y_train.to_numpy()
        T_train = self.T_train.to_numpy()
        X_train = self.X_train.to_numpy()
        X_test = self.X_test.to_numpy()
        true_cate_test = self.true_cate_test['CATE'].to_numpy()
        true_cate_train = self.true_cate_train['CATE'].to_numpy()
        
        est_t = TLearner(models=RandomForestRegressor())
        est_t.fit(Y=Y_train, T=T_train, X=X_train)

        estimated_cate_t_train = est_t.effect(X_train)
        estimated_cate_t_test = est_t.effect(X_test)

        T_MSE_train = mean_squared_error(true_cate_train, estimated_cate_t_train)
        T_MSE_test = mean_squared_error(true_cate_test, estimated_cate_t_test)
        

        return estimated_cate_t_train, estimated_cate_t_test, T_MSE_test, T_MSE_train
    



    def CF_DML(self) -> tuple[np.ndarray, float]:

        Y_train = self.Y_train.to_numpy()
        T_train = self.T_train.to_numpy()
        X_train = self.X_train.to_numpy()
        X_test = self.X_test.to_numpy()
        true_cate_train = self.true_cate_train['CATE'].to_numpy()
        true_cate_test = self.true_cate_test['CATE'].to_numpy()

    #Estimate the causal forest model
        est_cfdml = CausalForestDML(model_y='auto',
                            model_t=DummyClassifier(),
                            discrete_treatment=True,
                            cv=5,
                            n_estimators=1000,
                            random_state=42,
                            criterion='mse',
                            honest=True)

        est_cfdml.fit(Y=Y_train, T=T_train, X=X_train, W=None, cache_values=True)
        
        estimated_cate_cfdml_train = est_cfdml.effect(X_train)
        estimated_cate_cfdml_test = est_cfdml.effect(X_test)

        CF_DML_MSE_train = mean_squared_error(true_cate_train, estimated_cate_cfdml_train)
        CF_DML_MSE_test = mean_squared_error(true_cate_test, estimated_cate_cfdml_test)

        return estimated_cate_cfdml_train, estimated_cate_cfdml_test, CF_DML_MSE_train, CF_DML_MSE_test  




    def non_param_dml(self) -> tuple[np.ndarray, float]:

        Y_train = self.Y_train.to_numpy()
        T_train = self.T_train.to_numpy()
        X_train = self.X_train.to_numpy()
        X_test = self.X_test.to_numpy()
        true_cate_train = self.true_cate_train['CATE'].to_numpy()
        true_cate_test = self.true_cate_test['CATE'].to_numpy()

        est_npm = NonParamDML(
            model_y='auto',
            model_t=DummyClassifier(),
            model_final=RandomForestRegressor(),
            cv = 5,
            random_state = 42,
            discrete_treatment=True,
        )
        
        est_npm.fit(Y=Y_train, T=T_train, X=X_train, W=None, cache_values=True)

        estimated_cate_npm_train = est_npm.effect(X_train)
        estimated_cate_npm_test = est_npm.effect(X_test)

        NPM_MSE_Train = mean_squared_error(true_cate_train, estimated_cate_npm_train)
        NPM_MSE_Test = mean_squared_error(true_cate_test, estimated_cate_npm_test)

        return estimated_cate_npm_train, estimated_cate_npm_test, NPM_MSE_Train, NPM_MSE_Test

