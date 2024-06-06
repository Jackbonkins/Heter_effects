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



    
    def __init__(self, X_train: pd.DataFrame, T_train: pd.DataFrame, Y_train: pd.DataFrame, X_test: pd.DataFrame, 
                 T_test: pd.DataFrame, Y_test: pd.DataFrame, true_cate_test: pd.DataFrame) -> tuple[pd.DataFrame | np.array, float]:

        self.X_train = X_train
        self.T_train = T_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.T_test = T_test
        self.Y_test = Y_test
        self.true_cate_test = true_cate_test
    
    


    def ols_estimator(self) -> tuple[pd.DataFrame, float]:

        interactions_train = pd.DataFrame()
        interactions_test = pd.DataFrame()


        for col in self.X_train.columns:
            interactions_train[f'{col}_T'] = self.X_train[col]*self.T_train
            interactions_test[f'{col}_T'] = self.X_test[col]*self.T_test


        X_train_ols = self.X_train.join(interactions_train)
        X_test_ols = self.X_test.join([interactions_test, pd.DataFrame(self.T_test, columns=['T'])])


        #Create OLS object
        ols = LinearRegression()

        #Fit OLS 
        ols.fit(X_train_ols, self.Y_train)

        #Estimate CATE
        X_test_ols = X_test_ols[X_test_ols['T']==1]
        X_test_ols = X_test_ols[[col for col in X_test_ols.columns if col.startswith('X') and col.endswith('T')]]
        
        coeff = pd.DataFrame(ols.coef_, X_train_ols.columns).T
        coeff = coeff[[col for col in coeff.columns if col.startswith('X') and col.endswith('T')]].T
        
        estimated_cate_ols = X_test_ols.dot(coeff)
            
        #Calculate MSE
        true_cate_test = self.true_cate_test[self.true_cate_test['T'] == 1]
        true_cate_test = true_cate_test['CATE']

        OLS_MSE = mean_squared_error(true_cate_test, estimated_cate_ols)

        return estimated_cate_ols, OLS_MSE 
    



    def TLearner_estimator(self) -> tuple[np.ndarray, float]:

        Y_train = self.Y_train.to_numpy()
        T_train = self.T_train.to_numpy()
        X_train = self.X_train.to_numpy()
        X_test = self.X_test.to_numpy()
        true_cate_test = self.true_cate_test['CATE'].to_numpy()
        
        est_t = TLearner(models=RandomForestRegressor())
        est_t.fit(Y=Y_train, T=T_train, X=X_train)
        estimated_cate_t = est_t.effect(X_test)
        T_MSE = mean_squared_error(true_cate_test, estimated_cate_t)

        return estimated_cate_t, T_MSE
    



    def CF_DML(self) -> tuple[np.ndarray, float]:

        Y_train = self.Y_train.to_numpy()
        T_train = self.T_train.to_numpy()
        X_train = self.X_train.to_numpy()
        X_test = self.X_test.to_numpy()
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
        estimated_cate_cfdml = est_cfdml.effect(X_test)
        CF_DML_MSE = mean_squared_error(true_cate_test, estimated_cate_cfdml)
        
        return estimated_cate_cfdml, CF_DML_MSE 




    def non_param_dml(self) -> tuple[np.ndarray, float]:

        Y_train = self.Y_train.to_numpy()
        T_train = self.T_train.to_numpy()
        X_train = self.X_train.to_numpy()
        X_test = self.X_test.to_numpy()
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
        estimated_cate_npm = est_npm.effect(X_test)
        NPM_MSE = mean_squared_error(true_cate_test, estimated_cate_npm)

        return estimated_cate_npm, NPM_MSE

