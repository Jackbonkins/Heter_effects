
'''

Uses the causalml library to get the feature importance for the T-learner.

Saves the resulting array into a .npy file that is then used in feat_importance_analysis.ipynb.

'''

from causalml.inference.meta import BaseTRegressor as TLearner
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

data_files = ['train_df', 'X_train', 'Y_train', 'T_train', 'X_test', 'T_test']

#Loads the simulation data 
data_dict = {}
for file_name in data_files:
    with open(f'{file_name}.pkl', 'rb') as file:
        globals()[file_name] = pickle.load(file)

    key = f'{file_name}'
    data_dict[key] = globals()[file_name]


x_col = [col for col in data_dict['train_df'].columns if col.startswith('X')]

X_train = data_dict['X_train']
y_train = data_dict['Y_train']
T_train = data_dict['T_train']
X_test = data_dict['X_test']
T_test = data_dict['T_test']

#Estimates T-learner using the RF regressor
nuisance_model = RandomForestRegressor(random_state=220924, n_estimators=1000, n_jobs=-1)
T_learner = TLearner(nuisance_model)
T_learner.fit(X=X_train, treatment=T_train, y=y_train)
tlearner_cate = T_learner.predict(X_test, treatment=T_test)

#Get feature importance using permutation
feat_importance = T_learner.get_importance(X=X_test, tau=tlearner_cate, method='permutation', features=x_col, model_tau_feature=nuisance_model)

#Save the resulting features into an array
important_feats = np.array(feat_importance[1].index.values)
np.save('feat_importance_TLearner.npy', feat_importance)
np.save('causalml_cate_tlearner.npy', tlearner_cate)
