--------------------------------------
Files overview:
-------------------------------------- 

To reproduce the files of the paper, only **.ipynb** have to be executed. The **.py** scripts are mostly functions and methods that are then used in the jupyter notebooks.

The RMSE Files folder includes the .pkl files to replicate the main RMSE figures in the thesis. 

--------------------------------------
| File | Description |
| :---- | :-------- |
|Gen_data.py | Data generation class |
|Graph_feat.py | Sets functions to plot the descriptive statistics.|
|Descriptive.ipynb | Jupyter notebook that generates Figures 1, 2, B.1, B.2, and B.3, as well as Tables 1 and 2.|
|Methods_all_sample.py |Sets the estimation methods to be used in the study|
|Analysis_new.py|Specifies the RMSE simulations|
|Meta_Analysis.ipynb|Create simulations and save the RMSE values of all simulations in .pkl files|
|Graph_MSE.py |Sets functions to plot the results of the RMSE analysis|
|Graphical_analysis.ipynb|Generates the RMSE plots in Figures 3, 4, 5, 6, 7, B.4, and B.5 as well as Tables C.1, C.2, C.3, C.4, C.5, C.6, C.7, C.8|
|TLearner_feat_imp.py|Sets the T-learner and computes feature importance using the causalml library|
|feature_importance.py|Sets the function to get the partial dependence plots for the four estimators|
|Feat_importance_analysis.ipynb|Generates partial dependence plots depicted in Figures 8, 9, 10, and 11|
