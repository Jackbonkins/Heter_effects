-----------------------------------
Description
-----------------------------------
The repository includes files used for my master thesis titled "_Estimating Heterogeneous Treatment Effects in Randomized Controlled Trials: A Comparison of OLS and
Machine Learning Techniques_" at the Johannes Gutenberg University Mainz. The goal of the study was to compare how standard interactive linear regressions and machine learning techniques perform when estimating conditional average treatment effects (CATE) in randomized controlled trials of varied data complexity. Next to typical simulation parameters such as sample size and number of features, I also studied how complexity increases in terms of feature correlation in the DGP alters the performance of the selected estimators. 

All estimators were built on the premise of agnostic variable selection, meaning that the estimator had to be able to find the "correct" features in an increasingly spurious feature environment. For that, I used a "naive practitioner" approach to select the OLS interactions where all features in the data were interacted with the treatment indicator. The ML methods, the Two(T)-Learner, a Generalized Random Forest, and an R-Learner with a Generalized Random Forest in the final stage, used all available features to search for the relevant variables. The evaluation metrics for all methods were root mean squared error (RMSE) and a measure of feature importance, i.e., which variables explained **more** treatment heterogeneity.


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
