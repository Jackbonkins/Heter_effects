import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from econml.grf import CausalForest
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-colorblind')
matplotlib.use("pgf")
plt.rcParams.update({
    'pgf.texsystem': 'lualatex',  
    'font.size': 12,
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'pgf.rcfonts': False,
    'pgf.preamble': r'\usepackage{fontspec} \setmainfont{Times New Roman}',
})



#Function sorts features by their importance to the estimation
def get_important_feats(X_test: pd.DataFrame, feat_importance: np.ndarray):

    important_feats = X_test.columns[np.argsort(feat_importance)[::-1]]
    return important_feats



#Function is used to generate the PDP 
def partial_dependence_plots_ML(X_test: pd.DataFrame, important_feats: np.ndarray, est, setting: str):

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    
    for it, feature in enumerate(important_feats[:3]):
        ax = axes[it]
        
        grid = np.unique(np.percentile(X_test[feature], np.arange(0, 105, 5)))
        Zpd = pd.DataFrame(np.tile(np.median(X_test, axis=0, keepdims=True), (len(grid), 1)),
                        columns=X_test.columns)

        Zpd[feature] = grid

        if isinstance(est, CausalForestDML):            
            preds = est.effect(Zpd)
            lb, ub = est.effect_interval(Zpd)
            lb = lb.flatten()
            ub = ub.flatten()
            preds = preds.flatten()
            est_name = 'CFDML'
            ax.errorbar(Zpd[feature], preds, yerr=(preds - lb, ub - preds), fmt="--o", ecolor="black", elinewidth=0.75, color='darkorange', markerfacecolor='darkorange', markeredgecolor='darkorange')

            ax.plot(Zpd[feature], preds, color='darkorange')
            ax.set_xlabel(feature)

        elif isinstance(est, CausalForest):
            preds, lb, ub = est.predict(Zpd, interval=True, alpha=0.05)
            lb = lb.flatten()
            ub = ub.flatten()
            preds = preds.flatten()
            est_name = 'GRF'
            ax.errorbar(Zpd[feature], preds, yerr=(preds - lb, ub - preds), fmt="b--o", ecolor="black", elinewidth=0.75)

            ax.plot(Zpd[feature], preds, color='blue')
            ax.set_xlabel(feature)

        else:
            preds = est.effect(Zpd)
            preds = preds.flatten()
            est_name = 'T_Learner'
            ax.errorbar(Zpd[feature], preds, fmt="g--o", ecolor="black", elinewidth=0.5)

            ax.plot(Zpd[feature], preds, color='green')
            ax.set_xlabel(feature)
    
    fig.text(0.04, 0.5, 'Predicted CATE', va='center', ha='center', rotation='vertical')

    plt.tight_layout(rect=[0.05, 0, 1, 1]) 

    plt.savefig(fname=f'Feature_imp_{est_name}_{setting}', bbox_inches='tight')



#Plot the PDP of the OLS
def plot_OLS(X_test: pd.DataFrame, coeff: pd.DataFrame, conf_int: pd.DataFrame, setting: str):

    important_feats = X_test.columns[np.argsort(np.abs(coeff[0].to_numpy()))[::-1]]
    ci_subset = conf_int.loc[important_feats[:3]]

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

    for it, feature in enumerate(important_feats[:3]):
        ax = axes[it]
       
        grid = np.unique(np.percentile(X_test[feature], np.arange(0, 105, 5)))
        Zpd = pd.DataFrame(np.tile(np.median(X_test, axis=0, keepdims=True), (len(grid), 1)),
                        columns=X_test.columns)
        Zpd[feature] = grid
        estimated_cate_ols_test = Zpd.dot(coeff.T.to_numpy().flatten())

        lb =  np.tile(ci_subset['lower'].loc[feature], len(grid))
        ub =  np.tile(ci_subset['upper'].loc[feature], len(grid))

        errors = (ub - lb) / 2

        ax.errorbar(Zpd[feature], estimated_cate_ols_test, yerr=errors, fmt="r--o", ecolor="black", elinewidth=0.5)
        ax.plot(Zpd[feature], estimated_cate_ols_test, color='red')
        ax.set_xlabel(feature)
    
    fig.text(0.04, 0.5, 'Predicted CATE', va='center', ha='center', rotation='vertical')

    plt.tight_layout(rect=[0.05, 0, 1, 1])  

    plt.savefig(fname=f'Feature_imp_OLS_{setting}', bbox_inches='tight')







        
