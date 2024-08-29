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



def get_important_feats(X_test, feat_importance):
    important_feats = X_test.columns[np.argsort(feat_importance)[::-1]]
    return important_feats




def partial_dependence_plots(X_test, important_feats, est):

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
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
            ax.errorbar(Zpd[feature], preds, yerr=(preds - lb, ub - preds), fmt="r--o", ecolor="black", elinewidth=0.75)

        elif isinstance(est, CausalForest):
            preds, lb, ub = est.predict(Zpd, interval=True, alpha=0.05)
            lb = lb.flatten()
            ub = ub.flatten()
            preds = preds.flatten()
            est_name = 'GRF'
            ax.errorbar(Zpd[feature], preds, yerr=(preds - lb, ub - preds), fmt="r--o", ecolor="black", elinewidth=0.75)

        else:
            preds = est.effect(Zpd)
            preds = preds.flatten()
            est_name = 'T_Learner'
            ax.errorbar(Zpd[feature], preds, fmt="r--o", ecolor="black", elinewidth=0.5)

        ax.plot(Zpd[feature], preds, color='red')
        ax.set_xlabel(feature)
    
    # Set a common y-label
    fig.text(0.04, 0.5, 'Predicted CATE', va='center', ha='center', rotation='vertical')

    plt.tight_layout(rect=[0.05, 0, 1, 1])  # Adjust layout to make room for the y-label

    plt.savefig(fname=f'Feature_imp_{est_name}', bbox_inches='tight')