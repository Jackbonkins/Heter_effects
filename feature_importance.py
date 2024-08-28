import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from econml.grf import CausalForest
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-colorblind')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'Times New Roman',
    'font.size' : 12,
    'text.usetex': True,
    'pgf.rcfonts': False,
    })

def get_important_feats(X_test, feat_importance):
    important_feats = X_test.columns[np.argsort(feat_importance)[::-1]]
    return important_feats


def partial_dependence_plots(X_test, important_feats, est):

    plt.figure(figsize=(10, 7))
    for it, feature in enumerate(important_feats[:4]):
        plt.subplot(2, 2, it + 1)

        grid = np.unique(np.percentile(X_test[feature], np.arange(0, 105, 5)))
        Zpd = pd.DataFrame(np.tile(np.median(X_test, axis=0, keepdims=True), (len(grid), 1)),
                        columns=X_test.columns)

        Zpd[feature] = grid

        #Zpd = pd.DataFrame(np.tile(np.median(X_test, axis=0, keepdims=True), (len(X_test), 1)),
         #               columns=X_test.columns)
        

        if isinstance(est, CausalForestDML):            
            preds=est.effect(Zpd)
            lb, ub = est.effect_interval(Zpd)
            lb = lb.flatten()
            ub = ub.flatten()
            preds = preds.flatten()

            plt.errorbar(Zpd[feature], preds, yerr=(preds - lb, ub - preds), fmt="r--o", ecolor = "black", elinewidth=0.75)

        elif isinstance(est, CausalForest):
            preds, lb, ub = est.predict(Zpd, interval = True, alpha= 0.05)
            lb = lb.flatten()
            ub = ub.flatten()
            preds = preds.flatten()

            plt.errorbar(Zpd[feature], preds, yerr=(preds - lb, ub - preds), fmt="r--o", ecolor = "black", elinewidth=0.75)

        else:
            preds=est.predict(Zpd)
            preds = preds.flatten()
            plt.errorbar(Zpd[feature], preds, fmt="r--o", ecolor = "black", elinewidth=0.5)
        #plt.fill_between(Zpd[feature], preds - lb, preds + ub, color='lightblue', alpha=0.5, label='95% CI')


        plt.plot(Zpd[feature], preds, color='red', label='Predicted CATE')
        plt.xlabel(feature)
        plt.ylabel('Predicted CATE')
    plt.tight_layout()
    plt.savefig(f'cf-marginal-plots.png', dpi=600)
    plt.show()


#sim: SimulationStudy = SimulationStudy(p=35, mean_correlation=0.5, cor_variance=0.2, n=1500, no_feat_cate=3, non_linear='linear')
#simulation_linear = sim.create_dataset()
#train_df_linear, test_df_linear, X_train_linear, Y_train_linear, T_train_linear, X_test_linear, T_test_linear, y_test_linear, true_cate_train_linear, true_cate_test_linear = get_split(simulation_linear)

#estimated_cate_train, estimated_cate_test, ci_bounds, RMSE_test, RMSE_train = method.CF_DML(Y_train_linear, T_train_linear, X_train_linear, 
                                                                                            #X_test_linear, true_cate_train_linear, true_cate_test_linear)




#sim: SimulationStudy = SimulationStudy(p=35, mean_correlation=0.5, cor_variance=0.2, n=1500, no_feat_cate=3, non_linear='quadratic')
#simulation_quad = sim.create_dataset()
#train_df_quad, test_df_quad, X_train_quad, Y_train_quad, T_train_quad, X_test_quad, T_test_quad, y_test_quad, true_cate_train_quad, true_cate_test_quad = get_split(simulation_quad)
