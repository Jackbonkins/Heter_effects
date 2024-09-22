
'''

File is used to generate the MSE plots from the rmse dictionaries.


'''


import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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




#Sort dictionaries by correlation category and feature number
def custom_sort_rmse_total(key):
    feature_order = {'low': 1, 'medium': 2, 'high': 3}
    parts = key.split('_')
    number = int(parts[1])
    feature_level = parts[2]
    return (number, feature_order[feature_level])



#Apply custom sort function to the dictionaries
def sort_dict(rmse_dict: dict) -> dict:

    sorted_keys = sorted(rmse_dict.keys(), key=custom_sort_rmse_total)
    sorted_rmse_dict = {key: rmse_dict[key] for key in sorted_keys}

    return sorted_rmse_dict



#Create RMSE plot
def plot_rmse_analysis_test(rmse_dict: dict, analysis_type: str, data = 'Test',  p_list = [20,30,40,50], estimators = ['OLS', 'T-Learner', 'GRF', 'CF DML']):
    
    row_no = len(p_list)
    rmse_dict = sort_dict(rmse_dict)

    rmse_list = []

    for rmse_name in rmse_dict.keys():
        rmse_list.append(rmse_name)
    
    rmse_list = enumerate(rmse_list)
    p_list = enumerate(p_list)
    mean_correlation_list = enumerate(['0.1', '0.5', '0.8'])
    
    
    fig = plt.figure(figsize=(15,15))
    gs = fig.add_gridspec(nrows=row_no, ncols=3, hspace=0.4)
    axs = gs.subplots(sharey=True)
    
    
    handles, labels = [], []

    for j in range(0, row_no):
        p = next(p_list)[1]
        mean_correlation_list = enumerate(['0.1', '0.5', '0.8'])

        for i in range(0, 3):
            rmse_name = next(rmse_list)[1]
            colors = enumerate(['red', 'green', 'blue', 'darkorange'])
            mean_correlation = next(mean_correlation_list)[1]

            for est, rmse_df in rmse_dict[rmse_name].items():

                if est not in estimators:
                    color = next(colors)
                    continue
                else:
                    color = next(colors)
                    #marker = next(markers)
                    rmse_column = f'RMSE {data}'
                    axs[j, i].plot(rmse_df['n'], rmse_df[rmse_column], marker='o', color=color[1], label=est, linewidth=1)
                    axs[j,i].yaxis.set_major_locator(MaxNLocator(integer=True))
                
                
                    if j == 0 and i == 0:  # Collect labels and handles from the first subplot
                        handles.append(axs[j, i].plot([],[], marker='o', color=color[1], label=est)[0])
                        labels.append(est)
                
                

            axs[j, i].set_title(f'p $=$ {p}, Mean Correlation $=$ {mean_correlation}')
            axs[j,i].yaxis.set_major_locator(MaxNLocator(integer=True))

            
        
    fig.text(0.5, 0.05, 'No. of Observations', ha='center')

    if data == 'Train':
        fig.text(0.08, 0.5, 'RMSE Train', va='center', rotation='vertical')

    else:
        fig.text(0.08, 0.5, 'RMSE Test', va='center', rotation='vertical')

    
# Create a single legend for the entire figure in a box above the bottom axis label
    legend = fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.005), frameon=True)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_facecolor('white')
    
    plt.subplots_adjust(bottom=0.10)  # Adjust the bottom to make space for the legend and labels

    if 'OLS' not in estimators:
        plt.savefig(fname = f'{analysis_type}_RMSE_Analysis_{data}_no_OLS', bbox_inches = 'tight')
    else:
        plt.savefig(fname = f'{analysis_type}_RMSE_Analysis_{data}', bbox_inches = 'tight')



#Average RMSE values over feature numbers and sample size
def average_rmse_by_category(data_dict: dict, estimators = ['OLS', 'T-Learner', 'GRF', 'CF DML']):
    combined_dict = {}

    correlations = ['low', 'medium', 'high']
    
    for correlation in correlations:
        combined_dict[correlation] = {} 
        
        for estimator in estimators:
            combined_df = pd.DataFrame()
            
            keys_to_process = [key for key in data_dict if correlation in key]
            
            for key in keys_to_process:
                df = data_dict[key][estimator].copy() 
                df.set_index('n', inplace=True)
                df_train = df[['RMSE Test']]
                df = df_train.T
                combined_df = pd.concat([combined_df, df])
        
            avg_rmse = combined_df.mean(numeric_only=True).to_frame()
            avg_rmse.rename(columns={0: 'RMSE Test'}, inplace=True) 
            combined_dict[correlation][estimator] = avg_rmse

    return combined_dict
    



#Defines the function to plot the relationship between the RMSE and the correlation categories
def plot_rmse_corr(data_dict: dict):
    
    avg_mse_cat_dict = average_rmse_by_category(data_dict)
    
    plot_data = []

    # Loop over each correlation and estimator to extract RMSE values
    for correlation, estimator_dict in avg_mse_cat_dict.items():
        for estimator, df in estimator_dict.items():
            avg_rmse_test = df['RMSE Test'].mean()  # Get the mean RMSE Test value
            plot_data.append({
                'Correlation': correlation.capitalize(),
                'Estimator': estimator,
                'RMSE Test': avg_rmse_test
            })

    # Convert the list to a DataFrame
    plot_df = pd.DataFrame(plot_data)

    # Plot the data
    plt.figure(figsize=(8, 6))
    colors = enumerate(['red', 'green', 'blue', 'darkorange'])

    # Plot RMSE Train values
    for estimator in plot_df['Estimator'].unique():
        color = next(colors)
        subset = plot_df[plot_df['Estimator'] == estimator]
        plt.plot(subset['Correlation'], subset['RMSE Test'], marker='o', label=f'{estimator}', color=color[1])

    # Customize the plot
    plt.xlabel('Correlation')
    plt.ylabel('RMSE Test')
    plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.2), frameon=True)
    plt.savefig(fname=f'rmse_corr', bbox_inches='tight')




