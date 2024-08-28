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


def custom_sort(key):
    feature_order = {'low': 1, 'medium': 2, 'high': 3}
    parts = key.split('_')
    number = int(parts[1])
    feature_level = parts[2]
    return (number, feature_order[feature_level])



def sort_dict(rmse_dict: dict) -> dict:
    # Sort the keys using the custom sort function
    sorted_keys = sorted(rmse_dict.keys(), key=custom_sort)

    # Create a new sorted dictionary
    sorted_rmse_dict = {key: rmse_dict[key] for key in sorted_keys}

    return sorted_rmse_dict




def plot_rmse_analysis_test(rmse_dict: dict, analysis_type: str, data = 'Test',  p_list = [20,30,40,50]):
    
    

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
    axs = gs.subplots()
    
    handles, labels = [], []

    for j in range(0, row_no):
        p = next(p_list)[1]
        mean_correlation_list = enumerate(['0.1', '0.5', '0.8'])

        for i in range(0, 3):
            rmse_name = next(rmse_list)[1]
            #markers = enumerate(['o', 'x', '+', 'v', 's'])
            colors = enumerate(['red', 'green', 'blue', 'purple'])
            #colors = enumerate(['green', 'blue', 'purple'])
            mean_correlation = next(mean_correlation_list)[1]

            for est, rmse_df in rmse_dict[rmse_name].items():
                color = next(colors)
                #marker = next(markers)
                rmse_column = f'RMSE {data}'
                axs[j, i].plot(rmse_df['n'], rmse_df[rmse_column], marker='o', color=color[1], label=est, linewidth=1)
                
                if j == 0 and i == 0:  # Collect labels and handles from the first subplot
                    handles.append(axs[j, i].plot([],[], marker='o', color=color[1], label=est)[0])
                    labels.append(est)
                #scatter = axs[j, i].scatter(rmse_df['n'], rmse_df[rmse_column], s=20, marker='o-', color = color[1], label=est, linewidths=1)

                #if j == 0 and i == 0:  # Collect labels and handles from the first subplot
                 #   handles.append(scatter) 
                  #  labels.append(est)
                

            axs[j, i].set_title(f'p $=$ {p}, Mean Correlation $=$ {mean_correlation}')
            
        
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

    #plt.show()
    #plt.savefig(fname = f'{analysis_type}_RMSE_Analysis_{data}.pgf', bbox_inches = 'tight')
    plt.savefig(fname = f'{analysis_type}_RMSE_Analysis_{data}', bbox_inches = 'tight')






