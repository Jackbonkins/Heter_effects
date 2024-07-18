import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'Times New Roman',
    'font.size' : 12,
    'text.usetex': True,
    'pgf.rcfonts': False,
    })


def custom_sort(key):
    feature_order = {'low': 1, 'medium': 2, 'high': 3}
    parts = key.split('_')
    number = int(parts[1])
    feature_level = parts[2]
    return (number, feature_order[feature_level])



def sort_dict(mse_dict: dict) -> dict:
    # Sort the keys using the custom sort function
    sorted_keys = sorted(mse_dict.keys(), key=custom_sort)

    # Create a new sorted dictionary
    sorted_mse_dict = {key: mse_dict[key] for key in sorted_keys}

    return sorted_mse_dict



def plot_mse_analysis_test(mse_dict: dict, style: str, analysis_type: str, data = 'Test',  p_list = [20,30,40,50]):
    
    plt.style.use(style)

    row_no = len(p_list)
    mse_dict = sort_dict(mse_dict)

    mse_list = []

    for mse_name in mse_dict.keys():
        mse_list.append(mse_name)
    
    mse_list = enumerate(mse_list)
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
            mse_name = next(mse_list)[1]
            colors = enumerate(['red', 'green', 'blue', 'yellow', 'purple'])
            mean_correlation = next(mean_correlation_list)[1]

            for est, mse_df in mse_dict[mse_name].items():
                color = next(colors)
                mse_column = f'MSE {data}'
                scatter = axs[j, i].scatter(mse_df['n'], mse_df[mse_column], s=20, marker='o', color = color[1], label=est)

                if j == 0 and i == 0:  # Collect labels and handles from the first subplot
                    handles.append(scatter) 
                    labels.append(est)
                

            axs[j, i].set_title(f'p $=$ {p}, Mean Correlation $=$ {mean_correlation}')
            
        
    fig.text(0.5, 0.05, 'No. of Observations', ha='center')

    if data == 'Train':
        fig.text(0.08, 0.5, 'MSE Train', va='center', rotation='vertical')

    else:
        fig.text(0.08, 0.5, 'MSE Test', va='center', rotation='vertical')

    
# Create a single legend for the entire figure in a box above the bottom axis label
    legend = fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.005), frameon=True)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_facecolor('white')
    
    plt.subplots_adjust(bottom=0.10)  # Adjust the bottom to make space for the legend and labels

    #plt.show()
    plt.savefig(fname = f'{analysis_type}_MSE_Analysis_{data}.pgf', bbox_inches = 'tight')
    plt.savefig(fname = f'{analysis_type}_MSE_Analysis_{data}', bbox_inches = 'tight')






