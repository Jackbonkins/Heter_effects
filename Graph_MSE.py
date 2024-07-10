import matplotlib.pyplot as plt


def plot_mse_analysis_test(mse_list: list, p_list: list):
      
    plt.style.use('seaborn-v0_8')

    mse_list = enumerate(mse_list)
    p_list = enumerate(p_list)

    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(nrows=2, ncols=2, hspace=0.2)
    axs = gs.subplots(sharex=True, sharey=True)
    
    for j in range(0,2): 
        for i in range(0, 2):

            colors = enumerate(['red', 'green', 'blue', 'yellow'])
            mse_dict = next(mse_list)[1]
            
            for est, mse_df in mse_dict.items():
                color = next(colors)
                
                axs[j, i].scatter(mse_df['n'], mse_df['MSE Test'], s=20, marker='o', color = color[1], label=est)

            p = next(p_list)[1]
            axs[j, i].set_title(f'p $=$ {p}')


        if j == 0 and i==1:
            axs[j, i].legend()
        else:
            continue
    # Create a single legend for the entire figure

    #fig.suptitle('Influence of Individual Feature Values on the CATE Function', y=0.96)
    fig.text(0.5, 0.04, 'No. of Observations', ha='center')
    fig.text(0.04, 0.5, 'MSE Test', va='center', rotation='vertical')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    #for ax in axs.flat:
     #   ax.label_outer()