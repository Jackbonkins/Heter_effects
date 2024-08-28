import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-colorblind')
matplotlib.use('pgf')

#from matplotlib.backends.backend_pgf import FigureCanvasPgf
#matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)


plt.rcParams.update({
    'pgf.texsystem': 'lualatex',  # Use xelatex
    'font.size': 12,
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'pgf.rcfonts': False,
    'pgf.preamble': r'\usepackage{fontspec} \setmainfont{Times New Roman}',
})



from Gen_data import SimulationStudy



def plot_cate_feat_val(p: int, n: int, mean_correlation: float, no_feat_cate: int, function: str, total=True):

    

    sim: SimulationStudy = SimulationStudy(p=p, mean_correlation=mean_correlation, cor_variance=0.2, n=n, no_feat_cate=no_feat_cate, non_linear=function)
    simulation = sim.create_dataset()

    feat_no = sim.__getattribute__('p')
    cate_feat_no = sim.__getattribute__('no_feat_cate')
    
    y = simulation['CATE']

    if total is True:
            
        fig = plt.figure(figsize=(10,10))
        gs = fig.add_gridspec(nrows=5, ncols=4, hspace=0.4)
        axs = gs.subplots(sharex=True, sharey=True)

        feat = 0
        for j in range(0, 5): 
            for i in range(0, 4):
                feature = f'X{feat}'
                x = simulation[feature]
                        
                if feat < cate_feat_no:
                    axs[j, i].scatter(x, y, s=5, marker='o', color = 'red')
                    axs[j, i].set_title(f'$X_{{{feat}}}$')

                else:
                    axs[j, i].scatter(x, y, s=5, marker='o', color = 'blue')
                    axs[j, i].set_title(f'$X_{{{feat}}}$')

                if feat < feat_no-1:
                    feat += 1
                else:
                    feat = feat

    else:
        fig = plt.figure(figsize=(7,5))
        gs = fig.add_gridspec(nrows=1, ncols=cate_feat_no, hspace=0.4)
        axs = gs.subplots(sharex=True, sharey=True)

        feat = 0
        for i in range(0, 3):
            feature = f'X{feat}'
            x = simulation[feature]
                        
            axs[i].scatter(x, y, s=5, marker='o', color = 'red')
            axs[i].set_title(f'$X_{{{feat}}}$')


            feat += 1

    #fig.suptitle('Influence of Individual Feature Values on the CATE Function', y=0.96)
    fig.text(0.5, 0.001, 'Feature Values', ha='center')  # Adjust vertical position
    fig.text(0.05, 0.5, 'CATE', va='center', rotation='vertical')  # Adjust horizontal position

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    #plt.show()
    #plt.savefig(fname = f'Graph_feat_{function}.pgf', bbox_inches = 'tight')

    if total is True:
        plt.savefig(fname = f'Graph_feat_{function}', bbox_inches = 'tight')
    else:
        plt.savefig(fname = f'Graph_feat_CATE_{function}', bbox_inches = 'tight')

