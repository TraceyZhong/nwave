'''
ReceptiveFields for text data
'''

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams['figure.figsize'] = (8, 6) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'


from wavenet import wavenetSC

#################################
# Plotting func
#################################


def plot_colortable(colors, text_on = True):
    # ref: https://matplotlib.org/stable/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py
    nunit = len(colors) 
    side_length = int(np.sqrt(nunit))
    swatch_width = cell_width = cell_height = 32
    # set figs
    ncols = nrows = side_length
    width = cell_width * ncols
    height = cell_height * nrows
    dpi = 72
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    # set ax axis
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows), 0 )
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for unit_idx, nunit in enumerate(range(nunit)):
        row = unit_idx // ncols
        col = unit_idx % ncols
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col # + swatch_width
        
        if text_on:
            ax.text(text_pos_x + cell_width/2, y+cell_height/2, unit_idx, fontsize=6, horizontalalignment='center', verticalalignment='center')

        ax.add_patch(Rectangle(xy=(swatch_start_x, y), width=swatch_width, height=swatch_width, facecolor=colors[unit_idx], edgecolor='0.7')
        )

    return fig



def rescale(vec, qt):
    qtmin = np.quantile(vec, qt, axis = 1)[:,np.newaxis]
    qtmax = np.quantile(vec, 1-qt, axis = 1)[:, np.newaxis]
    return np.minimum(np.maximum((vec - qtmin)/(qtmax - qtmin), 0),1)

def get_colors(Vt, alpha = 0.5):
    _, n = Vt.shape
    colors = []
    for i in range(n):
        colors.append(( *Vt[:, i] , alpha))
    return colors

def plot_PCA(Phi, filename=''):
    U, S, Vt = np.linalg.svd(Phi.T , full_matrices=False)
    principal_score = U @ np.diag(S)[:,:3]
    principal_scoreT = rescale(principal_score.T, 0.05)
    colors = get_colors(principal_scoreT, alpha= 0.8)
    fig = plot_colortable(colors, text_on=False)
    if len(filename) > 0:
        fig.savefig(filename, bbox_inches='tight')
    plt.close()


#################################
# Compute Receptive fields
#################################

import argparse

def parse_argv():
    parser = argparse.ArgumentParser(prog='SC')
    parser.add_argument("--loader", type = str)
    parser.add_argument("--sparsity", type=float)
    parser.add_argument("--local", action="store_true")

    parser.add_argument("--we", type = int)
    parser.add_argument("--wi", type = int)
    
    return parser.parse_args()



## --- Experiment set up --- ##

def get_fpath_from_args(args):
    return "../result/{}tr{}we{}wi{}lrW0.1lrA0.1/".format(args.loader, args.sparsity, \
        args.we, args.wi)

if __name__=="__main__":

    args = parse_argv()

    fpath = get_fpath_from_args(args)

    Phi = np.load(fpath + "W1_t50000.npy")

    emb_dim, num_units = Phi.shape

    wnsc = wavenetSC(num_inputs=emb_dim, num_outputs=1, num_units=num_units, batch_size=emb_dim, lmda=.050, 
                lr_r=0.01, lr_W1=0.01, r1=3, r2=5, wi=30, we=5, sigmaE=3, \
                    target_sparsity=0.2, \
                        fpath = fpath)

    batch = np.eye(emb_dim)
    batch = batch - np.mean(batch, axis=1)
    batch = batch / np.std(batch, axis=1)


    wnsc.W1 = Phi
    wnsc.lmda = wnsc.cwsc.lmda = 0.007
    RC, s = wnsc.compute_activations(batch)

    np.save(fpath + "receptive_fields.npy", RC)

    plot_PCA(RC, fpath + 'RC.pdf')




