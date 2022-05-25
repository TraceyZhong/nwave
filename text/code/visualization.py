from utils import *

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8, 6) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'

import numpy as np 
from matplotlib.ticker import FuncFormatter

import pickle
import argparse

# np.random.seed(0)

WORDS_TO_SHOW = 5

def tsne_plot(words, w_labels, w_colors, codes, c_colors, fpath, noise = None, \
    srepr = None, show_key = True, cor = False):
    '''
    Let d be the embedding_dim
    codes : K * d
    c_labesl : K
    words : nK * d
    w_labels : nK
    whitenoise : something
    '''
    # normalize codes
    codes = codes/ np.sqrt(np.square(codes).sum(axis = 1))[:,np.newaxis]

    if noise is not None:
        X = np.concatenate((words, noise,codes), axis = 0)
        n_colors = np.repeat(c_colors.max() + 1, len(noise))
        # colors = np.concatenate((w_colors, n_colors, c_colors))
    else:
        X = np.concatenate((words, codes), axis = 0)
        # colors = np.concatenate((w_colors,  c_colors))

    tsne = TSNE(n_components = 2, init = 'random')
    X_2d = tsne.fit_transform(X)
    
    ncodes = len(c_colors)
    fig, ax = plt.subplots(figsize = (10,10), dpi = 200)
    # words
    ax.scatter(X_2d[:-ncodes][:,0], X_2d[:-ncodes][:,1], c = w_colors, \
        cmap = 'tab10', s = 40)
    # code words
    ax.scatter(X_2d[-ncodes:][:,0], X_2d[-ncodes:][:,1], c = c_colors, \
        cmap = 'tab10', s = 200, marker = '*')
    if show_key:    
        for i in range(len(w_labels)):
            ax.annotate(
                w_labels[i],
                xy = X_2d[i],
                fontsize = 10
                # textcoords='oppset points'
            )
    
    if not cor:
        ax.set_title("tSNE representation of codes by activation")
        fig.savefig(fpath + 'tSNE{}.png'.format(srepr))
    if cor:
        ax.set_title("tSNE representation of codes by correlation")
        fig.savefig(fpath + 'tSNE{}cor.png'.format(srepr))
    plt.close()

def get_tops(activity, K, n):
    '''
    activated : num_inputs * num_units
    K : number of codewords to observe
    n : number of words activated by that codeword
    d : embedding dim
    '''
    # TODO maybe not the top K abs, just the values..
    # select codes words that has the largest activation
    activity = np.abs(activity)
    activated = activity.sum(axis = 0)
    topK_idx = np.argpartition(activated, -K)[-K:]
    # select words that has the largest activation with the 
    topN_idx = np.argpartition(activity[:,topK_idx], -n, axis = 0)[-n:]

    return topK_idx, topN_idx

def get_tops_by_cor(codes, n):
    
    word_embeddings = np.load('../data/obj/normalizedwe.npy')
    cor = word_embeddings @ codes.T

    topN_idx = np.argpartition(cor, -n, axis = 0)[-n:]
    return topN_idx


def get_closest_words(topN_idx, fpath, write = True):
    n, K = topN_idx.shape

    with open('../data/obj/idxvocab.pkl', 'rb') as f:
        idxvocab = pickle.load(f)
    w_labels = np.vectorize(idxvocab.get)(topN_idx).T
    if write:
        np.savetxt(fpath + "closestWords.csv", w_labels, \
            fmt = '%s', delimiter=',')
    return w_labels.flatten(order = 'C')

def get_words_embedding(Phi, topK_idx, topN_idx, d):
    '''
    K : number of codewords to observe
    n : number of words activated by that codeword
    d : embedding dim
    '''   

    topN_idx = topN_idx[:WORDS_TO_SHOW, :]

    n, K = topN_idx.shape

    w_labels = get_closest_words(topN_idx, srepr, write = False)

    c_colors = np.arange(K)
    codes = Phi[:,topK_idx].T

    w_colors = np.repeat(c_colors, n)
    words = np.empty(shape = (K*n, d))

    # get embeddings for each word.
    word_embeddings = np.load('../data/obj/normalizedwe.npy')
    for i in c_colors:
        words[(i*n):(i+1)*n,:] = word_embeddings[topN_idx[:,i]]
    
    return codes, c_colors, words, w_colors, w_labels

def vis_model(model, K, n, srepr):
    d = model.num_inputs 
    topK_idx, topN_idx = get_tops(model.activity, K, n)
    get_closest_words(topN_idx, srepr)
    codes, c_colors, words, w_colors, w_labels = get_words_embedding(model.Phi, \
        topK_idx, topN_idx, d, srepr = srepr)
    tsne_plot(words, w_labels, w_colors, codes, c_colors, srepr=srepr, show_key=True)


def vis_data(activity, Phi, K, n, d, fpath):
    topK_idx, topN_idx = get_tops(activity, K, n)
    get_closest_words(topN_idx, fpath)
    codes, c_colors, words, w_colors, w_labels = get_words_embedding(Phi, \
        topK_idx, topN_idx, d)
    for i,s in enumerate([1992,3445,20]):
        np.random.seed(s)
        tsne_plot(words, w_labels, w_colors, codes, c_colors, fpath, srepr= str(i), show_key=True)
    fpathcor = fpath + "cor"
    topN_idx = get_tops_by_cor(codes, n)
    get_closest_words(topN_idx, fpathcor)    
    codes, c_colors, words, w_colors, w_labels = get_words_embedding(Phi, \
        topK_idx, topN_idx, d)
    for i,s in enumerate([1992,3445,20]):
        np.random.seed(s)
        tsne_plot(words, w_labels, w_colors, codes, c_colors, fpathcor, srepr=str(i), show_key=True)
    

def vis_error(error, fpath, train_start_step = 0):
    fig, (ax0, ax1, ax2, at) = plt.subplots(nrows=1, ncols = 4, figsize = (20,3))
    ax0.plot(error[:,0])
    ax0.set_title("reconstruction error")
    ax0.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%d" % (train_start_step+x)))
    ax1.plot(error[:,1])
    ax1.set_title("l1norm")
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%d" % (train_start_step+x)))
    ax2.plot(error[:,2])
    ax2.set_title("l0norm")
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%d" % (train_start_step+x)))
    at.plot(error[:,3])
    at.set_title("step of convergence")
    at.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%d" % (train_start_step+x)))
    
    fig.suptitle(fpath[10:].replace('/', ' ').strip())
    fig.savefig(fpath + 'errors{}ts.png'.format(train_start_step))
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("lmda", type=int)
    parser.add_argument("srepr", type=str)

    args = parser.parse_args()

    srepr = args.srepr
    fpath = create_dir(srepr)

    K = 10 # we load at the top K neurons 
    n = 20 # for each neuron, we look at the most associated words
    d = 100 # word embedding

    Phi = np.load(fpath + "Phi.npy")
    activity = np.load(fpath + "activity.npy")
    vis_data(activity, Phi, K, n, d, fpath)
    error = np.load(fpath + "errors.npy")
    vis_error(error, fpath)