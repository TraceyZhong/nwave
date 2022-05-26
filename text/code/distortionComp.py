import numpy as np
from pallet import Unigram97Loader

#################################
# Computationß
#################################


N = 5000

# compute the l2 norm 
def get_dist(arr, fname):
    result = np.empty((N,N))
    
    ## If the computation doesn't exceed memory
    # diff =  arr[np.newaxis, ...] - np.expand_dims(arr, axis=1)
    # result = np.sum(np.square(diff), axis = -1)
    # np.save(fname, result)

    ## If the computation exceeds memory
    for i in range(50):
        print("At iter {}".format(i))
        diff =  arr[np.newaxis, ...] - np.expand_dims(arr[i*100:(i+1)*100,:], axis=1)
        result[i*100:(i+1)*100,:] = np.sum(np.square(diff), axis = -1)
    
    np.save(fname, result)

fpath = "../result/unigram97Loadertr0.2we30wi5lrW0.1lrA0.1/"

uactivity = np.load(fpath + "uactivity.npy")[:N,:] # shape (N, 1600)
get_dist(uactivity, fpath + "distortion_activity.npy")

loader = Unigram97Loader(0)
word_embed = loader.word_embeddings[:N,:].astype('float')
get_dist(word_embed, fpath + "distortion_embed.npy")

from sklearn.manifold import TSNE
tsned = TSNE(n_components=2, init='random').fit_transform(word_embed)
get_dist(tsned, fpath + "distortion_tsned.npy")


#################################
# Visualization
#################################

## Load data 

fpath = "../result/unigram97Loadertr0.2we30wi5lrW0.1lrA0.1/"

demb = np.load(fpath + "distortion_embed.npy")
dtsned = np.load(fpath +"distortion_tsned.npy")
dact = np.load(fpath +"distortion_activity.npy")


## Compute R-squared statistic

from scipy.stats import linregress

def bootstrap(xmat, ymat, B = 100, dat_name = ""):
    indices = np.triu_indices_from(xmat)
    xdat = np.asarray(xmat[indices])
    ydat = np.asanyarray(ymat[indices])
    Rsq = np.empty(B)
    for i in range(B):
        this_sample = np.random.choice(len(xdat), size = len(xdat))
        this_xdat = xdat[this_sample]
        this_ydat = ydat[this_sample]
        _,_,r_value, _,_ = linregress(this_xdat, this_ydat)
        Rsq[i] = r_value
    print("Bootstrapped estimate of the standard deviation of Rsq in {} data is {:.2e}, mean is {:.2f}".format(dat_name, np.std(Rsq), np.mean(Rsq)))

bootstrap(demb, dtsned, dat_name = "tsned")
bootstrap(demb, dact, dat_name = "nwave")


## Subsample for plotting

n = 1000

subsample = np.random.randint(0, 5000*5000, n)
xs = subsample // 5000
ys = np.mod(subsample, 5000)

demb = demb[xs, ys]
dtsned = dtsned[xs, ys]
dact = dact[xs, ys]

# np.save(fpath + "demb_1000.npy", demb)
# np.save(fpath + "dtsned_1000.npy", dtsned)
# np.save(fpath + "dact_1000.npy", dact)


## Plotting
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(demb, dact, s=10, c="red", alpha = 0.3, label = "nwave")
ax2 = ax.twinx()
ax2.scatter(demb, dtsned, marker = "x", s=10, c="blue", alpha = 0.3,  label = "tsned")


ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)

scat1, lab1 = ax.get_legend_handles_labels()
scat2, lab2 = ax2.get_legend_handles_labels()
lgnd = ax.legend(scat1 + scat2, lab1 + lab2, loc=2, fontsize=10)
## Enlarge points in legends
for handle in lgnd.legendHandles:
    handle.set_sizes([6.0])

fig.suptitle("Title")
fig.savefig(fpath + "distortion.png")







    


