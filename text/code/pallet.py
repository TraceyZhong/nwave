'''
==========
Load word pairs.
==========
'''


## --- Word pairs with pmi larger than threshold --- ##
'''
ps = np.linspace(0,1,20)
np.quantile(M, ps)
array([-5.62402358, -1.00286085, -0.62600728, -0.36646464, -0.15152349,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.03084317,  0.21707535,
        0.41095752,  0.62671839,  0.88974549,  1.27789935,  9.40343387])
PMI_THRESHOLD = 0
0.3244387534030429
'''
import numpy as np

class LoaderFactory:
    """ The factory class for creating executors"""

    registry = {}
    """ Internal registry for available executors """

    @classmethod
    def register(cls, name: str):
        """ Class method to register Executor class to the internal registry.
        Args:
            name (str): The name of the executor.
        Returns:
            The Executor class itself.
        """

        def inner_wrapper(wrapped_class):
            if name in cls.registry:
                logger.warning('SC algorithm %s already exists. Will replace it', name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper
    # End register 

def load_pairs(batch_size, embedding_dim):
    subsampled_idx = np.random.randint(0, num_word_pairs, SUBSAMPLE_SIZE)
    prob = pmi_vals[subsampled_idx]
    prob = prob / np.abs(prob).sum()
    sampled_locs = np.random.choice(a = subsampled_idx, size = batch_size, p=prob)
    Xvs_idx = pmi_locs[0][sampled_locs]
    Xws_idx = pmi_locs[1][sampled_locs]

    word_pairs = np.empty(shape = (batch_size, 2, embedding_dim))
    word_pairs[:,0,:] = word_embeddings[Xvs_idx,:]
    word_pairs[:,1,:] = word_embeddings[Xws_idx,:]
    return word_pairs, np.array([Xvs_idx, Xws_idx])



class PairLoader:
    def __init__(self, batch_size, embedding_dim = 100, nemb=False, cos = False, **kwargs):
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        if not cos:
            pmi = np.load('../data/obj/pmi.npy')
            PMI_THRESHOLD = 0.01
        else:
            pmi = np.abs(np.load('../data/obj/cosEmb.npy'))
            PMI_THRESHOLD = 0.2
        # some words are not in the glove embedding
        pmi[:,(2867,2881)] = PMI_THRESHOLD - 1
        pmi[(2867,2881),:] = PMI_THRESHOLD - 1
        pmi = np.tril(pmi, k = -1)

        self.pmi_locs = list(np.where(pmi > PMI_THRESHOLD))
        self.pmi_vals = pmi[tuple(self.pmi_locs)]
        del pmi
        if not nemb:
            self.word_embeddings = np.load('../data/obj/word_embeddings.npy')
        else:
            self.word_embeddings = np.load('../data/obj/normalizedwe.npy')
        self.num_word_pairs = len(self.pmi_vals)
        self.SUBSAMPLE_SIZE = 1000

    def load_train_batch(self, batch_size = 0):
        if batch_size == 0:
            batch_size = self.batch_size
        subsampled_idx = np.random.randint(0, self.num_word_pairs, self.SUBSAMPLE_SIZE)
        prob = self.pmi_vals[subsampled_idx]
        prob = prob / np.abs(prob).sum()
        sampled_locs = np.random.choice(a = subsampled_idx, size = batch_size, p=prob)
        Xvs_idx = self.pmi_locs[0][sampled_locs]
        Xws_idx = self.pmi_locs[1][sampled_locs]

        word_pairs = np.empty(shape = (self.batch_size, 2, self.embedding_dim))
        word_pairs[:,0,:] = self.word_embeddings[Xvs_idx,:]
        word_pairs[:,1,:] = self.word_embeddings[Xws_idx,:]
        return word_pairs, np.array([Xvs_idx, Xws_idx])


# class UnigramPairLoader:
#     def __init__(self, batch_size, embedding_dim = 100, nemb=False, **kwargs):
#         self.embedding_dim = embedding_dim
#         self.batch_size = batch_size
#         self.word_embeddings = np.load('../data/obj/word_embeddings.npy')
#         idxFreq = np.load("../data/obj/idxFreq.npy")
#         assert idxFreq.shape[0] == self.word_embeddings.shape[0]
#         idxFreq[[2867,2881]] = 0
#         self.word_prob = idxFreq/idxFreq.sum()
#         self.num_vocabs = self.word_prob.shape[0]
#         del idxFreq
    
#     def load_train_batch(self):
#         Xvs_idx = np.random.choice(self.num_vocabs, size = self.batch_size, replcae=False, p=self.word_prob)
#         Xws_idx = np.random.choice(self.num_vocabs, size = self.batch_size, replace=False, p=self.word_prob)

#         word_pairs = np.empty(shape = (self.batch_size, 2, self.embedding_dim))
#         word_pairs[:,0,:] = self.word_embeddings[Xvs_idx,:]
#         word_pairs[:,1,:] = self.word_embeddings[Xws_idx,:]
#         return word_pairs, np.array([Xvs_idx, Xws_idx]) # (2, batch_size)


class BigramPairLoader:
    pass 

class ConditionPmiPairLoader:
    
    def __init__(self, batch_size, embedding_dim = 100, nemb=False, **kwargs):
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        PMI_THRESHOLD = 0.2
        self.conditionalPmi = np.load('../data/obj/conExpPmi.npy') # entry[i,j] is pmi[i,j]/\sum_j pmi[i,j]
        self.word_embeddings = np.load('../data/obj/word_embeddings.npy')
        idxFreq = np.load("../data/obj/idxFreq.npy")
        assert idxFreq.shape[0] == self.word_embeddings.shape[0]
        idxFreq[[2867,2881]] = 0
        self.word_prob = idxFreq/idxFreq.sum()
        self.num_vocabs = self.word_prob.shape[0]
        del idxFreq
    
    def load_train_batch(self):
        Xvs_idx = np.random.choice(self.num_vocabs, size = self.batch_size, p=self.word_prob)
        conditionalProb = self.conditionalPmi[Xvs_idx,:]
        Xws_idx = select_by_conditional_prob(conditionalProb)

        word_pairs = np.empty(shape = (self.batch_size, 2, self.embedding_dim))
        word_pairs[:,0,:] = self.word_embeddings[Xvs_idx,:]
        word_pairs[:,1,:] = self.word_embeddings[Xws_idx,:]
        return word_pairs, np.array([Xvs_idx, Xws_idx]) # (2, batch_size)


def select_by_conditional_prob(conditionalProb):
    nsample, nCorpus  = conditionalProb.shape
    Xws_idx = np.zeros(shape = (nsample,), dtype=int)
    for i in range(nsample):
        Xws_idx[i] = np.random.choice(nCorpus, size=1, p = conditionalProb[i])
    return Xws_idx

# import pickle
# with open('../data/obj/4vocabFreq.pkl', 'rb') as f:
#     vocabFreq = pickle.load(f)

# with open('../data/obj/idxvocab.pkl', 'rb') as f:
#     idxvocab = pickle.load(f)


class batch_loader:
    def __init__(self, test=False, num_vocabs=5044, batch_size = 0, nemb = False, **kwargs):
        self.test=test
        self.batch_size=batch_size
        self.num_vocabs=num_vocabs
        self.test_size = num_vocabs
        if not nemb:
            self.word_embeddings = np.load('../data/obj/word_embeddings.npy')
        else:
            self.word_embeddings = np.load('../data/obj/normalizedwe.npy')
        # self.freq = vocabFreq.values()
        # self.freq[2867] = 0
        # self.freq[2881] = 0
        self.cnt=0

    def load_train_batch(self):
        idx = np.random.randint(0, self.num_vocabs, size = self.batch_size)
        word_batch = self.word_embeddings[idx,:]
        return word_batch, idx
    
    def load_test_batch(self):
        idx = self.sequential_word_idx()
        word_batch = self.word_embeddings[idx,:]
        return word_batch, idx

    def load_word_batch(self):
        if not self.test:
            idx = np.random.randint(0, self.num_vocabs, size = self.batch_size)
        else:
            idx = self.sequential_word_idx()
        
        word_batch = self.word_embeddings[idx,:]
        return word_batch, idx
    
    def sequential_word_idx(self):
            if self.cnt > self.num_vocabs - self.batch_size:
                self.cnt=self.num_vocabs - self.batch_size
            idx = np.arange(self.cnt, self.cnt + self.batch_size)
            self.cnt += self.batch_size
            return idx


@LoaderFactory.register("unigramLoader")
class UnigramLoader:
    def __init__(self, batch_size, emb_dim = 100):
        self.test = False
        self.batch_size = batch_size
        self.cnt = 0
        self.emb_dim = emb_dim
        self.word_embeddings = np.load('../data/googleNgram/embed{}.npy'.format(emb_dim))
        self.word_freq = np.load("../data/googleNgram/1gramSortedFreq.npy")
        assert self.word_freq.shape[0] == self.word_embeddings.shape[0]
        self.num_train_vocabs = self.word_freq.shape[0]
        self.num_test_vocabs = 20000
        self.SUBSAMPLE_SIZE = 4096
    
    def __str__(self):
        return "Google 1 Gram freq; Glove Embedding with dim={}".format(self.emb_dim)
    
    def sample_word_idx(self, batch_size):
        subsampled_idx = np.random.randint(0, self.num_train_vocabs, self.SUBSAMPLE_SIZE)
        prob = self.word_freq[subsampled_idx]
        prob = prob / np.abs(prob).sum()
        sampled_locs = np.random.choice(a = subsampled_idx, size = batch_size, replace = False, p=prob)
        return sampled_locs

    def load_train_batch(self):
        sampled_idx = self.sample_word_idx(self.batch_size)
        word_batch = self.word_embeddings[sampled_idx,:]
        return word_batch, sampled_idx
    
    def load_test_batch(self):
        if self.cnt > self.num_test_vocabs - self.batch_size:
            self.cnt=self.num_test_vocabs - self.batch_size
        idx = np.arange(self.cnt, self.cnt + self.batch_size)
        self.cnt += self.batch_size
        word_batch = self.word_embeddings[idx,:]
        return word_batch, idx
    
    def load_by_idx(self, idx):
        word_batch = self.word_embeddings[idx,:]
        return word_batch, idx

@LoaderFactory.register("unigram97Loader")
class Unigram97Loader(UnigramLoader):
    def __init__(self, batch_size, emb_dim = 97):
        assert emb_dim == 97
        UnigramLoader.__init__(self, batch_size, emb_dim = 100)
        self.word_embeddings = np.delete(self.word_embeddings, [55, 58, 84], axis = 1)


class UniformLoader:
    def __init__(self, batch_size, emb_dim = 100):
        self.test = False
        self.batch_size = batch_size
        self.cnt = 0
        self.emb_dim = emb_dim
        self.word_embeddings = np.load('../data/googleNgram/embed{}.npy'.format(emb_dim))
        # self.word_freq = np.load("../data/googleNgram/1gramSortedFreq.npy")
        # assert self.word_freq.shape[0] == self.word_embeddings.shape[0]
        self.num_vocabs = self.word_embeddings.shape[0]
        self.num_test_vocabs = self.num_vocabs
        self.SUBSAMPLE_SIZE = 4096
    
    def __str__(self):
        return "Uniform; Glove Embedding with dim={}".format(self.emb_dim)

    def load_train_batch(self):
        sampled_idx = np.random.randint(0, self.num_vocabs, size = self.batch_size)
        word_batch = self.word_embeddings[sampled_idx,:]
        return word_batch, sampled_idx
    
    def load_test_batch(self):
        if self.cnt > self.num_vocabs - self.batch_size:
            self.cnt=self.num_vocabs - self.batch_size
        idx = np.arange(self.cnt, self.cnt + self.batch_size)
        self.cnt += self.batch_size
        word_batch = self.word_embeddings[idx,:]
        return word_batch, idx
@LoaderFactory.register("imageLoader")
class ImageLoader:
    def __init__(self, batch_size, emb_dim = 100):
        self.test = False
        self.batch_size = batch_size
        self.cnt = 0
        self.emb_dim = emb_dim
        sz = int(np.sqrt(emb_dim))
        assert sz**2 == emb_dim
        self.num_vocabs = 10000
        self.patches = load_patches(num_patches=self.num_vocabs ,sz= sz, normalize = True) 
    
    def load_train_batch(self):
        sampled_idx = np.random.randint(0, self.num_vocabs, size = self.batch_size)
        word_batch = self.patches[sampled_idx,:]
        return word_batch, sampled_idx

    def load_test_batch(self):
        if self.cnt > self.num_vocabs - self.batch_size:
            self.cnt=self.num_vocabs - self.batch_size
        idx = np.arange(self.cnt, self.cnt + self.batch_size)
        self.cnt += self.batch_size
        word_batch = self.patches[idx,:]
        return word_batch, idx



def load_patches(num_patches=10000, sz=16, normalize=False):
    #curl -O http://www.rctn.org/bruno/sparsenet/IMAGES.mat
    import scipy.io as scipio
    mat_images = scipio.loadmat('../data/IMAGES.mat')
    imgs = mat_images['IMAGES']
    H, W, num_images = imgs.shape
    patches = []
    # Get the coordinates of the upper left corner of randomly cropped image
    beginx = np.random.randint(0, W-sz, num_patches)
    beginy = np.random.randint(0, H-sz, num_patches)
    for i in range(num_patches):
        idx = np.random.randint(0, num_images)
        beginx = np.random.randint(0, W-sz)
        beginy = np.random.randint(0, H-sz)
        img = imgs[:, :, idx]
        crop = img[beginy:beginy+sz, beginx:beginx+sz].flatten()
        if normalize:
            crop = (crop - np.mean(crop))/np.std(crop)
        else:
            crop = crop - np.mean(crop)
        patches.append(crop)

    patches = np.array(patches)
    patches = patches.reshape(num_patches,sz*sz)
    # patches = patches.reshape(num_patches,sz,sz,1)
    patches = patches.astype('float32')
    return patches

@LoaderFactory.register("fairseqLoader")
class FairseqLoader:
    def __init__(self, batch_size, emb_dim = 100, cap = 3000):
        self.test = False
        self.batch_size = batch_size
        self.cnt = 0
        self.emb_dim = emb_dim
        self.word_embeddings = np.load('../data/wiki103/emb{}.npy'.format(emb_dim))
        self.word_freq = np.load("../data/wiki103/freqs.npy")
        assert self.word_freq.shape[0] == self.word_embeddings.shape[0]
        self.num_vocabs = self.word_freq.shape[0]
        self.num_train_vocabs = 128083
        self.num_test_vocabs = 20000
        self.word_freq = self.word_freq[:self.num_train_vocabs]
        self.word_freq[self.word_freq > cap] = cap
        self.SUBSAMPLE_SIZE = 4096
    
    def __str__(self):
        return "Google 1 Gram freq; Glove Embedding with dim={}".format(self.emb_dim)
    
    def sample_word_idx(self, batch_size):
        # NOTE we for Fairseq dataset we only sample the first 128083 words
        subsampled_idx = np.random.randint(0, self.num_train_vocabs, self.SUBSAMPLE_SIZE)
        prob = self.word_freq[subsampled_idx]
        prob = prob / np.abs(prob).sum()
        sampled_locs = np.random.choice(a = subsampled_idx, size = batch_size, replace = False, p=prob)
        return sampled_locs

    def load_train_batch(self):
        sampled_idx = self.sample_word_idx(self.batch_size)
        word_batch = self.word_embeddings[sampled_idx,:]
        return word_batch, sampled_idx
    
    def load_test_batch(self):
        if self.cnt > self.num_test_vocabs - self.batch_size:
            self.cnt=self.num_test_vocabs - self.batch_size
        idx = np.arange(self.cnt, self.cnt + self.batch_size)
        self.cnt += self.batch_size
        word_batch = self.word_embeddings[idx,:]
        return word_batch, idx

if __name__ == "__main__":
    loader = FairseqLoader(batch_size=100, emb_dim = 300)

    ws, idx = loader.load_train_batch()



