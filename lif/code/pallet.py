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

    registry = {}

    @classmethod
    def register(cls, name: str):

        def inner_wrapper(wrapped_class):
            if name in cls.registry:
                print('Loader %s already exists. Will replace it', name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper


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
    
    def load_train_batch(self, batch_size = 0):
        if batch_size == 0:
            batch_size = self.batch_size 
        sampled_idx = np.random.randint(0, self.num_vocabs, size = batch_size)
        word_batch = self.patches[sampled_idx,:]
        return word_batch, sampled_idx


    def load_test_batch(self):
        if self.cnt > self.num_vocabs - self.batch_size:
            self.cnt=self.num_vocabs - self.batch_size
        idx = np.arange(self.cnt, self.cnt + self.batch_size)
        self.cnt += self.batch_size
        word_batch = self.patches[idx,:]
        return word_batch, idx



def load_patches(num_patches=10000, sz=16, normalize=True):
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
