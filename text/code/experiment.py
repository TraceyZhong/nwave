import argparse

import numpy as np

from utils import *
from pallet import LoaderFactory

from wavenet import wavenetSC

from visualization import vis_error

def parse_argv():
    parser = argparse.ArgumentParser(prog='SC')
    parser.add_argument("--loader", type = str)
    parser.add_argument("--sparsity", type=float)
    parser.add_argument("--local", action="store_true")

    parser.add_argument("--we", type = int)
    parser.add_argument("--wi", type = int)
    
    return parser.parse_args()

args = parse_argv()

## --- Experiment set up --- ##

def get_fpath_from_args(args):
    return "../result/{}tr{}we{}wi{}lrW0.1lrA0.1/".format(args.loader, args.sparsity, \
        args.we, args.wi)

fpath = get_fpath_from_args(args)
mymkdir(fpath)

loader_name = args.loader
target_sparsity = args.sparsity
we = args.we
wi = args.wi 

emb_dim_dict = {
    "unigram97Loader": 97,
    "fairseqLoader": 300
}

batch_size = 128
emb_dim = emb_dim_dict[loader_name]
num_units = 1600
gradient_steps = 50000
if args.local:
    gradient_steps = 100


## --- A lot of set up --- ##
loaders = LoaderFactory.registry
LOADER = loaders[loader_name]
loader = LOADER(batch_size = batch_size, emb_dim=emb_dim)


wnsc = wavenetSC(num_inputs=emb_dim, num_outputs=1, num_units=num_units, batch_size=128, lmda=.050, 
            lr_r=0.01, lr_W1=0.01, r1=3, r2=5, wi=args.wi, we=args.we, sigmaE=3, \
                target_sparsity=target_sparsity, \
                    fpath = fpath)


# Train 
l2_loss, l1_loss, l0_loss, steps = wnsc.train_through_loader(loader, gradient_steps=gradient_steps)
wnsc.dump(fpath, gradient_steps)

errors = np.column_stack((l2_loss, l1_loss, l0_loss, steps))
vis_error(errors, fpath)


# with open(fpath + '%s_W1.npy' % 'fig_text', 'wb') as f:
#     np.save(f, wnsc.W1) # shape of W1 = (embedding X num_units)

## --- Test - collect all activations --- ##
if not args.local:
    loader.cnt = 0
    activity = np.zeros(shape=(loader.num_test_vocabs, num_units))

    for t in range((loader.num_test_vocabs-1)//loader.batch_size + 1):
        word_batch, wp_idx = loader.load_test_batch()
        try:
            activ, step = wnsc.compute_activations(word_batch)
            activity[wp_idx,:] = activ
        except RuntimeError as e:
            print(e) 

    with open(fpath + 'uactivity.npy', 'wb') as f:
        np.save(f, activity)












