import numpy as np
from scipy import sparse
import argparse
import pickle

def getAttributes(clazz):
    return {name: attr for name, attr in clazz.__dict__.items()
            if not name.startswith("__") 
            and not callable(attr)
            and not type(attr) is staticmethod
            and not type(attr) is classmethod}

class Config:
    
    lif_name = "kgliflight"
    dictlearner_name = "globalposnormreg"

    batch_size = 128
    num_units = 225
    emb_dim = 256
    num_euler_steps = 50
    lr_Phi = 0.1
    lr_VT = 0.001

    magnifier_base = 1000
    target_rate = 0.1
    target_l1 = 0.15
    VT = -3.763e-02

    DIRPREFIX = "l1rate{:.2f}".format(target_l1)

    WE = 5
    WI = 1
    RE = 3
    RI = 5

    @classmethod
    def get_dirname(cls):
        return "{}nunits{}we{:.1f}wi{:.1f}re{}ri{}".format(cls.DIRPREFIX, cls.num_units, cls.WE, cls.WI, cls.RE, cls.RI)
    
    @classmethod
    def dump(cls):
        dirname = cls.get_dirname()
        attributes = getAttributes(cls)
        fpath = "../result/{}/".format(dirname)
        mkdir(fpath)
        with open(fpath + "config", 'wb') as f:
            pickle.dump(attributes, f)
    
    @classmethod
    def load(cls, fpath):
        with open(fpath + "config", 'rb') as f:
            this_dict = pickle.load(f)  
        
        for k,v in this_dict.items():
            setattr(cls, k, v)


def write_line_to_file(fname, *content):

    with open(fname, 'a') as the_file:
        the_file.write('{}\n'.format(" ".join(map(str, content))))

class Log:

    def __init__(self, fname):
        self.fname = fname
        self.local = False
    
    def write(self, *content):
        write_line_to_file(self.fname, *content)
        if self.local == True:
            print(*content)


import os
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path) 
    return path  

def get_K_matrix(n, re, ri, wi=3, we=3, sigmaE = 3, inhibitory_ratio=0.2):
    assert re < ri
    resq = re**2 
    risq = ri**2
    
    # Set inhibitory neurons
    if n == 1600:
        inhibitory = []
        for i in range(20):
            for j in range(10):
                inhibitory.append(80*i + 4*j + 2*(i%2))
        inhibitory = np.array(inhibitory)
    else: 
        inhibitory = []
        sidelen = int(np.sqrt(n))
        for i in range(sidelen):
            for j in range(sidelen):
                if (i%2 == 0) and (j%2 == 0) and ((i+j)%4 == 0):
                    inhibitory.append(i*sidelen + j)

        inhibitory = np.array(inhibitory)

    side_length = int(np.sqrt(n))
    
    def find_distsq(i,j):
        xi, yi = i//side_length, i%side_length
        xj, yj = j//side_length, j%side_length
        return (xi-xj)**2+(yi-yj)**2
    
    # construct the W matrix
    W = np.zeros(shape = (n,n))
    for i in range(n):
        for j in range(n):
            # i row, j column
            distsq = find_distsq(i,j)
            if (distsq > resq):
                continue
            if j in inhibitory:
                W[i,j] = wi *  (distsq < risq)
            else:
                W[i,j] = we * np.exp(- distsq/2/sigmaE) * (distsq < resq)
    
    np.fill_diagonal(W, 0)
    
    return inhibitory, W


def get_sK_matrix(n, re, ri, wi=3, we=2, sigmaE = 2, inhibitory_ratio=0.2):
    inhibitory, K = get_K_matrix(n,re,ri,wi,we,sigmaE,inhibitory_ratio)
    sK = sparse.csr_matrix(K)
    return inhibitory, sK


def parse_argv(config: Config):
    parser = argparse.ArgumentParser(prog = 'nwave')

    structp = parser.add_argument_group("structureParameters")
    idp = parser.add_argument_group("ids")
    lifp = parser.add_argument_group("lif")
    dlp = parser.add_argument_group("dl")

    structp.add_argument("--num_units", type = int, nargs='?', default = 225)

    structp.add_argument("--we", type = float, nargs='?', default = 5)
    structp.add_argument("--wi", type = float, nargs='?', default = 1)
    structp.add_argument("--re", type = int, nargs='?', default = 3)
    structp.add_argument("--ri", type = int, nargs='?', default = 5)

    structp.add_argument("--target_l1", type = float, nargs='?', default = 0.15)

    idp.add_argument("--sid", type = int, nargs='?', default = 0)
    idp.add_argument("--dirprefix", type = str, nargs='?', default = "")

    lifp.add_argument("--lif_name", type = str, nargs = "?", default = "kgliflight")
    lifp.add_argument("--vt", type=float, nargs="?", default=-0)
    lifp.add_argument("--lr_vt", type=float, nargs="?", default=0.001)

    dlp.add_argument("--dl_name", type = str, nargs = "?", default = "globalposnormregl1")


    args = parser.parse_args()

    config.num_units = args.num_units

    config.RE = args.re
    config.RI = args.ri
    config.WE = args.we
    config.WI = args.wi

    config.target_l1 = args.target_l1

    config.SID = args.sid
    config.DIRPREFIX = args.dirprefix

    config.VT = args.vt
    config.lr_VT = args.lr_vt
    config.lif_name = args.lif_name

    config.dictlearner_name = args.dl_name

    return config

def print_all_combinations(ls, prefix):
    import re
    import itertools
    
    res = list(itertools.product(*ls))

    # filter
    res = [item for item in res if (item[0] > item[1] and item[2] < item [3])]
    
    for r in res:
        print(prefix + re.sub("[),(]", "", str(r)))

def calc_norms(matrix):
    return np.sqrt(np.square(matrix).sum(axis=0))