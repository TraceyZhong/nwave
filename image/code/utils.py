import os
import argparse
import time

import numpy as np
from scipy import sparse

def create_dir(srepr, algo):
    fpath = "../result/{}/{}/".format(algo, srepr)
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    print(fpath)
    return fpath

def get_Laplacian(n):
    pass

def get_W_matrix(n, r1, r2, wi=3, we=3, sigmaE = 3, inhibitory_ratio=0.2):
    assert r1 < r2
    r1sq = r1**2
    r2sq = r2**2
    # Random sample inhibitory neurons
    # inhibitory = np.random.choice(n, int(n*inhibitory_ratio))

    if n == 1600:
        inhibitory = []
        for i in range(20):
            for j in range(10):
                inhibitory.append(80*i + 4*j + 2*(i%2))
        inhibitory = np.array(inhibitory)
    else:
        inhibitory = np.random.choice(n, int(n*inhibitory_ratio))


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
            W[i,j] = - we * np.exp(- distsq/2/sigmaE) * (distsq < r1sq)
        for j in inhibitory:
            distsq = find_distsq(i,j)
            W[i,j] = wi *  (distsq < r2sq)
        # W[i,i] = np.abs((W[i,:])).sum() # + 2

    np.fill_diagonal(W, 0)

    mins = np.real(np.linalg.eigvals(W)).min()

    np.fill_diagonal(W, -mins + 1)

    return inhibitory, W


def get_sW_matrix(n, r1, r2, wi=3, we=2, sigmaE = 2, inhibitory_ratio=0.2):
    inhibitory, W = get_W_matrix(n,r1,r2,wi,we,sigmaE,inhibitory_ratio)
    sW = sparse.csr_matrix(W)
    return inhibitory, sW

def get_kernels(r1, r2, wi=3, we=2, sigmaE = 2, **kwargs):
    k_exc = np.zeros([2*r1+1, 2*r1+1])
    k_inh = np.zeros([2*r2+1, 2*r2+1])
    for i in range(2*r1+1):
        for j in range(2*r1+1):
            # i row, j column
            distsq = (i-r1)**2+(j-r1)**2
            k_exc[i,j] = np.exp(- distsq/2/sigmaE) * (distsq <= r1**2)
    k_exc = we * k_exc / np.sum(k_exc)
    for i in range(2*r2+1):
        for j in range(2*r2+1):
            # i row, j column
            distsq = (i-r2)**2+(j-r2)**2
            k_inh[i,j] = (distsq <= r2**2)
    k_inh = wi * k_inh / np.sum(k_inh)
    return k_exc, k_inh



def get_group_matrix(n, r1, r2, weight = "uniform"):
    '''
    n: number of units
    r1: side overlap among blocks
    r2: side length of each block
    return: num_groups * number of units
    '''
    assert r1 < r2
    l = int(np.sqrt(n))
    assert (l - r2) // (r2 - r1)
    num_group_by_side = (l - r2) // (r2 - r1) + 1
    group_mat = np.zeros(shape = (n, num_group_by_side*num_group_by_side))
    if weight == "uniform":
        for i in range(num_group_by_side):
            for j in range(num_group_by_side):
                group_idx = i*num_group_by_side + j
                block_i = (r2 - r1)*i  # [block_i, block_i + r2)
                block_j = (r2 - r1)*j  # [block_j, block_j + r2]
                # set entry value
                for ui in range(block_i, block_i + r2):
                    for uj in range(block_j, block_j + r2):
                        uij = ui * l + uj
                        group_mat[uij, group_idx] = 1
    if weight == "l2":
        for i in range(num_group_by_side):
            for j in range(num_group_by_side):
                group_idx = i*num_group_by_side + j
                block_i = (r2 - r1)*i  # [block_i, block_i + r2)
                block_j = (r2 - r1)*j  # [block_j, block_j + r2]
                center_i = block_i + int(r2/2)
                center_j = block_j + int(r2/2)
                for ui in range(block_i, block_i + r2):
                    for uj in range(block_j, block_j + r2):
                        uij = ui * l + uj
                        group_mat[uij, group_idx] = np.exp(- (ui - center_i)**2 - (uj - center_j)**2)

    # normalize over columns
    gm_norm = group_mat.sum(axis = 0)
    group_mat = group_mat / gm_norm
    return group_mat

def track_memory():
    import psutil
    import os
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss >> 20
    print('Memory used: {:.2f} MB'.format(memory))

def print_all_combinations(ls, prefix):
    import re
    import itertools

    res = list(itertools.product(*ls))

    for r in res:
        print(prefix + re.sub("[),(]", "", str(r)))




def construct_struct_matrix(n, r1, r2, l1 = False, l2 = False, donut = False, \
    offset = 0, normalize = True):
    l = int(np.sqrt(n))

    if (l1 and l2):
        RuntimeError("Either use l1 or l2. Can not use both")

    if l1:
        print("Using l1 distance for S mat")
        def find_dist(i,j):
            xi, yi = i//l, i%l
            xj, yj = j//l, j%l
            return np.abs(xi - xj) + np.abs(yi-yj)

    elif l2:
        print("Using l2 distance for S mat")
        def find_dist(i,j):
            xi, yi = i//l, i%l
            xj, yj = j//l, j%l
            return np.sqrt((xi-xj)**2+(yi-yj)**2)

    else:
        RuntimeError("Please specify how to construct the S mat")

    S = np.zeros((n, n))
    if not donut:
        print("Incremental inhibition")
        for i in range(n):
            for j in range(i):
                d = find_dist(i,j)
                if d>r1 and d<r2:
                    S[i,j]=np.sqrt(d)
                    S[j,i]=np.sqrt(d)

    else:
        print("Donut with homogeneous inhibition")
        for i in range(n):
            for j in range(i):
                d = find_dist(i,j)
                if d>r1 and d<r2:
                    S[i,j]=1
                    S[j,i]=1

    S -= offset
    np.fill_diagonal(S, 0)

    if normalize:
        print("normalize S")
        S = S / S.sum(axis = 0)

    return S

def span_grid(xx, yy):
    xv, yv = np.meshgrid(xx, yy, sparse = False)
    return np.dstack([xv, yv]).reshape((len(xx)*len(yy),2))

def write_line_to_file(file, *words):
    line = " ".join(map(str, words)) + "\n"
    with open(file, 'a') as f:
        f.write(line)

def parse_argv():
    parser = argparse.ArgumentParser(prog='SC')

    config = parser.add_argument_group('config', \
        description = "algorithm, loader and srepr")
    hyperp = parser.add_argument_group('hyperparameters')
    structp = parser.add_argument_group("structureParameters")
    exp = parser.add_argument_group("experiment")

    exp.add_argument("--train_steps", nargs= '?', type=int, const=1000, default=1000)

    config.add_argument("--algo", type = str)
    config.add_argument("--loader", type = str)
    config.add_argument("--emb_dim", type=int)
    config.add_argument("--nunit", type=int)
    config.add_argument("--srepr", type = str)

    hyperp.add_argument("--lmda", type=int)
    hyperp.add_argument("--lr_r", type=int)
    hyperp.add_argument("--lr_Phi", type=int)
    hyperp.add_argument("--lr_G", type=int)

    structp.add_argument("--groupAlgo", action="store_true")
    structp.add_argument("--r1", nargs='?', const=3, type=int, default=3)
    structp.add_argument("--r2", nargs='?', const=6, type=int, default=6)
    structp.add_argument("--donut", nargs='?', const=0, type=int, default=0)
    structp.add_argument("--ns", nargs='?', const=0, type=int, default=0)

    structp.add_argument("--wi", nargs='?', const=1, type=float, default=1)
    structp.add_argument("--we", nargs='?', const=1, type=float, default=1)
    structp.add_argument("--sigmaE", nargs='?', const=1, type=float, default=1)
    structp.add_argument("--leaky", nargs='?', const=30, type=float, default=30)


    return parser.parse_args()

def get_structPars(args):
    structp = {
        "r1": args.r1,
        "r2": args.r2,
        "wi": args.wi,
        "we": args.we,
        "sigmaE": args.sigmaE,
        "leaky": args.leaky
    }
    return structp



if __name__=="__main__":
    we = (2,3)
    sigmae = (1,2,3)
    leaky = (30, 40)
    lmba = (8, 10, 12)
    prefix = "sbatch run_wave.sh "
    # note the sequence is
    # lmba, we, sigmae, leaky
    ls = [lmba, we, sigmae, leaky]
    print_all_combinations(ls, prefix)
