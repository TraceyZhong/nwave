'''
==========
This module provides algorithms for training 2-layer neural networks with activations
given by wave dynamics.
==========
'''

import numpy as np
from tqdm import trange
from utils import *
from scipy import signal
from ista_sparse_coding import cwaveSC


class wavenetSC:
    '''
    sparse coding model with activations using waves
    '''
    # def __init__(self, **kwargs): #TODO
    #     assert "leaky" in kwargs
    #     assert "W" in kwargs # W is a sparse matrix
    #     super().__init__(**kwargs)
    #     self.k_exc= np.expand_dims(kwargs["W"][0], axis=0)
    #     self.k_inh = np.expand_dims(kwargs["W"][1], axis=0)
    #     self.leaky = kwargs["leaky"]
    #     self.side_len = int(np.sqrt(self.num_units))

    def soft_threshold(self, x):
        return np.maximum(x - self.lmda, 0) - np.maximum(-x - self.lmda, 0)

    def relu(self, x):
        return x*(x>0)

    def sigmoid(self, x):
        return 1./(1+np.exp(-x))

    def __init__(self, num_inputs, num_outputs, num_units, batch_size=1,
        eps = 5e-3, lmda=1e-2, lr_r=0.01, lr_W1=0.02, r1=3, r2=5, wi=1, we=5,
            sigmaE=3, leaky=0, target_sparsity=.1, **kwargs):
        self.D = num_inputs
        self.H1 = num_units
        self.K = num_outputs
        self.W1 = 0.3 * np.random.randn(self.D, self.H1)
        self.b1 = np.zeros((1,self.H1))
        self.W2 = 0.1 * np.random.randn(self.H1, self.K)
        self.b2 = np.zeros((1, self.K))
        self.side_len = int(np.sqrt(num_units))
        self.batch_size = batch_size
        self.lmda = lmda
        self.use_waves = True
        self.lr_W1 = lr_W1
        if leaky==0:
            self.leaky = wi+we
        self.target_sparsity = target_sparsity
        print('target sparsity: %.2f%%' % (100*target_sparsity))
        self.fpath = kwargs.get("fpath", "./")


        Wg= get_kernels(r1, r2, wi, we, sigmaE)
        W = np.expand_dims(Wg[0], axis=0)
        W = W[np.newaxis, :, :]
        W = W/np.max(W)
        self.k_grad = W/np.max(W)

        W = get_kernels(r1, r2, wi, we, sigmaE)
        self.cwsc = cwaveSC(
            num_inputs=3,
            num_units = self.side_len**2,
            batch_size = batch_size,
            lr_r = lr_r,
            lr_Phi = lr_W1,
            lmda = self.lmda,
            W = W,
            leaky=self.leaky
        )

    def dump(self, fpath, t):
        with open(fpath + 'W1_t%i.npy' % t, 'wb') as f:
            np.save(f, self.W1) # shape of W1 = (embedding X num_units)

    def load(self, fpath, t):
        self.W1 = np.load(fpath + 'W1_t%i.npy' % t)

    def reset_activation(self, batch_size = None):
        if batch_size is None:
            self.activation = np.zeros(self.activation.shape)
        else:
            print(self.activation.shape)
            self.activation = np.zeros((batch_size, *self.activation.shape[1:]))

    def compute_activations(self, X):
        stimulus = (X @ self.W1).reshape((X.shape[0], self.side_len, self.side_len))
        # stimulus = self.sigmoid(stimulus)

        self.cwsc.activation = np.zeros([X.shape[0], self.side_len, self.side_len, 2])
        for step in np.arange(1,50):
            self.cwsc.fit_activation(stimulus)
            dr = self.cwsc.activation_tm1 - self.cwsc.activation
            relative_error = np.sqrt(np.square(dr).sum()) / \
                (self.cwsc.eps + np.sqrt(np.square(self.cwsc.activation_tm1).sum()))
            if relative_error < self.cwsc.eps:
                break
            a = self.cwsc.activation[:,:,:,0]

        activations = self.cwsc.activation[:,:,:,0].reshape((X.shape[0], self.side_len * self.side_len))
        return activations, step

    def postprocess_activations(self, words, activations, gamma):
        #activations = np.maximum(activations, 0)
        #activations = np.round(100 * activations) / 100
        #words_hat = activations @ self.W1.T
        #gamma = np.diag(words_hat @ words.T) / (np.diag(words_hat @ words_hat.T) + 1e-08)
        #gamma = np.sum(np.diag(words_hat @ words.T)) / np.sum(np.diag(words_hat @ words_hat.T))
        #gamma = 0.1
        scalings = gamma * np.ones(activations.shape[0])
        return np.diag(scalings) @ activations
        #return activations

    def dictionary_gradient_step(self, words, activations):
        error = words - activations @ self.W1.T
        dW1 = error.T @ activations
        self.W1 += self.lr_W1 * (dW1 - np.vstack(np.mean(dW1, axis=1)))
        self.normalize_cols()

    def dictionary_smoothed_gradient_step(self, words, activations):
        a = activations.reshape((words.shape[0], self.side_len, self.side_len))[:, np.newaxis, :, :]
        Phi = self.W1.reshape((words.shape[1], self.side_len, self.side_len))[np.newaxis, :, :, :]
        aPhi = a*Phi # (bs, d, l, l)
        fit = signal.convolve(aPhi, self.k_grad, mode = "same") # (bs, d, l, l)
        error = words[:, :, np.newaxis, np.newaxis] - fit # (bs, d, l, l)
        dW1 = (a * error).sum(axis =0)
        self.W1 += self.lr_W1 * dW1.reshape((words.shape[1], self.side_len*self.side_len))
        self.normalize_cols()

    def dictionary_local_gradient_step(self, words, activations):
        D = np.square(activations).sum(axis = 0)
        dW1 = words.T @ activations - self.W1 * D
        self.W1 += self.lr_W1 * (dW1 - np.vstack(np.mean(dW1, axis=1)))
        self.normalize_cols()

    def normalize_cols(self):
        self.W1 = self.W1 / np.maximum(np.sqrt(np.square(self.W1).sum(axis=0)), 1e-8)

    def report_errors(self, words, activations):
        l2_error = np.square(words - activations @ self.W1.T).sum() / activations.shape[0]
        l1_norm = np.sum(np.abs(activations)) / activations.shape[0]
        l0_norm = np.sum(np.abs(activations) > 1e-4) / np.prod(activations.shape)
        return l2_error, l1_norm, l0_norm

    def train(self, X, gradient_steps=20000, initial_step=0, gamma=[]):
        l2_loss = []
        l1_loss = []
        l0_loss = []
        steps = []
        num_examples = X.shape[0]
        tbar = trange(initial_step, initial_step+gradient_steps, desc='Training', leave=False, miniters=int(100))

        for i in tbar:
            inds = np.random.randint(0, num_examples, self.batch_size)
            this_X = X[inds, :]

            # evaluate class scores, [N x K]
            activations, step = self.compute_activations(this_X)
            if len(gamma) > 0:
                if i % 500 == 0:
                    print('scaling by %f' % gamma[i])
                activations = self.postprocess_activations(this_X, activations, gamma[i])
            self.dictionary_gradient_step(this_X, activations)
            l2l, l1l, l0l = self.report_errors(this_X, activations)

            dlambda = l0l - self.target_sparsity
            self.cwsc.lmda = self.cwsc.lmda + .01 * dlambda
            #print("lambda=%f, dlambda=%f" % (self.lmda, dlambda))

            steps.append(step)
            l2_loss.append(l2l)
            l1_loss.append(l1l)
            l0_loss.append(l0l)

            if i % 100 == 0:
                tbar.set_description("loss=%.3f sparsity=%2.2f%% lmda=%.3f steps=%d" % \
                    (l2l, 100*l0l, self.cwsc.lmda, step))
                tbar.refresh()
                self.dump(self.fpath, i)

            #if i % 100 == 0:
            #    np.save('W1-%d' % i, self.W1)
            #    np.save('W2-%d' % i, self.W2)
            #    np.save('b2-%d' % i, self.b2)

        return l2_loss, l1_loss, l0_loss, steps

    def train_through_loader(self, loader, gradient_steps=10000, initial_step=0, gamma=[]):
        l2_loss = []
        l1_loss = []
        l0_loss = []
        steps = []
        tbar = trange(initial_step, initial_step+gradient_steps, desc='Training', leave=True, miniters=100)

        for i in tbar:
            this_X, _ = loader.load_train_batch()

            # evaluate class scores, [N x K]
            activations, step = self.compute_activations(this_X)
            if len(gamma) > 0:
                if i % 500 == 0:
                    print('scaling by %f' % gamma[i])
                activations = self.postprocess_activations(this_X, activations, gamma[i])
            self.dictionary_gradient_step(this_X, activations)
            l2l, l1l, l0l = self.report_errors(this_X, activations)

            dlambda = l0l - self.target_sparsity
            self.cwsc.lmda = self.cwsc.lmda + .01 * dlambda
            #print("lambda=%f, dlambda=%f" % (self.lmda, dlambda))

            steps.append(step)
            l2_loss.append(l2l)
            l1_loss.append(l1l)
            l0_loss.append(l0l)

            if i % 100 == 0:
                tbar.set_description("loss=%.3f sparsity=%2.2f%% lmda=%.3f steps=%d" % \
                    (l2l, 100*l0l, self.cwsc.lmda, step))
                tbar.refresh()
                self.dump(self.fpath, i)

            #if i % 100 == 0:
            #    np.save('W1-%d' % i, self.W1)
            #    np.save('W2-%d' % i, self.W2)
            #    np.save('b2-%d' % i, self.b2)

        return l2_loss, l1_loss, l0_loss, steps

    def train_local(self, X, gradient_steps=20000, initial_step=0):
        l2_loss = []
        l1_loss = []
        l0_loss = []
        steps = []
        num_examples = X.shape[0]
        tbar = trange(initial_step, initial_step+gradient_steps, desc='Training', leave=True)

        for i in tbar:
            inds = np.random.randint(0, num_examples, self.batch_size)
            this_X = X[inds, :]

            # evaluate class scores, [N x K]
            activations, step = self.compute_activations(this_X)
            self.dictionary_local_gradient_step(this_X, activations)
            l2l, l1l, l0l = self.report_errors(this_X, activations)

            steps.append(step)
            l2_loss.append(l2l)
            l1_loss.append(l1l)
            l0_loss.append(l0l)

            if i % 20 == 0:
                tbar.set_description("loss=%.3f sparsity=%2.2f%% lmda=%.3f steps=%d" % \
                    (l2l, 100*l0l, self.cwsc.lmda, step))
                tbar.refresh()

            #if i % 100 == 0:
            #    np.save('W1-%d' % i, self.W1)
            #    np.save('W2-%d' % i, self.W2)
            #    np.save('b2-%d' % i, self.b2)

        return l2_loss, l1_loss, l0_loss, steps


