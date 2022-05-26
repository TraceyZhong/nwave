'''
==========
This module provides algorithms for solving sparse coding problems using ista
methods.
==========
'''
import numpy as np
from scipy import signal
from abc import ABC, abstractmethod


class IscFactory:
    """ The factory class for creating executors"""

    registry = {}
    """ Internal registry for available executors """

    sregistry = {}
    """ registry for structure algorithm"""

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
                print('SC algorithm %s already exists. Will replace it' % name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    # End register

    @classmethod
    def sregister(cls, name: str):
        """ Class method to register Executor class to the internal registry.
        Args:
            name (str): The name of the executor.
        Returns:
            The Executor class itself.
        """

        def inner_wrapper(wrapped_class):
            if name in cls.registry:
                print('SC algorithm %s already exists. Will replace it' % name)
            cls.registry[name] = wrapped_class
            cls.sregistry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    # End register

def normalized_gram_matrix(a):
    gram = a @ a.T
    n=gram.shape[0]
    norm_gram = gram
    for i in range(n):
        norm=np.sqrt(gram[i,i])
        if norm>0:
            norm_gram[:,i] /= norm
            norm_gram[i,:] /= norm
        else:
            norm_gram[i,i] = 1
    return norm_gram


def correlation_matrix(a):
    cov=np.cov(a)
    n=cov.shape[0]
    corr = cov
    for i in range(n):
        std=np.sqrt(cov[i,i])
        if std>0:
            corr[:,i] /= std
            corr[i,:] /= std
        else:
            corr[i,i] = 1
    return corr


class _IstaSparseCoding(ABC):

    def __init__(self, num_inputs=0, num_units=0, batch_size=0, \
        max_a_fit = 1000, eps = 5e-3,
        lr_r=1e-2, lr_Phi=1e-2, lmda=1e-3, **kwargs):
        '''
        num_inputs: dimension of the embedding space
        num_units: number of code_words
        lr: learning rate
        lmda: soft thresholding parameter
        train_steps: number of training steps
        max_a_fit: maximum number of fitting a withing each training step
        '''
        # model parameter
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.batch_size = batch_size

        self.max_a_fit = max_a_fit
        self.eps = eps

        # algo hyper parameter
        self.lr_r = lr_r
        self.lr_Phi = lr_Phi
        self.lmda = lmda # soft thresholding parameter

        # Codebook
        Phi = np.random.randn(self.num_inputs, self.num_units)
        self.Phi = Phi * np.sqrt(1/self.num_inputs) # Each column is a codeword

        # Neuron activation
        self.activation = np.zeros((self.batch_size, self.num_units))


    def reset_activation(self, batch_size = None):
        if batch_size is None:
            self.activation = np.zeros(self.activation.shape)
        else:
            print(self.activation.shape)
            self.activation = np.zeros((batch_size, *self.activation.shape[1:]))

    def normalize_cols(self):
        self.Phi = self.Phi / np.maximum(np.sqrt(np.square(\
            self.Phi).sum(axis=0)), 1e-8)

    @abstractmethod
    def soft_threshold(self, x):
        return

    @abstractmethod
    def report_errors(self, words):
        return

    @abstractmethod
    def train_dictionary(self, words):
        return

    def fit_activation(self, words):
        error = words - self.activation @ self.Phi.T
        # error.shape = (bs, 2, num_inputs)
        dactivation = error @ self.Phi
        activation = self.activation + self.lr_r * dactivation
        self.activation = self.soft_threshold(activation)

    def train(self, words, step):

        self.reset_activation()
        self.normalize_cols()
        activation_tm1 = np.random.normal(size=self.activation.shape)

        for t in range(self.max_a_fit):

            self.fit_activation(words)

            dr = activation_tm1 - self.activation
            relative_error = np.sqrt(np.square(dr).sum()) / (self.eps + np.sqrt(np.square(activation_tm1).sum()))
            activation_tm1 = self.activation

            if relative_error < self.eps:
                self.train_dictionary(words)
                e2, e1, e0 = self.report_errors(words)
                return e2, e1, e0, t, self.activation

        raise RuntimeError("Error at patch {}. Relative error doesn't converge: {:.4f}".format(step, relative_error))

    def test(self, words, step):
        self.reset_activation()
        activation_tm1 = np.random.normal(size=self.activation.shape)

        for t in range(self.max_a_fit):
            self.fit_activation(words)

            dr = activation_tm1 - self.activation
            relative_error = np.sqrt(np.square(dr).sum()) / (self.eps + np.sqrt(np.square(activation_tm1).sum()))
            activation_tm1 = self.activation

            if relative_error < self.eps:

                e2, e1, e0 = self.report_errors(words)
                return e2, e1, e0, t, self.activation

        raise RuntimeError("Error at patch {}. Relative error doesn't converge: {:.4f}".format(step, relative_error))
@IscFactory.register("wsc")
class waveSC(_IstaSparseCoding):
    '''
    We specifically need a W matrix and a mask matrix
    the W matrix is not sparse
    '''

    def __init__(self, **kwargs):
        assert "inhibitory" in kwargs
        assert "W" in kwargs # W is a sparse matrix
        _IstaSparseCoding.__init__(self, **kwargs)
        self.Wt = kwargs["W"].T
        self.inhibitory = kwargs["inhibitory"]

    def soft_threshold(self, x):
        return np.maximum(x - self.lmda, 0) - np.maximum(-x - self.lmda, 0)

    def report_errors(self, word_batch):
        l2_error = np.square(word_batch - self.activation @ self.Phi.T).sum() / (self.batch_size)
        l1_norm = np.sum(np.abs(self.activation)) / (self.batch_size)
        l0_norm = np.sum(np.abs(self.activation) > 1e-4) / (self.batch_size * self.num_units)
        return l2_error, l1_norm, l0_norm

    def train_dictionary(self, words):
        error = words - self.activation @ self.Phi.T
        dPhi = error.T @ self.activation
        self.Phi += self.lr_Phi * dPhi
        self.Phi[:,self.inhibitory] = 0

    def fit_activation(self, stimulus):
        self.activation = self.activation + \
            self.lr_r * (stimulus - self.activation @ self.Wt)
        self.activation = self.soft_threshold(self.activation)

    # def get_stimulus(self, words):
    #     pass

    def train(self, words, step):

        self.reset_activation()
        self.normalize_cols()
        activation_tm1 = np.random.normal(size=self.activation.shape)

        # get stimulus
        stimulus = words @ self.Phi

        for t in range(self.max_a_fit):
            self.fit_activation(stimulus)

            dr = activation_tm1 - self.activation
            relative_error = np.sqrt(np.square(dr).sum()) / (self.eps + np.sqrt(np.square(activation_tm1).sum()))
            activation_tm1 = self.activation

            if relative_error < self.eps:
                self.train_dictionary(words)
                e2, e1, e0 = self.report_errors(words)
                return e2, e1, e0, t, self.activation

        raise RuntimeError("Error at patch {}. Relative error doesn't converge: {:.4f}".format(step, relative_error))

    def test(self, words, step):

        self.reset_activation()
        self.normalize_cols()
        activation_tm1 = np.random.normal(size=self.activation.shape)

        # get stimulus
        stimulus = words @ self.Phi

        for t in range(self.max_a_fit):
            self.fit_activation(stimulus)

            dr = activation_tm1 - self.activation
            relative_error = np.sqrt(np.square(dr).sum()) / (self.eps + np.sqrt(np.square(activation_tm1).sum()))
            activation_tm1 = self.activation

            if relative_error < self.eps:
                e2, e1, e0 = self.report_errors(words)
                return e2, e1, e0, t, self.activation

        raise RuntimeError("Error at patch {}. Relative error doesn't converge: {:.4f}".format(step, relative_error))

@IscFactory.register("swsc")
class swaveSC(waveSC):
    '''
    wave algorithm using sparse matrix
    '''
    def __init__(self, **kwargs):
        assert "inhibitory" in kwargs
        assert "W" in kwargs # W is a sparse matrix
        waveSC.__init__(self, **kwargs)
        self.W = kwargs["W"]
        self.inhibitory = kwargs["inhibitory"]
        self.activationT = np.transpose(self.activation)


    def fit_activationT(self, stimulusT):
        self.activationT = self.activationT + \
            self.lr_r * (stimulusT - self.W.dot(self.activationT))
        self.activationT = self.soft_threshold(self.activationT)

    def reset_activationT(self):
            self.activationT = np.zeros(self.activationT.shape)

    def train(self, words, step):

        self.reset_activation()
        self.normalize_cols()
        activationT_tm1 = np.random.normal(size=self.activationT.shape)

        # get stimulus
        stimulusT = (words @ self.Phi).T

        for t in range(self.max_a_fit):
            self.fit_activationT(stimulusT)


            dr = activationT_tm1 - self.activationT
            relative_error = np.sqrt(np.square(dr).sum()) / (self.eps + np.sqrt(np.square(activationT_tm1).sum()))
            activationT_tm1 = self.activationT

            if relative_error < self.eps:
                self.activation = self.activationT.T

                self.train_dictionary(words)
                e2, e1, e0 = self.report_errors(words)
                return e2, e1, e0, t, self.activation

        self.activation = self.activationT.T
        raise RuntimeError("Error at patch {}. Relative error doesn't converge: {:.4f}".format(step, relative_error))

    def test(self, words, step):

        self.reset_activation()
        self.normalize_cols()
        activationT_tm1 = np.random.normal(size=self.activationT.shape)

        # get stimulus
        stimulusT = (words @ self.Phi).T

        for t in range(self.max_a_fit):
            self.fit_activationT(stimulusT)

            dr = activationT_tm1 - self.activationT
            relative_error = np.sqrt(np.square(dr).sum()) / (self.eps + np.sqrt(np.square(activationT_tm1).sum()))
            activationT_tm1 = self.activationT

            if relative_error < self.eps:
                self.activation = self.activationT.T

                e2, e1, e0 = self.report_errors(words)
                return e2, e1, e0, t, self.activation

        self.activation = self.activationT.T
        raise RuntimeError("Error at patch {}. Relative error doesn't converge: {:.4f}".format(step, relative_error))

@IscFactory.sregister("cwsc")
class cwaveSC(_IstaSparseCoding):
    '''
    We specifically need W = (k_exc, k_inh)
    # inhibitory neurons = # excitatory neurons = self.num_units
    '''
    def __init__(self, **kwargs): #TODO
        assert "leaky" in kwargs
        assert "W" in kwargs # W is a sparse matrix
        super().__init__(**kwargs)
        self.k_exc= np.expand_dims(kwargs["W"][0], axis=0)
        self.k_inh = np.expand_dims(kwargs["W"][1], axis=0)
        self.leaky = kwargs["leaky"]
        self.side_len = int(np.sqrt(self.num_units))

    def soft_threshold(self, x):
        return np.maximum(x - self.lmda, 0) - np.maximum(-x - self.lmda, 0)

    def report_errors(self, word_batch):
        activation = self.activation[:,:,:,0].reshape([self.batch_size, self.num_units])
        l2_error = np.square(word_batch - activation @ self.Phi.T).sum() / (self.batch_size)
        l1_norm = np.sum(np.abs(activation)) / (self.batch_size)
        l0_norm = np.sum(np.abs(activation) > 1e-4) / (self.batch_size * self.num_units)
        return l2_error, l1_norm, l0_norm

    def train_dictionary(self, words):
        activation = self.activation[:,:,:,0].reshape([self.batch_size, self.num_units])
        error = words - activation @ self.Phi.T
        dPhi = error.T @ activation
        self.Phi += self.lr_Phi * dPhi
        self.normalize_cols()

    def fit_activation(self, stimulus):
        self.activation_tm1 = np.copy(self.activation)
        exc_input = signal.convolve(self.activation_tm1[:,:,:,0], self.k_exc, mode='same')
        inh_input = signal.convolve(self.activation_tm1[:,:,:,1], self.k_inh, mode='same')
        self.activation[:,:,:,0] = self.activation_tm1[:,:,:,0] + \
            self.lr_r * (- self.leaky * self.activation_tm1[:,:,:,0] + stimulus + exc_input - inh_input)

        self.activation[:,:,:,1] = self.activation_tm1[:,:,:,1] + \
            self.lr_r * (- self.leaky * self.activation_tm1[:,:,:,1] + exc_input)

        self.activation = self.soft_threshold(self.activation)

    def train(self, words, step):
        # get stimulus
        stimulus = (words @ self.Phi).reshape([self.batch_size, self.side_len, self.side_len])

        # initialize activation
        self.activation = np.zeros([self.batch_size, self.side_len, self.side_len, 2])

        for t in range(self.max_a_fit):
            self.fit_activation(stimulus)

            dr = self.activation_tm1 - self.activation
            relative_error = np.sqrt(np.square(dr).sum()) / (self.eps + np.sqrt(np.square(self.activation_tm1).sum()))

            #if t%50 == 0:
                #print("at step {:d} dr: ".format(t), np.sqrt(np.square(dr).sum()))

            if relative_error < self.eps:
                self.train_dictionary(words)
                e2, e1, e0 = self.report_errors(words)
                return e2, e1, e0, t, self.activation[:,:,:,0].reshape((self.batch_size, self.num_units))

        raise RuntimeError("Error at patch {}. Relative error doesn't converge: {:.4f}".format(step, relative_error))

    def get_activation_dynamic(self, words, max_a_fit):
        words = words.reshape((1,-1))
        res = np.empty(shape=(max_a_fit, self.num_units))
        errors = np.empty(shape=(max_a_fit, 3))
        stimulus = (words @ self.Phi).reshape([1, self.side_len, self.side_len])
        self.activation = np.zeros([1, self.side_len, self.side_len, 2])
        activation_tm1 = self.activation.copy()

        for t in range(max_a_fit):
            self.fit_activation(stimulus)
            res[t] = self.activation[:,:,:,0].reshape((1, self.num_units))

            relative_error = np.sqrt(np.square(activation_tm1 - \
                self.activation).sum()) / (self.eps + \
                    np.sqrt(np.square(activation_tm1).sum()))
            # print("At time {}, rerr: {}".format(t, relative_error))

            # Reconstruction error
            e2, e1, e0 = self.report_errors(words)
            errors[t] = [e2,e1,e0]

            if relative_error < self.eps:
                return res[:t,:], errors[:t,:]

            activation_tm1 = self.activation.copy()

        return res, errors




    def test(self, words, step):
        # get stimulus
        stimulus = (words @ self.Phi).reshape([self.batch_size, self.side_len, self.side_len])

        # initialize activation
        self.activation = np.zeros([self.batch_size, self.side_len, self.side_len, 2])

        for t in range(self.max_a_fit):
            self.fit_activation(stimulus)

            dr = self.activation_tm1 - self.activation
            relative_error = np.sqrt(np.square(dr).sum()) / (self.eps + np.sqrt(np.square(self.activation_tm1).sum()))

            #if t%50 == 0:
                #print("at step {:d} dr: ".format(t), np.sqrt(np.square(dr).sum()))

            if relative_error < self.eps:
                # self.train_dictionary(words)
                e2, e1, e0 = self.report_errors(words)
                return e2, e1, e0, t, self.activation[:,:,:,0].reshape((self.batch_size, self.num_units))

        raise RuntimeError("Error at patch {}. Relative error doesn't converge: {:.4f}".format(step, relative_error))

@IscFactory.sregister("dwsc")
class dwaveSC(_IstaSparseCoding):
    '''
    We specifically need W = (k_exc, k_inh)
    # inhibitory neurons = # excitatory neurons = self.num_units
    '''
    def __init__(self, **kwargs): #TODO
        assert "leaky" in kwargs
        assert "W" in kwargs # W is a sparse matrix
        super().__init__(**kwargs)
        self.k_exc= np.expand_dims(kwargs["W"][0], axis=0)
        self.k_inh = np.expand_dims(kwargs["W"][1], axis=0)
        self.leaky = kwargs["leaky"]
        self.side_len = int(np.sqrt(self.num_units))

    def soft_threshold(self, x):
        return np.maximum(x - self.lmda, 0) - np.maximum(-x - self.lmda, 0)

    def report_errors(self, word_batch):
        activation = self.activation[:,:,:,0].reshape([self.batch_size, self.num_units])
        l2_error = np.square(word_batch - activation @ self.Phi.T).sum() / (self.batch_size)
        l1_norm = np.sum(np.abs(activation)) / (self.batch_size)
        l0_norm = np.sum(np.abs(activation) > 1e-4) / (self.batch_size * self.num_units)
        return l2_error, l1_norm, l0_norm

    def train_dictionary(self, words):
        activation = self.activation[:,:,:,0].reshape([self.batch_size, self.num_units])
        error = words - activation @ self.Phi.T
        dPhi = error.T @ activation
        self.Phi += self.lr_Phi * dPhi
        self.normalize_cols()

    def fit_activation(self, stimulus):
        self.activation_tm1 = np.copy(self.activation)
        self.activation[:,:,:,1] = self.activation_tm1[:,:,:,1]
        self.activation[:,:,:,0] = self.soft_threshold(stimulus)

    def train(self, words, step):
        # get stimulus
        stimulus = (words @ self.Phi).reshape([self.batch_size, self.side_len, self.side_len])

        # initialize activation
        self.activation = np.zeros([self.batch_size, self.side_len, self.side_len, 2])

        for t in range(self.max_a_fit):
            self.fit_activation(stimulus)

            dr = self.activation_tm1 - self.activation
            relative_error = np.sqrt(np.square(dr).sum()) / (self.eps + np.sqrt(np.square(self.activation_tm1).sum()))

            #if t%50 == 0:
                #print("at step {:d} dr: ".format(t), np.sqrt(np.square(dr).sum()))

            if relative_error < self.eps:
                self.train_dictionary(words)
                e2, e1, e0 = self.report_errors(words)
                return e2, e1, e0, t, self.activation[:,:,:,0].reshape((self.batch_size, self.num_units))

        raise RuntimeError("Error at patch {}. Relative error doesn't converge: {:.4f}".format(step, relative_error))

    def get_activation_dynamic(self, words, max_a_fit):
        words = words.reshape((1,-1))
        res = np.empty(shape=(max_a_fit, self.num_units))
        errors = np.empty(shape=(max_a_fit, 3))
        stimulus = (words @ self.Phi).reshape([1, self.side_len, self.side_len])
        self.activation = np.zeros([1, self.side_len, self.side_len, 2])
        activation_tm1 = self.activation.copy()

        for t in range(max_a_fit):
            self.fit_activation(stimulus)
            res[t] = self.activation[:,:,:,0].reshape((1, self.num_units))

            relative_error = np.sqrt(np.square(activation_tm1 - \
                self.activation).sum()) / (self.eps + \
                    np.sqrt(np.square(activation_tm1).sum()))
            # print("At time {}, rerr: {}".format(t, relative_error))

            # Reconstruction error
            e2, e1, e0 = self.report_errors(words)
            errors[t] = [e2,e1,e0]

            if relative_error < self.eps:
                return res[:t,:], errors[:t,:]

            activation_tm1 = self.activation.copy()

        return res, errors




    def test(self, words, step):
        # get stimulus
        stimulus = (words @ self.Phi).reshape([self.batch_size, self.side_len, self.side_len])

        # initialize activation
        self.activation = np.zeros([self.batch_size, self.side_len, self.side_len, 2])

        for t in range(self.max_a_fit):
            self.fit_activation(stimulus)

            dr = self.activation_tm1 - self.activation
            relative_error = np.sqrt(np.square(dr).sum()) / (self.eps + np.sqrt(np.square(self.activation_tm1).sum()))

            #if t%50 == 0:
                #print("at step {:d} dr: ".format(t), np.sqrt(np.square(dr).sum()))

            if relative_error < self.eps:
                # self.train_dictionary(words)
                e2, e1, e0 = self.report_errors(words)
                return e2, e1, e0, t, self.activation[:,:,:,0].reshape((self.batch_size, self.num_units))

        raise RuntimeError("Error at patch {}. Relative error doesn't converge: {:.4f}".format(step, relative_error))


@IscFactory.register("isc")
class ISC(_IstaSparseCoding):

    def __init__(self, num_inputs=0, num_units=0, batch_size=0, \
        max_a_fit = 1000, eps = 5e-3,
        lr_r=1e-2, lr_Phi=1e-2, lmda=1e-3, **kwargs):
        _IstaSparseCoding.__init__(self, num_inputs, num_units, batch_size, \
        max_a_fit, eps, lr_r, lr_Phi, lmda)

    def soft_threshold(self, x):
        return np.maximum(x - self.lmda, 0) - np.maximum(-x - self.lmda, 0)

    def report_errors(self, word_batch):
        l2_error = np.square(word_batch - self.activation @ self.Phi.T).sum() / (self.batch_size)
        l1_norm = np.sum(np.abs(self.activation)) / (self.batch_size)
        l0_norm = np.sum(np.abs(self.activation) > 1e-4) / (self.batch_size * self.num_units)
        return l2_error, l1_norm, l0_norm

    def train_dictionary(self, word_batch):
        error = word_batch - self.activation @ self.Phi.T
        dPhi = error.T @ self.activation
        self.Phi += self.lr_Phi * dPhi


@IscFactory.sregister("groupIsc")
class GroupISC(ISC):

    def __init__(self, num_inputs, num_units, batch_size,
        group_mat,
        max_a_fit = 1000, eps = 5e-3,
            lr_r=1e-2, lr_Phi=1e-2, lmda=1e-3, **kwargs):
        # group mat: num_unit rows and num_groups columns and , G[i,j] weight of unit j in group i
        ISC.__init__(self, num_inputs, num_units, batch_size, \
            max_a_fit, eps, lr_r, lr_Phi, lmda)
        self.G = group_mat
        self.gMembership = np.transpose(group_mat > 0)

    def soft_threshold(self, x):
        # get group norm
        group_norm = np.maximum(np.sqrt(x**2 @ self.G),0.000000001)
        # thresholding factor
        gtf = np.maximum(1 - self.lmda/group_norm, 0) # (bs, num_group)
        # threshould outer product
        gtf = gtf[:,:,None] * self.gMembership #(bs, num_group, nunit)
        #
        thres = gtf.max(axis = 1)
        return thres * x

@IscFactory.sregister("lazyGroupIsc")
class LazyGroupISC(ISC):

    def __init__(self, num_inputs, num_units, batch_size,
        group_mat,
        max_a_fit = 1000, eps = 5e-3,
            lr_r=1e-2, lr_Phi=1e-2, lmda=1e-3, **kwargs):
        # group mat: num_unit rows and num_groups columns and , G[i,j] weight of unit j in group i
        ISC.__init__(self, num_inputs, num_units, batch_size, \
            max_a_fit, eps, lr_r, lr_Phi, lmda)
        self.cnt = 0
        self.EARLYREGISTRATION = 200
        self.G = group_mat
        self.gMembership = np.transpose(group_mat > 0)

    def soft_threshold(self, x):
        # get group norm
        group_norm = np.maximum(np.sqrt(x**2 @ self.G),0.000000001)
        # thresholding factor
        gtf = np.maximum(1 - self.lmda/group_norm, 0) # (bs, num_group)
        # threshould outer product
        gtf = gtf[:,:,None] * self.gMembership #(bs, num_group, nunit)
        #
        thres = gtf.max(axis = 1)
        return thres * x
@IscFactory.register("lisc")
class LISC(ISC):

    def __init__(self, num_inputs=0, num_units=0, batch_size=0, \
        max_a_fit = 1000, eps = 5e-3,
        lr_r=1e-2, lr_Phi=1e-2, lmda=1e-3, lr_G=1e-3, **kwargs):
        ISC.__init__(self, num_inputs=num_inputs, num_units=num_units, batch_size=batch_size, \
            max_a_fit = max_a_fit, eps =eps,
            lr_r=lr_r, lr_Phi=lr_Phi, lmda=lmda)
        self.G = np.identity(self.num_units)
        self.dG = np.identity(self.num_units)
        self.lr_G = lr_G

    def fit_activation(self, words):
        # error.shape = (bs, 2, num_inputs)
        dactivation = words @ self.Phi - self.activation @ self.G
        activation = self.activation + self.lr_r * dactivation
        self.activation = self.soft_threshold(activation)

    def train_dictionary(self, words):
    	# update Phi
        # local update
        D = np.square(self.activation).sum(axis = 0)
        dPhi = words.T @ self.activation - self.Phi * D

        # non-local update
        # error = words - self.activation @ self.Phi.T
        # dPhi = error.T @ self.activation
        self.Phi += self.lr_Phi * dPhi
        # update G
        # self.dG = correlation_matrix(self.activation.T)
        self.dG = normalized_gram_matrix(self.activation.T)
        self.G = (1-self.lr_G) * self.G + self.lr_G * self.dG

@IscFactory.register("jLisc")
class JLISC(ISC):

    def __init__(self, num_inputs=0, num_units=0, batch_size=0, \
        max_a_fit = 1000, eps = 5e-3,
        lr_r=1e-2, lr_Phi=1e-2, lmda=1e-3, lr_G=1e-3, **kwargs):
        ISC.__init__(self, num_inputs=num_inputs, num_units=num_units, batch_size=batch_size, \
            max_a_fit = max_a_fit, eps =eps,
            lr_r=lr_r, lr_Phi=lr_Phi, lmda=lmda)
        self.G = np.identity(self.num_units)
        self.dG = np.identity(self.num_units)
        self.lr_G = lr_G

    def fit_activation(self, words):
        # error.shape = (bs, 2, num_inputs)
        dactivation = words @ self.Phi - self.activation @ self.G
        activation = self.activation + self.lr_r * dactivation
        self.activation = self.soft_threshold(activation)

    def train_dictionary(self, words):
        # update G
        # self.dG = correlation_matrix(self.activation.T)
        self.dG = normalized_gram_matrix(self.activation.T)
        self.G = (1-self.lr_G) * self.G + self.lr_G * self.dG

    	# update Phi
        # local update \sum_{j} <a_i, a_j> G_{ij} Phi_i
        D = np.sum(self.activation.T @ self.activation * self.G, axis = 1)
        dPhi = words.T @ self.activation - self.Phi * D

        self.Phi += self.lr_Phi * dPhi




@IscFactory.register("lazyLisc")
class LazyLISC(LISC):
    def __init__(self, **kwargs):
        LISC.__init__(self, **kwargs)
        self.sleep = 0

    def train_dictionary(self, words):
    	# update Phi
        self.sleep += 1

        if self.sleep == 10:
            self.sleep = 0
            D = np.square(self.activation).sum(axis = 0)
            dPhi = words.T @ self.activation - self.Phi * D

            # error = words - self.activation @ self.Phi.T
            # dPhi = error.T @ self.activation
            self.Phi += self.lr_Phi * dPhi
        # update G
        # self.dG = correlation_matrix(self.activation.T)
        self.dG = normalized_gram_matrix(self.activation.T)
        self.G = (1-self.lr_G) * self.G + self.lr_G * self.dG


@IscFactory.register("withGLisc")
class withGLISC(LISC):
    def __init__(self, **kwargs):
        LISC.__init__(self, **kwargs)

    def train_dictionary(self, words):
    	# update Phi

        # D = np.square(self.activation).sum(axis = 0)
        # dPhi = words.T @ self.activation - self.Phi * D

        error = words - self.activation @ self.Phi.T
        dPhi = error.T @ self.activation
        self.Phi += self.lr_Phi * dPhi
        # update G
        # self.dG = correlation_matrix(self.activation.T)
        self.dG = normalized_gram_matrix(self.activation.T)
        self.G = (1-self.lr_G) * self.G + self.lr_G * self.dG


@IscFactory.register("noGlisc")
class NoGLISC(ISC):

    def __init__(self, num_inputs=0, num_units=0, batch_size=0, \
        max_a_fit = 1000, eps = 5e-3,
        lr_r=1e-2, lr_Phi=1e-2, lmda=1e-3, lr_G=1e-3, **kwargs):
        ISC.__init__(self, num_inputs=num_inputs, num_units=num_units, batch_size=batch_size, \
            max_a_fit = max_a_fit, eps =eps,
            lr_r=lr_r, lr_Phi=lr_Phi, lmda=lmda)
        self.G = np.identity(self.num_units)
        self.dG = np.identity(self.num_units)
        self.lr_G = lr_G

    # def fit_activation(self, words):
    #     # error.shape = (bs, 2, num_inputs)
    #     dactivation = words @ self.Phi - self.activation @ self.G
    #     activation = self.activation + self.lr_r * dactivation
    #     self.activation = self.soft_threshold(activation)

    def train_dictionary(self, words):
    	# update Phi

        D = np.square(self.activation).sum(axis = 0)
        dPhi = words.T @ self.activation - self.Phi * D

        # error = words - self.activation @ self.Phi.T
        # dPhi = error.T @ self.activation
        self.Phi += self.lr_Phi * dPhi
        # update G
        # self.dG = correlation_matrix(self.activation.T)
        self.dG = normalized_gram_matrix(self.activation.T)
        self.G = (1-self.lr_G) * self.G + self.lr_G * self.dG


@IscFactory.sregister("lssc")
class LSSC(LISC):
    def __init__(self, num_inputs, num_units, batch_size, \
        struct_mat,
        max_a_fit = 1000, eps = 5e-3,
        lr_r=1e-2, lr_Phi=1e-2, lr_G=5e-2, lmda=1e-3, **kwargs):
        LISC.__init__(self, num_inputs=num_inputs, num_units=num_units, batch_size=batch_size, \
            max_a_fit = max_a_fit, eps =eps,
            lr_r=lr_r, lr_Phi=lr_Phi, lr_G=lr_G, lmda=lmda)
        self.struct_mat = struct_mat

    def soft_threshold(self, x):
        thres = np.abs(x) @ self.struct_mat * self.lmda
        return np.maximum(x - thres, 0) - np.maximum(-x - thres, 0)



@IscFactory.sregister("gsc")
class GSC(ISC):
    def __init__(self, **kwargs):
        if not "Laplacian" in kwargs:
            RuntimeError("For graphical models we should to specify the Laplacian.")
        ISC.__init__(self, **kwargs)
        self.Laplacian = kwargs["Laplacian"]
        self.GLMDA = kwargs.get("GLMDA", 1)

    def fit_activation(self, words):
        error = words - self.activation @ self.Phi.T
        excitation = - self.activation @ self.Laplacian * self.GLMDA
        dactivation = error @ self.Phi + excitation
        activation = self.activation + self.lr_r * dactivation
        self.activation = self.soft_threshold(activation)

@IscFactory.sregister("ssc")
class SSC(ISC):
    def __init__(self, num_inputs, num_units, batch_size,
        struct_mat,
        max_a_fit = 1000, eps = 5e-3,
            lr_r=1e-2, lr_Phi=1e-2, lmda=1e-3, **kwargs):
        ISC.__init__(self, num_inputs, num_units, batch_size, \
            max_a_fit, eps, lr_r, lr_Phi, lmda)
        self.struct_mat = struct_mat


    def soft_threshold(self, x):
        thres = np.abs(x) @ self.struct_mat * self.lmda
        return np.maximum(x - thres, 0) - np.maximum(-x - thres, 0)

class SSCJ(SSC):
    def __init__(self, num_inputs, num_units, batch_size,
        struct_mat,
        max_a_fit = 1000, eps = 5e-3,
            lr_r=1e-2, lr_Phi=1e-2, lmda=1e-3):
        SSC.__init__(self, num_inputs, num_units, batch_size, \
            struct_mat,
            max_a_fit, eps, lr_r, lr_Phi, lmda)

    def soft_threshold(self, x):
        thres = np.abs(x) @ self.struct_mat * self.lmda
        return np.maximum(x - thres, 0) - np.maximum(-x - thres, 0)


class ParallelISC(_IstaSparseCoding):

    def __init__(self, num_inputs=0, num_units=0, batch_size=0, \
        max_a_fit = 1000, eps = 5e-3,
        lr_r=1e-2, lr_Phi=1e-2, lmda=1e-3):

        _IstaSparseCoding.__init__(self, num_inputs=num_inputs, num_units = num_units,\
            batch_size = batch_size, max_a_fit = max_a_fit, eps=eps, lr_r=lr_r, \
                lr_Phi = lr_Phi, lmda = lmda)

        self.activation = np.zeros((self.batch_size, 2, self.num_units))

    def soft_threshold(self, x):
        x = x * np.maximum(np.abs(x) - self.lmda, 0)/ (np.abs(x) + self.eps)
        return x

    def report_errors(self, word_pairs):
        l2_error = np.square(word_pairs - self.activation @ self.Phi.T).sum() / (self.batch_size * 2)
        l1_norm = np.sum(np.abs(self.activation)) / (self.batch_size * 2)
        l0_norm = np.sum(np.abs(self.activation) > 0) / (self.batch_size * 2 * self.num_units)
        return l2_error, l1_norm, l0_norm

    def train_dictionary(self, word_pairs):
        error = word_pairs - self.activation @ self.Phi.T
        dPhi = error[:,0,:].T @ self.activation[:,0,:]
        dPhi += error[:,1,:].T @ self.activation[:,1,:]
        self.Phi += self.lr_Phi * dPhi


class PairedISC(ParallelISC):

    def __init__(self, num_inputs=0, num_units=0, batch_size=0, \
        max_a_fit = 1000, eps = 5e-3,
        lr_r=1e-2, lr_Phi=1e-2, lmda=1e-3):
        ParallelISC.__init__(self, num_inputs = num_inputs, num_units = num_units, \
            batch_size = batch_size, max_a_fit = max_a_fit, \
                eps = eps, lr_r = lr_r, lr_Phi = lr_Phi, lmda = lmda)

    def soft_threshold(self, x):
        norm = np.expand_dims(np.abs(x).sum(axis = 1), axis = 1)
        x = x * np.maximum(norm - self.lmda, 0)/ (norm + self.eps)
        return x


class ParallelSSC(ParallelISC):
    def __init__(self, num_inputs=0, num_units=0, batch_size=0, \
        max_a_fit = 1000, eps = 5e-3,
        lr_r=1e-2, lr_Phi=1e-2, lmda=1e-3, struct_mat=0):
        ParallelSSC.__init__(self, num_inputs = num_inputs, num_units = num_units, batch_size = batch_size, \
            max_a_fit = max_a_fit, eps = eps, lr_r = lr_r, lr_Phi = lr_Phi, lmda = lmda)

        self.struct_mat = struct_mat

    def soft_threshold(self, x):
        thresh = np.abs(x) @ self.struct_mat * self.lmda
        x = x * np.maximum(np.abs(x) - thresh, 0)/ (np.abs(x) + self.eps)
        return x



class PairedSSC(PairedISC):
    def __init__(self, num_inputs=0, num_units=0, batch_size=0, \
        max_a_fit = 1000, eps = 5e-3,
        lr_r=1e-2, lr_Phi=1e-2, lmda=1e-3, struct_mat=0):
        PairedISC.__init__(self, num_inputs = num_inputs, num_units = num_units, batch_size = batch_size, \
            max_a_fit = max_a_fit, eps = eps, lr_r = lr_r, lr_Phi = lr_Phi, lmda = lmda)

        # construct the structured matrix
        self.struct_mat = struct_mat


    def soft_threshold(self, x):
        norm = np.expand_dims(np.sqrt(np.square(x).sum(axis = 1)), axis = 1)
        thresh = norm @ self.struct_mat * self.lmda
        x = x * np.maximum(norm - thresh, 0)/ (norm + self.eps)
        return x

class PairedSSCJ(PairedSSC):
    def __init__(self, num_inputs, num_units, batch_size, \
        struct_mat,
        max_a_fit = 1000, eps = 5e-3,
        lr_r=1e-2, lr_Phi=1e-2, lmda=1e-3):
        PairedSSC.__init__(self, num_inputs=num_inputs, num_units=num_units, batch_size=batch_size, \
        struct_mat = struct_mat,
        max_a_fit = max_a_fit, eps = eps, lr_r = lr_r, lr_Phi = lr_Phi, lmda = lmda)

    def soft_threshold(self, x):
        norm = np.expand_dims(np.sqrt(np.square(x).sum(axis = 1)), axis = 1)
        # norm = np.expand_dims(np.abs(x).sum(axis = 1), axis = 1)
        thresh = np.abs(x) @ self.struct_mat * self.lmda
        x = x * np.maximum(norm - thresh, 0)/ (norm + self.eps)
        return x
