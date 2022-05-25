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



class wavenet:
    '''
    Classification model with activations using waves
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

    def relu(self, x):
        return x*(x>0)

    def sigmoid(self, x):
        return 1./(1+np.exp(-x))

    def __init__(self, num_inputs, num_outputs, num_units, batch_size=1,
        eps = 5e-3, step_size=1e-2, lmda=1e-2, lr_r=0.01, lr_Phi=0.02, r1=3, r2=5, wi=1, we=5,
            sigmaE=3, leaky=0, use_waves=True, **kwargs):
        self.D = num_inputs
        self.H1 = num_units
        self.K = num_outputs
        self.step_size = step_size
        self.W1 = 0.3 * np.random.randn(self.D, self.H1)
        self.b1 = np.zeros((1,self.H1))
        self.W2 = 0.1 * np.random.randn(self.H1, self.K)
        self.b2 = np.zeros((1, self.K))
        self.side_len = int(np.sqrt(num_units))
        self.batch_size = batch_size
        self.lmda = lmda
        self.use_waves = use_waves
        if leaky==0:
            self.leaky = wi+we


        W = get_kernels(r1, r2, wi, we, sigmaE)
        self.cwsc = cwaveSC(
            num_inputs=3,
            num_units = self.side_len**2,
            batch_size = batch_size,
            lr_r = lr_r,
            lr_Phi = lr_Phi,
            lmda = self.lmda,
            W = W,
            leaky=self.leaky
        )

    def reset_activation(self, batch_size = None):
        if batch_size is None:
            self.activation = np.zeros(self.activation.shape)
        else:
            print(self.activation.shape)
            self.activation = np.zeros((batch_size, *self.activation.shape[1:]))

    def dactivation(self, dscores, X):
        dhidden = np.dot(dscores, self.W2.T)
        stimulus = (X @ self.W1)
        #activations = np.tanh(stimulus)
        #dhidden = dhidden * (1-np.square(activations))
        activations = self.sigmoid(stimulus)
        dhidden = dhidden * activations * (1-activations)
        return dhidden

    def derror_activation(self, dscores, X):
        derror = np.dot(dscores, self.W2.T)
        derror_activations, s = self.error_activation(derror)
        stimulus = (X @ self.W1)
        activations = self.sigmoid(stimulus)
        dhidden = derror_activations * activations * (1-activations)
        return dhidden

    def sigmoid(self, x):
        return 1./(1+np.exp(-x))

    def activation_soft_threshold(self, X):
        stimulus = (X @ self.W1).reshape((X.shape[0], self.side_len, self.side_len))
        activations = self.sigmoid(stimulus)
        activations = self.soft_threshold(activations).reshape((X.shape[0], self.side_len * self.side_len))
        return activations, 1

    def activation(self, X):
        if self.use_waves == False:
            return self.activation_soft_threshold(X)

        stimulus = (X @ self.W1).reshape((X.shape[0], self.side_len, self.side_len))
        stimulus = self.sigmoid(stimulus)

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

    def error_activation(self, derror):

        stimulus = derror.reshape((derror.shape[0], self.side_len, self.side_len))
        self.cwsc.activation = np.zeros([derror.shape[0], self.side_len, self.side_len, 2])
        for step in np.arange(1,50):
            self.cwsc.fit_activation(stimulus)
            dr = self.cwsc.activation_tm1 - self.cwsc.activation
            relative_error = np.sqrt(np.square(dr).sum()) / \
                (self.cwsc.eps + np.sqrt(np.square(self.cwsc.activation_tm1).sum()))
            if relative_error < self.cwsc.eps:
                break
            a = self.cwsc.activation[:,:,:,0]

        activations = self.cwsc.activation[:,:,:,0].reshape((derror.shape[0], self.side_len * self.side_len))
        return activations, step


    def clip(self, x, c):
        return np.sign(x)*np.minimum(np.abs(x), c)


    def train(self, X, y, gradient_steps=20000, initial_step=0):
        loss = []
        sparsity = []
        steps = []
        num_examples = X.shape[0]
        tbar = trange(initial_step, initial_step+gradient_steps, desc='Training', leave=True)

        for i in tbar:
            inds = np.random.randint(0, num_examples, self.batch_size)
            this_X = X[inds, :]
            this_y = y[inds]

            # evaluate class scores, [N x K]
            hidden_layer, step = self.activation(this_X)
            scores = np.dot(hidden_layer, self.W2) + self.b2
            steps.append(step)

            # compute the class probabilities
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

            # compute the loss
            correct_logprobs = -np.log(probs[range(self.batch_size), this_y])
            current_loss = np.sum(correct_logprobs)/self.batch_size
            loss.append(current_loss)

            current_sparsity = np.sum(np.abs(hidden_layer)>1e-5)/np.prod(hidden_layer.shape)
            sparsity.append(current_sparsity)

            if i % 20 == 0:
                tbar.set_description("loss=%.3f sparsity=%2.2f%% steps=%d" % \
                    (current_loss, 100*current_sparsity, step))
                tbar.refresh()

            # compute the gradient on scores
            dscores = np.array(probs)
            dscores[range(self.batch_size), this_y] -= 1
            dscores /= self.batch_size

            # backpropate the gradient to the parameters
            # first backprop into parameters W2 and b2
            dW2 = np.dot(hidden_layer.T, dscores)
            db2 = np.sum(dscores, axis=0, keepdims=True)

            # next backprop into hidden layer
            #dhidden = self.dactivation(dscores, this_X)
            dhidden = self.derror_activation(dscores, this_X)
            dhidden[np.abs(hidden_layer) <= 1e-5] = 0

            # finally into W,b
            dW1 = np.dot(this_X.T, dhidden)
            #db1 = np.sum(dhidden, axis=0, keepdims=True)

            # perform a parameter update
            self.W1 += -self.step_size * dW1
            #self.b1 += -self.step_size * db1
            self.W2 += -self.step_size * dW2
            self.W2 = self.clip(self.W2, 1)
            self.b2 += -self.step_size * db2

            #if i % 100 == 0:
            #    np.save('W1-%d' % i, self.W1)
            #    np.save('W2-%d' % i, self.W2)
            #    np.save('b2-%d' % i, self.b2)

        return loss, sparsity, steps

    def predict(self, X):
        hidden_layer, step = self.activation(X)
        scores = np.dot(hidden_layer, self.W2) + self.b2
        predicted_class = np.argmax(scores, axis=1)
        return predicted_class


class wavenetSC:
    '''
    sparse coding model with activations using waves
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

    def relu(self, x):
        return x*(x>0)

    def sigmoid(self, x):
        return 1./(1+np.exp(-x))

    def __init__(self, num_inputs, num_outputs, num_units, batch_size=1,
        eps = 5e-3, lmda=1e-2, lr_r=0.01, lr_W1=0.02, r1=3, r2=5, wi=1, we=5,
            sigmaE=3, leaky=0, **kwargs):
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

    def dictionary_gradient_step(self, words, activations):
        error = words - activations @ self.W1.T
        dW1 = error.T @ activations
        self.W1 += self.lr_W1 * dW1
        self.normalize_cols()

    def dictionary_local_gradient_step(self, words, activations):
        a = activations.reshape((words.shape[0], self.side_len, self.side_len))[:, np.newaxis, :, :]
        Phi = self.W1.reshape((words.shape[1], self.side_len, self.side_len))[np.newaxis, :, :, :]
        aPhi = a*Phi # (bs, d, l, l)
        fit = signal.convolve(aPhi, self.k_grad, mode = "same") # (bs, d, l, l)
        error = words[:, :, np.newaxis, np.newaxis] - fit # (bs, d, l, l)
        dW1 = (a * error).sum(axis =0)
        self.W1 += self.lr_W1 * dW1.reshape((words.shape[1], self.side_len*self.side_len))
        self.normalize_cols()

    def dictionary_smoothed_gradient_step(self, words, activations):
        D = np.square(activations).sum(axis = 0)
        dW1 = words.T @ activations - self.W1 * D
        self.W1 += self.lr_W1 * dW1
        self.normalize_cols()

    def normalize_cols(self):
        self.W1 = self.W1 / np.maximum(np.sqrt(np.square(self.W1).sum(axis=0)), 1e-8)

    def report_errors(self, words, activations):
        l2_error = np.square(words - activations @ self.W1.T).sum() / activations.shape[0]
        l1_norm = np.sum(np.abs(activations)) / activations.shape[0]
        l0_norm = np.sum(np.abs(activations) > 1e-4) / np.prod(activations.shape)
        return l2_error, l1_norm, l0_norm

    def train(self, X, gradient_steps=20000, initial_step=0):
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
            self.dictionary_gradient_step(this_X, activations)
            l2l, l1l, l0l = self.report_errors(this_X, activations)

            steps.append(step)
            l2_loss.append(l2l)
            l1_loss.append(l1l)
            l0_loss.append(l0l)

            if i % 20 == 0:
                tbar.set_description("loss=%.3f sparsity=%2.2f%% steps=%d" % \
                    (l2l, 100*l0l, step))
                tbar.refresh()

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
                tbar.set_description("loss=%.3f sparsity=%2.2f%% steps=%d" % \
                    (l2l, 100*l0l, step))
                tbar.refresh()

            #if i % 100 == 0:
            #    np.save('W1-%d' % i, self.W1)
            #    np.save('W2-%d' % i, self.W2)
            #    np.save('b2-%d' % i, self.b2)

        return l2_loss, l1_loss, l0_loss, steps
