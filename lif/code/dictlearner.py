'''
==========
Learning Algorithms:
collecting activities of neurons from stimulus 
==========
'''

from abc import ABC, abstractmethod
import pickle
import numpy as np 
from lif import LIF, KGLIF
from utils import Config, calc_norms

class DictLearnerFactory:

    registry = {}

    @classmethod
    def register(cls, name: str):

        def inner_wrapper(wrapped_class):
            if name in cls.registry:
                print('DictLearning algorithm %s already exists. Will replace it', name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

class _DictLearner(ABC):

    def __init__(self, config: Config, lif: LIF):
        self.acts = -1
        self.lif = lif 
        self.config = config
        # model config
        self.num_inputs = config.emb_dim
        self.num_units = config.num_units
        self.batch_size = config.batch_size

        # hyper parameters 
        self.lr_Phi = config.lr_Phi

        # codebook 
        self.Phi = np.random.randn(self.num_inputs, self.num_units) / np.sqrt(self.num_inputs)
    
    @abstractmethod
    def dump(self, t):
        pass 

    @abstractmethod
    def load(self, t):
        pass 

    @abstractmethod
    def train_dictionary(self, words, acts):
        pass
    
    def get_record(self, X, acts):
        avgFiringCount = acts.mean()
        avgFiringRate = np.mean(acts > 0)
        Xhat = acts @ self.Phi.T 
        reconstructionError = np.square(X - Xhat).sum()/self.batch_size # X has norm 16
        reconstructionNorm = np.sqrt(np.square(Xhat).sum(axis = 0)).mean()
        return (avgFiringCount, avgFiringRate, reconstructionError, reconstructionNorm)
    
    def train(self, words, step):

        stimulus = words @ self.Phi
        self.acts = self.lif.stimulate(stimulus)
        if step % 100 == 0:
            print("avg firing", self.acts.mean())
        self.train_dictionary(words, self.acts)

        record = self.get_record(words, self.acts)

        return record

@DictLearnerFactory.register("globalnorm")
class Global(_DictLearner):
    
    def __init__(self, config: Config, lif: LIF):
        _DictLearner.__init__(self, config, lif)
        self.acts = np.zeros((self.batch_size, self.num_units))
        self.dPhi = np.zeros(self.Phi.shape)

    def dump(self, t):
        with open(self.config.fpath + "global" + str(t), 'wb') as f:
            pickle.dump(self, f)
    
    def dump_codewords(self, t):
        np.save(self.config.fpath + "Phi" + str(t) + ".npy", self.Phi)

    def dump_acts(self, t):
        np.save(self.config.fpath + "acts" + str(t) + ".npy", self.acts)
    
    def load(self, t):
        with open(self.config.fpath + "global" + str(t), 'rb') as f:
            self.Phi = pickle.load(f).Phi
        
    def load_codewords(self, t):
        self.Phi = np.load(self.config.fpath + "Phi" + str(t) + ".npy")

    def train_dictionary(self, words, acts):
        error = words - acts @ self.Phi.T
        self.dPhi = error.T @ acts
        self.Phi += self.lr_Phi * self.dPhi/self.batch_size
        self.Phi = self.Phi / np.sqrt(np.square(self.Phi).sum(axis = 0) + 1e-8)

@DictLearnerFactory.register("globalposnormreg")
class GlobalPos(Global):

    def __init__(self, config, lif : KGLIF) -> None:
        super().__init__(config, lif)
        
        inh_loc_idx = lif.inh_loc_idx
        self.Phi[:, inh_loc_idx] = 0
        self.dPhi = np.zeros(self.Phi.shape)
        self.firing_UB = 2 * config.target_rate

        exc_loc = np.ones(self.num_units)
        exc_loc[inh_loc_idx] = 0
        self.exc_loc = exc_loc.astype(int)

    def get_stimulus(self, words):
        stimulus = words @ self.Phi * 1e-4
        stimulus[stimulus < 0] = 0
        return stimulus

    def train(self, words):
        stimulus = words @ self.Phi * 1e-4
        stimulus[stimulus < 0] = 0
        acts = self.lif.stimulate(stimulus) * self.exc_loc

        record = self.get_record(words, acts)
        if record[1] > self.firing_UB: # firing porportions can not exceed 0.2
            return (record[0],record[1],0,0)
        
        self.train_dictionary(words, acts)
        
        return record


@DictLearnerFactory.register("globalposnormregl1")
class GlobalPosNormRegL1(GlobalPos):
    def __init__(self, config, lif : KGLIF) -> None:
        super().__init__(config, lif)
        self.l1_UB = 1.1 * config.target_l1

    def train(self, words, step):
        stimulus = words @ self.Phi * 1e-4
        stimulus[stimulus < 0] = 0
        acts = self.lif.stimulate(stimulus) * self.exc_loc

        record = self.get_record(words, acts)
        if record[0] > self.l1_UB:
            return (record[0],record[1],0,0)
        
        self.train_dictionary(words, acts)
        return record