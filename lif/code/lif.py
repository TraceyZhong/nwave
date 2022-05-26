'''
==========
LIF dynamic
==========
'''
from abc import ABC, abstractmethod
import pickle
import numpy as np
from utils import Config, get_sK_matrix, Log

def calc_norms(matrix):
    return np.sqrt(np.square(matrix).sum(axis=0))

class LIFFactory:
    """ The factory class for LIF models
    Ref: https://medium.com/@geoffreykoh/implementing-the-factory-pattern-via-dynamic-registry-and-python-decorators-479fc1537bbe
    """

    registry = {}
    """ Internal registry for available LIF models """

    @classmethod
    def register(cls, name: str):

        def inner_wrapper(wrapped_class):
            if name in cls.registry:
                print('LIF model %s already exists. Will replace it', name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

class LIF(ABC):

    '''abstract method to implement
    init_internal_state()
    update_conductance()
    euler_step()
    '''

    def __init__(self, config: Config):

        self.config = config
        self.logger = config.logger 

        # model config
        self.num_units = config.num_units
        self.batch_size = config.batch_size
        self.shape = (self.batch_size, self.num_units)
        self.num_steps = config.num_euler_steps

        # physical constants
        self.GL = 5e-5 # leak conductance
        self.VL = -0.07 # leak reversal potential 
        self.VE = 0 # excitatory reversal potential 
        self.VI = -0.08 # inhibitory reversal potential 
        self.TREF = 5e-3 # refractory period 
        self.VT = -0.05 # voltage threshold
        self.VR = -0.07 # reset voltage
        self.C = 1e-6 # capacitance
        self.FE = 15e-6 # excitatory external synaptic conductance
        self.FI = 2e-6 # inhibitory synaptic conductance
        self.TERISE = 5e-4 # excitary rise time 
        self.TIRISE = 5e-4 # inhibitory rise time 
        self.TEDECAY = 2e-3 # excitary decay time 
        self.TIDECAY = 7e-3 # inhibitary decay time
        self.TUNIT = 5e-5 # time unit for each euler step
        self.TREFoverTUNIT = 1

    @abstractmethod
    def init_internal_state(self):
        self.voltage = np.zeros(self.shape)
        pass 

    @abstractmethod
    def reset(self):
        pass 

    @abstractmethod
    def dump(self, t):
        pass 

    @abstractmethod
    def load(self, t):
        pass

    def stimulate(self, stimulus):
        activities = np.zeros(self.shape)

        for t in range(self.num_steps):
            self.euler_step(stimulus, t)
            firing = self.spike()
            self.update_conductance(firing)
            activities += firing

        self.evolve(activities)
        self.reset()

        return activities

    def spike(self):
        hibernate = (self.ref_count_down > 0)
        self.voltage[hibernate] = self.VR 
        firing = self.voltage > self.VT
        self.voltage[firing] = self.VR
        self.ref_count_down = np.maximum(self.ref_count_down - 1, firing * self.TREFoverTUNIT)
        return firing

@LIFFactory.register("kgliflight")
class KGLIF(LIF):
    '''
    This model differs to the KG's model that
    1) refractory period is 1
    '''

    def __init__(self, config):
        super().__init__(config)
        inh_loc_idx, K = get_sK_matrix(config.num_units, config.RE, config.RI, \
            we = config.WE, wi = config.WI, sigmaE = 2, inhibitory_ratio=0.2)
        self.Kt = K.T * 1e-3
        self.TARGET_RATE = config.target_rate # target firing rate
        self.inh_loc_idx = inh_loc_idx
        self.lr_VT = config.lr_VT
        self.VT = config.VT
    
    def dump(self, t):
        pass 

    def load(self, t):
        pass 

    def init_internal_state(self):
        self.voltage = (self.VT - self.VR) * np.random.uniform(size=self.shape) + self.VR
        self.ref_count_down = np.full(self.shape, 0)
        self.exc_peripheral_synaptic_conductance = np.zeros(self.shape)
        self.inh_peripheral_synaptic_conductance = np.zeros(self.shape) # G in their paper

    def reset(self):
        self.init_internal_state()
    
    def update_conductance(self, firing):
        '''This is the G component in their paper.
        For this model, let's set this as just the last time firing
        '''
        self.exc_peripheral_synaptic_conductance = firing
        self.inh_peripheral_synaptic_conductance = firing

    def euler_step(self, stimulus, t):
        try: 
            exc_input = stimulus + self.FE + self.exc_peripheral_synaptic_conductance @ self.Kt
            inh_input = self.FI + self.inh_peripheral_synaptic_conductance @ self.Kt

            dvoltage = self.TUNIT/self.C * ( \
                - self.GL * (self.voltage - self.VL) \
                - exc_input * (self.voltage - self.VE) \
                - inh_input * (self.voltage - self.VI)
            )
            if t == self.num_steps - 1:
                exc_current = exc_input * np.abs(self.voltage - self.VE)
                inh_current = inh_input * np.abs(self.voltage - self.VI)
                self.logger.write("Last euler step with GL: {:.4e}, stimulus: {:.4e}, post_cond @ Kt: {:.4e}, voltage: {:.4e}, dvoltage: {:.4e}, inh/exc current ratio:".
                    format(self.GL, np.abs(stimulus).mean(), np.abs(self.exc_peripheral_synaptic_conductance @ self.Kt).mean(), \
                        np.abs(self.voltage).mean(), np.abs(dvoltage).mean()), np.mean(inh_current)/(np.mean(exc_current) + 1e-8) )
       
        except Exception as e:
            # TODO:
            # - print the correct values
            print("Error at {}th euler step".format(t), e)
            print("GL: {:.4e}, stimulus: {:.4e}, post_cond @ Kt: {:.4e}, voltage: {:.4e}".
                format(self.GL, np.abs(stimulus).mean(), np.abs(self.exc_peripheral_synaptic_conductance @ self.Kt).mean(), \
                    np.abs(self.voltage).mean()))
        finally:
            self.voltage += dvoltage
    
    def evolve(self, acts):
        # If firing rate is too high, increase the threshold VT
        dVT = ((acts > 0).mean() - self.TARGET_RATE)
        self.VT += dVT * self.lr_VT
        self.VT = max(self.VT, self.VR) 
        self.VT = min(self.VT, self.VE)

# firing rate calculated using l1 norm instead of l0
# (avg firings per neuron per simulation vs percent of neurons fired of simulation period)
@LIFFactory.register("kglight1")
class KGLIF1(KGLIF):
    def __init__(self, config):
        super().__init__(config)
        self.TARGET_L1 = config.target_l1
    
    def evolve(self, acts):
        dVT = (acts.mean() - self.TARGET_L1)
        self.VT += dVT * self.lr_VT
        self.VT = max(self.VT, self.VR) 
        self.VT = min(self.VT, self.VE)