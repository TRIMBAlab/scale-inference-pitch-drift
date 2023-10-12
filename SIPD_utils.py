'''

@Jon Cannon: jonathan.j.cannon@gmail.com
'''
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from scipy.stats import norm

def make_scale(means: np.ndarray, weights: np.ndarray, variances: np.ndarray):
    gmix = GaussianMixture(n_components=len(means), covariance_type='full')
    dummy_data = np.array([1,2,3,4,5])
    gmix.fit(dummy_data.reshape(-1, 1))  # Now it thinks it is trained
    gmix.weights_ = weights   # mixture weights (n_components,) 
    gmix.means_ = means          # mixture means (n_components,) 
    gmix.covariances_ = variances  # mixture cov (n_components,)
    return gmix

def get_pdf(scale: GaussianMixture, xlist):
    pdf = np.zeros(np.shape(xlist))
    for i in range(len(scale.means_)):
        #print(np.shape(scale.means_[i]))
        print(np.shape(scale.weights_[i]*norm.pdf(xlist, scale.means_[i], np.sqrt(scale.covariances_[i,0]))))
        pdf = pdf + scale.weights_[i]*norm.pdf(xlist, scale.means_[i], np.sqrt(scale.covariances_[i,0]))
    print(np.shape(pdf))
    return pdf

def gen_drift(noise: float, trend: float, length: int):
    drift = np.zeros(length)
    for i in range(1,length):
        drift[i] = drift[i-1] + norm.rvs(scale=noise, loc=trend)
    return drift

def gen_f0(scale: GaussianMixture, drift: np.ndarray):
    ''' Generates artificial data with a given scale and drift trajectory'''
    [pitches, labels] = scale.sample(len(drift))
    print(np.shape(pitches))
    #print(drift)
    np.random.shuffle(pitches)
    print("sum: ",np.shape(pitches+drift.reshape(-1, 1)))
    return pitches+drift.reshape(-1, 1)

@dataclass(init=True, repr=True)
class SIPDParams:
    ''' Configuration for SIPD algorithm - parameters'''

    n_reps: int = 7         # number of algorithm reps
    n_peaks: int = 7         # number of scale degrees
    drift_rate: float = 0.001 # expected drift rate
    df0 : float = 0.001      # f0 increment used for derivatives

class SIPD:
    ''' Base class for SIPD inference problems '''

    def __init__(self, params: SIPDParams):
        self.params = params
        self.all_f0 = []
        self.all_scales = []
        self.all_drift = []
        self.all_LLscores = []

    def infer_scale(self, f0: np.ndarray):
        ''' Takes a (de-drifted) f0 timeseries and fits distribution with a set of n_peaks Gaussian peaks '''
        print(np.shape(f0))
        scale = GaussianMixture(n_components=self.params.n_peaks, random_state = 0).fit(f0)
        
        return scale

    def infer_drift(self, f0: np.ndarray, scale: GaussianMixture):
        ''' Takes a f0 timeseries and a set of peaks and infers the drift timeseries '''
        drift_0 = self.infer_drift_fwd(f0, scale, 0)
        f0_backwards = np.flip(f0)
        drift_backwards = self.infer_drift_fwd(f0_backwards, scale, drift_0[-1])
        return np.flip(drift_backwards).reshape(-1,1)
    
    def infer_drift_fwd(self, f0: np.ndarray, scale: GaussianMixture, drift0: np.ndarray):
        ''' Takes a f0 timeseries, a set of peaks, and an initial drift and infers the drift timeseries in a single pass '''
        drift = np.zeros(len(f0))
        drift[0] = drift0
        
        for i in range(1, len(f0)):
            
            dedrifted_sample = f0[i] - drift[i-1]
            LL_up = scale.score_samples((dedrifted_sample + self.params.df0).reshape(1, -1))
            LL_down = scale.score_samples((dedrifted_sample - self.params.df0).reshape(1, -1))
            LL_deriv = (LL_up - LL_down)/(2*self.params.df0)
            drift[i] = drift[i-1] - LL_deriv * self.params.drift_rate
        return drift
    
    def run(self, f0: np.ndarray):
        
        drift_now = np.zeros(np.shape(f0))
        print("drift0: ", np.shape(drift_now))
        print("f0: ", np.shape(f0))
        self.all_drift.append(drift_now)
        scale_now = self.infer_scale(f0-drift_now+drift_now[0])
        self.all_scales.append(scale_now)
        self.all_LLscores.append(scale_now.score(f0-drift_now+drift_now[0]))
        
        for i in range(self.params.n_reps):
            drift_now = self.infer_drift(f0, scale_now)
            self.all_drift.append(drift_now)
            self.all_f0.append(f0-drift_now+drift_now[0])
            scale_now = self.infer_scale( f0-drift_now+drift_now[0])
            
            self.all_scales.append(scale_now)
            self.all_LLscores.append(scale_now.score(f0-drift_now+drift_now[0]))
        
        
        
