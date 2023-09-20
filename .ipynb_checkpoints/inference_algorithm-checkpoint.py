'''

@Jon Cannon: jonathan.j.cannon@gmail.com
'''
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

def make_scale(means: np.ndarray, weights: np.ndarray, variances: np.ndarray):
    gmix = mixture.GaussianMixture(n_components=len(means), covariance_type='full')
    gmix.fit(rand(10, 1))  # Now it thinks it is trained
    gmix.weights_ = weights   # mixture weights (n_components,) 
    gmix.means_ = means          # mixture means (n_components,) 
    gmix.covariances_ = variances  # mixture cov (n_components,)
    return gmix

def gen_drift(noise: float, trend: float, length: int):
    drift = np.zeros(length)
    for i in range(1,length):
        drift[i] = drift[i-1] + norm.rvs(scale=noise, loc=trend)
    return drift

def gen_f0(scale: GaussianMixture, drift: np.ndarray):
    ''' Generates artificial data with a given scale and drift trajectory'''
    pitches = scale.sample(len(drift))
    return pitches+drift

@dataclass(init=True, repr=True)
class SIPDParams:
    ''' Configuration for SIPD algorithm - parameters'''

    n_reps: int = 10         # number of algorithm reps
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
        scale = GaussianMixture(n_components=self.params.n_peaks, random_state = 0).fit(f0)
        
        return scale

    def infer_drift(self, f0: np.ndarray, scale: GaussianMixture):
        ''' Takes a f0 timeseries and a set of peaks and infers the drift timeseries '''
        drift_0 = infer_drift_fwd(self, f0, scale, 0)
        f0_backwards = np.flip(f0)
        drift_backwards = infer_drift_fwd(self, f0_backwards, scale, drift_0[-1])
        return np.flip(drift_backwards)
    
    def infer_drift_fwd(self, f0: np.ndarray, scale: GaussianMixture, drift0: np.ndarray):
        ''' Takes a f0 timeseries, a set of peaks, and an initial drift and infers the drift timeseries in a single pass '''
        drift = np.zeros(len(f0))
        drift[0] = drift0
        for i in range(1, len(f0)):
            dedrifted_sample = f0[i] - drift[i-1]
            LL_up = scale.score(dedrifted_sample + self.params.df0)
            LL_down = scale.score(dedrifted_sample - self.params.df0)
            LL_deriv = (LL_up - LL_down)/(2*self.params.df0)
            drift[i] = drift[i-1] + LL_deriv * self.params.drift_rate
        return drift
    
    def run(self, f0: np.ndarray):
        
        drift_now = np.zeros(len(f0))
        self.all_drift.append(drift_now)
        scale_now = infer_scale(self, f0-drift_now+drift_now[0])
        self.all_scales.append(scale_now)
        self.all_LLscore.append(scale_now.score(f0-drift_now+drift_now[0]))
        
        for i in range(self.params.n_reps):
            drift_now = infer_drift(self, f0, scale_now)
            self.all_drift.append(drift_now)
            scale_now = infer_scale(self, f0-drift_now+drift_now[0])
            self.all_scales.append(scale_now)
            self.all_LLscore.append(scale_now.score(f0-drift_now+drift_now[0]))
        
        


class mPIPPET(PIPPET):
    ''' PIPPET with multiple event streams '''

    def step(self, t_i: float, mu_prev: float, V_prev: float) -> tuple[float, float]:
        ''' Posterior update for a time step '''

        # Internal phase noise
        noise = np.sqrt(self.params.dt) * self.params.eta_mu * np.random.randn()

        # Sum dmu across event streams
        dmu_sum = 0
        for s_i in range(self.n_streams):
            dmu = self.streams[s_i].lambda_hat(mu_prev, V_prev)
            dmu *= (self.streams[s_i].mu_hat(mu_prev, V_prev) - mu_prev)
            dmu_sum += dmu
        mu = mu_prev + self.params.dt*(1 - dmu_sum) + noise

        # Sum dV across event streams
        dV_sum = 0
        for s_i in range(self.n_streams):
            dV = self.streams[s_i].lambda_hat(mu_prev, V_prev)
            dV *= (self.streams[s_i].V_hat(mu, mu_prev, V_prev) - V_prev)
            dV_sum += dV
        V = V_prev + self.params.dt*(self.params.sigma_phi**2 - dV_sum)

        # Update posterior based on events in any stream
        t_prev, t = self.ts[t_i-1], self.ts[t_i]
        for s_i in range(self.n_streams):
            if self.is_onset(t_prev, t, s_i):
                mu_new = self.streams[s_i].mu_hat(mu, V)
                V = self.streams[s_i].V_hat(mu_new, mu, V)
                mu = mu_new
                self.event_n[s_i] += 1
                self.idx_event.add(t_i)
                self.event_stream[t_i].add(s_i)

                self.surp[t_i, s_i, 0] = -np.log(self.streams[s_i].lambda_hat(mu_prev, V_prev)*self.params.dt)
                self.surp[t_i, s_i, 1] = -np.log(self.streams[s_i].lambda_hat(mu, V)*self.params.dt)
                self.grad[t_i, s_i] =  -np.log(self.streams[s_i].lambda_hat(mu_prev+.01, V_prev)*self.params.dt)
                self.grad[t_i, s_i] +=  np.log(self.streams[s_i].lambda_hat(mu_prev-.01, V_prev)*self.params.dt)
                self.grad[t_i, s_i] /= .02
            else:
                self.surp[t_i, s_i, 0] = -np.log(1-self.streams[s_i].lambda_hat(mu_prev, V_prev)*self.params.dt)
                self.surp[t_i, s_i, 1] = -np.log(1-self.streams[s_i].lambda_hat(mu, V)*self.params.dt)
                self.grad[t_i] =  -np.log(1-self.streams[s_i].lambda_hat(mu_prev+.01, V_prev)*self.params.dt)
                self.grad[t_i] +=  np.log(1-self.streams[s_i].lambda_hat(mu_prev-.01, V_prev)*self.params.dt)
                self.grad[t_i] /= .02

        return mu, V

    def run(self) -> None:
        ''' Step through entire stimulus, tracking sufficient statistics '''
        for i in range(1, self.n_ts):
            mu_prev = self.mu_s[i-1]
            V_prev = self.V_s[i-1]
            mu, V = self.step(i, mu_prev, V_prev)
            self.mu_s[i] = mu
            self.V_s[i] = V


class pPIPPET(PIPPET):
    ''' PIPPET with pattern (i.e. template) inference '''

    def __init__(self, params: PIPPETParams, prior: np.ndarray):
        super().__init__(params)

        # Track likelihoods and big Lambdas per pattern
        self.n_m = self.n_streams
        self.L_s = np.zeros(self.n_ts)
        self.L_ms = np.zeros((self.n_ts, self.n_m))
        self.p_m = np.zeros((self.n_ts, self.n_m))
        self.p_m[0] = prior
        self.p_m[0] = self.p_m[0]/self.p_m[0].sum()

        # Initialise big Lambdas using mu_0 and V_0
        for s_i, m in enumerate(self.streams):
            self.L_ms[0, s_i] = m.lambda_hat(self.mu_s[0], self.V_s[0])
        self.L_s[0] = np.sum(self.p_m[0] * self.L_ms[0])

    def step(self, s_i: int, mu_prev: float, V_prev: float, is_event: bool=False) -> tuple[float, float]:
        ''' Posterior step for a given pattern '''

        noise = np.sqrt(self.params.dt) * self.params.eta_mu * np.random.randn()

        dmu = self.streams[s_i].lambda_hat(mu_prev, V_prev)
        dmu *= (self.streams[s_i].mu_hat(mu_prev, V_prev) - mu_prev)
        mu = mu_prev + self.params.dt*(1 - dmu) + noise

        dV = self.streams[s_i].lambda_hat(mu_prev, V_prev)
        dV *= (self.streams[s_i].V_hat(mu, mu_prev, V_prev) - V_prev)
        V = V_prev + self.params.dt*(self.params.sigma_phi**2 - dV)

        if is_event:
            mu_new = self.streams[s_i].mu_hat(mu, V)
            V = self.streams[s_i].V_hat(mu_new, mu, V)
            mu = mu_new

        return mu, V

    def run(self) -> None:
        ''' Step through entire stimulus, for all patterns '''

        # For each time step
        for i in range(1, self.n_ts):
            lambda_prev = self.L_s[i-1]
            mu_prev = self.mu_s[i-1]
            V_prev = self.V_s[i-1]

            mu_ms = np.zeros(self.n_m)
            V_ms = np.zeros(self.n_m)

            t_prev, t = self.ts[i-1], self.ts[i]

            # For each pattern
            for s_i in range(self.n_m):
                lambda_m_prev = self.L_ms[i-1, s_i]
                prev_p_m = self.p_m[i-1, s_i]

                # Update p_m based on event observations (or absence of them)
                is_event = self.is_onset(t_prev, t, s_i)
                d_p_m = prev_p_m * (lambda_m_prev/lambda_prev - 1)
                if not is_event:
                    d_p_m *= -self.params.dt * lambda_prev
                self.p_m[i, s_i] = prev_p_m + d_p_m

                # Update posterior and lambda_m
                mu_m, V_m = self.step(s_i, mu_prev, V_prev, is_event)
                lambda_m = self.streams[s_i].lambda_hat(mu_m, V_m)

                self.L_ms[i, s_i] = lambda_m
                mu_ms[s_i] = mu_m
                V_ms[s_i] = V_m

                if is_event:
                    self.event_n[s_i] += 1
                    self.idx_event.add(i)
                    self.event_stream[i].add(s_i)

            # Marginalize across patterns
            self.mu_s[i] = np.sum(self.p_m[i] * mu_ms)
            self.L_s[i] = np.sum(self.p_m[i] * self.L_ms[i])
            self.V_s[i] = np.sum(self.p_m[i] * V_ms)
            self.V_s[i] += np.sum(self.p_m[i]*(1 - self.p_m[i])*np.power(mu_ms, 2))
            for m in range(self.n_m):
                for n in range(self.n_m):
                    if m != n:
                        self.V_s[i] -= self.p_m[i,m]*self.p_m[i,n]*mu_ms[m]*mu_ms[n]


class cPIPPET(PIPPET):
    ''' Oscillatory (wrapped) PIPPET '''

    def __init__(self, params: PIPPETParams):
        super().__init__(params)
        self.z_s = np.ones(self.n_ts, dtype=np.clongdouble)
        self.z_s[0] = np.exp(complex(-self.params.V_0/2, self.params.mu_0))

    def step(self, t_i: float, z_prev: complex, mu_prev: float, V_prev: float) -> complex:
        ''' Posterior update for a time step '''

        dz_sum = 0
        for s_i in range(self.n_streams):
            blambda = self.streams[s_i].zlambda(mu_prev, V_prev, self.params.tau)
            z_hat = self.streams[s_i].z_hat(mu_prev, V_prev, blambda, self.params.tau)
            if self.params.continuous_expectation:
                dz = blambda*(z_hat-z_prev)*self.params.dt
            else:
                dz = 0
            dz_sum += dz

        dz_par  =  -(self.params.sigma_phi**2)/2 * self.params.dt
        dz_perp = self.params.tau * self.params.dt
        z = z_prev * np.exp(1j*dz_perp + dz_par) - dz_sum

        #z = z_prev + z_prev*complex(-(self.params.sigma_phi**2)/2, self.params.tau)*self.params.dt - dz_sum
        z_norm = abs(z)
        if z_norm>1:
            z = z/z_norm * 0.9999
            print('o dang znorm '+str(z_norm))
        
        mu, V_s = PIPPETStream.z_mu_V(z)
        
        # Noise
        mu += np.sqrt(self.params.dt) * self.params.eta_mu * np.random.randn()
        V_s *= np.exp(np.sqrt(self.params.dt) * self.params.eta_V * np.random.randn())
        z = np.exp(complex(-V_s/2, mu))
        
        
        t_prev, t = self.ts[t_i-1], self.ts[t_i]
        for s_i in range(self.n_streams):
            if self.is_onset(t_prev, t, s_i):
                z = self.streams[s_i].z_hat(mu, V_s, self.streams[s_i].zlambda(mu, V_s, self.params.tau), self.params.tau)
                z_norm = abs(z)
                if z_norm>1:
                    z = z/z_norm * 0.9999
                    print('ono znorm '+str(z_norm))
                    print('Vprev ' +str(V_prev))
                    print('muprev ' +str(mu_prev))
                    print('zlam '+str(self.streams[s_i].zlambda(mu, V_s, self.params.tau)))
                self.event_n[s_i] += 1
                self.idx_event.add(t_i)
                self.event_stream[t_i].add(s_i)

                self.surp[t_i, s_i, 0] = -np.log(self.streams[s_i].lambda_hat(mu_prev, V_prev)*self.params.dt)
                self.surp[t_i, s_i, 1] = -np.log(self.streams[s_i].lambda_hat(mu, V_s)*self.params.dt)

                self.grad[t_i] =  -np.log(self.streams[s_i].zlambda(mu_prev+.01, V_prev, self.params.tau)*self.params.dt)
                self.grad[t_i] +=  np.log(self.streams[s_i].zlambda(mu_prev-.01, V_prev, self.params.tau)*self.params.dt)
                self.grad[t_i] /= .02
            else:
                self.surp[t_i, s_i, 0] = -np.log(1-self.streams[s_i].lambda_hat(mu_prev, V_prev)*self.params.dt)
                self.surp[t_i, s_i, 1] = -np.log(1-self.streams[s_i].lambda_hat(mu, V_s)*self.params.dt)
                self.grad[t_i] =  -np.log(1-self.streams[s_i].zlambda(mu_prev+.01, V_prev, self.params.tau)*self.params.dt)
                self.grad[t_i] +=  np.log(1-self.streams[s_i].zlambda(mu_prev-.01, V_prev, self.params.tau)*self.params.dt)
                self.grad[t_i] /= .02


        return z

    def run(self) -> None:
        ''' Step through entire stimulus, tracking sufficient statistics '''
        for i in range(1, self.n_ts):
            z_prev = self.z_s[i-1]
            mu_prev = self.mu_s[i-1]
            V_prev = self.V_s[i-1]
            z = self.step(i, z_prev, mu_prev, V_prev)
            mu, V = PIPPETStream.z_mu_V(z)
 
            self.mu_s[i], self.V_s[i] = mu, V
            self.z_s[i] = z

            
            
            

class movingPIPPET(cPIPPET):
    '''PIPPET with circular movement '''

    def __init__(self, params: PIPPETParams):
        super().__init__(params)
        self.alpha_s = np.ones(self.n_ts, dtype =np.clongdouble)
        self.alpha_s[0] = np.NaN
        self.tapping = False
        


    def step(self, t_i: float, z_prev: complex, mu_prev: float, V_prev: float, alpha_prev: float):
        ''' Posterior update for a time step '''
        alpha = alpha_prev * np.exp(1j * (self.params.tau )*self.params.dt)
        b = 1.3
        k = 2/b * np.tan(np.abs(z_prev)*b)
        alpha *= np.exp(1j * self.params.dt * self.params.movement_updating * k * np.sin(mu_prev-np.angle(alpha_prev)))
        alpha *= np.exp(1j * np.sqrt(self.params.dt) * self.params.eta_alpha* np.random.randn())
        
        if self.tapping:
            for phi in self.params.templates[0].e_means:
                
                
                
                # Add a tap event if it's that time
                if np.mod(np.angle(alpha_prev) - phi + np.pi, TWO_PI) < np.pi and np.mod(np.angle(alpha_prev) - phi + np.pi, TWO_PI) > np.pi/2 and np.mod(np.angle(alpha) - phi + np.pi, TWO_PI) > np.pi :
                    
                    if self.ts[t_i] - self.streams[0].e_times_p[-1] > .2:
                    
                        n_taps = self.streams[0].e_times_p.size
                        if self.params.verbose:
                            print('n_taps', n_taps)
                    
                        self.streams[0].e_times_p = np.insert(self.streams[0].e_times_p, n_taps, self.ts[t_i])
        
        z = super().step(t_i, z_prev, mu_prev, V_prev)
        
        if self.tapping:
            z = z + self.params.dt * self.params.movement_precision * (alpha_prev - z_prev)
            ##### DO THE MATH

            
        return z, alpha

    def run(self) -> None:
        ''' Step through entire stimulus, tracking sufficient statistics '''
        for i in range(1, self.n_ts):
            
            z_prev = self.z_s[i-1]
            mu_prev = self.mu_s[i-1]
            V_prev = self.V_s[i-1]
            alpha_prev = self.alpha_s[i-1]
            z, alpha = self.step(i, z_prev, mu_prev, V_prev, alpha_prev)
            mu, V = PIPPETStream.z_mu_V(z)
            # Noise
            
            if self.ts[i] > self.params.start_tapping and self.tapping==False:
                self.tapping=True
                alpha = z/np.abs(z)
                
            # Update
            self.mu_s[i], self.V_s[i] = mu, V
            self.z_s[i] = z
            self.alpha_s[i]=alpha



class gcPATIPPET(cPIPPET):
    ''' Oscillatory PIPPET Bank (gradient, circular)'''

    def __init__(self, params: PIPPETParams, taus: np.ndarray, prior: np.ndarray=None):
        super().__init__(params)
        self.taus = taus
        self.tau_centers = (taus[1:] + taus[:-1]) / 2
        self.dtau_list = np.diff(taus)
        self.n_bank = self.tau_centers.shape[0]
        # TODO
        self.L_s = np.zeros(self.n_ts)
        self.L_ms = np.zeros((self.n_ts, self.n_bank))
        self.p_m = np.zeros((self.n_ts, self.n_bank))
        if prior is None:
            prior = np.ones(self.n_bank)
        self.p_m[0] = prior
        self.p_m[0] = self.p_m[0]/(self.p_m[0] * self.dtau_list).sum()
        # Initialise big Lambdas using mu_0 and V_0
        for i, tau in enumerate(self.tau_centers):
            self.L_ms[0, i] = self.streams[0].zlambda(self.mu_s[0], self.V_s[0], 1)
        self.L_s[0] = np.sum(self.p_m[0] * self.L_ms[0])
        # TODO
        self.z_ms = np.ones((self.n_ts, self.n_bank), dtype=np.clongdouble)
        self.z_ms[0, :] = self.z_s[0]
        self.y_ms = np.ones((self.n_ts, self.n_bank), dtype=np.clongdouble)
        self.y_ms[0, :] = self.z_ms[0] * self.p_m[0] #* np.diff(self.taus)
        # TODO
        self.mu_ms = np.zeros((self.n_ts, self.n_bank))
        self.mu_ms[0] = self.mu_s[0]
        self.V_ms = np.zeros((self.n_ts, self.n_bank))
        self.V_ms[0] = self.V_s[0]
        self.integrated = np.zeros(self.n_ts, dtype=np.clongdouble)
        self.mu_avg = np.zeros(self.n_ts)
        self.V_avg = np.zeros(self.n_ts)
        self.tau_avg = np.zeros(self.n_ts)
        self.integrated[0] = np.sum(self.y_ms[0, :]*self.dtau_list.astype(complex))
        self.mu_avg[0] = np.angle(self.integrated[0])
        self.V_avg[0] = -2*np.log(np.abs(self.integrated[0]))
        self.tau_avg[0] = np.sum(self.p_m[0,:]*self.tau_centers*self.dtau_list)

    def step(self, t_i: float) -> complex:
        ''' Posterior update for a time step '''
        try:
            t_j = t_i - 1
            t_prev, t = self.ts[t_j], self.ts[t_i]
            is_event = self.is_onset(t_prev, t, 0)

            ys_prev = self.y_ms[t_j]
            ps_prev = self.p_m[t_j]
            lams_prev = self.L_ms[t_j]
            lam_prev = self.L_s[t_j]

            for i, tau in enumerate(self.tau_centers):
                y_prev = ys_prev[i]
                p_prev = ps_prev[i]
                dtau = self.taus[i+1] - self.taus[i]
                z_prev = y_prev/p_prev

                mu_prev, V_prev = PIPPETStream.z_mu_V(z_prev)

                p_flux_up = 0
                y_flux_up = 0
                if i != (self.n_bank-1):
                    p_flux_up = -(self.params.sigma_theta**2)/2
                    p_flux_up *= (ps_prev[i+1]-ps_prev[i])/(self.tau_centers[i+1]-self.tau_centers[i])
                    y_flux_up = -(self.params.sigma_theta**2)/2
                    y_flux_up *= (ys_prev[i+1]-ys_prev[i])/(self.tau_centers[i+1]-self.tau_centers[i])
                    #
                    p_flux_up += (ps_prev[i+1]+ps_prev[i])/2 * (self.params.tau_p - self.taus[i+1]) * self.params.tau_p_tendency
                    y_flux_up += (ys_prev[i+1]+ys_prev[i])/2 * (self.params.tau_p - self.taus[i+1]) * self.params.tau_p_tendency

                p_flux_down = 0
                y_flux_down = 0
                if i != 0:
                    p_flux_down =  -(self.params.sigma_theta**2)/2
                    p_flux_down *= (ps_prev[i]-ps_prev[i-1])/(self.tau_centers[i]-self.tau_centers[i-1])
                    y_flux_down =  -(self.params.sigma_theta**2)/2
                    y_flux_down *= (ys_prev[i]-ys_prev[i-1])/(self.tau_centers[i]-self.tau_centers[i-1])
                    #
                    p_flux_down += (ps_prev[i-1]+ps_prev[i])/2 * (self.params.tau_p - self.taus[i]) * self.params.tau_p_tendency
                    y_flux_down += (ys_prev[i-1]+ys_prev[i])/2 * (self.params.tau_p - self.taus[i]) * self.params.tau_p_tendency
                
                p_flux_out = self.params.sigma_theta_global * ps_prev[i]
                p_flux_in = self.params.sigma_theta_global * np.sum(ps_prev) / len(self.tau_centers)
                
                dp_c = (p_flux_down - p_flux_up + p_flux_in - p_flux_out)/dtau
                dy_c = (y_flux_down - y_flux_up  - ys_prev[i]/ps_prev[i] * p_flux_out)/dtau

                y_hat = p_prev * self.streams[0].z_hat(mu_prev, V_prev, lams_prev[i], 1) * lams_prev[i]/lam_prev
                p_hat = p_prev * lams_prev[i]/lam_prev

                y_hat = y_hat / np.maximum(1, abs(y_hat/p_hat)+.0001) ### numerical problem fixer

                # Coupling, p
                dp = self.params.dt * dp_c * self.params.tau_c_coef
                dp -= self.params.dt * lam_prev*(p_hat - p_prev)
                p = p_prev + dp
                

                # Coupling, y
                dy = self.params.dt * dy_c * self.params.tau_c_coef
                # Drift
                dy += self.params.dt * y_prev*complex(-(self.params.sigma_phi**2)/2, tau)
                
                
                # Expectation
                if self.params.continuous_expectation:
                    y = y_prev + dy - lam_prev*(y_hat-y_prev)*self.params.dt
                else:
                    y = y_prev + dy
                
                
                if is_event:
                    y = y_hat
                    p = p_hat
                
                p = np.maximum(p, 0.00001)
                y = y / np.maximum(1, abs(y/p)+.0001) ### numerical problem fixer

                mu, V = PIPPETStream.z_mu_V(y/p)
                if V<0:
                    print("V = ", V)
                    print(np.abs(y))
                    print(p)
                    print(np.abs(y/p))
                    
                self.mu_ms[t_i, i] = mu
                self.V_ms[t_i, i] = V
                self.L_ms[t_i, i] = self.streams[0].zlambda(mu, V, 1)
                self.y_ms[t_i, i] = y
                self.p_m[t_i, i] = p

            self.integrated[t_i] = np.sum(self.y_ms[t_i,:]*self.dtau_list.astype(complex))
            self.mu_avg[t_i] = np.angle(self.integrated[t_i])
            self.V_avg[t_i] = -2*np.log(np.abs(self.integrated[t_i]))
            self.tau_avg[t_i] = np.sum(self.p_m[t_i,:]*self.tau_centers*self.dtau_list)


            self.L_s[t_i] = np.sum(np.diff(self.taus) * self.p_m[t_i] * self.L_ms[t_i])

            if is_event:
                self.event_n[0] += 1
                self.idx_event.add(t_i)
                self.event_stream[t_i].add(0)
        except RuntimeWarning:
            print("y_prev =", y_prev, " p_prev =", p_prev, " zprev =", np.abs(y_prev/p_prev), " and ", self.streams[0].z_hat(mu_prev, V_prev, lams_prev[i], 1))
            breakpoint()

    def run(self) -> None:
        for i in range(1, self.n_ts):
            self.step(i)
            # TODO: noise!

            

class vcPATIPPET(cPIPPET):
    '''Variational circular phase and tempo inference'''
    
    
    def __init__(self, params: PIPPETParams):
        
        new_templates = []
        for p in params.templates:
            new_means = np.append(p.e_means,0)
            new_vars = np.append(p.e_vars,100)
            new_lambdas = np.append(p.e_lambdas, p.lambda_0)
            new_templates.append(TemplateParams(p.e_times, new_means, new_vars, new_lambdas, p.lambda_0, p.label))
        params.templates = new_templates
        
        super().__init__(params)
        
        self.V_thetas = np.zeros(self.n_ts)
        self.V_thetas[0] = self.params.V_theta_0

        self.V_zthetas = np.zeros(self.n_ts, dtype=np.clongdouble)
        self.V_zthetas[0] = self.params.S_0 * np.exp(1j*self.params.mu_0 - self.params.V_0/2)* self.params.V_theta_0
        

        self.S = np.ones(self.n_ts)
        self.S[0] = self.params.S_0 

        self.theta_bars = np.zeros((self.n_ts,))
        self.theta_bars[0] = self.params.theta_0
        
    
        self.M = np.arange(-40, 40+1, 1)
        self.N = np.arange(1, 40+1, 1)
        self.oM = np.ones(np.size(self.M))
        self.oN = np.ones(np.size(self.N))

    def step(self, t_i: float) -> complex:

        # Previous time step values
        z_prev = self.z_s[t_i-1]
        mu_prev = self.mu_s[t_i-1]
        V_prev = self.V_s[t_i-1]
        theta_bar_prev = self.theta_bars[t_i-1]
        S_prev = self.S[t_i-1]
        V_theta_prev = self.V_thetas[t_i-1]
        V_ztheta_prev = self.V_zthetas[t_i-1]

        # Event?
        t_prev, t = self.ts[t_i-1], self.ts[t_i]
        is_event = self.is_onset(t_prev, t, 0)
        dNt = int(is_event)

        #blambda = self.streams[0].zlambda_tempo(mu_prev, V_prev, V_theta_prev, S_prev)
        #z_hat = self.streams[0].z_hat_tempo(mu_prev, V_prev, V_theta_prev, S_prev, blambda)
        #theta_hat = self.streams[0].theta_hat(mu_prev, V_prev, V_theta_prev, S_prev, theta_bar_prev, blambda)

        dz_par  =  -(self.params.sigma_phi**2)/2 * self.params.dt
        dz_par += -S_prev * V_theta_prev * self.params.dt
        dz_perp = theta_bar_prev * self.params.dt
        z = z_prev * np.exp(1j*dz_perp + dz_par)
        if dz_par>0 and self.params.verbose:
            print('dz_par is positive ({} at {})'.format(dz_par, self.ts[t_i]))
        

        dtheta_bar =  self.params.tau_p_tendency*(self.params.tau_p - theta_bar_prev)*self.params.dt
        theta_bar = theta_bar_prev + dtheta_bar
        dV_theta = self.params.sigma_theta**2 / 2 * self.params.dt
        V_theta = V_theta_prev * np.exp(-2 * self.params.tau_p_tendency * self.params.dt) + dV_theta

        dV_ztheta_perp = theta_bar_prev * self.params.dt
        V_ztheta = V_ztheta_prev * np.exp(1j*dV_ztheta_perp  - self.params.tau_p_tendency * self.params.dt)
        V_ztheta += (V_theta_prev - S_prev**2 * V_theta_prev**2 - self.params.tau_p_tendency*(self.params.tau_p-theta_bar_prev))* z_prev*1j * self.params.dt
        
        S = (V_ztheta/(1j* z * V_theta)).real
        
        if abs(z)<0.0001:
            S = S_prev
            V_ztheta = V_ztheta_prev
            z = z_prev

        if is_event:
            if self.params.verbose:
                print('event')
                print(t)
            self.event_n[0] += 1
            self.idx_event.add(t_i)
            self.event_stream[t_i].add(0)
            
            mu = np.angle(z)
            V = -2*np.log(np.abs(z))
            
            if self.params.tempo_scaling:
                v = self.streams[0].params.e_vars * theta_bar_prev**2
                
            else:
                v = self.streams[0].params.e_vars
            
            
            C_0_0 = -(self.M**2) * ((V+v)/2).reshape(-1, 1) \
                   - 1j * self.M * (mu-self.streams[0].params.e_means).reshape(-1, 1) #\
                   
            C_0_ = (self.streams[0].params.e_lambdas/TWO_PI).reshape(-1, 1) * np.exp(C_0_0)
            
            
            blambda_i = np.sum(C_0_, 1).real
            blambda = np.sum(blambda_i)

            C_1_ = -(self.M**2) * (V/2) \
                   -(self.M+1)**2 * (v/2).reshape(-1, 1) \
                   - 1j * self.M * mu \
                   + 1j * (self.M+1) * self.streams[0].params.e_means.reshape(-1, 1)

            C_1_ = (self.streams[0].params.e_lambdas/TWO_PI).reshape(-1, 1) * np.exp(C_1_)
            
            
            z_hat = (1/blambda) * C_1_.sum()
            
            if abs(z_hat)>1:
                print('z_hat is too big ({} at {})'.format(abs(z_hat), self.ts[t_i]))
                z_hat = z_hat/abs(z_hat)
                
            
            theta_hat = theta_bar - (S * 1j*V_theta/blambda * (self.M * C_0_).sum()).real
            


            theta_bar_plus = theta_hat
            S_hat = max(0, (S  * -((self.M.reshape(1, -1)) * C_1_).sum()/blambda).real)
            
            
            
            V_theta_hat = V_theta - (theta_bar - theta_bar_plus)**2
            
            V_theta_hat += -(S**2*V_theta**2/blambda) * (self.M**2 * C_0_).sum()
            
            V_theta_hat = max(V_theta_hat, 0)
            
            theta_bar = theta_bar_plus
            z = z_hat
            #assert abs(z_hat)<1, 'z_hat is too big ({} at {})'.format(abs(z_hat), self.ts[t_i])
            
            V_theta = np.maximum(0, V_theta_hat.real)
            
            V_ztheta_hat =  S_hat * 1j * V_theta_hat * z_hat
            V_ztheta = V_ztheta_hat
            S = S_hat
            
        self.z_s[t_i] = z
        #assert abs(z)<1, 'z is too big ({} at {})'.format(abs(z_hat), self.ts[t_i])
        self.theta_bars[t_i] = theta_bar
        self.S[t_i] = S
        self.V_zthetas[t_i] = V_ztheta
        self.V_thetas[t_i] = V_theta

    def run(self) -> None:
        for i in range(1, self.n_ts):
            self.step(i)
            mu, V = PIPPETStream.z_mu_V(self.z_s[i])
            # TODO: add noise
            pass
            # Update
            self.mu_s[i], self.V_s[i] = mu, V
            self.z_s[i] = np.exp(complex(-V/2, mu))
            

if __name__ == "__main__":
    import pdb
    print('Debugger on - press \'c\' to continue examples, \'q\' to quit')
