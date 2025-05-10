
import attr
from tqdm import tqdm
from typing import Union

import numpy as np
from scipy.integrate import simpson

@attr.s(auto_attribs=True)
class StrongFieldIonizer : 
    """ --- Constructor --- """

    # Input parameters
    t_end: float = None
    t_start: float = None
    N_time: int = None

    wl: float = 0.057
    epsilon: float = 1
    intensity: float = 10e13 / 3.51e16
    phi: float = 0
    Nc: int = 1
    E0: float = -0.5

    # Computed parameters
    @property
    def t_axis(self) : return np.linspace(self.t_start, self.t_end, self.N_time)

    @property
    def T(self): return self.Nc * 2 * np.pi / self.wl

    @property
    def F0(self): return np.sqrt(self.intensity)

    @property
    def A0(self): return self.F0 / self.wl

    @property
    def Up(self): return 0.25 * self.A0**2

    """ --- Checker methods for error --- """

    def check_type_error(self, input_param, input_name: str) : 
        if not isinstance(input_param, (float, np.ndarray)) : 
            raise TypeError(f"Expected input <{input_name}> to be float or np.ndarray, got {type(input_param)}")

    def check_time_parameters(self):
        for param_name in ['t_start', 't_end', 'N_time']:
            if getattr(self, param_name) is None:
                raise ValueError(f"Parameter '{param_name}' must be set (not None).")

    """ --- Main methods --- """

    # Trivial time evolution phase 
    def time_evolution(self, tp: Union[float, np.ndarray], t0: float) : 
        self.check_type_error(tp, 'tp')

        return np.exp( -1j*self.E0*(tp - t0) ) 

    # Hydrogen 1s ground state (Fourier transformed)
    def ground_state_fourier(self, k: Union[float, np.ndarray]) : 
        return 2**(5/2)/(2*np.pi) * 1/( np.dot(k,k) +1)**2

    # Laserpulse envelope
    def get_envelope(self, t: Union[float, np.ndarray]) : 
        self.check_type_error(t, 't')
        
        return np.sin(np.pi*t/self.T)**2
    
    # Pulse vector potential
    def A(self, t: Union[float, np.ndarray]) : 
        self.check_type_error(t, 't')

        prefac = self.A0 * self.get_envelope(t) / ( np.sqrt( 1 + self.epsilon**2) )

        vec = np.stack([
                        np.zeros_like(t), 
                        self.epsilon*np.sin(self.wl * t + self.phi),
                        np.cos(self.wl * t + self.phi)
        ], axis=0)
        # stacks horizontally: [ [A_xs(t)], [A_ys(t)], [A_zs(t)] ]
        # returns: (3, len(t))
        return prefac * vec

    # Precompute alpha and beta for single or multiple times
    def get_alphas_and_betas(self, tps: Union[float, np.ndarray], t0: float, N: int) : 
        self.check_type_error(tps, 'tps')

        alpha_list = np.zeros(shape=(len(tps),3,))  # rows of [ A_x(t1), A_y(t1), A_z(t1) ]
        beta_list = np.zeros_like(tps)              # 1D row vector

        for i, tp in enumerate(tps) : 
            tgrid = np.linspace(t0, tp, N)
            A_of_ts = self.A(tgrid)
            # sum along horizontal to get: A_x(t_i)**2 + A_y(t_i)**2 + A_z(t_i)**2
            A2_of_ts = np.sum(A_of_ts**2, axis=0)  # shape: (len(tps),)

            alpha = simpson(y=A_of_ts, x=tgrid)
            beta = simpson(y=A2_of_ts, x=tgrid)
            
            alpha_list[i] = alpha
            beta_list[i] = beta

        # alpha_list: (len(tps),3)
        # beta_list: (len(tps),)
        return (alpha_list, beta_list)

    def calculate_matrix_elements(self, k_range: np.array, N_ks: int) : 
        self.check_time_parameters()

        # Precompute alpha and beta
        alphas, betas = self.get_alphas_and_betas(tps=self.t_axis, t0=self.t_start, N=self.N_time)

        A_tgrid = self.A(self.t_axis) # each column is Ax(t1), Ay(t1), Az(t1)

        # k-vectors
        k_ys = np.linspace(k_range[0], k_range[1], N_ks)
        k_zs = np.linspace(k_range[0], k_range[1], N_ks)

        result = np.zeros((N_ks, N_ks))

        total = N_ks ** 2
        pbar = tqdm(total=total, desc="Building matrix...", unit=' elements')

        for i, k_y in enumerate(k_ys) : 
            for j, k_z in enumerate(k_zs) : 
                k_vector = np.array([0.0, k_y, k_z])
                k2 = np.dot(k_vector, k_vector)
                
                prod = np.dot(A_tgrid.transpose(), k_vector)
                
                state = np.exp( -1j*0.5*k2*(self.t_axis - self.t_start) 
                            + 1j*np.dot(alphas, k_vector) 
                            + 1j*0.5*betas ) * self.time_evolution(self.t_axis, self.t_start)

                integrand = prod * state

                # Do Simpson integration
                integral = simpson(y=integrand, x=self.t_axis)

                res = -1j * self.ground_state_fourier(k_vector) * integral

                result[i,j] = np.abs(res)**2

                pbar.update(1)
        
        return A_tgrid, k_ys, k_zs, result     

    
