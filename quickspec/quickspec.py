import numpy as np

class initial_ps(object):
    """ primordial power spectrum, characterized by
        ln P = ln(amp) + (n_s -1)*ln(k/k_pivot) + n_r/2 * ln(k/k_pivot)^2
        """
    def __init__(self, amp=2.1e-9, n_s=0.95, n_r=0.0, k_pivot=0.05):

        self.amp     = amp
        self.n_s     = n_s
        self.n_r     = n_r
        self.k_pivot = k_pivot

    def p(self, k):
        return self.amp * np.exp( (self.n_s-1.) * np.log(k/self.k_pivot) + 0.5*self.n_r * np.log(k/self.k_pivot)**2 )
