import numpy as np
import scipy.integrate

# ===

def w_k_tophat(k):
    """ return the Fourier Transform of a tophat window function. """
    return 3./k**3*(np.sin(k) - k*np.cos(k))

def dw_k_tophat(k):
    """ return the derivative w.r.t. k of the fourier transform of a tophat window function. """
    return -9./k**4*(np.sin(k) - k*np.cos(k)) + 3./k**2 * np.sin(k)

# ===

class mps():
    """ base class for a matter power spectrum. """
    def __init__(self):
        pass

    def p_kx(self, k, x):
        """ returns the amplitude of the matter power spectrum at wavenumber k (in Mpc^{-1}) and conformal distance x (in Mpc). """
        return self.p_kz(k, self.cosmo.z_x(x))

    def sigma_rz(self, r, z, nk=10000 ):
        """ returns the variance in a smooth density field at redshift z on scale r (in Mpc).
                reference: Cooray and Sheth (2002, arxiv:0206508), Eq. 18
                     \sigma(r) = \sqrt{ \int dk k^2 P(k) / (2*pi^2) |W(k*r)|^2 }

                     W is a tophat with radius r.
                     Use dlnk for integration, nk parameter controls number of points.
        """
        def integrand(lnk):
            k  = np.exp(lnk); kr = k*r
            return (np.sin(kr) - kr*np.cos(kr))**2 / kr**3 * self.p_kz(k, z)

        dlnk = (np.log(self.kmax) - np.log(self.kmin)) / nk
        lnks = np.arange(0, nk) * dlnk + np.log(self.kmin)
        sigma2 = scipy.integrate.simps( integrand(lnks), dx=dlnk )
        sigma2 *= 9. / (r**3 * 2.*np.pi**2)

        return np.sqrt(sigma2)

    def dsigma2_rz(self, r, z, nk=10000):
        """ returns the derivative of sigma_rz with respect to r. """

        def integrand(lnk):
            k = np.exp(lnk); kr = k*r
            sinkr = np.sin(kr); kr_coskr = kr * np.cos(kr)
            return(-9.*(sinkr - kr_coskr)/kr**3 + 3./kr * sinkr) * (sinkr - kr_coskr) * self.p_kz(k, z)

        dlnk = (np.log(self.kmax) - np.log(self.kmin)) / nk
        lnks = np.arange(0, nk) * dlnk + np.log(self.kmin)
        dsigma2 = scipy.integrate.simps( integrand(lnks), dx=dlnk )
        dsigma2 *= 3. / (r**4 * np.pi**2)

        return dsigma2
