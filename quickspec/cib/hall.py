# Implementation of the single-spectra-energy-distribution (SSED) model of the  from Hall et. al. 2010 (arxiv:0912.4315)

import numpy as np

import quickspec as qs

alpha_mid_ir = -2.
nu_mid_ir    = 4954611330474.7109

def ssed_graybody(nu, Td=34., beta=2):
    return nu**(beta) * qs.units.planck(nu, Td)

def ssed(nu, Td=34., beta=2., alpha_mid_ir=alpha_mid_ir, nu_mid_ir=nu_mid_ir):
    """ Calculation of the SSED f_{\nu} defined between pages 4 and 5 of Hall et. al. """
    if np.isscalar(nu):
        if nu > nu_mid_ir:
            return ( ssed_graybody(nu_mid_ir, Td, beta) / (nu_mid_ir)**(alpha_mid_ir) * nu**(alpha_mid_ir) )
        else:
            return ssed_graybody(nu, Td, beta)
    else:
        ret = ssed_graybody(nu, Td, beta)
        ret[ np.where( nu > nu_mid_ir ) ] = (ssed_graybody(nu_mid_ir, Td, beta) / (nu_mid_ir)**(alpha_mid_ir) * nu**(alpha_mid_ir) )[ np.where( nu > nu_mid_ir ) ]
        return ret

def jbar(nu, z, x, zc=2., sigmaz=2., norm=7.5374829969423142e-15, ssed_kwargs={}):
    """ Eq. 10 of Hall et. al. nu in Hz, returns in Jansky. """
    # a * \chi^2 * exp( - (z-zc)/2/\sigma_z^2) f_{nu (1+z)}
    return 1./(1.+z) * x**2 * np.exp( -(z-zc)**2/(2.*sigmaz**2) ) * ssed( nu*(1+z), **ssed_kwargs ) * norm

class ssed_kern():
    def __init__(self, nu, b=1.0, jbar_kwargs={}, ssed_kwargs={}):
        self.nu = nu
        self.b  = b
        self.jbar_kwargs = jbar_kwargs
        self.ssed_kwargs = ssed_kwargs

    def w_lxz(self, l, x, z):
        # Ref: Hall et. al. Eq. 5. Cl = int dz 1/H(z)/chi(z)^2 (ab j(\nu, z))^2 P_lin(l/chi,z)
        return 1./(1.+z) * self.b * jbar(self.nu, z, x, ssed_kwargs=self.ssed_kwargs, **self.jbar_kwargs)
