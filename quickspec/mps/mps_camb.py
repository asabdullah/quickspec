import numpy as np

import scipy.interpolate

import quickspec as qs
import mps
import ipdb

zvec_high = np.linspace(2048, 11, 20)
zvec_low = np.linspace(10, 0, 60)
zvec_default = np.concatenate((zvec_high, zvec_low))

class mps_lin_camb(mps.mps_lin):
    '''
    Wrapper class to calculate the matter power spectrum using CAMB.

    INPUTS:
        cosmo [object]     Defines the cosmology.  Object of class quickspec.cosmo.lcdm
        sips [object]      Defines the initial power spectrum.  Object of quickspec.initial_ps
    '''

    def __init__(self, cosmo, sips, kmax=200., npoints=350, nonlinear=True,
                 zvec=zvec_default):
        try:
            import camb
        except:
            print "camb (http://camb.info/) could not be loaded. is it installed?"
            raise

        self.cosmo = cosmo
        # Define the model parameters
        par           = camb.model.CAMBparams()
        par.H0        = cosmo.H0
        par.omegab    = cosmo.omb
        par.omegac    = cosmo.omc
        par.omegav    = cosmo.oml
        par.NonLinear = nonlinear

        par.InitPower.set_params(As=sips.amp,
                                 ns=sips.n_s,
                                 nrun=sips.n_r,
                                 pivot_scalar=sips.k_pivot)

        par.set_matter_power(redshifts=zvec, kmax=kmax)

        self.par = par

        # Calculate the matter power spectrum
        camb_data = camb.get_results(par)
        p_kz = camb_data.get_matter_power_spectrum(minkh=1e-6, maxkh=kmax, npoints=npoints)
        kh, z, pk = camb_data.get_matter_power_spectrum(minkh=1e-6, maxkh=kmax, npoints=npoints)
        self.arr_z = z
        self.arr_k = kh * (self.cosmo.h)
        self.mat_p = pk / (self.cosmo.h)**3
        for iz, z in enumerate(self.arr_z):
            self.mat_p[iz, :] *= (1. + z)**2

        self.spl_p = scipy.interpolate.RectBivariateSpline(self.arr_z, np.log(self.arr_k), self.mat_p, kx=3, ky=3, s=0)

        self.kmin = self.arr_k[+0] * 0.999
        self.kmax = self.arr_k[-1] * 1.001

        self.zmin = np.min(zvec)
        self.zmax = np.max(zvec)

    def p_kz(self, k, z):
        """ returns the amplitude of the matter power spectrum at wavenumber k (in Mpc^{-1}) and conformal distance x (in Mpc). """

        assert(np.all(k >= self.kmin))
        assert(np.all(k <= self.kmax))
        assert(np.all(z >= self.zmin))
        assert(np.all(z <= self.zmax))

        k, z, s = qs.util.pair(k, z)
        ret = self.spl_p.ev(z, np.log(k))
        ret /= (1. + z)**2
        return ret.reshape(s)
