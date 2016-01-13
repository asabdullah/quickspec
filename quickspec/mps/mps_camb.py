import numpy as np

import scipy.interpolate

import quickspec as qs
import mps
import ipdb

class mps_lin_camb(mps.mps_lin):
    '''
    Wrapper class to calculate the matter power spectrum using CAMB.
    
    INPUTS:
        cosmo [object]     Defines the cosmology.  Object of class quickspec.cosmo.lcdm
        sips [object]      Defines the initial power spectrum.  Object of quickspec.initial_ps
    '''
    def __init__(self, cosmo, sips, kmax=200., npoints=250, nonlinear=False,
                 zvec = np.array([2048., 1024., 512., 256., 128., 64.,
                                  32., 16., 12., 8., 6., 4., 3., 2.,
                                  1.5, 1., 0.5, 0.])):
        try:
            import camb
        except:
            print "camb (http://camb.info/) could not be loaded. is it installed?"
            raise

        self.cosmo     = cosmo

        # Define the model parameters
        par = camb.model.CAMBparams()
        par.H0                = cosmo.H0
        par.omegab            = cosmo.omb
        par.omegac            = cosmo.omc
        par.omegav            = cosmo.oml
        par.NonLinear         = nonlinear

        par.InitPower.set_params(As               = sips.amp,
                                 ns               = sips.n_s,
                                 nrun             = sips.n_r,
                                 pivot_scalar     = sips.k_pivot)

        par.set_matter_power(redshifts=zvec, kmax=kmax, k_per_logint=100)

        self.par              = par

        # Calculate the matter power spectrum
        camb_data = camb.get_results(par)
        p_kz = camb_data.get_matter_power_spectrum(maxkh=kmax, npoints=npoints)

        self.arr_z = zvec[::-1]
        self.arr_k = p_kz[0][:] * (self.cosmo.h)
        self.mat_p = (p_kz[2][:,::-1].transpose()) / (self.cosmo.h)**3
        for iz, z in enumerate(self.arr_z):
            self.mat_p[:,iz] *= (1.+z)**2

        self.spl_p = scipy.interpolate.RectBivariateSpline(np.log(self.arr_k), self.arr_z, self.mat_p, kx=3, ky=3, s=0)

        self.kmin = self.arr_k[+0]*0.999
        self.kmax = self.arr_k[-1]*1.001

        self.zmin = np.min(zvec)
        self.zmax = np.max(zvec)

    def p_kz(self, k, z):
        """ returns the amplitude of the matter power spectrum at wavenumber k (in Mpc^{-1}) and conformal distance x (in Mpc). """

        assert( np.all( k >= self.kmin ) )
        assert( np.all( k <= self.kmax ) )
        assert( np.all( z >= self.zmin ) )
        assert( np.all( z <= self.zmax ) )

        k, z, s = qs.util.pair(k, z)
        ret  = self.spl_p.ev( np.log(k), z)
        ret /= (1.+z)**2
        return ret.reshape(s)
