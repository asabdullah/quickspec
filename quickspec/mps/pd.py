import numpy as np

import scipy.optimize
import scipy.interpolate

import quickspec as qs
import mps

class mps_pd(mps.mps):
    def __init__(self, mps, kmin=2.e-5, kmax=2.e3, npts=10, cache=False):
        """ Non-linear matter power spectrum using Peacock and Dodds (1996) fitting formula: astro-ph/9603031 """
        
        self.mps   = mps
        self.cosmo = mps.cosmo

        self.kmin = kmin
        self.kmax = kmax

        self.cache = cache

        if self.cache == True:
            self.arr_z  = np.array([2048., 1024., 512., 256., 128., 64., 32., 16.,
                                    12., 8., 6., 4., 3., 2., 1.5, 1., 0.5, 0.])[::-1]
            self.arr_k  = np.logspace( np.log10(kmin), np.log10(kmax), (np.log10(kmax) - np.log10(kmin))*npts )
            self.mat_lnkl = np.zeros( (len(self.arr_k), len(self.arr_z)) )
            for ik, k in qs.util.enumerate_progress(self.arr_k, label="mps::pd::init"):
                for iz, z in enumerate(self.arr_z):
                    self.mat_lnkl[ik, iz] = np.log( scipy.optimize.bisect(lambda tk: ( self.knl(tk,z) - k), kmin*0.1, kmax*10.) )
            self.spl_lnkl = scipy.interpolate.RectBivariateSpline(np.log(self.arr_k), self.arr_z, self.mat_lnkl, kx=3, ky=3, s=0)

    def p_kz(self, k, z):
        k, z, s = qs.util.pair(k, z)

        if self.cache == True:
            kl = np.exp(self.spl_lnkl.ev(np.log(k), z))
        else:
            kl = np.array([ scipy.optimize.bisect(lambda ttk: ( self.knl(ttk,tz) - tk), self.kmin, self.kmax) for (tk, tz) in zip(k, z) ])

        d2 = self.d2_kl(kl, z)
        return (d2 / k**3 * (2.*np.pi**2)).reshape(s)

    def knl(self, kl, z):
        d2 = self.d2_kl(kl, z)

        return kl * (1. + self.d2_kl(kl, z))**(1./3.)

    def d2_kl(self, k, z):
        logks = np.log( 0.5 * k * np.array( [0.999, 1.0, 1.001] ) )
        logps = np.log( self.mps.p_kz(np.exp(logks), z) )

        tn = 1. + qs.util.deriv(logks, logps)[1]/3.

        A = 0.482*tn**(-0.947)
        B = 0.226*tn**(-1.778)
        a = 3.310*tn**(-0.244)
        b = 0.862*tn**(-0.287)
        V = 11.55*tn**(-0.423)

        g = self.g_a(1./(1.+z))
        x = self.mps.p_kz(k, z) * k**3 / (2.*np.pi**2)

        ret = x * ( (1.+B*b*x + (A*x)**(a*b)) / (1. + ((A*x)**a * g**3 / (V*np.sqrt(x)))**b) )**(1./b)
        return ret

    def g_a(self, a): # Peacock and Dodds Eq. 9, 15/16
        c  = self.cosmo

        f = a + c.omm*(1.-a) + c.oml*(a**3-a)

        omm = c.omm / f
        omv = c.oml * a**3 / f

        return 2.5 * omm / (omm**(4./7.) - omv + (1. + 0.5*omm)*(1.+ omv/70.))
