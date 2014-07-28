# Fitting formulae for linear matter power spectra in CDM + Baryon cosmologies, from Bardeen, Bond, Kaiser, Szalay (1986)

import numpy as np

import scipy.interpolate

import quickspec as qs
import mps

class mps_lin_bbks(mps.mps_lin):
    def __init__(self, cosmo, n=1., sigma8=0.84):
        """ Linear matter power spectrum given by fitting function of Bardeen, Bond, Kaiser, Szalay (1986) """
        
        self.cosmo  = cosmo
        self.n      = n

        self.kmin   = 1.e-10
        self.kmax   = 1.e+10

        self.gamma  = cosmo.omm * cosmo.h * np.exp(-cosmo.omb*(1.+np.sqrt(2.*cosmo.h)/cosmo.omm))

        self.arr_z  = np.array([0.0, 0.5, 1., 1.5, 2., 3., 4., 6., 8., 12., 16., 32., 64., 128., 256., 512., 1300.])
        self.arr_h  = np.array([self.cosmo.G_z(z)**2 * (1.+z)**2 for z in self.arr_z])
        self.spl_h  = scipy.interpolate.UnivariateSpline( self.arr_z, self.arr_h, k=3, s=0 )

        self.norm   = 1.0
        tsigma8     = self.sigma_rz(8./cosmo.h, 0.0)
        self.norm   = sigma8 / tsigma8

    def p_kz(self, k, z):
        q = k*self.cosmo.h / self.gamma
        bbks = np.log(1.+2.34*q) / (2.24*q) * (1. + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-0.25)

        return k**self.n * bbks**2 * self.norm**2 * self.spl_h(z) / (1.+z)**2
