import numpy as np
import pylab as pl

# constants
h    = 6.626068e-34   # J.s planck constant
c    = 299792458.     # m/s speed of light
k    = 1.3806504e-23  # m^2 kg s^{-2} K^{-1} boltzmann constant

tcmb = 2.7255         # K cmb temperature (Fixsen et. al. 2009)

def planck(nu,T):
    """ returns the planck blackbody function (in W sr^{-1} Hz^{-1})
    at frequency \nu (in Hz) for a blackbody with temperature T (in K). """
    return 2*h*nu**3 / c**2 / (np.exp(h*nu/k/T) - 1.)

def dplanck_dt(nu,T):
    """ returns the derivative planck(nu,T) w.r.t. frequency (in W sr^{-1} Hz^{-2}). """
    return 2*h*nu**3 / c**2 / (np.exp(h*nu/k/T) - 1.)**2 * h*nu/k/T**2 * np.exp(h*nu/k/T)

def j2k(nu):
    """ returns the conversion factor between Jansky units and CMB Kelvin. """
    x = h*nu/(k*tcmb)
    g = (np.exp(x) - 1.)**2 / x**2 / np.exp(x)
    return c**2 / (2. * nu**2 * k) * g * 1.e-26

def k2j(nu):
    """ returns the conversion factor between CMB Kelvin and Jansky units. """
    return 1.0 / j2k(nu)
