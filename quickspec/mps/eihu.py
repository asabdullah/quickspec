# Fitting Formulae for linear matter power spectra in CDM + Baryon + Massive Neutrino (MDM) cosmologies,
# using the Eisenstein and Hu fitting formulae (http://arxiv.org/abs/astro-ph/9710252).
# Code adapted from http://background.uchicago.edu/~whu/transfer/power.c
/* Daniel J. Eisenstein & Wayne Hu, Institute for Advanced Study */

import numpy as np

import mps

class mps_lin_eihu(mps.mps_lin):
    def __init__(self, cosmo, deltaH=None, tilt=1, sigma8=None):
        self.cosmo = cosmo

        theta_cmb = 2.728 / 2.7 # Assume T_cmb = 2.728 K */

        num_degen_hdm = 1.0

        omhh     = cosmo.omm*cosmo.h**2
        obhh     = cosmo.omb*cosmo.h**2
        f_baryon = cosmo.omb/cosmo.omm
        f_hdm    = 0.0 / cosmo.omm
        f_cdm    = 1.0 - f_baryon - f_hdm
        f_cb     = f_cdm + f_baryon
        f_bnu     = f_baryon + f_hdm

        # Compute the equality scale.
        z_equality = 25000.0*omhh/theta_cmb**4 # Actually 1+z_eq
        k_equality = 0.0746*omhh/theta_cmb**2

        # Compute the drag epoch and sound horizon.
        z_drag_b1 = 0.313*omhh**(-0.419)*(1+0.607*omhh**(0.674))
        z_drag_b2 = 0.238*omhh**(0.223)
        z_drag    = 1291.*omhh**(0.251)/(1.0+0.659*omhh**(0.828))*(1.0+z_drag_b1*obhh**(z_drag_b2))
        y_drag    = z_equality/(1.0+z_drag)

        sound_horizon_fit = 44.5*np.log(9.83/omhh)/np.sqrt(1.0+10.0*obhh**(0.75))

        # Set up for the free-streaming & infall growth function.
        p_c = 0.25*(5.0-np.sqrt(1+24.0*f_cdm))
        p_cb = 0.25*(5.0-np.sqrt(1+24.0*f_cb))

        # Compute small-scale suppression.
        alpha_nu       = (f_cdm/f_cb*(5.0-2.*(p_c+p_cb))/(5.-4.*p_cb)*
                          (1+y_drag)**(p_cb-p_c)*
                          (1+f_bnu*(-0.553+0.126*f_bnu*f_bnu))/
                          (1-0.193*np.sqrt(f_hdm*num_degen_hdm)+0.169*f_hdm*num_degen_hdm**(0.2))*
                          (1+(p_c-p_cb)/2*(1+1/(3.-4.*p_c)/(7.-4.*p_cb))/(1+y_drag)))
        alpha_gamma = np.sqrt(alpha_nu)
        beta_c      = 1/(1-0.949*f_bnu)

        self.num_degen_hdm = num_degen_hdm
        self.theta_cmb = theta_cmb

        self.omhh = omhh

        self.f_hdm = f_hdm
        self.f_cb = f_cb
        self.z_equality = z_equality

        self.sound_horizon_fit = sound_horizon_fit

        self.p_cb = p_cb

        self.alpha_gamma = alpha_gamma
        self.beta_c = beta_c

        self.tilt = tilt
        self.deltaH = deltaH
        if self.deltaH == None:
            self.deltaH = self.cobenorm()

        self.kmin = 1.e-10
        self.kmax = 1.e+10

        self.norm   = 1.0
        if sigma8 != None:
            tsigma8     = self.sigma_rz(8./self.cosmo.h, 0.0)
            self.norm   = sigma8 / tsigma8

    def p_kz(self, kk, z):
        omega_denom    = self.cosmo.oml+(1.0+z)**2*(self.cosmo.omm*(1.0+z))
        omega_lambda_z = self.cosmo.oml/omega_denom;
        omega_matter_z = self.cosmo.omm*(1.0+z)**3/omega_denom;
        growth_k0      = self.z_equality/(1.0+z)*2.5*omega_matter_z/(omega_matter_z**(4.0/7.0)-omega_lambda_z+
                                                                     (1.0+omega_matter_z/2.0)*(1.0+omega_lambda_z/70.0));
        growth_to_z0   = self.z_equality*2.5*self.cosmo.omm/(self.cosmo.omm**(4.0/7.0)
                                                             -self.cosmo.oml +
                                                             (1.0+self.cosmo.omm/2.0)*(1.0+self.cosmo.oml/70.0));
        growth_to_z0   = growth_k0/growth_to_z0

        qq = kk/self.omhh*(self.theta_cmb)**2

        # Compute the scale-dependent growth functions
        y_freestream = 0.0
        if self.f_hdm != 0:
            y_freestream = 17.2*self.f_hdm*(1+0.488*self.f_hdm**(-7.0/6.0))*(self.num_degen_hdm*qq/self.f_hdm)**2
        temp1 = growth_k0**(1.0-self.p_cb)
        temp2 = (growth_k0/(1+y_freestream))**(0.7);
        growth_cb = (1.0+temp2)**(self.p_cb/0.7)*temp1;
        growth_cbnu = (self.f_cb**(0.7/self.p_cb)+temp2)**(self.p_cb/0.7)*temp1;

        # Compute the master function 
        gamma_eff = self.omhh*(self.alpha_gamma+(1.-self.alpha_gamma)/
                               (1+(kk*self.sound_horizon_fit*0.43)**4));
        qq_eff = qq*self.omhh/gamma_eff;

        tf_sup_L = np.log(2.71828+1.84*self.beta_c*self.alpha_gamma*qq_eff)
        tf_sup_C = 14.4+325/(1+60.5*qq_eff**(1.11))
        tf_sup = tf_sup_L/(tf_sup_L+tf_sup_C*(qq_eff)**2)

        if self.f_hdm != 0:
            qq_nu = 3.92*qq*np.sqrt(self.num_degen_hdm/self.f_hdm)
            max_fs_correction = 1+1.2*self.f_hdm**(0.64)*self.num_degen_hdm**(0.3+0.6*self.f_hdm)/(qq_nu**(-1.6)+qq_nu**(0.8));
        else:
            max_fs_correction = 1.0
        tf_master = tf_sup*max_fs_correction;

        # Now compute the CDM+HDM+baryon transfer functions
        tf_cb = tf_master*growth_cb/growth_k0
        return (2997.0*kk/self.cosmo.h)**(self.tilt+3.) * self.cosmo.h**2 * (self.deltaH * tf_cb * growth_to_z0)**2 / kk**3 * (2.*np.pi)**2 * self.norm**2

    def cobenorm(self):
        # Return the Bunn & White (1997) fit for delta_H
        # Given lambda, omega_m, qtensors, and tilt
        # Open model with tensors is from Hu & White

        n = self.tilt-1;
        return 1.94e-5*self.cosmo.omm**(-0.785-0.05*np.log(self.cosmo.omm))*np.exp(-0.95*n-0.169*n*n)
