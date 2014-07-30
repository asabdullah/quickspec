import units

class kern():
    def __init__(self, cosmo):
        self.cosmo = cosmo
        
        self.xlss  = cosmo.x_z(1100.)
        self.cfac  = 3. * cosmo.omm * (cosmo.H0 * 1.e3 / units.c)**2

    def w_lxz(self, l, x, z):
        return self.cfac * (1.+z) * (x/l)**2 * (1./x - 1./self.xlss)
