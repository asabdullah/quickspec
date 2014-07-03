#!/usr/bin/env python
#
# plot the grown factor as a function of redshift, with a vertical line marking lambda - matter equality.
#

import numpy as np
import pylab as pl

import quickspec as qs

lcdm = qs.cosmo.lcdm()

zs = np.arange(0., 100., 0.1)

pl.figure(figsize=(5,5))
pl.loglog( zs, [lcdm.G_z(z) for z in zs], c='k', label=r'$G(z)$' )
pl.loglog( zs, 1./(1.+zs), ls='--', c='r', label=r'$1/(1+z)$' )

pl.axvline(x=1./lcdm.aeq_lm() - 1., ls='--', color='k')

pl.xlabel(r'$z$')
pl.legend(); pl.setp( pl.gca().get_legend().get_frame(), visible=False )

pl.show()
