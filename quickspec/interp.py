import numpy as np

def lagrange(x, xv, yv, npts=3, check=True):
    """ lagrange interpolation """
    assert( npts > 1 ) #behavior not yet defined for npts <= 1.

    if check == True:
        try:
            assert( x >= xv[+0] )
            assert( x <= xv[-1] )
        except:
            print "qs::interp::lagrange. bounds error. "
            print "   xl, x, xh = (%2.2e, %2.2e, %2.2e)" % (xv[0], x, xv[-1])
            assert(0)

    dx    = int( np.floor(0.5*npts) )
    ixmin = min( max( np.searchsorted( xv, x, side='left' ) - dx, 0 ), len(xv) - npts )
    idxs  = np.arange(0,npts) + ixmin

    iv    = np.arange(0, npts)
    xs    = xv[idxs]
    ys    = yv[idxs]

    lx = 0.0
    for i, tx, ty in zip(iv, xs, ys):
        lx += 1. * ty * np.prod( (x - xs)[iv != i] ) / np.prod( (tx - xs)[iv != i] )

    return lx
