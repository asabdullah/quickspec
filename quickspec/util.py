import sys, time, os, urllib
import numpy as np

def enumerate_progress(list, label=''):
    """ version of python's builtin 'enumerate' iterator which displays a progress bar. """
    t0 = time.time()
    ni = len(list)
    for i, v in enumerate(list):
        yield i, v
        ppct = int(100. * (i-1) / ni)
        cpct = int(100. * (i+0) / ni)
        if cpct > ppct:
            dt = time.time() - t0
            dh = np.floor( dt / 3600. )
            dm = np.floor( np.mod(dt, 3600.) / 60.)
            ds = np.floor( np.mod(dt, 60) )
            sys.stdout.write( "\r [" + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] " +
                              label + " " + int(10. * cpct / 100)*"-" + "> " + ("%02d" % cpct) + r"%" )
            sys.stdout.flush()
    sys.stdout.write("\n"); sys.stdout.flush()

def deriv(xv, yv=None):
    """ numerical differentiation based on 3-point lagrange interpolation, following the IDL 'DERIV' function. """
    ni = len(xv)
    assert(ni >= 3)

    if yv == None:
        yv = xv
        xv = np.arange(0, ni)
    assert( len(yv) == len(xv) )

    idxs = {}
    for i in xrange(-1, 2):
        idxs[i] = np.arange(0, ni) + i
        idxs[i][0:1] = i+1
        idxs[i][ni-1:ni] = i+ni-2

    ret = np.zeros(ni)
    for i in xrange(-1, 2):
        num = np.zeros(ni)
        den = np.ones(ni)

        for j in xrange(-1, 2):
            if i==j: continue

            num += (xv - xv[idxs[j]])
            den *= (xv[idxs[i]] - xv[idxs[j]])

        ret += yv[idxs[i]] * num / den

    return ret

def download(url, fname):
    """ retrieve contents of url, copy to fname. """
    pct = 0
    def dlhook(count, block_size, total_size):
        ppct = int(100. * (count-1) * block_size / total_size)
        cpct = int(100. * (count+0) * block_size / total_size)
        if cpct > ppct: sys.stdout.write( "\r quickspec::util::download:: " + int(10. * cpct / 100)*"-" + "> " + ("%02d" % cpct) + r"%" ); sys.stdout.flush()

    try:
        urllib.urlretrieve(url, filename=fname, reporthook=dlhook)
        sys.stdout.write( "\n" ); sys.stdout.flush()
    except:
        print "quickspec::util::download failed! removing partial."
        os.remove(fname)

def pair(k, z):
    """ helper function to broadcast k and z to equally sized 1D arrays. """
    k = np.asarray(k)
    z = np.asarray(z)
    if (np.size(k) > 1) and (np.size(z) == 1):
        z = np.ones( np.size(k) ) * z
        s = np.shape(k)
    elif (np.size(z) > 1) and (np.size(k) == 1):
        k = np.ones( np.size(z) ) * k
        s = np.shape(z)
    else:
        assert( np.size(k) == np.size(z) )
        s = np.shape(k)

    return k.flatten(), z.flatten(), s
