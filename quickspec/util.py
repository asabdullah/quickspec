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
