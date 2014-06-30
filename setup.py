#!/usr/bin/env python

import glob

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('quickspec',parent_package,top_path)
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(name='quickspec',
          packages=['quickspec'],
          package_data={'quickspec': ['cib/data/Bethermin_2011_jbar/*']},
          configuration=configuration)
