dmdd
=========

This package enables simple simulation and Bayesian posterior analysis
of recoil-event data from dark-matter direct-detection experiments 
under a wide variety of scattering theories. It includes the following
features:

* Enables calculation of the nuclear-recoil rates for a wide range of non-relativistic and relativistic scattering operators, including non-standard momentum-, velocity-, and spin-dependent rates,
 
* Accounts for the correct nuclear response functions for each scattering operator, as given in Anand et al. (2013).
  
* Takes into account the natural abundances of isotopes for a variety of experimental target elements.

Installation
------------

Install either using pip::

    pip install dmdd

or by cloning the repository::

    git clone https://github.com/veragluscevic/dmdd.git
    cd dmdd
    python setup.py install

Usage
------

For a quick tour of usage, check out the `tutorial notebook <http://nbviewer.ipython.org/github/veragluscevic/dmdd/blob/master/dmdd_tutorial.ipynb>`_; for more complete documentation, `read the docs <http://dmdd.rtfd.org>`_; and for the most important formulas and definitions regarding rate_NR and rate_genNR, see `here <http://github.com/veragluscevic/dmdd/blob/master/rate_NR-and-genNR.pdf>`_.

Attribution
-----------

If you use this code in your research, please use the following BibTex
citation::

