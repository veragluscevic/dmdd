dmdd
=========

dmdd is a python package that enables simple simulation and Bayesian posterior analysis
of nuclear-recoil data from dark matter direct detection experiments 
for a wide variety of scattering theories. 

dmdd has the following features:

* Calculation of the nuclear-recoil rates for a variety of scattering scenarios, including non-standard momentum-, velocity-, and spin-dependent recoil rates. All rate and response functions from rate modules directly use incorporate from `Anand et al. (2013) <http://arxiv.org/abs/1308.6288>`_ (for non-relativistic operators, in rate_genNR and rate_NR) and `Gresham & Zurek (2014) <http://arxiv.org/abs/1401.3739>`_ (for UV-motivated scattering models in rate_UV). 
 
* Calculation of the nuclear response functions as presented in `Anand et al. (2013) <http://arxiv.org/abs/1308.6288>`_.
  
* Inclusion of natural abundances of isotopes for a variety of target elements: Xe, Ge, Ar, F, I, Na, He.

* Simple simulation of data (where data is a list of nuclear recoil energies) under different scattering models, including Poisson noise, following 

* Bayesian analysis (parameter estimation and model selection) of recoil data using MultiNest.
 

Dependencies
------------

For simulations:

* numpy, …

For posterior analysis:

* MultiNest

* PyMultinest




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

For a quick tour of usage, check out the `tutorial notebook <http://nbviewer.ipython.org/github/veragluscevic/dmdd/blob/master/dmdd_tutorial.ipynb>`_; for more complete documentation, `read the docs <http://dmdd.rtfd.org>`_; and for the most important formulas and definitions regarding the ``rate_NR`` and ``rate_genNR`` modules, see `here <http://github.com/veragluscevic/dmdd/blob/master/rate_NR-and-genNR.pdf>`_.

Attribution
-----------

This package was originally developed for to derive results of Gluscevic et al (2015). If you use this code in your research, please cite the following publications: Gluscevic et al (2015), Gresham and Zurek (2014), and `Anand et al. (2013) <http://arxiv.org/abs/1308.6288>`_; also please include the following ASCL ID [pending].


