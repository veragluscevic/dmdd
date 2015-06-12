dmdd
=======

A python package that enables simple simulation and Bayesian posterior analysis
of nuclear-recoil data from dark matter direct detection experiments 
for a wide variety of theories of dark matter-nucleon interactions. 

``dmdd`` has the following features:

* Calculation of the nuclear-recoil rates for various non-standard momentum-, velocity-, and spin-dependent scattering models. 
 
* Calculation of the appropriate nuclear response functions triggered by the chosen scattering model.
  
* Inclusion of natural abundances of isotopes for a variety of target elements: Xe, Ge, Ar, F, I, Na, He.

* Simple simulation of data (where data is a list of nuclear recoil energies, including Poisson noise) under different models. 

* Bayesian analysis (parameter estimation and model selection) of data using ``MultiNest``.

All rate and response functions directly implement the calculations of `Anand et al. (2013) <http://arxiv.org/abs/1308.6288>`_ and `Fitzpatrick et al. (2013) <https://inspirehep.net/record/1094068?ln=en>`_ (for non-relativistic operators, in ``rate_genNR`` and ``rate_NR``), and `Gresham & Zurek (2014) <http://arxiv.org/abs/1401.3739>`_ (for UV-motivated scattering models in ``rate_UV``). Simulations follow the prescription from `Gluscevic & Peter (2014) <http://adsabs.harvard.edu/abs/2014JCAP...09..040G>`_ and Gluscevic et al. (2015).

The code is being actively developed on `GitHub
<http://github.com/veragluscevic/dmdd>`_;  please feel free to
contribute pull requests or raise issues.  If you use this code in
your research, please cite GitHub page; ID request has been submitted to ASCL. 

Dependencies
------------

All of the package dependencies (listed below) are contained within the `Anaconda python distribution <http://continuum.io/downloads>`_, except for ``MultiNest`` and ``PyMultinest``. 

For simulations, you will need:

* basic python scientific packages (``numpy``, ``scipy``, ``matplotlib``)

* ``cython``

To do posterior analysis, you will also need:

* ``MultiNest``

* ``PyMultiNest``

To install these two, follow the instructions `here <http://astrobetter.com/wiki/MultiNest+Installation+Notes>`_.


Installation
------------

Install ``dmdd`` either using pip::

    pip install dmdd

or by cloning the repository::

    git clone https://github.com/veragluscevic/dmdd.git
    cd dmdd
    python setup.py install
    

Usage
------

For a quick tour of usage, check out the `tutorial notebook <https://github.com/veragluscevic/dmdd/blob/master/dmdd_tutorial.ipynb>`_; for more complete documentation, `read the docs <http://dmdd.rtfd.org>`_; and for the most important formulas and definitions regarding the ``rate_NR`` and ``rate_genNR`` modules, see also `here <http://github.com/veragluscevic/dmdd/blob/master/rate_NR-and-genNR.pdf>`_.


Basic Usage
------------

Here is a quick example of basic usage:

.. code-block:: python

    from dmdd import UV_Model, Experiment, MultinestRun

    model1 = UV_Model('SI_Higgs', ['mass', 'sigma_si'], fixed_params={'fnfp_si': 1})
    model2 = UV_Model('SD_fu', ['mass','sigma_sd'], fixed_params={'fnfp_sd': -1.1})

    xe = Experiment('Xe', 'xenon', 5, 40, 1000, eff.efficiency_Xe)

    run = MultinestRun('sim', [xe,ge], model1,{'mass':50.,'sigma_si':70.},
                   model2, prior_ranges={'mass':(1,1000), 'sigma_sd':(0.001,1000)})

    run.fit()
    run.visualize()



Attribution
-----------

This package was originally developed for to derive the results of Gluscevic et al (2015). If you use this code in your research, please cite this ASCL reference [pending], and the following publications: Gluscevic et al (2015), `Anand et al. (2013) <http://arxiv.org/abs/1308.6288>`_, `Fitzpatrick et al. (2013) <https://inspirehep.net/record/1094068?ln=en>`_, and `Gresham & Zurek (2014) <http://arxiv.org/abs/1401.3739>`_. 

API Documentation
-----------------

.. toctree::
   :maxdepth: 2

   rates
   api
