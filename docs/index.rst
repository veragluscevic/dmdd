dmdd
=======

dmdd is a python package that enables simple simulation and Bayesian posterior analysis
of nuclear-recoil data from dark matter direct detection experiments 
for a wide variety of scattering theories. 

dmdd has the following features:

* Calculation of the nuclear-recoil rates for a variety of scattering scenarios, including non-standard momentum-, velocity-, and spin-dependent recoil rates. All rate and response functions from rate modules directly use incorporate from `Anand et al. (2013) <http://arxiv.org/abs/1308.6288>`_ (for non-relativistic operators, in rate_genNR and rate_NR) and `Gresham & Zurek (2014) <http://arxiv.org/abs/1401.3739>`_ (for UV-motivated scattering models in rate_UV). 
 
* Calculation of the nuclear response functions as presented in `Anand et al. (2013) <http://arxiv.org/abs/1308.6288>`_.
  
* Inclusion of natural abundances of isotopes for a variety of target elements: Xe, Ge, Ar, F, I, Na, He.

* Simple simulation of data (where data is a list of nuclear recoil energies) under different scattering models, including Poisson noise, following 

* Bayesian analysis (parameter estimation and model selection) of recoil data using MultiNest.
 

The code is being actively developed on `GitHub
<http://github.com/veragluscevic/dmdd>`_;  please feel free to
contribute pull requests or raise issues.  If you use this code in
your research, please cite GitHub page; ID request has been submitted to ASCL. 

Installation
------------

Install either using pip::

    pip install dmdd

or by cloning the repository::

    git clone https://github.com/veragluscevic/dmdd.git
    cd dmdd
    python setup.py install


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

For more details on usage and interactive documentation, see the `tutorial
notebook <>`_. 

API Documentation
-----------------

.. toctree::
   :maxdepth: 2

   rates
   api
