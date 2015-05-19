dmdd
=======

This package enables simple simulation and Bayesian posterior analysis
of recoil-event data from dark-matter direct-detection experiments 
under a wide variety of scattering theories. It includes the following
features:

  * Enables calculation of the nuclear-recoil rates for a wide range of non-relativistic and relativistic scattering operators, including non-standard momentum-, velocity-, and spin-dependent rates,
  * Accounts for the correct nuclear response functions for each scattering operator, as given in Anand et al. (2013).
  * Takes into account the natural abundances of isotopes for a variety of experimental target elements.

The code is being actively developed on `GitHub
<http://github.com/veragluscevic/dmdd>`_;  please feel free to
contribute pull requests or raise issues.  If you use this code in
your research, please cite GitHub page; ID request is submitted to ASCL and will be available soon. 

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
