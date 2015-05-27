.. _rates:

Nuclear Recoil Rates
=========

``dmdd`` has three modules that let you calculate the differential rate :math:`dR/dE_R`, the total rate :math:`R(E_R)`, and the log(likelihood) :math:`\ln \mathcal{L}`  of nuclear-recoil events within two theoretical frameworks: 

I) ``rate_UV``: rates for a variety of UV-complete theories (as described in `Gresham & Zurek (2014) <http://arxiv.org/abs/1401.3739>`_ and Gluscevic et al., 2015). This takes form factors from formUV.pyx

II) ``rate_genNR``: rates for all non-relativistic scattering operators (automatically including interference terms) of `Fitzpatrick et al., 2013 <https://inspirehep.net/record/1094068?ln=en>`_. This takes form factors from formgenNR.pyx

III) ``rate_NR``: rates for individual nuclear responses compatible with the EFT (**not** automatically including interference terms) of `Fitzpatrick et al., 2013 <https://inspirehep.net/record/1094068?ln=en>`_. This takes form factors from formNR.pyx

For a specified target element, the natural abundance of its isotopes (with their specific response functions) is taken into account.

Each module is written in cython for fast rate calculations.

rate_UV
--------

.. automodule:: rate_UV
  :members: dRdQ, R, loglikelihood

rate_genNR
--------

.. automodule:: rate_genNR
  :members: dRdQ, R, loglikelihood

rate_NR
--------

.. automodule:: rate_NR
  :members: dRdQ, R, loglikelihood

