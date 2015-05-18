.. _rates:

Nuclear Recoil Rates
=========

``dmdd`` has two modules that let you calculate differential rate :math:`\frac{dR}{dE_R}` and total rate :math:`R(E_R)` of nuclear-recoil events within two theoretical frameworks: 

I) ``rate_UV``: rates for a variety of UV-complete theories (as described in `Gresham & Zurek (2014) <http://arxiv.org/abs/1401.3739>`_ and Gluscevic et al., 2015).

II) ``rate_genNR``: rates for all non-relativistic scattering operators (including interference terms) of `Fitzpatrick et al., 2013 <https://inspirehep.net/record/1094068?ln=en>`_.

Appropriate nuclear response functions (accompanied by the right momentum and energy dependencies of the rate) are automatically folded in, and for a specified target element natural abundance of its isotopes (with their specific response functions) are taken into account.

rate_UV
--------

This module is written in cython for fast rate calculations.

.. automodule:: rate_UV
  :members: dRdQ, R

rate_genNR
--------

.. automodule:: rate_genNR
  :members:

