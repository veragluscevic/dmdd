.. _rates:

Nuclear Recoil Rates
=========

``dmdd`` has two modules that let you calculate differential rate :math:`\frac{dR}{dE_R}` and total rate :math:`R(E_R)` of nuclear-recoil events within two theoretical frameworks: 

I) ``rate_UV``: rates for a variety of UV-complete theories (as described in `Gresham & Zurek (2014) <http://arxiv.org/abs/1401.3739>`_ and Gluscevic et al., 2015).

II) ``rate_genNR``: rates for all non-relativistic scattering operators (including interference terms) of `Fitzpatrick et al., 2013 <https://inspirehep.net/record/1094068?ln=en>`_.

Appropriate nuclear response functions (accompanied by the right momentum and energy dependencies of the rate) are automatically folded in, and for a specified target element natural abundance of its isotopes (with their specific response functions) are taken into account.

Installation of ``dmdd`` will also install the ``rate_UV`` 
and ``rate_genNR`` modules
(compiled from cython).   Because readthedocs can't build cython code, these
functions can't have the nice and pretty auto-generated sphinx documentation, 
so the documentation strings for the most important functions are just copied here.

rate_UV
--------

rate_UV.dRdQ()
^^^^^^

    This is the main differential nuclear-recoil rate function. Its output (in units of cts/keV/kg/s) is computed for any one of 18 different scattering scenarios (involving 9 different UV operators), by setting the appropriate ``sigma_*`` parameter to a non-zero value.

    
    :param Q:
        Nuclear recoil energies [keV]
    :type Q: ``ndarray``

    :param mass:
        Dark-matter particle mass [GeV]

    :param sigma_*:
        Various scattering cross sections [cm^2] off a proton. The value passed will be multiplied with a normalization factor given in ``dmdd.PAR_NORMS``. See explanation of suffixes below; the default ``PAR_NORMS`` are also listed.

    :param fnfp_*:
        Dimensionless ratio of neutron to proton coupling 

    :param v_lag,v_rms,v_esc:
        Lag, RMS, and escape velocities [km/s]. Note that ``v_rms`` is
        3/2x the standard RMS of a Maxwellian velocity distribution;
        that is, the default ``v_rms`` value = 220 km/s.

    :param rho_x:
        Dark matter energy density.

    :param element:
        Target nucleus element name (all lower case).
    :type element: ``str``

    Parameter suffixes:
    -------
    
    =========  ============================= ===== =====
    Suffix     Meaning                       norm  norm (massless)
    =========  ============================= ===== =====
    _si        spin-independent              1e-47 1e-48
    _sd        spin-dependent                1e-42 1e-43
    _anapole   anapole                       1e-40 1e-45
    _magdip    magnetic dipole               1e-38 1e-39
    _elecdip   electric dipole               1e-44 1e-45
    _LS        :math:`L \cdot S` generating  1e-44 1e-42
    _f1        pseudoscalar-scalar (DM-SM)   1e-47 1e-48
    _f2        scalar-pseudoscalar (DM-SM)   1e-42 1e-43
    _f3        pseudoscalar-pseudoscalar     1e-41 1e-42
    =========  ============================= ===== =====

    In all cases, the mediator can turn "massless" by appending _massless.


rate_UV.R()
^^^^^^
    Theoretical total integrated recoil-energy rate.

    Integrates :func:`dRdQ` between ``Qmin`` and ``Qmax`` using
    trapezoidal integration over 100 points.
    
    :param efficiency_function:
        Recoil-detection efficiency as a function of nuclear recoil energy.
    :type efficiency_function: ``function``

    :param mass:
        Dark-matter particle mass [GeV]

    :param sigma_*:
        Various scattering cross sections [in cm^2] off a proton. See :func:`dRdQ` for details.

    :param fnfp_*:
        Dimensionless ratio of neutron to proton coupling 

    :param v_lag,v_rms,v_esc:
        Lag, RMS, and escape velocities [km/s]. Note that ``v_rms`` is
        3/2x the standard RMS of a Maxwellian velocity distribution;
        that is, the default ``v_rms`` value = 220 km/s.

    :param rho_x:
        Dark matter energy density.

    :param element:
        Target nucleus element name (all lower case).
    :type element: ``str``

    :param Qmin,Qmax:
        Nuclear-recoil energy window of experiment.
    
    For reference on other paremeters, see :func:`dRdQ`.


loglikelihood()
^^^^^^^^^^^^^^^
    Log-likelihood of array of recoil energies
    
    :param Er: 
      Array of energies in keV.
      It can have as many entries as desired.
    :type Er: ``np.ndarray``
      
    :param efficiency: 
      Fractional efficiency as a function of energy.
      Efficiencies available:
        dmdd.eff.efficiency_Ar
        dmdd.eff.efficiency_Ge
        dmdd.eff.efficiency_I
        dmdd.eff.efficiency_KIMS
        dmdd.eff.efficiency_Xe
      right now, all of these are set to be constant and equal to 1.

    :type efficiency: ``function``
      
    :param Qmin,Qmax: 
      Minimum/maximum energy in keVNR of interest, e.g. detector threshold
      default 2., cutoff default 30.
    
    :param mass: 
      Dark matter mass in GeV.
      Default to 50. (optional)
    :type mass: ``float``

    :param sigma_*:
        Various scattering cross sections [in cm^2] off a proton. See :func:`dRdQ` for details.

    :param fnfp_*:
        Dimensionless ratio of neutron to proton coupling 

    :param v_lag: 
      Velocity of the solar system with respect to the Milky Way in km/s.
      Default to 220. (optional)
    :type v_lag: ``float``
    
    :param v_rms: 
      1.5 * (velocity dispersion in km/s) of the local DM distribution.
      Default to 220. (optional)
    :type v_rms: ``float``
    
    :param v_esc: 
      Escape velocity in km/s of a DM particle in the galactic rest frame.
      Default to 544. (optional)
    :type v_esc: ``float``
    
    :param element:
      Name of the detector element.
      Choice of: 'argon', 'fluorine', 'germanium', 'helium', 'iodine', 'sodium', or 'xenon'
      Default to 'xenon' (optional)
    :type element: ``str``
    
    :param rho_x:
      Local DM density in GeV/cm^3.
      Default to 0.3 (optional)
    :type rho_x: ``float``
    
    For reference on other paremeters, see :func:`dRdQ`.

rate_genNR
--------

rate_genNR.dRdQ()
^^^^^^^

    Differential recoil energy spectrum in counts/keV/kg/sec. Its output (in units of cts/keV/kg/s) is computed for any combination of 28 different couplings to nucleons, by setting the appropriate ``cXN_*`` parameters to a non-zero value. 

    :param Er:
      This is a list of energies in keV.
      The list must be entered as a numpy array, np.array([..]).
      It can have as many entries as desired.
    :type Er: ``np.ndarray``
      
    :param mass:
       Dark matter mass in GeV.
       Default to 50. (optional)
    :type mass: ``float``
    
    :param 28 cXN's:
      28 different np.arrays, all optional
      These are the EFT coefficients.
      
      X can be any number from 1-15 (except 2).
      N must be n or p.
      Any cXN that is entered is a list of coefficients.
      
      The list must be entered as a numpy array, np.array([..]).
      
      c1N and c4N must have three entries, any of which may be zero:

        -the first entry is the momentum-independent term.

        -the second entry is the coefficient of the q^2/mDM^2 term.

        -the third entry is the coefficient of the q^4/mDM^4 term.
        
      c3N and c5Y-c12N must have two entries, any of which may be zero:
      
          -the first entry is the momentum-independent term.

          -the second entry is the coefficient of the q^2/mDM^2 term.
          
      c13N-c15N must have one entry.
      All cXN have mass dimension negative two. The mass scale of the suppression is 500 GeV by default (may be adjusted; see below). By default all coefficients are set to zero.

    :param c_scale: 
      Suppression scale of all cXN coefficients in GeV.
      From a UV perspective, this could be mediator_mass/sqrt(couplings).
      Default 500. (optional)
    :type c_scale: ``float``

    :param v_lag: 
      Velocity of the solar system with respect to the Milky Way in km/s.
      Default to 220. (optional)
    :type v_lag: ``float``
    
    :param v_rms: 
       1.5 * (velocity dispersion in km/s) of the local DM distribution.
       Default to 220.
    
    :param v_esc: 
       Escape velocity in km/s of a DM particle in the galactic rest frame.
       Default to 544.
    
    :param element: 
      Name of the detector element.
      Choice of: 'argon', 'fluorine', 'germanium', 'helium', 'iodine', 'sodium', or 'xenon'
      Default to 'xenon'
    :type element: ``str``
      
    :param rho_x: 
      Local DM density in GeV/cm^3.
      Default to 0.3
      

    :return: dRdQ 
      array of differential recoil energy spectrum in
      counts/keV/kg/sec

rate_genNR.R()
^^^^^^^^
    Fractional observed events in counts/kg/sec
    
    Multiply this by an exposure in kg*sec to get a total number of observed events

    :param efficiency: 
      Fractional efficiency as a function of energy.
      Efficiencies available:
        dmdd.eff.efficiency_Ar
        dmdd.eff.efficiency_Ge
        dmdd.eff.efficiency_I
        dmdd.eff.efficiency_KIMS
        dmdd.eff.efficiency_Xe
      right now, all of these are set to be constant and equal to 1.
    :type efficiency: ``object``
    
    :param Qmin,Qmax: 
      Minimum/maximum energy in keVNR of interest, e.g. detector threshold
      default 2. and detector cutoff default 30.
        
    :param mass: 
      Dark matter mass in GeV.
      Default to 50. (optional)
    :type mass: ``float``
    
    :param 28 different cXN's:
        28 different np.arrays, all optional
        See :func:`dRdQ` for details.
    
    :param c_scale: 
      Suppression scale of all cXN coefficients in GeV.
      From a UV perspective, this could be mediator_mass/sqrt(couplings).
      Default 500. (optional)
    :type c_scale: ``float``

    :param v_lag: 
      Velocity of the solar system with respect to the Milky Way in km/s.
      Default to 220. (optional)
    :type v_lag: ``float``
    
    :param v_rms:
      1.5 * (velocity dispersion in km/s) of the local DM distribution.
      Default to 220. (optional)
    :type v_rms: ``float``
    
    :param v_esc:
      Escape velocity in km/s of a DM particle in the galactic rest frame.
      Default to 544. (optional)
    :type v_esc: ``float``
    
    :param element:
      Name of the detector element.
      Choice of: 'argon', 'fluorine', 'germanium', 'helium', 'iodine', 'sodium', or 'xenon'
      Default to 'xenon' (optional)
    :type element: ``str``
    
    :param rho_x:
      Local DM density in GeV/cm^3.
      Default to 0.3 (optional)
    :type rho_x: ``float``

loglikelihood()
^^^^^^^^^^^^^^^
    Log-likelihood of array of recoil energies
    
    :param Er: 
      Array of energies in keV.
      It can have as many entries as desired.
    :type Er: ``np.ndarray``
      
    :param efficiency: 
      Fractional efficiency as a function of energy.
      Efficiencies available:
        dmdd.eff.efficiency_Ar
        dmdd.eff.efficiency_Ge
        dmdd.eff.efficiency_I
        dmdd.eff.efficiency_KIMS
        dmdd.eff.efficiency_Xe
      right now, all of these are set to be constant and equal to 1.

    :type efficiency: ``function``
      
    :param Qmin,Qmax: 
      Minimum/maximum energy in keVNR of interest, e.g. detector threshold
      default 2., cutoff default 30.
    
    :param mass: 
      Dark matter mass in GeV.
      Default to 50. (optional)
    :type mass: ``float``
    
    :param 28 cXN's:
      28 different np.arrays, all optional
      These are the EFT coefficients.
      See :func:`dRdQ` for details.

    
    :param c_scale: 
      Suppression scale of all cXN coefficients in GeV.
      From a UV perspective, this could be mediator_mass/sqrt(couplings).
      Default 500. (optional)
    :type c_scale: ``float``

    :param v_lag: 
      Velocity of the solar system with respect to the Milky Way in km/s.
      Default to 220. (optional)
    :type v_lag: ``float``
    
    :param v_rms: 
      1.5 * (velocity dispersion in km/s) of the local DM distribution.
      Default to 220. (optional)
    :type v_rms: ``float``
    
    :param v_esc: 
      Escape velocity in km/s of a DM particle in the galactic rest frame.
      Default to 544. (optional)
    :type v_esc: ``float``
    
    :param element:
      Name of the detector element.
      Choice of: 'argon', 'fluorine', 'germanium', 'helium', 'iodine', 'sodium', or 'xenon'
      Default to 'xenon' (optional)
    :type element: ``str``
    
    :param rho_x:
      Local DM density in GeV/cm^3.
      Default to 0.3 (optional)
    :type rho_x: ``float``
