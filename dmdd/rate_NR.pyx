import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool

import math
import formNR as formNR
from helpers import trapz, eta, zeta
from dmdd.globals import PAR_NORMS
import dmdd.constants as const
DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef extern from "math.h":
    float INFINITY
    double exp(double)
    double sqrt(double)
    double erf(double)
    double log10(double)
    double log(double)
    

#constants for this module:
cdef DTYPE_t ratenorm = 1.68288e31 # this converts from cm**-1 * GeV**-2 to DRU = cts / keV / kg / sec

#physical constants from constants.py:
cdef DTYPE_t mN = const.NUCLEON_MASS # Nucleon mass in GeV
cdef DTYPE_t pmag = const.P_MAGMOM # proton magnetic moment, PDG Live
cdef DTYPE_t nmag = const.N_MAGMOM # neutron magnetic moment, PDG Live

#information about target nuclei:
eltshort = const.ELEMENT_INFO

############################# rates dR/dQ for each response:

# M response
@cython.boundscheck(False)
def dRdQM(np.ndarray[DTYPE_t] Er, DTYPE_t v_rms, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t std_coeff, DTYPE_t v2_coeff, DTYPE_t q2_coeff, DTYPE_t v2q2_coeff, DTYPE_t q4_coeff, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from the M nuclear response alone in units of cts/keV/kg/s. This is functionally equivalent to the standard spin-independent rate.

    :param Er:
        An array of keV energies
    :type Er: ``ndarray``
        
    :param V0:
        Velocity in km/s
    :type V0: ``float``

    :param v_lag:
        Lag velocity in km/s.
    :type v_lag: ``float``

    :param v_esc:
        Galactic escape velocity in km/s
    :type v_esc: ``float``

    :param mx:
        Dark matter particle mass in GeV
    :type mx: ``float``

    :param sigp:
        Dark-matter-proton scattering cross section normalized to
        give about 1 count at LUX when set to 1.
    :type sigp: ``float``
        
    :param fnfp:
        Dimensionless ratio of neutron to proton coupling.
    :type fnfp: ``float``

    :param elt:
        element name
    :type elt: ``str``

    :param rho_x: (optional)
        Local dark matter density.
    :type rho_x: ``float``
    
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, weight, v_independent
    cdef DTYPE_t ff

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    q_scale = 0.1
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_si'] / (2. * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        qsq = 2.*weight*mN*q # this is qsquared, which multiplies the coefficients
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        ff = formNR.factor_M(element_name,y_harm,fnfp)
        val_eta = eta(v_min,v_esc,v_rms,v_lag)
        val_zeta = zeta(v_min,v_esc,v_rms,v_lag)
        tot = v_independent * ( (std_coeff + q2_coeff*qsq/q_scale**2. + q4_coeff*qsq**2./q_scale**4.)*val_eta + (v2_coeff + v2q2_coeff*qsq/q_scale**2.)*val_zeta ) * ff
        out[i] = tot
    return out

# Sigma'' response
@cython.boundscheck(False)
def dRdQSigPP(np.ndarray[DTYPE_t] Er, DTYPE_t v_rms, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t std_coeff, DTYPE_t v2_coeff, DTYPE_t q2_coeff, DTYPE_t v2q2_coeff, DTYPE_t q4_coeff, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from the Sigma'' response in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQM`
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, weight, v_independent
    cdef DTYPE_t ff

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    q_scale = 0.1
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_si'] / (2. * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        qsq = 2.*weight*mN*q # this is qsquared, which multiplies the coefficients
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        ff = formNR.factor_SigPP(element_name,y_harm,fnfp)
        val_eta = eta(v_min,v_esc,v_rms,v_lag)
        val_zeta = zeta(v_min,v_esc,v_rms,v_lag)
        tot = v_independent * ( (std_coeff + q2_coeff*qsq/q_scale**2. + q4_coeff*qsq**2./q_scale**4.)*val_eta + (v2_coeff + v2q2_coeff*qsq/q_scale**2.)*val_zeta ) * ff
        out[i] = tot
    return out

# Sigma' response
@cython.boundscheck(False)
def dRdQSigP(np.ndarray[DTYPE_t] Er, DTYPE_t v_rms, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t std_coeff, DTYPE_t v2_coeff, DTYPE_t q2_coeff, DTYPE_t v2q2_coeff, DTYPE_t q4_coeff, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from the Sigma' response in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQM`
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, weight, v_independent
    cdef DTYPE_t ff

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    q_scale = 0.1
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_si'] / (2. * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        qsq = 2.*weight*mN*q # this is qsquared, which multiplies the coefficients
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        ff = formNR.factor_SigP(element_name,y_harm,fnfp)
        val_eta = eta(v_min,v_esc,v_rms,v_lag)
        val_zeta = zeta(v_min,v_esc,v_rms,v_lag)
        tot = v_independent * ( (std_coeff + q2_coeff*qsq/q_scale**2. + q4_coeff*qsq**2./q_scale**4.)*val_eta + (v2_coeff + v2q2_coeff*qsq/q_scale**2.)*val_zeta ) * ff
        out[i] = tot
    return out

# Phi'' response
@cython.boundscheck(False)
def dRdQPhiPP(np.ndarray[DTYPE_t] Er, DTYPE_t v_rms, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t v2_coeff, DTYPE_t q2_coeff, DTYPE_t v2q2_coeff, DTYPE_t q4_coeff, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from the Phi'' response in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQM`
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, weight, v_independent
    cdef DTYPE_t ff

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    q_scale = 0.1
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_si'] / (2. * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        qsq = 2.*weight*mN*q # this is qsquared, which multiplies the coefficients
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        ff = formNR.factor_PhiPP(element_name,y_harm,fnfp)
        val_eta = eta(v_min,v_esc,v_rms,v_lag)
        val_zeta = zeta(v_min,v_esc,v_rms,v_lag)
        tot = v_independent * ( (q2_coeff*qsq/q_scale**2. + q4_coeff*qsq**2./q_scale**4.)*val_eta + (v2_coeff + v2q2_coeff*qsq/q_scale**2.)*val_zeta ) * ff
        out[i] = tot
    return out

# Delta response
@cython.boundscheck(False)
def dRdQDelta(np.ndarray[DTYPE_t] Er, DTYPE_t v_rms, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t v2_coeff, DTYPE_t q2_coeff, DTYPE_t v2q2_coeff, DTYPE_t q4_coeff, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from the Delta response in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQM`
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, weight, v_independent
    cdef DTYPE_t ff

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    q_scale = 0.1
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_si'] / (2. * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        qsq = 2.*weight*mN*q # this is qsquared, which multiplies the coefficients
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        ff = formNR.factor_Delta(element_name,y_harm,fnfp)
        val_eta = eta(v_min,v_esc,v_rms,v_lag)
        val_zeta = zeta(v_min,v_esc,v_rms,v_lag)
        tot = v_independent * ( (q2_coeff*qsq/q_scale**2. + q4_coeff*qsq**2./q_scale**4.)*val_eta + (v2_coeff + v2q2_coeff*qsq/q_scale**2.)*val_zeta ) * ff
        out[i] = tot
    return out

# MPhi'' response
@cython.boundscheck(False)
def dRdQMPhiPP(np.ndarray[DTYPE_t] Er, DTYPE_t v_rms, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t v2_coeff, DTYPE_t q2_coeff, DTYPE_t v2q2_coeff, DTYPE_t q4_coeff, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from the M-Phi'' (interference) response in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQM`
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, weight, v_independent
    cdef DTYPE_t ff

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    q_scale = 0.1
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_si'] / (2. * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        qsq = 2.*weight*mN*q # this is qsquared, which multiplies the coefficients
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        ff = formNR.factor_MPhiPP(element_name,y_harm,fnfp)
        val_eta = eta(v_min,v_esc,v_rms,v_lag)
        val_zeta = zeta(v_min,v_esc,v_rms,v_lag)
        tot = v_independent * ( (q2_coeff*qsq/q_scale**2. + q4_coeff*qsq**2./q_scale**4.)*val_eta + (v2_coeff + v2q2_coeff*qsq/q_scale**2.)*val_zeta ) * ff
        out[i] = tot
    return out

# Sig'Delta response
@cython.boundscheck(False)
def dRdQSigPDelta(np.ndarray[DTYPE_t] Er, DTYPE_t v_rms, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t v2_coeff, DTYPE_t q2_coeff, DTYPE_t v2q2_coeff, DTYPE_t q4_coeff, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from the Sigma'-Delta (interference) response in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQM`
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, weight, v_independent
    cdef DTYPE_t ff

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    q_scale = 0.1
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_si'] / (2. * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        qsq = 2.*weight*mN*q # this is qsquared, which multiplies the coefficients
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        ff = formNR.factor_SigPDelta(element_name,y_harm,fnfp)
        val_eta = eta(v_min,v_esc,v_rms,v_lag)
        val_zeta = zeta(v_min,v_esc,v_rms,v_lag)
        tot = v_independent * ( (q2_coeff*qsq/q_scale**2. + q4_coeff*qsq**2./q_scale**4.)*val_eta + (v2_coeff + v2q2_coeff*qsq/q_scale**2.)*val_zeta ) * ff
        out[i] = tot
    return out

#######################  "Master" rate functions

@cython.boundscheck(False)
def dRdQ(np.ndarray[DTYPE_t] Q, DTYPE_t mass=50.,
         DTYPE_t sigma_M=0., DTYPE_t sigma_SigPP=0., DTYPE_t sigma_SigP=0., DTYPE_t sigma_PhiPP=0., DTYPE_t sigma_Delta=0., DTYPE_t sigma_MPhiPP=0., DTYPE_t sigma_SigPDelta=0.,
         DTYPE_t fnfp_M=1., DTYPE_t fnfp_SigPP=1., DTYPE_t fnfp_SigP=1., DTYPE_t fnfp_PhiPP=1., DTYPE_t fnfp_Delta=1., DTYPE_t fnfp_MPhiPP=1., DTYPE_t fnfp_SigPDelta=1.,
         DTYPE_t stdco_M=1., DTYPE_t stdco_SigPP=1., DTYPE_t stdco_SigP=1.,
         DTYPE_t q2co_M=0., DTYPE_t q2co_SigPP=0., DTYPE_t q2co_SigP=0., DTYPE_t q2co_PhiPP=1., DTYPE_t q2co_Delta=1., DTYPE_t q2co_MPhiPP=1., DTYPE_t q2co_SigPDelta=1.,
         DTYPE_t q4co_M=0., DTYPE_t q4co_SigPP=0., DTYPE_t q4co_SigP=0., DTYPE_t q4co_PhiPP=0., DTYPE_t q4co_Delta=0., DTYPE_t q4co_MPhiPP=0., DTYPE_t q4co_SigPDelta=0.,
         DTYPE_t v2co_M=0., DTYPE_t v2co_SigPP=0., DTYPE_t v2co_SigP=0., DTYPE_t v2co_PhiPP=0., DTYPE_t v2co_Delta=0., DTYPE_t v2co_MPhiPP=0., DTYPE_t v2co_SigPDelta=0.,
         DTYPE_t v2q2co_M=0., DTYPE_t v2q2co_SigPP=0., DTYPE_t v2q2co_SigP=0., DTYPE_t v2q2co_PhiPP=0., DTYPE_t v2q2co_Delta=0., DTYPE_t v2q2co_MPhiPP=0., DTYPE_t v2q2co_SigPDelta=0.,
         DTYPE_t v_lag=220., DTYPE_t v_rms=220., DTYPE_t v_esc=544., DTYPE_t rho_x=0.3,
         str element='xenon'):
    """
    differential rate for a particular EFT response. Invoking multiple responses is hazardous since some responses interfere. Use rate_genNR for more complicated EFT scenarios.
    
    :param Er:
      This is a list of energies in keV.
      The list must be entered as a numpy array, np.array([..]).
      It can have as many entries as desired.
    :type Er: ``np.array``
    
    :param mass:
      Dark matter mass in GeV.
      Optional, default to 50.
    :type mass: ``float``
    
    :param sigma_*:
      dark-matter-proton scattering cross section normalized to give about 1 count at LUX when set to 1.
      * should be replaced by M, SigP, SigPP, PhiPP, Delta, MPhiPP, SigPDelta. (optional)
    :type sigma_*: ``float``
    
    :param fnfp_*:
      dimensionless ratio of neutron to proton coupling
      * should be replaced by M, SigP, SigPP, PhiPP, Delta, MPhiPP, SigPDelta. (optional)
    :type fnfp_*: ``float``
    
    :param v2co_*:
      coefficient of terms in the cross section that get multiplied by v^2
      * should be replaced by M, SigP, SigPP, PhiPP, Delta, MPhiPP, SigPDelta. (optional)
    :type v2co_*: ``float``
    
    :param q2co_*:
      coefficient of terms in the cross section that get multiplied by q^2/m_N^2
      * should be replaced by M, SigP, SigPP, PhiPP, Delta, MPhiPP, SigPDelta. (optional)
    :type q2co_*: ``float``
    
    :param v2q2co_*:
      coefficient of terms in the cross section that get multiplied by v^2*q^2/m_N^2
      * should be replaced by M, SigP, SigPP, PhiPP, Delta, MPhiPP, SigPDelta. (optional)
    :type v2q2co_*: ``float``
    
    :param q4co_*:
      coefficient of terms in the cross section that get multiplied by q^4/m_N^4
      * should be replaced by M, SigP, SigPP, PhiPP, Delta, MPhiPP, SigPDelta. (optional)
    :type q4co_*: ``float``

    :param v_lag:
      Velocity of the solar system with respect to the Milky Way in km/s.
      Optional, default to 220.
    :type v_lag: ``float``
    
    :param v_rms:
      1.5 * (velocity dispersion in km/s) of the local DM distribution.
      Optional, default to 220.
    :type v_rms: ``float``
    
    :param v_esc:
      Escape velocity in km/s of a DM particle in the galactic rest frame.
      Optional, default to 544.
    :type v_esc: ``float``
    
    :param element:
      Name of the detector element.
      Choice of:
        'argon',
        'fluorine',
        'germanium',
        'helium',
        'iodine',
        'sodium',
        'xenon'.
      *Optional*, default to 'xenon'
    :type element: ``str``
    
    :param rho_x:
      Local DM density in GeV/cm^3.
      Optional, default to 0.3
    :type rho_x: ``float``
    """
    
    cdef np.ndarray[DTYPE_t] sum
    sum = np.zeros(len(Q))
    if sigma_M!= 0.:
        sum += dRdQM(Q, v_rms, v_lag, v_esc, mass, sigma_M, stdco_M, v2co_M, q2co_M, v2q2co_M, q4co_M, fnfp_M, element, rho_x=rho_x)
    if sigma_SigPP!= 0.:
        sum += dRdQSigPP(Q, v_rms, v_lag, v_esc, mass, sigma_SigPP, stdco_SigPP, v2co_SigPP, q2co_SigPP, v2q2co_SigPP, q4co_SigPP, fnfp_SigPP, element, rho_x=rho_x)
    if sigma_SigP!= 0.:
        sum += dRdQSigP(Q, v_rms, v_lag, v_esc, mass, sigma_SigP, stdco_SigP, v2co_SigP, q2co_SigP, v2q2co_SigP, q4co_SigP, fnfp_SigP, element, rho_x=rho_x)
    if sigma_PhiPP!= 0.:
        sum += dRdQPhiPP(Q, v_rms, v_lag, v_esc, mass, sigma_PhiPP, v2co_PhiPP, q2co_PhiPP, v2q2co_PhiPP, q4co_PhiPP, fnfp_PhiPP, element, rho_x=rho_x)
    if sigma_Delta!= 0.:
        sum += dRdQDelta(Q, v_rms, v_lag, v_esc, mass, sigma_Delta, v2co_Delta, q2co_Delta, v2q2co_Delta, q4co_Delta, fnfp_Delta, element, rho_x=rho_x)
    if sigma_MPhiPP!= 0.:
        sum += dRdQMPhiPP(Q, v_rms, v_lag, v_esc, mass, sigma_MPhiPP, v2co_MPhiPP, q2co_MPhiPP, v2q2co_MPhiPP, q4co_MPhiPP, fnfp_MPhiPP, element, rho_x=rho_x)
    if sigma_SigPDelta!= 0.:
        sum += dRdQSigPDelta(Q, v_rms, v_lag, v_esc, mass, sigma_SigPDelta, v2co_SigPDelta, q2co_SigPDelta, v2q2co_SigPDelta, q4co_SigPDelta, fnfp_SigPDelta, element, rho_x=rho_x)
    return sum
    


@cython.boundscheck(False)
def R(object efficiency_fn, DTYPE_t mass=50.,
         DTYPE_t sigma_M=0., DTYPE_t sigma_SigPP=0., DTYPE_t sigma_SigP=0., DTYPE_t sigma_PhiPP=0., DTYPE_t sigma_Delta=0., DTYPE_t sigma_MPhiPP=0., DTYPE_t sigma_SigPDelta=0.,
         DTYPE_t fnfp_M=1., DTYPE_t fnfp_SigPP=1., DTYPE_t fnfp_SigP=1., DTYPE_t fnfp_PhiPP=1., DTYPE_t fnfp_Delta=1., DTYPE_t fnfp_MPhiPP=1., DTYPE_t fnfp_SigPDelta=1.,
         DTYPE_t stdco_M=1., DTYPE_t stdco_SigPP=1., DTYPE_t stdco_SigP=1.,
         DTYPE_t q2co_M=0., DTYPE_t q2co_SigPP=0., DTYPE_t q2co_SigP=0., DTYPE_t q2co_PhiPP=0., DTYPE_t q2co_Delta=0., DTYPE_t q2co_MPhiPP=0., DTYPE_t q2co_SigPDelta=0.,
         DTYPE_t q4co_M=0., DTYPE_t q4co_SigPP=0., DTYPE_t q4co_SigP=0., DTYPE_t q4co_PhiPP=0., DTYPE_t q4co_Delta=0., DTYPE_t q4co_MPhiPP=0., DTYPE_t q4co_SigPDelta=0.,
         DTYPE_t v2co_M=0., DTYPE_t v2co_SigPP=0., DTYPE_t v2co_SigP=0., DTYPE_t v2co_PhiPP=0., DTYPE_t v2co_Delta=0., DTYPE_t v2co_MPhiPP=0., DTYPE_t v2co_SigPDelta=0.,
         DTYPE_t v2q2co_M=0., DTYPE_t v2q2co_SigPP=0., DTYPE_t v2q2co_SigP=0., DTYPE_t v2q2co_PhiPP=0., DTYPE_t v2q2co_Delta=0., DTYPE_t v2q2co_MPhiPP=0., DTYPE_t v2q2co_SigPDelta=0.,
         DTYPE_t v_lag=220., DTYPE_t v_rms=220., DTYPE_t v_esc=544., DTYPE_t rho_x=0.3,
         str element='xenon', DTYPE_t Qmin=2., DTYPE_t Qmax=30.):
    """
    Theoretical total integrated recoil-energy rate.

    Integrates :func:`dRdQ` between ``Qmin`` and ``Qmax`` using
    trapezoidal integration over 100 points.

    :param efficiency_fn:
      Fractional efficiency as a function of energy.
      Efficiencies available:
        dmdd.eff.efficiency_Ar,
        dmdd.eff.efficiency_Ge,
        dmdd.eff.efficiency_I,
        dmdd.eff.efficiency_KIMS,
        dmdd.eff.efficiency_Xe,
      right now, all of these are set to be constant and equal to 1.
    :type efficiency: ``object``

    :param Qmin:
        Nuclear-recoil energy threshold of experiment [keVNR]. *Optional*, default 2.
    :type Qmin: ``float``

    :param Qmax:
        Upper bound of nuclear-recoil energy window of experiment [keVNR]. *Optional*, default 30.
    :type Qmax: ``float``
    
    For reference on other parameters, see :func:`dRdQ`.
      

    :return:
      total recoil energy rate in counts/kg/sec
    """
    cdef unsigned int npoints = 100 
    cdef unsigned int i
    cdef DTYPE_t result
    cdef DTYPE_t expQmin = log10(Qmin)
    cdef DTYPE_t expQmax = log10(Qmax)
    cdef DTYPE_t expQstep = (expQmax - expQmin)/(npoints - 1)
    cdef np.ndarray[DTYPE_t] Qs = np.empty(npoints,dtype=float)

    for i in xrange(npoints):
        expQ = expQmin + i*expQstep
        Qs[i] = 10**expQ
        
    cdef np.ndarray[DTYPE_t] dRdQs = dRdQ(Qs, mass=mass,
                                          v_lag=v_lag, v_rms=v_rms, v_esc= v_esc, rho_x=rho_x,
                                          element=element,
                                          sigma_M=sigma_M, sigma_SigPP=sigma_SigPP, sigma_SigP=sigma_SigP, sigma_PhiPP=sigma_PhiPP, sigma_Delta=sigma_Delta, sigma_MPhiPP=sigma_MPhiPP, sigma_SigPDelta=sigma_SigPDelta,
          fnfp_M=fnfp_M,  fnfp_SigPP=fnfp_SigPP,  fnfp_SigP=fnfp_SigP,  fnfp_PhiPP=fnfp_PhiPP,  fnfp_Delta=fnfp_Delta,  fnfp_MPhiPP=fnfp_MPhiPP,  fnfp_SigPDelta=fnfp_SigPDelta,
          stdco_M=stdco_M,  stdco_SigPP=stdco_SigPP,  stdco_SigP=stdco_SigP,
          q2co_M=q2co_M,  q2co_SigPP=q2co_SigPP,  q2co_SigP=q2co_SigP,  q2co_PhiPP=q2co_PhiPP,  q2co_Delta=q2co_Delta,  q2co_MPhiPP=q2co_MPhiPP,  q2co_SigPDelta=q2co_SigPDelta,
          q4co_M=q4co_M,  q4co_SigPP=q4co_SigPP,  q4co_SigP=q4co_SigP,  q4co_PhiPP=q4co_PhiPP,  q4co_Delta=q4co_Delta,  q4co_MPhiPP=q4co_MPhiPP,  q4co_SigPDelta=q4co_SigPDelta,
          v2co_M=v2co_M,  v2co_SigPP=v2co_SigPP,  v2co_SigP=v2co_SigP,  v2co_PhiPP=v2co_PhiPP,  v2co_Delta=v2co_Delta,  v2co_MPhiPP=v2co_MPhiPP,  v2co_SigPDelta=v2co_SigPDelta,
          v2q2co_M=v2q2co_M,  v2q2co_SigPP=v2q2co_SigPP,  v2q2co_SigP=v2q2co_SigP,  v2q2co_PhiPP=v2q2co_PhiPP,  v2q2co_Delta=v2q2co_Delta,  v2q2co_MPhiPP=v2q2co_MPhiPP,  v2q2co_SigPDelta=v2q2co_SigPDelta) * efficiency_fn(Qs)
    result = trapz(dRdQs,Qs)
    return result



@cython.boundscheck(False)
def loglikelihood(np.ndarray[DTYPE_t] Q, object efficiency_fn, DTYPE_t mass=50.,
         DTYPE_t sigma_M=0., DTYPE_t sigma_SigPP=0., DTYPE_t sigma_SigP=0., DTYPE_t sigma_PhiPP=0., DTYPE_t sigma_Delta=0., DTYPE_t sigma_MPhiPP=0., DTYPE_t sigma_SigPDelta=0.,
         DTYPE_t fnfp_M=1., DTYPE_t fnfp_SigPP=1., DTYPE_t fnfp_SigP=1., DTYPE_t fnfp_PhiPP=1., DTYPE_t fnfp_Delta=1., DTYPE_t fnfp_MPhiPP=1., DTYPE_t fnfp_SigPDelta=1.,
         DTYPE_t stdco_M=1., DTYPE_t stdco_SigPP=1., DTYPE_t stdco_SigP=1.,
         DTYPE_t q2co_M=0., DTYPE_t q2co_SigPP=0., DTYPE_t q2co_SigP=0., DTYPE_t q2co_PhiPP=0., DTYPE_t q2co_Delta=0., DTYPE_t q2co_MPhiPP=0., DTYPE_t q2co_SigPDelta=0.,
         DTYPE_t q4co_M=0., DTYPE_t q4co_SigPP=0., DTYPE_t q4co_SigP=0., DTYPE_t q4co_PhiPP=0., DTYPE_t q4co_Delta=0., DTYPE_t q4co_MPhiPP=0., DTYPE_t q4co_SigPDelta=0.,
         DTYPE_t v2co_M=0., DTYPE_t v2co_SigPP=0., DTYPE_t v2co_SigP=0., DTYPE_t v2co_PhiPP=0., DTYPE_t v2co_Delta=0., DTYPE_t v2co_MPhiPP=0., DTYPE_t v2co_SigPDelta=0.,
         DTYPE_t v2q2co_M=0., DTYPE_t v2q2co_SigPP=0., DTYPE_t v2q2co_SigP=0., DTYPE_t v2q2co_PhiPP=0., DTYPE_t v2q2co_Delta=0., DTYPE_t v2q2co_MPhiPP=0., DTYPE_t v2q2co_SigPDelta=0.,
                  DTYPE_t v_lag=220., DTYPE_t v_rms=220., DTYPE_t v_esc=544., DTYPE_t rho_x=0.3,
                  str element='xenon', DTYPE_t Qmin=2., DTYPE_t Qmax=30., DTYPE_t exposure=1., energy_resolution=True):
    """
    This is the main log(likelihood) for any combination of EFT responses.
    
    For reference on free parameters, see :func:`dRdQ` and :func:`R`.
      

    :return:
      log(likelihood) for an arbitrary rate to produce an observed array of recoil events
    """
    cdef unsigned int i
    cdef long Nevents = len(Q)
    cdef np.ndarray[DTYPE_t] out
    cdef DTYPE_t Nexp
    cdef DTYPE_t tot = 0.
    cdef DTYPE_t Tobs = exposure * 24. * 3600. * 365.
   
    cdef DTYPE_t Rate = R(efficiency_fn, mass=mass, v_rms=v_rms, v_lag=v_lag, v_esc=v_esc, rho_x=rho_x,
        sigma_M=sigma_M, sigma_SigPP=sigma_SigPP, sigma_SigP=sigma_SigP, sigma_PhiPP=sigma_PhiPP, sigma_Delta=sigma_Delta, sigma_MPhiPP=sigma_MPhiPP, sigma_SigPDelta=sigma_SigPDelta,
          fnfp_M=fnfp_M,  fnfp_SigPP=fnfp_SigPP,  fnfp_SigP=fnfp_SigP,  fnfp_PhiPP=fnfp_PhiPP,  fnfp_Delta=fnfp_Delta,  fnfp_MPhiPP=fnfp_MPhiPP,  fnfp_SigPDelta=fnfp_SigPDelta,
          stdco_M=stdco_M,  stdco_SigPP=stdco_SigPP,  stdco_SigP=stdco_SigP,
          q2co_M=q2co_M,  q2co_SigPP=q2co_SigPP,  q2co_SigP=q2co_SigP,  q2co_PhiPP=q2co_PhiPP,  q2co_Delta=q2co_Delta,  q2co_MPhiPP=q2co_MPhiPP,  q2co_SigPDelta=q2co_SigPDelta,
          q4co_M=q4co_M,  q4co_SigPP=q4co_SigPP,  q4co_SigP=q4co_SigP,  q4co_PhiPP=q4co_PhiPP,  q4co_Delta=q4co_Delta,  q4co_MPhiPP=q4co_MPhiPP,  q4co_SigPDelta=q4co_SigPDelta,
          v2co_M=v2co_M,  v2co_SigPP=v2co_SigPP,  v2co_SigP=v2co_SigP,  v2co_PhiPP=v2co_PhiPP,  v2co_Delta=v2co_Delta,  v2co_MPhiPP=v2co_MPhiPP,  v2co_SigPDelta=v2co_SigPDelta,
          v2q2co_M=v2q2co_M,  v2q2co_SigPP=v2q2co_SigPP,  v2q2co_SigP=v2q2co_SigP,  v2q2co_PhiPP=v2q2co_PhiPP,  v2q2co_Delta=v2q2co_Delta,  v2q2co_MPhiPP=v2q2co_MPhiPP,  v2q2co_SigPDelta=v2q2co_SigPDelta,
          Qmin=Qmin, Qmax=Qmax, element=element)

    Nexp = Rate * Tobs
    if Nevents==0 and Nexp==0.:
        return 0.
    tot += Nevents * log(Nexp) - Nexp 
    if energy_resolution:
        tot -= Nevents * log(Rate) 
        out = dRdQ(Q, mass=mass, v_lag=v_lag, v_rms=v_rms, v_esc= v_esc, rho_x=rho_x, element=element,
                                        sigma_M=sigma_M, sigma_SigPP=sigma_SigPP, sigma_SigP=sigma_SigP, sigma_PhiPP=sigma_PhiPP, sigma_Delta=sigma_Delta, sigma_MPhiPP=sigma_MPhiPP, sigma_SigPDelta=sigma_SigPDelta,
          fnfp_M=fnfp_M,  fnfp_SigPP=fnfp_SigPP,  fnfp_SigP=fnfp_SigP,  fnfp_PhiPP=fnfp_PhiPP,  fnfp_Delta=fnfp_Delta,  fnfp_MPhiPP=fnfp_MPhiPP,  fnfp_SigPDelta=fnfp_SigPDelta,
          stdco_M=stdco_M,  stdco_SigPP=stdco_SigPP,  stdco_SigP=stdco_SigP,
          q2co_M=q2co_M,  q2co_SigPP=q2co_SigPP,  q2co_SigP=q2co_SigP,  q2co_PhiPP=q2co_PhiPP,  q2co_Delta=q2co_Delta,  q2co_MPhiPP=q2co_MPhiPP,  q2co_SigPDelta=q2co_SigPDelta,
          q4co_M=q4co_M,  q4co_SigPP=q4co_SigPP,  q4co_SigP=q4co_SigP,  q4co_PhiPP=q4co_PhiPP,  q4co_Delta=q4co_Delta,  q4co_MPhiPP=q4co_MPhiPP,  q4co_SigPDelta=q4co_SigPDelta,
          v2co_M=v2co_M,  v2co_SigPP=v2co_SigPP,  v2co_SigP=v2co_SigP,  v2co_PhiPP=v2co_PhiPP,  v2co_Delta=v2co_Delta,  v2co_MPhiPP=v2co_MPhiPP,  v2co_SigPDelta=v2co_SigPDelta,
          v2q2co_M=v2q2co_M,  v2q2co_SigPP=v2q2co_SigPP,  v2q2co_SigP=v2q2co_SigP,  v2q2co_PhiPP=v2q2co_PhiPP,  v2q2co_Delta=v2q2co_Delta,  v2q2co_MPhiPP=v2q2co_MPhiPP,  v2q2co_SigPDelta=v2q2co_SigPDelta) * efficiency_fn(Q)
        
    
        for i in range(Nevents):
            if out[i]==0.:
                return -1.*INFINITY #if an event is seen where the model expects zero events (behind the V_lag), this model is excluded, and loglikelihood=-Infinity.
            tot += log(out[i])
            
    return tot
