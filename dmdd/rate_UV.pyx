import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool

import math
import formUV as formUV
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
cdef DTYPE_t ratenorm = 1.68288e31 # conversion from cm**-1 * GeV**-2 to DRU = cts / keV / kg / sec

#physical constants from constants.py:
cdef DTYPE_t mN = const.NUCLEON_MASS # Nucleon mass in GeV
cdef DTYPE_t pmag = const.P_MAGMOM # proton magnetic moment, PDG Live
cdef DTYPE_t nmag = const.N_MAGMOM # neutron magnetic moment, PDG Live

#information about target nuclei:
eltshort = const.ELEMENT_INFO


############################# rates dR/dQ for each operator:

#spin-independent
@cython.boundscheck(False)
def dRdQSI(np.ndarray[DTYPE_t] Er, DTYPE_t V0, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from the spin-independent (scalar-scalar) scattering cross section in units of cts/keV/kg/s.

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
    :type elt: ``str``

    :param rho_x:
        Local dark matter density in GeV/cm^3. Optional, default 0.3
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
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_si'] / (2. * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        ff = formUV.factor_SI(element_name,y_harm,fnfp)
        val_eta = eta(v_min,v_esc,V0,v_lag)
        tot = v_independent * val_eta * ff
        out[i] = tot
    return out

@cython.boundscheck(False)
def dRdQSI_massless(np.ndarray[DTYPE_t] Er, DTYPE_t V0, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from the spin-independent (scalar-scalar) scattering cross section *with massless mediator* in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQSI` 
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, weight, qref, q_squared, v_independent
    cdef DTYPE_t ff

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    qref = 0.1
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_si_massless'] / (2. * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        q_squared = 2.*weight*mN*q
        ff = formUV.factor_SI(element_name,y_harm,fnfp)
        val_eta = eta(v_min,v_esc,V0,v_lag)
        tot = v_independent * (qref**2./q_squared)**2. * val_eta * ff
        out[i] = tot
    return out

#spin-dependent
@cython.boundscheck(False)
def dRdQSD(np.ndarray[DTYPE_t] Er, DTYPE_t V0, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from the spin-dependent (axial-axial) scattering cross section in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQSI`.
    
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
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_sd'] / (2 * mx * 3 * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        ff = formUV.factor_SD(element_name,y_harm,fnfp)
        val_eta = eta(v_min,v_esc,V0,v_lag)
        tot = v_independent * val_eta * ff
        out[i]=tot
    return out

#spin-dependent
@cython.boundscheck(False)
def dRdQSD_massless(np.ndarray[DTYPE_t] Er, DTYPE_t V0, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from the spin-dependent (axial-axial) scattering cross section *with massless mediator* in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQSI`.
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, weight, qref, q_squared, v_independent
    cdef DTYPE_t ff

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    qref = 0.1
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_sd_massless'] / (2 * mx * 3 * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        q_squared = 2.*weight*mN*q
        ff = formUV.factor_SD(element_name,y_harm,fnfp)
        val_eta = eta(v_min,v_esc,V0,v_lag)
        tot = v_independent * (qref**2./q_squared)**2. * val_eta * ff
        out[i]=tot
    return out

#anapole - assuming that the DM is spin 1/2 (relevant for the factor C_\chi)
@cython.boundscheck(False)
def dRdQana(np.ndarray[DTYPE_t] Er, DTYPE_t V0, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from the anapole moment scattering cross section in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQSI`.
    
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, val_zeta, weight, v_independent
    cdef DTYPE_t ff_v_sq, ff_v_std

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_anapole'] / (2 * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        ff_v_std = formUV.factor_anapole_v_std(element_name,y_harm,mx,b_harm)
        ff_v_sq = formUV.factor_anapole_v_sq(element_name,y_harm)
        val_eta = eta(v_min,v_esc,V0,v_lag)
        val_zeta = zeta(v_min,v_esc,V0,v_lag)
        tot = v_independent * (val_zeta * ff_v_sq + val_eta * ff_v_std)
        out[i]=tot
    return out

#anapole - assuming that the DM is spin 1/2 (relevant for the factor C_\chi)
@cython.boundscheck(False)
def dRdQana_massless(np.ndarray[DTYPE_t] Er, DTYPE_t V0, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from the anapole moment *with massless mediator* scattering cross section in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQSI`.
    
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, val_zeta, weight, qref, q_squared, v_independent
    cdef DTYPE_t ff_v_sq, ff_v_std

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    qref = 0.1
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_anapole_massless'] / (2 * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        q_squared = 2.*weight*mN*q
        ff_v_std = formUV.factor_anapole_v_std(element_name,y_harm,mx,b_harm)
        ff_v_sq = formUV.factor_anapole_v_sq(element_name,y_harm)
        val_eta = eta(v_min,v_esc,V0,v_lag)
        val_zeta = zeta(v_min,v_esc,V0,v_lag)
        tot = v_independent * (qref**2./q_squared)**2. * (val_zeta * ff_v_sq + val_eta * ff_v_std)
        out[i]=tot
    return out

#magnetic dipole - assuming that the DM is spin 1/2 (relevant for the factor C_\chi)
@cython.boundscheck(False)
def dRdQmagdip(np.ndarray[DTYPE_t] Er, DTYPE_t V0, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from the magnetic dipole moment scattering cross section in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQSI`.
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, val_zeta, weight, scale, qref, q_squared, v_independent
    cdef DTYPE_t ff_v_sq, ff_v_std

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    scale = 1.
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_magdip'] / (2 * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        q_squared = 2.*weight*mN*q
        ff_v_std = formUV.factor_magdip_v_std(element_name,y_harm,mx,b_harm)
        ff_v_sq = formUV.factor_magdip_v_sq(element_name,y_harm)
        val_eta = eta(v_min,v_esc,V0,v_lag)
        val_zeta = zeta(v_min,v_esc,V0,v_lag)
        tot = v_independent * q_squared/(scale**2) * ( val_zeta * ff_v_sq +  val_eta * ff_v_std )
        out[i]=tot
    return out

#magnetic dipole - assuming that the DM is spin 1/2 (relevant for the factor C_\chi)
@cython.boundscheck(False)
def dRdQmagdip_massless(np.ndarray[DTYPE_t] Er, DTYPE_t V0, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from the magnetic dipole moment *with massless mediator* scattering cross section in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQSI`.
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, val_zeta, weight, scale, qref, q_squared, v_independent
    cdef DTYPE_t ff_v_sq, ff_v_std

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    scale = 1.
    qref = 0.1
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_magdip_massless'] / (2 * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        q_squared = 2.*weight*mN*q
        ff_v_std = formUV.factor_magdip_v_std(element_name,y_harm,mx,b_harm)
        ff_v_sq = formUV.factor_magdip_v_sq(element_name,y_harm)
        val_eta = eta(v_min,v_esc,V0,v_lag)
        val_zeta = zeta(v_min,v_esc,V0,v_lag)
        tot = v_independent * qref**4./(scale**2.*q_squared) * ( val_zeta * ff_v_sq +  val_eta * ff_v_std )
        out[i]=tot
    return out

#electric dipole - assuming that the DM is spin 1/2 (relevant for the factor C_\chi)
@cython.boundscheck(False)
def dRdQelecdip(np.ndarray[DTYPE_t] Er, DTYPE_t V0, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the the rate from electric dipole moment scattering cross section in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQSI`.
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, weight, scale, v_independent
    cdef DTYPE_t ff

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    scale = 1.
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_elecdip'] / (2 * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        q_squared = 2.*weight*mN*q
        ff = formUV.factor_elecdip(element_name,y_harm)
        val_eta = eta(v_min,v_esc,V0,v_lag)
        tot = v_independent * q_squared/(scale**2) * val_eta * ff
        out[i]=tot
    return out

#electric dipole - assuming that the DM is spin 1/2 (relevant for the factor C_\chi)
@cython.boundscheck(False)
def dRdQelecdip_massless(np.ndarray[DTYPE_t] Er, DTYPE_t V0, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the the rate from electric dipole moment *with massless mediator* scattering cross section in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQSI`.
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, weight, scale, qref, q_squared, v_independent
    cdef DTYPE_t ff

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    scale = 1.
    qref = 0.1
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_elecdip_massless'] / (2 * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        q_squared = 2.*weight*mN*q
        ff = formUV.factor_elecdip(element_name,y_harm)
        val_eta = eta(v_min,v_esc,V0,v_lag)
        tot = v_independent * qref**4./(scale**2.*q_squared) * val_eta * ff
        out[i]=tot
    return out

#LS generating - assuming that the DM is spin 1/2 (relevant for the factor C_\chi)
@cython.boundscheck(False)
def dRdQLS(np.ndarray[DTYPE_t] Er, DTYPE_t V0, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the the rate from a :L dot S generating: scattering cross section in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQSI`.
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, weight, q_squared, v_independent
    cdef DTYPE_t ff

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_LS'] / (2 * mx * m_reduced_sq)
    qref = 0.1
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        q_squared = 2.*weight*mN*q
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].    
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        ff_vstd = formUV.factor_LS_vstd(element_name,y_harm,fnfp,mx,b_harm)
        ff_vsq = formUV.factor_LS_vsq(element_name,y_harm,fnfp)
        val_eta = eta(v_min,v_esc,V0,v_lag)
        val_zeta = zeta(v_min,v_esc,V0,v_lag)
        tot = v_independent * q_squared/(qref**2.) * ( val_eta * ff_vstd + val_zeta * ff_vsq )
        out[i]=tot
    return out

#LS generating - assuming that the DM is spin 1/2 (relevant for the factor C_\chi)
@cython.boundscheck(False)
def dRdQLS_massless(np.ndarray[DTYPE_t] Er, DTYPE_t V0, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the the rate from a :L dot S generating: *with massless mediator* scattering cross section in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQSI`.
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, weight, qref, v_independent
    cdef DTYPE_t ff

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    qref = 0.1
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_LS_massless'] / (2 * mx * m_reduced_sq) * (qref**2./(mN**2))**2
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].    
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        ff = formUV.factor_LS(element_name,y_harm,fnfp,mx)
        val_eta = eta(v_min,v_esc,V0,v_lag)
        tot = v_independent * val_eta * ff
        out[i]=tot
    return out

#pseudoscalar f1 (PS-S)
@cython.boundscheck(False)
def dRdQf1(np.ndarray[DTYPE_t] Er, DTYPE_t v_rms, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from a pseudoscalar-scalar (CP-odd coupling to DM, CP-even coupling to SM) cross section in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQSI`.
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, weight, qref, q_squared, v_independent
    cdef DTYPE_t ff

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    qref = 0.1
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_f1'] / (2 * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        q_squared = 2.*weight*mN*q
        ff = formUV.factor_PS_S(element_name,y_harm,fnfp)
        val_eta = eta(v_min,v_esc,v_rms,v_lag)
        tot = v_independent * q_squared/qref**2 * val_eta * ff
        out[i]=tot
    return out

#pseudoscalar f1 (PS-S)
@cython.boundscheck(False)
def dRdQf1_massless(np.ndarray[DTYPE_t] Er, DTYPE_t v_rms, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from a pseudoscalar-scalar (CP-odd coupling to DM, CP-even coupling to SM) *with massless mediator* cross section in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQSI`.
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, weight, qref, q_squared, v_independent
    cdef DTYPE_t ff

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    qref = 0.1
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_f1_massless'] / (2 * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        q_squared = 2.*weight*mN*q
        ff = formUV.factor_PS_S(element_name,y_harm,fnfp)
        val_eta = eta(v_min,v_esc,v_rms,v_lag)
        tot = v_independent * (qref**2./q_squared) * val_eta * ff
        out[i]=tot
    return out

#pseudoscalar f2 (S-PS)
@cython.boundscheck(False)
def dRdQf2(np.ndarray[DTYPE_t] Er, DTYPE_t v_rms, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from a scalar-pseudoscalar (CP-even coupling to DM, CP-odd coupling to SM) cross section in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQSI`.
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, weight, qref, q_squared, v_independent
    cdef DTYPE_t ff

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    qref= 0.1
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_f2'] / (2 * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        q_squared = 2.*weight*mN*q
        ff = formUV.factor_S_PS(element_name,y_harm,fnfp)
        val_eta = eta(v_min,v_esc,v_rms,v_lag)
        tot = v_independent * q_squared/qref**2 * val_eta * ff
        out[i]=tot
    return out

#pseudoscalar f2 (S-PS)
@cython.boundscheck(False)
def dRdQf2_massless(np.ndarray[DTYPE_t] Er, DTYPE_t v_rms, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from a scalar-pseudoscalar (CP-even coupling to DM, CP-odd coupling to SM) *with massless mediator* cross section in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQSI`.
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, weight, qref, q_squared, v_independent
    cdef DTYPE_t ff

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    qref= 0.1
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_f2_massless'] / (2 * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        q_squared = 2.*weight*mN*q
        ff = formUV.factor_S_PS(element_name,y_harm,fnfp)
        val_eta = eta(v_min,v_esc,v_rms,v_lag)
        tot = v_independent * qref**2./q_squared * val_eta * ff
        out[i]=tot
    return out

#pseudoscalar f3 (PS-PS)
@cython.boundscheck(False)
def dRdQf3(np.ndarray[DTYPE_t] Er, DTYPE_t v_rms, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from a pseudoscalar-pseudoscalar (CP-odd coupling to DM and to SM) cross section in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQSI`.
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(elt.title())

    cdef DTYPE_t q, tot, m_reduced_sq, y_harm, b_harm, v_min, val_eta, weight, qref, q_squared, v_independent
    cdef DTYPE_t ff

    weight = eltshort[elt]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    qref = 0.1
    m_reduced_sq = mx**2.*mN**2./(mx+mN)**2.
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_f3'] / (2 * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        q_squared = 2.*weight*mN*q
        ff = formUV.factor_PS_PS(element_name,y_harm,fnfp)
        val_eta = eta(v_min,v_esc,v_rms,v_lag)
        tot = v_independent * (q_squared/(qref**2.))**2. * val_eta * ff
        out[i]=tot
    return out

#pseudoscalar f3 (PS-PS)
@cython.boundscheck(False)
def dRdQf3_massless(np.ndarray[DTYPE_t] Er, DTYPE_t v_rms, DTYPE_t v_lag, DTYPE_t v_esc, DTYPE_t mx, DTYPE_t sigp, DTYPE_t fnfp, str elt, DTYPE_t rho_x=0.3):
    """
    This is the rate from a pseudoscalar-pseudoscalar (CP-odd coupling to DM and to SM) *with massless mediator* cross section in units of cts/keV/kg/s.

    Takes same parameters as :func:`dRdQSI`.
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
    v_independent = ratenorm * rho_x * sigp * PAR_NORMS['sigma_f3_massless'] / (2 * mx * m_reduced_sq)
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mx/(weight*mN+mx)) *3.*10.**5
        ff = formUV.factor_PS_PS(element_name,y_harm,fnfp)
        val_eta = eta(v_min,v_esc,v_rms,v_lag)
        tot = v_independent * val_eta * ff
        out[i]=tot
    return out

#######################  "Master" rate functions. 

@cython.boundscheck(False)
def dRdQ(np.ndarray[DTYPE_t] Q, DTYPE_t mass=50.,
         DTYPE_t sigma_si=0.,DTYPE_t sigma_sd=0.,
         DTYPE_t sigma_anapole=0.,DTYPE_t sigma_magdip=0., DTYPE_t sigma_elecdip=0.,
         DTYPE_t sigma_LS=0., DTYPE_t sigma_f1=0., DTYPE_t sigma_f2=0., DTYPE_t sigma_f3=0.,
         DTYPE_t sigma_si_massless=0.,DTYPE_t sigma_sd_massless=0.,
         DTYPE_t sigma_anapole_massless=0.,DTYPE_t sigma_magdip_massless=0., DTYPE_t sigma_elecdip_massless=0.,
         DTYPE_t sigma_LS_massless=0., DTYPE_t sigma_f1_massless=0., DTYPE_t sigma_f2_massless=0., DTYPE_t sigma_f3_massless=0.,
         DTYPE_t fnfp_si=1., DTYPE_t fnfp_sd=1.,
         DTYPE_t fnfp_anapole=1., DTYPE_t fnfp_magdip=1., DTYPE_t fnfp_elecdip=1.,
         DTYPE_t fnfp_LS=1., DTYPE_t fnfp_f1=1., DTYPE_t fnfp_f2=1., DTYPE_t fnfp_f3=1.,
         DTYPE_t fnfp_si_massless=0., DTYPE_t fnfp_sd_massless=1.,
         DTYPE_t fnfp_anapole_massless=1., DTYPE_t fnfp_magdip_massless=1., DTYPE_t fnfp_elecdip_massless=1.,
         DTYPE_t fnfp_LS_massless=1., DTYPE_t fnfp_f1_massless=1., DTYPE_t fnfp_f2_massless=1., DTYPE_t fnfp_f3_massless=1.,
         DTYPE_t v_lag=220., DTYPE_t v_rms=220., DTYPE_t v_esc=544., DTYPE_t rho_x=0.3,
         str element='xenon'):
    """
    This is the main differential nuclear-recoil rate function. Its output (in units of cts/keV/kg/s) is computed for any one of 28 different scattering operators, by setting the appropriate ``sigma_*`` parameter to a non-zero value.

    
    :param Q:
        Nuclear recoil energies [keV]
    :type Q: ``ndarray``

    :param mass:
        Dark-matter particle mass [GeV]. *Optional*, default 50.
    :type mass: ``float``

    :param sigma_*:
        Various scattering cross sections [cm^2] off a **proton**. The symbol * should be replaced with a suffix from the list below. The value passed will be multiplied with a normalization factor given in ``dmdd.PAR_NORMS``. See explanation of suffixes and values of ``PAR_NORMS`` below. *Optional*, default all 0.
    :type sigma_*: ``float``

    :param fnfp_*:
        Dimensionless ratio of neutron to proton coupling. *Optional*, default all 1.
    :type fnfp_*: ``float``

    :param v_lag:
        Lag velocity [km/s]. *Optional*, default 220.
    :type v_lag: ``float``

    :param v_rms:
        RMS velocity [km/s]. Note that ``v_rms`` is 3/2x the standard RMS of a Maxwellian velocity distribution; that is, the default ``v_rms`` value = 220.
    :type v_rms: ``float``

    :param v_esc:
        Escape velocity [km/s]. *Optional*, default 544.
    :type v_esc: ``float``

    :param rho_x:
        Local dark matter mass density [GeV/cm^3]. *Optional*, default 0.3
    :type rho_x: ``float``

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

    In all suffixes, the mediator is specified to be "massless" by appending _massless.

      

    :return: 
      array of differential recoil energy spectrum in counts/keV/kg/sec

    """
    
    cdef np.ndarray[DTYPE_t] sum
    sum = np.zeros(len(Q))
    if sigma_si!= 0.:
        sum += dRdQSI(Q, v_rms, v_lag, v_esc, mass, sigma_si, fnfp_si, element, rho_x=rho_x)
    if sigma_sd!= 0.:
        sum += dRdQSD(Q, v_rms, v_lag, v_esc, mass, sigma_sd, fnfp_sd, element, rho_x=rho_x)
    if sigma_anapole!= 0.:
        sum += dRdQana(Q, v_rms, v_lag, v_esc, mass, sigma_anapole, fnfp_anapole, element, rho_x=rho_x)
    if sigma_magdip!= 0.:
        sum += dRdQmagdip(Q, v_rms, v_lag, v_esc, mass, sigma_magdip, fnfp_magdip, element, rho_x=rho_x)
    if sigma_elecdip!= 0.:
        sum += dRdQelecdip(Q, v_rms, v_lag, v_esc, mass, sigma_elecdip, fnfp_elecdip, element, rho_x=rho_x)
    if sigma_LS!= 0.:
        sum += dRdQLS(Q, v_rms, v_lag, v_esc, mass, sigma_LS, fnfp_LS, element, rho_x=rho_x)
    if sigma_f1!= 0.:
        sum += dRdQf1(Q, v_rms, v_lag, v_esc, mass, sigma_f1, fnfp_f1, element, rho_x=rho_x)
    if sigma_f2!= 0.:
        sum += dRdQf2(Q, v_rms, v_lag, v_esc, mass, sigma_f2, fnfp_f2, element, rho_x=rho_x)
    if sigma_f3!= 0.:
        sum += dRdQf3(Q, v_rms, v_lag, v_esc, mass, sigma_f3, fnfp_f3, element, rho_x=rho_x)
    if sigma_si_massless!= 0.:
        sum += dRdQSI_massless(Q, v_rms, v_lag, v_esc, mass, sigma_si_massless, fnfp_si_massless, element, rho_x=rho_x)
    if sigma_sd_massless!= 0.:
        sum += dRdQSD_massless(Q, v_rms, v_lag, v_esc, mass, sigma_sd_massless, fnfp_sd_massless, element, rho_x=rho_x)
    if sigma_anapole_massless!= 0.:
        sum += dRdQana_massless(Q, v_rms, v_lag, v_esc, mass, sigma_anapole_massless, fnfp_anapole_massless, element, rho_x=rho_x)
    if sigma_magdip_massless!= 0.:
        sum += dRdQmagdip_massless(Q, v_rms, v_lag, v_esc, mass, sigma_magdip_massless, fnfp_magdip_massless, element, rho_x=rho_x)
    if sigma_elecdip_massless!= 0.:
        sum += dRdQelecdip_massless(Q, v_rms, v_lag, v_esc, mass, sigma_elecdip_massless, fnfp_elecdip_massless, element, rho_x=rho_x)
    if sigma_LS_massless!= 0.:
        sum += dRdQLS_massless(Q, v_rms, v_lag, v_esc, mass, sigma_LS_massless, fnfp_LS_massless, element, rho_x=rho_x)
    if sigma_f1_massless!= 0.:
        sum += dRdQf1_massless(Q, v_rms, v_lag, v_esc, mass, sigma_f1_massless, fnfp_f1_massless, element, rho_x=rho_x)
    if sigma_f2_massless!= 0.:
        sum += dRdQf2_massless(Q, v_rms, v_lag, v_esc, mass, sigma_f2_massless, fnfp_f2_massless, element, rho_x=rho_x)
    if sigma_f3_massless!= 0.:
        sum += dRdQf3_massless(Q, v_rms, v_lag, v_esc, mass, sigma_f3_massless, fnfp_f3_massless, element, rho_x=rho_x)
    return sum
    


@cython.boundscheck(False)
def R(object efficiency_fn, DTYPE_t mass=50.,
         DTYPE_t sigma_si=0.,DTYPE_t sigma_sd=0.,
         DTYPE_t sigma_anapole=0.,DTYPE_t sigma_magdip=0., DTYPE_t sigma_elecdip=0.,
         DTYPE_t sigma_LS=0., DTYPE_t sigma_f1=0., DTYPE_t sigma_f2=0., DTYPE_t sigma_f3=0.,
         DTYPE_t sigma_si_massless=0.,DTYPE_t sigma_sd_massless=0.,
         DTYPE_t sigma_anapole_massless=0.,DTYPE_t sigma_magdip_massless=0., DTYPE_t sigma_elecdip_massless=0.,
         DTYPE_t sigma_LS_massless=0., DTYPE_t sigma_f1_massless=0., DTYPE_t sigma_f2_massless=0., DTYPE_t sigma_f3_massless=0.,
         DTYPE_t fnfp_si=1., DTYPE_t fnfp_sd=1.,
         DTYPE_t fnfp_anapole=1., DTYPE_t fnfp_magdip=1., DTYPE_t fnfp_elecdip=1.,
         DTYPE_t fnfp_LS=1., DTYPE_t fnfp_f1=1., DTYPE_t fnfp_f2=1., DTYPE_t fnfp_f3=1.,
         DTYPE_t fnfp_si_massless=0., DTYPE_t fnfp_sd_massless=1.,
         DTYPE_t fnfp_anapole_massless=1., DTYPE_t fnfp_magdip_massless=1., DTYPE_t fnfp_elecdip_massless=1.,
         DTYPE_t fnfp_LS_massless=1., DTYPE_t fnfp_f1_massless=1., DTYPE_t fnfp_f2_massless=1., DTYPE_t fnfp_f3_massless=1.,
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
                                          fnfp_si=fnfp_si, fnfp_sd=fnfp_sd, fnfp_si_massless=fnfp_si_massless, fnfp_sd_massless=fnfp_sd_massless,
                                          fnfp_anapole=fnfp_anapole, fnfp_magdip=fnfp_magdip, fnfp_elecdip=fnfp_elecdip, fnfp_anapole_massless=fnfp_anapole_massless, fnfp_magdip_massless=fnfp_magdip_massless, fnfp_elecdip_massless=fnfp_elecdip_massless,
                                          fnfp_LS=fnfp_LS, fnfp_f1=fnfp_f1, fnfp_f2=fnfp_f2, fnfp_f3=fnfp_f3, fnfp_LS_massless=fnfp_LS_massless, fnfp_f1_massless=fnfp_f1_massless, fnfp_f2_massless=fnfp_f2_massless, fnfp_f3_massless=fnfp_f3_massless,
                                          sigma_si= sigma_si, sigma_sd=sigma_sd, sigma_si_massless= sigma_si_massless, sigma_sd_massless=sigma_sd_massless,
                                          sigma_anapole=sigma_anapole, sigma_magdip=sigma_magdip, sigma_elecdip=sigma_elecdip, sigma_anapole_massless=sigma_anapole_massless, sigma_magdip_massless=sigma_magdip_massless, sigma_elecdip_massless=sigma_elecdip_massless,
                                          sigma_LS=sigma_LS, sigma_f1=sigma_f1, sigma_f2=sigma_f2, sigma_f3=sigma_f3, sigma_LS_massless=sigma_LS_massless, sigma_f1_massless=sigma_f1_massless, sigma_f2_massless=sigma_f2_massless, sigma_f3_massless=sigma_f3_massless) * efficiency_fn(Qs)
    result = trapz(dRdQs,Qs)
    return result



@cython.boundscheck(False)
def loglikelihood(np.ndarray[DTYPE_t] Q, object efficiency_fn, DTYPE_t mass=50.,
         DTYPE_t sigma_si=0.,DTYPE_t sigma_sd=0.,
         DTYPE_t sigma_anapole=0.,DTYPE_t sigma_magdip=0., DTYPE_t sigma_elecdip=0.,
         DTYPE_t sigma_LS=0., DTYPE_t sigma_f1=0., DTYPE_t sigma_f2=0., DTYPE_t sigma_f3=0.,
         DTYPE_t sigma_si_massless=0.,DTYPE_t sigma_sd_massless=0.,
         DTYPE_t sigma_anapole_massless=0.,DTYPE_t sigma_magdip_massless=0., DTYPE_t sigma_elecdip_massless=0.,
         DTYPE_t sigma_LS_massless=0., DTYPE_t sigma_f1_massless=0., DTYPE_t sigma_f2_massless=0., DTYPE_t sigma_f3_massless=0.,
         DTYPE_t fnfp_si=1., DTYPE_t fnfp_sd=1.,
         DTYPE_t fnfp_anapole=1., DTYPE_t fnfp_magdip=1., DTYPE_t fnfp_elecdip=1.,
         DTYPE_t fnfp_LS=1., DTYPE_t fnfp_f1=1., DTYPE_t fnfp_f2=1., DTYPE_t fnfp_f3=1.,
         DTYPE_t fnfp_si_massless=1., DTYPE_t fnfp_sd_massless=1.,
         DTYPE_t fnfp_anapole_massless=1., DTYPE_t fnfp_magdip_massless=1., DTYPE_t fnfp_elecdip_massless=1.,
         DTYPE_t fnfp_LS_massless=1., DTYPE_t fnfp_f1_massless=1., DTYPE_t fnfp_f2_massless=1., DTYPE_t fnfp_f3_massless=1.,
                  DTYPE_t v_lag=220., DTYPE_t v_rms=220., DTYPE_t v_esc=544., DTYPE_t rho_x=0.3,
                  str element='xenon', DTYPE_t Qmin=2., DTYPE_t Qmax=30., DTYPE_t exposure=1., energy_resolution=True):
    """
    This is the main log(likelihood) for any combination of UV models.
    
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
   
    cdef DTYPE_t Rate = R(efficiency_fn, mass=mass,
                                         v_rms=v_rms, v_lag=v_lag, v_esc=v_esc, rho_x=rho_x,
                                          fnfp_si=fnfp_si, fnfp_sd=fnfp_sd,
                                          fnfp_anapole=fnfp_anapole, fnfp_magdip=fnfp_magdip, fnfp_elecdip=fnfp_elecdip,
                                          fnfp_LS=fnfp_LS, fnfp_f1=fnfp_f1, fnfp_f2=fnfp_f2, fnfp_f3=fnfp_f3,
                                          sigma_si= sigma_si, sigma_sd=sigma_sd,
                                          sigma_anapole=sigma_anapole, sigma_magdip=sigma_magdip, sigma_elecdip=sigma_elecdip,
                                          sigma_LS=sigma_LS, sigma_f1=sigma_f1, sigma_f2=sigma_f2, sigma_f3=sigma_f3,
                                          fnfp_si_massless=fnfp_si_massless, fnfp_sd_massless=fnfp_sd_massless,
                                          fnfp_anapole_massless=fnfp_anapole_massless, fnfp_magdip_massless=fnfp_magdip_massless, fnfp_elecdip_massless=fnfp_elecdip_massless,
                                          fnfp_LS_massless=fnfp_LS_massless, fnfp_f1_massless=fnfp_f1_massless, fnfp_f2_massless=fnfp_f2_massless, fnfp_f3_massless=fnfp_f3_massless,
                                          sigma_si_massless = sigma_si_massless, sigma_sd_massless=sigma_sd_massless,
                                          sigma_anapole_massless=sigma_anapole_massless, sigma_magdip_massless=sigma_magdip_massless, sigma_elecdip_massless=sigma_elecdip_massless,
                                          sigma_LS_massless=sigma_LS_massless, sigma_f1_massless=sigma_f1_massless, sigma_f2_massless=sigma_f2_massless, sigma_f3_massless=sigma_f3_massless,                                         Qmin=Qmin, Qmax=Qmax, element=element)

    Nexp = Rate * Tobs
    if Nevents==0 and Nexp==0.:
        return 0.
    tot += Nevents * log(Nexp) - Nexp 
    if energy_resolution:
        tot -= Nevents * log(Rate) 
        out = dRdQ(Q, mass=mass,
                                        v_lag=v_lag, v_rms=v_rms, v_esc= v_esc, rho_x=rho_x, element=element,
                                        fnfp_si=fnfp_si, fnfp_sd=fnfp_sd,
                                          fnfp_anapole=fnfp_anapole, fnfp_magdip=fnfp_magdip, fnfp_elecdip=fnfp_elecdip,
                                          fnfp_LS=fnfp_LS, fnfp_f1=fnfp_f1, fnfp_f2=fnfp_f2, fnfp_f3=fnfp_f3,
                                          sigma_si= sigma_si, sigma_sd=sigma_sd,
                                          sigma_anapole=sigma_anapole, sigma_magdip=sigma_magdip, sigma_elecdip=sigma_elecdip,
                                          sigma_LS=sigma_LS, sigma_f1=sigma_f1, sigma_f2=sigma_f2, sigma_f3=sigma_f3,
                                          fnfp_si_massless=fnfp_si_massless, fnfp_sd_massless=fnfp_sd_massless,
                                          fnfp_anapole_massless=fnfp_anapole_massless, fnfp_magdip_massless=fnfp_magdip_massless, fnfp_elecdip_massless=fnfp_elecdip_massless,
                                          fnfp_LS_massless=fnfp_LS_massless, fnfp_f1_massless=fnfp_f1_massless, fnfp_f2_massless=fnfp_f2_massless, fnfp_f3_massless=fnfp_f3_massless,
                                          sigma_si_massless = sigma_si_massless, sigma_sd_massless=sigma_sd_massless,
                                          sigma_anapole_massless=sigma_anapole_massless, sigma_magdip_massless=sigma_magdip_massless, sigma_elecdip_massless=sigma_elecdip_massless,
                                          sigma_LS_massless=sigma_LS_massless, sigma_f1_massless=sigma_f1_massless, sigma_f2_massless=sigma_f2_massless, sigma_f3_massless=sigma_f3_massless) * efficiency_fn(Q)
        
    
        for i in range(Nevents):
            if out[i]==0.:
                return -1.*INFINITY #if an event is seen where the model expects zero events (behind the V_lag), this model is excluded, and loglikelihood=-Infinity.
            tot += log(out[i])
            
    return tot
