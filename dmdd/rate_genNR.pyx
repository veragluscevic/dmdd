import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool

import math
import formgenNR as formgenNR
from helpers import trapz, eta, zeta
from dmdd.globals import PAR_NORMS
import dmdd.constants as const
DTYPE = np.float
ctypedef np.float_t DTYPE_t
cdef DTYPE_t pi = np.pi

cdef extern from "math.h":
    float INFINITY
    double exp(double)
    double sqrt(double)
    double erf(double)
    double log10(double)
    double log(double)

#constants for this module:
cdef DTYPE_t c_scale_global=500. # units for the c_i as given below
cdef DTYPE_t ratenorm = 0.0654869 # this converts from cm**-3 * GeV**-4 to DRU = cts / keV / kg / sec

#physical constants from constants.py:
cdef DTYPE_t mN = const.NUCLEON_MASS # Nucleon mass in GeV
cdef DTYPE_t pmag = const.P_MAGMOM # proton magnetic moment, PDG Live
cdef DTYPE_t nmag = const.N_MAGMOM # neutron magnetic moment, PDG Live

#information about target nuclei:
eltshort = const.ELEMENT_INFO


#######################  "Master" rate functions. 

def dRdQ(np.ndarray[DTYPE_t] Er, DTYPE_t mass=50., np.ndarray[DTYPE_t] c1p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c3p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c4p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c5p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c6p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c7p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c8p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c9p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c10p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c11p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c12p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c13p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c14p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c15p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c1n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c3n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c4n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c5n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c6n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c7n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c8n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c9n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c10n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c11n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c12n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c13n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c14n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c15n=np.array([0.,0.,0.]), DTYPE_t v_lag=220., DTYPE_t v_rms=220., DTYPE_t v_esc=544., str element='xenon', DTYPE_t rho_x=0.3, DTYPE_t c_scale=c_scale_global):
    """
    Differential recoil energy spectrum in counts/keV/kg/sec

    :param Er:
      This is a list of energies in keV.
      The list must be entered as a numpy array, np.array([..]).
      It can have as many entries as desired.
    :type Er: ``np.ndarray``
      
    :param mass:
       Dark matter mass in GeV.
       *Optional*, default to 50.
    :type mass: ``float``
    
    :param cXN's:
      28 different np.arrays, all optional. These are the EFT coefficients.
      
      X can be any number from 1-15 (except 2).
      N must be n or p.
      Any cXN that is entered is a list of coefficients.
      
      The list must be entered as a numpy array, np.array([..]).
      
      **c1N and c4N must have three entries,** any of which may be zero:

        -the first entry is a momentum-independent term.

        -the second entry is the coefficient of a term that multiplies q^2/mDM^2.

        -the third entry is the coefficient of a term that multiplies q^4/mDM^4.
        
      **c3N and c5Y-c12N must have two entries,** any of which may be zero:
      
          -the first entry is a momentum-independent term.

          -the second entry is the coefficient of a term that multiplies q^2/mDM^2.
          
      **c13N-c15N must have one entry.**
      
      All cXN have mass dimension negative two.
      The mass scale of the suppression is 500 GeV by default (may be adjusted; see `c_scale`).

    :param c_scale:
      Suppression scale of all cXN coefficients in GeV.
      From a UV perspective, this is roughly mediator_mass/sqrt(couplings).
      *Optional*, default 500.
    :type c_scale: ``float``

    :param v_lag:
      Velocity of the solar system with respect to the Milky Way in km/s.
      *Optional*, default to 220.
    :type v_lag: ``float``
    
    :param v_rms: 
       1.5 * (velocity dispersion in km/s) of the local DM distribution.
       *Optional*, default to 220.
    :type v_rms: ``float``
    
    :param v_esc: 
       Escape velocity in km/s of a DM particle in the galactic rest frame.
       *Optional*, default to 544.
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
      *Optional*, default to 0.3
    :type rho_x: ``float``
      

    :return:
      array of differential recoil energy spectrum in counts/keV/kg/sec
  
    """
    cdef int npts = len(Er)
    cdef np.ndarray[DTYPE_t] out = np.empty(npts,dtype=float)
    cdef int i
    cdef element_name = str(element.title())

    cdef DTYPE_t q, tot, y_harm, b_harm, m_reduced_sq, v_min, val_eta, weight, v_independent
    cdef DTYPE_t ff_M_std, ff_M_v2, ff_M_q2, ff_M_v2q2, ff_M_q4, ff_SigPP_std, ff_SigPP_v2, ff_SigPP_q2, ff_SigPP_v2q2, ff_SigPP_q4, ff_SigP_std, ff_SigP_v2, ff_SigP_q2, ff_SigP_v2q2, ff_SigP_q4, ff_PhiPP_q2, ff_PhiPP_q4, ff_Delta_q2, ff_Delta_q4, ff_MPhiPP_q2, ff_MPhiPP_q4, ff_SigPDelta_q2

    weight = eltshort[element]['weight']
    b_harm = 5.0677*(41.467/(45.*weight**(-1./3.) - 25.*weight**(-2./3.)))**0.5 #this is in [1/GeV]
    m_reduced_sq = weight**2.*mN**2.*mass**2./(weight*mN+mass)**2.
    q_scale = 0.1
    v_independent = ratenorm * rho_x / (2. * pi * mass) / c_scale_global**4. * (c_scale_global / c_scale)**4.
    for i in range(npts):
        q = Er[i]*10**-6. #converting Er from keV-->GeV.
        qsq = 2.*weight*mN*q # this is qsquared, which multiplies the coefficients
        y_harm = weight*mN*q*b_harm**2/2. #this takes q in [GeV].
        v_min = ((2.*weight*mN*q))**0.5/(2.*weight*mN*mass/(weight*mN+mass)) *3.*10.**5
        ff_M_std = formgenNR.factor_M(element_name,y_harm, c1p[0]*c1p[0], c1p[0]*c1n[0], c1n[0]*c1p[0], c1n[0]*c1n[0])
        ff_M_v2 = formgenNR.factor_M(element_name,y_harm, c8p[0]*c8p[0]/4., c8p[0]*c8n[0]/4., c8n[0]*c8p[0]/4., c8n[0]*c8n[0]/4.)
        ff_M_q2 = formgenNR.factor_M(element_name,y_harm, 2*c1p[1]*c1p[0]*mN**2./mass**2.+c11p[0]*c11p[0]/4.-c8p[0]*c8p[0]*mN**2./(16.*m_reduced_sq), (c1p[1]*c1n[0]+c1p[0]*c1n[1])*mN**2./mass**2.+c11p[0]*c11n[0]/4.-c8p[0]*c8n[0]*mN**2./(16.*m_reduced_sq), (c1p[1]*c1n[0]+c1p[0]*c1n[1])*mN**2./mass**2.+c11n[0]*c11p[0]/4.-c8n[0]*c8p[0]*mN**2./(16.*m_reduced_sq), 2*c1n[1]*c1n[0]*mN**2./mass**2.+c11n[0]*c11n[0]/4.-c8n[0]*c8n[0]*mN**2./(16.*m_reduced_sq))
        ff_M_v2q2 = formgenNR.factor_M(element_name,y_harm, c5p[0]*c5p[0]/4.+2.*c8p[1]*c8p[0]/4., c5p[0]*c5n[0]/4.+(c8p[1]*c8n[0]+c8p[0]*c8n[1])/4., c5n[0]*c5p[0]/4.+(c8n[1]*c8p[0]+c8n[0]*c8p[1])/4., c5n[0]*c5n[0]/4.+2.*c8n[1]*c8n[0]/4.)
        ff_M_q4 = formgenNR.factor_M(element_name,y_harm, (c1p[1]*c1p[1]+2*c1p[2]*c1p[0])*mN**4./mass**4.+(2.*c11p[1]*c11p[0]/4.-2.*c8p[1]*c8p[0]*mN**2./(16.*m_reduced_sq))*mN**2/mass**2.-c5p[0]*c5p[0]*mN**2./(16.*m_reduced_sq), (c1p[1]*c1n[1]+c1p[2]*c1n[0]+c1p[0]*c1n[2])*mN**4./mass**4.+((c11p[1]*c11n[0]+c11p[0]*c11n[1])/4.-(c8p[1]*c8n[0]+c8p[0]*c8n[1])*mN**2./(16.*m_reduced_sq))*mN**2/mass**2.-c5p[0]*c5n[0]*mN**2./(16.*m_reduced_sq), (c1n[1]*c1p[1]+c1n[2]*c1p[0]+c1n[0]*c1p[2])*mN**4./mass**4.+((c11n[1]*c11p[0]+c11n[0]*c11p[1])/4.-(c8n[1]*c8p[0]+c8n[0]*c8p[1])*mN**2./(16.*m_reduced_sq))*mN**2/mass**2.-c5n[0]*c5p[0]*mN**2./(16.*m_reduced_sq), (c1n[1]*c1n[1]+2*c1n[2]*c1n[0])*mN**4./mass**4.+(2.*c11n[1]*c11n[0]/4.-2.*c8n[1]*c8n[0]*mN**2./(16.*m_reduced_sq))*mN**2/mass**2.-c5n[0]*c5n[0]*mN**2./(16.*m_reduced_sq))
        ff_SigPP_std = formgenNR.factor_SigPP(element_name,y_harm, c4p[0]*c4p[0]/16., c4p[0]*c4n[0]/16., c4n[0]*c4p[0]/16., c4n[0]*c4n[0]/16.)
        ff_SigPP_v2 = formgenNR.factor_SigPP(element_name,y_harm, c12p[0]*c12p[0]/16., c12p[0]*c12n[0]/16., c12n[0]*c12p[0]/16., c12n[0]*c12n[0]/16.)
        ff_SigPP_q2 = formgenNR.factor_SigPP(element_name,y_harm, 2.*c4p[1]*c4p[0]*mN**2./mass**2./16.+c10p[0]*c10p[0]/4.+(c4p[0]*c6p[0]+c6p[0]*c4p[0])/16.-c12p[0]*c12p[0]*mN**2./(64.*m_reduced_sq), (c4p[1]*c4n[0]+c4p[0]*c4n[1])*mN**2./mass**2./16.+c10p[0]*c10n[0]/4.+(c4p[0]*c6n[0]+c6p[0]*c4n[0])/16.-c12p[0]*c12n[0]*mN**2./(64.*m_reduced_sq), (c4n[1]*c4p[0]+c4n[0]*c4p[1])*mN**2./mass**2./16.+c10n[0]*c10p[0]/4.+(c4n[0]*c6p[0]+c6n[0]*c4p[0])/16.-c12n[0]*c12p[0]*mN**2./(64.*m_reduced_sq), 2.*c4n[1]*c4n[0]*mN**2./mass**2./16.+c10n[0]*c10n[0]/4.+(c4n[0]*c6n[0]+c6n[0]*c4n[0])/16.-c12n[0]*c12n[0]*mN**2./(64.*m_reduced_sq))
        ff_SigPP_v2q2 = formgenNR.factor_SigPP(element_name,y_harm, c13p[0]*c13p[0]/16.+2.*c12p[1]*c12p[0]/16., c13p[0]*c13n[0]/16.+(c12p[1]*c12n[0]+c12p[0]*c12n[1])/16., c13n[0]*c13p[0]/16.+(c12n[1]*c12p[0]+c12n[0]*c12p[1])/16., c13n[0]*c13n[0]/16.+2.*c12n[1]*c12n[0]/16.)
        ff_SigPP_q4 = formgenNR.factor_SigPP(element_name,y_harm, (c4p[1]*c4p[1]+2*c4p[2]*c4p[0])*mN**4./mass**4./16.+(2.*c10p[1]*c10p[0]/4.+2.*(c4p[1]*c6p[0]+c6p[1]*c4p[0])/16.-2.*c12p[1]*c12p[0]*mN**2./(64.*m_reduced_sq))*mN**2./mass**2.+c6p[0]*c6p[0]/16.-c13p[0]*c13p[0]*mN**2./(64.*m_reduced_sq), (c4p[1]*c4n[1]+c4p[2]*c4n[0]+c4p[0]*c4n[2])*mN**4./mass**4./16.+((c10p[1]*c10n[0]+c10p[0]*c10n[1])/4.+(c4p[1]*c6n[0]+c4p[0]*c6n[1]+c6p[1]*c4n[0]+c6p[0]*c4n[1])/16.-(c12p[1]*c12n[0]+c12p[0]*c12n[1])*mN**2./(64.*m_reduced_sq))*mN**2./mass**2.+c6p[0]*c6n[0]/16.-c13p[0]*c13n[0]*mN**2./(64.*m_reduced_sq), (c4n[1]*c4p[1]+c4n[2]*c4p[0]+c4n[0]*c4p[2])*mN**4./mass**4./16.+((c10n[1]*c10p[0]+c10n[0]*c10p[1])/4.+(c4n[1]*c6p[0]+c4n[0]*c6p[1]+c6n[1]*c4p[0]+c6n[0]*c4p[1])/16.-(c12n[1]*c12p[0]+c12n[0]*c12p[1])*mN**2./(64.*m_reduced_sq))*mN**2./mass**2.+c6n[0]*c6p[0]/16.-c13n[0]*c13p[0]*mN**2./(64.*m_reduced_sq), (c4n[1]*c4n[1]+2*c4n[2]*c4n[0])*mN**4./mass**4./16.+(2.*c10n[1]*c10n[0]/4.+2.*(c4n[1]*c6n[0]+c6n[0]*c4n[1])/16.-2.*c12n[1]*c12n[0]*mN**2./(64.*m_reduced_sq))*mN**2./mass**2.+c6n[0]*c6n[0]/16.-c13n[0]*c13n[0]*mN**2./(64.*m_reduced_sq))
        ff_SigP_std = formgenNR.factor_SigP(element_name,y_harm, c4p[0]*c4p[0]/16., c4p[0]*c4n[0]/16., c4n[0]*c4p[0]/16., c4n[0]*c4n[0]/16.)
        ff_SigP_v2 = formgenNR.factor_SigP(element_name,y_harm, c12p[0]*c12p[0]/32.+c7p[0]*c7p[0]/8., c12p[0]*c12n[0]/32.+c7p[0]*c7n[0]/8., c12n[0]*c12p[0]/32.+c7n[0]*c7p[0]/8., c12n[0]*c12n[0]/32.+c7n[0]*c7n[0]/8.)
        ff_SigP_q2 = formgenNR.factor_SigP(element_name,y_harm, 2.*c4p[1]*c4p[0]/16.*mN**2./mass**2.+c9p[0]*c9p[0]/16.-c12p[0]*c12p[0]*mN**2./(128.*m_reduced_sq)-c7p[0]*c7p[0]*mN**2./(32.*m_reduced_sq), (c4p[1]*c4n[0]+c4p[0]*c4n[1])/16.*mN**2./mass**2.+c9p[0]*c9n[0]/16.-c12p[0]*c12n[0]*mN**2./(128.*m_reduced_sq)-c7p[0]*c7n[0]*mN**2./(32.*m_reduced_sq), (c4n[1]*c4p[0]+c4n[0]*c4p[1])/16.*mN**2./mass**2.+c9n[0]*c9p[0]/16.-c12n[0]*c12p[0]*mN**2./(128.*m_reduced_sq)-c7n[0]*c7p[0]*mN**2./(32.*m_reduced_sq), 2.*c4n[1]*c4n[0]/16.*mN**2./mass**2.+c9n[0]*c9n[0]/16.-c12n[0]*c12n[0]*mN**2./(128.*m_reduced_sq)-c7n[0]*c7n[0]*mN**2./(32.*m_reduced_sq))
        ff_SigP_v2q2 = formgenNR.factor_SigP(element_name,y_harm, c3p[0]*c3p[0]/8.+(c14p[0]*c14p[0]-c12p[0]*c15p[0]-c15p[0]*c12p[0])/32.+2.*(c12p[1]*c12p[0]/32.+c7p[1]*c7p[0]/8.), c3p[0]*c3n[0]/8.+(c14p[0]*c14n[0]-c12p[0]*c15n[0]-c15p[0]*c12n[0])/32.+(c12p[1]*c12n[0]+c12p[0]*c12n[1])/32.+(c7p[1]*c7n[0]+c7p[0]*c7n[1])/8., c3n[0]*c3p[0]/8.+(c14n[0]*c14p[0]-c12n[0]*c15p[0]-c15n[0]*c12p[0])/32.+(c12n[1]*c12p[0]+c12n[0]*c12p[1])/32.+(c7n[1]*c7p[0]+c7n[0]*c7p[1])/8., c3n[0]*c3n[0]/8.+(c14n[0]*c14n[0]-c12n[0]*c15n[0]-c15n[0]*c12n[0])/32.+2.*(c12n[1]*c12n[0]/32.+c7n[1]*c7n[0]/8.))
        ff_SigP_q4 = formgenNR.factor_SigP(element_name,y_harm, (c4p[1]*c4p[1]+2.*c4p[2]*c4p[0])/16.*mN**4./mass**4.+(2.*c9p[1]*c9p[0]/16.-2.*c12p[1]*c12p[0]*mN**2./(128.*m_reduced_sq)-2.*c7p[1]*c7p[0]*mN**2./(32.*m_reduced_sq))*mN**2./mass**2.+(c15p[0]*c12p[0]+c12p[0]*c15p[0]-c14p[0]*c14p[0]-4.*c3p[0]*c3p[0])*mN**2./(128.*m_reduced_sq), (c4p[1]*c4n[1]+c4p[2]*c4n[0]+c4p[0]*c4n[2])/16.*mN**4./mass**4.+((c9p[1]*c9n[0]+c9p[0]*c9n[1])/16.-(c12p[1]*c12n[0]+c12p[0]*c12n[1])*mN**2./(128.*m_reduced_sq)-(c7p[1]*c7n[0]+c7p[0]*c7n[1])*mN**2./(32.*m_reduced_sq))*mN**2./mass**2.+(c15p[0]*c12n[0]+c12p[0]*c15n[0]-c14p[0]*c14n[0]-4.*c3p[0]*c3n[0])*mN**2./(128.*m_reduced_sq), (c4n[1]*c4p[1]+c4n[2]*c4p[0]+c4n[0]*c4p[2])/16.*mN**4./mass**4.+((c9n[1]*c9p[0]+c9n[0]*c9p[1])/16.-(c12n[1]*c12p[0]+c12n[0]*c12p[1])*mN**2./(128.*m_reduced_sq)-(c7n[1]*c7p[0]+c7n[0]*c7p[1])*mN**2./(32.*m_reduced_sq))*mN**2./mass**2.+(c15n[0]*c12p[0]+c12n[0]*c15p[0]-c14n[0]*c14p[0]-4.*c3n[0]*c3p[0])*mN**2./(128.*m_reduced_sq), (c4n[1]*c4n[1]+2.*c4n[2]*c4n[0])/16.*mN**4./mass**4.+(2.*c9n[1]*c9n[0]/16.-2.*c12n[1]*c12n[0]*mN**2./(128.*m_reduced_sq)-2.*c7n[1]*c7n[0]*mN**2./(32.*m_reduced_sq))*mN**2./mass**2.+(c15n[0]*c12n[0]+c12n[0]*c15n[0]-c14n[0]*c14n[0]-4.*c3n[0]*c3n[0])*mN**2./(128.*m_reduced_sq))
        ff_PhiPP_q2 = formgenNR.factor_PhiPP(element_name,y_harm, c12p[0]*c12p[0]/16., c12p[0]*c12n[0]/16., c12n[0]*c12p[0]/16., c12n[0]*c12n[0]/16.)
        ff_PhiPP_q4 = formgenNR.factor_PhiPP(element_name,y_harm, 2.*c12p[1]*c12p[0]/16.*mN**2./mass**2.+c3p[0]*c3p[0]/4.-(c12p[0]*c15p[0]+c15p[0]*c12p[0])/16., (c12p[1]*c12n[0]+c12p[0]*c12n[1])/16.*mN**2./mass**2.+c3p[0]*c3n[0]/4.-(c12p[0]*c15n[0]+c15p[0]*c12n[0])/16., (c12n[1]*c12p[0]+c12n[0]*c12p[1])/16.*mN**2./mass**2.+c3n[0]*c3p[0]/4.-(c12n[0]*c15p[0]+c15n[0]*c12p[0])/16., 2.*c12n[1]*c12n[0]/16.*mN**2./mass**2.+c3n[0]*c3n[0]/4.-(c12n[0]*c15n[0]+c15n[0]*c12n[0])/16.)
        ff_Delta_q2 = formgenNR.factor_Delta(element_name,y_harm, c8p[0]*c8p[0]/4., c8p[0]*c8n[0]/4., c8n[0]*c8p[0]/4., c8n[0]*c8n[0]/4.)
        ff_Delta_q4 = formgenNR.factor_Delta(element_name,y_harm, 2.*c8p[1]*c8p[0]/4.*mN**2./mass**2.+c5p[0]*c5p[0]/4., (c8p[1]*c8n[0]+c8p[0]*c8n[1])/4.*mN**2./mass**2.+c5p[0]*c5n[0]/4., (c8n[1]*c8p[0]+c8n[0]*c8p[1])/4.*mN**2./mass**2.+c5n[0]*c5p[0]/4., 2.*c8n[1]*c8n[0]/4.*mN**2./mass**2.+c5n[0]*c5n[0]/4.)
        ff_MPhiPP_q2 = formgenNR.factor_MPhiPP(element_name,y_harm, c3p[0]*c1p[0]+c12p[0]*c11p[0]/4., c3p[0]*c1n[0]+c12p[0]*c11n[0]/4., c3n[0]*c1p[0]+c12n[0]*c11p[0]/4., c3n[0]*c1n[0]+c12n[0]*c11n[0]/4.)
        ff_MPhiPP_q4 = formgenNR.factor_MPhiPP(element_name,y_harm, (c3p[1]*c1p[0]+c3p[0]*c1p[1]+c12p[1]*c11p[0]/4.+c12p[0]*c11p[1]/4.)*mN**2./mass**2.-c15p[0]*c11p[0]/4., (c3p[1]*c1n[0]+c3p[0]*c1n[1]+c12p[1]*c11n[0]/4.+c12p[0]*c11n[1]/4.)*mN**2./mass**2.-c15p[0]*c11n[0]/4., (c3n[1]*c1p[0]+c3n[0]*c1p[1]+c12n[1]*c11p[0]/4.+c12n[0]*c11p[1]/4.)*mN**2./mass**2.-c15n[0]*c11p[0]/4., (c3n[1]*c1n[0]+c3n[0]*c1n[1]+c12n[1]*c11n[0]/4.+c12n[0]*c11n[1]/4.)*mN**2./mass**2.-c15n[0]*c11n[0]/4.)
        ff_SigPDelta_q2 = formgenNR.factor_SigPDelta(element_name,y_harm, (c5p[0]*c4p[0]-c8p[0]*c9p[0])/4., (c5p[0]*c4n[0]-c8p[0]*c9n[0])/4., (c5n[0]*c4p[0]-c8n[0]*c9p[0])/4., (c5n[0]*c4n[0]-c8n[0]*c9n[0])/4.)
        ff_SigPDelta_q4 = formgenNR.factor_SigPDelta(element_name,y_harm, (c5p[1]*c4p[0]+c5p[0]*c4p[1]-c8p[1]*c9p[0]-c8p[0]*c9p[1])/4., (c5p[1]*c4n[0]+c5p[0]*c4n[1]-c8p[1]*c9n[0]-c8p[0]*c9n[1])/4., (c5n[1]*c4p[0]+c5n[0]*c4p[1]-c8n[1]*c9p[0]-c8n[0]*c9p[1])/4., (c5n[1]*c4n[0]+c5n[0]*c4n[1]-c8n[1]*c9n[0]-c8n[0]*c9n[1])/4.)
        val_eta = eta(v_min,v_esc,v_rms,v_lag)
        val_zeta = zeta(v_min,v_esc,v_rms,v_lag)
        tot = v_independent * ( (ff_M_std + ff_SigPP_std + ff_SigP_std + qsq/mN**2.*(ff_M_q2 + ff_SigPP_q2 + ff_SigP_q2 + ff_PhiPP_q2 + ff_Delta_q2 + ff_MPhiPP_q2 + ff_SigPDelta_q2) + qsq**2./mN**4.*(ff_M_q4 + ff_SigPP_q4 + ff_SigP_q4 + ff_PhiPP_q4 + ff_Delta_q4 + ff_MPhiPP_q4 + ff_SigPDelta_q4))*val_eta + (ff_M_v2 + ff_SigPP_v2 + ff_SigP_v2 + qsq/mN**2.*(ff_M_v2q2 + ff_SigPP_v2q2 + ff_SigP_v2q2))*val_zeta )
        out[i] = tot
    return out



@cython.boundscheck(False)
def R(object efficiency_fn, DTYPE_t mass=50., np.ndarray[DTYPE_t] c1p=np.array([1.,0.,0.]), np.ndarray[DTYPE_t] c3p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c4p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c5p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c6p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c7p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c8p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c9p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c10p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c11p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c12p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c13p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c14p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c15p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c1n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c3n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c4n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c5n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c6n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c7n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c8n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c9n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c10n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c11n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c12n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c13n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c14n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c15n=np.array([0.,0.,0.]), DTYPE_t v_lag=220., DTYPE_t v_rms=220., DTYPE_t v_esc=544., DTYPE_t rho_x=0.3, DTYPE_t c_scale=c_scale_global, str element='xenon', DTYPE_t Qmin=2., DTYPE_t Qmax=30.):
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
        
    cdef np.ndarray[DTYPE_t] dRdQs = dRdQ(Qs, v_lag=v_lag, v_rms=v_rms, v_esc=v_esc, mass=mass, c1p=c1p, c3p=c3p, c4p=c4p, c5p=c5p, c6p=c6p, c7p=c7p, c8p=c8p, c9p=c9p, c10p=c10p, c11p=c11p, c12p=c12p, c13p=c13p, c14p=c14p, c15p=c15p, c1n=c1n, c3n=c3n, c4n=c4n, c5n=c5n, c6n=c6n, c7n=c7n, c8n=c8n, c9n=c9n, c10n=c10n, c11n=c11n, c12n=c12n, c13n=c13n, c14n=c14n, c15n=c15n, element=element, rho_x=rho_x, c_scale=c_scale) * efficiency_fn(Qs)
    result = trapz(dRdQs,Qs)
    return result



@cython.boundscheck(False)
def loglikelihood(np.ndarray[DTYPE_t] Q, object efficiency_fn, DTYPE_t mass=50., np.ndarray[DTYPE_t] c1p=np.array([1.,0.,0.]), np.ndarray[DTYPE_t] c3p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c4p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c5p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c6p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c7p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c8p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c9p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c10p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c11p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c12p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c13p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c14p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c15p=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c1n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c3n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c4n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c5n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c6n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c7n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c8n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c9n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c10n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c11n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c12n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c13n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c14n=np.array([0.,0.,0.]), np.ndarray[DTYPE_t] c15n=np.array([0.,0.,0.]), DTYPE_t v_lag=220., DTYPE_t v_rms=220., DTYPE_t v_esc=544., DTYPE_t rho_x=0.3, DTYPE_t c_scale=c_scale_global, str element='xenon', DTYPE_t Qmin=2., DTYPE_t Qmax=30., DTYPE_t exposure=1., energy_resolution=True):
    """
    This is the main log(likelihood) for any combination of EFT coefficients.
    
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
   
    cdef DTYPE_t Rate = R(efficiency_fn, mass=mass, v_rms=v_rms, v_lag=v_lag, v_esc=v_esc, rho_x=rho_x, c_scale=c_scale, c1p=c1p, c3p=c3p, c4p=c4p, c5p=c5p, c6p=c6p, c7p=c7p, c8p=c8p, c9p=c9p, c10p=c10p, c11p=c11p, c12p=c12p, c13p=c13p, c14p=c14p, c15p=c15p, c1n=c1n, c3n=c3n, c4n=c4n, c5n=c5n, c6n=c6n, c7n=c7n, c8n=c8n, c9n=c9n, c10n=c10n, c11n=c11n, c12n=c12n, c13n=c13n, c14n=c14n, c15n=c15n, Qmin=Qmin, Qmax=Qmax, element=element)

    Nexp = Rate * Tobs
    if Nevents==0 and Nexp==0.:
        return 0.
    tot += Nevents * log(Nexp) - Nexp
    if energy_resolution:
        tot -= Nevents * log(Rate) 
        out = dRdQ(Q, mass=mass, v_lag=v_lag, v_rms=v_rms, v_esc= v_esc, rho_x=rho_x, c_scale=c_scale, element=element, c1p=c1p, c3p=c3p, c4p=c4p, c5p=c5p, c6p=c6p, c7p=c7p, c8p=c8p, c9p=c9p, c10p=c10p, c11p=c11p, c12p=c12p, c13p=c13p, c14p=c14p, c15p=c15p, c1n=c1n, c3n=c3n, c4n=c4n, c5n=c5n, c6n=c6n, c7n=c7n, c8n=c8n, c9n=c9n, c10n=c10n, c11n=c11n, c12n=c12n, c13n=c13n, c14n=c14n, c15n=c15n) * efficiency_fn(Q)
        
    
        for i in range(Nevents):
            if out[i]==0.:
                return -1.*INFINITY #if an event is seen where the model expects zero events (behind the V_lag), this model is excluded, and loglikelihood=-Infinity.
            tot += log(out[i])
            
    return tot
