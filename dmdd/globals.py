import numpy as np
    
FNFP_PARAM_NAMES = ['fnfp_si', 'fnfp_sd', 'fnfp_anapole', 'fnfp_magdip', 'fnfp_elecdip',
                    'fnfp_LS', 'fnfp_f1', 'fnfp_f2', 'fnfp_f3', 'fnfp_si_massless', 'fnfp_sd_massless',
                    'fnfp_anapole_massless', 'fnfp_magdip_massless', 'fnfp_elecdip_massless', 'fnfp_LS_massless',
                    'fnfp_f1_massless', 'fnfp_f2_massless', 'fnfp_f3_massless']

PAR_NORMS = { 
    'sigma_si':1.e-47,
    'sigma_sd':1.e-42,
    'sigma_anapole':1.e-40,
    'sigma_magdip':1.e-38,
    'sigma_elecdip':1.e-44,
    'sigma_LS':1.e-44,
    'sigma_f1':1.e-47,
    'sigma_f2':1.e-42,
    'sigma_f3':1.e-41,
    'sigma_si_massless':1.e-48,
    'sigma_sd_massless':1.e-43,
    'sigma_anapole_massless':1.e-45,
    'sigma_magdip_massless':1.e-39,
    'sigma_elecdip_massless':1.e-45,
    'sigma_LS_massless':1.e-42,
    'sigma_f1_massless':1.e-48,
    'sigma_f2_massless':1.e-43,
    'sigma_f3_massless':1.e-42
    }

PAR_NORM_EXPONENTS = { 
    'sigma_si': np.log10(PAR_NORMS['sigma_si']),
    'sigma_sd': np.log10(PAR_NORMS['sigma_sd']),
    'sigma_anapole': np.log10(PAR_NORMS['sigma_anapole']),
    'sigma_magdip': np.log10(PAR_NORMS['sigma_magdip']),
    'sigma_elecdip': np.log10(PAR_NORMS['sigma_elecdip']),
    'sigma_LS': np.log10(PAR_NORMS['sigma_LS']),
    'sigma_f1': np.log10(PAR_NORMS['sigma_f1']),
    'sigma_f2': np.log10(PAR_NORMS['sigma_f2']),
    'sigma_f3': np.log10(PAR_NORMS['sigma_f3']),
    'sigma_si_massless': np.log10(PAR_NORMS['sigma_si_massless']),
    'sigma_sd_massless': np.log10(PAR_NORMS['sigma_sd_massless']),
    'sigma_anapole_massless': np.log10(PAR_NORMS['sigma_anapole_massless']),
    'sigma_magdip_massless': np.log10(PAR_NORMS['sigma_magdip_massless']),
    'sigma_elecdip_massless': np.log10(PAR_NORMS['sigma_elecdip_massless']),
    'sigma_LS_massless': np.log10(PAR_NORMS['sigma_LS_massless']),
    'sigma_f1_massless': np.log10(PAR_NORMS['sigma_f1_massless']),
    'sigma_f2_massless': np.log10(PAR_NORMS['sigma_f2_massless']),
    'sigma_f3_massless': np.log10(PAR_NORMS['sigma_f3_massless']),
    }



PARAM_TEX = {
    'mass': r'$m_\chi$ [GeV]',
    'sigma_si': r'$\sigma_{{p}}^{{SI}}$ [$10^{{{:.0f}}}$ cm$^2$]'.format(PAR_NORM_EXPONENTS['sigma_si']),
    'sigma_sd': r'$\sigma_{{p}}^{{SD}}$ [$10^{{{:.0f}}}$ cm$^2$]'.format(PAR_NORM_EXPONENTS['sigma_sd']),
    'sigma_anapole': r'$\sigma_{{p}}^{{ana}}$ [$10^{{{:.0f}}}$ cm$^2$]'.format(PAR_NORM_EXPONENTS['sigma_anapole']),
    'sigma_magdip': r'$\sigma_{{p}}^{{MD}}$ [$10^{{{:.0f}}}$ cm$^2$]'.format(PAR_NORM_EXPONENTS['sigma_magdip']),
    'sigma_elecdip': r'$\sigma_{{p}}^{{ED}}$ [$10^{{{:.0f}}}$ cm$^2$]'.format(PAR_NORM_EXPONENTS['sigma_elecdip']),
    'sigma_LS': r'$\sigma_{{p}}^{{LS}}$ [$10^{{{:.0f}}}$ cm$^2$]'.format(PAR_NORM_EXPONENTS['sigma_LS']),
    'sigma_f1': r'$\sigma_{{p}}^{{f_1}}$ [$10^{{{:.0f}}}$ cm$^2$]'.format(PAR_NORM_EXPONENTS['sigma_f1']),
    'sigma_f2': r'$\sigma_{{p}}^{{f_2}}$ [$10^{{{:.0f}}}$ cm$^2$]'.format(PAR_NORM_EXPONENTS['sigma_f2']),
    'sigma_f3': r'$\sigma_{{p}}^{{f_3}}$ [$10^{{{:.0f}}}$ cm$^2$]'.format(PAR_NORM_EXPONENTS['sigma_f3']),
    'sigma_si_massless': r'$\sigma_{{p}}^{{milli}}$ [$10^{{{:.0f}}}$ cm$^2$]'.format(PAR_NORM_EXPONENTS['sigma_si_massless']),
    'sigma_sd_massless': r'$\sigma_{{p}}^{{SD, light}}$ [$10^{{{:.0f}}}$ cm$^2$]'.format(PAR_NORM_EXPONENTS['sigma_sd_massless']),
    'sigma_anapole_massless': r'$\sigma_{{p}}^{{ana, light}}$ [$10^{{{:.0f}}}$ cm$^2$]'.format(PAR_NORM_EXPONENTS['sigma_anapole_massless']),
    'sigma_magdip_massless': r'$\sigma_{{p}}^{{MD, light}}$ [$10^{{{:.0f}}}$ cm$^2$]'.format(PAR_NORM_EXPONENTS['sigma_magdip_massless']),
    'sigma_elecdip_massless': r'$\sigma_{{p}}^{{ED, light}}$ [$10^{{{:.0f}}}$ cm$^2$]'.format(PAR_NORM_EXPONENTS['sigma_elecdip_massless']),
    'sigma_LS_massless': r'$\sigma_{{p}}^{{LS, light}}$ [$10^{{{:.0f}}}$ cm$^2$]'.format(PAR_NORM_EXPONENTS['sigma_LS_massless']),
    'sigma_f1_massless': r'$\sigma_{{p}}^{{f_1, light}}$ [$10^{{{:.0f}}}$ cm$^2$]'.format(PAR_NORM_EXPONENTS['sigma_f1_massless']),
    'sigma_f2_massless': r'$\sigma_{{p}}^{{f_2, light}}$ [$10^{{{:.0f}}}$ cm$^2$]'.format(PAR_NORM_EXPONENTS['sigma_f2_massless']),
    'sigma_f3_massless': r'$\sigma_{{p}}^{{f_3, light}}$ [$10^{{{:.0f}}}$ cm$^2$]'.format(PAR_NORM_EXPONENTS['sigma_f3_massless']),   
    'fnfp_si': r'$f_n/f_p$',
    'fnfp_sd': r'$f_n/f_p$',
    'fnfp_anapole': r'$f_n/f_p$',
    'fnfp_magdip': r'$f_n/f_p$',
    'fnfp_elecdip': r'$f_n/f_p$',
    'fnfp_LS': r'$f_n/f_p$',
    'fnfp_f1': r'$f_n/f_p$',
    'fnfp_f2': r'$f_n/f_p$',
    'fnfp_f3': r'$f_n/f_p$',
    'fnfp_si_massless': r'$f_n/f_p$',
    'fnfp_sd_massless': r'$f_n/f_p$',
    'fnfp_anapole_massless': r'$f_n/f_p$',
    'fnfp_magdip_massless': r'$f_n/f_p$',
    'fnfp_elecdip_massless': r'$f_n/f_p$',
    'fnfp_LS_massless': r'$f_n/f_p$',
    'fnfp_f1_massless': r'$f_n/f_p$',
    'fnfp_f2_massless': r'$f_n/f_p$',
    'fnfp_f3_massless':r'$f_n/f_p$',
    }


MODELNAME_COLOR = { 
    'SI_Higgs': 'k',
    'Millicharge': 'Magenta',
    'SD_fu': 'DarkBlue',
    'Anapole': 'BlueViolet',
    'Mag.dip.heavy': 'DarkGreen',
    'Mag.dip.light': 'Maroon',
    'Elec.dip.heavy': 'LightGreen',
    'Elec.dip.light': 'Red',
    'f1': 'k',
    'f2_flavor-universal': 'DarkBlue',
    'f2_Higgs': 'blue',
    'f3_flavor-universal': 'Maroon',
    'f3_Higgs': 'Red',
    'LS': 'Magenta'
    }

    
MODELNAME_TEX = { 
    'SI_Higgs': 'SI Higgs',
    'Millicharge': 'Millicharge',
    'SD_fu': 'SD flavor-univ.',
    'Anapole': 'Anapole',
    'Mag.dip.heavy': 'Mag. dip. heavy',
    'Mag.dip.light': 'Mag. dip. light',
    'Elec.dip.heavy': 'Elec. dip. heavy',
    'Elec.dip.light': 'Elec. dip. light',
    'f1': '$f_1$',
    'f2_flavor-universal': '$f_2$ flavor-univ.',
    'f2_Higgs': '$f_2$, Higgs',
    'f3_flavor-universal': '$f_3$ flavor-univ.',
    'f3_Higgs': '$f_3$, Higgs',
    'LS': '$L \cdot S$'
    }
