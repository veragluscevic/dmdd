import os,os.path,shutil
import numpy as np
import pickle
import matplotlib.pyplot as plt

import dmdd
import dmdd_efficiencies as eff


def check_min_mass(element='fluorine', Qmin=1., v_esc=544., v_lag=220., mx_guess=1.):
    experiment = dmdd.Experiment('test',element,Qmin, 40.,100., eff.efficiency_unit)
    res = experiment.find_min_mass(v_esc=v_esc, v_lag=v_lag, mx_guess=mx_guess)
    print res,'GeV'
    if res<0:
        print 'Problem: try another mx_guess...'
    
    

def make_UVmodels(return_models=False):
    SI_Higgs = dmdd.UV_Model('SI_Higgs', ['mass', 'sigma_si'], fixed_params={'fnfp_si': 1})
    milicharge = dmdd.UV_Model('Milicharge', ['mass', 'sigma_si_massless'], fixed_params={'fnfp_si_massless': 0})
    SD_flavoruniversal = dmdd.UV_Model('SD_fu', ['mass','sigma_sd'], fixed_params={'fnfp_sd': -1.1})
    anapole = dmdd.UV_Model('Anapole', ['mass','sigma_anapole'])
    magdip_heavy = dmdd.UV_Model('Mag.dip.heavy', ['mass','sigma_magdip'])
    magdip_0 = dmdd.UV_Model('Mag.dip.light', ['mass','sigma_magdip_massless'])
    elecdip_heavy = dmdd.UV_Model('Elec.dip.heavy', ['mass','sigma_elecdip'])
    elecdip_0 = dmdd.UV_Model('Elec.dip.light', ['mass','sigma_elecdip_massless'])
    f1 = dmdd.UV_Model('f1', ['mass','sigma_f1'], fixed_params={'fnfp_f1': 1.})
    f2_Higgs = dmdd.UV_Model('f2_Higgs', ['mass','sigma_f2'], fixed_params={'fnfp_f2': -0.05})
    #f2_flavoruniversal = dmdd.UV_Model('f2_flavor-universal', ['mass','sigma_f2'], fixed_params={'fnfp_f2': 1.})
    f3_Higgs = dmdd.UV_Model('f3_Higgs', ['mass','sigma_f3'], fixed_params={'fnfp_f3': -0.05})
    #f3_flavoruniversal = dmdd.UV_Model('f3_flavor-universal', ['mass','sigma_f3'], fixed_params={'fnfp_f3': 1.})
    LS = dmdd.UV_Model('LS', ['mass','sigma_LS'], fixed_params={'fnfp_LS': 0.})

    models = [SI_Higgs, milicharge, SD_flavoruniversal, anapole,
              magdip_heavy, magdip_0, elecdip_heavy, elecdip_0,
              f1, f2_Higgs, f3_Higgs, LS]

    if return_models:
        return models

def make_experiments(return_experiments=False):
    xe = dmdd.Experiment('Xe','xenon',5., 40.,100., eff.efficiency_unit)
    ge = dmdd.Experiment('Ge','germanium',0.4, 100.,100., eff.efficiency_unit)
    if return_experiments:
        return [xe,ge]


def test_MultinestRun(mass=50,test_fits=False):

    SI_Higgs = dmdd.UV_Model('SI_Higgs', ['mass', 'sigma_si'], fixed_params={'fnfp_si': 1})
    elecdip_heavy = dmdd.UV_Model('Elec.dip.heavy', ['mass','sigma_elecdip'])

    experiment = make_experiments(return_experiments=True)
    
    simmodel = SI_Higgs
    fitmodel1 = SI_Higgs
    fitmodel2 = elecdip_heavy
    
    pardic = {'sigma_si': 70.,'mass': mass}
    simname = 'simtest'
    
    testrun1 = dmdd.MultinestRun(simname, experiment, simmodel, pardic,
                                 fitmodel1, prior_ranges={'mass':(1,1000),
                                                          'sigma_si':(0.001,100000),
                                                          'sigma_elecdip':(0.001,100000)})
    data1 = np.loadtxt(testrun1.simulations[0].datafile)

    pardic = {'sigma_si': 70.0007,'mass': mass}
    testrun2 = dmdd.MultinestRun(simname, experiment, simmodel, pardic,
                                 fitmodel2, empty_run=False,
                                 prior_ranges={'mass':(1,1000),
                                                'sigma_si':(0.001,100000),
                                                'sigma_elecdip':(0.001,100000)})
    data2 = np.loadtxt(testrun1.simulations[0].datafile)

    #simulation datafile should be created only for the first instance of MultinestRun:
    assert np.allclose(data1, data2) 

    if test_fits:
    
        testrun1.fit()        
        testrun1.visualize()
        testrun2.fit()
        testrun2.visualize()


        if (not os.path.exists(testrun1.chains_file)) or (not os.path.exists(testrun1.pickle_file)) or (not os.path.exists(testrun1.stats_file)):
            raise AssertionError('Stats or chains or pickle are not created or are erased.')

        plotfile1 = testrun1.chainspath + '2d_posterior_mass_vs_sigma_si.pdf'
        plotfile2 = testrun1.chainspath + '{}_theoryfitdata_Ge.pdf'.format(simname)
        plotfile3 = testrun1.chainspath + '{}_theoryfitdata_Xe.pdf'.format(simname)
        if (not os.path.exists(plotfile1)) or (not os.path.exists(plotfile2)) or (not os.path.exists(plotfile3)):
            raise AssertionError('Plots are not created or are erased.')

        if (not os.path.exists(testrun2.chains_file)) or (not os.path.exists(testrun2.pickle_file)) or (not os.path.exists(testrun2.stats_file)):
            raise AssertionError('Stats or chains or pickle are not created.')

        plotfile1 = testrun2.chainspath + '2d_posterior_mass_vs_sigma_elecdip.pdf'
        plotfile2 = testrun2.chainspath + '{}_theoryfitdata_Ge.pdf'.format(simname)
        plotfile3 = testrun2.chainspath + '{}_theoryfitdata_Xe.pdf'.format(simname)
        if (not os.path.exists(plotfile1)) or (not os.path.exists(plotfile2)) or (not os.path.exists(plotfile3)):
            raise AssertionError('Plots are not created.')


def test_UVrate():
    
    experiment = dmdd.Experiment('Xe','xenon',5., 40.,10000., eff.efficiency_unit)
    models = make_UVmodels(return_models=True)
    mass = 40.
    qs = np.array([15.])
    v_lag = 200.
    v_rms = 100.
    v_esc = 600.
    rho_x = 0.4
    
    sigma_names = {}
    fnfp_names = {}
    fnfp_vals = {}
    for m in models:
        sigma_names[m.name] = m.param_names[1]    
        if len(m.fixed_params)>0:
            fnfp_names[m.name] = m.fixed_params.keys()[0]
            fnfp_vals[m.name] = m.fixed_params.values()[0] 
        else:
            fnfp_names[m.name] = None
            fnfp_vals[m.name] = None

    dRdQs = np.zeros(len(models))
    Rs = np.zeros(len(models))
    for i,m in enumerate(models):
        kwargs = {sigma_names[m.name]:1.}
        if fnfp_names[m.name] is not None:
            kwargs[fnfp_names[m.name]] = fnfp_vals[m.name]

        dRdQs[i] = dmdd.rate_UV.dRdQ(qs, mass=mass, element=experiment.element,
                                        v_lag=v_lag, v_rms=v_rms, v_esc=v_esc, rho_x=rho_x,
                                        **kwargs)
        Rs[i] = dmdd.rate_UV.R(eff.efficiency_unit, mass=mass, element=experiment.element,
                                        Qmin=experiment.Qmin, Qmax=experiment.Qmax,
                                        v_lag=v_lag, v_rms=v_rms, v_esc=v_esc, rho_x=rho_x,
                                        **kwargs)
    #print 'dRdQs = {}\n'.format(dRdQs)
    #print 'Rs = {}\n'.format(Rs)
    dRdQs_correct = [  1.27974652e-12,   1.67031585e-13,   6.28936205e-13,   7.76864477e-13,
                       7.71724584e-13,   5.66164037e-13,   8.40579288e-13,   6.16678247e-13,
                       4.72480605e-13,   2.59857470e-16,   9.59390104e-16,   1.14295679e-13]

    Rs_correct = [  6.15358778e-11,   3.10857259e-11,   3.14982315e-11,   4.14119198e-11,
                    1.82181891e-11,   3.84877268e-11,   2.35638282e-11,   5.50063883e-11,
                    1.34702925e-11,   5.82472177e-15,   1.64213483e-14,   2.26028126e-12]

    assert np.allclose(dRdQs_correct, dRdQs)
    assert np.allclose(Rs_correct, Rs)

    ###
    qs = np.array([8.3,15.7])
    logtest1 = dmdd.rate_UV.loglikelihood(qs, eff.efficiency_unit, mass=mass,
                                              sigma_si=1.,fnfp_si=1.,
                                                element=experiment.element,
                                                Qmin=experiment.Qmin, Qmax=experiment.Qmax,
                                                exposure=experiment.exposure,energy_resolution=True,
                                                v_lag=v_lag, v_rms=v_rms, v_esc=v_esc, rho_x=rho_x)
    logtest2 = dmdd.rate_UV.loglikelihood(qs, eff.efficiency_unit, mass=mass,
                                              sigma_si=1.,fnfp_si=1.,
                                                element=experiment.element,
                                                Qmin=experiment.Qmin, Qmax=experiment.Qmax,
                                                exposure=experiment.exposure,energy_resolution=False,
                                                v_lag=v_lag, v_rms=v_rms, v_esc=v_esc, rho_x=rho_x)

    #print 'logtest1_correct={}'.format(logtest1)
    #print 'logtest2_correct={}'.format(logtest2)
    logtest1_correct=-19.7010967514
    logtest2_correct=-13.4747945274

    assert np.isclose(logtest1_correct, logtest1)
    assert np.isclose(logtest2_correct, logtest2) 



