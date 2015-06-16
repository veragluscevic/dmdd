try:
    import rate_NR 
    import rate_genNR  
    import rate_UV 
    import dmdd_efficiencies as eff
    import dmdd_plot as dp
except ImportError:
    pass
    
import os,os.path,shutil
import pickle
import logging
import time

on_rtd = False

try:
    import numpy.random as random
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import poisson
    from scipy.interpolate import UnivariateSpline as interpolate
    from scipy.optimize import fsolve
except ImportError:
    on_rtd = True
    np = None
    pass

try:
  import pymultinest
except ImportError:
  logging.warning('pymultinest not imported!')

if not on_rtd:
    from constants import *
    from globals import *

try:
    MAIN_PATH = os.environ['DMDD_MAIN_PATH']
except KeyError:
    logging.warning('DMDD_MAIN_PATH environment variable not defined, defaulting to:   ~/.dmdd')
    MAIN_PATH = os.path.expanduser('~/.dmdd') #os.getcwd()

SIM_PATH = MAIN_PATH + '/simulations_uv/'
CHAINS_PATH = MAIN_PATH + '/chains_uv/'
RESULTS_PATH = MAIN_PATH + '/results_uv'

if not os.path.exists(SIM_PATH):
    os.makedirs(SIM_PATH)
if not os.path.exists(CHAINS_PATH):
    os.makedirs(CHAINS_PATH)
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)




"""
Example usage of the objects here defined:

model1 = UV_Model('SI_Higgs', ['mass', 'sigma_si'], fixed_params={'fnfp_si': 1})
model2 = UV_Model('SD_fu', ['mass','sigma_sd'], fixed_params={'fnfp_sd': -1.1})

xe = Experiment('Xe', 'xenon', 5, 40, 1000, eff.efficiency_Xe)

run = MultinestRun('sim', [xe,ge], model1,{'mass':50.,'sigma_si':70.},
                   model2, prior_ranges={'mass':(1,1000), 'sigma_sd':(0.001,1000)})

run.fit()
run.visualize()
"""

class MultinestRun(object):
    """This object controls a single simulated data set and its MultiNest analysis.

    This is a "master" class of ``dmdd`` that makes use of all other
    objects. It takes in experimental parameters, particle-physics
    parameters, and astrophysical parameters, and then generates a
    simulation (if it doesn't already exist), and prepares to perform
    ``MultiNest`` analysis of simulated data.  It has methods to do a
    ``MultiNest`` run (:meth:`MultinestRun.fit()`) and to visualize outputs
    (:meth:`.visualize()`). :class:`Model` used for simulation does not have to
    be the same as the :class:`Model` used for fitting. Simulated spectra from
    multiple experiments will be analyzed jointly if ``MultiNest`` run is
    initialized with a list of appropriate :class:`Experiment` objects. 

    The likelihod function is an argument of the fitting model (:class:`Model`
    object); for UV models it is set to :func:`dmdd.rate_UV.loglikelihood`, and
    for models that would correspond to ``rate_genNR``,
    :func:`dmdd.rate_genNR.loglikelihood` should be used. Both likelihood functions include the
    Poisson factor, and (if ``energy_resolution=True`` of the :class:`Experiment`
    at hand) the factors that evaluate probability of each individual
    event (i.e. each recoil-energy measurement), given the fitting scattering model.  

    MultiNest-related files produced by this object will go to
    a directory, under
    ``$DMDD_MAIN_PATH``, with the name defined by the parameters
    passed.  This directory name will be accessible via ``self.chainspath``
    after the object is initialized.


    :param sim_name:
      The name of the simulation (e.g. 'sim1')
    :type sim_name: ``str``

    :param experiments:
      A list of :class:`Experiment` objects, or a single such object.
    :type experiments: ``list``
      
    :param sim_model:
      The true underlying model for the simulations (name cannot have spaces).
    :type sim_model: :class:`Model`
      
    :param param_values: 
      The values of the parameters for ``sim_model``.  
    :type param_values: ``dict``
      
    :param fit_model: 
      The model for MultiNest to fit to the data.  Does not
      have to be the same as ``sim_model``, but can be. Its name
      cannot have spaces.
    :type fit_model: :class:`Model`

    :param prior_ranges:
      Dictionary of prior ranges for parameters of fit_model.
      e.g. {'mass':(1,1000), 'sigma_si':(0.1,1e4), etc....}
    :type prior_ranges: ``dict``
      
    :param prior: 
      either 'logflat' or 'flat'
    :type prior: ``str``
      
    :param sim_root: 
      The path under which to store the simulations.
    :type sim_root: ``str``
      
    :param chains_root: 
      The path under which to store the Multinest chains.
    :type chains_root: ``str``

    :param force_sim: 
      If `True`, then redo the simulations no matter what.  If `False`,
      then the simulations will be redone if and only if the given simulation
      parameters don't match what has already been simulated for this sim_name.
    :type force_sim: ``bool``
      
    :param asimov: 
      Do asimov simulations.  Not currently implemented.

    :param nbins_asimov: 
      Number of asimov bins.

    :param n_live_points,evidence_tolerance,sampling_efficiency,resume,basename:
      Parameters to pass to MultiNest, defined in the `PyMultiNest documentation
      <https://johannesbuchner.github.io/PyMultiNest/>`_.

    :param silent:
      If ``True``, then print messages will be suppressed.

    :param empty_run:
       if ``True``, then simulations are not initialized.

    """
    def __init__(self, sim_name, experiments, sim_model,
                 param_values, fit_model, prior_ranges,
                 prior='logflat',
                 sim_root=SIM_PATH, chains_root=CHAINS_PATH,
                 force_sim=False,
                 asimov=False, nbins_asimov=20,
                 n_live_points=2000, evidence_tolerance=0.1,
                 sampling_efficiency=0.3, resume=False, basename='1-',
                 silent=False, empty_run=False):
       
        if type(experiments) == Experiment:
            experiments = [experiments]
            
        self.silent = silent
            
        self.sim_name = sim_name
        self.experiments = experiments
        self.sim_model = sim_model
        self.fit_model = fit_model

        self.param_values = param_values
        self.prior_ranges = prior_ranges
        self.prior = prior
        self.simulations = []
    
        self.mn_params = {}
        self.mn_params['n_live_points'] = n_live_points
        self.mn_params['evidence_tolerance'] = evidence_tolerance
        self.mn_params['sampling_efficiency'] = sampling_efficiency
        self.mn_params['resume'] = resume
        self.mn_params['outputfiles_basename'] = basename
    
        self.asimov = asimov
        self.nbins_asimov = nbins_asimov
    
        #build folder names where chains are stored
        self.foldername = sim_name
        experiment_names = [ex.name for ex in experiments]
        experiment_names.sort()
        for experiment in experiment_names:
            self.foldername += '_{}'.format(experiment)

        all_params = dict(param_values)
        for k in self.sim_model.fixed_params:
            all_params[k] = self.sim_model.fixed_params[k]
            
        all_param_names = self.sim_model.param_names + self.sim_model.fixed_params.keys()
        inds = np.argsort(all_param_names)        
        sorted_param_names = np.array(all_param_names)[inds]
        sorted_param_values = [all_params[par] for par in sorted_param_names]
        for parname, parval in zip(sorted_param_names, sorted_param_values):
            self.foldername += '_{}_{:.2f}'.format(parname, parval)      
        self.foldername += '_fitfor'
        
        inds = np.argsort(fit_model.param_names)
        sorted_fitparam_names = np.array(fit_model.param_names)[inds]
        for par in sorted_fitparam_names:
            self.foldername += '_{}'.format(par)


        
        if len(self.fit_model.fixed_params) > 0:
            self.foldername += '_fixed'

            keys = self.fit_model.fixed_params.keys()
            inds = np.argsort(keys)
            sorted_fixedparam_names = np.array(keys)[inds]
            sorted_fixedparam_values = [self.fit_model.fixed_params[par] for par in sorted_fixedparam_names]
            for parname, parval in zip(sorted_fixedparam_names, sorted_fixedparam_values):
                self.foldername += '_{}_{:.2f}'.format(parname, parval)   

        self.foldername += '_{}_nlive{}'.format(prior,self.mn_params['n_live_points'])  
        self.chainspath = '{}/{}/'.format(chains_root,self.foldername)
        self.chainsfile = self.chainspath + '/' + self.mn_params['outputfiles_basename'] + 'post_equal_weights.dat'


        if not empty_run:
            #make simulations, one for each experiment
            for experiment in experiments:
                self.simulations.append(Simulation(sim_name,
                                                   experiment, sim_model,
                                                   param_values,
                                                   path=sim_root,
                                                   force_sim=force_sim,
                                                   asimov=asimov,
                                                   nbins_asimov=nbins_asimov,
                                                   silent=self.silent))

    def return_chains_loglike(self):
        """
        Returns MultiNest chains and equal-weighted posteriors.
        
        """
        data = np.loadtxt(self.chainspath + '/' + self.mn_params['outputfiles_basename'] + 'post_equal_weights.dat')
        return data[:,:-1], data[:,-1]

    def global_bestfit(self):
        """
        Returns maximum a posteriori values for parameters. 
        """
        samples = np.loadtxt(self.chainsfile)
        posterior = samples[:,-1]
        max_index = posterior.argmax()
        return samples[max_index,:-1]
    
    def loglikelihood_total(self,cube,ndim,nparams):
        """
        Log-likelihood function used by MultiNest.
    
        :param cube, ndim, nparams: 
          Params required by MulitNest.
        """
        res = 0
        fit_paramvals = {}
        for i in xrange(ndim):
            par = self.fit_model.param_names[i]
            fit_paramvals[par] = cube[i]
    
        for sim in self.simulations:
            kwargs = self.fit_model.default_rate_parameters.copy()
            for kw,val in fit_paramvals.iteritems():
                kwargs[kw] = val
            for kw,val in sim.experiment.parameters.iteritems():
                kwargs[kw] = val
            kwargs['energy_resolution'] = sim.experiment.energy_resolution
            
            res += self.fit_model.loglikelihood(sim.Q, sim.experiment.efficiency, **kwargs)

        return res

  
  
    def logflat_prior(self, cube, ndim, nparams):
        """
        Logflat prior, passed to MultiNest.

        Converts unit cube into correct parameter values
        based on log-flat prior within range defined by
        ``self.prior_ranges``.
        
        """
        params = self.fit_model.param_names
        for i in xrange(ndim):
            # if the i-th param is fnfp, then the range
            # might go to negative values, must force flat prior:
            if params[i] in FNFP_PARAM_NAMES:
                cube_min,cube_max = self.prior_ranges[params[i]]
                cube[i] = cube[i] * (cube_max - cube_min) + cube_min
            else:
                cube_min,cube_max = self.prior_ranges[params[i]]
                pow = (np.log10(cube_max) - np.log10(cube_min))*cube[i] + np.log10(cube_min)
                cube[i] = 10**pow
  
 
    def flat_prior(self, cube, ndim, nparams):
        """
        Flat prior, passed to MultiNest.

        Converts unit cube into correct parameter values
        based on flat prior within range defined by
        ``self.prior_ranges``.
        
        """
        params = self.fit_model.param_names
        for i in xrange(ndim):
            cube_min,cube_max = self.prior_ranges[params[i]]
            cube[i] = cube[i] * (cube_max - cube_min) + cube_min


    def get_evidence(self):
        """
        Returns evidence from stats file produced by MultiNest.
        """
        filename = self.chainspath + self.mn_params['outputfiles_basename'] + 'stats.dat'
        try:
            fev = open(filename,'r')
        except IOError,e:
            print e
            return 0
    
        line = fev.readline()
        line2 = fev.readline()
    
        line = line.split()
        line2 = line2.split()

        ln_evidence = float(line[5])
        fev.close()
        return ln_evidence
  
  
    def fit(self, force_run=False):
        """
        Runs MultiNest; parameters set by object initialization.

        :param force_run:
            If ``True``, then fit will re-run; by default,
            it will not, unless the simulation data has changed, or chains don't exist.
              
        """
        start = time.time()
        
        # make dictionary of things to be compared:
        # data for each simulation, multinest, and fitting parameters
        pickle_content = {}
        pickle_content['mn_params'] = self.mn_params
        del pickle_content['mn_params']['resume'] 

        inds = np.argsort(self.fit_model.param_names)
        sorted_fitparam_names = np.array(self.fit_model.param_names)[inds]
        pickle_content['fit_param_names'] = sorted_fitparam_names
        pickle_content['fixed_params'] = self.fit_model.fixed_params
        pickle_content['prior'] = self.prior

        pickle_content['prior_ranges'] = {kw:self.prior_ranges[kw] for kw in self.fit_model.param_names}            
    
        pickle_content['data'] = {}
        pickle_content['sim_folders'] = {}
        for sim in self.simulations:
            pickle_content['data'][sim.experiment.name] = sim.Q
            pickle_content['sim_folders'][sim.experiment.name] = sim.file_basename
      

        #define filename of pickle file:
        pickle_file = self.chainspath + 'run_parameters.pkl'
        stats_file = self.chainspath + self.mn_params['outputfiles_basename'] + 'stats.dat'
        chains_file = self.chainspath + self.mn_params['outputfiles_basename'] + 'post_equal_weights.dat'

        self.pickle_file = pickle_file
        self.stats_file = stats_file
        self.chains_file = chains_file
    
        if (not os.path.exists(chains_file)) or (not os.path.exists(pickle_file)) or (not os.path.exists(stats_file)):
            force_run = True
            print 'Chains, pickle, or stats file(s) not found. Forcing run.\n\n'
        else:
            fin = open(pickle_file,'rb')
            pickle_old = pickle.load(fin)
            fin.close()
            try:
                if not compare_dictionaries(pickle_old, pickle_content):
                    force_run = True
                    print 'Run pickle file not a match. Forcing run.\n\n'
            except:
                raise
                          
        # if not all run params are as they need to be, force run.  
        # if force_run is True, run MultiNest:  
        if force_run:
            if os.path.exists(self.chainspath):
                shutil.rmtree(self.chainspath)
            os.makedirs(self.chainspath)
            cwd = os.getcwd()
            os.chdir(self.chainspath) #go where multinest chains will be; do this because that's how multinest works
      
            if self.prior == 'logflat':
                prior_function = self.logflat_prior
            elif self.prior == 'flat':
                prior_function = self.flat_prior
            else:
                raise ValueError('Unknown prior: {}'.format(self.prior))
        
            nparams = len(self.fit_model.param_names)
            pymultinest.run(self.loglikelihood_total, prior_function, nparams, **self.mn_params)
    
            #create pickle file with all info defining this run.
            fout = open(pickle_file,'wb')
            pickle.dump(pickle_content, fout)
            fout.close()
      
            os.chdir(cwd) # go back to whatever directory you were in before
      
 
        #check at the end that everything for created that was supposed to:
        #(this might not be the case, if you ran out of storage space, or if the run was interrupted in the middle.)
        if (not os.path.exists(chains_file)) or (not os.path.exists(pickle_file)) or (not os.path.exists(stats_file)):
            raise RuntimeError('for {}: chains file, or Multinest pickle file, or stats file still does not exist!\n\n'.format(self.chainspath))

        end = time.time()
        if not self.silent:
            print '\n Fit took {:.12f} minutes.\n'.format((end - start) / 60.)
        
    def visualize(self, **kwargs):
        """
        Makes plots of data for each experiment with theoretical and best-fit models.

        Also makes 2-d posteriors for each fitted parameter vs. every other.
        These plots get saved to ``self.chainspath``.

        :param **kwargs:
            Keyword arguments passed to :func:`dmdd.dmdd_plot.plot_2d_posterior`.
        """

        #make theory, data, and fit plots for each experiment:
        fitmodel_dRdQ_params = self.fit_model.default_rate_parameters
        param_values = self.global_bestfit()
        if len(self.fit_model.fixed_params) > 0:
            for k,v in self.fit_model.fixed_params.iteritems():
                fitmodel_dRdQ_params[k] = v 
        for i,k in enumerate(self.fit_model.param_names): 
            fitmodel_dRdQ_params[k] = param_values[i]
                
        for sim in self.simulations:
            filename = self.chainspath + '/{}_theoryfitdata_{}.pdf'.format(self.sim_name, sim.experiment.name)
            Qbins, Qhist, xerr, yerr, Qbins_theory, Qhist_theory, binsize = sim.plot_data(make_plot=False,
                                                                                          return_plot_items=True)
            fitmodel_dRdQ_params['element'] = sim.experiment.element
            fitmodel_dRdQ = sim.model.dRdQ(Qbins_theory,**fitmodel_dRdQ_params)
            Ntot = sim.N
            Qhist_fit = fitmodel_dRdQ*binsize*sim.experiment.exposure*YEAR_IN_S*sim.experiment.efficiency(Qbins_theory)

            if self.fit_model.modelname_tex is None:
                if self.fit_model.name in MODELNAME_TEX:
                    fitmodel_title = MODELNAME_TEX[self.fit_model.name]
                else:
                    fitmodel_title = self.fit_model.name
            if self.sim_model.modelname_tex is None:
                if self.sim_model.name in MODELNAME_TEX:
                    simmodel_title = MODELNAME_TEX[self.sim_model.name]
                else:
                    simmodel_title = self.sim_model.name
            dp.plot_theoryfitdata(Qbins, Qhist, xerr, yerr, Qbins_theory, Qhist_theory, Qhist_fit,
                                    filename=filename, save_file=True, Ntot=Ntot, 
                                    fitmodel=fitmodel_title, simmodel=simmodel_title,
                                    experiment=sim.experiment.name, labelfont=18, legendfont=17,titlefont=20, mass=self.param_values['mass'])
            
        #make 2d posterior plots:
        samples = np.loadtxt(self.chainspath + self.mn_params['outputfiles_basename'] + 'post_equal_weights.dat')
        nparams = len(self.fit_model.param_names)
        if nparams > 1:
            for i,par in enumerate(self.fit_model.param_names):
                for j in np.arange(i+1,nparams):
                    xlabel = PARAM_TEX[self.fit_model.param_names[i]]
                    ylabel = PARAM_TEX[self.fit_model.param_names[j]] 
                    savefile = self.chainspath + '2d_posterior_{}_vs_{}.pdf'.format(self.fit_model.param_names[i], self.fit_model.param_names[j])
              
                    if (self.fit_model.param_names[i] in self.sim_model.param_names):
                        input_x = self.param_values[self.fit_model.param_names[i]]
                    elif (self.fit_model.param_names[i] in self.sim_model.fixed_params.keys()):
                        input_x = self.sim_model.fixed_params[self.fit_model.param_names[i]] 
                    else:
                        input_x = 0.
                    if (self.fit_model.param_names[j] in self.sim_model.param_names):
                        input_y = self.param_values[self.fit_model.param_names[j]]
                    elif (self.fit_model.param_names[j] in self.sim_model.fixed_params.keys()):
                        input_y = self.sim_model.fixed_params[self.fit_model.param_names[j]]
                    else:
                        input_y = 0.
                    dp.plot_2d_posterior(samples[:,i], samples[:,j],
                                         input_x=input_x, input_y=input_y,
                                         savefile=savefile, xlabel=xlabel, ylabel=ylabel, **kwargs)
                    
                    
      
class Simulation(object):
    """
    A simulation of dark-matter direct-detection data under a given experiment and scattering model.

    This object handles a single simulated data set (nuclear recoil energy
    spectrum). It is generaly initialized and used by the :class:`MultinestRun`
    object, but can be used stand-alone.  

    Simulation data will only be generated if a simulation with the right
    parameters and name does not already exist, or if ``force_sim=True`` is
    provided upon :class:`Simulation` initialization; if the data exist, it will
    just be read in.  (Data is a list of nuclear recoil energies of
    "observed" events.) Initializing :class:`Simulation` with given parameters
    for the first time will produce 3 files, located by default at
    ``$DMDD_PATH/simulations`` (or ``./simulations`` if ``$DMDD_PATH`` not
    defined):  

      - .dat file with a list of nuclear-recoil energies (keV), drawn from a
      Poisson distribution with an expected number of events given by the
      underlying scattering model.
      
      - .pkl file with all relevant initialization parameters for record
      
      - .pdf plot of the simulated recoil-energy spectrum with simulated
      data points (with Poisson error bars) on top of the underlying model 

    
    :param name:
        Identifier for simulation (e.g. 'sim1')
    :type name: ``str``

    :param experiment: 
        Experiment for simulation.
    :type experiment: :class:`Experiment`

    :param model: 
        Model under which to simulate data.
    :type model: :class:`Model`

    :param parvals:
       Values of model parameters. Must contain the
       same parameters as ``model``.
    :type parvals:  ``dict``  

    :param path: 
      The path under which to store the simulations.
    :type path: ``str``
      
    :param force_sim: 
      If ``True``, then redo the simulations no matter what.  If ``False``,
      then the simulations will be redone if and only if the given simulation
      parameters don't match what has already been simulated
      for this simulation name.
    :type force_sim: ``bool``

    :param asimov: 
      Do asimov simulations.  Not currently implemented.

    :param nbins_asimov: 
      Number of asimov bins.

    :param plot_nbins:
       Number of bins to bin data in for rate plot.

    :param plot_theory:
       Whether to plot the "true" theoretical rate curve along
       with the simulated data.

    :param silent:
        If ``True``, then print messages will be suppressed.

    """
    def __init__(self, name, experiment,
                 model, parvals,
                 path=SIM_PATH, force_sim=False,
                 asimov=False, nbins_asimov=20,
                 plot_nbins=20, plot_theory=True, 
                 silent=False):
        self.silent = silent
        if not set(parvals.keys())==set(model.param_names):
            raise ValueError('Must pass parameter value dictionary corresponding exactly to model.param_names')
        
        self.model = model #underlying model
        self.experiment = experiment
    
        #build param_values from parvals
        self.param_values = [parvals[par] for par in model.param_names]
        self.param_names = list(self.model.param_names)
        for k,v in self.model.fixed_params.items():
            self.param_values.append(v)
            self.param_names.append(k)
        self.name = name
        self.path = path
        self.asimov = asimov
        self.nbins_asimov = nbins_asimov
    
       
        self.file_basename = '{}_{}'.format(name,experiment.name)

        
        inds = np.argsort(self.param_names)
        sorted_parnames = np.array(self.param_names)[inds]
        sorted_parvals = np.array(self.param_values)[inds]
        for parname, parval in zip(sorted_parnames, sorted_parvals):
            self.file_basename += '_{}_{:.2f}'.format(parname, parval)

        #calculate total expected rate
        dRdQ_params = model.default_rate_parameters.copy()
        allpars = model.default_rate_parameters.copy()
        allpars['simname'] = self.name
        for i,par in enumerate(model.param_names): #model parameters
            dRdQ_params[par] = self.param_values[i]
            allpars[par] = self.param_values[i]
        
        for kw,val in experiment.parameters.iteritems(): #add experiment parameters
            allpars[kw] = val
        dRdQ_params['element'] = experiment.element
        self.dRdQ_params = dRdQ_params
    
        self.model_Qgrid = np.linspace(experiment.Qmin,experiment.Qmax,1000)
        efficiencies = experiment.efficiency(self.model_Qgrid)
        self.model_dRdQ = self.model.dRdQ(self.model_Qgrid,**dRdQ_params)
        R_integrand =  self.model_dRdQ * efficiencies
        self.model_R = np.trapz(R_integrand,self.model_Qgrid)
        self.model_N = self.model_R * experiment.exposure * YEAR_IN_S
    
    
        #create dictionary of all parameters relevant to simulation
        self.allpars = allpars
        self.allpars['experiment'] = experiment.name    

        #record dictionary for relevant coupling normalizations only:
        norm_dict = {}
        for kw in model.param_names:
            if kw in PAR_NORMS:
                norm_dict[kw] = PAR_NORMS[kw]
        self.allpars['norms'] = norm_dict
                          
      
        self.datafile = '{}/{}.dat'.format(self.path,self.file_basename)
        self.plotfile = '{}/{}.pdf'.format(self.path,self.file_basename)
        self.picklefile = '{}/{}.pkl'.format(self.path,self.file_basename)
    
        #control to make sure simulations are forced if they need to be
        if os.path.exists(self.picklefile) and os.path.exists(self.datafile):
            fin = open(self.picklefile,'rb')
            allpars_old = pickle.load(fin)
            fin.close()
            if not compare_dictionaries(self.allpars,allpars_old):
                print('Existing simulation does not match current parameters.  Forcing simulation.\n\n')
                force_sim = True
                
        else:
            print 'Simulation data and/or pickle file does not exist. Forcing simulation.\n\n'
            force_sim = True
      
        if force_sim:
            if asimov:
                raise ValueError('Asimov simulations not yet implemented!')
            else:
                Q = self.simulate_data()
                np.savetxt(self.datafile,Q)
                fout = open(self.picklefile,'wb')
                pickle.dump(self.allpars,fout)
                fout.close()
                self.Q = np.atleast_1d(Q)
                self.N = len(self.Q)
        else:
            if asimov:
                raise ValueError('Asimov simulations not yet implemented!')
            else:
                Q = np.loadtxt(self.datafile)
                self.Q = np.atleast_1d(Q)
                self.N = len(self.Q)
                

        if asimov:
            raise ValueError('Asimov not yet implemented!')
        else:
            self.N = len(self.Q)

        if force_sim or (not os.path.exists(self.plotfile)):
            self.plot_data(plot_nbins=plot_nbins, plot_theory=plot_theory, save_plot=True)
        else:
            if not self.silent:
                print "simulation had %i events (expected %.0f)." % (self.N,self.model_N)
    
        
    def simulate_data(self):
        """
        Do Poisson simulation of data according to scattering model's dR/dQ.
        """
        Nexpected = self.model_N
        if Nexpected > 0:
            npts = 10000
            Nevents = poisson.rvs(Nexpected)
      
            Qgrid = np.linspace(self.experiment.Qmin,self.experiment.Qmax,npts)
            efficiency = self.experiment.efficiency(Qgrid)
            pdf = self.model.dRdQ(Qgrid,**self.dRdQ_params) * efficiency / self.model_R
            cdf = pdf.cumsum()
            cdf /= cdf.max()
            u = random.rand(Nevents)
            Q = np.zeros(Nevents)
            for i in np.arange(Nevents):
                Q[i] = Qgrid[np.absolute(cdf - u[i]).argmin()]
        else:
            Q = np.array([])
            Nevents = 0
            Nexpected = 0

        if not self.silent:
            print "simulated: %i events (expected %.0f)." % (Nevents,Nexpected)
        return Q
	    
    def plot_data(self, plot_nbins=20, plot_theory=True, save_plot=True,
                  make_plot=True, return_plot_items=False):
        """
        Plot simuated data.

        
        
        :param plot_nbins:
            Number of bins for plotting.

        :param plot_theory:
            Whether to overplot the theory rate curve on top of the
            data points.

        :param save_plot:
            Whether to save plot under ``self.plotfile``.

        :param make_plot:
            Whether to make the plot.  No reason really to ever
            be false unless you only want the "plot items"
            returned if ``return_plot_items=True`` is passed.

        :param return_plot_items:
            If ``True``, then function will return lots of things.
            
        """

        Qhist,bins = np.histogram(self.Q,plot_nbins)
        Qbins = (bins[1:]+bins[:-1])/2. 
        binsize = Qbins[1]-Qbins[0] #valid only for uniform gridding.
        Qwidths = (bins[1:]-bins[:-1])/2.
        xerr = Qwidths
        yerr = Qhist**0.5

        Qhist_theory = self.model_dRdQ*binsize*self.experiment.exposure*YEAR_IN_S*self.experiment.efficiency(self.model_Qgrid)
        Qbins_theory = self.model_Qgrid

        if make_plot:
            plt.figure()
            plt.title('%s (total events = %i)' % (self.experiment.name,self.N), fontsize=18)
            xlabel = 'Nuclear recoil energy [keV]'
            ylabel = 'Number of events'
            ax = plt.gca()
            fig = plt.gcf()
            xlabel = ax.set_xlabel(xlabel,fontsize=18)
            ylabel = ax.set_ylabel(ylabel,fontsize=18)
            if plot_theory:
                if self.model.name in MODELNAME_TEX.keys():
                    label='True model ({})'.format(MODELNAME_TEX[self.model.name])
                else:
                    label='True model'
                plt.plot(Qbins_theory, Qhist_theory,lw=3,
                         color='blue',
                         label=label)
            plt.errorbar(Qbins, Qhist,xerr=xerr,yerr=yerr,marker='o',color='black',linestyle='None',label='Simulated data')
     
            plt.legend(prop={'size':20},numpoints=1)
            if save_plot:
                plt.savefig(self.plotfile, bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')


        if return_plot_items:
            return Qbins, Qhist, xerr, yerr, Qbins_theory, Qhist_theory, binsize
    
class Model(object):
    """
    A generic class describing a dark-matter scattering model.

    This object facilitates handling of a "hypothesis" that describes the
    scattering interaction at hand (to be used either to simulate recoil
    spectra, or to fit them). There is an option
    to give any parameter a fixed value, which will not be varied
    if the model is used to fit data. 
    
    Subclassed by :class:`UV_Model`.
    
    :param name: 
        Name of the model, matching the operator(s) name.
        It cannot have spaces.
    :type name: ``str``

    :param param_names:
        Names of the parameters.
    :type param_names: ``list``

    :param dRdQ_fn:
        Appropriate rate function.
    :type dRdQ_fn: ``function``

    :param loglike_fn:
        Function that returns the log-likelihood of an array of event energies, given experimental
        and astrophysical parameters. Must take ``Q, eff_fn, **kwargs``
        as arguments.
    :type loglike_fn: ``function``

    :param default_rate_parameters: 
        Default parameters to be passed to rate function.
    :type default_rate_parameters: ``dict``

    :param tex_names: 
        Dictionary of LaTeX names of parameters.
    :type tex_names: ``dict``
        
    :param fixed_params:
        Parameters of model that are not intended to be fit for. 
    :type fixed_params: ``dict``
    
    """
    
    def __init__(self, name, param_names,
                 dRdQ_fn, loglike_fn,
                 default_rate_parameters, tex_names=None,
                 fixed_params=None,
                 modelname_tex=None):
        """
            fixed_params: dictionary
            
            tex_names is dictionary
        """
        self.name = name
        self.param_names = param_names
        self.dRdQ = dRdQ_fn
        self.loglikelihood = loglike_fn
        self.default_rate_parameters = default_rate_parameters
        if fixed_params is None:
            fixed_params = {}
        self.fixed_params = fixed_params
        for k,v in fixed_params.items():
            self.default_rate_parameters[k] = v
    
        if tex_names is None:
            tex_names = {p:p for p in param_names}
        self.tex_names = tex_names
        self.modelname_tex = modelname_tex

      
class UV_Model(Model):
    """
    Subclass of Model implementing UV-complete scattering models.
    Rate function and log-likelihood function are taken from the
    ``rate_UV`` module.
    
    """
    def __init__(self,name,param_names,**kwargs):
        default_rate_parameters = dict(mass=50., sigma_si=0., sigma_sd=0., sigma_anapole=0., sigma_magdip=0., sigma_elecdip=0.,
                                    sigma_LS=0., sigma_f1=0., sigma_f2=0., sigma_f3=0.,
                                    sigma_si_massless=0., sigma_sd_massless=0.,
                                    sigma_anapole_massless=0., sigma_magdip_massless=0.,  sigma_elecdip_massless=0.,
                                    sigma_LS_massless=0.,  sigma_f1_massless=0.,  sigma_f2_massless=0.,  sigma_f3_massless=0.,
                                    fnfp_si=1.,  fnfp_sd=1.,
                                    fnfp_anapole=1.,  fnfp_magdip=1.,  fnfp_elecdip=1.,
                                    fnfp_LS=1.,  fnfp_f1=1.,  fnfp_f2=1.,  fnfp_f3=1.,
                                    fnfp_si_massless=1.,  fnfp_sd_massless=1.,
                                    fnfp_anapole_massless=1.,  fnfp_magdip_massless=1.,  fnfp_elecdip_massless=1.,
                                    fnfp_LS_massless=1.,  fnfp_f1_massless=1.,  fnfp_f2_massless=1.,  fnfp_f3_massless=1.,
                                    v_lag=220.,  v_rms=220.,  v_esc=544.,  rho_x=0.3)
        
        Model.__init__(self,name,param_names,
                       rate_UV.dRdQ,
                       rate_UV.loglikelihood,
                       default_rate_parameters,
                       **kwargs)  

        
class Experiment(object):
    """
    An object representing a dark-matter direct-detection experiment.

    This object packages all the information that defines a single
    "experiment". For statistical analysis, a list of these objects is
    passed to initialize an instance of a :class:`MultinestRun` object, or to
    initialize an instance of a :class:`Simulation` object.  It can also be used
    on its own to explore the capabilities of an
    experiment with given characteristics. Experiments set up here can either have perfect energy
    resolution in a given analysis window, or no resolution (controlled by
    the parameter ``energy_resolution``, default being ``True``).

    
    :param name:
        Name of experiment.
    :type name: ``str``

    :param element:
        Detector target element.  Only single-element targets currently supported.
    :type element: ``str``
        
    :param Qmin,Qmax:
        Nuclear-recoil energy range of experiment [in keV].

    :param exposure:
        Total exposure of experiment [kg-years].

    :param efficiency_fn:
        Efficiency as a function of nuclear recoil energy.
    :type efficiency_fn: ``function``

    :param tex_name:
        Optional; provide this if you want a specific tex name on plots.
    :type tex_name: ``str``
        
    :param energy_resolution:
        If ``True``, then the energy of recoil events will be taken
        into account in likelihood analyses using this experiment;
        otherwise, not (e.g., for bubble-chamber experiments).
    :type energy_resolution: ``bool``

    """
    #pass the name of element instead of A, use natural isotope abundances for now.
    def __init__(self, name, element, Qmin, Qmax, exposure,
                 efficiency_fn, 
                 tex_name=None, energy_resolution=True):
        """
        Exposure in kg-yr
        """
      
        #implement exps with multiple nuclei?
        self.energy_resolution = energy_resolution
        self.name = name
        self.Qmin = Qmin
        self.Qmax = Qmax
        self.exposure = exposure
        self.element = element
        self.efficiency = efficiency_fn
        self.parameters = {'Qmin':Qmin,
                           'Qmax':Qmax,
                           'exposure':exposure,
                           'element':element}
        if tex_name is None:
            tex_name = name

    def NminusNbg(self, sigma_val, sigma_name='sigma_si', fnfp_name='fnfp_si', fnfp_val=None,
                    mass=50., Nbackground=4,
                    v_esc=540., v_lag=220., v_rms=220., rho_x=0.3):
        """
        Expected number of events minus background

        :param sigma_val:
            Scattering cross-section for interaction with proton [cm^2]

        :param sigma_name:
            Which sigma this corresponds to (i.e., which argument of :func:`rate_UV.R`)
        :type fnfp_name: ``str``

        :param fnfp_name:
            Which fnfp to use.
        :type fnfp_name: ``str``

        :param fnfp_val:
            Value of fnfp (optional).

        :param mass:
            Dark-matter particle mass in GeV.

        :param Nbackground:
            Number of background events expected.

        :param v_esc,v_lag,v_rms,rho_x:
            Passed to :func:`rate_UV.R`.
            
        """
        kwargs = {
            'mass': mass,
            sigma_name: sigma_val,
            'v_lag': v_lag,
            'v_rms': v_rms,
            'v_esc': v_esc,
            'rho_x': rho_x,
            'element': self.element,
            'Qmin': self.Qmin,
            'Qmax': self.Qmax,
            }

        if fnfp_val is not None:
            kwargs[fnfp_name] = fnfp_val
            
        Nexpected = rate_UV.R(self.efficiency, **kwargs) * YEAR_IN_S * self.exposure
        return Nexpected - Nbackground

    def sigma_limit(self, sigma_name='sigma_si', fnfp_name='fnfp_si', fnfp_val=None,
                    mass=50., Nbackground=4, sigma_guess = 1.e10, mx_guess=1.,
                    v_esc=540., v_lag=220., v_rms=220., rho_x=0.3):
        """
        Returns value of sigma at which expected number of dark-matter induced recoil events is equal to the number of expected background events, N = Nbg, in order to get a rough projected exclusion for this experiment.

        :param sigma_guess:
            Initial guess for solver.

        :param mx_guess:
            Initial guess for dark-matter particle mass in order to find the minimum mass
            detectable from experiment (:meth:`Experiment.find_min_mass`).
        
        For other arguments, see :meth:`Experiment.NminusNbg`
        """
        if mass < self.find_min_mass(mx_guess = mx_guess):
            return np.inf
        res = fsolve(self.NminusNbg, sigma_guess, xtol=1e-3, args=(sigma_name, fnfp_name, fnfp_val,
                                                                    mass, Nbackground,
                                                                    v_esc, v_lag, v_rms, rho_x))
        return res[0]

    def sigma_exclusion(self, sigma_name, fnfp_name='fnfp_si', fnfp_val=None,
                        mass_max=5000, Nbackground=4, mx_guess=1., sigma_guess=1.e10,
                        v_esc=540., v_lag=220., v_rms=220., rho_x=0.3,
                        mass_spacing='log', nmass_points=100, make_plot=False,ymax=None):
        """
        Makes exclusion curve for a chosen sigma parameter.

        Calculates :meth:`Experiment.sigma_limit` for a grid of masses,
        and interpolates.
        
        :param sigma_name:
            Name of cross-section to exclude.
        :type sigma_name: ``str``

        :param mass_spacing:
            'log' (logarithmic) or 'lin' (linear) spacing for mass grid.

        :param nmass_points:
            Number of points to calculate for mass grid.

        :param make_plot:
            Whether to make the plot. If ``False``, then function
            will return arrays of ``mass, sigma``.

        :param ymax:
            Set the y maximum of plot axis.
        
        For other parameters, see :meth:`Experiment.sigma_limit`
        """
        mass_min = self.find_min_mass(v_esc=v_esc, v_lag=v_lag, mx_guess=mx_guess)
        if mass_spacing=='lin':
            masses = np.linspace(mass_min, mass_max, nmass_points)
        else:
            masses = np.logspace(np.log10(mass_min), np.log10(mass_max), nmass_points)
        sigmas = np.zeros(nmass_points)
        
        for i,m in enumerate(masses):
            sigmas[i] = self.sigma_limit(sigma_name=sigma_name, fnfp_name=fnfp_name, fnfp_val=fnfp_val,
                                            mass=m, Nbackground=Nbackground,sigma_guess=sigma_guess,
                                            v_esc=v_esc, v_lag=v_lag, v_rms=v_rms, rho_x=rho_x)

        if make_plot:
            plt.figure()
            plt.loglog(masses, sigmas * PAR_NORMS[sigma_name], lw=3, color='k')
            plt.xlabel(PARAM_TEX['mass'])
            plt.ylabel(PARAM_TEX[sigma_name])
            figtitle = 'Limits from {}'.format(self.name)
            if fnfp_val is not None:
                figtitle += ' (for $f_n/f_p = {}$)'.format(fnfp_val)
            plt.title(figtitle, fontsize=18)
            plt.ylim(ymax=ymax)
            plt.show()

        return masses, sigmas
        

    def VminusVesc(self, mx, v_esc=540., v_lag=220.):
        """
        This function returns the value of the minimum velocity needed to produce
        recoil of energy Qmin, minus escape velocity in Galactic frame.

        See Eq 2.3 in (Gluscevic & Peter, 2014).

        Zero of this function gives minimal dark-matter particle mass mx that can be detected with this
        Experiment. This is usually called by :meth:`Experiment.find_min_mass`.

        :param mx:
   	    WIMP mass [GeV]
        
        :param v_esc: 
           escape velocity in Galactic frame [km/sec]
                 
        :param v_lag:
          rotational velocity of the Milky Way [km/sec]

        :return:
            Vmin - Vesc

        """
        
        v_esc_lab = v_esc + v_lag
    
        mT = NUCLEAR_MASSES[self.element]
        q = self.Qmin / GEV_IN_KEV
        mu = mT * mx / ( mT + mx )
        
        res = mT * q /( 2. * mu**2 ) - (v_esc_lab / C_KMSEC)**2.
        return res
    
    def find_min_mass(self, v_esc=540., v_lag=220., mx_guess=1.):
        """
        This finds the minimum dark-matter particle mass detectable with this experiment,
        by finding a zero of VminusVesc.


        :param mx_guess:
          guess-value [GeV].

        Other parameters documented in :meth:`Experiment.VminusVesc`.
        """
        
        res = fsolve(self.VminusVesc, mx_guess, xtol=1e-3, args=(v_esc, v_lag))
        
        return res[0]
            
        

        
############################################
############################################

def compare_dictionaries(d1,d2,debug=False,rtol=1e-5):
    """Returns True if dictionaries are identical; false if not.
    
    It works with multi-level dicts.
    If elements are arrays, then numpy's array compare is used
    """
    if not set(d1.keys())==set(d2.keys()):
        if debug:
            print 'keys not equal.'
            print d1.keys(),d2.keys()
        return False
    for k in d1.keys():
        if type(d1[k]) != type(d2[k]):
            return False
        elif type(d1[k])==dict:
            if not compare_dictionaries(d1[k],d2[k],rtol=rtol):
                if debug:
                    print 'dictionaries not equal for {}.'.format(k)
                return False
        elif type(d1[k])==type(np.array([1,2])):
            if not np.all(d1[k]==d2[k]):
                if debug:
                    print 'arrays for {} not equal:'.format(k)
                    print d1[k], d2[k]
                return False

        #make sure floats are close in value, down to rtol relative precision:
        elif type(d1[k])==float:
            if not np.isclose(d1[k], d2[k], rtol=rtol):
                return False
        else:
            if d1[k] != d2[k]:
                if debug:
                    'values for {} not equal: {}, {}.'.format(k,d1[k],d2[k])
                return False
    return True


############################################
############################################
def Nexpected(element, Qmin, Qmax, exposure, efficiency,
              sigma_name, sigma_val, fnfp_name=None, fnfp_val=None,
              mass=50.,
              v_esc=540., v_lag=220., v_rms=220., rho_x=0.3):
    """
    NOTE: This is only set up for models in rate_UV.
    """
              

    kwargs = {
        'mass': mass,
        sigma_name: sigma_val,
        'v_lag': v_lag,
        'v_rms': v_rms,
        'v_esc': v_esc,
        'rho_x': rho_x,
        'element': element,
        'Qmin': Qmin,
        'Qmax': Qmax
        }
    if (fnfp_val is not None) and  (fnfp_name is not None):
        kwargs[fnfp_name] = fnfp_val
        
    res = rate_UV.R(efficiency, **kwargs) * YEAR_IN_S * exposure
    return res





############################################
############################################
