import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import gaussian_kde as kde
import re
from scipy.integrate import quad
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl


import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec
import glob
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times','Palatino']})
rc('text', usetex=True)


import dmdd_efficiencies as eff
import constants as const
import rate_UV 

mpl.rcParams['xtick.major.size']=8
mpl.rcParams['ytick.major.size']=8
mpl.rcParams['xtick.labelsize']=18
mpl.rcParams['ytick.labelsize']=18


##############################################
def plot_2d_posterior(x, y,xlabel='', ylabel='',
                      input_x=None, input_y=None, input_color='red', fontsize=22,
                      contour_colors=('SkyBlue','Aqua'), alpha=0.7,
                      xmin=None, xmax=None, ymin=None, ymax=None,
                      title='', plot_samples=False, samples_color='gray',
                      contour_lw=2, savefile=None, plot_contours=True,
                      show_legend=False):
    """
    The main chains plotting routine, for visualizing 2d posteriors...
    """
    n = 100

    points = np.array([x,y])
    posterior = kde(points)

    if xmin is None:
        xmin=x.min()

    if xmax is None:
        xmax=x.max()

    if ymin is None:
        ymin=y.min()

    if ymax is None:
        ymax=y.max()


    step_x = ( xmax - xmin ) / n
    step_y = ( ymax - ymin ) / n
    grid_pars = np.mgrid[0:n,0:n]
    par_x = grid_pars[0]*step_x + xmin
    par_y = grid_pars[1]*step_y + ymin
    grid_posterior = grid_pars[0]*0.

    for i in range(n):
        for j in range(n):
            grid_posterior[i][j] = posterior([par_x[i][j],par_y[i][j]])

    pl.figure()
    ax = pl.gca()
    pl.title(title, fontsize=fontsize)
    fig = pl.gcf()
    xlabel = ax.set_xlabel(xlabel,fontsize=fontsize)
    ylabel = ax.set_ylabel(ylabel,fontsize=fontsize)
    if plot_samples:
        pl.plot(x,y,'o',ms=1, mfc=samplescolor, mec=samplescolor)
    pl.plot(input_x,input_y,'x',mew=3,ms=15,color=input_color,label='input',zorder=4)
    if plot_contours:
        percentage_integral = np.array([0.95,0.68,0.])
        contours = 0.* percentage_integral		
        num_epsilon_steps = 1000.
        epsilon = grid_posterior.max()/num_epsilon_steps
        epsilon_marks = np.arange(num_epsilon_steps + 1)
        posterior_marks = grid_posterior.max() - epsilon_marks * epsilon
        posterior_norm = grid_posterior.sum()
        for j in np.arange(len(percentage_integral)):
            for i in epsilon_marks:
                posterior_integral = grid_posterior[np.where(grid_posterior>posterior_marks[i])].sum()/posterior_norm
                if posterior_integral > percentage_integral[j]:
                    break
            contours[j] = posterior_marks[i]
        contours[-1]=grid_posterior.max()
        pl.contour(par_x, par_y, grid_posterior, contours, linewidths=contour_lw,colors='k',zorder=3)
        pl.contourf(par_x, par_y, grid_posterior, contours,colors=contour_colors,alpha=alpha,zorder=2)
        
    pl.xlim(xmin=xmin,xmax=xmax)
    pl.ylim(ymin=ymin,ymax=ymax)
    if show_legend: 
        pl.legend(prop={'size':20},numpoints=1)
    if savefile is None:
        return par_x, par_y, grid_posterior, contours
    else:
        pl.savefig(savefile, bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')
    


##################################
def ln_evidence(filename='/data/verag/dmdd_2014/chains_uv/sim1_Xe_mass_10.00_sigma_si_7862.18_fitfor_mass_sigma_si_logflat_nlive2000/1-stats.dat'):
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

###########################################
def plot_theoryfitdata(Qbins, Qhist, xerr, yerr, Qbins_theory, Qhist_theory, Qhist_fit,
                       filename=None, save_file=True, Ntot=None, fitmodel=None, simmodel=None,
                       experiment=None, labelfont=18, legendfont=16,titlefont=20, mass=None):

    plt.figure()
    title = ''
    if experiment is not None:
        title += experiment
    if Ntot is not None:
        title = title + ' (total events: {}'.format(Ntot)
    if mass is not None:
        title += ', mass: {:.0f} GeV'.format(mass)
    title += ')'
    plt.title(title, fontsize=titlefont)
    
    xlabel = 'Nuclear recoil energy [keV]'
    ylabel = 'Number of events'
    ax = plt.gca()
    fig = plt.gcf()
    xlabel = ax.set_xlabel(xlabel,fontsize=labelfont)
    ylabel = ax.set_ylabel(ylabel,fontsize=labelfont)
    
    label1 = 'True model'
    if simmodel is not None:
        label1 += ' ({})'.format(simmodel)
    plt.plot(Qbins_theory, Qhist_theory,lw=3,color='blue',label=label1)
    plt.errorbar(Qbins, Qhist,xerr=xerr,yerr=yerr,marker='o',color='black',linestyle='None',label='Simulated data')
    label2 = '$\mathcal{L}_{max}$'
    if fitmodel is not None:
        label2 += ' (fit {})'.format(fitmodel)
    plt.plot(Qbins_theory, Qhist_fit,'--',lw=3,color='red',label=label2)

    plt.legend(prop={'size':legendfont},numpoints=1)
    if save_file:
        plt.savefig(filename, bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')
      
############################################
############################################
def spectrum(element, efficiency=eff.efficiency_unit,
             Qmin=1, Qmax=100, exposure=1000, 
             sigma_name='sigma_si', sigma_val=1., fnfp_name=None, fnfp_val=None,
             mass=50.,
             v_esc=540., v_lag=220., v_rms=220., rho_x=0.3):
    """
    NOTE: This is only set up for models in rate_UV.
    """

    qs = np.linspace(Qmin,Qmax,1000)  

    kwargs = {
        'mass': mass,
        sigma_name: sigma_val,
        'v_lag': v_lag,
        'v_rms': v_rms,
        'v_esc': v_esc,
        'rho_x': rho_x,
        'element': element
        }
    if (fnfp_val is not None) and  (fnfp_name is not None):
        kwargs[fnfp_name] = fnfp_val
        
    res = rate_UV.dRdQ(qs, **kwargs) * const.YEAR_IN_S * exposure * efficiency( qs )
    return qs,res

####################
def plot_spectrum(element, efficiency=eff.efficiency_unit,
                    Qmin=1, Qmax=100, exposure=1000, 
                    sigma_name='sigma_si', sigma_val=1.,
                    fnfp_name=None, fnfp_val=None,
                    mass=50.,
                    v_esc=540., v_lag=220., v_rms=220., rho_x=0.3,
                    saveplot=False, title='spectrum', 
                    fontsize=20, filename=None,
                    **kwargs):
    """
    NOTE: This is only set up for models in rate_UV.
    """

    qs,res = spectrum(element, efficiency,
                           Qmin=Qmin, Qmax=Qmax, exposure=exposure,
                           sigma_name=sigma_name, sigma_val=sigma_val,
                           fnfp_name=fnfp_name, fnfp_val=fnfp_val,
                           mass=50.,
                           v_esc=v_esc, v_lag=v_lag, v_rms=v_rms, rho_x=rho_x)

    
    plt.figure()
    plt.plot(qs, res, lw=3, **kwargs)
    xlabel = 'Nuclear recoil energy [keV]'
    ylabel = 'Number of events'
    ax = plt.gca()
    fig = plt.gcf()
    xlabel = ax.set_xlabel(xlabel,fontsize=fontsize)
    ylabel = ax.set_ylabel(ylabel,fontsize=fontsize)
    plt.xlim(xmin=Qmin,xmax=Qmax)
    plt.title(title, fontsize=fontsize)
    if saveplot:
        if filename is None:
            filename = 'spectra_{}_{:.0f}GeV'.format(element,mass)
            if len(label)>0:
                filename += '_{}'.format(label)
            filename += '.pdf'
        pl.savefig(filename, bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')

