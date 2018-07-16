"""
plot_spec.py
CASA Data Reduction Pipeline - Plot continuum SED and
line-to-continuum ratio vs frequency
Trey V. Wenger September 2017 - V1.0
Trey V. Wenger June 2018 - ignore multiple components except (a)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

__VERSION__ = "1.0"

def main(specfile,title=None,sed_plot='cont_sed.pdf',fluxtype='flux',
         line2cont_plot='line2cont.pdf',te_plot='te.pdf',
         xmin=4000,xmax=10000):
    """
    Read output file from calc_te.py and plot continuum flux vs.
    frequency and line-to-continuum ratio vs. frequency
    
    Inputs:
    specfile = file containing calc_te.py output
    title = title for plots
    sed_plot = where you want to save the continuum SED plot
    line2cont_plot = where you want to save the line-to-continuum
                     ratio vs. frequency plot
    te_plot = where you want to save the te vs frequency plot
    xmin = minimum frequency axis (MHz)
    xmax = maximum frequency axis (MHz)

    Returns:
      Nothing
    """
    #
    # Read data
    #
    data = np.genfromtxt(specfile,dtype=None,names=True)
    #
    # Ignore multiple components except (a)
    #
    multcomps = ['(b)','(c)','(d)','(e)']
    is_multcomp = np.array([lineid[-3:] in multcomps for lineid in data['lineid']])
    data = data[~is_multcomp]
    #
    # Determine which are Hnalpha lines and which are stacked
    #
    is_Hnalpha = np.array([lineid[0] == 'H' for lineid in data['lineid']])
    is_stacked = ~is_Hnalpha
    is_all = np.where(['all' in lineid for lineid in data['lineid']])[0]
    if len(is_all) > 0:
        is_all = is_all[0]
    else:
        is_all = None
    # plot limits
    xfit = np.linspace(xmin,xmax,100)
    #
    # Plot continuum SED
    #
    print("Plotting continuum SED...")
    plt.ioff()
    fig, ax = plt.subplots()
    plt.xlim(xmin,xmax)
    isnan = ((np.isnan(data['cont'][is_Hnalpha])) | (data['cont'][is_Hnalpha] < 0.))
    ax.errorbar(data['frequency'][is_Hnalpha][~isnan],
                data['cont'][is_Hnalpha][~isnan],
                yerr=data['rms'][is_Hnalpha][~isnan],fmt='o',color='k')
    if len(data['cont'][is_Hnalpha][~isnan]) > 4:
        # fit line 
        fit,cov = np.polyfit(data['frequency'][is_Hnalpha][~isnan],
                             data['cont'][is_Hnalpha][~isnan],1,
                             w=1./data['rms'][is_Hnalpha][~isnan],cov=True)
        ax.plot(xfit,np.poly1d(fit)(xfit),'k-',zorder=10,
                label=r'$F_{{\nu,\rm C}} = ({0:.3f}\pm{1:.3f})\nu + ({2:.1f}\pm{3:.1f})$'.format(fit[0],np.sqrt(cov[0,0]),fit[1],np.sqrt(cov[1,1])))
        # fit power law
        ispos = data['cont'][is_Hnalpha][~isnan] > 0.
        if len(data['cont'][is_Hnalpha][~isnan][ispos]) > 4:
            fit,cov = np.polyfit(np.log10(data['frequency'][is_Hnalpha][~isnan][ispos]),
                                np.log10(data['cont'][is_Hnalpha][~isnan][ispos]),1,
                                w=data['cont'][is_Hnalpha][~isnan][ispos]*np.log(10.)/data['rms'][is_Hnalpha][~isnan][ispos],cov=True)
            yfit = lambda x: 10.**np.poly1d(fit)(np.log10(x))
            ax.plot(xfit,yfit(xfit),'k--',zorder=10,
                    label=r'$F_{{\nu,\rm C}} \propto \nu^{{({0:.2f}\pm{1:.2f})}}$'.format(fit[0],np.sqrt(cov[0,0])))
        ax.legend(loc='upper right',fontsize=10)
    # plot stacked
    xlim = ax.get_xlim()
    if is_all is not None:
        ax.plot(xlim,[data['cont'][is_all],data['cont'][is_all]],'k-')
        ax.fill_between(xlim,
                        [data['cont'][is_all]-data['rms'][is_all],data['cont'][is_all]-data['rms'][is_all]],
                        [data['cont'][is_all]+data['rms'][is_all],data['cont'][is_all]+data['rms'][is_all]],
                        color='k',alpha=0.5,edgecolor='none')
    ax.set_xlim(xlim)
    ax.set_xlabel('Frequency (MHz)')
    if fluxtype == 'flux':
        ax.set_ylabel('Flux (mJy)')
    else:
        ax.set_ylabel('Flux Density (mJy/beam)')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(sed_plot)
    plt.close(fig)
    print("Done!")
    #
    # Plot line-to-continuum ratio vs frequency
    #
    print("Plotting line-to-continuum ratio...")
    # fix small errors
    data['e_line2cont'][data['e_line2cont'] == 0.0] = 0.001
    fig, ax = plt.subplots()
    plt.xlim(xmin,xmax)
    isnan = ((np.isnan(data['line2cont'][is_Hnalpha])) | (data['line2cont'][is_Hnalpha] < 0.))
    ax.errorbar(data['frequency'][is_Hnalpha][~isnan],data['line2cont'][is_Hnalpha][~isnan],
                yerr=data['e_line2cont'][is_Hnalpha][~isnan],fmt='o',color='k')
    # limits
    ax.plot(data['frequency'][is_Hnalpha][isnan],
            3.*data['rms'][is_Hnalpha][isnan]/data['cont'][is_Hnalpha][isnan],
            linestyle='none',marker=r'$\downarrow$',markersize=20,color='k')
    if len(data['line2cont'][is_Hnalpha][~isnan]) > 4:
        # fit line 
        fit,cov = np.polyfit(data['frequency'][is_Hnalpha][~isnan],
                             data['line2cont'][is_Hnalpha][~isnan],1,
                             w=1./data['e_line2cont'][is_Hnalpha][~isnan],cov=True)
        ax.plot(xfit,np.poly1d(fit)(xfit),'k-',zorder=10,
                label=r'$F_{{\nu,\rm L}}/F_{{\nu,\rm C}} = ({0:.3f}\pm{1:.3f})\times10^{{-3}}\nu + ({2:.3f}\pm{3:.3f})$'.format(fit[0]*1.e3,np.sqrt(cov[0,0])*1.e3,fit[1],np.sqrt(cov[1,1])))
        # fit power law
        ispos = data['line2cont'][is_Hnalpha][~isnan] > 0.
        if len(data['line2cont'][is_Hnalpha][~isnan][ispos]) > 4:
            fit,cov = np.polyfit(np.log10(data['frequency'][is_Hnalpha][~isnan][ispos]),
                                np.log10(data['line2cont'][is_Hnalpha][~isnan][ispos]),1,
                                w=data['line2cont'][is_Hnalpha][~isnan][ispos]*np.log(10.)/data['e_line2cont'][is_Hnalpha][~isnan][ispos],cov=True)
            yfit = lambda x: 10.**np.poly1d(fit)(np.log10(x))
            ax.plot(xfit,yfit(xfit),'k--',zorder=10,
                    label=r'$F_{{\nu,\rm L}}/F_{{\nu,\rm C}} \propto \nu^{{({0:.2f}\pm{1:.2f})}}$'.format(fit[0],np.sqrt(cov[0,0])))
        ax.legend(loc='upper left',fontsize=10)
    # plot stacked
    xlim = ax.get_xlim()
    if is_all is not None:
        ax.plot(xlim,
                [data['line2cont'][is_all],data['line2cont'][is_all]],'k-')
        ax.fill_between(xlim,
                        [data['line2cont'][is_all]-data['e_line2cont'][is_all],data['line2cont'][is_all]-data['e_line2cont'][is_all]],
                        [data['line2cont'][is_all]+data['e_line2cont'][is_all],data['line2cont'][is_all]+data['e_line2cont'][is_all]],
                        color='k',alpha=0.5,edgecolor='none')
    ax.set_xlim(xlim)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Line-to-Continuum Ratio')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(line2cont_plot)
    plt.close(fig)
    print("Done!")
    #
    # Plot Te vs frequency
    #
    print("Plotting electron temperature...")
    # fix small errors
    data['e_elec_temp'][data['e_elec_temp'] == 0.0] = 0.001
    fig, ax = plt.subplots()
    plt.xlim(xmin,xmax)
    isnan = ((np.isnan(data['elec_temp'][is_Hnalpha])) | (data['elec_temp'][is_Hnalpha] < 0.))
    ax.errorbar(data['frequency'][is_Hnalpha][~isnan],data['elec_temp'][is_Hnalpha][~isnan],
                yerr=data['e_elec_temp'][is_Hnalpha][~isnan],fmt='o',color='k')
    if len(data['elec_temp'][is_Hnalpha][~isnan]) > 4:
        # fit line 
        fit,cov = np.polyfit(data['frequency'][is_Hnalpha][~isnan],
                             data['elec_temp'][is_Hnalpha][~isnan],1,
                             w=1./data['e_elec_temp'][is_Hnalpha][~isnan],cov=True)
        ax.plot(xfit,np.poly1d(fit)(xfit),'k-',zorder=10,
                label=r'$T_e = ({0:.2f}\pm{1:.2f})\nu + ({2:.1f}\pm{3:.1f})$'.format(fit[0],np.sqrt(cov[0,0]),fit[1],np.sqrt(cov[1,1])))
        # fit power law
        ispos = data['elec_temp'][is_Hnalpha][~isnan] > 0.
        if len(data['elec_temp'][is_Hnalpha][~isnan][ispos]) > 4:
            fit,cov = np.polyfit(np.log10(data['frequency'][is_Hnalpha][~isnan][ispos]),
                                np.log10(data['elec_temp'][is_Hnalpha][~isnan][ispos]),1,
                                w=data['elec_temp'][is_Hnalpha][~isnan][ispos]*np.log(10.)/data['e_elec_temp'][is_Hnalpha][~isnan][ispos],cov=True)
            yfit = lambda x: 10.**np.poly1d(fit)(np.log10(x))
            ax.plot(xfit,yfit(xfit),'k--',zorder=10,
                    label=r'$T_e \propto \nu^{{({0:.2f}\pm{1:.2f})}}$'.format(fit[0],np.sqrt(cov[0,0])))
        ax.legend(loc='upper left',fontsize=10)
    # plot stacked
    xlim = ax.get_xlim()
    if is_all is not None:
        ax.plot(xlim,
                [data['elec_temp'][is_all],data['elec_temp'][is_all]],'k-')
        ax.fill_between(xlim,
                        [data['elec_temp'][is_all]-data['e_elec_temp'][is_all],data['elec_temp'][is_all]-data['e_elec_temp'][is_all]],
                        [data['elec_temp'][is_all]+data['e_elec_temp'][is_all],data['elec_temp'][is_all]+data['e_elec_temp'][is_all]],
                        color='k',alpha=0.5,edgecolor='none')
    ax.set_xlim(xlim)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Electron Temperature (K)')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(te_plot)
    plt.close(fig)
    print("Done!")
    plt.ion()
