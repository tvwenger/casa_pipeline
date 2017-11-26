"""
plot_spec.py
CASA Data Reduction Pipeline - Plot continuum SED and
line-to-continuum ratio vs frequency
Trey V. Wenger September 2017 - V1.0
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

__VERSION__ = "1.0"

def main(specfile,sed_plot='cont_sed.pdf',
         line2cont_plot='line2cont.pdf',te_plot='te.pdf'):
    """
    Read output file from calc_te.py and plot continuum flux vs.
    frequency and line-to-continuum ratio vs. frequency
    
    Inputs:
    specfile = file containing calc_te.py output
    sed_plot = where you want to save the continuum SED plot
    line2cont_plot = where you want to save the line-to-continuum
                     ratio vs. frequency plot
    te_plot = where you want to save the te vs frequency plot

    Returns:
      Nothing
    """
    #
    # Read data
    #
    data = np.genfromtxt(specfile,dtype=None,names=True)
    #
    # Determine which are Hnalpha lines and which are stacked
    #
    is_Hnalpha = np.array([lineid[0] == 'H' for lineid in data['lineid']])
    is_stacked = ~is_Hnalpha
    is_all = np.where(['all' in lineid for lineid in data['lineid']])[0][0]
    #
    # Plot continuum SED
    #
    plt.ioff()
    fig, ax = plt.subplots()
    ax.errorbar(data['frequency'][is_Hnalpha],data['cont'][is_Hnalpha],
                yerr=data['rms'][is_Hnalpha],fmt='o',color='k')
    # fit line 
    fit,cov = np.polyfit(data['frequency'][is_Hnalpha],data['cont'][is_Hnalpha],1,w=1./data['rms'][is_Hnalpha],cov=True)
    ax.plot(data['frequency'][is_Hnalpha],np.poly1d(fit)(data['frequency'][is_Hnalpha]),'k-',zorder=10,
            label=r'$F_{{\nu,\rm C}} = ({0:.3f}\pm{1:.3f})\nu + ({2:.1f}\pm{3:.1f})$'.format(fit[0],np.sqrt(cov[0,0]),fit[1],np.sqrt(cov[1,1])))
    # fit power law
    fit,cov = np.polyfit(np.log10(data['frequency'][is_Hnalpha]),np.log10(data['cont'][is_Hnalpha]),1,w=data['cont'][is_Hnalpha]*np.log(10.)/data['rms'][is_Hnalpha],cov=True)
    yfit = lambda x: 10.**np.poly1d(fit)(np.log10(x))
    ax.plot(data['frequency'][is_Hnalpha],yfit(data['frequency'][is_Hnalpha]),'k--',zorder=10,
            label=r'$F_{{\nu,\rm C}} \propto \nu^{{({0:.2f}\pm{1:.2f})}}$'.format(fit[0],np.sqrt(cov[0,0])))
    ax.legend(loc='upper right',fontsize=10)
    # plot stacked
    xlim = ax.get_xlim()
    ax.plot(xlim,
            [data['cont'][is_all],data['cont'][is_all]],'k-')
    ax.fill_between(xlim,
                    [data['cont'][is_all]-data['rms'][is_all],data['cont'][is_all]-data['rms'][is_all]],
                    [data['cont'][is_all]+data['rms'][is_all],data['cont'][is_all]+data['rms'][is_all]],
                    color='k',alpha=0.5,edgecolor='none')
    ax.set_xlim(xlim)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Flux Density (mJy/beam)')
    fig.tight_layout()
    fig.savefig(sed_plot)
    plt.close(fig)
    #
    # Plot line-to-continuum ratio vs frequency
    #
    fig, ax = plt.subplots()
    ax.errorbar(data['frequency'][is_Hnalpha],data['line2cont'][is_Hnalpha],
                yerr=data['e_line2cont'][is_Hnalpha],fmt='o',color='k')
    # fit line 
    fit,cov = np.polyfit(data['frequency'][is_Hnalpha],data['line2cont'][is_Hnalpha],1,w=1./data['e_line2cont'][is_Hnalpha],cov=True)
    ax.plot(data['frequency'][is_Hnalpha],np.poly1d(fit)(data['frequency'][is_Hnalpha]),'k-',zorder=10,
            label=r'$F_{{\nu,\rm L}}/F_{{\nu,\rm C}} = ({0:.3f}\pm{1:.3f})\times10^{{-3}}\nu + ({2:.3f}\pm{3:.3f})$'.format(fit[0]*1.e3,np.sqrt(cov[0,0])*1.e3,fit[1],np.sqrt(cov[1,1])))
    # fit power law
    fit,cov = np.polyfit(np.log10(data['frequency'][is_Hnalpha]),np.log10(data['line2cont'][is_Hnalpha]),1,w=data['line2cont'][is_Hnalpha]*np.log(10.)/data['e_line2cont'][is_Hnalpha],cov=True)
    yfit = lambda x: 10.**np.poly1d(fit)(np.log10(x))
    ax.plot(data['frequency'][is_Hnalpha],yfit(data['frequency'][is_Hnalpha]),'k--',zorder=10,
            label=r'$F_{{\nu,\rm L}}/F_{{\nu,\rm C}} \propto \nu^{{({0:.2f}\pm{1:.2f})}}$'.format(fit[0],np.sqrt(cov[0,0])))
    ax.legend(loc='upper left',fontsize=10)
    # plot stacked
    xlim = ax.get_xlim()
    ax.plot(xlim,
            [data['line2cont'][is_all],data['line2cont'][is_all]],'k-')
    ax.fill_between(xlim,
                    [data['line2cont'][is_all]-data['e_line2cont'][is_all],data['line2cont'][is_all]-data['e_line2cont'][is_all]],
                    [data['line2cont'][is_all]+data['e_line2cont'][is_all],data['line2cont'][is_all]+data['e_line2cont'][is_all]],
                    color='k',alpha=0.5,edgecolor='none')
    ax.set_xlim(xlim)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Line-to-Continuum Ratio')
    fig.tight_layout()
    fig.savefig(line2cont_plot)
    plt.close(fig)
    #
    # Plot Te vs frequency
    #
    fig, ax = plt.subplots()
    ax.errorbar(data['frequency'][is_Hnalpha],data['elec_temp'][is_Hnalpha],
                yerr=data['e_elec_temp'][is_Hnalpha],fmt='o',color='k')
    # fit line 
    fit,cov = np.polyfit(data['frequency'][is_Hnalpha],data['elec_temp'][is_Hnalpha],1,w=1./data['e_elec_temp'][is_Hnalpha],cov=True)
    ax.plot(data['frequency'][is_Hnalpha],np.poly1d(fit)(data['frequency'][is_Hnalpha]),'k-',zorder=10,
            label=r'$T_e = ({0:.2f}\pm{1:.2f})\nu + ({2:.1f}\pm{3:.1f})$'.format(fit[0],np.sqrt(cov[0,0]),fit[1],np.sqrt(cov[1,1])))
    # fit power law
    fit,cov = np.polyfit(np.log10(data['frequency'][is_Hnalpha]),np.log10(data['elec_temp'][is_Hnalpha]),1,w=data['elec_temp'][is_Hnalpha]*np.log(10.)/data['e_elec_temp'][is_Hnalpha],cov=True)
    yfit = lambda x: 10.**np.poly1d(fit)(np.log10(x))
    ax.plot(data['frequency'][is_Hnalpha],yfit(data['frequency'][is_Hnalpha]),'k--',zorder=10,
            label=r'$T_e \propto \nu^{{({0:.2f}\pm{1:.2f})}}$'.format(fit[0],np.sqrt(cov[0,0])))
    ax.legend(loc='upper left',fontsize=10)
    # plot stacked
    xlim = ax.get_xlim()
    ax.plot(xlim,
            [data['elec_temp'][is_all],data['elec_temp'][is_all]],'k-')
    ax.fill_between(xlim,
                    [data['elec_temp'][is_all]-data['e_elec_temp'][is_all],data['elec_temp'][is_all]-data['e_elec_temp'][is_all]],
                    [data['elec_temp'][is_all]+data['e_elec_temp'][is_all],data['elec_temp'][is_all]+data['e_elec_temp'][is_all]],
                    color='k',alpha=0.5,edgecolor='none')
    ax.set_xlim(xlim)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Electron Temperature (K)')
    fig.tight_layout()
    fig.savefig(te_plot)
    plt.close(fig)
    plt.ion()
