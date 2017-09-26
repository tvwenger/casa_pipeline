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

def main(specfile,sed_plot='cont_sed.png',
         line2cont_plot='line2cont.png'):
    """
    Read output file from calc_te.py and plot continuum flux vs.
    frequency and line-to-continuum ratio vs. frequency
    
    Inputs:
    specfile = file containing calc_te.py output
    sed_plot = where you want to save the continuum SED plot
    line2cont_plot = where you want to save the line-to-continuum
                     ratio vs. frequency plot

    Returns:
      Nothing
    """
    #
    # Read data
    #
    data = np.genfromtxt(specfile,dtype=None,names=True)
    #
    # Remove stacked line
    #
    is_stacked = data['lineid'] == b'stacked'
    data = data[~is_stacked]
    #
    # Plot continuum SED
    #
    fig, ax = plt.subplots()
    ax.errorbar(data['frequency'],data['cont'],yerr=data['rms'],
                fmt='o',color='k')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Flux Density (mJy/beam)')
    fig.tight_layout()
    fig.savefig(sed_plot)
    plt.close(fig)
    #
    # Plot line-to-continuum ratio vs frequency
    #
    fig, ax = plt.subplots()
    ax.errorbar(data['frequency'],data['line2cont'],yerr=data['e_line2cont'],
                fmt='o',color='k')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Line-to-Continuum Ratio')
    fig.tight_layout()
    fig.savefig(line2cont_plot)
    plt.close(fig)

if __name__ == "__main__":
    specfile='{0}.peak.specinfo.txt'.format(sys.argv[1])
    sed_plot='{0}.cont_sed.png'.format(sys.argv[1])
    line2cont_plot='{0}.line2cont.png'.format(sys.argv[1])
    main(specfile,sed_plot,line2cont_plot)
