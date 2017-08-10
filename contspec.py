"""
contspec.py
CASA Data Reduction Pipeline - Continuum SED plotting script
Trey V. Wenger July 2017 - V1.0
"""

import __main__ as casa # import casa namespace
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import logging.config
import ConfigParser
import shutil

__VERSION__ = "1.0"

# load logging configuration file
logging.config.fileConfig('logging.conf')

def setup(config=None):
    """
    Perform setup tasks: find line and continuum spectral windows

    Inputs:
      config  = ConfigParser object for this project

    Returns:
      (my_cont_spws,my_line_spws,refant)
      my_cont_spws    = comma-separated string of continuum spws
      my_line_spws    = comma-separated string of line spws
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # check config
    #
    if config is None:
        logger.critical("Error: Need to supply a config")
        raise ValueError("Config is None") 
    #
    # Get continuum and line spws from configuration file
    #
    my_cont_spws = config.get("Spectral Windows","Continuum")
    my_line_spws = config.get("Spectral Windows","Line")
    logger.info("Found continuum spws: {0}".format(my_cont_spws))
    logger.info("Found line spws: {0}".format(my_line_spws))
    return (my_cont_spws,my_line_spws)

def smooth_all(field,my_cont_spws='',config=None,overwrite=False,
               linetype='clean'):
    """
    Smooth all continuum images to worst resolution of
    any individual image

    Inputs:
      field        = field to analyze
      my_cont_spws = comma separated string of continuum spws
      config       = ConfigParser object for this project
      overwrite    = if True, overwrite steps as necessary
                     if False, skip steps if output already exists
      linetype     = 'clean' or 'dirty' depending on which you want
                     to plot

    Returns:
      beam_area = beam size area in pixels
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # check config
    #
    if config is None:
        logger.critical("Error: Need to supply a config")
        raise ValueError("Config is None") 
    #
    # Find beam major axes, minor axes, and position angles for all
    # available images
    #
    logger.info("Finding largest synthesized beam")
    bmaj = []
    bmin = []
    bpa = []
    for spw in my_cont_spws.split(','):
        lineimage = '{0}.spw{1}.cont.{2}.pbcor'.format(field,spw,linetype)
        if not os.path.isdir(lineimage):
            logger.warn("{0} not found!".format(lineimage))
            continue
        bmaj.append(casa.imhead(imagename=lineimage,mode='get',
                                hdkey='beammajor')['value'])
        bmin.append(casa.imhead(imagename=lineimage,mode='get',
                                hdkey='beamminor')['value'])
        bpa.append(casa.imhead(imagename=lineimage,mode='get',
                               hdkey='beampa')['value'])
    #
    # Smooth available images to maximum beam size + 2*cell_size
    # and mean position angle
    #
    cell_size = float(config.get("Clean","cell").replace('arcsec',''))
    bmaj_target = np.max(bmaj)+2.*cell_size
    bmin_target = np.max(bmaj)+2.*cell_size
    bpa_target = np.mean(bpa)
    #
    # Compute beam area in pixels
    #
    beam_area = np.pi * bmaj_target * bmin_target / cell_size**2.
    #
    # Smooth
    #
    logger.info("Smoothing all images to")
    logger.info("Major axis: {0} arcsec".format(bmaj_target))
    logger.info("Minor axis: {0} arcsec".format(bmin_target))
    logger.info("Position angle: {0} degs".format(bpa_target))
    bmaj_target = {'unit':'arcsec','value':bmaj_target}
    bmin_target = {'unit':'arcsec','value':bmin_target}
    bpa_target = {'unit':'deg','value':bpa_target}
    for spw in my_cont_spws.split(','):
        imagename = '{0}.spw{1}.cont.{2}.pbcor'.format(field,spw,linetype)
        if os.path.isdir(imagename):
            outfile = '{0}.spw{1}.cont.{2}.imsmooth'.format(field,spw,linetype)
            casa.imsmooth(imagename=imagename,kernel='gauss',
                          targetres=True,major=bmaj_target,minor=bmin_target,
                          pa=bpa_target,outfile=outfile,overwrite=overwrite)
    logger.info("Done!")
    return beam_area

def plot_sed(field,beam_area,my_cont_spws='',config=None,overwrite=False,
             linetype='clean',outplot='cont_sed.pdf'):
    """
    Extract fluxes and plot SED.

    Inputs:
      field        = field to analyze
      beam_area    = beam size area in pixels
      my_cont_spws = comma separated string of continuum spws
      config       = ConfigParser object for this project
      overwrite    = if True, overwrite steps as necessary
                     if False, skip steps if output already exists
      linetype     = 'clean' or 'dirty' depending on which you want
                     to plot
      outplot      = where to save plot

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # check config
    #
    if config is None:
        logger.critical("Error: Need to supply a config")
        raise ValueError("Config is None")
    #
    # Extract fluxes and frequencies
    #
    fluxes = np.array([])
    err_fluxes = np.array([])
    freqs = np.array([])
    for spw in my_cont_spws.split(','):
        imagename = '{0}.spw{1}.cont.{2}.imsmooth'.format(field,spw,linetype)
        region = '{0}.imsmooth.reg'.format(field)
        err_region = '{0}_err.imsmooth.reg'.format(field)
        if os.path.isdir(imagename):
            freq = casa.imhead(imagename=imagename,mode='get',
                               hdkey='crval4')['value']
            flux = casa.imstat(imagename=imagename,region=region)['flux']
            rms = casa.imstat(imagename=imagename,region=err_region)['rms']
            region_size = casa.imstat(imagename=imagename,region=region)['npts']
            err_flux = rms * region_size/beam_area
            freqs = np.append(freqs,freq)
            fluxes = np.append(fluxes,flux)
            err_fluxes = np.append(err_fluxes,err_flux)
    #
    # Generate plot
    #
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #freqs = np.log10(freqs/1.e9)
    #fluxes = np.log10(fluxes*1.e3)
    #err_fluxes = np.log10(err_fluxes*1.e3)
    freqs = freqs/1.e9
    fluxes = fluxes*1.e3
    err_fluxes = err_fluxes*1.e3
    ax.errorbar(freqs,fluxes,yerr=err_fluxes,fmt='o',color='k')
    #
    # Fit and plot line
    #
    fit,cov = np.polyfit(freqs,fluxes,deg=1,w=1./err_fluxes,cov=True)
    xfit = np.linspace(np.min(freqs),np.max(freqs),100)
    yfit = fit[0]*xfit + fit[1]
    ax.plot(xfit,yfit,'r-')
    ax.text(0.1,0.9,r'Flux/mJy = ${0:.2f}\nu + {1:.2f}$'.format(fit[0],fit[1]),transform=ax.transAxes)
    #
    # Fit spectral index
    #
    fit,cov = np.polyfit(np.log10(freqs),np.log10(fluxes),deg=1,w=1./np.log10(err_fluxes),cov=True)
    alpha = fit[0]
    e_alpha = np.sqrt(cov[0][0])
    ax.text(0.1,0.8,r'$\alpha = {0:.2f}\pm{1:.2f}$'.format(alpha,e_alpha),transform=ax.transAxes)
    #
    # Labels and save
    #
    ax.set_xlabel('log(Frequency/GHz)')
    ax.set_ylabel('log(Flux/mJy)')
    plt.title('{0} Continuum SED ({1})'.format(field,linetype))
    plt.savefig(outplot)
    plt.close(fig)

def main(field,config_file='',overwrite=False,linetype='clean',
         outplot='cont_sed.pdf'):
    """
    Smooth all continuum spw images to common beamsize, extract flux
    from region, plot continuum SED with error bars

    Inputs:
      field       = field to analyze
      config_file = filename of the configuration file for this project
      overwrite   = if True, overwrite steps as necessary
                    if False, skip steps if output already exists
      linetype = 'clean' or 'dirty'

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Check inputs
    #
    if not os.path.exists(config_file):
        logger.critical('Configuration file not found')
        raise ValueError('Configuration file not found!')
    #
    # load configuration file
    #
    config = ConfigParser.ConfigParser()
    logger.info("Reading configuration file {0}".format(config_file))
    config.read(config_file)
    logger.info("Done.")
    #
    # initial setup
    #
    my_cont_spws,my_line_spws = setup(config=config)
    #
    # Smooth all continuum images to common beam
    #
    beam_area = smooth_all(field,my_cont_spws=my_cont_spws,config=config,
                           overwrite=overwrite,linetype=linetype)
    #
    # Tell user to make region and error region
    #
    print("Please create a region file called {0}.imsmooth.reg".format(field))
    print("containing the object, as well as a region file called")
    print("{0}_err.imsmooth.reg containing a blank region".format(field))
    _ = raw_input("Press <CR> to continue...")
    #
    # Extract fluxes from regions and make SED plot
    #
    plot_sed(field,beam_area,my_cont_spws,config=config,overwrite=overwrite,
             linetype=linetype,outplot=outplot)
