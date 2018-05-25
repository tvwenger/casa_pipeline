"""
smooth.py
CASA Data Reduction Pipeline - Smooth images to common beam size
Trey V. Wenger Jun 2016 - V1.0
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

def combeam_line_spws(field,my_line_spws='',linetype='clean',
                      overwrite=False):
    """
    Smooth each line spw to common beam within that spw (i.e. if
    beam is varying over frequency within that spw)

    Inputs:
      field        = field to analyze
      my_line_spws = comma separated string of line spws

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    logger.info("Smoothing each line spw to common beam")
    for spw in my_line_spws.split(','):
        logger.info("Working on spw {0}".format(spw))
        imagename='{0}.spw{1}.channel.{2}.pbcor.image'.format(field,spw,linetype)
        if not os.path.isdir(imagename):
            logger.warn("{0} was not found!".format(imagename))
            continue
        outfile='{0}.spw{1}.channel.{2}.combeam.pbcor'.format(field,spw,linetype)
        head = casa.imhead(imagename=imagename)
        if 'perplanebeams' in head.keys():
            casa.imsmooth(imagename=imagename,kernel='commonbeam',
                          outfile=outfile,overwrite=overwrite)
        else:
            logger.info("Only one beam in image.")
            shutil.copytree(imagename,outfile)
        logger.info("Done!")
    logger.info("Done!")

def smooth_all(field,my_line_spws='',config=None,overwrite=False,
               linetype='clean',conttype='clean'):
    """
    Smooth all line and continuum images to worst resolution of
    any individual image

    Inputs:
      field        = field to analyze
      my_line_spws = comma separated string of line spws
      config       = ConfigParser object for this project
      overwrite    = if True, overwrite steps as necessary
                     if False, skip steps if output already exists
      linetype = 'clean' or 'dirty', etc.
      conttype = 'clean' or 'dirty', etc.

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
    # Find beam major axes, minor axes, and position angles for all
    # available images
    #
    logger.info("Finding largest synthesized beam")
    bmajs = []
    bmins = []
    bpas = []
    contimage = '{0}.cont.mfs.{1}.image.tt0'.format(field,conttype)
    contpbimage = '{0}.cont.mfs.{1}.pbcor.image.tt0'.format(field,conttype)
    contresimage = '{0}.cont.mfs.{1}.residual.tt0'.format(field,conttype)
    if not os.path.isdir(contimage):
        logger.warn("{0} not found!".format(contimage))
    else:
        bmajs.append(casa.imhead(imagename=contimage,mode='get',
                                 hdkey='beammajor')['value'])
        bmins.append(casa.imhead(imagename=contimage,mode='get',
                                 hdkey='beamminor')['value'])
        bpas.append(casa.imhead(imagename=contimage,mode='get',
                                hdkey='beampa')['value'])
    for spw in my_line_spws.split(','):
        lineimage = '{0}.spw{1}.channel.{2}.combeam.pbcor'.format(field,spw,linetype)
        if not os.path.isdir(lineimage):
            logger.warn("{0} not found!".format(lineimage))
            continue
        bmajs.append(casa.imhead(imagename=lineimage,mode='get',
                                hdkey='beammajor')['value'])
        bmins.append(casa.imhead(imagename=lineimage,mode='get',
                                hdkey='beamminor')['value'])
        bpas.append(casa.imhead(imagename=lineimage,mode='get',
                                hdkey='beampa')['value'])
    #
    # Smooth available images to maximum (circular) beam size
    # + 0.1 pixel size (otherwise imsmooth will complain)
    #
    cell_size = float(config.get("Clean","cell").replace('arcsec',''))
    bmaj_target = np.max(bmajs)+0.1*cell_size
    bmin_target = np.max(bmajs)+0.1*cell_size
    bpa_target = 0.
    logger.info("Smoothing all images to")
    logger.info("Major axis: {0} arcsec".format(bmaj_target))
    logger.info("Minor axis: {0} arcsec".format(bmin_target))
    logger.info("Position angle: {0} degs".format(bpa_target))
    bmaj_target = {'unit':'arcsec','value':bmaj_target}
    bmin_target = {'unit':'arcsec','value':bmin_target}
    bpa_target = {'unit':'deg','value':bpa_target}
    # Smooth continuum
    if os.path.isdir(contimage):
        logger.info("Working on continuum image")
        outfile='{0}.cont.mfs.{1}.imsmooth'.format(field,conttype)
        casa.imsmooth(imagename=contimage,kernel='gauss',
                      targetres=True,major=bmaj_target,minor=bmin_target,
                      pa=bpa_target,outfile=outfile,overwrite=overwrite)
        casa.exportfits(imagename=outfile,fitsimage='{0}.fits'.format(outfile),
                        overwrite=True,history=False)
        logger.info("Done!")
    if os.path.isdir(contpbimage):
        logger.info("Working on continuum PBCorr image")
        outfile='{0}.cont.mfs.{1}.imsmooth.pbcor'.format(field,conttype)
        casa.imsmooth(imagename=contpbimage,kernel='gauss',
                      targetres=True,major=bmaj_target,minor=bmin_target,
                      pa=bpa_target,outfile=outfile,overwrite=overwrite)
        casa.exportfits(imagename=outfile,fitsimage='{0}.fits'.format(outfile),
                        overwrite=True,history=False)
        logger.info("Done!")
    if os.path.isdir(contresimage):
        logger.info("Working on continuum residual image")
        outfile='{0}.cont.mfs.{1}.imsmooth.residual'.format(field,conttype)
        casa.imsmooth(imagename=contresimage,kernel='gauss',
                      targetres=True,major=bmaj_target,minor=bmin_target,
                      pa=bpa_target,outfile=outfile,overwrite=overwrite)
        casa.exportfits(imagename=outfile,fitsimage='{0}.fits'.format(outfile),
                        overwrite=True,history=False)
        logger.info("Done!")
    # Smooth lines
    for spw in my_line_spws.split(','):
        lineimage = '{0}.spw{1}.channel.{2}.combeam.pbcor'.format(field,spw,linetype)
        if os.path.isdir(lineimage):
            logger.info("Working on spw {0}".format(spw))
            outfile = '{0}.spw{1}.channel.{2}.imsmooth.pbcor'.format(field,spw,linetype)
            casa.imsmooth(imagename=lineimage,kernel='gauss',
                        targetres=True,major=bmaj_target,minor=bmin_target,
                        pa=bpa_target,outfile=outfile,overwrite=overwrite)
            casa.exportfits(imagename=outfile,fitsimage='{0}.fits'.format(outfile),
                            overwrite=True,history=False,velocity=True)
            logger.info("Done!")
    logger.info("Done!")

def main(field,config_file='',overwrite=False,
         linetype='clean',conttype='clean'):
    """
    Smooth all line spw images to common beam

    Inputs:
      field       = field to analyze
      config_file = filename of the configuration file for this project
      overwrite   = if True, overwrite steps as necessary
                    if False, skip steps if output already exists
      linetype = 'clean' or 'dirty', etc.
      conttype = 'clean' or 'dirty', etc.

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
    # smooth line images to common beam
    #
    combeam_line_spws(field,my_line_spws=my_line_spws,linetype=linetype,
                      overwrite=overwrite)
    #
    # Smooth all line and continuum images to common beam
    #
    smooth_all(field,my_line_spws=my_line_spws,config=config,
               overwrite=overwrite,linetype=linetype,conttype=conttype)
