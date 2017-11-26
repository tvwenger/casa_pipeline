"""
linestack.py
CASA Data Reduction Pipeline - Line stacking script
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

def combeam_line_spws(field,my_line_spws='',
                      linetype='clean'):
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
        imagename='{0}.spw{1}.channel.{2}.pbcor'.format(field,spw,linetype)
        if not os.path.isdir(imagename):
            logger.warn("{0} was not found!".format(imagename))
            continue
        outfile='{0}.spw{1}.channel.{2}.pbcor.combeam'.format(field,spw,linetype)
        if os.path.isdir(outfile):
            logger.info("{0} exists.".format(outfile))
            continue
        casa.imsmooth(imagename=imagename,kernel='commonbeam',
                      outfile=outfile)
    logger.info("Done!")

def smooth_all(field,my_line_spws='',config=None,overwrite=False,
               linetype='clean'):
    """
    Smooth all line and continuum images to worst resolution of
    any individual image, name images with lineID
    instead of spectral window

    Inputs:
      field        = field to analyze
      my_line_spws = comma separated string of line spws
      config       = ConfigParser object for this project
      overwrite    = if True, overwrite steps as necessary
                     if False, skip steps if output already exists

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
    contimage = '{0}.cont.mfs.clean.pbcor'.format(field)
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
        lineimage = '{0}.spw{1}.channel.{2}.pbcor.combeam'.format(field,spw,linetype)
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
    contimage = '{0}.cont.mfs.clean.pbcor'.format(field)
    # Smooth continuum
    if os.path.isdir(contimage):
        outfile='{0}.cont.mfs.imsmooth'.format(field)
        casa.imsmooth(imagename=contimage,kernel='gauss',
                      targetres=True,major=bmaj_target,minor=bmin_target,
                      pa=bpa_target,outfile=outfile,overwrite=overwrite)
        casa.exportfits(imagename=outfile,fitsimage='{0}.fits'.format(outfile),
                        overwrite=True,history=False)
    # Smooth lines, rename with lineid
    for spw in my_line_spws.split(','):
        spw_ind = config.get("Spectral Windows","Line").split(',').index(spw)
        lineid = config.get("Clean","lineids").split(',')[spw_ind]
        lineimage = '{0}.spw{1}.channel.{2}.pbcor.combeam'.format(field,spw,linetype)
        outfile = '{0}.{1}.channel.{2}.imsmooth'.format(field,lineid,linetype)
        casa.imsmooth(imagename=lineimage,kernel='gauss',
                      targetres=True,major=bmaj_target,minor=bmin_target,
                      pa=bpa_target,outfile=outfile,overwrite=overwrite)
        casa.exportfits(imagename=outfile,fitsimage='{0}.fits'.format(outfile),
                        overwrite=True,history=False,velocity=True)
    logger.info("Done!")

def stack_line(field,stackedimage,
               my_line_spws='',config=None,overwrite=False,
               linetype='clean'):
    """
    Stack line images

    Inputs:
      field        = field to analyze
      stackedimage = what to name the stacked image
      my_line_spws = spectral windows to stack
      overwrite    = if True, overwrite steps as necessary
                     if False, skip steps if output already exists

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Get lineids
    #
    lineids = []
    for spw in my_line_spws.split(','):
        spw_ind = config.get("Spectral Windows","Line").split(',').index(spw)
        lineids += [config.get("Clean","lineids").split(',')[spw_ind]]
    logger.info("Stacking lines {0}".format(lineids))
    images = ['{0}.{1}.channel.{2}.imsmooth'.format(field,lineid,linetype) for lineid in lineids]
    ims = ['IM{0}'.format(foo) for foo in range(len(images))]
    myexp =  '({0})/{1}'.format('+'.join(ims),str(float(len(images))))
    casa.immath(imagename=images,outfile=stackedimage,mode='evalexpr',
                expr=myexp)
    casa.exportfits(imagename=stackedimage,fitsimage='{0}.fits'.format(stackedimage),
                    overwrite=True,history=False,velocity=True)
    logger.info("Done!")

def main(field,stackedimage,spws='',config_file='',overwrite=False,
         linetype='clean'):
    """
    Continuum subtract, smooth line images to common beam, 
    smooth all line and continuum images to common beam (rename spw 
    to lineID), stack line images, create line-to-continuum image

    Inputs:
      field       = field to analyze
      stackedimage = what to name the stacked image
      spws        = spectral windows to stack. if '', stack all line spws
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
    my_cont_spws,all_line_spws = setup(config=config)
    if spws == '':
        my_line_spws = all_line_spws
    else:
        my_line_spws = spws
    logger.info("Considering line spws: {0}".format(my_line_spws))
    #
    # smooth line images to common beam
    #
    combeam_line_spws(field,my_line_spws=my_line_spws,
                      linetype=linetype)
    #
    # Smooth all line and continuum images to common beam, rename
    # by lineid
    #
    smooth_all(field,my_line_spws=my_line_spws,config=config,
               overwrite=overwrite,linetype=linetype)
    #
    # Stack line images
    #
    stack_line(field,stackedimage,my_line_spws=my_line_spws,config=config,
               overwrite=overwrite,linetype=linetype)
