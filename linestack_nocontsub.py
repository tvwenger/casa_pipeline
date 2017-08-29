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

def combeam_line_spws(field,my_line_spws='',overwrite=False,
                      linetype='clean'):
    """
    Smooth each line spw to common beam within that spw (i.e. if
    beam is varying over frequency within that spw)

    Inputs:
      field        = field to analyze
      my_line_spws = comma separated string of line spws
      overwrite    = if True, overwrite steps as necessary
                     if False, skip steps if output already exists

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
            if overwrite:
                logger.info("Overwriting {0}".format(outfile))
                shutil.rmtree(outfile)
            else:
                logger.info("{0} exists.".format(outfile))
                continue
        casa.imsmooth(imagename=imagename,kernel='commonbeam',
                      outfile=outfile,overwrite=overwrite)
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
    bmaj = []
    bmin = []
    bpa = []
    contimage = '{0}.cont.mfs.clean.pbcor'.format(field)
    if not os.path.isdir(contimage):
        logger.warn("{0} not found!".format(contimage))
    else:
        bmaj.append(casa.imhead(imagename=contimage,mode='get',
                                hdkey='beammajor')['value'])
        bmin.append(casa.imhead(imagename=contimage,mode='get',
                                hdkey='beamminor')['value'])
        bpa.append(casa.imhead(imagename=contimage,mode='get',
                               hdkey='beampa')['value'])
    for spw in my_line_spws.split(','):
        lineimage = '{0}.spw{1}.channel.{2}.pbcor.combeam'.format(field,spw,linetype)
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
    logger.info("Smoothing all images to")
    logger.info("Major axis: {0} arcsec".format(bmaj_target))
    logger.info("Minor axis: {0} arcsec".format(bmin_target))
    logger.info("Position angle: {0} degs".format(bpa_target))
    bmaj_target = {'unit':'arcsec','value':bmaj_target}
    bmin_target = {'unit':'arcsec','value':bmin_target}
    bpa_target = {'unit':'deg','value':bpa_target}
    contimage = '{0}.cont.mfs.clean.pbcor'.format(field)
    if os.path.isdir(contimage):
        outfile='{0}.cont.mfs.imsmooth'.format(field)
        casa.imsmooth(imagename=contimage,kernel='gauss',
                      targetres=True,major=bmaj_target,minor=bmin_target,
                      pa=bpa_target,outfile=outfile,overwrite=overwrite)
    for spw,lineid in zip(my_line_spws.split(','),config.get('Clean','lineids').split(',')):
        lineimage = '{0}.spw{1}.channel.{2}.pbcor.combeam'.format(field,spw,linetype)
        if os.path.isdir(lineimage):
            outfile = '{0}.{1}.channel.{2}.imsmooth'.format(field,lineid,linetype)
            casa.imsmooth(imagename=lineimage,kernel='gauss',
                          targetres=True,major=bmaj_target,minor=bmin_target,
                          pa=bpa_target,outfile=outfile,overwrite=overwrite)
    logger.info("Done!")

def stack_line(field,lineids=[],config=None,overwrite=False,
               linetype='clean'):
    """
    Stack line images

    Inputs:
      field        = field to analyze
      lineids      = lines to stack, if empty use all lines
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
    # Check if we supplied lineids
    #
    if len(lineids) == 0:
        lineids = config.get("Clean","lineids").split(',')
    logger.info("Stacking lines {0}".format(lineids))
    images = ['{0}.{1}.channel.{2}.imsmooth'.format(field,lineid,linetype) for lineid in lineids]
    ims = ['IM{0}'.format(foo) for foo in range(len(images))]
    myexp =  '({0})/{1}'.format('+'.join(ims),str(float(len(images))))
    outfile='{0}.Halpha_{1}lines.channel.{2}.image'.format(field,str(len(images)),linetype)
    casa.immath(imagename=images,outfile=outfile,mode='evalexpr',
                expr=myexp)
    logger.info("Done!")
    return outfile

def moment0_image(stackedimage='',overwrite=False):
    """
    Stack line images

    Inputs:
      stackedimage = filename of line-stacked image
      overwrite    = if True, overwrite steps as necessary
                     if False, skip steps if output already exists

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    outfile = stackedimage+'.mom0'
    if os.path.isdir(outfile):
        if overwrite:
            logger.info("Overwriting {0}".format(outfile))
            shutil.rmtree(outfile)
        else:
            logger.info("Found {0}".format(outfile))
            return outfile
    logger.info("Creating moment 0 map")
    casa.immoments(stackedimage,moments=0,outfile=outfile)
    logger.info("Done!")
    return outfile

def linetocont_image(field,moment0image='',overwrite=False,
                     linetype='clean'):
    """
    Stack line images

    Inputs:
      field        = field to analyze
      moment0image = filename of line-stacked moment 0 image
      overwrite    = if True, overwrite steps as necessary
                     if False, skip steps if output already exists

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    logger.info("Creating line-to-continuum image")
    if not os.path.isdir(moment0image):
        logger.warn("{0} does not exist".format(moment0image))
        return
    contimage='{0}.cont.mfs.imsmooth'.format(field)
    if not os.path.isdir(contimage):
        logger.warn("{0} does not exist".format(contimage))
        return
    outfile='{0}.linetocont.{1}.image'.format(field,linetype)
    images = [moment0image,contimage]
    myexp = 'IM0/IM1'
    casa.immath(imagename=images,outfile=outfile,mode='evalexpr',
                expr=myexp)
    logger.info("Done!")

def main(field,lineids=[],config_file='',overwrite=False,
         linetype='clean'):
    """
    Continuum subtract, smooth line images to common beam, 
    smooth all line and continuum images to common beam (rename spw 
    to lineID), stack line images, create line-to-continuum image

    Inputs:
      field       = field to analyze
      lineids     = lines to stack, if empty all lines
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
    # smooth line images to common beam
    #
    combeam_line_spws(field,my_line_spws=my_line_spws,overwrite=overwrite,
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
    stackedimage = stack_line(field,lineids=lineids,config=config,
                              overwrite=overwrite,linetype=linetype)
    #
    # make moment 0 image
    #
    moment0image = moment0_image(stackedimage,overwrite=overwrite)
    #
    # create line-to-continuum image using stacked line image
    #
    linetocont_image(field,moment0image=moment0image,overwrite=overwrite,
                     linetype=linetype)
