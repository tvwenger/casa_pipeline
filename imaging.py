"""
imaging.py
CASA Data Reduction Pipeline - Imaging Script
Trey V. Wenger Jun 2016 - V1.0
"""

import __main__ as casa
import os
import numpy as np
import glob
import logging
import logging.config
import ConfigParser
import shutil

__VERSION__ = "1.0"

# load logging configuration file
logging.config.fileConfig('logging.conf')

def setup(vis='',config=None):
    """
    Perform setup tasks: find line and continuum spectral windows
                         get clean parameters

    Inputs:
      vis     = measurement set
      config  = ConfigParser object for this project

    Returns:
      (my_cont_spws,my_line_spws,clean_params)
      my_cont_spws    = comma-separated string of continuum spws
      my_line_spws    = comma-separated string of line spws
      clean_params    = dictionary of clean parameters read from config file
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
    #
    # Get clean parameters from configuration file
    #
    lineids = config.get("Clean","lineids").split(',')
    restfreqs = config.get("Clean","restfreqs").split(',')
    imsize = [int(foo) for foo in config.get("Clean","imsize").split(',')]
    cell = config.getfloat("Clean","cell")
    cell = "{0}arcsec".format(cell)
    weighting = config.get("Clean","weighting")
    robust = config.getfloat("Clean","robust")
    multiscale = [int(foo) for foo in config.get("Clean","multiscale").split(',') if foo != '']
    gain = config.getfloat("Clean","gain")
    cyclefactor = config.getfloat("Clean","cyclefactor")
    velstart = config.getfloat("Clean","velstart")
    chanwidth = config.getfloat("Clean","chanwidth")
    nchan = config.getint("Clean","nchan")
    chanbuffer = config.getint("Clean","chanbuffer")
    cvelstart = "{0}km/s".format(velstart-(chanbuffer*chanwidth))
    velstart = "{0}km/s".format(velstart)
    chanwidth = "{0}km/s".format(chanwidth)
    cvelnchan = nchan+2*chanbuffer
    outframe = config.get("Clean","outframe")
    veltype = config.get("Clean","veltype")
    clean_params = {"lineids":lineids,"restfreqs":restfreqs,"imsize":imsize,
                    "cell":cell,"weighting":weighting,"robust":robust,
                    "multiscale":multiscale,"gain":gain,"cyclefactor":cyclefactor,
                    "velstart":velstart,"chanwidth":chanwidth,
                    "nchan":nchan,"outframe":outframe,"veltype":veltype,
                    "cvelstart":cvelstart,"cvelnchan":cvelnchan}
    return (my_cont_spws,my_line_spws,clean_params)

def regrid_velocity(vis='',spws='',config=None,clean_params={}):
    """
    Re-grid velocity axis of each spectral window

    Inputs:
      vis     = measurement set
      spws    = comma-separated list of spectral windows to regrid
      config  = ConfigParser object for this project
      clean_params = dictionary of clean parameters

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
    # Re-grid velocity axis of line spectral windows
    #
    for spw in spws.split(','):
        regrid_vis = vis+'.spw{0}.cvel'.format(spw)
        if os.path.isdir(regrid_vis):
            logger.info("Found {0}".format(regrid_vis))
            continue
        logger.info("Regridding velocity axis of spw {0}".format(spw))
        spw_ind = config.get("Spectral Windows","Line").split(',').index(spw)
        restfreq = config.get("Clean","restfreqs").split(',')[spw_ind]
        casa.cvel(vis=vis,outputvis=regrid_vis,spw=spw,restfreq=restfreq,mode='velocity',
                  start=clean_params['cvelstart'],width=clean_params['chanwidth'],
                  nchan=clean_params['cvelnchan'],outframe=clean_params['outframe'],
                  veltype=clean_params['veltype'],interpolation='fftshift')
        logger.info("Done.")

def mfs_clean_cont(field='',vis='',my_cont_spws='',clean_params={}):
    """
    Clean continuum spws using multi-frequency synthesis

    Inputs:
      field        = field to be cleaned
      vis          = measurement set
      my_cont_spws = comma-separated string of continuum spws
      clean_params = dictionary of clean parameters

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Clean continuum
    #
    imagename='{0}.cont.clean'.format(field)
    logger.info("MFS cleaning continuum...")
    casa.clean(vis=vis,imagename=imagename,field=field,spw=my_cont_spws,
               threshold='0mJy',niter=10000,interactive=True,
               imagermode='csclean',mode='mfs',multiscale=clean_params['multiscale'],
               gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'],
               
               usescratch=True)
    logger.info("Done.")
    #
    # Primary beam correction
    #
    logger.info("Performing primary beam correction...")
    casa.impbcor(imagename='{0}.image'.format(imagename),
                 pbimage='{0}.flux'.format(imagename),
                 outfile='{0}.pbcor'.format(imagename))
    logger.info("Done.")

def dirty_image_line(field='',vis='',spws='',my_line_spws='',
                     clean_params={},config=None):
    """
    Dirty image all line spws

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      spws    = comma-separated list of spectral windows to image
      my_line_spws = comma-separated string of line spws
      clean_params = dictionary of clean parameters
      config       = ConfigParser object for this project

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
    for spw in spws.split(','):
        #
        # Get restfreq
        #
        spw_ind = my_line_spws.split(',').index(spw)
        restfreq = config.get("Clean","restfreqs").split(',')[spw_ind]
        #
        # clean spw
        #
        imagename='{0}.spw{1}.dirty'.format(field,spw)
        logger.info("Cleaning spw {0} (restfreq: {1})...".format(spw,restfreq))
        regrid_vis = vis+'.spw{0}.cvel'.format(spw)
        casa.clean(vis=regrid_vis,imagename=imagename,field=field,spw='0',
                   threshold='0mJy',niter=0,interactive=False,
                   imagermode='csclean',mode='velocity',multiscale=clean_params['multiscale'],
                   gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
                   imsize=clean_params['imsize'],cell=clean_params['cell'],
                   weighting=clean_params['weighting'],robust=clean_params['robust'],
                   restfreq=restfreq,start=clean_params['velstart'],width=clean_params['chanwidth'],
                   nchan=clean_params['nchan'],
                   outframe=clean_params['outframe'],veltype=clean_params['veltype'],
                   usescratch=True)
        logger.info("Done.")
        #
        # Primary beam correction
        #
        logger.info("Performing primary beam correction...")
        casa.impbcor(imagename='{0}.image'.format(imagename),
                     pbimage='{0}.flux'.format(imagename),
                     outfile='{0}.pbcor'.format(imagename))
        logger.info("Done.")


def mfs_clean_line(field='',vis='',spws='',clean_params={}):
    """
    Clean line spws using multi-frequency synthesis

    Inputs:
      field        = field to be cleaned
      vis          = measurement set
      spws         = comma-separated string of line spws to image
      clean_params = dictionary of clean parameters

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Clean line
    #
    imagename='{0}.spws_{1}.cont.clean'.format(field,spws)
    logger.info("MFS cleaning line spws...")
    casa.clean(vis=vis,imagename=imagename,field=field,spw=spws,
               threshold='0mJy',niter=10000,interactive=True,
               imagermode='csclean',mode='mfs',multiscale=clean_params['multiscale'],
               gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'],
               usescratch=True)
    logger.info("Done.")
    #
    # Primary beam correction
    #
    logger.info("Performing primary beam correction...")
    casa.impbcor(imagename='{0}.image'.format(imagename),
                 pbimage='{0}.flux'.format(imagename),
                 outfile='{0}.pbcor'.format(imagename))
    logger.info("Done.")

def manual_clean_line(field='',vis='',spw='',my_line_spws='',mask='',
                      clean_params={},config=None):
    """
    Clean a line spw to get clean threshold

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      spw          = spw to clean
      my_line_spws = comma-separated string of all line spws
      mask         = clean mask to use
      clean_params = dictionary of clean parameters
      config       = ConfigParser object for this project

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
    # Get restfreq
    #
    spw_ind = my_line_spws.split(',').index(spw)
    restfreq = config.get("Clean","restfreqs").split(',')[spw_ind]
    #
    # clean spw
    #
    imagename='{0}.spw{1}.clean'.format(field,spw)
    logger.info("Cleaning spw {0} (restfreq: {1})...".format(spw,restfreq))
    regrid_vis = vis+'.spw{0}.cvel'.format(spw)
    casa.clean(vis=regrid_vis,imagename=imagename,field=field,spw='0',
               threshold='0mJy',niter=10000,interactive=True,
               imagermode='csclean',mode='velocity',multiscale=clean_params['multiscale'],
               gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'],
               restfreq=restfreq,start=clean_params['velstart'],width=clean_params['chanwidth'],
               nchan=clean_params['nchan'],
               outframe=clean_params['outframe'],veltype=clean_params['veltype'],
               usescratch=True,mask=mask)
    logger.info("Done.")

def auto_clean_line(field='',vis='',spws='',my_line_spws='',mask='',
                    clean_params={},threshold='',config=None):
    """
    Clean all line spws non-interactively

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      spws         = comma-separated string of spws being cleaned
      my_line_spws = comma-separated string of line spws
      mask         = clean mask to use
      clean_params = dictionary of clean parameters
      threshold    = clean threshold
      config       = ConfigParser object for this project

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
    for spw in spws.split(','):
        #
        # Get restfreq
        #
        spw_ind = my_line_spws.split(',').index(spw)
        restfreq = config.get("Clean","restfreqs").split(',')[spw_ind]
        #
        # clean spw
        #
        imagename='{0}.spw{1}.clean'.format(field,spw)
        logger.info("Cleaning spw {0} (restfreq: {1})...".format(spw,restfreq))
        regrid_vis = vis+'.spw{0}.cvel'.format(spw)
        casa.clean(vis=regrid_vis,imagename=imagename,field=field,spw='0',
                   threshold=threshold,niter=10000,interactive=False,
                   imagermode='csclean',mode='velocity',multiscale=clean_params['multiscale'],
                   gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
                   imsize=clean_params['imsize'],cell=clean_params['cell'],
                   weighting=clean_params['weighting'],robust=clean_params['robust'],
                   restfreq=restfreq,start=clean_params['velstart'],width=clean_params['chanwidth'],
                   nchan=clean_params['nchan'],
                   outframe=clean_params['outframe'],veltype=clean_params['veltype'],
                   usescratch=True,mask=mask)
        logger.info("Done.")
        #
        # Primary beam correction
        #
        logger.info("Performing primary beam correction...")
        casa.impbcor(imagename='{0}.image'.format(imagename),
                     pbimage='{0}.flux'.format(imagename),
                     outfile='{0}.pbcor'.format(imagename))
        logger.info("Done.")

def main(field,vis='',spws='',config_file=''):
    """
    Image, and clean a field

    Inputs:
      field       = field name to clean
      vis         = measurement set containing all data for field
      spws        = comma-separated string of spectral windows to image
                    if empty, image all line spectral windows
      config_file = filename of the configuration file for this project

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
    threshold = None
    my_cont_spws,my_line_spws,clean_params = setup(vis=vis,config=config)
    if spws == '':
        spws = my_line_spws
    #
    # Regrid velocity axis
    #
    regrid_velocity(vis=vis,spws=spws,config=config,clean_params=clean_params)
    #
    # Prompt the user with a menu for each option
    #
    while True:
        print("0. Manually mfs clean continuum image")
        print("1. Dirty clean ALL line spectral windows")
        print("2. Manually mfs clean SELECTED line spectral windows")
        print("3. Manually clean line ANY spectral window to get clean threshold")
        print("4. Set line spectral window clean threshold")
        print("5. Automatically clean SELECTED line spectral windows")
        print("q [quit]")
        answer = raw_input("> ")
        if answer == '0':
            mfs_clean_cont(field=field,vis=vis,my_cont_spws=my_cont_spws,clean_params=clean_params)
        elif answer == '1':
            dirty_image_line(field=field,vis=vis,spws=spws,
                             my_line_spws=my_line_spws,
                             clean_params=clean_params,config=config)
        elif answer == '2':
            mfs_clean_line(field=field,vis=vis,spws=spws,
                           clean_params=clean_params)
        elif answer == '3':
            print("Which spw do you want to clean?")
            spw = raw_input('> ')
            manual_clean_line(field=field,vis=vis,spw=spw,
                              my_line_spws=my_line_spws,
                              mask='{0}.spws_{1}.cont.clean.mask'.format(field,spws),
                              clean_params=clean_params,config=config)
        elif answer == '4':
            print("Please enter the threshold (i.e. 1.1mJy)")
            threshold = raw_input('> ')
        elif answer == '5':
            if threshold is None:
                logger.warn("Must set threshold first!")
            else:
                auto_clean_line(field=field,vis=vis,spws=spws,
                                my_line_spws=my_line_spws,
                                mask='{0}.spws_{1}.cont.clean.mask'.format(field,spws),
                                clean_params=clean_params,
                                threshold=threshold,config=config)
        elif answer.lower() == 'q' or answer.lower() == 'quit':
            break
        else:
            print("Input not recognized.")
