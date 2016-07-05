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

def setup(config=None):
    """
    Perform setup tasks: find line and continuum spectral windows
                         get clean parameters

    Inputs:
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
    cell = config.get("Clean","cell")
    weighting = config.get("Clean","weighting")
    robust = config.getfloat("Clean","robust")
    multiscale = [int(foo) for foo in config.get("Clean","multiscale").split(',') if foo != '']
    gain = config.getfloat("Clean","gain")
    cyclefactor = config.getfloat("Clean","cyclefactor")
    velstart = config.get("Clean","velstart")
    chanwidth = config.get("Clean","chanwidth")
    nchan = config.getint("Clean","nchan")
    outframe = config.get("Clean","outframe")
    veltype = config.get("Clean","veltype")
    clean_params = {"lineids":lineids,"restfreqs":restfreqs,"imsize":imsize,
                    "cell":cell,"weighting":weighting,"robust":robust,
                    "multiscale":multiscale,"gain":gain,"cyclefactor":cyclefactor,
                    "velstart":velstart,"chanwidth":chanwidth,
                    "nchan":nchan,"outframe":outframe,"veltype":veltype}
    return (my_cont_spws,my_line_spws,clean_params)

def mfs_cont(field='',vis='',my_cont_spws='',clean_params={}):
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
    logger.info("Cleaning continuum...")
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

def mfs_line(field='',vis='',spws=[],clean_params={}):
    """
    Clean line spws using multi-frequency synthesis

    Inputs:
      field        = field to be cleaned
      vis          = measurement set
      spws         = list of line spws to clean
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
    imagename='{0}.spws_{1}.cont.clean'.format(field,'_'.join(spws))
    logger.info("Cleaning continuum...")
    casa.clean(vis=vis,imagename=imagename,field=field,spw=','.join(spws),
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
    #
    # Copy clean mask to other spws being cleaned
    #
    logger.info("Copying clean mask to spws {1}".format(spws))
    oldmaskfile = '{0}.mask'.format(imagename,spw)
    for spw in spws:
        newmaskfile = '{0}.spw{1}.clean.mask'.format(field,spw)
        shutil.copytree(oldmaskfile,newmaskfile)
    logger.info("Done!")

def manual_clean_line(field='',vis='',spw='',spws=[],my_line_spws='',
                      clean_params={},config=None):
    """
    Clean a line spw to get clean threshold

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      spw          = spw to clean
      spws         = list of other spws being cleaned
      my_line_spws = comma-separated string of all line spws
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
    casa.clean(vis=vis,imagename=imagename,field=field,spw=spw,
               threshold='0mJy',niter=10000,interactive=True,
               imagermode='csclean',mode='velocity',multiscale=clean_params['multiscale'],
               gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'],
               nchan=clean_params['nchan'],start=clean_params['velstart'],
               width=clean_params['chanwidth'],restfreq=restfreq,
               outframe=clean_params['outframe'],veltype=clean_params['veltype'],
               usescratch=True)
    logger.info("Done.")

def auto_clean_line(field='',vis='',spws=[],my_line_spws='',
                    clean_params={},threshold='',config=None):
    """
    Clean all line spws non-interactively

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      spws         = list of spws to be cleaned
      my_line_spws = comma-separated string of line spws
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
    for spw in spws:
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
        casa.clean(vis=vis,imagename=imagename,field=field,spw=spw,
                   threshold=threshold,niter=10000,interactive=False,
                   imagermode='csclean',mode='velocity',multiscale=clean_params['multiscale'],
                   gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
                   imsize=clean_params['imsize'],cell=clean_params['cell'],
                   weighting=clean_params['weighting'],robust=clean_params['robust'],
                   nchan=clean_params['nchan'],start=clean_params['velstart'],
                   width=clean_params['chanwidth'],restfreq=restfreq,
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

def main(field,vis='',spws=[],config_file=''):
    """
    Combine, image, and clean a field

    Inputs:
      field       = field name to clean
      vis         = measurement set containing all data for field
      spws        = spectral windows to image
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
    my_cont_spws,my_line_spws,clean_params = setup(config=config)
    #
    # Prompt the user with a menu for each option
    #
    while True:
        print("0. Manually mfs clean continuum image")
        print("1. Manually mfs clean line spectral windows, copy clean mask to all spws")
        print("2. Manually clean line spectral window to get clean threshold")
        print("3. Set line spectral window clean threshold")
        print("4. Automatically clean line spectral windows")
        print("q [quit]")
        answer = raw_input("> ")
        if answer == '0':
            mfs_clean_cont(field=field,vis=vis,my_cont_spws=my_cont_spws,clean_params=clean_params)
        elif answer == '1':
            mfs_clean_line(field=field,vis=vis,spws=spws,
                           clean_params=clean_params)
        elif answer == '2':
            print("Which spw do you want to clean?")
            spw = raw_input('> ')
            manual_clean_line(field=field,vis=vis,spw=spw,spws=spws,
                              my_line_spws=my_line_spws,
                              clean_params=clean_params,config=config)
        elif answer == '3':
            print("Please enter the threshold (i.e. 1.1mJy)")
            threshold = raw_input('> ')
        elif answer == '4':
            if threshold is None:
                logger.warn("Must set threshold first!")
            else:
                auto_clean_line(field=field,vis=vis,spws=spws,
                                my_line_spws=my_line_spws,
                                clean_params=clean_params,
                                threshold=threshold,config=config)
        elif answer.lower() == 'q' or answer.lower() == 'quit':
            break
        else:
            print("Input not recognized.")
