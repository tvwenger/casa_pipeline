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
    velstart = config.get("Clean","velstart")
    chanwidth = config.get("Clean","chanwidth")
    nchan = config.getint("Clean","nchan")
    outframe = config.get("Clean","outframe")
    veltype = config.get("Clean","veltype")
    clean_params = {"lineids":lineids,"restfreqs":restfreqs,"imsize":imsize,
                    "cell":cell,"weighting":weighting,"robust":robust,
                    "multiscale":multiscale,"velstart":velstart,"chanwidth":chanwidth,
                    "nchan":nchan,"outframe":outframe,"veltype":veltype}
    return (my_cont_spws,my_line_spws,clean_params)

def clean_cont(field='',vises=[],my_cont_spws='',clean_params={}):
    """
    Clean continuum spws

    Inputs:
      field        = field to be cleaned
      vises        = list of measurement sets to combine
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
    casa.clean(vis=vises,imagename=imagename,field=field,spw=my_cont_spws,
               threshold='0mJy',niter=10000,interactive=True,
               imagermode='csclean',mode='mfs',multiscale=clean_params['multiscale'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'])
    logger.info("Done.")
    #
    # Primary beam correction
    #
    logger.info("Performing primary beam correction...")
    casa.impbcor(imagename='{0}.image'.format(imagename),
                 pbimage='{0}.flux'.format(imagename),
                 outfile='{0}.pbcor'.format(imagename))
    logger.info("Done.")

def dirty_line(field='',vises=[],my_line_spws='',clean_params={}):
    """
    Dirty image line spws

    Inputs:
      field        = field to be imaged
      vises        = list of measurement sets to combine
      my_line_spws = comma-separated string of line spws
      clean_params = dictionary of clean parameters

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Dirty image each spw
    #
    for spw in my_line_spws.split(','):
        imagename='{0}.spw{1}.dirty'.format(field,spw)
        logger.info("Dirty imaging spw {0}...".format(spw))
        casa.clean(vis=vises,imagename=imagename,field=field,spw=spw,
                   threshold='0mJy',niter=0,interactive=False,
                   imagermode='csclean',mode='mfs',multiscale=clean_params['multiscale'],
                   imsize=clean_params['imsize'],cell=clean_params['cell'],
                   weighting=clean_params['weighting'],robust=clean_params['robust'])
        logger.info("Done.")

def clean_first_line(field='',vises=[],my_line_spws='',clean_params={}):
    """
    Clean first line spw to get clean threshold

    Inputs:
      field        = field to be imaged
      vises        = list of measurement sets to combine
      my_line_spws = comma-separated string of line spws
      clean_params = dictionary of clean parameters

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Check that region file exists
    #
    spw = my_line_spws.split(',')[0]
    restfreq = clean_params['restfreqs'][0]
    mask='{0}.spw{1}.reg'.format(field,spw)
    if not os.path.exists(mask):
        logger.critical("Region file does not exist: {0}".format(mask))
        raise ValueError("Region file does not exist: {0}".format(mask))
    #
    # clean spw
    #
    imagename='{0}.spw{1}.clean'.format(field,spw)
    logger.info("Cleaning spw {0}...".format(spw))
    casa.clean(vis=vises,imagename=imagename,field=field,spw=spw,
               threshold='0mJy',niter=10000,interactive=True,mask=mask,
               imagermode='csclean',mode='velocity',multiscale=clean_params['multiscale'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'],
               nchan=clean_params['nchan'],start=clean_params['velstart'],
               width=clean_params['chanwidth'],restfreq=restfreq,
               outframe=clean_params['outframe'],veltype=clean_params['veltype'])
    logger.info("Done.")

def clean_line(field='',vises=[],my_line_spws='',clean_params={},threshold=''):
    """
    Clean all line spws non-interactively

    Inputs:
      field        = field to be imaged
      vises        = list of measurement sets to combine
      my_line_spws = comma-separated string of line spws
      clean_params = dictionary of clean parameters
      threshold    = clean threshold

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Check that region file exists
    #
    for spw,restfreq in zip(my_line_spws.split(','),clean_params['restfreqs']):
        mask='{0}.spw{1}.reg'.format(field,spw)
        if not os.path.exists(mask):
            logger.critical("Region file does not exist: {0}".format(mask))
            raise ValueError("Region file does not exist: {0}".format(mask))
        #
        # clean spw
        #
        imagename='{0}.spw{1}.clean'.format(field,spw)
        logger.info("Cleaning spw {0}...".format(spw))
        casa.clean(vis=vises,imagename=imagename,field=field,spw=spw,
                   threshold=threshold,niter=10000,interactive=False,mask=mask,
                   imagermode='csclean',mode='velocity',multiscale=clean_params['multiscale'],
                   imsize=clean_params['imsize'],cell=clean_params['cell'],
                   weighting=clean_params['weighting'],robust=clean_params['robust'],
                   nchan=clean_params['nchan'],start=clean_params['velstart'],
                   width=clean_params['chanwidth'],restfreq=restfreq,
                   outframe=clean_params['outframe'],veltype=clean_params['veltype'])
        logger.info("Done.")
        #
        # Primary beam correction
        #
        logger.info("Performing primary beam correction...")
        casa.impbcor(imagename='{0}.image'.format(imagename),
                     pbimage='{0}.flux'.format(imagename),
                     outfile='{0}.pbcor'.format(imagename))
        logger.info("Done.")

def main(field,vises=[],config_file=''):
    """
    Combine, image, and clean a field

    Inputs:
      field       = field name to clean
      vises       = list of measurement sets from which to collect data for this
                    source
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
    if len(vises) == 0:
        logger.critical('Must supply vises containing calibrated measurement sets!')
        raise ValueError('Must supply vises containing measurement sets')
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
        print("0. Manually clean continuum image")
        print("1. Dirty image line spectral windows")
        print("2. Manually clean first line spectral window")
        print("3. Set line spectral window clean threshold")
        print("4. Automatically clean line spectral windows")
        print("q [quit]")
        answer = raw_input("> ")
        if answer == '0':
            clean_cont(field=field,vises=vises,my_cont_spws=my_cont_spws,clean_params=clean_params)
        elif answer == '1':
            dirty_line(field=field,vises=vises,my_line_spws=my_line_spws,clean_params=clean_params)
            print("Please open each dirty image in the CASA viewer and define a clean region.")
            print("Save the CASA region in a file with name {field}.spw{spw_number}.reg")
        elif answer == '2':
            clean_first_line(field=field,vises=vises,my_line_spws=my_line_spws,clean_params=clean_params)
        elif answer == '3':
            print("Please enter the threshold (i.e. 1.1mJy)")
            threshold = raw_input('> ')
        elif answer == '4':
            if threshold is None:
                logger.warn("Must set threshold first!")
            else:
                clean_line(field=field,vises=vises,my_line_spws=my_line_spws,clean_params=clean_params,
                           threshold=threshold)
        elif answer.lower() == 'q' or answer.lower() == 'quit':
            break
        else:
            print("Input not recognized.")
