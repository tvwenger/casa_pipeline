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
    nterms = config.getint("Clean","nterms")
    outertaper = ["{0}arcsec".format(config.getfloat("Clean","outertaper"))]
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
                    "cvelstart":cvelstart,"cvelnchan":cvelnchan,
                    "nterms":nterms,"outertaper":outertaper}
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
        casa.cvel2(vis=vis,outputvis=regrid_vis,spw=spw,restfreq=restfreq,mode='velocity',
                   start=clean_params['cvelstart'],width=clean_params['chanwidth'],
                   nchan=clean_params['cvelnchan'],outframe=clean_params['outframe'],
                   veltype=clean_params['veltype'],interpolation='linear')
        #casa.mstransform(vis=vis,outputvis=regrid_vis,datacolumn='data',
        #                 spw=spw,regridms=True,
        #                 restfreq=restfreq,mode='velocity',
        #                 start=clean_params['cvelstart'],width=clean_params['chanwidth'],
        #                 nchan=clean_params['cvelnchan'],outframe=clean_params['outframe'],
        #                 veltype=clean_params['veltype'],interpolation='fftshift')
        logger.info("Done.")

def mfs_dirty_cont(field='',vis='',my_cont_spws='',clean_params={}):
    """
    Dirty image continuum spws using multi-frequency synthesis

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
    imagename='{0}.cont.mfs.dirty'.format(field)
    logger.info("Generating dirty continuum image (MFS)...")
    casa.clean(vis=vis,imagename=imagename,field=field,spw=my_cont_spws,
               threshold='0mJy',niter=0,interactive=False,nterms=clean_params['nterms'],
               imagermode='csclean',mode='mfs',multiscale=clean_params['multiscale'],
               gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'],
               uvtaper=True,outertaper=clean_params['outertaper'])
    logger.info("Done.")
    #
    # Primary beam correction
    #
    logger.info("Performing primary beam correction...")
    casa.impbcor(imagename='{0}.image.tt0'.format(imagename),
                 pbimage='{0}.flux'.format(imagename),
                 outfile='{0}.pbcor'.format(imagename))
    logger.info("Done.")

def channel_dirty_cont(field='',vis='',my_cont_spws='',clean_params={}):
    """
    Dirty image continuum spws channel-by-channel

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
    imagename='{0}.cont.chan.dirty'.format(field)
    logger.info("Generating dirty continuum cube (channel)...")
    casa.clean(vis=vis,imagename=imagename,field=field,spw=my_cont_spws,
               threshold='0mJy',niter=0,interactive=False,
               imagermode='csclean',mode='channel',multiscale=clean_params['multiscale'],
               gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'],
               uvtaper=True,outertaper=clean_params['outertaper'],
               outframe=clean_params['outframe'])
    logger.info("Done.")
    #
    # Primary beam correction
    #
    logger.info("Performing primary beam correction...")
    casa.impbcor(imagename='{0}.image'.format(imagename),
                 pbimage='{0}.flux'.format(imagename),
                 outfile='{0}.pbcor'.format(imagename))
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
    imagename='{0}.cont.mfs.clean'.format(field)
    logger.info("Cleaning continuum image (MFS)...")
    casa.clean(vis=vis,imagename=imagename,field=field,spw=my_cont_spws,
               threshold='0mJy',niter=10000,interactive=True,nterms=clean_params['nterms'],
               imagermode='csclean',mode='mfs',multiscale=clean_params['multiscale'],
               gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'],
               uvtaper=True,outertaper=clean_params['outertaper'])
    logger.info("Done.")
    #
    # Primary beam correction
    #
    logger.info("Performing primary beam correction...")
    casa.impbcor(imagename='{0}.image.tt0'.format(imagename),
                 pbimage='{0}.flux'.format(imagename),
                 outfile='{0}.pbcor'.format(imagename))
    logger.info("Done.")

def channel_clean_cont(field='',vis='',my_cont_spws='',clean_params={}):
    """
    Clean continuum channels

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
    imagename='{0}.cont.chan.clean'.format(field)
    logger.info("Cleaning continuum cube (channel)...")
    casa.clean(vis=vis,imagename=imagename,field=field,spw=my_cont_spws,
               threshold='0mJy',niter=10000,interactive=True,
               imagermode='csclean',mode='channel',multiscale=clean_params['multiscale'],
               gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'],
               uvtaper=True,outertaper=clean_params['outertaper'],
               outframe=clean_params['outframe'])
    logger.info("Done.")
    #
    # Primary beam correction
    #
    logger.info("Performing primary beam correction...")
    casa.impbcor(imagename='{0}.image'.format(imagename),
                 pbimage='{0}.flux'.format(imagename),
                 outfile='{0}.pbcor'.format(imagename))
    logger.info("Done.")

def mfs_dirty_cont_spws(field='',vis='',my_cont_spws='',
                        clean_params={}):
    """
    Dirty image each continuum spw (MFS)

    Inputs:
      field        = field to be imaged
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
    for spw in my_cont_spws.split(','):
        #
        # Clean
        #
        imagename='{0}.spw{1}.mfs.dirty'.format(field,spw)
        logger.info("Generating dirty image of spw {0} (MFS)...".format(spw))
        casa.clean(vis=vis,imagename=imagename,field=field,spw=spw,
                   threshold='0mJy',niter=0,interactive=False,nterms=clean_params['nterms'],
                   imagermode='csclean',mode='mfs',multiscale=clean_params['multiscale'],
                   gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
                   imsize=clean_params['imsize'],cell=clean_params['cell'],
                   weighting=clean_params['weighting'],robust=clean_params['robust'],
                   uvtaper=True,outertaper=clean_params['outertaper'])
        logger.info("Done.")
        #
        # Primary beam correction
        #
        logger.info("Performing primary beam correction...")
        casa.impbcor(imagename='{0}.image.tt0'.format(imagename),
                     pbimage='{0}.flux'.format(imagename),
                     outfile='{0}.pbcor'.format(imagename))
        logger.info("Done.")

def mfs_clean_cont_spws(field='',vis='',my_cont_spws='',
                        clean_params={}):
    """
    Clean each continuum spw (MFS)

    Inputs:
      field        = field to be imaged
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
    for spw in my_cont_spws.split(','):
        #
        # Clean
        #
        imagename='{0}.spw{1}.mfs.clean'.format(field,spw)
        logger.info("Cleaning continuum spw {0} (MFS)...".format(spw))
        casa.clean(vis=vis,imagename=imagename,field=field,spw=spw,
                   threshold='0mJy',niter=100000,interactive=True,nterms=clean_params['nterms'],
                   imagermode='csclean',mode='mfs',multiscale=clean_params['multiscale'],
                   gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
                   imsize=clean_params['imsize'],cell=clean_params['cell'],
                   weighting=clean_params['weighting'],robust=clean_params['robust'],
                   uvtaper=True,outertaper=clean_params['outertaper'])
        logger.info("Done.")
        #
        # Primary beam correction
        #
        logger.info("Performing primary beam correction...")
        casa.impbcor(imagename='{0}.image.tt0'.format(imagename),
                     pbimage='{0}.flux'.format(imagename),
                     outfile='{0}.pbcor'.format(imagename))
        logger.info("Done.")

def mfs_clean_cont_spw(field='',vis='',spw='',mask='',
                        clean_params={}):
    """
    Clean single continuum spw (MFS)

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      spw          = spectral window to clean
      mask         = clean mask
      clean_params = dictionary of clean parameters

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Clean
    #
    imagename='{0}.spw{1}.mfs.clean'.format(field,spw)
    logger.info("Cleaning continuum spw {0} (MFS)...".format(spw))
    casa.clean(vis=vis,imagename=imagename,field=field,spw=spw,
               threshold='0mJy',niter=100000,interactive=True,nterms=clean_params['nterms'],
               imagermode='csclean',mode='mfs',multiscale=clean_params['multiscale'],
               gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'],
               uvtaper=True,outertaper=clean_params['outertaper'],
               mask=mask)
    logger.info("Done.")
    #
    # Primary beam correction
    #
    logger.info("Performing primary beam correction...")
    casa.impbcor(imagename='{0}.image.tt0'.format(imagename),
                 pbimage='{0}.flux'.format(imagename),
                 outfile='{0}.pbcor'.format(imagename))
    logger.info("Done.")

def auto_mfs_clean_cont_spws(field='',vis='',my_cont_spws='',
                             threshold='',mask='',
                             clean_params={}):
    """
    Clean each continuum spw (MFS)

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      my_cont_spws = comma-separated string of continuum spws
      threshold    = clean threshold
      mask         = clean mask
      clean_params = dictionary of clean parameters

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    for spw in my_cont_spws.split(','):
        #
        # Clean
        #
        imagename='{0}.spw{1}.mfs.clean'.format(field,spw)
        logger.info("Automatically cleaning continuum spw {0} (MFS)...".format(spw))
        casa.clean(vis=vis,imagename=imagename,field=field,spw=spw,
                   threshold=threshold,niter=100000,interactive=False,nterms=clean_params['nterms'],
                   imagermode='csclean',mode='mfs',multiscale=clean_params['multiscale'],
                   gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
                   imsize=clean_params['imsize'],cell=clean_params['cell'],
                   weighting=clean_params['weighting'],robust=clean_params['robust'],
                   uvtaper=True,outertaper=clean_params['outertaper'],
                   mask=mask)
        logger.info("Done.")
        #
        # Primary beam correction
        #
        logger.info("Performing primary beam correction...")
        casa.impbcor(imagename='{0}.image.tt0'.format(imagename),
                     pbimage='{0}.flux'.format(imagename),
                     outfile='{0}.pbcor'.format(imagename))
        logger.info("Done.")

def channel_dirty_line_spws(field='',vis='',my_line_spws='',
                            clean_params={},config=None):
    """
    Dirty image all line spws

    Inputs:
      field        = field to be imaged
      vis          = measurement set
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
    for spw in my_line_spws.split(','):
        #
        # Get restfreq
        #
        spw_ind = my_line_spws.split(',').index(spw)
        restfreq = config.get("Clean","restfreqs").split(',')[spw_ind]
        #
        # clean spw
        #
        imagename='{0}.spw{1}.channel.dirty'.format(field,spw)
        logger.info("Dirty imaging spw {0} (restfreq: {1})...".format(spw,restfreq))
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
                   uvtaper=True,outertaper=clean_params['outertaper'])
        logger.info("Done.")
        #
        # Primary beam correction
        #
        logger.info("Performing primary beam correction...")
        casa.impbcor(imagename='{0}.image'.format(imagename),
                     pbimage='{0}.flux'.format(imagename),
                     outfile='{0}.pbcor'.format(imagename))
        logger.info("Done.")

def channel_clean_line_spws(field='',vis='',my_line_spws='',
                            clean_params={},config=None):
    """
    Clean all line spws manually

    Inputs:
      field        = field to be imaged
      vis          = measurement set
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
    for spw in my_line_spws.split(','):
        #
        # Get restfreq
        #
        spw_ind = my_line_spws.split(',').index(spw)
        restfreq = config.get("Clean","restfreqs").split(',')[spw_ind]
        #
        # clean spw
        #
        imagename='{0}.spw{1}.channel.clean'.format(field,spw)
        logger.info("Cleaning spw {0} (restfreq: {1})...".format(spw,restfreq))
        regrid_vis = vis+'.spw{0}.cvel'.format(spw)
        #
        # Need to generate dirty image first because fucking velocity
        # channels are messed up the first time.
        #
        casa.clean(vis=regrid_vis,imagename=imagename,field=field,spw='0',
                   threshold='0mJy',niter=0,interactive=False,
                   imagermode='csclean',mode='velocity',multiscale=clean_params['multiscale'],
                   gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
                   imsize=clean_params['imsize'],cell=clean_params['cell'],
                   weighting=clean_params['weighting'],robust=clean_params['robust'],
                   restfreq=restfreq,start=clean_params['velstart'],width=clean_params['chanwidth'],
                   nchan=clean_params['nchan'],
                   outframe=clean_params['outframe'],veltype=clean_params['veltype'],
                   uvtaper=True,outertaper=clean_params['outertaper'])
        #
        # Now actually clean it
        #
        casa.clean(vis=regrid_vis,imagename=imagename,field=field,spw='0',
                   threshold='0mJy',niter=10000,interactive=True,
                   imagermode='csclean',mode='velocity',multiscale=clean_params['multiscale'],
                   gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
                   imsize=clean_params['imsize'],cell=clean_params['cell'],
                   weighting=clean_params['weighting'],robust=clean_params['robust'],
                   restfreq=restfreq,start=clean_params['velstart'],width=clean_params['chanwidth'],
                   nchan=clean_params['nchan'],
                   outframe=clean_params['outframe'],veltype=clean_params['veltype'],
                   uvtaper=True,outertaper=clean_params['outertaper'])
        logger.info("Done.")
        #
        # Primary beam correction
        #
        logger.info("Performing primary beam correction...")
        casa.impbcor(imagename='{0}.image'.format(imagename),
                     pbimage='{0}.flux'.format(imagename),
                     outfile='{0}.pbcor'.format(imagename))
        logger.info("Done.")

def channel_clean_line_spw(field='',vis='',spw='',my_line_spws='',
                           mask='',clean_params={},config=None):
    """
    Clean all line spws manually

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
    imagename='{0}.spw{1}.channel.clean'.format(field,spw)
    logger.info("Cleaning spw {0} (restfreq: {1})...".format(spw,restfreq))
    regrid_vis = vis+'.spw{0}.cvel'.format(spw)
    #
    # Need to generate dirty image first because fucking velocity
    # channels are messed up the first time.
    #
    casa.clean(vis=regrid_vis,imagename=imagename,field=field,spw='0',
               threshold='0mJy',niter=0,interactive=False,
               imagermode='csclean',mode='velocity',multiscale=clean_params['multiscale'],
               gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'],
               restfreq=restfreq,start=clean_params['velstart'],width=clean_params['chanwidth'],
               nchan=clean_params['nchan'],
               outframe=clean_params['outframe'],veltype=clean_params['veltype'],
               uvtaper=True,outertaper=clean_params['outertaper'])
    #
    # Now actually clean it
    #
    casa.clean(vis=regrid_vis,imagename=imagename,field=field,spw='0',
               threshold='0mJy',niter=10000,interactive=True,
               imagermode='csclean',mode='velocity',multiscale=clean_params['multiscale'],
               gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'],
               restfreq=restfreq,start=clean_params['velstart'],width=clean_params['chanwidth'],
               nchan=clean_params['nchan'],
               outframe=clean_params['outframe'],veltype=clean_params['veltype'],
               uvtaper=True,outertaper=clean_params['outertaper'],
               mask=mask)
    logger.info("Done.")
    #
    # Primary beam correction
    #
    logger.info("Performing primary beam correction...")
    casa.impbcor(imagename='{0}.image'.format(imagename),
                 pbimage='{0}.flux'.format(imagename),
                 outfile='{0}.pbcor'.format(imagename))
    logger.info("Done.")

def auto_channel_clean_line_spws(field='',vis='',my_line_spws='',mask='',
                                 clean_params={},threshold='',config=None):
    """
    Clean all line spws non-interactively

    Inputs:
      field        = field to be imaged
      vis          = measurement set
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
    for spw in my_line_spws.split(','):
        #
        # Get restfreq
        #
        spw_ind = my_line_spws.split(',').index(spw)
        restfreq = config.get("Clean","restfreqs").split(',')[spw_ind]
        #
        # clean spw
        #
        imagename='{0}.spw{1}.channel.clean'.format(field,spw)
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
                   uvtaper=True,outertaper=clean_params['outertaper'],
                   mask=mask)
        logger.info("Done.")
        #
        # Primary beam correction
        #
        logger.info("Performing primary beam correction...")
        casa.impbcor(imagename='{0}.image'.format(imagename),
                     pbimage='{0}.flux'.format(imagename),
                     outfile='{0}.pbcor'.format(imagename))
        logger.info("Done.")

def contplot(field,my_cont_spws=''):
    """
    Generate pdf document of continuum diagnostic plots:
    1. Dirty image
    2. Clean image
    3. Residual image
    4. alpha image
    5. alpha error image
    6. Beta image

    Inputs:
      field  = field we're plotting
      my_cont_spws = continuum spectral windows

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    goodplots = []
    #
    # Generate combined continuum spw images
    #
    logger.info("Generating combined continuum images...")
    #
    # Dirty image
    #
    raster = {'file':'{0}.cont.mfs.dirty.image.tt0'.format(field),
              'colormap':'Greyscale 2',
              'colorwedge':True}
    out = '{0}.cont.mfs.dirty.image.png'.format(field)
    casa.imview(raster=raster,zoom=2,out=out)
    if os.path.exists(out): goodplots.append(out)
    #
    # Clean image
    #
    raster = {'file':'{0}.cont.mfs.clean.image.tt0'.format(field),
              'colormap':'Greyscale 2',
              'colorwedge':True}
    out = '{0}.cont.mfs.clean.image.png'.format(field)
    casa.imview(raster=raster,zoom=2,out=out)
    if os.path.exists(out): goodplots.append(out)
    #
    # Residual image
    #
    raster = {'file':'{0}.cont.mfs.clean.residual.tt0'.format(field),
              'colormap':'Greyscale 2',
              'colorwedge':True}
    out = '{0}.cont.mfs.clean.residual.png'.format(field)
    casa.imview(raster=raster,zoom=2,out=out)
    if os.path.exists(out): goodplots.append(out)
    #
    # Alpha image
    #
    raster = {'file':'{0}.cont.mfs.clean.image.alpha'.format(field),
              'colormap':'Greyscale 2',
              'colorwedge':True,
              'range':[-5,5]}
    out = '{0}.cont.mfs.clean.alpha.png'.format(field)
    casa.imview(raster=raster,zoom=2,out=out)
    if os.path.exists(out): goodplots.append(out)
    #
    # Alpha error
    #
    raster = {'file':'{0}.cont.mfs.clean.image.alpha.error'.format(field),
              'colormap':'Greyscale 2',
              'colorwedge':True,
              'range':[0,1]}
    out = '{0}.cont.mfs.clean.alpha.error.png'.format(field)
    casa.imview(raster=raster,zoom=2,out=out)
    if os.path.exists(out): goodplots.append(out)
    #
    # Beta image
    #
    raster = {'file':'{0}.cont.mfs.clean.image.beta'.format(field),
              'colormap':'Greyscale 2',
              'colorwedge':True,
              'range':[-5,5]}
    out = '{0}.cont.mfs.clean.beta.png'.format(field)
    casa.imview(raster=raster,zoom=2,out=out)
    if os.path.exists(out): goodplots.append(out)
    logger.info("Done.")
    #
    # Loop over spws and generate images
    #
    for spw in my_cont_spws.split(','):
        logger.info("Generating spw {0} continuum images...".format(spw))
        #
        # Dirty image
        #
        raster = {'file':'{0}.spw{1}.mfs.dirty.image.tt0'.format(field,spw),
                  'colormap':'Greyscale 2',
                  'colorwedge':True}
        out = '{0}.spw{1}.mfs.dirty.image.png'.format(field,spw)
        casa.imview(raster=raster,zoom=2,out=out)
        if os.path.exists(out): goodplots.append(out)
        #
        # Clean image
        #
        raster = {'file':'{0}.spw{1}.mfs.clean.image.tt0'.format(field,spw),
                  'colormap':'Greyscale 2',
                  'colorwedge':True}
        out = '{0}.spw{1}.mfs.clean.image.png'.format(field,spw)
        casa.imview(raster=raster,zoom=2,out=out)
        if os.path.exists(out): goodplots.append(out)
        #
        # Residual image
        #
        raster = {'file':'{0}.spw{1}.mfs.clean.residual.tt0'.format(field,spw),
                  'colormap':'Greyscale 2',
                  'colorwedge':True}
        out = '{0}.spw{1}.mfs.clean.residual.png'.format(field,spw)
        casa.imview(raster=raster,zoom=2,out=out)
        if os.path.exists(out): goodplots.append(out)
        logger.info("Done.")
    #
    # Generate PDF of plots
    #
    # need to fix filenames so LaTeX doesn't complain
    for i in range(len(goodplots)):
        goodplots[i] = '{'+goodplots[i].split('.png')[0]+'}.png'
    logger.info("Generating PDF...")
    with open('{0}.contplots.tex'.format(field),'w') as f:
        f.write(r"\documentclass{article}"+"\n")
        f.write(r"\usepackage{graphicx}"+"\n")
        f.write(r"\usepackage[margin=0.1cm]{geometry}"+"\n")
        f.write(r"\begin{document}"+"\n")
        for i in range(0,len(goodplots),6):
            f.write(r"\begin{figure}"+"\n")
            f.write(r"\centering"+"\n")
            f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i]+"}\n")
            if len(goodplots) > i+3: f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i+3]+"}\n")
            f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i+1]+"}\n")
            if len(goodplots) > i+4: f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i+4]+"}\n")
            f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i+2]+"}\n")
            if len(goodplots) > i+5: f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i+5]+"}\n")
            f.write(r"\end{figure}"+"\n")
            f.write(r"\clearpage"+"\n")
        f.write(r"\end{document}")
    os.system('pdflatex -interaction=batchmode {0}.contplots.tex'.format(field))
    logger.info("Done.")

def main(field,vis='',config_file=''):
    """
    Generate continuum and line images in various ways

    Inputs:
      field       = field name to clean
      vis         = measurement set containing all data for field
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
    cont_threshold = None
    line_threshold = None
    my_cont_spws,my_line_spws,clean_params = setup(vis=vis,config=config)
    #
    # Regrid velocity axis
    #
    regrid_velocity(vis=vis,spws=my_line_spws,config=config,clean_params=clean_params)
    #
    # Prompt the user with a menu for each option
    #
    while True:
        print("0.  Dirty image combined continuum spws (MFS)")
        print("1.  Dirty image combined continuum spws (channel)")
        print("2.  Manually clean combined continuum spws (MFS)")
        print("3.  Manually clean combined continuum spws (channel)")
        print("4.  Dirty image each continuum spw (MFS)")
        print("5.  Manually clean each continuum spw (MFS)")
        print("6.  Manually clean single continuum spw (MFS)")
        print("7.  Set continuum spw clean threshold")
        print("8.  Automatically clean each continuum spw (MFS)")
        print("9.  Dirty image each line spw (channel)")
        print("10. Manually clean each line spw (channel)")
        print("11. Manually clean single line spw (channel)")
        print("12. Set line spw clean threshold")
        print("13. Automatically clean each line spw (channel)")
        print("14. Generate continuum diagnostic plots")
        print("q [quit]")
        answer = raw_input("> ")
        if answer == '0':
            mfs_dirty_cont(field=field,vis=vis,
                           my_cont_spws=my_cont_spws,
                           clean_params=clean_params)
        elif answer == '1':
            channel_dirty_cont(field=field,vis=vis,
                               my_cont_spws=my_cont_spws,
                               clean_params=clean_params)
        elif answer == '2':
            mfs_clean_cont(field=field,vis=vis,
                           my_cont_spws=my_cont_spws,
                           clean_params=clean_params)
        elif answer == '3':
            channel_clean_cont(field=field,vis=vis,
                               my_cont_spws=my_cont_spws,
                               clean_params=clean_params)
        elif answer == '4':
            mfs_dirty_cont_spws(field=field,vis=vis,
                                my_cont_spws=my_cont_spws,
                                clean_params=clean_params)
        elif answer == '5':
            mfs_clean_cont_spws(field=field,vis=vis,
                                my_cont_spws=my_cont_spws,
                                clean_params=clean_params)
        elif answer == '6':
            print("Which spw do you want to clean?")
            spw = raw_input('> ')
            mask = '{0}.cont.mfs.clean.mask'.format(field)
            mfs_clean_cont_spw(field=field,vis=vis,spw=spw,
                               mask=mask,clean_params=clean_params)
        elif answer == '7':
            print("Enter continuum clean threshold (i.e. 1.1mJy)")
            cont_threshold = raw_input('> ')
        elif answer == '8':
            if cont_threshold is None:
                logger.warn("Must set continuum clean threshold first!")
            else:
                mask = '{0}.cont.mfs.clean.mask'.format(field)
                auto_mfs_clean_cont_spws(field=field,vis=vis,
                                         my_cont_spws=my_cont_spws,
                                         threshold=cont_threshold,
                                         mask=mask,
                                         clean_params=clean_params)
        elif answer == '9':
            channel_dirty_line_spws(field=field,vis=vis,
                                    my_line_spws=my_line_spws,
                                    clean_params=clean_params,
                                    config=config)
        elif answer == '10':
            channel_clean_line_spws(field=field,vis=vis,
                                    my_line_spws=my_line_spws,
                                    clean_params=clean_params,
                                    config=config)
        elif answer == '11':
            print("Which spw do you want to clean?")
            spw = raw_input('> ')
            mask = '{0}.cont.mfs.clean.mask'.format(field)
            channel_clean_line_spw(field=field,vis=vis,spw=spw,
                                   my_line_spws=my_line_spws,
                                   mask=mask,clean_params=clean_params,
                                   config=config)
        elif answer == '12':
            print("Enter line clean threshold (i.e. 1.1mJy)")
            line_threshold = raw_input('> ')
        elif answer == '13':
            if line_threshold is None:
                logger.warn("Must set line clean threshold first!")
            else:
                mask = '{0}.cont.mfs.clean.mask'.format(field)
                auto_channel_clean_line_spws(field=field,vis=vis,
                                my_line_spws=my_line_spws,
                                threshold=line_threshold,mask=mask,
                                clean_params=clean_params,
                                config=config)
        elif answer == '14':
            contplot(field,my_cont_spws=my_cont_spws)
        elif answer.lower() == 'q' or answer.lower() == 'quit':
            break
        else:
            print("Input not recognized.")
