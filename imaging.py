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
import matplotlib.pyplot as plt

__VERSION__ = "1.0"

# load logging configuration file
logging.config.fileConfig('logging.conf')

# Maximum number of iterations allowed for cleaning (prevents
# endless auto-cleaning when there is a problem).
_MAX_ITER = 500

# Number of iterations per cycle during interactive clean
_NPERCYCLE = 10

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
    freqstart = "{0}MHz".format(config.getfloat("Clean","freqstart"))
    freqwidth = "{0}MHz".format(config.getfloat("Clean","freqwidth"))
    nfreqchan = config.getint("Clean","nfreqchan")
    clean_params = {"lineids":lineids,"restfreqs":restfreqs,"imsize":imsize,
                    "cell":cell,"weighting":weighting,"robust":robust,
                    "multiscale":multiscale,"gain":gain,"cyclefactor":cyclefactor,
                    "velstart":velstart,"chanwidth":chanwidth,
                    "nchan":nchan,"outframe":outframe,"veltype":veltype,
                    "cvelstart":cvelstart,"cvelnchan":cvelnchan,
                    "freqstart":freqstart,"freqwidth":freqwidth,
                    "nfreqchan":nfreqchan,
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
    # Dirty image continuum
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
                 outfile='{0}.pbcor'.format(imagename),
                 overwrite=True)
    logger.info("Done.")
    #
    # Export to fits
    #
    logger.info("Exporting fits file...")
    casa.exportfits(imagename='{0}.pbcor'.format(imagename),
                    fitsimage='{0}.pbcor.fits'.format(imagename),
                    overwrite=True)
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
    # Dirty image continuum
    #
    imagename='{0}.cont.channel.dirty'.format(field)
    logger.info("Generating dirty continuum cube (channel)...")
    casa.clean(vis=vis,imagename=imagename,field=field,spw=my_cont_spws,
               threshold='0mJy',niter=0,interactive=False,
               imagermode='csclean',mode='channel',multiscale=clean_params['multiscale'],
               gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'],
               uvtaper=True,outertaper=clean_params['outertaper'])
    logger.info("Done.")
    #
    # Primary beam correction
    #
    logger.info("Performing primary beam correction...")
    casa.impbcor(imagename='{0}.image'.format(imagename),
                 pbimage='{0}.flux'.format(imagename),
                 outfile='{0}.pbcor'.format(imagename),
                 overwrite=True)
    logger.info("Done.")
    #
    # Export to fits
    #
    logger.info("Exporting fits file...")
    casa.exportfits(imagename='{0}.pbcor'.format(imagename),
                    fitsimage='{0}.pbcor.fits'.format(imagename),
                    overwrite=True)
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
               threshold='0mJy',niter=_MAX_ITER,interactive=True,npercycle=_NPERCYCLE,nterms=clean_params['nterms'],
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
                 outfile='{0}.pbcor'.format(imagename),
                 overwrite=True)
    logger.info("Done.")
    #
    # Export to fits
    #
    logger.info("Exporting fits file...")
    casa.exportfits(imagename='{0}.pbcor'.format(imagename),
                    fitsimage='{0}.pbcor.fits'.format(imagename),
                    overwrite=True)
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
    imagename='{0}.cont.channel.clean'.format(field)
    logger.info("Cleaning continuum cube (channel)...")
    casa.clean(vis=vis,imagename=imagename,field=field,spw=my_cont_spws,
               threshold='0mJy',niter=_MAX_ITER,interactive=True,npercycle=_NPERCYCLE,
               imagermode='csclean',mode='channel',multiscale=clean_params['multiscale'],
               gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'],
               uvtaper=True,outertaper=clean_params['outertaper'])
    logger.info("Done.")
    #
    # Primary beam correction
    #
    logger.info("Performing primary beam correction...")
    casa.impbcor(imagename='{0}.image'.format(imagename),
                 pbimage='{0}.flux'.format(imagename),
                 outfile='{0}.pbcor'.format(imagename),
                 overwrite=True)
    logger.info("Done.")
    #
    # Export to fits
    #
    logger.info("Exporting fits file...")
    casa.exportfits(imagename='{0}.pbcor'.format(imagename),
                    fitsimage='{0}.pbcor.fits'.format(imagename),
                    overwrite=True)
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
        # Dirty image
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
                     outfile='{0}.pbcor'.format(imagename),
                     overwrite=True)
        logger.info("Done.")
        #
        # Export to fits
        #
        logger.info("Exporting fits file...")
        casa.exportfits(imagename='{0}.pbcor'.format(imagename),
                        fitsimage='{0}.pbcor.fits'.format(imagename),
                        overwrite=True)
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
                   threshold='0mJy',niter=_MAX_ITER,interactive=True,npercycle=_NPERCYCLE,nterms=clean_params['nterms'],
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
                     outfile='{0}.pbcor'.format(imagename),
                     overwrite=True)
        logger.info("Done.")
        #
        # Export to fits
        #
        logger.info("Exporting fits file...")
        casa.exportfits(imagename='{0}.pbcor'.format(imagename),
                        fitsimage='{0}.pbcor.fits'.format(imagename),
                        overwrite=True)
        logger.info("Done.")

def mfs_clean_cont_spw(field='',vis='',spw='',
                        clean_params={}):
    """
    Clean single continuum spw (MFS)

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      spw          = spectral window to clean
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
               threshold='0mJy',niter=_MAX_ITER,interactive=True,npercycle=_NPERCYCLE,nterms=clean_params['nterms'],
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
                 outfile='{0}.pbcor'.format(imagename),
                 overwrite=True)
    logger.info("Done.")
    #
    # Export to fits
    #
    logger.info("Exporting fits file...")
    casa.exportfits(imagename='{0}.pbcor'.format(imagename),
                    fitsimage='{0}.pbcor.fits'.format(imagename),
                    overwrite=True)
    logger.info("Done.")

def auto_mfs_clean_cont_spws(field='',vis='',my_cont_spws='',
                             manual_spw='',threshold='',
                             clean_params={}):
    """
    Clean each continuum spw (MFS) using mask and threshold determined
    from manually cleaning another continuum spw. Delete that
    continuum spw so each is automatically cleaned in the same way.

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      my_cont_spws = comma-separated string of continuum spws
      manual_spw   = spectral window that was cleaned manually
      threshold    = clean threshold
      clean_params = dictionary of clean parameters

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Use mask from manually cleaned spw
    #
    mask = '{0}.spw{1}.mfs.clean.mask'.format(field,manual_spw)
    #
    # Delete image of manually cleaned spw
    #
    imagename = '{0}.spw{1}.mfs.clean'.format(field,manual_spw)
    shutil.rmtree('{0}.image'.format(imagename))
    shutil.rmtree('{0}.pbcor'.format(imagename))
    shutil.rmtree('{0}.residual'.format(imagename))
    shutil.rmtree('{0}.model'.format(imagename))
    shutil.rmtree('{0}.psf'.format(imagename))
    shutil.rmtree('{0}.flux'.format(imagename))
    #
    # Loop over spectral windows
    #
    for spw in my_cont_spws.split(','):
        #
        # Clean
        #
        imagename='{0}.spw{1}.mfs.clean'.format(field,spw)
        logger.info("Automatically cleaning continuum spw {0} (MFS)...".format(spw))
        casa.clean(vis=vis,imagename=imagename,field=field,spw=spw,
                   threshold=threshold,niter=_MAX_ITER,interactive=False,nterms=clean_params['nterms'],
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
                     outfile='{0}.pbcor'.format(imagename),
                     overwrite=True)
        logger.info("Done.")
        #
        # Export to fits
        #
        logger.info("Exporting fits file...")
        casa.exportfits(imagename='{0}.pbcor'.format(imagename),
                        fitsimage='{0}.pbcor.fits'.format(imagename),
                        overwrite=True)
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
        # dirty image spw
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
                     outfile='{0}.pbcor'.format(imagename),
                     overwrite=True)
        logger.info("Done.")
        #
        # Export to fits
        #
        logger.info("Exporting fits file...")
        casa.exportfits(imagename='{0}.pbcor'.format(imagename),
                        fitsimage='{0}.pbcor.fits'.format(imagename),
                        velocity=True,overwrite=True)
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
                   threshold='0mJy',niter=_MAX_ITER,interactive=True,npercycle=_NPERCYCLE,
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
                     outfile='{0}.pbcor'.format(imagename),
                     overwrite=True)
        logger.info("Done.")
        #
        # Export to fits
        #
        logger.info("Exporting fits file...")
        casa.exportfits(imagename='{0}.pbcor'.format(imagename),
                        fitsimage='{0}.pbcor.fits'.format(imagename),
                        velocity=True,overwrite=True)
        logger.info("Done.")


def channel_clean_line_spw(field='',vis='',spw='',my_line_spws='',
                           clean_params={},config=None):
    """
    Clean single line spws manually

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      spw          = spw to clean
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
               threshold='0mJy',niter=_MAX_ITER,interactive=True,npercycle=_NPERCYCLE,
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
                 outfile='{0}.pbcor'.format(imagename),
                 overwrite=True)
    logger.info("Done.")
    #
    # Export to fits
    #
    logger.info("Exporting fits file...")
    casa.exportfits(imagename='{0}.pbcor'.format(imagename),
                    fitsimage='{0}.pbcor.fits'.format(imagename),
                    velocity=True,overwrite=True)
    logger.info("Done.")


def auto_channel_clean_line_spws(field='',vis='',my_line_spws='',
                                 manual_spw='',threshold='',
                                 clean_params={},config=None):
    """
    Clean all line spws non-interactively using clean mask and threshold
    determined by manually cleaning a single line spectral window.
    Delete the image of that manually cleaned spw so each spw is
    cleaned uniformly.

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      my_line_spws = comma-separated string of line spws
      manual_spw   = spw that was cleaned manually
      threshold    = clean threshold
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
    # Use mask from manually cleaned spw
    #
    mask = '{0}.spw{1}.channel.clean.mask'.format(field,manual_spw)
    #
    # Delete image of manually cleaned spw
    #
    imagename = '{0}.spw{1}.channel.clean'.format(field,manual_spw)
    shutil.rmtree('{0}.image'.format(imagename))
    shutil.rmtree('{0}.pbcor'.format(imagename))
    shutil.rmtree('{0}.residual'.format(imagename))
    shutil.rmtree('{0}.model'.format(imagename))
    shutil.rmtree('{0}.psf'.format(imagename))
    shutil.rmtree('{0}.flux'.format(imagename))
    #
    # Loop over spectral windows
    #
    for spw in my_line_spws.split(','):
        #
        # Get restfreq
        #
        spw_ind = my_line_spws.split(',').index(spw)
        restfreq = config.get("Clean","restfreqs").split(',')[spw_ind]
        #
        # Dirty image so we can copy mask
        #
        imagename='{0}.spw{1}.channel.clean'.format(field,spw)
        regrid_vis = vis+'.spw{0}.cvel'.format(spw)
        casa.clean(vis=regrid_vis,imagename=imagename,field=field,spw='0',
                   threshold=threshold,niter=0,interactive=False,
                   imagermode='csclean',mode='velocity',multiscale=clean_params['multiscale'],
                   gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
                   imsize=clean_params['imsize'],cell=clean_params['cell'],
                   weighting=clean_params['weighting'],robust=clean_params['robust'],
                   restfreq=restfreq,start=clean_params['velstart'],width=clean_params['chanwidth'],
                   nchan=clean_params['nchan'],
                   outframe=clean_params['outframe'],veltype=clean_params['veltype'],
                   uvtaper=True,outertaper=clean_params['outertaper'])
        #
        # Copy mask
        #
        logger.info("Copying mask for spw {0}...".format(spw,restfreq))
        casa.makemask(mode='expand',
                      inpimage='{0}.image'.format(imagename),
                      inpmask=mask,
                      output='{0}.mask'.format(imagename),
                      inpfreqs=[0])
        logger.info("Done.")
        #
        # clean spw
        #
        logger.info("Cleaning spw {0} (restfreq: {1})...".format(spw,restfreq))
        casa.clean(vis=regrid_vis,imagename=imagename,field=field,spw='0',
                   threshold=threshold,niter=_MAX_ITER,interactive=False,
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
                     outfile='{0}.pbcor'.format(imagename),
                     overwrite=True)
        logger.info("Done.")
        #
        # Export to fits
        #
        logger.info("Exporting fits file...")
        casa.exportfits(imagename='{0}.pbcor'.format(imagename),
                        fitsimage='{0}.pbcor.fits'.format(imagename),
                        velocity=True,overwrite=True)
        logger.info("Done.")


def contplot(field,my_cont_spws='',clean_params={}):
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
      clean_params = dictionary of clean parameters

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    goodplots = []
    center_x = int(clean_params['imsize'][0]/2)
    center_y = int(clean_params['imsize'][1]/2)
    blc = [center_x-int(center_x/2),center_y-int(center_y/2)]
    trc = [center_x+int(center_x/2),center_y+int(center_y/2)]
    zoom = {'blc':blc,'trc':trc}
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
    casa.imview(raster=raster,zoom=zoom,out=out)
    if os.path.exists(out): goodplots.append(out)
    #
    # Clean image
    #
    raster = {'file':'{0}.cont.mfs.clean.image.tt0'.format(field),
              'colormap':'Greyscale 2',
              'colorwedge':True}
    out = '{0}.cont.mfs.clean.image.png'.format(field)
    casa.imview(raster=raster,zoom=zoom,out=out)
    if os.path.exists(out): goodplots.append(out)
    #
    # Residual image
    #
    raster = {'file':'{0}.cont.mfs.clean.residual.tt0'.format(field),
              'colormap':'Greyscale 2',
              'colorwedge':True}
    out = '{0}.cont.mfs.clean.residual.png'.format(field)
    casa.imview(raster=raster,zoom=zoom,out=out)
    if os.path.exists(out): goodplots.append(out)
    #
    # Alpha image
    #
    raster = {'file':'{0}.cont.mfs.clean.image.alpha'.format(field),
              'colormap':'Greyscale 2',
              'colorwedge':True,
              'range':[-5,5]}
    out = '{0}.cont.mfs.clean.alpha.png'.format(field)
    casa.imview(raster=raster,zoom=zoom,out=out)
    if os.path.exists(out): goodplots.append(out)
    #
    # Alpha error
    #
    raster = {'file':'{0}.cont.mfs.clean.image.alpha.error'.format(field),
              'colormap':'Greyscale 2',
              'colorwedge':True,
              'range':[0,1]}
    out = '{0}.cont.mfs.clean.alpha.error.png'.format(field)
    casa.imview(raster=raster,zoom=zoom,out=out)
    if os.path.exists(out): goodplots.append(out)
    #
    # Beta image
    #
    raster = {'file':'{0}.cont.mfs.clean.image.beta'.format(field),
              'colormap':'Greyscale 2',
              'colorwedge':True,
              'range':[-5,5]}
    out = '{0}.cont.mfs.clean.beta.png'.format(field)
    casa.imview(raster=raster,zoom=zoom,out=out)
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
        casa.imview(raster=raster,zoom=zoom,out=out)
        if os.path.exists(out): goodplots.append(out)
        #
        # Clean image
        #
        raster = {'file':'{0}.spw{1}.mfs.clean.image.tt0'.format(field,spw),
                  'colormap':'Greyscale 2',
                  'colorwedge':True}
        out = '{0}.spw{1}.mfs.clean.image.png'.format(field,spw)
        casa.imview(raster=raster,zoom=zoom,out=out)
        if os.path.exists(out): goodplots.append(out)
        #
        # Residual image
        #
        raster = {'file':'{0}.spw{1}.mfs.clean.residual.tt0'.format(field,spw),
                  'colormap':'Greyscale 2',
                  'colorwedge':True}
        out = '{0}.spw{1}.mfs.clean.residual.png'.format(field,spw)
        casa.imview(raster=raster,zoom=zoom,out=out)
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

def lineplot(field,my_line_spws='',clean_params={}):
    """
    Generate pdf document of spectral line diagnostic plots:
    For each spectral window
    1. spectrum, center pixel

    Inputs:
      field  = field we're plotting
      my_line_spws = line spectral windows
      clean_params = dictionary of clean parameters

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    goodplots = []
    center_x = int(clean_params['imsize'][0]/2)
    center_y = int(clean_params['imsize'][1]/2)
    blc = [center_x-int(center_x/2),center_y-int(center_y/2)]
    trc = [center_x+int(center_x/2),center_y+int(center_y/2)]
    center_chan = int(clean_params['nchan']/2)
    zoom = {'channel':center_chan,'blc':blc,'trc':trc}
    #
    # Loop over spws and generate images
    #
    for spw,lineid in zip(my_line_spws.split(','),clean_params['lineids']):
        logger.info("Generating spw {0} line images...".format(spw))
        #
        # Center pixel spectrum
        #
        imagename = '{0}.spw{1}.channel.clean.image'.format(field,spw)
        specfile = '{0}.spw{1}.channel.clean.image.specflux'.format(field,spw)
        region='circle[[100pix,100pix],0.1pix]'
        casa.specflux(imagename=imagename,region=region,function='mean',
                      unit='km/s',logfile=specfile,overwrite=True)
        if os.path.exists(specfile):
            spec = np.genfromtxt(specfile,dtype=None,comments='#',
                                 names=['channel','npix','frequency',
                                        'velocity','flux'])
            plt.ioff()
            fig,ax = plt.subplots()
            ax.plot(spec['velocity'],1000.*spec['flux'],'k-')
            ax.set_xlabel('Velocity (km/s)')
            ax.set_ylabel('Flux (mJy)')
            ax.set_xlim(np.min(spec['velocity']),np.max(spec['velocity']))
            ax.set_title(lineid)
            out = '{0}.spw{1}.channel.clean.specflux.png'.format(field,spw)
            fig.savefig(out)
            plt.close(fig)
            plt.ion()
            goodplots.append(out)
    #
    # Generate PDF of plots
    #
    # need to fix filenames so LaTeX doesn't complain
    for i in range(len(goodplots)):
        goodplots[i] = '{'+goodplots[i].split('.png')[0]+'}.png'
    logger.info("Generating PDF...")
    with open('{0}.lineplots.tex'.format(field),'w') as f:
        f.write(r"\documentclass{article}"+"\n")
        f.write(r"\usepackage{graphicx}"+"\n")
        f.write(r"\usepackage[margin=0.1cm]{geometry}"+"\n")
        f.write(r"\begin{document}"+"\n")
        for i in range(0,len(goodplots),6):
            f.write(r"\begin{figure}"+"\n")
            f.write(r"\centering"+"\n")
            f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i]+"}\n")
            if len(goodplots) > i+1: f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i+1]+"}\n")
            if len(goodplots) > i+2: f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i+2]+"}\n")
            if len(goodplots) > i+3: f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i+3]+"}\n")
            if len(goodplots) > i+4: f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i+4]+"}\n")
            if len(goodplots) > i+5: f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i+5]+"}\n")
            f.write(r"\end{figure}"+"\n")
            f.write(r"\clearpage"+"\n")
        f.write(r"\end{document}")
    os.system('pdflatex -interaction=batchmode {0}.lineplots.tex'.format(field))
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
        print("7.  Automatically clean each continuum spw (MFS)")
        print("8.  Dirty image each line spw (channel)")
        print("9.  Manually clean each line spw (channel)")
        print("10. Manually clean single line spw (channel)")
        print("11. Automatically clean each line spw (channel)")
        print("12. Generate continuum diagnostic plots")
        print("13. Generate spectral line diagnostic plots")
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
            mfs_clean_cont_spw(field=field,vis=vis,spw=spw,
                               clean_params=clean_params)
        elif answer == '7':
            print("Which spw did you clean manually?")
            manual_spw = raw_input('> ')
            print("Enter clean threshold (i.e. 1.1mJy)")
            cont_threshold = raw_input('> ')
            auto_mfs_clean_cont_spws(field=field,vis=vis,
                                     my_cont_spws=my_cont_spws,
                                     manual_spw=manual_spw,
                                     threshold=cont_threshold,
                                     clean_params=clean_params)
        elif answer == '8':
            channel_dirty_line_spws(field=field,vis=vis,
                                    my_line_spws=my_line_spws,
                                    clean_params=clean_params,
                                    config=config)
        elif answer == '9':
            channel_clean_line_spws(field=field,vis=vis,
                                    my_line_spws=my_line_spws,
                                    clean_params=clean_params,
                                    config=config)
        elif answer == '10':
            print("Which spw do you want to clean?")
            spw = raw_input('> ')
            channel_clean_line_spw(field=field,vis=vis,spw=spw,
                                   my_line_spws=my_line_spws,
                                   clean_params=clean_params,
                                   config=config)
        elif answer == '11':
            print("Which spw did you clean manually?")
            manual_spw = raw_input('> ')
            print("Enter clean threshold (i.e. 1.1mJy)")
            line_threshold = raw_input('> ')
            auto_channel_clean_line_spws(field=field,vis=vis,
                                         my_line_spws=my_line_spws,
                                         manual_spw=manual_spw,
                                         threshold=line_threshold,
                                         clean_params=clean_params,
                                         config=config)
        elif answer == '12':
            contplot(field,my_cont_spws=my_cont_spws,
                     clean_params=clean_params)
        elif answer == '13':
            lineplot(field,my_line_spws=my_line_spws,
                     clean_params=clean_params)
        elif answer.lower() == 'q' or answer.lower() == 'quit':
            break
        else:
            print("Input not recognized.")
