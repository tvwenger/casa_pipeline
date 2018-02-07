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
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.patches import Ellipse

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

def mfs_dirty_cont(field='',vis='',my_cont_spws='',clean_params={},
                   uvtaper=False):
    """
    Dirty image continuum spws using multi-frequency synthesis

    Inputs:
      field        = field to be cleaned
      vis          = measurement set
      my_cont_spws = comma-separated string of continuum spws
      clean_params = dictionary of clean parameters
      uvtaper      = if True, apply UV tapering

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
    if uvtaper:
        imagename = imagename + '.uvtaper'
    logger.info("Generating dirty continuum image (MFS)...")
    casa.clean(vis=vis,imagename=imagename,field=field,spw=my_cont_spws,
               threshold='0mJy',niter=0,interactive=False,nterms=clean_params['nterms'],
               imagermode='csclean',mode='mfs',multiscale=clean_params['multiscale'],
               gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'],
               uvtaper=uvtaper,outertaper=clean_params['outertaper'])
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
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.image.tt0'.format(imagename),
                    fitsimage='{0}.image.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.residual.tt0'.format(imagename),
                    fitsimage='{0}.residual.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.image.alpha'.format(imagename),
                    fitsimage='{0}.alpha.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.image.alpha.error'.format(imagename),
                    fitsimage='{0}.alpha.error.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.image.beta'.format(imagename),
                    fitsimage='{0}.beta.fits'.format(imagename),
                    overwrite=True,history=False)
    logger.info("Done.")

def channel_dirty_cont(field='',vis='',my_cont_spws='',clean_params={},
                       uvtaper=False):
    """
    Dirty image continuum spws channel-by-channel

    Inputs:
      field        = field to be cleaned
      vis          = measurement set
      my_cont_spws = comma-separated string of continuum spws
      clean_params = dictionary of clean parameters
      uvtaper      = if True, apply UV tapering

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
    if uvtaper:
        imagename = imagename + '.uvtaper'
    logger.info("Generating dirty continuum cube (channel)...")
    casa.clean(vis=vis,imagename=imagename,field=field,spw=my_cont_spws,
               threshold='0mJy',niter=0,interactive=False,
               imagermode='csclean',mode='channel',multiscale=clean_params['multiscale'],
               gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'],
               uvtaper=uvtaper,outertaper=clean_params['outertaper'])
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
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.image'.format(imagename),
                    fitsimage='{0}.image.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.residual'.format(imagename),
                    fitsimage='{0}.residual.fits'.format(imagename),
                    overwrite=True,history=False)    
    logger.info("Done.")

def mfs_clean_cont(field='',vis='',my_cont_spws='',clean_params={},
                   uvtaper=False):
    """
    Clean continuum spws using multi-frequency synthesis

    Inputs:
      field        = field to be cleaned
      vis          = measurement set
      my_cont_spws = comma-separated string of continuum spws
      clean_params = dictionary of clean parameters
      uvtaper      = if True, apply UV tapering

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
    if uvtaper:
        imagename = imagename + '.uvtaper'
    logger.info("Cleaning continuum image (MFS)...")
    casa.clean(vis=vis,imagename=imagename,field=field,spw=my_cont_spws,
               threshold='0mJy',niter=_MAX_ITER,interactive=True,npercycle=_NPERCYCLE,nterms=clean_params['nterms'],
               imagermode='csclean',mode='mfs',multiscale=clean_params['multiscale'],
               gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'],
               uvtaper=uvtaper,outertaper=clean_params['outertaper'])
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
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.image.tt0'.format(imagename),
                    fitsimage='{0}.image.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.residual.tt0'.format(imagename),
                    fitsimage='{0}.residual.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.image.alpha'.format(imagename),
                    fitsimage='{0}.alpha.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.image.alpha.error'.format(imagename),
                    fitsimage='{0}.alpha.error.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.image.beta'.format(imagename),
                    fitsimage='{0}.beta.fits'.format(imagename),
                    overwrite=True,history=False)

    logger.info("Done.")

def channel_clean_cont(field='',vis='',my_cont_spws='',clean_params={},
                       uvtaper=False):
    """
    Clean continuum channels

    Inputs:
      field        = field to be cleaned
      vis          = measurement set
      my_cont_spws = comma-separated string of continuum spws
      clean_params = dictionary of clean parameters
      uvtaper      = if True, apply UV tapering

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
    if uvtaper:
        imagename = imagename + '.uvtaper'
    logger.info("Cleaning continuum cube (channel)...")
    casa.clean(vis=vis,imagename=imagename,field=field,spw=my_cont_spws,
               threshold='0mJy',niter=_MAX_ITER,interactive=True,npercycle=_NPERCYCLE,
               imagermode='csclean',mode='channel',multiscale=clean_params['multiscale'],
               gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'],
               uvtaper=uvtaper,outertaper=clean_params['outertaper'])
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
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.image'.format(imagename),
                    fitsimage='{0}.image.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.residual'.format(imagename),
                    fitsimage='{0}.residual.fits'.format(imagename),
                    overwrite=True,history=False)
    logger.info("Done.")

def mfs_dirty_cont_spws(field='',vis='',my_cont_spws='',
                        clean_params={},uvtaper=False):
    """
    Dirty image each continuum spw (MFS)

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      my_cont_spws = comma-separated string of continuum spws
      clean_params = dictionary of clean parameters
      uvtaper      = if True, apply UV tapering

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
        if uvtaper:
            imagename = imagename + '.uvtaper'
        logger.info("Generating dirty image of spw {0} (MFS)...".format(spw))
        casa.clean(vis=vis,imagename=imagename,field=field,spw=spw,
                   threshold='0mJy',niter=0,interactive=False,nterms=clean_params['nterms'],
                   imagermode='csclean',mode='mfs',multiscale=clean_params['multiscale'],
                   gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
                   imsize=clean_params['imsize'],cell=clean_params['cell'],
                   weighting=clean_params['weighting'],robust=clean_params['robust'],
                   uvtaper=uvtaper,outertaper=clean_params['outertaper'])
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
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.image.tt0'.format(imagename),
                        fitsimage='{0}.image.fits'.format(imagename),
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.residual.tt0'.format(imagename),
                        fitsimage='{0}.residual.fits'.format(imagename),
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.image.alpha'.format(imagename),
                        fitsimage='{0}.alpha.fits'.format(imagename),
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.image.alpha.error'.format(imagename),
                        fitsimage='{0}.alpha.error.fits'.format(imagename),
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.image.beta'.format(imagename),
                        fitsimage='{0}.beta.fits'.format(imagename),
                        overwrite=True,history=False)
        logger.info("Done.")

def mfs_clean_cont_spws(field='',vis='',my_cont_spws='',
                        clean_params={},uvtaper=False):
    """
    Clean each continuum spw (MFS)

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      my_cont_spws = comma-separated string of continuum spws
      clean_params = dictionary of clean parameters
      uvtaper      = if True, apply UV tapering

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
        if uvtaper:
            imagename = imagename + '.uvtaper'
        logger.info("Cleaning continuum spw {0} (MFS)...".format(spw))
        casa.clean(vis=vis,imagename=imagename,field=field,spw=spw,
                   threshold='0mJy',niter=_MAX_ITER,interactive=True,npercycle=_NPERCYCLE,nterms=clean_params['nterms'],
                   imagermode='csclean',mode='mfs',multiscale=clean_params['multiscale'],
                   gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
                   imsize=clean_params['imsize'],cell=clean_params['cell'],
                   weighting=clean_params['weighting'],robust=clean_params['robust'],
                   uvtaper=uvtaper,outertaper=clean_params['outertaper'])
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
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.image.tt0'.format(imagename),
                        fitsimage='{0}.image.fits'.format(imagename),
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.residual.tt0'.format(imagename),
                        fitsimage='{0}.residual.fits'.format(imagename),
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.image.alpha'.format(imagename),
                        fitsimage='{0}.alpha.fits'.format(imagename),
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.image.alpha.error'.format(imagename),
                        fitsimage='{0}.alpha.error.fits'.format(imagename),
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.image.beta'.format(imagename),
                        fitsimage='{0}.beta.fits'.format(imagename),
                        overwrite=True,history=False)
        logger.info("Done.")

def mfs_clean_cont_spw(field='',vis='',spw='',
                        clean_params={},uvtaper=False):
    """
    Clean single continuum spw (MFS)

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      spw          = spectral window to clean
      clean_params = dictionary of clean parameters
      uvtaper      = if True, apply UV tapering

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
    if uvtaper:
        imagename = imagename + '.uvtaper'
    logger.info("Cleaning continuum spw {0} (MFS)...".format(spw))
    casa.clean(vis=vis,imagename=imagename,field=field,spw=spw,
               threshold='0mJy',niter=_MAX_ITER,interactive=True,npercycle=_NPERCYCLE,nterms=clean_params['nterms'],
               imagermode='csclean',mode='mfs',multiscale=clean_params['multiscale'],
               gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
               imsize=clean_params['imsize'],cell=clean_params['cell'],
               weighting=clean_params['weighting'],robust=clean_params['robust'],
               uvtaper=uvtaper,outertaper=clean_params['outertaper'])
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
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.image.tt0'.format(imagename),
                    fitsimage='{0}.image.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.residual.tt0'.format(imagename),
                    fitsimage='{0}.residual.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.image.alpha'.format(imagename),
                    fitsimage='{0}.alpha.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.image.alpha.error'.format(imagename),
                    fitsimage='{0}.alpha.error.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.image.beta'.format(imagename),
                    fitsimage='{0}.beta.fits'.format(imagename),
                    overwrite=True,history=False)
    logger.info("Done.")

def auto_mfs_clean_cont_spws(field='',vis='',my_cont_spws='',
                             manual_spw='',threshold='',
                             clean_params={},uvtaper=False):
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
      uvtaper      = if True, apply UV tapering

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
    mask = '{0}.spw{1}.mfs.clean'.format(field,manual_spw)
    if uvtaper:
        mask = mask + '.uvtaper'
    mask = mask + '.mask'
    #
    # Delete image of manually cleaned spw
    #
    imagename = '{0}.spw{1}.mfs.clean'.format(field,manual_spw)
    if uvtaper:
        imagename = imagename + '.uvtaper'
    if os.path.isdir('{0}.image'.format(imagename)):
        shutil.rmtree('{0}.image'.format(imagename))
    if os.path.isdir('{0}.pbcor'.format(imagename)):
        shutil.rmtree('{0}.pbcor'.format(imagename))
    if os.path.isdir('{0}.residual'.format(imagename)):
        shutil.rmtree('{0}.residual'.format(imagename))
    if os.path.isdir('{0}.model'.format(imagename)):
        shutil.rmtree('{0}.model'.format(imagename))
    if os.path.isdir('{0}.psf'.format(imagename)):
        shutil.rmtree('{0}.psf'.format(imagename))
    if os.path.isdir('{0}.flux'.format(imagename)):
        shutil.rmtree('{0}.flux'.format(imagename))
    #
    # Loop over spectral windows
    #
    for spw in my_cont_spws.split(','):
        #
        # Clean
        #
        imagename='{0}.spw{1}.mfs.clean'.format(field,spw)
        if uvtaper:
            imagename = imagename + '.uvtaper'
        logger.info("Automatically cleaning continuum spw {0} (MFS)...".format(spw))
        casa.clean(vis=vis,imagename=imagename,field=field,spw=spw,
                   threshold=threshold,niter=_MAX_ITER,interactive=False,nterms=clean_params['nterms'],
                   imagermode='csclean',mode='mfs',multiscale=clean_params['multiscale'],
                   gain=clean_params['gain'],cyclefactor=clean_params['cyclefactor'],
                   imsize=clean_params['imsize'],cell=clean_params['cell'],
                   weighting=clean_params['weighting'],robust=clean_params['robust'],
                   uvtaper=uvtaper,outertaper=clean_params['outertaper'],
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
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.image.tt0'.format(imagename),
                        fitsimage='{0}.image.fits'.format(imagename),
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.residual.tt0'.format(imagename),
                        fitsimage='{0}.residual.fits'.format(imagename),
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.image.alpha'.format(imagename),
                        fitsimage='{0}.alpha.fits'.format(imagename),
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.image.alpha.error'.format(imagename),
                        fitsimage='{0}.alpha.error.fits'.format(imagename),
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.image.beta'.format(imagename),
                        fitsimage='{0}.beta.fits'.format(imagename),
                        overwrite=True,history=False)
        logger.info("Done.")

def channel_dirty_line_spws(field='',vis='',my_line_spws='',
                            clean_params={},config=None,
                            uvtaper=False):
    """
    Dirty image all line spws

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      my_line_spws = comma-separated string of line spws to image
      clean_params = dictionary of clean parameters
      config       = ConfigParser object for this project
      uvtaper      = if True, apply UV tapering

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
        spw_ind = config.get("Spectral Windows","Line").split(',').index(spw)
        restfreq = config.get("Clean","restfreqs").split(',')[spw_ind]
        #
        # dirty image spw
        #
        imagename='{0}.spw{1}.channel.dirty'.format(field,spw)
        if uvtaper:
            imagename = imagename + '.uvtaper'
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
                   uvtaper=uvtaper,outertaper=clean_params['outertaper'])
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
                        velocity=True,overwrite=True,history=False)
        casa.exportfits(imagename='{0}.image'.format(imagename),
                        fitsimage='{0}.image.fits'.format(imagename),
                        velocity=True,overwrite=True,history=False)
        casa.exportfits(imagename='{0}.residual'.format(imagename),
                        fitsimage='{0}.residual.fits'.format(imagename),
                        velocity=True,overwrite=True,history=False)
        logger.info("Done.")

def channel_clean_line_spws(field='',vis='',my_line_spws='',
                            clean_params={},config=None,
                            uvtaper=False):
    """
    Clean all line spws manually

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      my_line_spws = comma-separated string of line spws to image
      clean_params = dictionary of clean parameters
      config       = ConfigParser object for this project
      uvtaper      = if True, apply UV tapering

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
        spw_ind = config.get("Spectral Windows","Line").split(',').index(spw)
        restfreq = config.get("Clean","restfreqs").split(',')[spw_ind]
        #
        # clean spw
        #
        imagename='{0}.spw{1}.channel.clean'.format(field,spw)
        if uvtaper:
            imagename = imagename + '.uvtaper'
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
                   uvtaper=uvtaper,outertaper=clean_params['outertaper'])
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
                   uvtaper=uvtaper,outertaper=clean_params['outertaper'])
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
                        velocity=True,overwrite=True,history=False)
        casa.exportfits(imagename='{0}.image'.format(imagename),
                        fitsimage='{0}.image.fits'.format(imagename),
                        velocity=True,overwrite=True,history=False)
        casa.exportfits(imagename='{0}.residual'.format(imagename),
                        fitsimage='{0}.residual.fits'.format(imagename),
                        velocity=True,overwrite=True,history=False)
        logger.info("Done.")


def channel_clean_line_spw(field='',vis='',spw='',
                           clean_params={},config=None,
                           uvtaper=False):
    """
    Clean single line spws manually

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      spw          = spw to clean
      clean_params = dictionary of clean parameters
      config       = ConfigParser object for this project
      uvtaper      = if True, apply UV tapering

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
    spw_ind = config.get("Spectral Windows","Line").split(',').index(spw)
    restfreq = config.get("Clean","restfreqs").split(',')[spw_ind]
    #
    # clean spw
    #
    imagename='{0}.spw{1}.channel.clean'.format(field,spw)
    if uvtaper:
        imagename = imagename + '.uvtaper'
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
               uvtaper=uvtaper,outertaper=clean_params['outertaper'])
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
               uvtaper=uvtaper,outertaper=clean_params['outertaper'])
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
                    velocity=True,overwrite=True,history=False)
    casa.exportfits(imagename='{0}.image'.format(imagename),
                    fitsimage='{0}.image.fits'.format(imagename),
                    velocity=True,overwrite=True,history=False)
    casa.exportfits(imagename='{0}.residual'.format(imagename),
                    fitsimage='{0}.residual.fits'.format(imagename),
                    velocity=True,overwrite=True,history=False)
    logger.info("Done.")


def auto_channel_clean_line_spws(field='',vis='',my_line_spws='',
                                 manual_spw='',threshold='',
                                 clean_params={},config=None,
                                 uvtaper=False):
    """
    Clean all line spws non-interactively using clean mask and threshold
    determined by manually cleaning a single line spectral window.
    Delete the image of that manually cleaned spw so each spw is
    cleaned uniformly.

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      my_line_spws = comma-separated string of line spws to image
      manual_spw   = spw that was cleaned manually
      threshold    = clean threshold
      clean_params = dictionary of clean parameters
      config       = ConfigParser object for this project
      uvtaper      = if True, apply UV tapering

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
    mask = '{0}.spw{1}.channel.clean'.format(field,manual_spw)
    if uvtaper:
        mask = mask + '.uvtaper'
    mask = mask + '.mask'
    #
    # Delete image of manually cleaned spw
    #
    imagename = '{0}.spw{1}.channel.clean'.format(field,manual_spw)
    if uvtaper:
        imagename = imagename + '.uvtaper'
    if os.path.isdir('{0}.image'.format(imagename)):
        shutil.rmtree('{0}.image'.format(imagename))
    if os.path.isdir('{0}.pbcor'.format(imagename)):
        shutil.rmtree('{0}.pbcor'.format(imagename))
    if os.path.isdir('{0}.residual'.format(imagename)):
        shutil.rmtree('{0}.residual'.format(imagename))
    if os.path.isdir('{0}.model'.format(imagename)):
        shutil.rmtree('{0}.model'.format(imagename))
    if os.path.isdir('{0}.psf'.format(imagename)):
        shutil.rmtree('{0}.psf'.format(imagename))
    if os.path.isdir('{0}.flux'.format(imagename)):
        shutil.rmtree('{0}.flux'.format(imagename))
    #
    # Loop over spectral windows
    #
    for spw in my_line_spws.split(','):
        #
        # Get restfreq
        #
        spw_ind = config.get("Spectral Windows","Line").split(',').index(spw)
        restfreq = config.get("Clean","restfreqs").split(',')[spw_ind]
        #
        # Dirty image so we can copy mask
        #
        imagename='{0}.spw{1}.channel.clean'.format(field,spw)
        if uvtaper:
            imagename = imagename + '.uvtaper'
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
                   uvtaper=uvtaper,outertaper=clean_params['outertaper'])
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
                   uvtaper=uvtaper,outertaper=clean_params['outertaper'])
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
                        velocity=True,overwrite=True,history=False)
        casa.exportfits(imagename='{0}.image'.format(imagename),
                        fitsimage='{0}.image.fits'.format(imagename),
                        velocity=True,overwrite=True,history=False)
        casa.exportfits(imagename='{0}.residual'.format(imagename),
                        fitsimage='{0}.residual.fits'.format(imagename),
                        velocity=True,overwrite=True,history=False)
        logger.info("Done.")


def contplot(field,clean_params={}):
    """
    Generate pdf document of continuum diagnostic plots, both with
    and without uv-tapering
    1. Dirty image
    2. Clean image
    3. Residual image
    4. alpha image
    5. alpha error image
    6. Beta image

    Inputs:
      field  = field we're plotting
      clean_params = dictionary of clean parameters

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    logger.info("Generating continuum images...")
    #
    # Setup parameters for zooming
    #
    center_x = int(clean_params['imsize'][0]/2)
    center_y = int(clean_params['imsize'][1]/2)
    # corners of plots for zooming
    x_min = center_x-int(center_x/3)
    x_max = center_x+int(center_x/3)
    y_min = center_y-int(center_y/3)
    y_max = center_y+int(center_y/3)
    #
    # Loop over all plot filenames
    #
    fitsfiles = ['{0}.cont.mfs.dirty.image.fits'.format(field),
                 '{0}.cont.mfs.clean.image.fits'.format(field),
                 '{0}.cont.mfs.clean.residual.fits'.format(field),
                 '{0}.cont.mfs.clean.alpha.fits'.format(field),
                 '{0}.cont.mfs.clean.alpha.error.fits'.format(field),
                 '{0}.cont.mfs.clean.beta.fits'.format(field),
                 '{0}.cont.mfs.dirty.uvtaper.image.fits'.format(field),
                 '{0}.cont.mfs.clean.uvtaper.image.fits'.format(field),
                 '{0}.cont.mfs.clean.uvtaper.residual.fits'.format(field),
                 '{0}.cont.mfs.clean.uvtaper.alpha.fits'.format(field),
                 '{0}.cont.mfs.clean.uvtaper.alpha.error.fits'.format(field),
                 '{0}.cont.mfs.clean.uvtaper.beta.fits'.format(field)]
    titles = ['{0} - Dirty'.format(field),
              '{0} - Clean'.format(field),
              '{0} - Residual'.format(field),
              '{0} - Spectral Index'.format(field),
              '{0} - Spectral Index Error'.format(field),
              '{0} - Beta'.format(field),
              '{0} - Dirty - UV taper'.format(field),
              '{0} - Clean - UV taper'.format(field),
              '{0} - Residual - UV taper'.format(field),
              '{0} - Spectral Index - UV taper'.format(field),
              '{0} - Spectral Index Error - UV taper'.format(field),
              '{0} - Beta - UV taper'.format(field)]
    labels = ['Flux Density (Jy/beam)',
              'Flux Density (Jy/beam)',
              'Flux Density (Jy/beam)',
              'Spectral Index',
              'Spectral Index Error',
              'Beta',
              'Flux Density (Jy/beam)',
              'Flux Density (Jy/beam)',
              'Flux Density (Jy/beam)',
              'Spectral Index',
              'Spectral Index Error',
              'Beta']
    vlims = [(None,None),
             (None,None),
             (None,None),
             (-2,1),
             (0,1),
             (-1,1),
             (None,None),
             (None,None),
             (None,None),
             (-2,1),
             (0,1),
             (-1,1)]
    for fitsfile,title,label,vlim in zip(fitsfiles,titles,labels,vlims):
        #
        # Open fits file, generate WCS
        #
        hdu = fits.open(fitsfile)[0]
        wcs = WCS(hdu.header)
        #
        # Generate figure
        #
        plt.ioff()
        fig = plt.figure()
        ax = plt.subplot(projection=wcs.sub(['celestial']))
        ax.set_title(title)
        cax = ax.imshow(hdu.data[0,0],
                        origin='lower',interpolation='none',
                        cmap='binary',vmin=vlim[0],vmax=vlim[1])
        # ax.grid(True,color='black',ls='solid')
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
        ax.coords[0].set_major_formatter('hh:mm:ss')
        ax.set_xlabel('RA (J2000)')
        ax.set_ylabel('Declination (J2000)')
        #
        # Add Galactic l,b locus
        #
        # overlay = ax.get_coords_overlay('galactic')
        # overlay[0].set_axislabel('Galactic Longitude (deg)')
        # overlay[1].set_axislabel('Galactic Latitude (deg)')
        # overlay[0].set_major_formatter('d.dd')
        # overlay[1].set_major_formatter('d.dd')
        # overlay.grid(True,color='black',ls='dotted')
        #
        # Plot beam, if it is defined
        #
        if 'BMAJ' in hdu.header.keys():
            cell = float(clean_params['cell'].replace('arcsec',''))
            beam_maj = hdu.header['BMAJ']*3600./cell
            beam_min = hdu.header['BMIN']*3600./cell
            beam_pa = hdu.header['BPA']
            ellipse = Ellipse((center_x-int(center_x/4),
                            center_y-int(center_y/4)),
                            beam_min,beam_maj,angle=beam_pa,
                            fill=True,zorder=10,hatch='///',
                            edgecolor='black',facecolor='white')
            ax.add_patch(ellipse)
        #
        # Plot colorbar
        #
        cbar = fig.colorbar(cax,fraction=0.046,pad=0.04)
        cbar.set_label(label)
        #
        # Re-scale to fit, then save
        #
        fig.savefig(fitsfile.replace('.fits','.pdf'),
                    bbox_inches='tight')
        plt.close(fig)
        plt.ion()
    #
    # Generate PDF of plots
    #
    # need to fix filenames so LaTeX doesn't complain
    outplots = ['{'+fn.replace('.fits','')+'}.pdf' for fn in fitsfiles]
    logger.info("Generating PDF...")
    with open('{0}.contplots.tex'.format(field),'w') as f:
        f.write(r"\documentclass{article}"+"\n")
        f.write(r"\usepackage{graphicx}"+"\n")
        f.write(r"\usepackage[margin=0.1cm]{geometry}"+"\n")
        f.write(r"\begin{document}"+"\n")
        for i in range(0,len(outplots),6):
            f.write(r"\begin{figure}"+"\n")
            f.write(r"\centering"+"\n")
            f.write(r"\includegraphics[width=0.45\textwidth]{"+outplots[i]+"}\n")
            if len(outplots) > i+3: f.write(r"\includegraphics[width=0.45\textwidth]{"+outplots[i+3]+"}\n")
            f.write(r"\includegraphics[width=0.45\textwidth]{"+outplots[i+1]+"}\n")
            if len(outplots) > i+4: f.write(r"\includegraphics[width=0.45\textwidth]{"+outplots[i+4]+"}\n")
            f.write(r"\includegraphics[width=0.45\textwidth]{"+outplots[i+2]+"}\n")
            if len(outplots) > i+5: f.write(r"\includegraphics[width=0.45\textwidth]{"+outplots[i+5]+"}\n")
            f.write(r"\end{figure}"+"\n")
            f.write(r"\clearpage"+"\n")
        f.write(r"\end{document}")
    os.system('pdflatex -interaction=batchmode {0}.contplots.tex'.format(field))
    logger.info("Done.")

def lineplot(field,line_spws='',clean_params={}):
    """
    Generate pdf document of spectral line diagnostic plots:
    For each spectral window
    1. Dirty image, center channel
    2. Clean image, center channel
    3. Residual image, center channel
    4. Spectrum, center pixel

    Inputs:
      field  = field we're plotting
      line_spws = line spectral windows to plot
      clean_params = dictionary of clean parameters

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    logger.info("Generating line images...")
    #
    # Setup parameters for zooming
    #
    center_x = int(clean_params['imsize'][0]/2)
    center_y = int(clean_params['imsize'][1]/2)
    # corners of plots for zooming
    x_min = center_x-int(center_x/3)
    x_max = center_x+int(center_x/3)
    y_min = center_y-int(center_y/3)
    y_max = center_y+int(center_y/3)
    #
    # Loop over spectral windows
    #
    goodplots = []
    for spw in line_spws.split(','):
        # check that this spectral window exists
        if not os.path.exists('{0}.spw{1}.channel.clean.uvtaper.image.fits'.format(field,spw)):
            continue
        #
        # Loop over all plot filenames
        #
        fitsfiles = ['{0}.spw{1}.channel.dirty.uvtaper.image.fits'.format(field,spw),
                     '{0}.spw{1}.channel.clean.uvtaper.image.fits'.format(field,spw),
                     '{0}.spw{1}.channel.clean.uvtaper.residual.fits'.format(field,spw)]
        titles = ['{0} - spw: {1} - Dirty - UV taper'.format(field,spw),
                  '{0} - spw: {1} - Clean - UV taper'.format(field,spw),
                  '{0} - spw: {1} - Residual - UV taper'.format(field,spw)]
        labels = ['Flux Density (Jy/beam)',
                  'Flux Density (Jy/beam)',
                  'Flux Density (Jy/beam)']
        vlims = [(None,None),
                 (None,None),
                 (None,None)]
        for fitsfile,title,label,vlim in zip(fitsfiles,titles,labels,vlims):
            #
            # Open fits file, generate WCS
            #
            hdulist = fits.open(fitsfile)
            hdu = hdulist[0]
            wcs = WCS(hdu.header)
            #
            # Generate figure
            #
            plt.ioff()
            fig = plt.figure()
            ax = plt.subplot(projection=wcs.sub(['celestial']))
            ax.set_title(title)
            center_chan = hdu.data.shape[1]/2
            img = ax.imshow(hdu.data[0,center_chan],
                            origin='lower',interpolation='none',
                            cmap='binary',vmin=vlim[0],vmax=vlim[1])
            # ax.grid(True,color='black',ls='solid')
            ax.set_xlim(x_min,x_max)
            ax.set_ylim(y_min,y_max)
            ax.coords[0].set_major_formatter('hh:mm:ss')
            ax.set_xlabel('RA (J2000)')
            ax.set_ylabel('Declination (J2000)')
            #ax.set_position([0.2,0.15,0.6,0.6])
            #
            # Add Galactic l,b locus
            #
            # overlay = ax.get_coords_overlay('galactic')
            # overlay[0].set_axislabel('Galactic Longitude (deg)')
            # overlay[1].set_axislabel('Galactic Latitude (deg)')
            # overlay[0].set_major_formatter('d.dd')
            # overlay[1].set_major_formatter('d.dd')
            # overlay.grid(True,color='black',ls='dotted')
            #
            # Plot beam, if it is defined
            #
            if 'BMAJ' in hdu.header.keys():
                cell = float(clean_params['cell'].replace('arcsec',''))
                beam_maj = hdu.header['BMAJ']*3600./cell
                beam_min = hdu.header['BMIN']*3600./cell
                beam_pa = hdu.header['BPA']
                ellipse = Ellipse((center_x-int(center_x/4),
                                center_y-int(center_y/4)),
                                beam_min,beam_maj,angle=beam_pa,
                                fill=True,zorder=10,hatch='///',
                                edgecolor='black',facecolor='white')
                ax.add_patch(ellipse)
            elif len(hdulist) > 1:
                hdu = hdulist[1]
                cell = float(clean_params['cell'].replace('arcsec',''))
                beam_maj = hdu.data['BMAJ'][center_chan]/cell
                beam_min = hdu.data['BMIN'][center_chan]/cell
                beam_pa = hdu.data['BPA'][center_chan]
                ellipse = Ellipse((center_x-int(center_x/4),
                                  center_y-int(center_y/4)),
                                  beam_min,beam_maj,angle=beam_pa,
                                  fill=True,zorder=10,hatch='///',
                                  edgecolor='black',facecolor='white')
                ax.add_patch(ellipse)
            #
            # Plot colorbar
            #
            cbar = fig.colorbar(img,fraction=0.046,pad=0.04)
            cbar.set_label(label)
            #
            # Re-scale to fit, then save
            #
            fig.savefig(fitsfile.replace('.fits','.pdf'),
                        bbox_inches='tight')
            plt.close(fig)
            plt.ion()
            goodplots.append(fitsfile.replace('.fits','.pdf'))
        #
        # Generate spectrum
        #
        fitsfile = '{0}.spw{1}.channel.clean.uvtaper.image.fits'.format(field,spw)
        hdu = fits.open(fitsfile)[0]
        spec = hdu.data[0,:,center_x,center_y]
        velo = (np.arange(len(spec))*hdu.header['CDELT3'] + hdu.header['CRVAL3'])/1000.
        #
        # Generate figure
        #
        plt.ioff()
        fig = plt.figure()
        ax = plt.subplot()
        ax.plot(velo,spec,'k-')
        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('Flux Density (Jy/beam)')
        ax.set_xlim(np.min(velo),np.max(velo))
        ybuff = 0.1*(np.max(spec)-np.min(spec))
        ax.set_ylim(np.min(spec)-ybuff,np.max(spec)+ybuff)
        ax.set_title('{0} - spw {1} - Center'.format(field,spw))
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(fitsfile.replace('.fits','.spec.pdf'),
                    bbox_inches='tight')
        plt.close(fig)
        plt.ion()
        goodplots.append(fitsfile.replace('.fits','.spec.pdf'))
    #
    # Generate PDF of plots
    #
    # need to fix filenames so LaTeX doesn't complain
    goodplots = ['{'+fn.replace('.pdf','')+'}.pdf' for fn in goodplots]
    logger.info("Generating PDF...")
    with open('{0}.lineplots.tex'.format(field),'w') as f:
        f.write(r"\documentclass{article}"+"\n")
        f.write(r"\usepackage{graphicx}"+"\n")
        f.write(r"\usepackage[margin=0.1cm]{geometry}"+"\n")
        f.write(r"\begin{document}"+"\n")
        for i in range(0,len(goodplots),4):
            f.write(r"\begin{figure}"+"\n")
            f.write(r"\centering"+"\n")
            f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i]+"}\n")
            if len(goodplots) > i+1: f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i+1]+"}\n")
            if len(goodplots) > i+2: f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i+2]+"}\n")
            if len(goodplots) > i+3: f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i+3]+"}\n")
            f.write(r"\end{figure}"+"\n")
            f.write(r"\clearpage"+"\n")
        f.write(r"\end{document}")
    os.system('pdflatex -interaction=batchmode {0}.lineplots.tex'.format(field))
    logger.info("Done.")

def main(field,vis='',spws='',config_file='',uvtaper=False):
    """
    Generate continuum and line images in various ways

    Inputs:
      field       = field name to clean
      vis         = measurement set containing all data for field
      spws        = comma-separated list of line spws to clean
                    if empty, clean all line spws
      config_file = filename of the configuration file for this project
      uvtaper     = if True, apply UV tapering

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
    my_cont_spws,all_line_spws,clean_params = setup(vis=vis,config=config)
    if spws == '':
        my_line_spws = all_line_spws
    else:
        my_line_spws = spws
    logger.info("Considering line spws: {0}".format(my_line_spws))
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
                           clean_params=clean_params,
                           uvtaper=uvtaper)
        elif answer == '1':
            channel_dirty_cont(field=field,vis=vis,
                               my_cont_spws=my_cont_spws,
                               clean_params=clean_params,
                               uvtaper=uvtaper)
        elif answer == '2':
            mfs_clean_cont(field=field,vis=vis,
                           my_cont_spws=my_cont_spws,
                           clean_params=clean_params,
                           uvtaper=uvtaper)
        elif answer == '3':
            channel_clean_cont(field=field,vis=vis,
                               my_cont_spws=my_cont_spws,
                               clean_params=clean_params,
                               uvtaper=uvtaper)
        elif answer == '4':
            mfs_dirty_cont_spws(field=field,vis=vis,
                                my_cont_spws=my_cont_spws,
                                clean_params=clean_params,
                                uvtaper=uvtaper)
        elif answer == '5':
            mfs_clean_cont_spws(field=field,vis=vis,
                                my_cont_spws=my_cont_spws,
                                clean_params=clean_params,
                                uvtaper=uvtaper)
        elif answer == '6':
            print("Which spw do you want to clean?")
            spw = raw_input('> ')
            mfs_clean_cont_spw(field=field,vis=vis,spw=spw,
                               clean_params=clean_params,
                               uvtaper=uvtaper)
        elif answer == '7':
            print("Which spw did you clean manually?")
            manual_spw = raw_input('> ')
            print("Enter clean threshold (i.e. 1.1mJy)")
            cont_threshold = raw_input('> ')
            auto_mfs_clean_cont_spws(field=field,vis=vis,
                                     my_cont_spws=my_cont_spws,
                                     manual_spw=manual_spw,
                                     threshold=cont_threshold,
                                     clean_params=clean_params,
                                     uvtaper=uvtaper)
        elif answer == '8':
            channel_dirty_line_spws(field=field,vis=vis,
                                    my_line_spws=my_line_spws,
                                    clean_params=clean_params,
                                    config=config,
                                    uvtaper=uvtaper)
        elif answer == '9':
            channel_clean_line_spws(field=field,vis=vis,
                                    my_line_spws=my_line_spws,
                                    clean_params=clean_params,
                                    config=config,
                                    uvtaper=uvtaper)
        elif answer == '10':
            print("Which spw do you want to clean?")
            spw = raw_input('> ')
            channel_clean_line_spw(field=field,vis=vis,spw=spw,
                                   clean_params=clean_params,
                                   config=config,
                                   uvtaper=uvtaper)
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
                                         config=config,
                                         uvtaper=uvtaper)
        elif answer == '12':
            contplot(field,clean_params=clean_params)
        elif answer == '13':
            lineplot(field,line_spws=all_line_spws,
                     clean_params=clean_params)
        elif answer.lower() == 'q' or answer.lower() == 'quit':
            break
        else:
            print("Input not recognized.")
