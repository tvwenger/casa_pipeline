"""
imaging_tclean.py
CASA Data Reduction Pipeline - Imaging Script
This verison uses tclean to re-bin and re-grid the data and to
generate automatic clean masks.
Trey V. Wenger Jun 2016 - V1.0
"""

import __main__ as casa
import os
import gc
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

def setup(vis='',config=None,uvtaper=False):
    """
    Perform setup tasks: find line and continuum spectral windows
                         get clean parameters

    Inputs:
      vis     = measurement set
      config  = ConfigParser object for this project

    Returns:
      (my_cont_spws,my_line_spws,cp)
      my_cont_spws    = comma-separated string of continuum spws
      my_line_spws    = comma-separated string of line spws
      cp    = dictionary of clean parameters read from config file
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
    cp = {}
    cp["lineids"] = config.get("Clean","lineids").split(',')
    cp["restfreqs"] = config.get("Clean","restfreqs").split(',')
    # clean info
    cp["imsize"] = [int(foo) for foo in config.get("Clean","imsize").split(',')]
    cp["pblimit"] = config.getfloat("Clean","pblimit")
    cp["cell"] = "{0}arcsec".format(config.getfloat("Clean","cell"))
    cp["weighting"] = config.get("Clean","weighting")
    cp["robust"] = config.getfloat("Clean","robust")
    cp["scales"] = [int(foo) for foo in config.get("Clean","scales").split(',') if foo != '']
    cp["gain"] = config.getfloat("Clean","gain")
    cp["cyclefactor"] = config.getfloat("Clean","cyclefactor")
    cp["lightniter"] = config.getint("Clean","lightniter")
    cp["maxniter"] = config.getint("Clean","maxniter")
    cp["nrms"] = config.getfloat("Clean","nrms")
    cp["contpbchan"] = config.get("Clean","contpbchan")
    cp["nterms"] = config.getint("Clean","nterms")
    if uvtaper:
        cp["outertaper"] = ["{0}arcsec".format(config.getfloat("Clean","outertaper"))]
    else:
        cp["outertaper"] = []
    # automasking
    cp["contpbmask"] = config.getfloat("Clean","contpbmask")
    cp["contsidelobethreshold"] = config.getfloat("Clean","contsidelobethreshold")
    cp["contnoisethreshold"] = config.getfloat("Clean","contnoisethreshold")
    cp["contlownoisethreshold"] = config.getfloat("Clean","contlownoisethreshold")
    cp["contnegativethreshold"] = config.getfloat("Clean","contnegativethreshold")
    cp["contsmoothfactor"] = config.getfloat("Clean","contsmoothfactor")
    cp["contminbeamfrac"] = config.getfloat("Clean","contminbeamfrac")
    cp["contcutthreshold"] = config.getfloat("Clean","contcutthreshold")
    cp["contgrowiterations"] = config.getint("Clean","contgrowiterations")
    cp["linepbmask"] = config.getfloat("Clean","linepbmask")
    cp["linesidelobethreshold"] = config.getfloat("Clean","linesidelobethreshold")
    cp["linenoisethreshold"] = config.getfloat("Clean","linenoisethreshold")
    cp["linelownoisethreshold"] = config.getfloat("Clean","linelownoisethreshold")
    cp["linenegativethreshold"] = config.getfloat("Clean","linenegativethreshold")
    cp["linesmoothfactor"] = config.getfloat("Clean","linesmoothfactor")
    cp["lineminbeamfrac"] = config.getfloat("Clean","lineminbeamfrac")
    cp["linecutthreshold"] = config.getfloat("Clean","linecutthreshold")
    cp["linegrowiterations"] = config.getint("Clean","linegrowiterations")
    # re-grid info
    velstart = config.getfloat("Clean","velstart")
    cp["velstart"] = "{0}km/s".format(velstart)
    chanwidth = config.getfloat("Clean","chanwidth")
    cp["chanwidth"] = "{0}km/s".format(chanwidth)
    cp["nchan"] = config.getint("Clean","nchan")
    cp["chanbuffer"] = config.getint("Clean","chanbuffer")
    cp["cvelstart"] = "{0}km/s".format(velstart-(cp["chanbuffer"]*chanwidth))
    cp["cvelnchan"] = cp["nchan"]+2*cp["chanbuffer"]
    cp["lineoutframe"] = config.get("Clean","lineoutframe")
    cp["veltype"] = config.get("Clean","veltype")
    cp["interpolation"] = config.get("Clean","interpolation")
    cp["contoutframe"] = config.get("Clean","contoutframe")
    return (my_cont_spws,my_line_spws,cp)

def regrid_velocity(vis='',spws='',config=None,cp={}):
    """
    Re-grid velocity axis of each spectral window

    Inputs:
      vis     = measurement set
      spws    = comma-separated list of spectral windows to regrid
      config  = ConfigParser object for this project
      cp      = dictionary of clean parameters

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
        regrid_vis = vis.replace('.ms','')+'.spw{0}.cvel.ms'.format(spw)
        if os.path.isdir(regrid_vis):
            logger.info("Found {0}".format(regrid_vis))
            continue
        logger.info("Regridding velocity axis of spw {0}".format(spw))
        spw_ind = config.get("Spectral Windows","Line").split(',').index(spw)
        restfreq = config.get("Clean","restfreqs").split(',')[spw_ind]
        casa.cvel2(vis=vis,outputvis=regrid_vis,spw=spw,restfreq=restfreq,mode='velocity',
                   start=cp['cvelstart'],width=cp['chanwidth'],
                   nchan=cp['cvelnchan'],outframe=cp['lineoutframe'],
                   veltype=cp['veltype'],interpolation=cp['interpolation'])
        # casa.mstransform(vis=vis,outputvis=regrid_vis,spw=spw,
        #                  regridms=True,datacolumn='data',
        #                  restfreq=restfreq,mode='velocity',
        #            start=cp['cvelstart'],width=cp['chanwidth'],
        #            nchan=cp['cvelnchan'],outframe=cp['lineoutframe'],
        #            veltype=cp['veltype'],interpolation=cp['interpolation'])
        logger.info("Done.")

def mfs_dirty_cont(field='',vis='',my_cont_spws='',cp={},
                   uvtaper=False):
    """
    Dirty image continuum spws using multi-frequency synthesis

    Inputs:
      field        = field to be cleaned
      vis          = measurement set
      my_cont_spws = comma-separated string of continuum spws
      cp = dictionary of clean parameters
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
    casa.tclean(vis=vis,imagename=imagename,field=field,spw=my_cont_spws,
                specmode='mfs',threshold='0mJy',niter=0,
                nterms=cp['nterms'],deconvolver='mtmfs',scales=cp['scales'],
                gain=cp['gain'],cyclefactor=cp['cyclefactor'],
                imsize=cp['imsize'],pblimit=-1.0,cell=cp['cell'],
                weighting=cp['weighting'],robust=cp['robust'],
                uvtaper=cp['outertaper'],pbcor=False)
    logger.info("Done.")
    #
    # Primary beam correction using PB of center channel
    #
    logger.info("Performing primary beam correction...")
    spwlist = [int(spw) for spw in my_cont_spws.split(',')]
    weightlist = [1.0 for spw in spwlist]
    chanlist = ','.join([cp['contpbchan'] for foo in my_cont_spws.split(',')])
    casa.widebandpbcor(vis=vis,imagename=imagename,
                       nterms=cp['nterms'],pbmin=cp['pblimit'],threshold='0.1mJy',
                       spwlist=spwlist,weightlist=weightlist,chanlist=chanlist)
    logger.info("Done.")
    #
    # Export to fits
    #
    logger.info("Exporting fits file...")
    casa.exportfits(imagename='{0}.image.tt0'.format(imagename),
                    fitsimage='{0}.image.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.residual.tt0'.format(imagename),
                    fitsimage='{0}.residual.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.pbcor.image.tt0'.format(imagename),
                    fitsimage='{0}.pbcor.image.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.pbcor.image.alpha'.format(imagename),
                    fitsimage='{0}.pbcor.image.alpha.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.pbcor.image.alpha.error'.format(imagename),
                    fitsimage='{0}.pbcor.image.alpha.error.fits'.format(imagename),
                    overwrite=True,history=False)
    logger.info("Done.")

def mfs_clean_cont(field='',vis='',my_cont_spws='',cp={},
                   uvtaper=False):
    """
    Clean continuum spws using multi-frequency synthesis

    Inputs:
      field        = field to be cleaned
      vis          = measurement set
      my_cont_spws = comma-separated string of continuum spws
      cp = dictionary of clean parameters
      uvtaper      = if True, apply UV tapering

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Lightly clean continuum to get RMS threshold
    #
    imagename='{0}.cont.mfs.lightclean'.format(field)
    if uvtaper:
        imagename = imagename + '.uvtaper'
    logger.info("Lightly cleaning continuum image (MFS)...")
    casa.tclean(vis=vis,imagename=imagename,field=field,spw=my_cont_spws,
                specmode='mfs',threshold='0mJy',niter=cp["lightniter"],
                usemask='auto-multithresh',pbmask=cp['contpbmask'],
                sidelobethreshold=cp['contsidelobethreshold'],
                noisethreshold=cp['contnoisethreshold'],
                lownoisethreshold=cp['contlownoisethreshold'],
                negativethreshold=cp['contnegativethreshold'],
                smoothfactor=cp['contsmoothfactor'],minbeamfrac=cp['contminbeamfrac'],
                cutthreshold=cp['contcutthreshold'],growiterations=cp['contgrowiterations'],
                nterms=cp['nterms'],deconvolver='mtmfs',
                scales=cp['scales'],
                gain=cp['gain'],cyclefactor=cp['cyclefactor'],
                imsize=cp['imsize'],pblimit=-1.0,cell=cp['cell'],
                weighting=cp['weighting'],robust=cp['robust'],
                uvtaper=cp['outertaper'],pbcor=False)
    logger.info("Done.")
    #
    # Get RMS of residuals outside of clean mask
    #
    dat = casa.imstat(imagename='{0}.residual.tt0'.format(imagename),
                      axes=[0,1],mask="'{0}.mask' == 0".format(imagename))
    logger.info("Max un-masked RMS: {0:.2f} mJy/beam".format(1000.*np.max(dat['rms'])))
    logger.info("Mean un-masked RMS: {0:.2f} mJy/beam".format(1000.*np.mean(dat['rms'])))
    logger.info("Median un-masked RMS: {0:.2f} mJy/beam".format(1000.*np.median(dat['rms'])))
    logger.info("Max un-masked MAD*1.4826: {0:.2f} mJy/beam".format(1000.*1.4826*np.max(dat['medabsdevmed'])))
    logger.info("Mean un-masked MAD*1.4826: {0:.2f} mJy/beam".format(1000.*1.4826*np.mean(dat['medabsdevmed'])))
    logger.info("Median un-masked MAD*1.4826: {0:.2f} mJy/beam".format(1000.*1.4826*np.median(dat['medabsdevmed'])))
    logger.info("Using median MADS*1.4826 times {0} (user-defined) as threshold".format(cp['nrms']))
    threshold = '{0:.2f}mJy'.format(cp["nrms"]*1000.*1.4826*np.median(dat['medabsdevmed']))
    #
    # Clean to threshold
    #
    imagename='{0}.cont.mfs.clean'.format(field)
    if uvtaper:
        imagename = imagename + '.uvtaper'
    logger.info("Cleaning continuum image (MFS) to threshold: {0}...".format(threshold))
    casa.tclean(vis=vis,imagename=imagename,field=field,spw=my_cont_spws,
                specmode='mfs',threshold=threshold,niter=cp['maxniter'],
                usemask='auto-multithresh',pbmask=cp['contpbmask'],
                sidelobethreshold=cp['contsidelobethreshold'],
                noisethreshold=cp['contnoisethreshold'],
                lownoisethreshold=cp['contlownoisethreshold'],
                negativethreshold=cp['contnegativethreshold'],
                smoothfactor=cp['contsmoothfactor'],minbeamfrac=cp['contminbeamfrac'],
                cutthreshold=cp['contcutthreshold'],growiterations=cp['contgrowiterations'],
                nterms=cp['nterms'],deconvolver='mtmfs',
                scales=cp['scales'],
                gain=cp['gain'],cyclefactor=cp['cyclefactor'],
                imsize=cp['imsize'],pblimit=-1.0,cell=cp['cell'],
                weighting=cp['weighting'],robust=cp['robust'],
                uvtaper=cp['outertaper'],pbcor=False)
    logger.info("Done.")
    #
    # Primary beam correction using PB of center channel
    #
    logger.info("Performing primary beam correction...")
    spwlist = [int(spw) for spw in my_cont_spws.split(',')]
    weightlist = [1.0 for spw in spwlist]
    chanlist = ','.join([cp['contpbchan'] for foo in my_cont_spws.split(',')])
    casa.widebandpbcor(vis=vis,imagename=imagename,
                       nterms=cp['nterms'],pbmin=cp['pblimit'],threshold='0.1mJy',
                       spwlist=spwlist,weightlist=weightlist,chanlist=chanlist)
    logger.info("Done.")
    #
    # Export to fits
    #
    logger.info("Exporting fits file...")
    casa.exportfits(imagename='{0}.image.tt0'.format(imagename),
                    fitsimage='{0}.image.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.mask'.format(imagename),
                    fitsimage='{0}.mask.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.residual.tt0'.format(imagename),
                    fitsimage='{0}.residual.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.pbcor.image.tt0'.format(imagename),
                    fitsimage='{0}.pbcor.image.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.pbcor.image.alpha'.format(imagename),
                    fitsimage='{0}.pbcor.image.alpha.fits'.format(imagename),
                    overwrite=True,history=False)
    casa.exportfits(imagename='{0}.pbcor.image.alpha.error'.format(imagename),
                    fitsimage='{0}.pbcor.image.alpha.error.fits'.format(imagename),
                    overwrite=True,history=False)
    logger.info("Done.")

def mfs_dirty_spws(field='',vis='',my_spws='',
                        cp={},uvtaper=False):
    """
    Dirty image each suuplied spw (MFS)

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      my_spws      = comma-separated string of spws to image
      cp = dictionary of clean parameters
      uvtaper      = if True, apply UV tapering

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    for spw in my_spws.split(','):
        #
        # Dirty image
        #
        imagename='{0}.spw{1}.mfs.dirty'.format(field,spw)
        if uvtaper:
            imagename = imagename + '.uvtaper'
        logger.info("Generating dirty image of spw {0} (MFS)...".format(spw))
        casa.tclean(vis=vis,imagename=imagename,field=field,spw=spw,
                    specmode='mfs',threshold='0mJy',niter=0,
                    deconvolver='multiscale',scales=cp['scales'],
                    gain=cp['gain'],cyclefactor=cp['cyclefactor'],
                    imsize=cp['imsize'],pblimit=-1.0,cell=cp['cell'],
                    weighting=cp['weighting'],robust=cp['robust'],
                    uvtaper=cp['outertaper'],pbcor=False)
        logger.info("Done.")
        #
        # Generate primary beam image
        #
        pbimagename='{0}.spw{1}.mfs.pb'.format(field,spw)
        if uvtaper:
            pbimagename = pbimagename + '.uvtaper'
        logger.info("Generating primary beam image of spw {0} (MFS)...".format(spw))
        casa.tclean(vis=vis,imagename=pbimagename,field=field,spw=spw,
                    specmode='mfs',threshold='0mJy',niter=0,
                    deconvolver='multiscale',scales=cp['scales'],
                    gain=cp['gain'],cyclefactor=cp['cyclefactor'],
                    imsize=cp['imsize'],pblimit=cp['pblimit'],cell=cp['cell'],
                    weighting=cp['weighting'],robust=cp['robust'],
                    uvtaper=cp['outertaper'],pbcor=False)
        logger.info("Done.")
        #
        # Primary beam correction
        #
        pbimagename='{0}.spw{1}.mfs.pb'.format(field,spw)
        if uvtaper:
            pbimagename = pbimagename + '.uvtaper'
        logger.info("Performing primary beam correction...")
        casa.impbcor(imagename='{0}.image'.format(imagename),
                     pbimage='{0}.pb'.format(pbimagename),
                     outfile='{0}.pbcor.image'.format(imagename))
        logger.info("Done.")
        #
        # Export to fits
        #
        logger.info("Exporting fits file...")
        casa.exportfits(imagename='{0}.image'.format(imagename),
                        fitsimage='{0}.image.fits'.format(imagename),
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.residual'.format(imagename),
                        fitsimage='{0}.residual.fits'.format(imagename),
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.pbcor.image'.format(imagename),
                        fitsimage='{0}.pbcor.image.fits'.format(imagename),
                        overwrite=True,history=False)
        logger.info("Done.")

def mfs_clean_spws(field='',vis='',my_spws='',
                   cp={},uvtaper=False):
    """
    Clean each supplied spw (MFS)

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      my_spws = comma-separated string of spws to image
      cp = dictionary of clean parameters
      uvtaper      = if True, apply UV tapering

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    for spw in my_spws.split(','):
        #
        # Lightly clean to get threshold
        #
        imagename='{0}.spw{1}.mfs.lightclean'.format(field,spw)
        if uvtaper:
            imagename = imagename + '.uvtaper'
        logger.info("Lightly cleaning spw {0} (MFS)...".format(spw))
        casa.tclean(vis=vis,imagename=imagename,field=field,spw=spw,
                    specmode='mfs',threshold='0mJy',niter=cp["lightniter"],
                    usemask='auto-multithresh',pbmask=cp['linepbmask'],
                    sidelobethreshold=cp['linesidelobethreshold'],
                    noisethreshold=cp['linenoisethreshold'],
                    lownoisethreshold=cp['linelownoisethreshold'],
                    negativethreshold=cp['linenegativethreshold'],
                    smoothfactor=cp['linesmoothfactor'],minbeamfrac=cp['lineminbeamfrac'],
                    cutthreshold=cp['linecutthreshold'],growiterations=cp['linegrowiterations'],
                    deconvolver='multiscale',scales=cp['scales'],
                    gain=cp['gain'],cyclefactor=cp['cyclefactor'],
                    imsize=cp['imsize'],pblimit=-1.0,cell=cp['cell'],
                    weighting=cp['weighting'],robust=cp['robust'],
                    uvtaper=cp['outertaper'],pbcor=False)
        logger.info("Done.")
        #
        # Get RMS of residuals
        #
        dat = casa.imstat(imagename='{0}.residual'.format(imagename),
                          axes=[0,1],mask="'{0}.mask' == 0".format(imagename))
        logger.info("Max un-masked RMS: {0:.2f} mJy/beam".format(1000.*np.max(dat['rms'])))
        logger.info("Mean un-masked RMS: {0:.2f} mJy/beam".format(1000.*np.mean(dat['rms'])))
        logger.info("Median un-masked RMS: {0:.2f} mJy/beam".format(1000.*np.median(dat['rms'])))
        logger.info("Max un-masked MAD*1.4826: {0:.2f} mJy/beam".format(1000.*1.4826*np.max(dat['medabsdevmed'])))
        logger.info("Mean un-masked MAD*1.4826: {0:.2f} mJy/beam".format(1000.*1.4826*np.mean(dat['medabsdevmed'])))
        logger.info("Median un-masked MAD*1.4826: {0:.2f} mJy/beam".format(1000.*1.4826*np.median(dat['medabsdevmed'])))
        logger.info("Using median MADS*1.4826 times {0} (user-defined) as threshold".format(cp['nrms']))
        threshold = '{0:.2f}mJy'.format(cp["nrms"]*1000.*1.4826*np.median(dat['medabsdevmed']))
        #
        # Clean to threshold
        #
        imagename='{0}.spw{1}.mfs.clean'.format(field,spw)
        if uvtaper:
            imagename = imagename + '.uvtaper'
        logger.info("Cleaning spw {0} (MFS) to threshold: {1}...".format(spw,threshold))
        casa.tclean(vis=vis,imagename=imagename,field=field,spw=spw,
                    specmode='mfs',threshold=threshold,niter=cp["maxniter"],
                    usemask='auto-multithresh',pbmask=cp['linepbmask'],
                    sidelobethreshold=cp['linesidelobethreshold'],
                    noisethreshold=cp['linenoisethreshold'],
                    lownoisethreshold=cp['linelownoisethreshold'],
                    negativethreshold=cp['linenegativethreshold'],
                    smoothfactor=cp['linesmoothfactor'],minbeamfrac=cp['lineminbeamfrac'],
                    cutthreshold=cp['linecutthreshold'],growiterations=cp['linegrowiterations'],
                    deconvolver='multiscale',scales=cp['scales'],
                    gain=cp['gain'],cyclefactor=cp['cyclefactor'],
                    imsize=cp['imsize'],pblimit=-1.0,cell=cp['cell'],
                    weighting=cp['weighting'],robust=cp['robust'],
                    uvtaper=cp['outertaper'],pbcor=False)
        logger.info("Done.")
        #
        # Primary beam correction
        #
        pbimagename='{0}.spw{1}.mfs.pb'.format(field,spw)
        if uvtaper:
            pbimagename = pbimagename + '.uvtaper'
        logger.info("Performing primary beam correction...")
        casa.impbcor(imagename='{0}.image'.format(imagename),
                     pbimage='{0}.pb'.format(pbimagename),
                     outfile='{0}.pbcor.image'.format(imagename))
        logger.info("Done.")
        #
        # Export to fits
        #
        logger.info("Exporting fits file...")
        casa.exportfits(imagename='{0}.image'.format(imagename),
                        fitsimage='{0}.image.fits'.format(imagename),
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.mask'.format(imagename),
                        fitsimage='{0}.mask.fits'.format(imagename),
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.residual'.format(imagename),
                        fitsimage='{0}.residual.fits'.format(imagename),
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.pbcor.image'.format(imagename),
                        fitsimage='{0}.pbcor.image.fits'.format(imagename),
                        overwrite=True,history=False)
        logger.info("Done.")

def channel_dirty_spws(field='',vis='',my_spws='',
                       cp={},spwtype='line',regrid=False,
                       config=None,uvtaper=False):
    """
    Dirty image all line spws

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      my_spws      = comma-separated string of spws to image
      cp           = dictionary of clean parameters
      spwtype      = 'line' or 'cont', determine which channel
                     parameters to use
      regrid       = if True, look for CVEL2 measurement set
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
    # Set channel parameters
    #
    if spwtype == 'cont':
        restfreqs = [None for spw in my_spws.split(',')]
        start = None
        width = None
        nchan = None
        outframe = cp['contoutframe']
        veltype = None
        interpolation = None
    elif spwtype == 'line':
        spw_inds = [config.get("Spectral Windows","Line").split(',').index(spw) for spw in my_spws.split(',')]
        restfreqs = [config.get("Clean","restfreqs").split(',')[spw_ind] for spw_ind in spw_inds]
        start = cp['velstart']
        width = cp['chanwidth']
        nchan = cp['nchan']
        outframe = cp['lineoutframe']
        veltype = cp['veltype']
        interpolation = cp['interpolation']
    else:
        logger.critical("Error: spwtype {0} not supported".format(spwtype))
        raise ValueError("Invalid spwtype")
    #
    # Loop over spws
    #
    for spw,restfreq in zip(my_spws.split(','),restfreqs):
        #
        # dirty image spw
        #
        if regrid:
            myvis = vis.replace('.ms','')+'.spw{0}.cvel.ms'.format(spw)
            myspw = '0'
        else:
            myvis = vis
            myspw = spw
        imagename='{0}.spw{1}.channel.dirty'.format(field,spw)
        if uvtaper:
            imagename = imagename + '.uvtaper'
        logger.info("Dirty imaging spw {0} (restfreq: {1})...".format(spw,restfreq))
        casa.tclean(vis=myvis,imagename=imagename,field=field,spw=myspw,
                    specmode='cube',threshold='0mJy',niter=0,
                    deconvolver='multiscale',scales=cp['scales'],
                    gain=cp['gain'],cyclefactor=cp['cyclefactor'],
                    imsize=cp['imsize'],pblimit=-1.0,cell=cp['cell'],
                    weighting=cp['weighting'],robust=cp['robust'],
                    restfreq=restfreq,start=start,width=width,nchan=nchan,
                    outframe=outframe,veltype=veltype,interpolation=interpolation,
                    uvtaper=cp['outertaper'],pbcor=False)
        logger.info("Done.")
        #
        # Generate primary beam image
        #
        pbimagename='{0}.spw{1}.channel.pb'.format(field,spw)
        if uvtaper:
            pbimagename = pbimagename + '.uvtaper'
        logger.info("Generating primary beam image of spw {0} (MFS)...".format(spw))
        casa.tclean(vis=myvis,imagename=pbimagename,field=field,spw=myspw,
                    specmode='cube',threshold='0mJy',niter=0,
                    deconvolver='multiscale',scales=cp['scales'],
                    gain=cp['gain'],cyclefactor=cp['cyclefactor'],
                    imsize=cp['imsize'],pblimit=cp['pblimit'],cell=cp['cell'],
                    weighting=cp['weighting'],robust=cp['robust'],
                    restfreq=restfreq,start=start,width=width,nchan=nchan,
                    outframe=outframe,veltype=veltype,interpolation=interpolation,
                    uvtaper=cp['outertaper'],pbcor=False)
        logger.info("Done.")
        #
        # Primary beam correction
        #
        pbimagename='{0}.spw{1}.channel.pb'.format(field,spw)
        if uvtaper:
            pbimagename = pbimagename + '.uvtaper'
        logger.info("Performing primary beam correction...")
        casa.impbcor(imagename='{0}.image'.format(imagename),
                     pbimage='{0}.pb'.format(pbimagename),
                     outfile='{0}.pbcor.image'.format(imagename))
        logger.info("Done.")
        #
        # Export to fits
        #
        logger.info("Exporting fits file...")
        casa.exportfits(imagename='{0}.image'.format(imagename),
                        fitsimage='{0}.image.fits'.format(imagename),
                        velocity=True,overwrite=True,history=False)
        casa.exportfits(imagename='{0}.residual'.format(imagename),
                        fitsimage='{0}.residual.fits'.format(imagename),
                        velocity=True,overwrite=True,history=False)
        casa.exportfits(imagename='{0}.pbcor.image'.format(imagename),
                        fitsimage='{0}.pbcor.image.fits'.format(imagename),
                        velocity=True,overwrite=True,history=False)
        logger.info("Done.")

def channel_clean_spws(field='',vis='',my_spws='',
                       cp={},spwtype='line',regrid=False,
                       config=None,uvtaper=False):
    """
    Clean all supplied spws by channel using clean mask from MFS
    images.

    Inputs:
      field        = field to be imaged
      vis          = measurement set
      my_spws      = comma-separated string of spws to image
      cp           = dictionary of clean parameters
      spwtype      = 'line' or 'cont', determine which channel
                     parameters to use
      regrid       = if True, look for CVEL2 measurement set
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
    # Set channel parameters
    #
    if spwtype == 'cont':
        restfreqs = [None for spw in my_spws.split(',')]
        start = None
        width = None
        nchan = None
        outframe = cp['contoutframe']
        veltype = None
        interpolation = None
    elif spwtype == 'line':
        spw_inds = [config.get("Spectral Windows","Line").split(',').index(spw) for spw in my_spws.split(',')]
        restfreqs = [config.get("Clean","restfreqs").split(',')[spw_ind] for spw_ind in spw_inds]
        start = cp['velstart']
        width = cp['chanwidth']
        nchan = cp['nchan']
        outframe = cp['lineoutframe']
        veltype = cp['veltype']
        interpolation = cp['interpolation']
    else:
        logger.critical("Error: spwtype {0} not supported".format(spwtype))
        raise ValueError("Invalid spwtype")
    #
    # Loop over spws
    #
    for spw,restfreq in zip(my_spws.split(','),restfreqs):
        #
        # Lightly clean spw
        #
        if regrid:
            myvis = vis.replace('.ms','')+'.spw{0}.cvel.ms'.format(spw)
            myspw = '0'
        else:
            myvis = vis
            myspw = spw
        imagename='{0}.spw{1}.channel.lightclean'.format(field,spw)
        if uvtaper:
            imagename = imagename + '.uvtaper'
            mask = '{0}.spw{1}.mfs.clean.uvtaper.mask'.format(field,spw)
        else:
            mask = '{0}.spw{1}.mfs.clean.mask'.format(field,spw)
        if not os.path.isdir(mask):
            logger.critical("Error: {0} does not exist".format(mask))
            raise ValueError("{0} does not exist".format(mask))
        logger.info("Lightly cleaning spw {0} (restfreq: {1})...".format(spw,restfreq))
        logger.info("Using mask: {0}".format(mask))
        casa.tclean(vis=myvis,imagename=imagename,field=field,spw=myspw,
                    specmode='cube',threshold='0mJy',niter=cp['lightniter']*cp['nchan'],
                    mask=mask,
                    deconvolver='multiscale',scales=cp['scales'],
                    gain=cp['gain'],cyclefactor=cp['cyclefactor'],
                    imsize=cp['imsize'],pblimit=-1.0,cell=cp['cell'],
                    weighting=cp['weighting'],robust=cp['robust'],
                    restfreq=restfreq,start=start,width=width,nchan=nchan,
                    outframe=outframe,veltype=veltype,interpolation=interpolation,
                    uvtaper=cp['outertaper'],pbcor=False)
        #
        # Get RMS of residuals
        #
        dat = casa.imstat(imagename='{0}.residual'.format(imagename),
                          axes=[0,1],mask="'{0}.mask' == 0".format(imagename))
        logger.info("Max un-masked RMS: {0:.2f} mJy/beam".format(1000.*np.max(dat['rms'])))
        logger.info("Mean un-masked RMS: {0:.2f} mJy/beam".format(1000.*np.mean(dat['rms'])))
        logger.info("Median un-masked RMS: {0:.2f} mJy/beam".format(1000.*np.median(dat['rms'])))
        logger.info("Max un-masked MAD*1.4826: {0:.2f} mJy/beam".format(1000.*1.4826*np.max(dat['medabsdevmed'])))
        logger.info("Mean un-masked MAD*1.4826: {0:.2f} mJy/beam".format(1000.*1.4826*np.mean(dat['medabsdevmed'])))
        logger.info("Median un-masked MAD*1.4826: {0:.2f} mJy/beam".format(1000.*1.4826*np.median(dat['medabsdevmed'])))
        logger.info("Using median MADS*1.4826 times {0} (user-defined) as threshold".format(cp['nrms']))
        threshold = '{0:.2f}mJy'.format(cp["nrms"]*1000.*1.4826*np.median(dat['medabsdevmed']))
        #
        # Deep clean to threshold
        #
        imagename='{0}.spw{1}.channel.clean'.format(field,spw)
        if uvtaper:
            imagename = imagename + '.uvtaper'
            mask = '{0}.spw{1}.mfs.clean.uvtaper.mask'.format(field,spw)
        else:
            mask = '{0}.spw{1}.mfs.clean.mask'.format(field,spw)
        if not os.path.isdir(mask):
            logger.critical("Error: {0} does not exist".format(mask))
            raise ValueError("{0} does not exist".format(mask))
        logger.info("Cleaning spw {0} (restfreq: {1}) to threshold: {2}...".format(spw,restfreq,threshold))
        logger.info("Using mask: {0}".format(mask))
        casa.tclean(vis=myvis,imagename=imagename,field=field,spw=myspw,
                    specmode='cube',threshold=threshold,niter=cp['maxniter']*cp['nchan'],
                    mask=mask,
                    deconvolver='multiscale',scales=cp['scales'],
                    gain=cp['gain'],cyclefactor=cp['cyclefactor'],
                    imsize=cp['imsize'],pblimit=-1.0,cell=cp['cell'],
                    weighting=cp['weighting'],robust=cp['robust'],
                    restfreq=restfreq,start=start,width=width,nchan=nchan,
                    outframe=outframe,veltype=veltype,interpolation=interpolation,
                    uvtaper=cp['outertaper'],pbcor=False)
        logger.info("Done.")
        #
        # Primary beam correction
        #
        pbimagename='{0}.spw{1}.channel.pb'.format(field,spw)
        if uvtaper:
            pbimagename = pbimagename + '.uvtaper'
        logger.info("Performing primary beam correction...")
        casa.impbcor(imagename='{0}.image'.format(imagename),
                     pbimage='{0}.pb'.format(pbimagename),
                     outfile='{0}.pbcor.image'.format(imagename))
        logger.info("Done.")
        #
        # Export to fits
        #
        logger.info("Exporting fits file...")
        casa.exportfits(imagename='{0}.image'.format(imagename),
                        fitsimage='{0}.image.fits'.format(imagename),
                        velocity=True,overwrite=True,history=False)
        casa.exportfits(imagename='{0}.mask'.format(imagename),
                        fitsimage='{0}.mask.fits'.format(imagename),
                        overwrite=True,history=False)
        casa.exportfits(imagename='{0}.residual'.format(imagename),
                        fitsimage='{0}.residual.fits'.format(imagename),
                        velocity=True,overwrite=True,history=False)
        casa.exportfits(imagename='{0}.pbcor.image'.format(imagename),
                        fitsimage='{0}.pbcor.image.fits'.format(imagename),
                        velocity=True,overwrite=True,history=False)
        logger.info("Done.")

def contplot(field,uvtaper=False,cp={}):
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
      uvtaper = if True, use uv-tapered images
      cp = dictionary of clean parameters

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    logger.info("Generating continuum images...")
    #
    # Get center pixels
    #
    center_x = int(cp['imsize'][0]/2)
    center_y = int(cp['imsize'][1]/2)
    #
    # Loop over all plot filenames
    #
    if uvtaper:
        fitsfiles = ['{0}.cont.mfs.dirty.uvtaper.image.fits'.format(field),
                 '{0}.cont.mfs.clean.uvtaper.image.fits'.format(field),
                 '{0}.cont.mfs.clean.uvtaper.residual.fits'.format(field),
                 '{0}.cont.mfs.clean.uvtaper.pbcor.image.fits'.format(field),
                 '{0}.cont.mfs.clean.uvtaper.pbcor.image.alpha.fits'.format(field),
                 '{0}.cont.mfs.clean.uvtaper.pbcor.image.alpha.error.fits'.format(field)]
        maskfiles = ['{0}.cont.mfs.clean.uvtaper.mask.fits'.format(field),
                 '{0}.cont.mfs.clean.uvtaper.mask.fits'.format(field),
                 '{0}.cont.mfs.clean.uvtaper.mask.fits'.format(field),
                 '{0}.cont.mfs.clean.uvtaper.mask.fits'.format(field),
                 '',
                 '']
        titles = ['{0} - UVTap/Dirty'.format(field),
              '{0} - UVTap/Clean'.format(field),
              '{0} - UVTap/Residual'.format(field),
              '{0} - UVTap/PBCorr'.format(field),
              '{0} - UVTap/PBCorr/Alpha'.format(field),
              '{0} - UVTap/PBCorr/Alpha Error'.format(field)]
        labels = ['Flux Density (Jy/beam)',
              'Flux Density (Jy/beam)',
              'Flux Density (Jy/beam)',
              'Flux Density (Jy/beam)',
              '',
              '']
        vlims = [(None,None),
             (None,None),
             (None,None),
             (None,None),
             (-1,0),
             (0,0.2)]
    else:
        fitsfiles = ['{0}.cont.mfs.dirty.image.fits'.format(field),
                 '{0}.cont.mfs.clean.image.fits'.format(field),
                 '{0}.cont.mfs.clean.residual.fits'.format(field),
                 '{0}.cont.mfs.clean.pbcor.image.fits'.format(field),
                 '{0}.cont.mfs.clean.pbcor.image.alpha.fits'.format(field),
                 '{0}.cont.mfs.clean.pbcor.image.alpha.error.fits'.format(field)]
        maskfiles = ['{0}.cont.mfs.clean.mask.fits'.format(field),
                 '{0}.cont.mfs.clean.mask.fits'.format(field),
                 '{0}.cont.mfs.clean.mask.fits'.format(field),
                 '{0}.cont.mfs.clean.mask.fits'.format(field),
                 '',
                 '']
        titles = ['{0} - Dirty'.format(field),
              '{0} - Clean'.format(field),
              '{0} - Residual'.format(field),
              '{0} - PBCorr'.format(field),
              '{0} - PBCorr/Alpha'.format(field),
              '{0} - PBCorr/Alpha Error'.format(field)]
        labels = ['Flux Density (Jy/beam)',
              'Flux Density (Jy/beam)',
              'Flux Density (Jy/beam)',
              'Flux Density (Jy/beam)',
              '',
              '']
        vlims = [(None,None),
             (None,None),
             (None,None),
             (None,None),
             (-1,0),
             (0,0.2)]
    for fitsfile,maskfile,title,label,vlim in \
      zip(fitsfiles,maskfiles,titles,labels,vlims):
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
        ax.coords[0].set_major_formatter('hh:mm:ss')
        ax.set_xlabel('RA (J2000)')
        ax.set_ylabel('Declination (J2000)')
        #
        # Plot clean mask as contour
        #
        if maskfile != '':
            maskhdu = fits.open(maskfile)[0]
            contour = ax.contour(maskhdu.data[0,0],levels=[0.5],
                                 origin='lower',colors='r',linewidths=2)
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
            cell = float(cp['cell'].replace('arcsec',''))
            beam_maj = hdu.header['BMAJ']*3600./cell
            beam_min = hdu.header['BMIN']*3600./cell
            beam_pa = hdu.header['BPA']
            ellipse = Ellipse((center_x-int(3.*center_x/4),
                            center_y-int(3.*center_y/4)),
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
    fname = '{0}.contplots.tex'.format(field)
    if uvtaper:
        fname = '{0}.uvtaper.contplots.tex'.format(field)
    with open(fname,'w') as f:
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
    os.system('pdflatex -interaction=batchmode {0}'.format(fname))
    logger.info("Done.")

def lineplot(field,line_spws='',uvtaper=False,cp={}):
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
      uvtaper = if True, use uv-tapered images
      cp = dictionary of clean parameters

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
    center_x = int(cp['imsize'][0]/2)
    center_y = int(cp['imsize'][1]/2)
    #
    # Get center pixels
    #
    goodplots = []
    for spw in line_spws.split(','):
        # check that this spectral window exists
        fname = '{0}.spw{1}.channel.clean.image.fits'.format(field,spw)
        if uvtaper:
            fname = '{0}.spw{1}.channel.clean.uvtaper.image.fits'.format(field,spw)
        if not os.path.exists(fname):
            continue
        #
        # Loop over all plot filenames
        #
        if uvtaper:
            fitsfiles = ['{0}.spw{1}.channel.dirty.uvtaper.image.fits'.format(field,spw),
                     '{0}.spw{1}.channel.clean.uvtaper.image.fits'.format(field,spw),
                     '{0}.spw{1}.channel.clean.uvtaper.residual.fits'.format(field,spw),
                     '{0}.spw{1}.channel.clean.uvtaper.pbcor.image.fits'.format(field,spw)]
            maskfile = '{0}.spw{1}.channel.clean.uvtaper.mask.fits'.format(field,spw)
            titles = ['{0} - spw: {1} - Dirty/UVTap'.format(field,spw),
                  '{0} - spw: {1} - Clean/UVTap'.format(field,spw),
                  '{0} - spw: {1} - Residual/UVTap'.format(field,spw),
                  '{0} - spw: {1} - UVTap/PBCorr'.format(field,spw)]
            labels = ['Flux Density (Jy/beam)',
                  'Flux Density (Jy/beam)',
                  'Flux Density (Jy/beam)',
                  'Flux Density (Jy/beam)']
            vlims = [(None,None),
                 (None,None),
                 (None,None),
                 (None,None)]
        else:
            fitsfiles = ['{0}.spw{1}.channel.dirty.image.fits'.format(field,spw),
                     '{0}.spw{1}.channel.clean.image.fits'.format(field,spw),
                     '{0}.spw{1}.channel.clean.residual.fits'.format(field,spw),
                     '{0}.spw{1}.channel.clean.pbcor.image.fits'.format(field,spw)]
            maskfile = '{0}.spw{1}.channel.clean.mask.fits'.format(field,spw)
            titles = ['{0} - spw: {1} - Dirty'.format(field,spw),
                  '{0} - spw: {1} - Clean'.format(field,spw),
                  '{0} - spw: {1} - Residual'.format(field,spw),
                  '{0} - spw: {1} - PBCorr'.format(field,spw)]
            labels = ['Flux Density (Jy/beam)',
                  'Flux Density (Jy/beam)',
                  'Flux Density (Jy/beam)',
                  'Flux Density (Jy/beam)']
            vlims = [(None,None),
                 (None,None),
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
            ax.coords[0].set_major_formatter('hh:mm:ss')
            ax.set_xlabel('RA (J2000)')
            ax.set_ylabel('Declination (J2000)')
            #
            # Plot clean mask as contour
            #
            maskhdu = fits.open(maskfile)[0]
            contour = ax.contour(maskhdu.data[0,center_chan],levels=[0.5],
                                 origin='lower',colors='r',linewidths=2)
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
                cell = float(cp['cell'].replace('arcsec',''))
                beam_maj = hdu.header['BMAJ']*3600./cell
                beam_min = hdu.header['BMIN']*3600./cell
                beam_pa = hdu.header['BPA']
                ellipse = Ellipse((center_x-int(3.*center_x/4),
                                center_y-int(3.*center_y/4)),
                                beam_min,beam_maj,angle=beam_pa,
                                fill=True,zorder=10,hatch='///',
                                edgecolor='black',facecolor='white')
                ax.add_patch(ellipse)
            elif len(hdulist) > 1:
                hdu = hdulist[1]
                cell = float(cp['cell'].replace('arcsec',''))
                beam_maj = hdu.data['BMAJ'][center_chan]/cell
                beam_min = hdu.data['BMIN'][center_chan]/cell
                beam_pa = hdu.data['BPA'][center_chan]
                ellipse = Ellipse((center_x-int(3.*center_x/4),
                                  center_y-int(3.*center_y/4)),
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
        if uvtaper:
            fitsfile = '{0}.spw{1}.channel.clean.uvtaper.image.fits'.format(field,spw)
        else:
            fitsfile = '{0}.spw{1}.channel.clean.image.fits'.format(field,spw)
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
        if uvtaper:
            ax.set_title('{0} - spw {1} - UVTap/Center'.format(field,spw))
        else:
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
    fname = '{0}.lineplots.tex'.format(field)
    if uvtaper:
        fname = '{0}.uvtaper.lineplots.tex'.format(field)
    with open(fname,'w') as f:
        f.write(r"\documentclass{article}"+"\n")
        f.write(r"\usepackage{graphicx}"+"\n")
        f.write(r"\usepackage[margin=0.1cm]{geometry}"+"\n")
        f.write(r"\begin{document}"+"\n")
        for i in range(0,len(goodplots),5):
            f.write(r"\begin{figure}"+"\n")
            f.write(r"\centering"+"\n")
            f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i]+"}\n")
            if len(goodplots) > i+1: f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i+1]+"}\n")
            if len(goodplots) > i+2: f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i+2]+"}\n")
            if len(goodplots) > i+3: f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i+3]+"}\n")
            if len(goodplots) > i+4: f.write(r"\includegraphics[width=0.45\textwidth]{"+goodplots[i+4]+"}\n")
            f.write(r"\end{figure}"+"\n")
            f.write(r"\clearpage"+"\n")
        f.write(r"\end{document}")
    os.system('pdflatex -interaction=batchmode {0}'.format(fname))
    logger.info("Done.")

def main(field,vis='',spws='',config_file='',
         uvtaper=False,regrid=False,auto=''):
    """
    Generate continuum and line images in various ways using autoclean

    Inputs:
      field       = field name to clean
      vis         = measurement set containing all data for field
      spws        = comma-separated list of line spws to clean
                    if empty, clean all line spws
      config_file = filename of the configuration file for this project
      uvtaper     = if True, apply UV tapering
      regrid      = if True, use CVEL2 to re-grid the velocity axes
      auto        = if not an empty string, it is a comma separated
                    list of menu items to perform, i.e.
                    auto='0,1,4,5,6'

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
    # initial setup, get all line and cont spws
    #
    my_cont_spws,all_line_spws,cp = setup(vis=vis,config=config,uvtaper=uvtaper)
    if spws == '':
        my_line_spws = all_line_spws
    else:
        my_line_spws = spws
    #
    # Determine which spws actually have data
    #
    logger.info("Checking line spws...")
    good_line_spws = []
    for spw in my_line_spws.split(','):
        foo = None
        foo = casa.visstat(vis=vis,spw=spw)
        if foo is not None:
            good_line_spws.append(spw)
    my_line_spws = ','.join(good_line_spws)
    logger.info("Using line spws: {0}".format(my_line_spws))
    logger.info("Checking cont spws...")
    good_cont_spws = []
    for spw in my_cont_spws.split(','):
        foo = None
        foo = casa.visstat(vis=vis,spw=spw)
        if foo is not None:
            good_cont_spws.append(spw)
    my_cont_spws = ','.join(good_cont_spws)
    logger.info("Using cont spws: {0}".format(my_cont_spws))
    #
    # Re-grid
    #
    if regrid:
        regrid_velocity(vis=vis,spws=my_line_spws,config=config,cp=cp)
    #
    # Prompt the user with a menu for each option, or auto-do them
    #
    auto_items = auto.split(',')
    auto_ind = 0
    while True:
        if len(auto) == 0:
            print("0. Dirty image combined continuum spws (MFS; multi-term; multi-scale)")
            print("1. Autoclean combined continuum spws (MFS; multi-term; multi-scale)")
            print("2. Dirty image each continuum spw (MFS; multi-scale)")
            print("3. Autoclean each continuum spw (MFS; multi-scale)")
            print("4. Dirty image each continuum spw (channel; multi-scale)")
            print("5. Autoclean each continuum spw (channel; multi-scale)")
            print("6. Dirty image each line spw (MFS; multi-scale)")
            print("7. Autoclean each line spw (MFS; multi-scale)")
            print("8. Dirty image each line spw (channel; multi-scale)")
            print("9. Autoclean each line spw (channel; multi-scale)")
            print("10. Generate continuum and line diagnostic plots")
            print("q [quit]")
            answer = raw_input("> ")
        else:
            answer = auto_items[auto_ind]
            auto_ind += 1
        if answer == '0':
            mfs_dirty_cont(field=field,vis=vis,
                           my_cont_spws=my_cont_spws,
                           cp=cp,
                           uvtaper=uvtaper)
        elif answer == '1':
            mfs_clean_cont(field=field,vis=vis,
                           my_cont_spws=my_cont_spws,
                           cp=cp,
                           uvtaper=uvtaper)
        elif answer == '2':
            mfs_dirty_spws(field=field,vis=vis,
                           my_spws=my_cont_spws,
                           cp=cp,
                           uvtaper=uvtaper)
        elif answer == '3':
            mfs_clean_spws(field=field,vis=vis,
                           my_spws=my_cont_spws,
                           cp=cp,
                           uvtaper=uvtaper)
        elif answer == '4':
            channel_dirty_spws(field=field,vis=vis,
                                    my_spws=my_cont_spws,
                                    cp=cp,spwtype='cont',
                                    regrid=False,
                                    config=config,
                                    uvtaper=uvtaper)
        elif answer == '5':
            channel_clean_spws(field=field,vis=vis,
                                    my_spws=my_cont_spws,
                                    cp=cp,spwtype='cont',
                                    regrid=False,
                                    config=config,
                                    uvtaper=uvtaper)
        elif answer == '6':
            mfs_dirty_spws(field=field,vis=vis,
                           my_spws=my_line_spws,
                           cp=cp,
                           uvtaper=uvtaper)
        elif answer == '7':
            mfs_clean_spws(field=field,vis=vis,
                           my_spws=my_line_spws,
                           cp=cp,
                           uvtaper=uvtaper)
        elif answer == '8':
            channel_dirty_spws(field=field,vis=vis,
                                    my_spws=my_line_spws,
                                    cp=cp,spwtype='line',
                                    regrid=regrid,
                                    config=config,
                                    uvtaper=uvtaper)
        elif answer == '9':
            channel_clean_spws(field=field,vis=vis,
                                    my_spws=my_line_spws,
                                    cp=cp,spwtype='line',
                                    regrid=regrid,
                                    config=config,
                                    uvtaper=uvtaper)
        elif answer == '10':
            contplot(field,uvtaper=uvtaper,cp=cp)
            lineplot(field,line_spws=all_line_spws,uvtaper=uvtaper,
                     cp=cp)
        elif answer.lower() == 'q' or answer.lower() == 'quit':
            break
        else:
            print("Input not recognized.")
        if auto_ind >= len(auto_items):
            break
