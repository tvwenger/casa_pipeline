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

class ClickPlot:
    """
    Generic class for generating and interacting with matplotlib figures
    """
    def __init__(self,num):
        self.fig = plt.figure(num)
        plt.clf()
        print "Created figure",self.fig.number
        self.ax = self.fig.add_subplot(111)
        self.clickbutton = []
        self.clickx_data = []
        self.clicky_data = []

    def onclick(self,event):
        """
        Handle click event
        """
        if event.button not in [1,3]:
            return
        try:
            print "Click at ({0:.2f},{1:.2f})".format(event.xdata,
                                                      event.ydata)
        except ValueError:
            return
        self.clickbutton.append(event.button)
        self.clickx_data.append(event.xdata)
        self.clicky_data.append(event.ydata)

    def get_line_free_regions(self,xdata,ydata,xlabel=None,ylabel=None):
        """
        Using click events to get the line free regions of a
        spectrum
        """
        self.ax.clear()
        self.ax.plot(xdata,ydata,'k-')
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.clickbutton = []
        self.clickx_data = []
        self.clicky_data = []
        print "Left click to select start of line-free-region."
        print "Left click again to select end of line-free-region."
        print "Repeat as necessary."
        print "Right click when done."
        cid = self.fig.canvas.mpl_connect('button_press_event',
                                          self.onclick)
        self.fig.show()
        while True:
            self.fig.waitforbuttonpress()
            if 3 in self.clickbutton:
                break
            elif 1 in self.clickbutton:
                self.ax.axvline(self.clickx_data[-1])
        self.fig.canvas.mpl_disconnect(cid)
        # remove last element (right-click)
        self.clickx_data = self.clickx_data[0:-1]
        # check that there are an even number, otherwise remove last
        # element
        if len(self.clickx_data) % 2 != 0:
            self.clickx_data = self.clickx_data[0:-1]
        regions = zip(self.clickx_data[::2],self.clickx_data[1::2])
        return regions

def contsub(field,my_line_spws='',overwrite=False):
    """
    Subtact continuum from line spws, save line-free-region RMS
    to file.

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
    logger.info("Performing continuum subtraction")
    with open('{0}.line_rms.txt'.format(field),'w') as f:
        for spw in my_line_spws.split(','):
            imagename='{0}.spw{1}.clean.pbcor'.format(field,spw)
            if not os.path.isdir(imagename):
                logger.warn("{0} was not found!".format(imagename))
                continue
            region='{0}.spw{1}.reg'.format(field,spw)
            if not os.path.exists(region):
                logger.warn("{0} was not found!".format(region))
                region='{0}.reg'.format(field)
                if not os.path.exists(region):
                    logger.warn("{0} was not found, skipping...".format(region))
                    continue
            linefile='{0}.spw{1}.clean.line'.format(field,spw)
            contfile='{0}.spw{1}.clean.cont'.format(field,spw)
            if os.path.isdir(linefile) or os.path.isdir(contfile):
                if overwrite:
                    logger.info("Overwriting {0}".format(linefile))
                    logger.info("Overwriting {0}".format(contfile))
                    shutil.rmtree(linefile)
                    shutil.rmtree(contfile)
                else:
                    logger.info("{0} or {1} exists.".format(linefile,contfile))
                    continue
            stats = casa.imstat(imagename=imagename,region=region,axes=[0,1])
            myplot = ClickPlot(0)
            regs = myplot.get_line_free_regions(range(len(stats['flux'])),
                                                stats['flux'],
                                                xlabel='channel',
                                                ylabel='flux (Jy)')
            chans = []
            fluxdata = np.array([])
            for idx,reg in enumerate(regs):
                start,end = int(reg[0]),int(reg[1])
                if start < 0:
                    start = 0
                if end >= len(stats['flux']):
                    end = len(stats['flux'])-1
                chans.append('{0}~{1}'.format(start,end))
                fluxdata = np.append(fluxdata,stats['flux'][start:end])
            chans = ','.join(chans)
            casa.imcontsub(imagename=imagename,linefile=linefile,
                           contfile=contfile,chans=chans)
            rms = np.sqrt(np.mean((fluxdata-np.mean(fluxdata))**2.))
            f.write('{0:2} {1:6.3f}\n'.format(spw,rms))
    logger.info("Done!")

def combeam_line_spws(field,my_line_spws='',overwrite=False):
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
        imagename='{0}.spw{1}.clean.line'.format(field,spw)
        if not os.path.isdir(imagename):
            logger.warn("{0} was not found!".format(imagename))
            continue
        outfile='{0}.spw{1}.clean.line.combeam'.format(field,spw)
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

def smooth_all(field,my_line_spws='',config=None,overwrite=False):
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
    contimage = '{0}.cont.clean.pbcor'.format(field)
    print(contimage)
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
        lineimage = '{0}.spw{1}.clean.line.combeam'.format(field,spw)
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
    contimage = '{0}.cont.clean.pbcor'.format(field)
    if os.path.isdir(contimage):
        outfile='{0}.cont.imsmooth'.format(field)
        casa.imsmooth(imagename=contimage,kernel='gauss',
                      targetres=True,major=bmaj_target,minor=bmin_target,
                      pa=bpa_target,outfile=outfile,overwrite=overwrite)
    for spw,lineid in zip(my_line_spws.split(','),config.get('Clean','lineids').split(',')):
        lineimage = '{0}.spw{1}.clean.line.combeam'.format(field,spw)
        if os.path.isdir(lineimage):
            outfile = '{0}.{1}.imsmooth'.format(field,lineid)
            casa.imsmooth(imagename=lineimage,kernel='gauss',
                          targetres=True,major=bmaj_target,minor=bmin_target,
                          pa=bpa_target,outfile=outfile,overwrite=overwrite)
    logger.info("Done!")

def stack_line(field,lineids=[],config=None,overwrite=False):
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
    images = ['{0}.{1}.imsmooth'.format(field,lineid) for lineid in lineids]
    ims = ['IM{0}'.format(foo) for foo in range(len(images))]
    myexp =  '({0})/{1}'.format('+'.join(ims),str(float(len(images))))
    outfile='{0}.Halpha_{1}lines.image'.format(field,str(len(images)))
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

def linetocont_image(field,moment0image='',overwrite=False):
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
    contimage='{0}.cont.imsmooth'.format(field)
    if not os.path.isdir(contimage):
        logger.warn("{0} does not exist".format(contimage))
        return
    outfile='{0}.linetocont.image'.format(field)
    images = [moment0image,contimage]
    myexp = 'IM0/IM1'
    casa.immath(imagename=images,outfile=outfile,mode='evalexpr',
                expr=myexp)
    logger.info("Done!")

def main(field,lineids=[],config_file='',overwrite=False):
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
    # Continuum subtact line images
    #
    contsub(field,my_line_spws=my_line_spws,overwrite=overwrite)
    #
    # smooth line images to common beam
    #
    combeam_line_spws(field,my_line_spws=my_line_spws,overwrite=overwrite)
    #
    # Smooth all line and continuum images to common beam, rename
    # by lineid
    #
    smooth_all(field,my_line_spws=my_line_spws,config=config,
               overwrite=overwrite)
    #
    # Stack line images
    #
    stackedimage = stack_line(field,lineids=lineids,config=config,
                              overwrite=overwrite)
    #
    # make moment 0 image
    #
    moment0image = moment0_image(stackedimage,overwrite=overwrite)
    #
    # create line-to-continuum image using stacked line image
    #
    linetocont_image(field,moment0image=moment0image,overwrite=overwrite)
