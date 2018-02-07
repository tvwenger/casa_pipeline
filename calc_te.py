"""
calc_te.py
CASA Data Reduction Pipeline - Calculate electron temperature
Trey V. Wenger, Wesley Red July 2017 - V1.0
"""

import __main__ as casa # import casa namespace
import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import logging
import logging.config
import ConfigParser
import shutil
import itertools

__VERSION__ = "1.0"

# load logging configuration file
logging.config.fileConfig('logging.conf')

def gaussian(xdata,amp,center,sigma):
    """
    Compute gaussian function
    """
    ydata = amp*np.exp(-(xdata-center)**2./(2.*sigma**2.))
    return ydata

class ClickPlot:
    """
    Generic class for generating and interacting with matplotlib figures
    """
    def __init__(self,num):
        self.fig = plt.figure(num,figsize=(8,6))
        plt.clf()
        self.ax = self.fig.add_subplot(111)
        self.clickbutton = []
        self.clickx_data = []
        self.clicky_data = []

    def onclick(self,event):
        """
        Handle click event
        """
        if event.button not in [1,3]:
            self.clickbutton.append(-1)
            return
        if not event.inaxes:
            self.clickbutton.append(-1)
            return
        self.clickbutton.append(event.button)
        self.clickx_data.append(event.xdata)
        self.clicky_data.append(event.ydata)

    def get_line_free_regions(self,xdata,ydata,xlabel=None,ylabel=None,title=None):
        """
        Using click events to get the line free regions of a
        spectrum
        """
        self.ax.clear()
        self.ax.grid(False)
        #self.ax.plot(xdata,ydata,'k-')
        self.ax.step(xdata,ydata,'k-',where='mid')
        self.ax.set_title(title.replace('_','\_'))
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlim(np.min(xdata),np.max(xdata))
        yrange = np.max(ydata)-np.min(ydata)
        ymin = np.min(ydata)-0.10*yrange
        ymax = np.max(ydata)+0.10*yrange
        self.ax.set_ylim(ymin,ymax)
        self.clickbutton = []
        self.clickx_data = []
        self.clicky_data = []
        print "Left click to select start of line-free-region."
        print "Left click again to select end of line-free-region."
        print "Repeat as necessary."
        print "Right click when done."
        cid = self.fig.canvas.mpl_connect('button_press_event',
                                          self.onclick)
        self.fig.tight_layout()
        self.fig.show()
        nregions = []
        while True:
            self.fig.waitforbuttonpress()
            if self.clickbutton[-1] == 3:
                if len(nregions) == 0 or len(nregions) % 2 != 0:
                    continue
                else:
                    break
            elif self.clickbutton[-1] == 1:
                if self.clickx_data[-1] < np.min(xdata):
                    nregions.append(np.min(xdata))
                elif self.clickx_data[-1] > np.max(xdata):
                    nregions.append(np.max(xdata))
                else:
                    nregions.append(self.clickx_data[-1])
                self.ax.axvline(nregions[-1])
                self.fig.show()
        self.fig.canvas.mpl_disconnect(cid)
        regions = zip(nregions[::2],nregions[1::2])
        return regions

    def auto_get_line_free_regions(self,xdata,ydata,xlabel=None,ylabel=None,title=None):
        """
        Automatically get the line free regions of a
        spectrum
        """
        self.ax.clear()
        self.ax.grid(False)
        #self.ax.plot(xdata,ydata,'k-')
        self.ax.step(xdata,ydata,'k-',where='mid')
        self.ax.set_title(title.replace('_','\_'))
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlim(np.min(xdata),np.max(xdata))
        yrange = np.max(ydata)-np.min(ydata)
        ymin = np.min(ydata)-0.10*yrange
        ymax = np.max(ydata)+0.10*yrange
        self.ax.set_ylim(ymin,ymax)
        self.fig.tight_layout()
        self.fig.show()
        #
        # Iterate fitting a 3rd order polynomial baseline and
        # rejecting outliers until no new outliers
        # do this on data smoothed by gaussian 3 channels
        #
        smoy = gaussian_filter(ydata,sigma=5.)
        self.ax.plot(xdata,smoy,'g-')
        self.fig.show()
        outliers = np.array([False]*len(xdata))
        while True:
            pfit = np.polyfit(xdata[~outliers],smoy[~outliers],3)
            yfit = np.poly1d(pfit)
            new_smoy = smoy - yfit(xdata)
            rms = np.sqrt(np.mean(new_smoy[~outliers]**2.))
            new_outliers = np.abs(new_smoy) > 3.*rms
            if np.sum(new_outliers) == np.sum(outliers):
                break
            outliers = new_outliers
        #
        # line-free regions are all channels without outliers
        #
        regions = []
        chans = range(len(xdata))
        for val,ch in itertools.groupby(chans,lambda x: outliers[x]):
            if not val: # if not an outlier
                chs = list(ch)
                regions.append([xdata[chs[0]],xdata[chs[-1]]])
                self.ax.axvline(xdata[chs[0]])
                self.ax.axvline(xdata[chs[-1]])
                self.fig.show()
        plt.pause(0.5)
        #print("Click anywhere to continue")
        #self.fig.waitforbuttonpress()
        return regions

    def plot_contfit(self,xdata,ydata,contfit,
                     xlabel=None,ylabel=None,title=None,auto=False):
        """
        Plot data and continuum fit
        """
        self.ax.clear()
        self.ax.grid(False)
        self.ax.plot(xdata,contfit(xdata),'r-')
        #self.ax.plot(xdata,ydata,'k-')
        self.ax.step(xdata,ydata,'k-',where='mid')
        self.ax.set_title(title.replace('_','\_'))
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlim(np.min(xdata),np.max(xdata))
        yrange = np.max(ydata)-np.min(ydata)
        ymin = np.min(ydata)-0.10*yrange
        ymax = np.max(ydata)+0.10*yrange
        self.ax.set_ylim(ymin,ymax)
        self.fig.tight_layout()
        self.fig.show()
        plt.pause(0.5)
        if not auto:
            print("Click anywhere to continue")
            self.fig.waitforbuttonpress()

    def get_gaussian_estimates(self,xdata,ydata,xlabel=None,ylabel=None,title=None):
        """
        Using click events to get the gaussian fit estimates
        """
        self.ax.clear()
        self.ax.grid(False)
        self.ax.axhline(0,color='k')
        #self.ax.plot(xdata,ydata,'k-')
        self.ax.step(xdata,ydata,'k-',where='mid')
        self.ax.set_title(title.replace('_','\_'))
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        xmin = -250
        xmax = 150
        self.ax.set_xlim(xmin,xmax)
        ydata_cut = ydata[np.argmin(np.abs(xdata-xmin)):np.argmin(np.abs(xdata-xmax))]
        yrange = np.max(ydata_cut)-np.min(ydata_cut)
        ymin = np.min(ydata_cut)-0.10*yrange
        ymax = np.max(ydata_cut)+0.10*yrange
        self.ax.set_ylim(ymin,ymax)
        self.clickbutton = []
        self.clickx_data = []
        self.clicky_data = []
        print "Right click to skip fitting this line, or:"
        print "Left click to select start of line region"
        cid = self.fig.canvas.mpl_connect('button_press_event',
                                          self.onclick)
        self.fig.tight_layout()
        self.fig.show()
        self.fig.waitforbuttonpress()
        if 3 in self.clickbutton:
            return None,None,None,None
        self.ax.axvline(self.clickx_data[-1])
        self.fig.show()
        line_start = self.clickx_data[-1]
        print "Left click to select center of line"
        self.fig.waitforbuttonpress()
        self.ax.axvline(self.clickx_data[-1])
        self.fig.show()
        center_guess = self.clickx_data[-1]
        print "Left click to select width of line"
        self.fig.waitforbuttonpress()
        self.ax.axvline(self.clickx_data[-1])
        self.fig.show()
        sigma_guess = self.clickx_data[-1]-center_guess
        print "Left click to select end of line region."
        self.fig.waitforbuttonpress()
        self.ax.axvline(self.clickx_data[-1])
        self.fig.show()
        line_end = self.clickx_data[-1]
        self.fig.canvas.mpl_disconnect(cid)
        return line_start,center_guess,sigma_guess,line_end

    def auto_get_gaussian_estimates(self,xdata,ydata,xlabel=None,ylabel=None,title=None):
        """
        Automatically get the gaussian fit estimates
        """
        self.ax.clear()
        self.ax.grid(False)
        self.ax.axhline(0,color='k')
        #self.ax.plot(xdata,ydata,'k-')
        self.ax.step(xdata,ydata,'k-',where='mid')
        self.ax.set_title(title.replace('_','\_'))
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        xmin = -250
        xmax = 150
        self.ax.set_xlim(xmin,xmax)
        ydata_cut = ydata[np.argmin(np.abs(xdata-xmin)):np.argmin(np.abs(xdata-xmax))]
        yrange = np.max(ydata_cut)-np.min(ydata_cut)
        ymin = np.min(ydata_cut)-0.10*yrange
        ymax = np.max(ydata_cut)+0.10*yrange
        self.ax.set_ylim(ymin,ymax)
        self.fig.tight_layout()
        self.fig.show()
        #
        # Iterate rejecting outliers until no new outliers
        # do this on data smoothed by gaussian 3 channels
        #
        smoy = gaussian_filter(ydata,sigma=5.)
        self.ax.plot(xdata,smoy,'g-')
        self.fig.show()
        outliers = np.array([False]*len(xdata))
        while True:
            rms = np.sqrt(np.mean(smoy[~outliers]**2.))
            new_outliers = np.abs(smoy) > 3.*rms
            if np.sum(new_outliers) == np.sum(outliers):
                break
            outliers = new_outliers
        #
        # Group outlier regions where ydata values are positive
        # and keep the widest region
        #
        line = np.array([])
        chans = range(len(xdata))
        for val,ch in itertools.groupby(chans,lambda x: outliers[x]):
            if val: # if an outlier
                chs = np.array(list(ch))
                # skip if outliers are negative
                if np.sum(ydata[chs]) < 0.:
                    continue
                # skip if this region is smaller than 4 channels
                if len(chs) < 4:
                    continue
                # skip if fewer than 4 (smoothed) values in this region > 5 rms
                if np.sum(smoy[chs] > 5.*rms) < 4:
                    continue
                # skip if this region is smaller than the saved region
                if len(chs) < len(line):
                    continue
                line = xdata[chs]
        # no line to fit if line is empty
        if len(line) == 0:
            plt.pause(0.5)
            #self.fig.waitforbuttonpress()
            return None,None,None,None
        line_start = np.min(line)
        self.ax.axvline(line_start)
        line_center = np.mean(line)
        self.ax.axvline(line_center)
        line_end = np.max(line)
        self.ax.axvline(line_end)
        line_width = (line_end-line_center)/2.
        self.ax.axvline(line_center+line_width)
        self.fig.show()
        plt.pause(0.5)
        #self.fig.waitforbuttonpress()
        return line_start,line_center,line_width,line_end

    def plot_fit(self,xdata,ydata,amp,center,sigma,
                 xlabel=None,ylabel=None,title=None,
                 outfile=None,auto=False):
        """
        Plot data and fit and residuals
        """
        self.ax.clear()
        self.ax.grid(False)
        self.ax.axhline(0,color='k')
        yfit = gaussian(xdata,amp,center,sigma)
        residuals = ydata-yfit
        #self.ax.plot(xdata,residuals,'m-')
        self.ax.step(xdata,residuals,'m-',where='mid')
        self.ax.plot(xdata,yfit,'r-')
        #self.ax.plot(xdata,ydata,'k-')
        self.ax.step(xdata,ydata,'k-',where='mid')
        self.ax.set_title(title.replace('_','\_'))
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        xmin = -250
        xmax = 150
        self.ax.set_xlim(xmin,xmax)
        ydata_cut = ydata[np.argmin(np.abs(xdata-xmin)):np.argmin(np.abs(xdata-xmax))]
        yrange = np.max(ydata_cut)-np.min(ydata_cut)
        ymin = np.min(ydata_cut)-0.10*yrange
        ymax = np.max(ydata_cut)+0.10*yrange
        self.ax.set_ylim(ymin,ymax)
        self.fig.tight_layout()
        self.fig.savefig(outfile)
        self.fig.show()
        plt.pause(0.5)
        if not auto:
            print("Click anywhere to continue")
            self.fig.waitforbuttonpress()

def calc_te(imagename,region,fluxtype,freq=None,auto=False):
    """
    Extract spectrum from region, measure RRL shape and continuum
    brightness, calculate electron temperature

    Inputs:
      imagename = image to analyze
      region = region file
      fluxtype = what type of flux to measure ('flux' or 'mean')
      freq = if None, get frequency from image header (MHz)

    Returns:
      te = electron temperature (K)
      e_te = error on electron temperature (K)
      line_to_cont = line to continuum ratio
      e_line_to_cont = error
      line_brightness = line strength (mJy)
      e_line_brightness = error
      line_fwhm = line width (km/s)
      e_line_fwhm = error
      cont_brightness = continuum brightness (mJy)
      rms = error (mJy)
      freq = frequency
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Open image, extract spectrum from region
    #
    logfile = '{0}.{1}.specflux'.format(imagename,region)
    casa.specflux(imagename=imagename,region=region,function=fluxtype,
                  logfile=logfile,overwrite=True)
    if fluxtype == 'flux':
        ylabel = 'Flux (mJy)'
    else:
        ylabel = 'Flux Density (mJy/beam)'
    #
    # Import spectrum and plot it
    #
    try:
        specdata = np.genfromtxt(logfile,comments='#',dtype=None,
                                 names=('channel','npts','freq','velocity','flux'))
    except:
        # region is all NaNs (i.e. outside of primary beam)
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan, np.nan)
    specdata['flux'] = specdata['flux']*1000. # Jy -> mJy
    myplot = ClickPlot(1)
    #
    # Remove NaNs (if any)
    #
    is_nan = np.isnan(specdata['flux'])
    specdata_flux = specdata['flux'][~is_nan]
    specdata_velocity = specdata['velocity'][~is_nan]
    #
    # Get line-free regions
    #
    title = '{0}\n{1}'.format(imagename,region)
    if auto:
        regions = myplot.auto_get_line_free_regions(specdata_velocity,specdata_flux,
                                                    xlabel='Velocity (km/s)',ylabel=ylabel,
                                                    title=title)
    else:
        regions = myplot.get_line_free_regions(specdata_velocity,specdata_flux,
                                               xlabel='Velocity (km/s)',ylabel=ylabel,
                                               title=title)
    #
    # Extract line free velocity and flux
    #
    line_free_mask = np.zeros(specdata_velocity.size,dtype=bool)
    for reg in regions:
        line_free_mask[(specdata_velocity>reg[0])&(specdata_velocity<reg[1])] = True
    line_free_velocity = specdata_velocity[line_free_mask]
    line_free_flux = specdata_flux[line_free_mask]
    #
    # Fit baseline as polyonimal order 3
    #
    pfit = np.polyfit(line_free_velocity,line_free_flux,3)
    contfit = np.poly1d(pfit)
    myplot.plot_contfit(specdata_velocity,specdata_flux,contfit,
                        xlabel='Velocity (km/s)',ylabel=ylabel,
                        title=title,auto=auto)
    #
    # Subtract continuum
    #
    flux_contsub = specdata_flux - contfit(specdata_velocity)
    line_free_flux_contsub = line_free_flux - contfit(line_free_velocity)
    #
    # Calculate average continuum
    #
    cont_brightness = np.mean(line_free_flux)
    #
    # Compute RMS
    #
    rms = np.sqrt(np.mean(line_free_flux_contsub**2.))
    #
    # Re-plot spectrum, get Gaussian fit estimates
    #
    if auto:
        line_start,center_guess,sigma_guess,line_end = \
        myplot.auto_get_gaussian_estimates(specdata_velocity,flux_contsub,
                                           xlabel='Velocity (km/s)',ylabel=ylabel,title=title)
    else:
        line_start,center_guess,sigma_guess,line_end = \
        myplot.get_gaussian_estimates(specdata_velocity,flux_contsub,
                                      xlabel='Velocity (km/s)',ylabel=ylabel,title=title)
    if None in [line_start,center_guess,sigma_guess,line_end]:
        line_brightness = np.nan
        e_line_brightness = np.nan
        line_center = np.nan
        e_line_center = np.nan
        line_sigma = np.nan
        e_line_sigma = np.nan
        line_fwhm = np.nan
        e_line_fwhm = np.nan
    else:
        center_idx = np.argmin(np.abs(specdata_velocity-center_guess))
        amp_guess = flux_contsub[center_idx]
        #
        # Extract line velocity and fluxes
        #
        line_mask = (specdata_velocity>line_start)&(specdata_velocity<line_end)
        line_flux = flux_contsub[line_mask]
        line_velocity = specdata_velocity[line_mask]
        #
        # Fit gaussian to data
        #
        try:
            popt,pcov = curve_fit(gaussian,line_velocity,line_flux,
                                  p0=(amp_guess,center_guess,sigma_guess),
                                  sigma=np.ones(line_flux.size)*rms)
            line_brightness = popt[0]
            e_line_brightness = np.sqrt(pcov[0][0])
            line_center = popt[1]
            e_line_center = np.sqrt(pcov[1][1])
            line_sigma = np.abs(popt[2])
            e_line_sigma = np.sqrt(np.abs(pcov[2][2]))
            line_fwhm = 2.*np.sqrt(2.*np.log(2.))*line_sigma
            e_line_fwhm = 2.*np.sqrt(2.*np.log(2.))*e_line_sigma
        except:
            line_brightness = np.nan
            e_line_brightness = np.nan
            line_center = np.nan
            e_line_center = np.nan
            line_sigma = np.nan
            e_line_sigma = np.nan
            line_fwhm = np.nan
            e_line_fwhm = np.nan            
    #
    # Plot fit
    #
    outfile='{0}.{1}.spec.pdf'.format(imagename,region)
    myplot.plot_fit(specdata_velocity,flux_contsub,line_brightness,line_center,line_sigma,
                    xlabel='Velocity (km/s)',ylabel=ylabel,title=title,
                    outfile=outfile,auto=auto)
    #
    # Calculate electron temperature
    #
    if freq is None:
        freq = np.mean(specdata['freq'])
    line_to_cont = line_brightness/cont_brightness
    e_line_to_cont = line_to_cont * np.sqrt(rms**2./cont_brightness**2. + e_line_brightness**2./line_brightness**2.)
    y = 0.08 # Balser 2011 default value
    te = (7103.3*(freq/1000.)**1.1/line_to_cont/line_fwhm/(1.+y))**0.87
    e_te = 0.87*te*np.sqrt(e_line_fwhm**2./line_fwhm**2. + rms**2./cont_brightness**2. + e_line_brightness**2./line_brightness**2.)
    #
    # Return results
    #
    return (te, e_te, line_to_cont, e_line_to_cont, line_brightness, e_line_brightness, line_fwhm, e_line_fwhm,
            line_center, e_line_center, cont_brightness, rms, freq)

def main(field,region,
         stackedimages=[],stackedlabels=[],stackedfreqs=[],
         fluxtype='flux',
         lineids=[],linetype='dirty',
         outfile='electron_temps.txt',config_file=None,auto=False):
    """
   Extract spectrum from region in each RRL image, measure continuum
   brightess and fit Gaussian to measure RRL properties. Also fit stacked 
   line. Compute electron temperature.

    Inputs:
      field       = field to analyze
      region = filename of region where to extract spectrum
      stackedimagelabels = what to call the stacked image data in outfile
      stackedimages = filenames of stacked images
      fluxtype = what type of flux to measure ('flux' or 'mean')
      lineids     = lines to stack, if empty all lines
      linetype = 'clean' or 'dirty'
      outfile = where the results go
      config_file = configuration file

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
    # Check if we supplied lineids
    #
    if len(lineids) == 0:
        lineids = config.get("Clean","lineids").split(',')
        goodlineids = []
        for lineid in lineids:
            if os.path.isdir('{0}.{1}.channel.{2}.imsmooth'.format(field,lineid,linetype)):
                goodlineids.append(lineid)
        lineids = goodlineids
    #
    # Set-up file
    #
    with open(outfile,'w') as f:
        # 0       1           2      3      4        5        6     7      8        9        10        11          12        13          
        # lineid  frequency   velo   e_velo line     e_line   fwhm  e_fwhm cont     rms      line2cont e_line2cont elec_temp e_elec_temp 
        # #       MHz         km/s   km/s   mJy/beam mJy/beam km/s  km/s   mJy/beam mJy/beam                       K         K           
        # H122a   9494.152594 -100.0 50.0   1000.0   100.0    100.0 10.0   1000.0   100.0    0.050     0.001       10000.0   1000.0      
        # stacked 9494.152594 -100.0 50.0   1000.0   100.0    100.0 10.0   1000.0   100.0    0.050     0.001       10000.0   1000.0      
        # 1234567 12345678902 123456 123456 12345678 12345678 12345 123456 12345678 12345678 123456789 12345678901 123456789 12345678901 
        #
        headerfmt = '{0:12} {1:12} {2:6} {3:6} {4:8} {5:8} {6:5} {7:6} {8:8} {9:8} {10:9} {11:11} {12:9} {13:11}\n'
        rowfmt = '{0:12} {1:12.6f} {2:6.1f} {3:6.1f} {4:8.1f} {5:8.1f} {6:5.1f} {7:6.1f} {8:8.1f} {9:8.1f} {10:9.3f} {11:11.3f} {12:9.1f} {13:11.1f}\n'
        f.write(headerfmt.format('lineid','frequency','velo','e_velo',
                                 'line','e_line','fwhm','e_fwhm',
                                 'cont','rms','line2cont','e_line2cont',
                                 'elec_temp','e_elec_temp'))
        if fluxtype == 'flux':
            fluxunit = 'mJy'
        else:
            fluxunit = 'mJy/beam'
        f.write(headerfmt.format('#','MHz','km/s','km/s',
                                 fluxunit,fluxunit,'km/s','km/s',
                                 fluxunit,fluxunit,'','','K','K'))
        #
        # Compute electron temperature for each individual RRL
        # 
        for lineid in lineids:
            #
            # Generate file names
            # 
            imagename = '{0}.{1}.channel.{2}.imsmooth'.format(field,lineid,linetype)
            te, e_te, line_to_cont, e_line_to_cont, line_brightness, e_line_brightness, line_fwhm, e_line_fwhm, \
                line_center,e_line_center,cont_brightness, rms, freq = calc_te(imagename,region,fluxtype, auto=auto)
            f.write(rowfmt.format(lineid, freq,
                                  line_center, e_line_center,
                                  line_brightness, e_line_brightness,
                                  line_fwhm, e_line_fwhm,
                                  cont_brightness, rms,
                                  line_to_cont, e_line_to_cont,
                                  te, e_te))
        #
        # Compute electron temperature for stacked line images
        #
        for image,label,freq in zip(stackedimages,stackedlabels,stackedfreqs):
            te, e_te, line_to_cont, e_line_to_cont, line_brightness, e_line_brightness, line_fwhm, e_line_fwhm, \
                line_center, e_line_center, cont_brightness, rms, freq = calc_te(image,region,fluxtype,freq=freq, auto=auto)
            f.write(rowfmt.format(label, freq,
                              line_center, e_line_center,
                              line_brightness, e_line_brightness,
                              line_fwhm, e_line_fwhm,
                              cont_brightness, rms,
                              line_to_cont, e_line_to_cont,
                              te, e_te))
    #
    # Generate TeX file of all plots
    #
    logger.info("Generating PDF...")
    plots = ['{0}.{1}.channel.{2}.imsmooth.{3}.spec.pdf'.format(field,lineid,linetype,region) for lineid in lineids]
    plots = plots + ['{0}.{1}.spec.pdf'.format(image,region) for image in stackedimages]
    # remove plots that don't exist
    plots = [plot for plot in plots if os.path.exists(plot)]
    # fix filenames so LaTeX doesn't complain
    plots = ['{'+fn.replace('.pdf','')+'}.pdf' for fn in plots]
    with open('{0}.spectra.tex'.format(region),'w') as f:
        f.write(r"\documentclass{article}"+"\n")
        f.write(r"\usepackage{graphicx}"+"\n")
        f.write(r"\usepackage[margin=0.1cm]{geometry}"+"\n")
        f.write(r"\begin{document}"+"\n")
        for i in range(0,len(plots),6):
            f.write(r"\begin{figure}"+"\n")
            f.write(r"\centering"+"\n")
            if len(plots) > i: f.write(r"\includegraphics[width=0.45\textwidth]{"+plots[i]+"}\n")
            if len(plots) > i+3: f.write(r"\includegraphics[width=0.45\textwidth]{"+plots[i+3]+"}\n")
            if len(plots) > i+1: f.write(r"\includegraphics[width=0.45\textwidth]{"+plots[i+1]+"}\n")
            if len(plots) > i+4: f.write(r"\includegraphics[width=0.45\textwidth]{"+plots[i+4]+"}\n")
            if len(plots) > i+2: f.write(r"\includegraphics[width=0.45\textwidth]{"+plots[i+2]+"}\n")
            if len(plots) > i+5: f.write(r"\includegraphics[width=0.45\textwidth]{"+plots[i+5]+"}\n")
            f.write(r"\end{figure}"+"\n")
            f.write(r"\clearpage"+"\n")
        f.write(r"\end{document}")
    os.system('pdflatex -interaction=batchmode {0}.spectra.tex'.format(region))
    logger.info("Done.")
        
