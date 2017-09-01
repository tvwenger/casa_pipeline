"""
calc_te.py
CASA Data Reduction Pipeline - Calculate electron temperature
Trey V. Wenger, Wesley Red July 2017 - V1.0
"""

import __main__ as casa # import casa namespace
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import logging
import logging.config
import ConfigParser
import shutil

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

    def get_line_free_regions(self,xdata,ydata,xlabel=None,ylabel=None,title=None):
        """
        Using click events to get the line free regions of a
        spectrum
        """
        self.ax.clear()
        self.ax.plot(xdata,ydata,'k-')
        self.ax.set_title(title)
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

    def get_gaussian_estimates(self,xdata,ydata,xlabel=None,ylabel=None,title=None):
        """
        Using click events to get the gaussian fit estimates
        """
        self.ax.clear()
        self.ax.plot(xdata,ydata,'k-')
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.clickbutton = []
        self.clickx_data = []
        self.clicky_data = []
        print "Left click to select start of line region"
        print "Left click to select center of line"
        print "Left click to select width of line"
        print "Left click again to select end of line region."
        cid = self.fig.canvas.mpl_connect('button_press_event',
                                          self.onclick)
        self.fig.show()
        self.fig.waitforbuttonpress()
        self.ax.axvline(self.clickx_data[-1])
        line_start = self.clickx_data[-1]
        self.fig.waitforbuttonpress()
        self.ax.axvline(self.clickx_data[-1])
        center_guess = self.clickx_data[-1]
        self.fig.waitforbuttonpress()
        self.ax.axvline(self.clickx_data[-1])
        sigma_guess = self.clickx_data[-1]-center_guess
        self.fig.waitforbuttonpress()
        self.ax.axvline(self.clickx_data[-1])
        line_end = self.clickx_data[-1]
        self.fig.canvas.mpl_disconnect(cid)
        return line_start,center_guess,sigma_guess,line_end

    def plot_fit(self,xdata,ydata,amp,center,sigma,xlabel=None,ylabel=None,title=None):
        """
        Plot data and fit and residuals
        """
        self.ax.clear()
        yfit = gaussian(xdata,amp,center,sigma)
        residuals = ydata-yfit
        self.ax.plot(xdata,residuals,'m-')
        self.ax.plot(xdata,yfit,'r-')
        self.ax.plot(xdata,ydata,'k-')
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.fig.show()
        print("Click anywhere to continue")
        self.fig.waitforbuttonpress()

def calc_te(imagename,region,freq=None):
    """
    Extract spectrum from region, measure RRL shape and continuum
    brightness, calculate electron temperature

    Inputs:
      imagename = image to analyze
      region = region file
      freq = if None, get frequency from image header

    Returns:
      te = electron temperature (K)
      e_te = error on electron temperature (K)
      line_to_cont = line to continuum ratio
      e_line_to_cont = error
      line_brightness = line strength (Jy)
      e_line_brightness = error
      line_fwhm = line width (km/s)
      e_line_fwhm = error
      cont_brightness = continuum brightness (Jy)
      rms = error
      freq = frequency
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Open image, extract spectrum from region
    #
    logfile = '{0}.specflux'.format(imagename)
    casa.specflux(imagename=imagename,region=region,logfile=logfile)
    #
    # Import spectrum and plot it
    #
    specdata = np.genfromtxt(logfile,comments='#',dtype=None,
                             names=('channel','npts','freq','velocity','flux'))
    myplot = ClickPlot(1)
    #
    # Get line-free regions
    #
    regions = myplot.get_line_free_regions(specdata['velocity'],specdata['flux'],
                                           xlabel='Velocity (km/s)',ylabel='Flux (Jy)',title=imagename)
    #
    # Extract line free velocity and flux
    #
    line_free_mask = np.zeros(specdata['velocity'].size,dtype=bool)
    for region in regions:
        line_free_mask[(specdata['velocity']>region[0])&(specdata['velocity']<region[1])] = True
    line_free_velocity = specdata['velocity'][line_free_mask]
    line_free_flux = specdata['flux'][line_free_mask]
    #
    # Fit and remove continuum (degree = 0, constant value)
    #
    pfit = np.polyfit(line_free_velocity,line_free_flux,0)
    cont_brightness = pfit[0]
    #
    # Compute RMS
    #
    rms = np.sqrt(np.mean((line_free_flux-cont_brightness)**2.))
    #
    # Subtract continuum
    #
    flux_contsub = specdata['flux'] - cont_brightness
    #
    # Re-plot spectrum, get Gaussian fit estimates
    #
    line_start,center_guess,sigma_guess,line_end = \
     myplot.get_gaussian_estimates(specdata['velocity'],flux_contsub,
                                                         xlabel='Velocity (km/s)',ylabel='Flux (Jy)',title=imagename)
    center_idx = np.argmin(np.abs(specdata['velocity']-center_guess))
    amp_guess = flux_contsub[center_idx]
    #
    # Extract line velocity and fluxes
    #
    line_mask = (specdata['velocity']>line_start)&(specdata['velocity']<line_end)
    line_flux = flux_contsub[line_mask]
    line_velocity = specdata['velocity'][line_mask]
    #
    # Fit gaussian to data
    #
    popt,pcov = curve_fit(gaussian,line_velocity,line_flux,
                                       p0=(amp_guess,center_guess,sigma_guess),
                                       sigma=np.ones(line_flux.size)*rms)
    line_brightness = popt[0]
    e_line_brightness = np.sqrt(pcov[0][0])
    line_center = popt[1]
    e_line_center = np.sqrt(pcov[1][1])
    line_sigma = popt[2]
    e_line_sigma = np.sqrt(pcov[2][2])
    line_fwhm = 2.*np.sqrt(2.*np.log(2.))*line_sigma
    e_line_fwhm = 2.*np.sqrt(2.*np.log(2.))*e_line_sigma
    #
    # Plot fit
    #
    myplot.plot_fit(specdata['velocity'],flux_contsub,line_brightness,line_center,line_sigma,
                             xlabel='Velocity (km/s)',ylabel='Flux (Jy)',title=imagename)
    #
    # Calculate electron temperature
    #
    if freq is None:
        freq = np.mean(specdata['freq'])/1000.
    line_to_cont = line_brightness/cont_brightness
    e_line_to_cont = line_to_cont * np.sqrt(rms**2./cont_brightness**2. + e_line_brightness**2./line_brightness**2.)
    y = 0.08 # Balser 2011 default value
    te = (7103.3*freq**1.1/line_to_cont/line_fwhm/(1.+y))**0.87
    e_te = 0.87*te*np.sqrt(e_line_fwhm**2./line_fwhm**2. + rms**2./cont_brightness**2. + e_line_brightness**2./line_brightness**2.)
    #
    # Return results
    #
    return (te, e_te, line_to_cont, e_line_to_cont, line_brightness, e_line_brightness, line_fwhm, e_line_fwhm,
               cont_brightness, rms, freq)

def main(field,lineids=[],linetype='dirty',outfile='electron_temps.txt'):
    """
   Extract spectrum from region in each RRL image, measure continuum
   brightess and fit Gaussian to measure RRL properties. Also fit stacked 
   line. Compute electron temperature.

    Inputs:
      field       = field to analyze
      lineids     = lines to stack, if empty all lines
      linetype = 'clean' or 'dirty'
      outfile = where the results go

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Set-up file
    #
    region = '{0}.reg'.format(field)
    with open(outfile,'w') as f:
        f.write('{0:13} {1:5} {2:5} {3:6} {4:9} {5:11} {6:5} {7:6} {8:9} {9:11} {10:7} {11:7}\n'.\
                   format('lineid','freq','line','e_line','line_fwhm','e_line_fwhm','cont','rms','line2cont','e_line2cont','te','e_te'))
        f.write('#{0:12} {1:5} {2:5} {3:6} {4:9} {5:11} {6:5} {7:6} {8:9} {9:11} {10:7} {11:7}\n'.\
                   format('','GHz','mJy','mJy','km/s','km/s','mJy','mJy','','','K','K'))
        #
        # Compute electron temperature for each individual RRL
        # 
        freqs = np.array([])
        for lineid in lineids:
            #
            # Generate file names
            # 
            imagename = '{0}.{1}.channel.{2}.imsmooth'.format(field,lineid,linetype)
            te, e_te, line_to_cont, e_line_to_cont, line_brightness, e_line_brightness, line_fwhm, e_line_fwhm, \
                cont_brightness, rms, freq = calc_te(imagename,region)
            f.write('{0:13} {1:5.3f} {2:5.2f} {3:6.2f} {4:9.2f} {5:11.2f} {6:5.2f} {7:6.2f} {8:9.3f} {9:11.3f} {10:7.1f} {11:7.1f}\n'.\
                       format(lineid, freq, line_brightness*1000., e_line_brightness*1000., line_fwhm, e_line_fwhm, cont_brightness*1000.,
                                   rms*1000., line_to_cont, e_line_to_cont, te, e_te))
            freqs = np.append(freqs,freq)
        #
        # Compute electron temperature for stacked line image
        #
        lineid = 'Halpha_{0}lines'.format(len(lineids))
        imagename = '{0}.{1}.channel.{2}.image'.format(field,lineid,linetype)
        te, e_te, line_to_cont, e_line_to_cont, line_brightness, e_line_brightness, line_fwhm, e_line_fwhm, \
                cont_brightness, rms, freq = calc_te(imagename,region,freq=np.mean(freqs))
        f.write('{0:13} {1:5.3f} {2:5.2f} {3:6.2f} {4:9.2f} {5:11.2f} {6:5.2f} {7:6.2f} {8:9.3f} {9:11.3f} {10:7.1f} {11:7.1f}\n'.\
                       format(lineid, freq, line_brightness*1000., e_line_brightness*1000., line_fwhm, e_line_fwhm, cont_brightness*1000.,
                                   rms*1000., line_to_cont, e_line_to_cont, te, e_te))
