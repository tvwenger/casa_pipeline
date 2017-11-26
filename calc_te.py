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
        self.fig = plt.figure(num,figsize=(8,6))
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

    def plot_contfit(self,xdata,ydata,contfit,
                     xlabel=None,ylabel=None,title=None):
        """
        Plot data and continuum fit
        """
        self.ax.clear()
        self.ax.plot(xdata,contfit(xdata),'r-')
        self.ax.plot(xdata,ydata,'k-')
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
        print("Click anywhere to continue")
        self.fig.waitforbuttonpress()

    def get_gaussian_estimates(self,xdata,ydata,xlabel=None,ylabel=None,title=None):
        """
        Using click events to get the gaussian fit estimates
        """
        self.ax.clear()
        self.ax.plot(xdata,ydata,'k-')
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
        line_start = self.clickx_data[-1]
        print "Left click to select center of line"
        self.fig.waitforbuttonpress()
        self.ax.axvline(self.clickx_data[-1])
        center_guess = self.clickx_data[-1]
        print "Left click to select width of line"
        self.fig.waitforbuttonpress()
        self.ax.axvline(self.clickx_data[-1])
        sigma_guess = self.clickx_data[-1]-center_guess
        print "Left click to select end of line region."
        self.fig.waitforbuttonpress()
        self.ax.axvline(self.clickx_data[-1])
        line_end = self.clickx_data[-1]
        self.fig.canvas.mpl_disconnect(cid)
        return line_start,center_guess,sigma_guess,line_end

    def plot_fit(self,xdata,ydata,amp,center,sigma,
                 xlabel=None,ylabel=None,title=None,
                 outfile=None):
        """
        Plot data and fit and residuals
        """
        self.ax.clear()
        yfit = gaussian(xdata,amp,center,sigma)
        residuals = ydata-yfit
        self.ax.plot(xdata,residuals,'m-')
        self.ax.plot(xdata,yfit,'r-')
        self.ax.plot(xdata,ydata,'k-')
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
        print("Click anywhere to continue")
        self.fig.waitforbuttonpress()

def calc_te(imagename,region,fluxtype,freq=None):
    """
    Extract spectrum from region, measure RRL shape and continuum
    brightness, calculate electron temperature

    Inputs:
      imagename = image to analyze
      region = region file
      fluxtype = what type of flux to measure ('flux density' or 'mean')
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
    logfile = '{0}.specflux'.format(imagename)
    casa.specflux(imagename=imagename,region=region,function=fluxtype,
                  logfile=logfile,overwrite=True)
    if fluxtype == 'flux density':
        ylabel = 'Flux (mJy)'
    else:
        ylabel = 'Flux Density (mJy/beam)'
    #
    # Import spectrum and plot it
    #
    specdata = np.genfromtxt(logfile,comments='#',dtype=None,
                             names=('channel','npts','freq','velocity','flux'))
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
    regions = myplot.get_line_free_regions(specdata_velocity,specdata_flux,
                                           xlabel='Velocity (km/s)',ylabel=ylabel,title=imagename)
    #
    # Extract line free velocity and flux
    #
    line_free_mask = np.zeros(specdata_velocity.size,dtype=bool)
    for region in regions:
        line_free_mask[(specdata_velocity>region[0])&(specdata_velocity<region[1])] = True
    line_free_velocity = specdata_velocity[line_free_mask]
    line_free_flux = specdata_flux[line_free_mask]
    #
    # Fit baseline as polyonimal, iterate to find best fit, plot fit
    #
    rsss = []
    npolys = []
    contfits = []
    for npoly in range(0,11):
        pfit = np.polyfit(line_free_velocity,line_free_flux,npoly)
        contfit = np.poly1d(pfit)
        rss = np.sum((line_free_flux-contfit(line_free_velocity))**2.)
        rss = rss/(len(line_free_velocity)-npoly-1)
        rsss.append(rss)
        npolys.append(npoly)
        contfits.append(contfit)
    ind = np.argmin(rsss)
    npoly = npolys[ind]
    contfit = contfits[ind]
    logger.info("Best baseline fit has polynomial order {0}".format(npoly))
    myplot.plot_contfit(specdata_velocity,specdata_flux,contfit,
                        xlabel='Velocity (km/s)',ylabel=ylabel,
                        title=imagename)
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
    line_start,center_guess,sigma_guess,line_end = \
     myplot.get_gaussian_estimates(specdata_velocity,flux_contsub,
                                   xlabel='Velocity (km/s)',ylabel=ylabel,title=imagename)
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
    outfile='{0}.spec.pdf'.format(imagename)
    myplot.plot_fit(specdata_velocity,flux_contsub,line_brightness,line_center,line_sigma,
                    xlabel='Velocity (km/s)',ylabel=ylabel,title=imagename,
                    outfile=outfile)
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
            line_center, e_line_center, cont_brightness, rms, freq, npoly)

def main(field,region,
         stackedimages=[],stackedlabels=[],stackedfreqs=[],
         fluxtype='flux density',
         lineids=[],linetype='dirty',
         outfile='electron_temps.txt',config_file=None):
    """
   Extract spectrum from region in each RRL image, measure continuum
   brightess and fit Gaussian to measure RRL properties. Also fit stacked 
   line. Compute electron temperature.

    Inputs:
      field       = field to analyze
      region = filename of region where to extract spectrum
      stackedimagelabels = what to call the stacked image data in outfile
      stackedimages = filenames of stacked images
      fluxtype = what type of flux to measure ('flux density' or 'mean')
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
        # 0       1           2      3      4        5        6     7      8        9        10        11          12        13          14
        # lineid  frequency   velo   e_velo line     e_line   fwhm  e_fwhm cont     rms      line2cont e_line2cont elec_temp e_elec_temp npoly
        # #       MHz         km/s   km/s   mJy/beam mJy/beam km/s  km/s   mJy/beam mJy/beam                       K         K           
        # H122a   9494.152594 -100.0 50.0   1000.0   100.0    100.0 10.0   1000.0   100.0    0.050     0.001       10000.0   1000.0      50
        # stacked 9494.152594 -100.0 50.0   1000.0   100.0    100.0 10.0   1000.0   100.0    0.050     0.001       10000.0   1000.0      50
        # 1234567 12345678902 123456 123456 12345678 12345678 12345 123456 12345678 12345678 123456789 12345678901 123456789 12345678901 50
        #
        headerfmt = '{0:12} {1:12} {2:6} {3:6} {4:8} {5:8} {6:5} {7:6} {8:8} {9:8} {10:9} {11:11} {12:9} {13:11} {14:5}\n'
        rowfmt = '{0:12} {1:12.6f} {2:6.1f} {3:6.1f} {4:8.1f} {5:8.1f} {6:5.1f} {7:6.1f} {8:8.1f} {9:8.1f} {10:9.3f} {11:11.3f} {12:9.1f} {13:11.1f} {14:2}\n'
        f.write(headerfmt.format('lineid','frequency','velo','e_velo',
                                 'line','e_line','fwhm','e_fwhm',
                                 'cont','rms','line2cont','e_line2cont',
                                 'elec_temp','e_elec_temp','npoly'))
        if fluxtype == 'fluxdensity':
            fluxunit = 'mJy'
        else:
            fluxunit = 'mJy/beam'
        f.write(headerfmt.format('#','MHz','km/s','km/s',
                                 fluxunit,fluxunit,'km/s','km/s',
                                 fluxunit,fluxunit,'','','K','K',''))
        #
        # Compute electron temperature for each individual RRL
        # 
        for lineid in lineids:
            #
            # Generate file names
            # 
            imagename = '{0}.{1}.channel.{2}.imsmooth'.format(field,lineid,linetype)
            te, e_te, line_to_cont, e_line_to_cont, line_brightness, e_line_brightness, line_fwhm, e_line_fwhm, \
                line_center,e_line_center,cont_brightness, rms, freq, npoly = calc_te(imagename,region,fluxtype)
            f.write(rowfmt.format(lineid, freq,
                                  line_center, e_line_center,
                                  line_brightness, e_line_brightness,
                                  line_fwhm, e_line_fwhm,
                                  cont_brightness, rms,
                                  line_to_cont, e_line_to_cont,
                                  te, e_te, npoly))
        #
        # Compute electron temperature for stacked line images
        #
        for image,label,freq in zip(stackedimages,stackedlabels,stackedfreqs):
            te, e_te, line_to_cont, e_line_to_cont, line_brightness, e_line_brightness, line_fwhm, e_line_fwhm, \
                line_center, e_line_center, cont_brightness, rms, freq, npoly = calc_te(image,region,fluxtype,freq=freq)
            f.write(rowfmt.format(label, freq,
                              line_center, e_line_center,
                              line_brightness, e_line_brightness,
                              line_fwhm, e_line_fwhm,
                              cont_brightness, rms,
                              line_to_cont, e_line_to_cont,
                              te, e_te, npoly))
    #
    # Generate TeX file of all plots
    #
    logger.info("Generating PDF...")
    plots = ['{0}.{1}.channel.{2}.imsmooth.spec.pdf'.format(field,lineid,linetype) for lineid in lineids]
    plots = plots + ['{0}.spec.pdf'.format(image) for image in stackedimages]
    # fix filenames so LaTeX doesn't complain
    plots = ['{'+fn.replace('.pdf','')+'}.pdf' for fn in plots]
    with open('{0}.spectra.tex'.format(field),'w') as f:
        f.write(r"\documentclass{article}"+"\n")
        f.write(r"\usepackage{graphicx}"+"\n")
        f.write(r"\usepackage[margin=0.1cm]{geometry}"+"\n")
        f.write(r"\begin{document}"+"\n")
        for i in range(0,len(plots),6):
            f.write(r"\begin{figure}"+"\n")
            f.write(r"\centering"+"\n")
            f.write(r"\includegraphics[width=0.45\textwidth]{"+plots[i]+"}\n")
            if len(plots) > i+3: f.write(r"\includegraphics[width=0.45\textwidth]{"+plots[i+3]+"}\n")
            f.write(r"\includegraphics[width=0.45\textwidth]{"+plots[i+1]+"}\n")
            if len(plots) > i+4: f.write(r"\includegraphics[width=0.45\textwidth]{"+plots[i+4]+"}\n")
            f.write(r"\includegraphics[width=0.45\textwidth]{"+plots[i+2]+"}\n")
            if len(plots) > i+5: f.write(r"\includegraphics[width=0.45\textwidth]{"+plots[i+5]+"}\n")
            f.write(r"\end{figure}"+"\n")
            f.write(r"\clearpage"+"\n")
        f.write(r"\end{document}")
    os.system('pdflatex -interaction=batchmode {0}.spectra.tex'.format(field))
    logger.info("Done.")
        
