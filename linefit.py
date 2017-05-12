"""
linefit.py
CASA Data Reduction Pipeline - Line fitting script
Trey V. Wenger May 2017 - V1.0
"""

import __main__ as casa # import casa namespace
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

__VERSION__ = "1.0"

def gaussians(x,cont,*p):
    # params is amp1, center1, fwhm1, amp2, center2, fwhm2, etc.
    n_gauss = int(len(p)/3)
    y = np.sum([p[i+0]*np.exp(-(x-p[i+1])**2./(2*(p[i+2]/(2.*np.sqrt(2.*np.log(2.))))**2.))
                for i in np.arange(n_gauss)],axis=0)
    y = y + cont
    return y

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

    def get_gauss_estimates(self,xdata,ydata,xlabel=None,ylabel=None):
        """
        Using click events to get the gaussian fit estimates
        """
        self.ax.clear()
        self.ax.plot(xdata,ydata,'k-')
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.clickbutton = []
        self.clickx_data = []
        self.clicky_data = []
        print "Left click to select start of Gaussian region"
        print "For each Gaussian: Left click to select peak of Gaussian"
        print "For each Gaussian: Left click to select half-maximum of Gaussian"
        print "Left click to select end of Gaussian region"
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
        start = self.clickx_data[0]
        centers = np.array(self.clickx_data[1:-1:2])
        widths = np.array(self.clickx_data[2:-1:2])
        end = self.clickx_data[-1]
        return (start,centers,widths,end)

def fit(imagename,region):
    """
    Fit Gaussian(s) to spectrum extracted from image within a region

    Inputs:
      imagename  = CASA image file to fit
      region = CASA region file

    Returns:
      (peaks, centers, fwhms, continuum, rms)
    """
    stats = casa.imstat(imagename=imagename,region=region,axes=[0,1])
    myplot = ClickPlot(0)
    chans = np.arange(len(stats['max']))
    fluxes =  np.array(stats['max'])
    #
    # Get line-free regions
    #
    regs = myplot.get_line_free_regions(chans,fluxes,
                                        xlabel='channel',
                                        ylabel='peak flux density (Jy/beam)')
    fluxdata = np.array([])
    for idx,reg in enumerate(regs):
        xstart = chans[np.argmin(np.abs(reg[0]-chans))]
        xend = chans[np.argmin(np.abs(reg[1]-chans))]
        fluxdata = np.append(fluxdata,fluxes[xstart:xend])
    rms = np.sqrt(np.mean((fluxdata-np.mean(fluxdata))**2.))
    #
    # Fit-gaussians
    #
    estimates = myplot.get_gauss_estimates(chans,fluxes,
                                           xlabel='channel',
                                           ylabel='peak flux density (Jy/beam)')
    start, centers, widths, end = estimates
    start_ind = np.argmin(np.abs(start-chans))
    end_ind = np.argmin(np.abs(end-chans))
    center_inds = np.array([np.argmin(np.abs(c-chans)) for c in centers])
    width_inds = np.array([np.argmin(np.abs(w-chans)) for w in widths])
    xdata = chans[start_ind:end_ind]
    ydata = fluxes[start_ind:end_ind]
    p0_amps = fluxes[center_inds]
    p0_centers = chans[center_inds]
    p0_fwhms = chans[width_inds]-p0_centers
    p0_cont = np.mean(fluxdata)
    n_gauss = len(p0_amps)
    p0_params = []
    for i in np.arange(n_gauss):
        p0_params += p0_amps[i]
        p0_params += p0_centers[i]
        p0_params += p0_fwhm[i]
    p0 = (p0_cont,p0_params)
    print(p0_params)
    popt, pcov = curve_fit(gaussians,xdata,ydata,p0=p0)
    #
    # Return results
    #
    return (popt[0],popt[1],popt[2],popt[3],rms)
