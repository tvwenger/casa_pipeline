"""
plotreg.py
CASA Data Reduction Pipeline - Plot regions on images
Trey V. Wenger Jan 2018 - V1.0
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerPatch

__VERSION__ = "1.0"

class HandlerEllipse(HandlerPatch):
    """
    This adds the ability to create ellipses within a matplotlib
    legend
    """
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = Ellipse(xy=center, width=height + xdescent,
                    height=height + ydescent, fill=False)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def main(images,regions,colors,labels,shrdsfile=None,fluxlimit=0.,
         wisefile=None,sigma=0.,levels=[5.,10.,20.,50.,100.]):
    """
    Plot some regions on top of some images with specified colors,
    and create PDF.

    Inputs:
      images = list of fits image names to plot
      regions = list of region filenames to plot
      colors = what colors to plot each region
      labels = label for regions
      shrdsfile = if not None, path to SHRDS candidates data file
                 This will plot the SHRDS candidate regions on top
      fluxlimit = only plot WISE regions brighter than this peak
                  continuum flux density (mJy/beam)
      wisefile = if not None, path to WISE positions data file
                 This will plot the WISE regions on top
      sigma = if > 0., will plot colormap with contours at
              levels * sigma
      levels = list of contour levels

    Returns:
      Nothing
    """
    levels = np.array(levels)
    outimages = []
    for image in images:
        #
        # Open fits file, generate WCS
        #
        hdu = fits.open(image)[0]
        wcs = WCS(hdu.header)
        #
        # Generate figure
        #
        plt.ioff()
        fig = plt.figure()
        wcs_celest = wcs.sub(['celestial'])
        ax = plt.subplot(projection=wcs_celest)
        ax.set_title(image.replace('.fits',''))
        # image
        cax = ax.imshow(hdu.data[0,0],origin='lower',
                        interpolation='none',cmap='viridis')
        # contours
        if sigma > 0.:
            con = ax.contour(hdu.data[0,0],origin='lower',
                             levels=levels*sigma,colors='k',linewidths=0.2)
        xlen,ylen = hdu.data[0,0].shape
        ax.coords[0].set_major_formatter('hh:mm:ss')
        ax.set_xlabel('RA (J2000)')
        ax.set_ylabel('Declination (J2000)')
        #
        # Adjust limits
        #
        ax.set_xlim(0.1*xlen,0.9*xlen)
        ax.set_ylim(0.1*ylen,0.9*ylen)
        #
        # Plot colorbar
        #
        cbar = fig.colorbar(cax,fraction=0.046,pad=0.04)
        cbar.set_label('Flux Density (Jy/beam)')
        #
        # Plot beam, if it is defined
        #
        pixsize = hdu.header['CDELT2'] # deg
        if 'BMAJ' in hdu.header.keys():
            beam_maj = hdu.header['BMAJ']/pixsize # pix
            beam_min = hdu.header['BMIN']/pixsize # pix
            beam_pa = hdu.header['BPA']
            ellipse = Ellipse((1./8.*xlen,1./8.*ylen),
                            beam_min,beam_maj,angle=beam_pa,
                            fill=True,zorder=10,hatch='///',
                            edgecolor='black',facecolor='white')
            ax.add_patch(ellipse)
        #
        # Plot regions
        #
        for reg,col,lab in zip(regions,colors,labels):
            if not os.path.exists(reg):
                continue
            # read second line in region file
            with open(reg,'r') as f:
                f.readline()
                data = f.readline()
                # handle point region
                if 'ellipse' in data:
                    splt = data.split(' ')
                    RA = splt[1].replace('[[','').replace(',','')
                    RA_h, RA_m, RA_s = RA.split(':')
                    RA = '{0}h{1}m{2}s'.format(RA_h,RA_m,RA_s)
                    dec = splt[2].replace('],','')
                    dec_d, dec_m, dec_s, dec_ss = dec.split('.')
                    dec = '{0}d{1}m{2}.{3}s'.format(dec_d,dec_m,dec_s,dec_ss)
                    coord = SkyCoord(RA,dec)
                    ax.plot(coord.ra.value,coord.dec.value,'+',color=col,
                            markersize=10,transform=ax.get_transform('world'),
                            label=lab)
                # handle point region
                elif 'poly' in data:
                    splt = data.split('[[')[1]
                    splt = splt.split(']]')[0]
                    parts = splt.split(' ')
                    RAs = []
                    decs = []
                    for ind in range(0,len(parts),2):
                        RA = parts[ind].replace('[','').replace(',','')
                        RA_h, RA_m, RA_s = RA.split(':')
                        RA = '{0}h{1}m{2}s'.format(RA_h,RA_m,RA_s)
                        dec = parts[ind+1].replace('],','')
                        dec_d, dec_m, dec_s, dec_ss = dec.split('.')
                        dec = '{0}d{1}m{2}.{3}s'.format(dec_d,dec_m,dec_s,dec_ss)
                        coord = SkyCoord(RA,dec)
                        RAs.append(coord.ra.value)
                        decs.append(coord.dec.value)
                    RAs.append(RAs[0])
                    decs.append(decs[0])
                    ax.plot(RAs,decs,marker=None,linestyle='solid',color=col,
                            transform=ax.get_transform('world'),
                            label=lab,zorder=110)
        #
        # Add regions legend
        #
        if len(regions) > 0:
            region_legend = plt.legend(loc='upper right',fontsize=10)
            ax.add_artist(region_legend)
        #
        # Plot SHRDS candidate regions
        #
        if shrdsfile is not None:
            shrdsdata = np.genfromtxt(shrdsfile,dtype=None,delimiter=',',encoding='UTF-8',
                                      usecols=(0,1,4,5,6,7),skip_header=1,
                                      names=('name','GName','RA','Dec','diameter','flux'))
            RA = np.zeros(len(shrdsdata))
            Dec = np.zeros(len(shrdsdata))
            for i,dat in enumerate(shrdsdata):
                parts = [float(part) for part in dat['RA'].split(':')]
                RA[i] = 360./24.*(parts[0]+parts[1]/60.+parts[2]/3600.)
                parts = [float(part) for part in dat['Dec'].split(':')]
                Dec[i] = np.abs(parts[0])+parts[1]/60.+parts[2]/3600.
                if '-' in dat['Dec']:
                    Dec[i] = -1.*Dec[i]
            # limit only to regions with centers within image
            corners = wcs_celest.calc_footprint()
            min_RA = np.min(corners[:,0])
            max_RA = np.max(corners[:,0])
            RA_range = max_RA - min_RA
            #min_RA += RA_range
            #max_RA -= RA_range
            min_Dec = np.min(corners[:,1])
            max_Dec = np.max(corners[:,1])
            Dec_range = max_Dec - min_Dec
            #min_Dec += Dec_range
            #max_Dec -= Dec_range
            good = (min_RA < RA)&(RA < max_RA)&(min_Dec < Dec)&(Dec < max_Dec)&(shrdsdata['flux']>fluxlimit)
            # plot them
            shrdsdata = shrdsdata[good]
            RA = RA[good]
            Dec = Dec[good]
            for R,D,dat in zip(RA,Dec,shrdsdata):
                xpos,ypos = wcs_celest.wcs_world2pix(R,D,1)
                size = dat['diameter']/3600./pixsize
                ell = Ellipse((xpos,ypos),size,size,
                               color='m',fill=False,linestyle='dashed',zorder=105)
                ax.add_patch(ell)
                ax.text(R,D,dat['GName'],transform=ax.get_transform('world'),fontsize=10,zorder=105)
        #
        # Plot WISE regions
        #
        if wisefile is not None:
            wisedata = np.genfromtxt(wisefile,dtype=None,names=True,encoding='UTF-8')
            # limit only to regions with centers within image
            corners = wcs_celest.calc_footprint()
            min_RA = np.min(corners[:,0])
            max_RA = np.max(corners[:,0])
            RA_range = max_RA - min_RA
            #min_RA += RA_range
            #max_RA -= RA_range
            min_Dec = np.min(corners[:,1])
            max_Dec = np.max(corners[:,1])
            Dec_range = max_Dec - min_Dec
            #min_Dec += Dec_range
            #max_Dec -= Dec_range
            good = (min_RA < wisedata['RA'])&(wisedata['RA'] < max_RA)&(min_Dec < wisedata['Dec'])&(wisedata['Dec'] < max_Dec)
            # plot them
            wisedata = wisedata[good]
            for dat in wisedata:
                xpos,ypos = wcs_celest.wcs_world2pix(dat['RA'],dat['Dec'],1)
                size = dat['Size']*2./3600./pixsize
                ell = Ellipse((xpos,ypos),size,size,
                               color='y',fill=False,linestyle='dashed',zorder=100)
                ax.add_patch(ell)
                ax.text(dat['RA'],dat['Dec'],dat['GName'],transform=ax.get_transform('world'),fontsize=10,zorder=100)
        #
        # Add WISE+SHRDS legend
        #
        if shrdsfile is not None or wisefile is not None:
            patches = []
            if shrdsfile is not None:
                ell = Ellipse((0,0),0.1,0.1,color='m',fill=False,
                              linestyle='dashed',label='SHRDS Candidates')
                patches.append(ell)
            if wisefile is not None:
                ell = Ellipse((0,0),0.1,0.1,color='y',fill=False,
                              linestyle='dashed',label='WISE Catalog')
                patches.append(ell)
            wise_legend = plt.legend(handles=patches,loc='lower right',fontsize=10,
                                     handler_map={Ellipse: HandlerEllipse()})
            ax.add_artist(wise_legend)
        #
        # Re-scale to fit, then save
        #
        fig.savefig(image.replace('.fits','.reg.pdf'),
                    bbox_inches='tight')
        plt.close(fig)
        plt.ion()
        outimages.append(image.replace('.fits','.reg.pdf'))
