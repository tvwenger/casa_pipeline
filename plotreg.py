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
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.patches import Ellipse

__VERSION__ = "1.0"

def main(images,regions,colors,labels,shrdsfile=None,fluxlimit=0.):
    """
    Plot some regions on top of some images with specified colors,
    and create PDF.

    Inputs:
      images = list of fits image names to plot
      regions = list of region filenames to plot
      colors = what colors to plot each region
      labels = label for regions
      shrdsdata = if not None, path to SHRDS candidates data file
                 This will plot the WISE regions on top
      fluxlimit = only plot WISE regions brighter than this peak
                  continuum flux density (mJy/beam)

    Returns:
      Nothing
    """
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
        cax = ax.imshow(hdu.data[0,0],
                        origin='lower',interpolation='none',cmap='binary',
                        norm=LogNorm(vmin=0.001))
        xlen,ylen = hdu.data[0,0].shape
        ax.coords[0].set_major_formatter('hh:mm:ss')
        ax.set_xlabel('RA (J2000)')
        ax.set_ylabel('Declination (J2000)')
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
        # Plot colorbar
        #
        cbar = fig.colorbar(cax,fraction=0.046,pad=0.04)
        cbar.set_label('Flux Density (Jy/beam)')
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
                            label=lab)
        #
        # Plot WISE regions
        #
        if shrdsfile is not None:
            shrdsdata = np.genfromtxt(shrdsfile,dtype=None,delimiter=',',
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
            min_RA += 0.25 * RA_range
            max_RA -= 0.25 * RA_range
            min_Dec = np.min(corners[:,1])
            max_Dec = np.max(corners[:,1])
            Dec_range = max_Dec - min_Dec
            min_Dec += 0.25 * Dec_range
            max_Dec -= 0.25 * Dec_range
            good = (min_RA < RA)&(RA < max_RA)&(min_Dec < Dec)&(Dec < max_Dec)&(shrdsdata['flux']>fluxlimit)
            # plot them
            shrdsdata = shrdsdata[good]
            RA = RA[good]
            Dec = Dec[good]
            for R,D,dat in zip(RA,Dec,shrdsdata):
                xpos,ypos = wcs_celest.wcs_world2pix(R,D,1)
                size = dat['diameter']/3600./pixsize
                ell = Ellipse((xpos,ypos),size,size,
                               color='m',fill=False,linestyle='dashed')
                ax.add_patch(ell)
                ax.text(R,D,dat['name']+'\n'+dat['GName'],transform=ax.get_transform('world'),fontsize=6)
        #
        # Re-scale to fit, then save
        #
        ax.legend(loc='best',fontsize=10)
        fig.savefig(image.replace('.fits','.reg.pdf'),
                    bbox_inches='tight')
        plt.close(fig)
        plt.ion()
        outimages.append(image.replace('.fits','.reg.pdf'))

    
    
        
