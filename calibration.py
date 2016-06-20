"""
calibration.py
CASA Data Reduction Pipeline - Calibration Script
Trey V. Wenger Jun 2016 - V1.0
"""

import __main__ as casa
import os
import numpy as np
import glob
import re
import time
import pickle
import logging
import logging.config
import ConfigParser

__VERSION__ = "1.0"

# load logging configuration file
logging.config.fileConfig('logging.conf')

def natural_sort(l):
    """
    Natural sort an alphanumeric list

    Inputs:
      l        = alphanumeric list to be sorted

    Returns:
      sorted_l = naturally sorted list
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def get_refant(vis=''):
    """
    Return the best reference antenna to use. For now, just prompt
    the user
    TODO: make this smarter but also efficient

    Inputs:
      vis       = memasurement set

    Returns:
      refant    = the reference antenna
    """
    #casa.plotants(vis=vis)
    refant = raw_input("Refant? ")
    return refant

def setup(vis='',config=None):
    """
    Perform setup tasks: get reference antenna, generate listobs
    file, find line and continuum spectral windows, generate a list
    of primary calibrators, secondary calibrators, flux calibrators,
    and science targets, create needed directories.

    Inputs:
      vis     = measurement set
      config  = ConfigParser object for this project

    Returns:
      (my_cont_spws,my_line_spws,flux_cals,primary_cals,
            secondary_cals,science_targets,refant)
      my_cont_spws    = comma-separated string of continuum spws
      my_line_spws    = comma-separated string of line spws
      flux_cals       = list of flux calibrators
      primary_cals    = list of primary calibrators
      secondary_cals  = list of secondary calibrators
      science_targets = list of science targets
      refant          = reference antenna
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
    # find good referance antenna
    #
    logger.info("Looking for good reference antenna...")
    refant = get_refant(vis=vis)
    if refant is None:
        logger.critical("Error: No good referance antenna found!")
        raise ValueError("No good referance antenna found!")
    logger.info("Done. Found reference antenna: {0}".format(refant))
    #
    # Generate listobs file
    #
    if not os.path.isfile('listobs.txt'):
        logger.info("Generating listobs file...")
        casa.listobs(vis=vis,listfile='listobs.txt')
        logger.info("Done.")
    #
    # Get continuum and line spws from configuration file
    #
    my_cont_spws = config.get("Spectral Windows","Continuum")
    my_line_spws = config.get("Spectral Windows","Line")
    logger.info("Found continuum spws: {0}".format(my_cont_spws))
    logger.info("Found line spws: {0}".format(my_line_spws))
    #
    # get field names
    #
    logger.info("Looking for field names...")
    fields = casa.vishead(vis=vis,mode='get',hdkey='field')[0]
    logger.info("Found fields:")
    logger.info('{0}'.format(fields))
    #
    # Get primary calibrator fields if they are not in config
    #
    if config.get('Calibrators','Primary Calibrators') == '':
        primary_cals = []
        logger.info("Looking for primary calibrators...")
        with open('listobs.txt') as f:
            for line in f:
                for field in fields:
                    if field in line and "CALIBRATE_BANDPASS" in line \
                      and field not in primary_cals:
                        primary_cals.append(field)
        logger.info("Done")
    else:
        primary_cals = [field for field in config.get('Calibrators','Primary Calibrators').splitlines() if field in fields]
    logger.info("Primary calibrators: {0}".format(primary_cals))
    #
    # Get Secondary calibrator fields if they are not in config
    #
    if config.get('Calibrators','Secondary Calibrators') == '':
        secondary_cals = []
        logger.info("Looking for secondary calibrators...")
        with open('listobs.txt') as f:
            for line in f:
                for field in fields:
                    if field in line and ("CALIBRATE_AMPLI" in line or "CALIBRATE_PHASE" in line) \
                      and field not in secondary_cals:
                        secondary_cals.append(field)
        logger.info("Done")
    else:
        secondary_cals = [field for field in config.get('Calibrators','Secondary Calibrators').splitlines() if field in fields]
    logger.info("Secondary calibrators: {0}".format(secondary_cals))
    #
    # Get flux calibrator fields if they are not in config
    #
    if config.get('Calibrators','Flux Calibrators') == '':
        flux_cals = []
        logger.info("Looking for flux calibrators...")
        with open('listobs.txt') as f:
            for line in f:
                for field in fields:
                    if field in line and "CALIBRATE_FLUX" in line \
                      and field not in flux_cals:
                        flux_cals.append(field)
        logger.info("Done")
    else:
        flux_cals = [field for field in config.get('Calibrators','Flux Calibrators').splitlines() if field in fields]
    logger.info("Flux calibrators: {0}".format(flux_cals))
    #
    # Check that flux calibrators are in primary calibrator list
    # if not, add them
    #
    for flux_cal in flux_cals:
        if flux_cal not in primary_cals:
            primary_cals.append(flux_cal)
    #
    # Get science targets
    # 
    science_targets = []
    logger.info("Looking for science targets...")
    for field in fields:
        if field not in primary_cals+secondary_cals:
            science_targets.append(field)
    logger.info("Done")
    logger.info("Science targets: {0}".format(science_targets))
    #
    # create directories for figures
    #
    if not os.path.isdir('calib_figures'):
        logger.info("Creating calib_figures directory...")
        os.makedirs('calib_figures')
        logger.info("Done.")
    if not os.path.isdir('scitarg_figures'):
        logger.info("Creating scitarg_figures directory...")
        os.makedirs('scitarg_figures')
        logger.info("Done.")
    return (my_cont_spws,my_line_spws,flux_cals,primary_cals,
            secondary_cals,science_targets,refant)

def preliminary_flagging(vis='',my_line_spws='',my_cont_spws='',config=None):
    """
    Perform preliminary flagging: shadowed antennas, quack,
    flags from configuration file, then tfcrop all raw data

    Inputs:
      vis          = measurement set
      my_line_spws = comma-separated string of line spws
      my_cont_spws = comma-separated string of continuum spws
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
    # save initial flag state
    #
    logger.info("Saving flag state...")
    casa.flagmanager(vis=vis,mode='save',
                     versionname='starting_flags_{0}'.format(time.strftime('%Y%m%d%H%M%S',time.gmtime())))
    logger.info("Done.")
    #
    # Flag shadowed antennas
    #
    logger.info("Flagging shadowed antennas...")
    casa.flagdata(vis=vis,mode='shadow',tolerance=-3.0,
                  flagbackup=False,extendflags=False)
    logger.info("Done.")
    #
    # Flag the beginning of each scan
    #
    logger.info("Flagging the beginning of each scan (quack)...")
    casa.flagdata(vis=vis,mode='quack',quackinterval=6,
                  flagbackup=False,extendflags=False)
    logger.info("Done.")
    #
    # Flag antennas from configuration file
    #
    antenna = config.get("Flags","Antenna")
    if antenna != '':
        logger.info("Flagging antennas from configuration file: {0}".format(antenna))
        casa.flagdata(vis=vis,mode='manual',antenna=antenna,
                      flagbackup=False,extendflags=False)
        logger.info("Done.")
    #
    # Flag line channels from configuration file
    #
    badchans = config.get("Flags","Line Channels").split(',')
    if badchans[0] != '':
        logger.info("Flagging line channels from configuration file: {0}".format(badchans))
        badchans = ';'.join(badchans)
        line_spws = ','.join([i+':'+badchans for i in my_line_spws.split(',')])
        casa.flagdata(vis=vis,mode='manual',spw=line_spws,
                      flagbackup=False,extendflags=False)
        logger.info("Done.")
    #
    # Flag continuum channels from configuration file
    #
    badchans = config.get("Flags","Continuum Channels").split(',')
    if badchans[0] != '':
        logger.info("Flagging continuum channels from configuration file: {0}".format(badchans))
        badchans = ';'.join(badchans)
        cont_spws = ','.join([i+':'+badchans for i in my_cont_spws.split(',')])
        casa.flagdata(vis=vis,mode='manual',spw=cont_spws,
                      flagbackup=False,extendflags=False)
        logger.info("Done.")
    #
    # Run tfcrop on all fields
    #
    logger.info("Running tfcrop on raw data column...")
    casa.flagdata(vis=vis,mode='tfcrop',
                  timefit='poly',freqfit='poly',
                  flagbackup=False,datacolumn='data',
                  extendflags=False)
    logger.info("Done.")
    #
    # Save the flags
    #
    logger.info("Saving flag state...")
    casa.flagmanager(vis=vis,mode='save',
                     versionname='preliminary_{0}'.format(time.strftime('%Y%m%d%H%M%S',time.gmtime())))
    logger.info("Done.")

def auto_flag_calibrators(vis='',primary_cals=[],secondary_cals=[]):
    """
    Perform automatic flagging of calibrators using rflag on
    calibrated data or tfcrop on raw data

    Inputs:
      vis            = measurement set
      primary_cals   = list of primary calibrators (must include flux cals)
      secondary_cals = list of secondary calibrators

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # check if calibrators have corrected datacolumn
    #
    field = ','.join(primary_cals+secondary_cals)
    stat = None
    logger.info("Checking if ms contains corrected data column...")
    stat = casa.visstat(vis=vis,field=field,spw='0',datacolumn='corrected')
    if stat is None:
        logger.info("Done. ms does not contain corrected data column.")
        datacolumn='data'
    else:
        logger.info("Done. ms does contain corrected data column.")
        datacolumn='corrected'
    #
    # Run rflag on calibrated data
    #
    if datacolumn == 'corrected':
        logger.info("Running rflag on corrected data column...")
        casa.flagdata(vis=vis,mode='rflag',field=field,
                      flagbackup=False,datacolumn=datacolumn,
                      extendflags=False)
        logger.info("Done.")
    #
    # Run tfcrop on uncalibrated data
    #
    else:
        logger.info("Running tfcrop on raw data column...")
        casa.flagdata(vis=vis,mode='tfcrop',field=field,
                      timefit='poly',freqfit='poly',
                      flagbackup=False,datacolumn='data',
                      extendflags=False)
        logger.info("Done.")
    #
    # Save the flags
    #
    logger.info("Saving flag state...")
    casa.flagmanager(vis=vis,mode='save',
                     versionname='autoflag_{0}'.format(time.strftime('%Y%m%d%H%M%S',time.gmtime())))
    logger.info("Done.")

def gen_calibrator_plots(vis='',primary_cals=[],secondary_cals=[],
                         config=None):
    """
    Generate visibility plots for calibrators

    Inputs:
      vis            = measurement set
      primary_cals   = list of primary calibrators (must include flux cals)
      secondary_cals = list of secondary calibrators

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
    # check if calibrators have corrected datacolumn
    #
    field = ','.join(primary_cals+secondary_cals)
    stat = None
    logger.info("Checking if ms contains corrected data column...")
    stat = casa.visstat(vis=vis,spw='0',field=field,datacolumn='corrected')
    if stat is None:
        logger.info("Done. ms does not contain corrected data column.")
        datacolumn='data'
    else:
        logger.info("Done. ms does contain corrected data column.")
        datacolumn='corrected'
    #
    # Generate the plots
    #
    logger.info("Generating plots for manual inspection...")
    plotnum=0
    plots = []
    for field in primary_cals+secondary_cals:
        #
        # Amplitude vs UV-distance (in wavelength units)
        #
        casa.plotms(vis=vis,xaxis='uvwave',yaxis='amp',field=field,
                    ydatacolumn=datacolumn,iteraxis='spw',
                    coloraxis='baseline',correlation=config.get('Polarization','Polarization'),
                    title='PlotID: {0} Field: {1}'.format(plotnum,field),
                    plotfile='calib_figures/{0}.png'.format(plotnum),
                    overwrite=True,showgui=False,exprange='all')
        plots.append({'field':field,'xaxis':'uvwave','yaxis':'amp','avgtime':'','avgchannel':''})
        plotnum += 1
         #
        # Amplitude vs Time
        #
        casa.plotms(vis=vis,xaxis='time',yaxis='amp',field=field,
                    ydatacolumn=datacolumn,iteraxis='spw',
                    coloraxis='baseline',correlation=config.get('Polarization','Polarization'),
                    title='PlotID: {0} Field: {1}'.format(plotnum,field),
                    avgchannel='1e7',
                    plotfile='calib_figures/{0}.png'.format(plotnum),
                    overwrite=True,showgui=False,exprange='all')
        plots.append({'field':field,'xaxis':'time','yaxis':'amp','avgtime':'','avgchannel':'1e7'})
        plotnum += 1
        #
        # Phase vs Time
        #
        casa.plotms(vis=vis,xaxis='time',yaxis='phase',field=field,
                    ydatacolumn=datacolumn,iteraxis='spw',
                    coloraxis='baseline',correlation=config.get('Polarization','Polarization'),
                    title='PlotID: {0} Field: {1}'.format(plotnum,field),
                    avgchannel='1e7',
                    plotfile='calib_figures/{0}.png'.format(plotnum),
                    overwrite=True,showgui=False,exprange='all')
        plots.append({'field':field,'xaxis':'time','yaxis':'phase','avgtime':'','avgchannel':'1e7'})
        plotnum += 1
        #
        # Amplitude vs Channel
        #
        casa.plotms(vis=vis,xaxis='channel',yaxis='amp',field=field,
                    ydatacolumn=datacolumn,iteraxis='spw',
                    coloraxis='baseline',correlation=config.get('Polarization','Polarization'),
                    title='PlotID: {0} Field: {1}'.format(plotnum,field),
                    avgtime='1e7',
                    plotfile='calib_figures/{0}.png'.format(plotnum),
                    overwrite=True,showgui=False,exprange='all')
        plots.append({'field':field,'xaxis':'channel','yaxis':'amp','avgtime':'1e7','avgchannel':''})
        plotnum += 1
        #
        # Phase vs Channel
        # 
        casa.plotms(vis=vis,xaxis='channel',yaxis='phase',field=field,
                    ydatacolumn=datacolumn,iteraxis='spw',
                    coloraxis='baseline',correlation=config.get('Polarization','Polarization'),
                    title='PlotID: {0} Field: {1}'.format(plotnum,field),
                    avgtime='1e7',
                    plotfile='calib_figures/{0}.png'.format(plotnum),
                    overwrite=True,showgui=False,exprange='all')
        plots.append({'field':field,'xaxis':'channel','yaxis':'phase','avgtime':'1e7','avgchannel':''})
        plotnum += 1
    logger.info("Done.")
    #
    # Generate PDF to display plots
    #
    logger.info("Generating PDF...")
    num_plots = plotnum
    iplot = 0
    with open('calibrator_plots.tex','w') as f:
        f.write(r"\documentclass{article}"+"\n")
        f.write(r"\usepackage{graphicx}"+"\n")
        f.write(r"\usepackage[margin=0.1cm]{geometry}"+"\n")
        f.write(r"\begin{document}"+"\n")
        f.write(r"\begin{figure}"+"\n")
        f.write(r"\centering"+"\n")
        for plotnum in range(num_plots):
            fnames=glob.glob("calib_figures/{0}_*.png".format(plotnum))
            fnames = natural_sort(fnames)
            for fname in fnames:
                if iplot > 0 and iplot % 6 == 0:
                    f.write(r"\end{figure}"+"\n")
                    f.write(r"\clearpage"+"\n")
                    f.write(r"\begin{figure}"+"\n")
                    f.write(r"\centering"+"\n")
                elif iplot > 0 and iplot % 2 == 0:
                    f.write(r"\end{figure}"+"\n")
                    f.write(r"\begin{figure}"+"\n")
                    f.write(r"\centering"+"\n")
                f.write(r"\includegraphics[width=0.45\textwidth]{"+fname+"}\n")
                iplot+=1
        f.write(r"\end{figure}"+"\n")
        f.write(r"\end{document}"+"\n")
    os.system('pdflatex -interaction=batchmode calibrator_plots.tex')
    logger.info("Done.")
    #
    # Save plot list to a pickle object
    #
    logger.info("Saving plot list to pickle...")
    with open('calibrator_plots.pkl','w') as f:
        pickle.dump(plots,f)
    logger.info("Done.")

def flag(vis='',all_fields=[]):
    """
    Interactively flag

    Inputs:
      vis            = measurement set
      all_fields     = list of fields currently being flagged
                       (i.e. list of calibrators if we are flagging
                        calibrators)

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Build list of flag commands
    #
    flag_commands = []
    while True:
        #
        # Prompt user for field, scan, spectral window, time range,
        # antenna, and correlation to flag
        #
        print("Field? Empty = {0}".format(','.join(all_fields)))
        field = raw_input()
        if field == '':
            field = ','.join(all_fields)
        if field not in ','.join(all_fields):
            print('{0} is not in {1}'.format(field,all_fields))
            continue
        print("Scan? Empty = all scans (ex. 0 to flag scan 0, 1~3 to flag scans 1, 2, and 3)")
        scan = raw_input()
        print("Spectral window and channels? Empty = all spws (ex. 2:100~150;200:250,5:1200~1210 is spw 2, chans 100 to 150  and 200 to 250, and spw 5 chans 1200 to 1210)")
        spw = raw_input()
        print("Time range? Empty = all times (ex. 10:23:45~10:23:55)")
        timerange = raw_input()
        print("Antenna or baseline? Empty = all antennas/baselines (ex. CA01 to flag ant 1 or CA01&CA02 to flag 1-2 baseline)")
        antenna = raw_input()
        print("Correlation? Empty = all correlations (XX,YY,XY,YX)")
        correlation = raw_input()
        #
        # Build flag command
        #
        parts = []
        if len(field) > 0:
            parts.append("field='{0}'".format(field))
        if len(scan) > 0:
            parts.append("scan='{0}'".format(scan))
        if len(spw) > 0:
            parts.append("spw='{0}'".format(spw))
        if len(timerange) > 0:
            parts.append("timerange='{0}'".format(timerange))
        if len(antenna) > 0:
            parts.append("antenna='{0}'".format(antenna))
        if len(correlation) > 0:
            parts.append("correlation='{0}'".format(correlation))
        flag_commands.append(' '.join(parts))
        #
        # Confirm with user, or append more flag commands
        #
        print("Will execute:")
        print("flagdata(vis='{0}',mode='list',flagbackup=False,extendflags=False,".format(vis))
        for icmd,cmd in enumerate(flag_commands):
            if len(flag_commands) == 1:
                print("         inpfile=[\"{0}\"])".format(cmd))
            elif icmd == 0:
                print("         inpfile=[\"{0}\",".format(cmd))
            elif icmd == len(flag_commands)-1:
                print("                  \"{0}\"])".format(cmd))
            else:
                print("                  \"{0}\",".format(cmd))
        print("Proceed [y/n] or add another flag command [a]?")
        go = raw_input()
        #
        # Execute flag command
        #
        if go.lower() == 'y':
            logger.info("Executing:")
            logger.info("flagdata(vis='{0}',mode='list',flagbackup=False,extendflags=False,".format(vis))
            for icmd,cmd in enumerate(flag_commands):
                if len(flag_commands) == 1:
                    logger.info("         inpfile=[\"{0}\"])".format(cmd))
                elif icmd == 0:
                    logger.info("         inpfile=[\"{0}\",".format(cmd))
                elif icmd == len(flag_commands)-1:
                    logger.info("                  \"{0}\"])".format(cmd))
                else:
                    logger.info("                  \"{0}\",".format(cmd))
            #
            # Save flag command to manual flags list
            #
            with open('manual_flags.txt','a') as f:
                for cmd in flag_commands:
                    f.write(time.strftime('%Y%m%d%H%M%S',time.gmtime())+': '+cmd+'\n')
            casa.flagdata(vis=vis,mode='list',flagbackup=False,extendflags=False,
                          inpfile=flag_commands)
            break
        #
        # Append another flag command
        #
        elif go.lower() == 'a':
            continue
        #
        # Quit flagging
        #
        else:
            print("Aborting...")
            break

def manual_flag_calibrators(vis='',primary_cals=[],secondary_cals=[],
                            config=None):
    """
    Interactively plot and flag the calibrators

    Inputs:
      vis            = measurement set
      primary_cals   = list of primary calibrators (must include flux cals)
      secondary_cals = list of secondary calibrators

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
    # check if calibrators have corrected datacolumn
    #
    field = ','.join(primary_cals+secondary_cals)
    stat = None
    logger.info("Checking if ms contains corrected data column...")
    stat = casa.visstat(vis=vis,spw='0',field=field,datacolumn='corrected')
    if stat is None:
        logger.info("Done. ms does not contain corrected data column.")
        datacolumn='data'
    else:
        logger.info("Done. ms does contain corrected data column.")
        datacolumn='corrected'
    #
    # Read the plot list from the pickle object
    #
    logger.info("Reading plot list from pickle...")
    with open('calibrator_plots.pkl','r') as f:
        plots = pickle.load(f)
    num_plots = len(plots)
    logger.info("Done.")
    #
    # Display menu option to user
    #
    logger.info("Please inspect calibrator_plots.pdf then perform manual calibrations.")
    while True:
        print("f - flag some data")
        print("plot id number - generate interactive version of plot with this id")
        print("quit - end this flagging session")
        answer = raw_input()
        #
        # Flag some data
        #
        if answer.lower() == 'f':
            flag(vis,all_fields=primary_cals+secondary_cals)
        #
        # Stop interactively plotting and flagging
        #
        elif answer.lower() == 'quit':
            break
        #
        # Generate interactive plot
        #
        else:
            try:
                plotid = int(answer)
            except ValueError:
                print("Plot ID not valid!")
                continue
            if plotid >= num_plots:
                print("Plot ID not valid!")
                continue
            casa.plotms(vis=vis,xaxis=plots[plotid]['xaxis'],yaxis=plots[plotid]['yaxis'],field=plots[plotid]['field'],
                        ydatacolumn=datacolumn,iteraxis='spw',
                        coloraxis='baseline',correlation=config.get('Polarization','Polarization'),
                        title='PlotID: {0} Field: {1}'.format(plotid,plots[plotid]['field']),
                        avgchannel=plots[plotid]['avgchannel'],avgtime=plots[plotid]['avgtime'])
    #
    # Save the flags
    #
    logger.info("Saving flag state...")
    casa.flagmanager(vis=vis,mode='save',
                     versionname='manualflag_{0}'.format(time.strftime('%Y%m%d%H%M%S',time.gmtime())))
    logger.info("Done.")

def calibrate_calibrators(vis='',primary_cals=[],secondary_cals=[],
                          flux_cals=[],my_line_spws='',
                          my_cont_spws='',refant='',config=None):
    """
    Calculate calibration solutions (bandpass, delays, complex gains)
    and apply the calibration solutions to the calibrators

    Inputs:
      vis            = measurement set
      primary_cals   = list of primary calibrators (must include flux cals)
      secondary_cals = list of secondary calibrators
      flux_cals      = list of flux calibrators
      my_line_spws   = comma-separated string of line spws
      my_cont_spws   = comma-separated string of continuum spws
      refant         = reference antenna
      config         = ConfigParser object for this project

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
    # set the model for the flux calibrators
    #
    logger.info("Setting the flux calibrator models...")
    for flux_cal in flux_cals:
        #
        # If flux calibrator model is supplied in config, use that
        #
        manual_flux_cals = config.get('Flux Calibrator Models','Name').splitlines()
        if flux_cal in manual_flux_cals:
            # get index of flux_cal in manual_flux_cals
            flux_idx = manual_flux_cals.index(flux_cal)
            # get reference frequency
            reffreq = config.get('Flux Calibrator Models','Reference Frequency').splitlines()
            reffreq = reffreq[flux_idx]
            # get fluxdensity and convert to proper units
            fluxdensity = config.get('Flux Calibrator Models','Log Flux Density').splitlines()
            fluxdensity = [10.**float(fluxdensity[flux_idx]),0.,0.,0.]
            # get spectral index coefficients
            spix = config.get('Flux Calibrator Models','Spectral Index Coefficients').splitlines()
            spix = spix[flux_idx]
            spix = [float(i) for i in spix.split(',')]
            # Run setjy in manual mode
            casa.setjy(vis=vis,field=flux_cal,scalebychan=True,
                       standard='manual',fluxdensity=fluxdensity,
                       spix=spix,reffreq=reffreq)
        #
        # Otherwise, use CASA model
        #
        else:
            casa.setjy(vis=vis,field=flux_cal,scalebychan=True)
    logger.info("Done.")
    #
    # pre-bandpass calibration delay calibration on primary calibrators
    # (linear slope in phase vs frequency)
    #
    field = ','.join(primary_cals)
    logger.info("Calculating delay calibration table for primary calibrators...")
    if os.path.isdir('delays.cal'):
        casa.rmtables('delays.cal')
    casa.gaincal(vis=vis,caltable='delays.cal',field=field,
                 refant=refant,gaintype='K',minblperant=1)
    if not os.path.isdir('delays.cal'):
        logger.critical('Problem with delay calibration')
        raise ValueError('Problem with delay calibration!')
    logger.info("Done.")
    #
    # integration timescale phase calibration
    # (phase vs time)
    #
    logger.info("Calculating phase calibration table on integration timescales for primary calibrators...")
    if os.path.isdir('phase_int.cal'):
        casa.rmtables('phase_int.cal')
    casa.gaincal(vis=vis,caltable="phase_int.cal",field=field,
                 solint="int",calmode="p",refant=refant,
                 gaintype="G",minsnr=2.0,minblperant=1,
                 gaintable=['delays.cal'])
    if not os.path.isdir('phase_int.cal'):
        logger.critical('Problem with integration-timescale phase calibration')
        raise ValueError('Problem with integration-timescale phase calibration!')
    logger.info("Done.")
    #
    # bandpass calibration for continuum spws. Combine all scans,
    # average some channels as defined in configuration file
    #
    logger.info("Calculating bandpass calibration table for primary calibrators...")
    if os.path.isdir('bandpass.cal'):
        casa.rmtables('bandpass.cal')
    chan_avg = config.get('Calibration','Continuum Channels')
    if chan_avg == '':
        solint='inf'
    else:
        solint='inf,{0}chan'.format(chan_avg)
    casa.bandpass(vis=vis,caltable='bandpass.cal',field=field,
                  spw=my_cont_spws,refant=refant,solint=solint,
                  combine='scan',solnorm=True,minblperant=1,
                  gaintable=['delays.cal','phase_int.cal'])
    #
    # bandpass calibration for line spws. Combine all scans,
    # average some channels as defined in configuration file,
    # append to continuum channel bandpass calibration table
    #
    chan_avg = config.get('Calibration','Line Channels')
    if chan_avg == '':
        solint='inf'
    else:
        solint='inf,{0}chan'.format(chan_avg)
    casa.bandpass(vis=vis,caltable='bandpass.cal',field=field,
                  spw=my_line_spws,refant=refant,solint=solint,
                  combine='scan',solnorm=True,minblperant=1,append=True,
                  gaintable=['delays.cal','phase_int.cal'])
    if not os.path.isdir('bandpass.cal'):
        logger.critical('Problem with bandpass calibration')
        raise ValueError('Problem with bandpass calibration!')
    logger.info("Done.")
    #
    # integration timescale phase corrections for all calibrators
    # required for accurate amplitude calibration
    #
    field = ','.join(primary_cals+secondary_cals)
    logger.info("Re-calculating the phase calibration table on integration timescales for all calibrators...")
    if os.path.isdir('phase_int.cal'):
        casa.rmtables('phase_int.cal')
    casa.gaincal(vis=vis,caltable="phase_int.cal",field=field,
                solint="int",calmode="p",refant=refant,
                gaintype="G",minsnr=2.0,minblperant=1,
                gaintable=['delays.cal','bandpass.cal'])
    if not os.path.isdir('phase_int.cal'):
        logger.critical('Problem with integration-timescale phase calibration')
        raise ValueError('Problem with integration-timescale phase calibration!')
    logger.info("Done.")
    #
    # scan timescale phase corrections for all calibrators
    # required to apply to science targets
    #
    logger.info("Calculating the phase calibration table on scan timescales for all calibrators...")
    if os.path.isdir('phase_scan.cal'):
        casa.rmtables('phase_scan.cal')
    casa.gaincal(vis=vis,caltable="phase_scan.cal",field=field,
                 solint="inf",calmode="p",refant=refant,
                 gaintype="G",minsnr=2.0,minblperant=1,
                 gaintable=['delays.cal','bandpass.cal'])
    if not os.path.isdir('phase_scan.cal'):
        logger.critical('Problem with scan-timescale phase calibration')
        raise ValueError('Problem with scan-timescale phase calibration!')
    logger.info("Done.")
    #
    # scan timescale amplitude corrections using
    # integration timescale phase calibration
    #
    logger.info("Calculating the amplitude calibration table on scan timescales for all calibrators...")
    if os.path.isdir('apcal.cal'):
        casa.rmtables('apcal.cal')
    casa.gaincal(vis=vis,caltable="apcal.cal",field=field,
                 solint="inf",calmode="ap",refant=refant,
                 minsnr=2.0,minblperant=1,
                 gaintable=['delays.cal','bandpass.cal','phase_int.cal'])
    if not os.path.isdir('apcal.cal'):
        logger.critical('Problem with amplitude calibration')
        raise ValueError('Problem with amplitude calibration!')
    logger.info("Done.")
    #
    # set the flux scale
    #
    logger.info("Calculating the flux calibration table...")
    if os.path.isdir('flux.cal'):
        casa.rmtables('flux.cal')
    casa.fluxscale(vis=vis,caltable="apcal.cal",fluxtable="flux.cal",
                   reference=','.join(flux_cals),incremental=True)
    if not os.path.isdir('flux.cal'):
        logger.critical('Problem with flux calibration')
        raise ValueError('Problem with flux calibration!')
    logger.info("Done.")
    #
    # apply calibration solutions to calibrators
    #
    logger.info("Applying calibration tables to all calibrators...")
    for field in primary_cals+secondary_cals:
        casa.applycal(vis=vis,field=field,calwt=False,
                      gaintable=['delays.cal','bandpass.cal',
                                 'phase_int.cal','apcal.cal',
                                 'flux.cal'],
                      gainfield=['','',field,field,field],
                      flagbackup=False)
    logger.info("Done.")
    #
    # save flag state
    #
    casa.flagmanager(vis=vis,mode='save',
                     versionname='calibrate_{0}'.format(time.strftime('%Y%m%d%H%M%S',time.gmtime())))

def calibrate_sciencetargets(vis='',science_targets=[]):
    """
    Apply calibration solutions to science targets

    Inputs:
      vis             = measurement set
      science_targets = list of science targets

    Returns:
      Nothing    
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    for field in science_targets:
        #
        # use all fields in delays and bandpass
        # use nearest field in complex gains and flux
        #
        logger.info("Applying calibration solutions to {0}".format(field))
        casa.applycal(vis=vis,field=field,calwt=False,
                      gaintable=['delays.cal','bandpass.cal',
                                 'phase_int.cal','apcal.cal',
                                 'flux.cal'],
                      gainfield=['','','nearest','nearest','nearest'],
                      flagbackup=False)
    #
    # save the flags
    #
    logger.info("Saving flag state...")
    casa.flagmanager(vis=vis,mode='save',
                     versionname='calibrate_{0}'.format(time.strftime('%Y%m%d%H%M%S',time.gmtime())))
    logger.info("Done.")

def auto_flag_sciencetargets(vis='',science_targets=[]):
    """
    Perform automatic flagging of calibrated science targets using
    rflag

    Inputs:
      vis             = measurement set
      science_targets = list of science targets

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Perform automatic flagging on science targets
    #
    field = ','.join(science_targets)
    datacolumn='corrected'
    logger.info("Running rflag on all correlations...")
    casa.flagdata(vis=vis,mode='rflag',field=field,
                  flagbackup=False,datacolumn=datacolumn,
                  extendflags=False)
    logger.info("Done.")
    #
    # Save the flags
    #
    logger.info("Saving flag state...")
    casa.flagmanager(vis=vis,mode='save',
                     versionname='autoflag_{0}'.format(time.strftime('%Y%m%d%H%M%S',time.gmtime())))
    logger.info("Done.")

def gen_sciencetarget_plots(vis='',science_targets=[],config=None):
    """
    Generate science target visibility plots

    Inputs:
      vis             = measurement set
      science_targets = list of science targets

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    datacolumn='corrected'
    #
    # check config
    #
    if config is None:
        logger.critical("Error: Need to supply a config")
        raise ValueError("Config is None")
    #
    # Generate the plots
    #
    logger.info("Generating plots for manual inspection...")
    plotnum=0
    plots = []
    for field in science_targets:
        #
        # Amplitude vs UV-distance (in wavelength units)
        #
        casa.plotms(vis=vis,xaxis='uvwave',yaxis='amp',field=field,
                    ydatacolumn=datacolumn,iteraxis='spw',
                    coloraxis='baseline',correlation=config.get('Polarization','Polarization'),
                    title='PlotID: {0} Field: {1}'.format(plotnum,field),
                    plotfile='scitarg_figures/{0}.png'.format(plotnum),
                    overwrite=True,showgui=False,exprange='all')
        plots.append({'field':field,'xaxis':'uvwave','yaxis':'amp','avgtime':'','avgchannel':''})
        plotnum += 1
        #
        # Amplitude vs Time
        #
        casa.plotms(vis=vis,xaxis='time',yaxis='amp',field=field,
                    ydatacolumn=datacolumn,iteraxis='spw',
                    coloraxis='baseline',correlation=config.get('Polarization','Polarization'),
                    title='PlotID: {0} Field: {1}'.format(plotnum,field),
                    avgchannel='1e7',
                    plotfile='scitarg_figures/{0}.png'.format(plotnum),
                    overwrite=True,showgui=False,exprange='all')
        plots.append({'field':field,'xaxis':'time','yaxis':'amp','avgtime':'','avgchannel':'1e7'})
        plotnum += 1
        #
        # Amplitude vs Channel
        #
        casa.plotms(vis=vis,xaxis='channel',yaxis='amp',field=field,
                    ydatacolumn=datacolumn,iteraxis='spw',
                    coloraxis='baseline',correlation=config.get('Polarization','Polarization'),
                    title='PlotID: {0} Field: {1}'.format(plotnum,field),
                    avgtime='1e7',
                    plotfile='scitarg_figures/{0}.png'.format(plotnum),
                    overwrite=True,showgui=False,exprange='all')
        plots.append({'field':field,'xaxis':'channel','yaxis':'amp','avgtime':'1e7','avgchannel':''})
        plotnum += 1
    logger.info("Done.")
    #
    # Generate PDF to display plots
    #
    logger.info("Generating tex document...")
    num_plots = plotnum
    iplot = 0
    with open('scitarg_plots.tex','w') as f:
        f.write(r"\documentclass{article}"+"\n")
        f.write(r"\usepackage{graphicx}"+"\n")
        f.write(r"\usepackage[margin=0.1cm]{geometry}"+"\n")
        f.write(r"\begin{document}"+"\n")
        f.write(r"\begin{figure}"+"\n")
        f.write(r"\centering"+"\n")
        for plotnum in range(num_plots):
            fnames=glob.glob("scitarg_figures/{0}_*.png".format(plotnum))
            fnames = natural_sort(fnames)
            for fname in fnames:
                if iplot > 0 and iplot % 6 == 0:
                    f.write(r"\end{figure}"+"\n")
                    f.write(r"\clearpage"+"\n")
                    f.write(r"\begin{figure}"+"\n")
                    f.write(r"\centering"+"\n")
                elif iplot > 0 and iplot % 2 == 0:
                    f.write(r"\end{figure}"+"\n")
                    f.write(r"\begin{figure}"+"\n")
                    f.write(r"\centering"+"\n")
                f.write(r"\includegraphics[width=0.45\textwidth]{"+fname+"}\n")
                iplot+=1
        f.write(r"\end{figure}"+"\n")
        f.write(r"\end{document}"+"\n")
    os.system('pdflatex -interaction=batchmode scitarg_plots.tex')
    logger.info("Done.")
    #
    # Save the plot list to the pickle object
    #
    logger.info("Saving plot list to pickle...")
    with open('scitarg_plots.pkl','w') as f:
        pickle.dump(plots,f)
    logger.info("Done.")

def manual_flag_sciencetargets(vis='',science_targets=[],config=None):
    """
    Interactively plot and flag the science targets

    Inputs:
      vis             = measurement set
      science_targets = list of science targets

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    datacolumn='corrected'
    #
    # check config
    #
    if config is None:
        logger.critical("Error: Need to supply a config")
        raise ValueError("Config is None")
    #
    # Read plot list from pickle object
    #
    logger.info("Reading plot list from pickle...")
    with open("scitarg_plots.pkl","r") as f:
        plots = pickle.load(f)
    num_plots = len(plots)
    logger.info("Done.")
    #
    # Prompt user with menu
    #
    logger.info("Please inspect scitarg_plots.pdf then perform manual calibrations.")
    while True:
        print("f - flag some data")
        print("plot id number - generate interactive version of plot with this id")
        print("quit - end this flagging session")
        answer = raw_input()
        #
        # Flag some data
        #
        if answer.lower() == 'f':
            flag(vis=vis,all_fields=science_targets)
        #
        # Stop flagging
        #
        elif answer.lower() == 'quit':
            break
        #
        # Generate plotms figure
        #
        else:
            try:
                plotid = int(answer)
            except ValueError:
                print("Plot ID not valid!")
                continue
            if plotid >= num_plots:
                print("Plot ID not valid!")
                continue
            casa.plotms(vis=vis,xaxis=plots[plotid]['xaxis'],yaxis=plots[plotid]['yaxis'],field=plots[plotid]['field'],
                        ydatacolumn=datacolumn,iteraxis='spw',
                        coloraxis='baseline',correlation=config.get('Polarization','Polarization'),
                        title='PlotID: {0} Field: {1}'.format(plotid,plots[plotid]['field']),
                        avgchannel=plots[plotid]['avgchannel'],avgtime=plots[plotid]['avgtime'])
    #
    # Save the flags
    #
    logger.info("Saving flag state...")
    casa.flagmanager(vis=vis,mode='save',
                     versionname='manualflag_{0}'.format(time.strftime('%Y%m%d%H%M%S',time.gmtime())))
    logger.info("Done.")

def split_fields(vis='',primary_cals=[],secondary_cals=[],science_targets=[]):
    """
    Split calibrated fields into own measurement sets with naming format:
    {field_name}_calibrated.ms

    Inputs:
      vis             = measurement set
      primary_cals    = list of primary calibrators
      secondary_cals  = list of secondary calibrators
      science_targets = list of science targets

    Returns:
      Nothing
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    logger.info("Splitting fields...")
    for field in primary_cals+secondary_cals+science_targets:
        outputvis = '{0}_calibrated.ms'.format(field)
        logger.info("Splitting {0} to {1}".format(field,outputvis))
        casa.split(vis=vis,outputvis=outputvis,field=field,keepflags=False)
    logger.info("Done!")

def main(vis='',config_file='',auto=False):
    """
    Run the CASA data reduction pipeline

    Inputs:
      vis         = measurement set
      config_file = filename of the configuration file for this project
      auto        = if True, automatically run the pipeline and do not
                    prompt the user for anything other than the
                    reference antenna at the start

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
    if not os.path.isdir(vis):
        logger.critical('Measurement set not found!')
        raise ValueError('Measurement set not found!')
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
    my_cont_spws,my_line_spws,flux_cals,primary_cals,secondary_cals,science_targets,refant = \
      setup(vis=vis,config=config)
    #
    # if auto, just do automatic routine
    # - Preliminary Flagging
    # - Calibrate and apply to calibrators
    # - Auto-flag calibrators
    # - Re-calibrate and apply to calibrators
    # - Generate plotms figures for calibrators
    # - Apply calibrations to science targets
    # - Auto flag science targets
    # - Generate plotms figures for science targets
    #
    if auto:
        start_time = time.time()
        preliminary_flagging(vis=vis,my_line_spws=my_line_spws,
                             my_cont_spws=my_cont_spws,config=config)
        calibrate_calibrators(vis=vis,primary_cals=primary_cals,
                              secondary_cals=secondary_cals,
                              flux_cals=flux_cals,
                              my_line_spws=my_line_spws,
                              my_cont_spws=my_cont_spws,
                              refant=refant,config=config)
        auto_flag_calibrators(vis=vis,primary_cals=primary_cals,
                              secondary_cals=secondary_cals)
        calibrate_calibrators(vis=vis,primary_cals=primary_cals,
                              secondary_cals=secondary_cals,
                              flux_cals=flux_cals,
                              my_line_spws=my_line_spws,
                              my_cont_spws=my_cont_spws,
                              refant=refant,config=config)
        gen_calibrator_plots(vis=vis,primary_cals=primary_cals,
                             secondary_cals=secondary_cals,
                             config=config)
        calibrate_sciencetargets(vis=vis,science_targets=science_targets)
        auto_flag_sciencetargets(vis=vis,science_targets=science_targets)
        gen_sciencetarget_plots(vis=vis,science_targets=science_targets,
                                config=config)
        run_time = time.time() - start_time
        hrs = int(run_time/3600.)
        mins = int((run_time-3600.*hrs)/60.)
        secs = run_time-3600.*hrs-60.*mins
        print("Runtime: {0:02}h {1:02}m {2:02.2f}s".format(hrs,mins,secs))
        return
    #
    # Prompt the user with a menu for each option
    #
    while True:
        print("0. Flag from configuration file, qvack, shadowed antennas, and tfcrop (auto-flag) all fields")
        print("1. Auto-flag calibrators")
        print("2. Generate plotms figures for calibrators")
        print("3. Manually flag calibrators")
        print("4. Calculate and apply calibration solutions to calibrators")
        print("5. Apply calibration solutions to science targets")
        print("6. Auto-flag science targets")
        print("7. Generate plotms figures for science targets")
        print("8. Manually flag science targets")
        print("9. Split calibrated fields")
        print("q [quit]")
        answer = raw_input("> ")
        if answer == '0':
            preliminary_flagging(vis=vis,my_line_spws=my_line_spws,
                                 my_cont_spws=my_cont_spws,config=config)
        elif answer == '1':
            auto_flag_calibrators(vis=vis,primary_cals=primary_cals,
                                  secondary_cals=secondary_cals)            
        elif answer == '2':
            gen_calibrator_plots(vis=vis,primary_cals=primary_cals,
                                 secondary_cals=secondary_cals,
                                 config=config)
        elif answer == '3':
            manual_flag_calibrators(vis=vis,primary_cals=primary_cals,
                                    secondary_cals=secondary_cals,
                                    config=config)
        elif answer == '4':
            calibrate_calibrators(vis=vis,primary_cals=primary_cals,
                                  secondary_cals=secondary_cals,
                                  flux_cals=flux_cals,
                                  my_line_spws=my_line_spws,
                                  my_cont_spws=my_cont_spws,
                                  refant=refant,config=config)
        elif answer == '5':
            calibrate_sciencetargets(vis=vis,science_targets=science_targets)
        elif answer == '6':
            auto_flag_sciencetargets(vis=vis,science_targets=science_targets)
        elif answer == '7':
            gen_sciencetarget_plots(vis=vis,science_targets=science_targets,
                                    config=config)
        elif answer == '8':
            manual_flag_sciencetargets(vis=vis,science_targets=science_targets,
                                       config=config)
        elif answer == '8':
            split_fields(vis=vis,primary_cals=primary_cals,secondary_cals=secondary_cals,
                         science_targets=science_targets)
        elif answer.lower() == 'q' or answer.lower() == 'quit':
            break
        else:
            print("Input not recognized.")
