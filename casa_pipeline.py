"""
casa_pipeline.py
SHRDS ATCA Data Reduction Pipeline - CASA version
Trey V. Wenger 2015     - V1.0
Trey V. Wenger Jun 2016 - V1.1
               moved plot generation to separate functions
               added logger
               fixed bug in flagging function that looped you back
               in to a new flagging prompt after applying the flags
               add timestamp to preliminary flagging backup save file
               changed auto-flag process. Use tfcrop on uncalibrated
               data, then calibrate, then use rflag, then generate plots
               DO NOT extend the flags - this deletes entire sources
               if the online flags already flagged too much data.
"""

import __main__ as casa
import os
import numpy as np
import glob
import re
from time import gmtime, strftime
import pickle
import logging
import logging.config

__VERSION__ = "1.1"

# load logging configuration file
logging.config.fileConfig('logging.conf')

# Set defaults
# default project
_DEFAULT_PROJECT="C2963"
# default calibrators
_DEFAULT_PRIMARY_CALS=["1934-638","0823-500"]
_DEFAULT_SECONDARY_CALS=["1036-52","j1322-6532","1613-586",
                         "0906-47","1714-397","1714-336"]
# fluxscale coefficients from Miriad
# had to dig in the f**king source code to find 0823
# FYI, apparently 1934 doesn't agree with VLA
# and 0823 is off by 10-15% according to Suarez+2015
# S = fluxdensity[0]*log(f/reffreq)^(spix[0] + spix[1]*log(f) + spix[2]*log(f)^2)
_REFFREQ = {'1934-638':'1MHz','0823-500':'1MHz'}
_FLUXDENSITY = {'1934-638':[10**-30.7667,0,0,0],'0823-500':[10**-51.0361,0,0,0]}
_SPIX = {'1934-638':[26.4908,-7.0977,0.605334],'0823-500':[41.4101,-10.7771,0.90468]}


_CABB_CONFIGS = [
    {"project":"C2482",
     "iffreqs":[7000,9900, # MHz
              6488,6680,6872,7096,7320,7544,7800,6104,7688,
              6296,9164,9484,9804,10156,10508,10700,9740,8908],
     "lineids":['cont1','cont2',
              'H100a','H99a','H98a','H97a','H96a','H95a','H94a','H102a','KCl',
              'H101a','H89a','H88a','H87a','H86a','H85a','H133d','H125g','H129g'],
     "restfreqs":[7.000,9.900, # GHz
                6.478760,6.676070,6.881490,7.095410,7.318290,7.550610,7.792870,6.106850,7.68907,
                6.289140,9.173320,9.487820,9.816860,10.161300,10.522040,10.695910,9.748560,8.879180]},
    {"project":"C2963",
     "iffreqs":[5505,8540, # MHz
                4609,4737,4865,4993,5153,5281,5441,5601,5761,5921,6113,6305,6465,
                7548,7804,8060,8316,8572,9180,9500],
     "lineids":['cont1','cont2',
                'H112a','H111a','H110a','H109a','H108a','H107a','H106a','H105a','H104a','H103a','H102a','H101a','H100a',
                'H95a','H94a','H93a','H92a','H91a','H89a','H88a'],
     "restfreqs":[5.505,8.540, # GHz
                  4.618790,4.744184,4.874158,5.008924,5.148704,5.293733,5.444262,5.600551,5.762881,5.931546,6.106857,6.289145,6.478761,
                  7.550616,7.792872,8.045604,8.309384,8.584823,9.173323,9.487823]}
    ]

def natural_sort(l):
    """
    Natural sort an alphanumeric list
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def get_refant(vis='',primarycals=[],secondarycals=[],
               my_line_spws='',my_cont_spws=''):
    # start logger
    logger = logging.getLogger("main")
    fields = casa.vishead(vis=vis,mode='get',hdkey='field')[0]
    for ant in ['CA01','CA02','CA03','CA04','CA05']:
        is_good = True
        for calib in primarycals+secondarycals:
            if calib not in fields:
                continue
            for spw in my_line_spws.split(',')+my_cont_spws.split(','):
                x = None
                x = casa.visstat(vis=vis,field=calib,antenna=ant,spw=spw)
                if x is None:
                    is_good=False
                    logger.info("{0} rejected because of {1} spw {2}".format(ant,calib,spw))
                    break
            if not is_good:
                break
        if is_good:
            return ant
    # if we get here, no good antenna was found. Just pick one, then
    # CASA will choose a different one as necessary
    return 'CA01'

def setup(vis='',project=''):
    """
    Perform setup tasks
    Return cabb_config, info from listobs, list of fields
    """
    # start logger
    logger = logging.getLogger("main")
    logger.info("Looking for CABB configuration...")
    cabb_config = None
    for c in _CABB_CONFIGS:
        if c['project'] == project:
            cabb_config = c
    if cabb_config is None:
        logger.critical("Error: project ID not found in available configurations")
        raise ValueError("Error: project ID not found in available configurations")
    logger.info("Found CABB configuration.")
    # Read listobs file and find spw info
    if not os.path.isfile('listobs.txt'):
        logger.info("Generating listobs file...")
        casa.listobs(vis=vis,listfile='listobs.txt')
        logger.info("Done.")
    logger.info("Reading listobs file...")
    with open('listobs.txt') as f:
        field_start_line = -1
        field_end_line = -1
        spw_start_line = -1
        spw_end_line = -1
        for line_num,line in enumerate(f):
            if "Fields:" in line:
                field_start_line = line_num+1
            if "Spectral Windows:" in line:
                field_end_line = line_num-1
                spw_start_line = line_num+1
            if "Sources:" in line:
                spw_end_line = line_num-1
        total_lines = line_num
    logger.info("Done.")
    # determine spws with 128 vs 512 channels
    if spw_start_line == -1 or spw_end_line == -1:
        logger.critical("Error: could not find spectral window list in listobs.txt")
        raise ValueError("Error: could not find spectral window list in listobs.txt")
    else:
        spw_table = np.genfromtxt('listobs.txt',
                                skip_header=spw_start_line+1,
                                skip_footer=total_lines-spw_end_line,
                                dtype=None,comments=';')
        my_cont_spws = ",".join([str(id) for (id, numchans) in
                                zip(spw_table['f0'],spw_table['f1'])
                                if numchans == 33])
        my_line_spws = ",".join([str(id) for (id, numchans) in
                                zip(spw_table['f0'],spw_table['f1'])
                                if numchans == 2049])
    logger.info("Found continuum spws: {0}".format(my_cont_spws))
    logger.info("Found line spws: {0}".format(my_line_spws))
    # get field names
    logger.info("Looking for field names...")
    fields = casa.vishead(vis=vis,mode='get',hdkey='field')[0]
    logger.info("Found fields:")
    logger.info('{0}'.format(fields))
    # make sure figures path exists
    if not os.path.isdir('calib_figures'):
        logger.info("Creating calib_figures directory...")
        os.makedirs('calib_figures')
        logger.info("Done.")
    if not os.path.isdir('scitarg_figures'):
        logger.info("Creating scitarg_figures directory...")
        os.makedirs('scitarg_figures')
        logger.info("Done.")
    return cabb_config,my_cont_spws,my_line_spws,fields

def preliminary_flagging(vis='',my_line_spws=''):
    """
    Flag antenna 6, birdies, qvack, and shadowed antennas
    """
    # start logger
    logger = logging.getLogger("main")
    # save initial flag state
    logger.info("Saving flag state...")
    casa.flagmanager(vis=vis,mode='save',
                     versionname='starting_flags_{0}'.format(strftime('%Y%m%d%H%M%S',gmtime())))
    logger.info("Done.")
    # Flag antenna6
    logger.info("Flagging antenna 6...")
    casa.flagdata(vis=vis,mode='manual',antenna='CA06',
                  flagbackup=False,extendflags=False)
    logger.info("Done.")
    # Flag shadowed antennas
    logger.info("Flagging shadowed antennas...")
    casa.flagdata(vis=vis,mode='shadow',tolerance=-3.0,
                  flagbackup=False,extendflags=False)
    logger.info("Done.")
    # Flag the beginning of each scan
    logger.info("Flagging the beginning of each scan (quack)...")
    casa.flagdata(vis=vis,mode='quack',quackinterval=6,
                  flagbackup=False,extendflags=False)
    logger.info("Done.")
    # Flag birdies in line spws
    logger.info("Flagging birdies in line spws...")
    badchans = ['{0};{1}'.format(foo,bar) for (foo,bar) in zip(range(127,1920,128),range(128,1920,128))]
    badchans = ';'.join(badchans)
    line_spws = ','.join([i+':'+badchans
                          for i in my_line_spws.split(',')])
    casa.flagdata(vis=vis,mode='manual',spw=line_spws,
                  flagbackup=False,extendflags=False)
    logger.info("Done.")
    # Run tfcrop on all fields
    logger.info("Running tfcrop on raw data column...")
    casa.flagdata(vis=vis,mode='tfcrop',
                  timefit='poly',freqfit='poly',
                  flagbackup=False,datacolumn='data',
                  extendflags=False)
    logger.info("Done.")
    logger.info("Saving flag state...")
    casa.flagmanager(vis=vis,mode='save',
                     versionname='preliminary_{0}'.format(strftime('%Y%m%d%H%M%S',gmtime())))
    logger.info("Done.")

def auto_flag_calibrators(vis='',primarycals=[],secondarycals=[],
                         all_fields=[]):
    """
    Perform automatic flagging of calibrators using rflag and tfcrop
    """
    # start logger
    logger = logging.getLogger("main")
    # perform round of automatic flagging
    field = ','.join([i for i in primarycals+secondarycals if i in all_fields])
    # check if calibrators have corrected datacolumn
    stat = None
    logger.info("Checking if ms contains corrected data column...")
    stat = casa.visstat(vis=vis,field=field,spw='0',datacolumn='corrected')
    if stat is None:
        logger.info("Done. ms does not contain corrected data column.")
        datacolumn='data'
    else:
        logger.info("Done. ms does contain corrected data column.")
        datacolumn='corrected'
    if datacolumn == 'corrected':
        logger.info("Running rflag on corrected data column...")
        casa.flagdata(vis=vis,mode='rflag',field=field,
                      flagbackup=False,datacolumn=datacolumn,
                      extendflags=False)
        logger.info("Done.")
    else:
        logger.info("Running tfcrop on raw data column...")
        casa.flagdata(vis=vis,mode='tfcrop',field=field,
                      timefit='poly',freqfit='poly',
                      flagbackup=False,datacolumn='data',
                      extendflags=False)
        logger.info("Done.")
    logger.info("Saving flag state...")
    casa.flagmanager(vis=vis,mode='save',
                     versionname='autoflag_{0}'.format(strftime('%Y%m%d%H%M%S',gmtime())))
    logger.info("Done.")

def gen_calibrator_plots(vis='',primarycals=[],secondarycals=[],
                         all_fields=[]):
    """
    Generate visibility plots for inspection
    """
    # start logger
    logger = logging.getLogger("main")
    field = ','.join([i for i in primarycals+secondarycals if i in all_fields])
    # check if calibrators have corrected datacolumn
    stat = None
    logger.info("Checking if ms contains corrected data column...")
    stat = casa.visstat(vis=vis,spw='0',field=field,datacolumn='corrected')
    if stat is None:
        logger.info("Done. ms does not contain corrected data column.")
        datacolumn='data'
    else:
        logger.info("Done. ms does contain corrected data column.")
        datacolumn='corrected'
    # generate plots for manual inspection
    logger.info("Generating plots for manual inspection...")
    plotnum=0
    plots = []
    for field in [i for i in primarycals+secondarycals if i in all_fields]:
        casa.plotms(vis=vis,xaxis='real',yaxis='imag',field=field,
                    ydatacolumn=datacolumn,iteraxis='spw',
                    coloraxis='baseline',correlation='XX,YY',
                    title='PlotID: {0} Field: {1}'.format(plotnum,field),
                    plotfile='calib_figures/{0}.png'.format(plotnum),
                    overwrite=True,showgui=False,exprange='all')
        plots.append({'field':field,'xaxis':'real','yaxis':'imag','avgtime':'','avgchannel':''})
        plotnum += 1
        casa.plotms(vis=vis,xaxis='time',yaxis='amp',field=field,
                    ydatacolumn=datacolumn,iteraxis='spw',
                    coloraxis='baseline',correlation='XX,YY',
                    title='PlotID: {0} Field: {1}'.format(plotnum,field),
                    avgchannel='1e7',
                    plotfile='calib_figures/{0}.png'.format(plotnum),
                    overwrite=True,showgui=False,exprange='all')
        plots.append({'field':field,'xaxis':'time','yaxis':'amp','avgtime':'','avgchannel':'1e7'})
        plotnum += 1
        casa.plotms(vis=vis,xaxis='time',yaxis='phase',field=field,
                    ydatacolumn=datacolumn,iteraxis='spw',
                    coloraxis='baseline',correlation='XX,YY',
                    title='PlotID: {0} Field: {1}'.format(plotnum,field),
                    avgchannel='1e7',
                    plotfile='calib_figures/{0}.png'.format(plotnum),
                    overwrite=True,showgui=False,exprange='all')
        plots.append({'field':field,'xaxis':'time','yaxis':'phase','avgtime':'','avgchannel':'1e7'})
        plotnum += 1
        casa.plotms(vis=vis,xaxis='channel',yaxis='amp',field=field,
                    ydatacolumn=datacolumn,iteraxis='spw',
                    coloraxis='baseline',correlation='XX,YY',
                    title='PlotID: {0} Field: {1}'.format(plotnum,field),
                    avgtime='1e7',
                    plotfile='calib_figures/{0}.png'.format(plotnum),
                    overwrite=True,showgui=False,exprange='all')
        plots.append({'field':field,'xaxis':'channel','yaxis':'amp','avgtime':'1e7','avgchannel':''})
        plotnum += 1
        casa.plotms(vis=vis,xaxis='channel',yaxis='phase',field=field,
                    ydatacolumn=datacolumn,iteraxis='spw',
                    coloraxis='baseline',correlation='XX,YY',
                    title='PlotID: {0} Field: {1}'.format(plotnum,field),
                    avgtime='1e7',
                    plotfile='calib_figures/{0}.png'.format(plotnum),
                    overwrite=True,showgui=False,exprange='all')
        plots.append({'field':field,'xaxis':'channel','yaxis':'phase','avgtime':'1e7','avgchannel':''})
        plotnum += 1
    logger.info("Done.")
    logger.info("Generating tex document...")
    num_plots = plotnum
    iplot = 0
    with open('calibrator_plots.tex','w') as f:
        f.write(r"\documentclass{article}"+"\n")
        f.write(r"\usepackage{graphicx}"+"\n")
        f.write(r"\usepackage[margin=0.1cm]{geometry}"+"\n")
        f.write(r"\begin{document}"+"\n")
        f.write(r"\begin{figure}"+"\n")
        f.write(r"\centering"+"\n")
        for plotnum in range(num_plots-1):
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
    os.system('pdflatex calibrator_plots.tex')
    logger.info("Done.")
    logger.info("Saving plot list to pickle...")
    with open('calibrator_plots.pkl','w') as f:
        pickle.dump(plots,f)
    logger.info("Done.")

def manual_flag_calibrators(vis='',primarycals=[],secondarycals=[],
                            all_fields=[]):
    """
    manually flag calibrators
    """
    # start logger
    logger = logging.getLogger("main")
    field = ','.join([i for i in primarycals+secondarycals if i in all_fields])
    # check if calibrators have corrected datacolumn
    stat = None
    logger.info("Checking if ms contains corrected data column...")
    stat = casa.visstat(vis=vis,spw='0',field=field,datacolumn='corrected')
    if stat is None:
        logger.info("Done. ms does not contain corrected data column.")
        datacolumn='data'
    else:
        logger.info("Done. ms does contain corrected data column.")
        datacolumn='corrected'
    # perform round of manual flagging
    logger.info("Reading plot list from pickle...")
    with open('calibrator_plots.pkl','r') as f:
        plots = pickle.load(f)
    num_plots = len(plots)
    logger.info("Done.")
    logger.info("Please inspect calibrator_plots.pdf then perform manual calibrations.")
    while True:
        print("f - flag some data")
        print("plot id number - generate interactive version of plot with this id")
        print("quit - end this flagging session")
        answer = raw_input()
        if answer.lower() == 'f':
            flag_commands = []
            while True:
                print("Field? Empty = all calibrator fields")
                field = raw_input()
                if field == '':
                    field = ','.join([i for i in primarycals+secondarycals if i in all_fields])
                print("Spectral window and channels? Empty = all spws (ex. 2:100~150;200:250,5:1200~1210 is spw 2, chans 100 to 150  and 200 to 250, and spw 5 chans 1200 to 1210)")
                spw = raw_input()
                print("Time range? Empty = all times (ex. 10:23:45~10:23:55)")
                timerange = raw_input()
                print("Antenna or baseline? Empty = all antennas/baselines (ex. CA01 to flag ant 1 or CA01&CA02 to flag 1-2 baseline)")
                antenna = raw_input()
                print("Correlation? Empty = all correlations (XX,YY,XY,YX)")
                correlation = raw_input()
                parts = []
                if len(field) > 0:
                    parts.append("field='{0}'".format(field))
                if len(spw) > 0:
                    parts.append("spw='{0}'".format(spw))
                if len(timerange) > 0:
                    parts.append("timerange='{0}'".format(timerange))
                if len(antenna) > 0:
                    parts.append("antenna='{0}'".format(antenna))
                if len(correlation) > 0:
                    parts.append("correlation='{0}'".format(correlation))
                flag_commands.append(' '.join(parts))
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
                    with open('manual_flags.txt','a') as f:
                        for cmd in flag_commands:
                            f.write(strftime('%Y%m%d%H%M%S',gmtime())+': '+cmd+'\n')
                    casa.flagdata(vis=vis,mode='list',flagbackup=False,extendflags=False,
                                  inpfile=flag_commands)
                    break
                elif go.lower() == 'a':
                    continue
                else:
                    print("Aborting...")
                    break
        elif answer.lower() == 'quit':
            break
        else:
            try:
                plotid = int(answer)
            except ValueError:
                print("Plot ID not valid!")
                continue
            if plotid < num_plots:
                casa.plotms(vis=vis,xaxis=plots[plotid]['xaxis'],yaxis=plots[plotid]['yaxis'],field=plots[plotid]['field'],
                            ydatacolumn=datacolumn,iteraxis='spw',
                            coloraxis='baseline',correlation='XX,YY',
                            title='PlotID: {0} Field: {1}'.format(plotid,plots[plotid]['field']),
                            avgchannel=plots[plotid]['avgchannel'],avgtime=plots[plotid]['avgtime'])
    logger.info("Saving flag state...")
    casa.flagmanager(vis=vis,mode='save',
                     versionname='manualflag_{0}'.format(strftime('%Y%m%d%H%M%S',gmtime())))
    logger.info("Done.")

def calibrate_calibrators(vis='',primarycals=[],secondarycals=[],
                          all_fields=[],my_line_spws='',
                          my_cont_spws=''):
    """
    Calculate calibration solutions and apply to calibrators
    """
    # start logger
    logger = logging.getLogger("main")
    # find good referance antenna
    logger.info("Looking for good reference antenna...")
    refant = get_refant(vis=vis,primarycals=primarycals,
                        secondarycals=secondarycals,my_line_spws=my_line_spws,
                        my_cont_spws=my_cont_spws)
    if refant is None:
        logger.critical("No good referance antenna found!")
        raise ValueError("No good referance antenna found!")
    logger.info("Done. Found reference antenna: {0}".format(refant))
    # find first primary cal in all fields
    fluxcal = None
    for cal in primarycals:
        if cal in all_fields:
            fluxcal=cal
            break
    if fluxcal is None:
        logger.critical("No good flux calibrator found!")
        raise ValueError("No good flux calibrator found!")
    # initial delay calibration on bandpass calibrators
    logger.info("Calculating delay calibration table for primary calibrators...")
    field = ','.join([i for i in primarycals if i in all_fields])
    if os.path.isdir('delays.cal'):
        casa.rmtables('delays.cal')
    casa.gaincal(vis=vis,caltable='delays.cal',field=field,
                 refant=refant,gaintype='K',minblperant=1)
    logger.info("Done.")
    # short timescale phase calibration
    logger.info("Calculating phase calibration table on integration timescales for primary calibrators...")
    if os.path.isdir('phase_int.cal'):
        casa.rmtables('phase_int.cal')
    casa.gaincal(vis=vis,caltable="phase_int.cal",field=field,
                 solint="int",calmode="p",refant=refant,gaintype="G",
                 gaintable=['delays.cal'],minblperant=1)
    logger.info("Done.")
    # bandpass calibration on all channels of continuum bands
    logger.info("Calculating bandpass calibration table for primary calibrators...")
    if os.path.isdir('bandpass.cal'):
        casa.rmtables('bandpass.cal')
    casa.bandpass(vis=vis,caltable='bandpass.cal',field=field,
                  spw=my_cont_spws,refant=refant,solint='inf',
                  combine='scan',solnorm=True,minblperant=1,
                  gaintable=['delays.cal','phase_int.cal'])
    # bandpass calibration average 64 channels of line bands
    casa.bandpass(vis=vis,caltable='bandpass.cal',field=field,
                  spw=my_line_spws,refant=refant,solint='inf,64chan',
                  combine='scan',solnorm=True,minblperant=1,append=True,
                  gaintable=['delays.cal','phase_int.cal'])
    logger.info("Done.")
    # apply bandpass solutions back to primary calibrators
    logger.info("Applying calibration tables to primary calibrators...")
    for field in [i for i in primarycals if i in all_fields]:
        casa.applycal(vis=vis,field=field,calwt=False,
                      gaintable=['delays.cal','phase_int.cal','bandpass.cal'],
                      gainfield=field,flagbackup=False)
    logger.info("Done.")
    # set flux scale
    logger.info("Setting the flux scale...")
    casa.setjy(vis=vis,field=fluxcal,scalebychan=True,
               standard='manual',
               fluxdensity=_FLUXDENSITY[fluxcal],
               spix=_SPIX[fluxcal],
               reffreq=_REFFREQ[fluxcal])
    logger.info("Done.")
    # re-calculate short-timescale phase corrections for all calibrators
    logger.info("Re-calculating the phase calibration table on integration timescales for all calibrators...")
    field = ','.join([i for i in primarycals+secondarycals if i in all_fields])
    if os.path.isdir('phase_int.cal'):
        casa.rmtables('phase_int.cal')
    casa.gaincal(vis=vis,caltable="phase_int.cal",field=field,minblperant=1,
                solint="int",calmode="p",refant=refant,gaintype="G",
                gaintable=['delays.cal','bandpass.cal'])
    logger.info("Done.")
    # scan-timescale phase corrections
    logger.info("Calculating the phase calibration table on scan timescales for all calibrators...")
    if os.path.isdir('phase_scan.cal'):
        casa.rmtables('phase_scan.cal')
    casa.gaincal(vis=vis,caltable="phase_scan.cal",field=field,minblperant=1,
                solint="inf",calmode="p",refant=refant,gaintype="G",
                gaintable=['delays.cal','bandpass.cal'])
    logger.info("Done.")
    # amplitude corrections with short phase
    logger.info("Calculating the amplitude and phase calibration table on scan timescales for all calibrators...")
    if os.path.isdir('apcal.cal'):
        casa.rmtables('apcal.cal')
    casa.gaincal(vis=vis,caltable="apcal.cal",field=field,minblperant=1,
                 solint="inf",calmode="ap",refant=refant,
                 gaintable=['delays.cal','bandpass.cal','phase_int.cal'])
    logger.info("Done.")
    # set the flux scale
    logger.info("Calculating the flux calibration table...")
    if os.path.isdir('flux.cal'):
        casa.rmtables('flux.cal')
    casa.fluxscale(vis=vis,caltable="apcal.cal",fluxtable="flux.cal",
                   reference=fluxcal,incremental=True)
    logger.info("Done.")
    # apply solutions back to calibrators
    logger.info("Applying calibration tables to all calibrators...")
    for field in [i for i in primarycals+secondarycals if i in all_fields]:
        casa.applycal(vis=vis,field=field,calwt=False,
                      gaintable=['delays.cal','bandpass.cal',
                                 'phase_int.cal','apcal.cal',
                                 'flux.cal'],
                      gainfield=['','',field,field,field],
                      flagbackup=False)
    logger.info("Done.")
    # save flag state
    casa.flagmanager(vis=vis,mode='save',
                     versionname='calibrate_{0}'.format(strftime('%Y%m%d%H%M%S',gmtime())))

def calibrate_sciencetargets(vis='',primarycals=[],secondarycals=[],
                             all_fields=[]):
    """
    Apply calibration solutions to science targets
    """
    # start logger
    logger = logging.getLogger("main")
    for field in all_fields:
        if field in primarycals+secondarycals:
            continue
        # use all fields in delays/bandpass
        # use nearest field in gain
        logger.info("Applying calibration solutions to {0}".format(field))
        casa.applycal(vis=vis,field=field,calwt=False,
                      gaintable=['delays.cal','bandpass.cal',
                                 'phase_int.cal','apcal.cal',
                                 'flux.cal'],
                      gainfield=['','','nearest','nearest','nearest'],
                      flagbackup=False)
    # save flag state
    logger.info("Saving flag state...")
    casa.flagmanager(vis=vis,mode='save',
                     versionname='calibrate_{0}'.format(strftime('%Y%m%d%H%M%S',gmtime())))
    logger.info("Done.")

def auto_flag_sciencetargets(vis='',primarycals=[],secondarycals=[],
                             all_fields=[]):
    """
    Perform automatic flagging of science targets using rflag and tfcrop
    """
    # start logger
    logger = logging.getLogger("main")
    # perform round of automatic flagging
    field = ','.join([i for i in all_fields if i not in primarycals+secondarycals])
    datacolumn='corrected'
    logger.info("Running rflag on all correlations...")
    casa.flagdata(vis=vis,mode='rflag',field=field,
                  flagbackup=False,datacolumn=datacolumn,
                  extendflags=False)
    logger.info("Done.")
    logger.info("Saving flag state...")
    casa.flagmanager(vis=vis,mode='save',
                     versionname='autoflag_{0}'.format(strftime('%Y%m%d%H%M%S',gmtime())))
    logger.info("Done.")

def gen_sciencetarget_plots(vis='',primarycals=[],secondarycals=[],
                            all_fields=[]):
    """
    Generate visibility plots for inspection
    """
    # start logger
    logger = logging.getLogger("main")
    datacolumn='corrected'
    # generate plots for manual inspection
    logger.info("Generating plots for manual inspection...")
    plotnum=0
    plots = []
    for field in [i for i in all_fields if i not in primarycals+secondarycals]:
        casa.plotms(vis=vis,xaxis='real',yaxis='imag',field=field,
                    ydatacolumn=datacolumn,iteraxis='spw',
                    coloraxis='baseline',correlation='XX,YY',
                    title='PlotID: {0} Field: {1}'.format(plotnum,field),
                    plotfile='scitarg_figures/{0}.png'.format(plotnum),
                    overwrite=True,showgui=False,exprange='all')
        plots.append({'field':field,'xaxis':'real','yaxis':'imag','avgtime':'','avgchannel':''})
        plotnum += 1
        casa.plotms(vis=vis,xaxis='time',yaxis='amp',field=field,
                    ydatacolumn=datacolumn,iteraxis='spw',
                    coloraxis='baseline',correlation='XX,YY',
                    title='PlotID: {0} Field: {1}'.format(plotnum,field),
                    avgchannel='1e7',
                    plotfile='scitarg_figures/{0}.png'.format(plotnum),
                    overwrite=True,showgui=False,exprange='all')
        plots.append({'field':field,'xaxis':'time','yaxis':'amp','avgtime':'','avgchannel':'1e7'})
        plotnum += 1
        casa.plotms(vis=vis,xaxis='channel',yaxis='amp',field=field,
                    ydatacolumn=datacolumn,iteraxis='spw',
                    coloraxis='baseline',correlation='XX,YY',
                    title='PlotID: {0} Field: {1}'.format(plotnum,field),
                    avgtime='1e7',
                    plotfile='scitarg_figures/{0}.png'.format(plotnum),
                    overwrite=True,showgui=False,exprange='all')
        plots.append({'field':field,'xaxis':'channel','yaxis':'amp','avgtime':'1e7','avgchannel':''})
        plotnum += 1
    logger.info("Done.")
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
        for plotnum in range(num_plots-1):
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
    os.system('pdflatex scitarg_plots.tex')
    logger.info("Done.")
    logger.info("Saving plot list to pickle...")
    with open('scitarg_plots.pkl','w') as f:
        pickle.dump(plots,f)
    logger.info("Done.")

def manual_flag_sciencetargets(vis='',primarycals=[],secondarycals=[],
                               all_fields=[],auto=False):
    """
    manually flag science targets
    """
    # start logger
    logger = logging.getLogger("main")
    datacolumn='corrected'
    logger.info("Reading plot list from pickle...")
    with open("scitarg_plots.pkl","r") as f:
        plots = pickle.load(f)
    num_plots = len(plots)
    logger.info("Done.")
    logger.info("Please inspect scitarg_plots.pdf then perform manual calibrations.")
    while True:
        print("f - flag some data")
        print("plot id number - generate interactive version of plot with this id")
        print("quit - end this flagging session")
        answer = raw_input()
        if answer.lower() == 'f':
            flag_commands = []
            while True:
                print("Field? Empty = all science target fields")
                field = raw_input()
                if field == '':
                    field = ','.join([i for i in all_fields if i not in primarycals+secondarycals])
                print("Spectral window and channels? Empty = all spws (ex. 2:100~150;200:250,5:1200~1210 is spw 2, chans 100 to 150  and 200 to 250, and spw 5 chans 1200 to 1210)")
                spw = raw_input()
                print("Time range? Empty = all times (ex. 10:23:45~10:23:55)")
                timerange = raw_input()
                print("Antenna or baseline? Empty = all antennas/baselines (ex. CA01 to flag ant 1 or CA01&CA02 to flag 1-2 baseline)")
                antenna = raw_input()
                print("Correlation? Empty = all correlations (XX,YY,XY,YX)")
                correlation = raw_input()
                parts = []
                if len(field) > 0:
                    parts.append("field='{0}'".format(field))
                if len(spw) > 0:
                    parts.append("spw='{0}'".format(spw))
                if len(timerange) > 0:
                    parts.append("timerange='{0}'".format(timerange))
                if len(antenna) > 0:
                    parts.append("antenna='{0}'".format(antenna))
                if len(correlation) > 0:
                    parts.append("correlation='{0}'".format(correlation))
                flag_commands.append(' '.join(parts))
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
                    with open('manual_flags.txt','a') as f:
                        for cmd in flag_commands:
                            f.write(strftime('%Y%m%d%H%M%S',gmtime())+': '+cmd+'\n')
                    casa.flagdata(vis=vis,mode='list',flagbackup=False,extendflags=False,
                                  inpfile=flag_commands)
                    break
                elif go.lower() == 'a':
                    continue
                else:
                    print("Aborting...")
                    break
        elif answer.lower() == 'quit':
            break
        else:
            try:
                plotid = int(answer)
            except ValueError:
                print("Plot ID not valid!")
                continue
            if plotid < num_plots:
                casa.plotms(vis=vis,xaxis=plots[plotid]['xaxis'],yaxis=plots[plotid]['yaxis'],field=plots[plotid]['field'],
                            ydatacolumn=datacolumn,iteraxis='spw',
                            coloraxis='baseline',correlation='XX,YY',
                            title='PlotID: {0} Field: {1}'.format(plotid,plots[plotid]['field']),
                            avgchannel=plots[plotid]['avgchannel'],avgtime=plots[plotid]['avgtime'])
    logger.info("Saving flag state...")
    casa.flagmanager(vis=vis,mode='save',
                     versionname='manualflag_{0}'.format(strftime('%Y%m%d%H%M%S',gmtime())))
    logger.info("Done.")

def main(vis,project=_DEFAULT_PROJECT,
         primarycals=_DEFAULT_PRIMARY_CALS,
         secondarycals=_DEFAULT_SECONDARY_CALS,auto=False):
    # start logger
    logger = logging.getLogger("main")
    # initial setup
    cabb_config,my_cont_spws,my_line_spws,all_fields = \
        setup(vis=vis,project=project)
    # if auto, just do automatic routine
    # - Preliminary Flagging
    # - Calibrate and apply to calibrators
    # - Auto-flag calibrators
    # - Re-calibrate and apply to calibrators
    # - Generate plotms figures for calibrators
    # - Apply calibrations to science targets
    # - Auto flag science targets
    # - Generate plotms figures for science targets
    if auto:
        # preliminary flagging
        preliminary_flagging(vis=vis,my_line_spws=my_line_spws)
        # calibrate calibrators
        calibrate_calibrators(vis=vis,primarycals=primarycals,
                              secondarycals=secondarycals,
                              all_fields=all_fields,
                              my_line_spws=my_line_spws,
                              my_cont_spws=my_cont_spws)
        # auto flag calibrators
        auto_flag_calibrators(vis=vis,primarycals=primarycals,
                              secondarycals=secondarycals,
                              all_fields=all_fields)
        # re-calibrate calibrators
        calibrate_calibrators(vis=vis,primarycals=primarycals,
                              secondarycals=secondarycals,
                              all_fields=all_fields,
                              my_line_spws=my_line_spws,
                              my_cont_spws=my_cont_spws)
        # generate calibrator plots
        gen_calibrator_plots(vis=vis,primarycals=primarycals,
                             secondarycals=secondarycals,
                             all_fields=all_fields)
        # apply calibration solutions to science targets
        calibrate_sciencetargets(vis=vis,primarycals=primarycals,
                                 secondarycals=secondarycals,
                                 all_fields=all_fields)
        # auto-flag the science targets
        auto_flag_sciencetargets(vis=vis,
                                 primarycals=primarycals,
                                 secondarycals=secondarycals,
                                 all_fields=all_fields)
        # generate science target plots
        gen_sciencetarget_plots(vis=vis,primarycals=primarycals,
                                secondarycals=secondarycals,
                                all_fields=all_fields)
        return
    # menu items
    while True:
        print("0. Flag antenna 6, birdies, qvack, shadowed antennas, tfcrop (auto-flag) all fields")
        print("1. Auto-flag calibrators")
        print("2. Generate plotms figures for calibrators")
        print("3. Manually flag calibrators")
        print("4. Calculate and apply calibration solutions to calibrators")
        print("5. Apply calibration solutions to science targets")
        print("6. Auto-flag science targets")
        print("7. Generate plotms figures for science targets")
        print("8. Manually flag science targets")
        print("q [quit]")
        answer = raw_input("SHRDS> ")
        if answer == '0':
            # preliminary flagging
            preliminary_flagging(vis=vis,my_line_spws=my_line_spws)
        elif answer == '1':
            # auto flag calibrators
            auto_flag_calibrators(vis=vis,primarycals=primarycals,
                                  secondarycals=secondarycals,
                                  all_fields=all_fields)
        elif answer == '2':
            # generate calibrator plots
            gen_calibrator_plots(vis=vis,primarycals=primarycals,
                                 secondarycals=secondarycals,
                                 all_fields=all_fields)
        elif answer == '3':
            # manually flag calibrators
            manual_flag_calibrators(vis=vis,primarycals=primarycals,
                                    secondarycals=secondarycals,
                                    all_fields=all_fields)
        elif answer == '4':
            # calibrate calibrators
            calibrate_calibrators(vis=vis,primarycals=primarycals,
                                  secondarycals=secondarycals,
                                  all_fields=all_fields,
                                  my_line_spws=my_line_spws,
                                  my_cont_spws=my_cont_spws)
        elif answer == '5':
            # apply calibration solutions to science targets
            calibrate_sciencetargets(vis=vis,primarycals=primarycals,
                                   secondarycals=secondarycals,
                                   all_fields=all_fields)
        elif answer == '6':
            # auto-flag the science targets
            auto_flag_sciencetargets(vis=vis,
                                     primarycals=primarycals,
                                     secondarycals=secondarycals,
                                     all_fields=all_fields)
        elif answer == '7':
            # generate science target plots
            gen_sciencetarget_plots(vis=vis,primarycals=primarycals,
                                    secondarycals=secondarycals,
                                    all_fields=all_fields)
        elif answer == '8':
            # manually flag science targets
            manual_flag_sciencetargets(vis=vis,primarycals=primarycals,
                                    secondarycals=secondarycals,
                                    all_fields=all_fields)
        elif answer.lower() == 'q' or answer.lower() == 'quit':
            break
        else:
            print("Input not recognized.")
