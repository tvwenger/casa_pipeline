"""
selfcal.py
CASA Data Reduction Pipeline - Self Calibration Script
Trey V. Wenger Jun 2016 - V1.0
"""

import __main__ as casa
import os
import time
import logging
import logging.config
import ConfigParser

__VERSION__ = "1.0"

# load logging configuration file
logging.config.fileConfig('logging.conf')

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
    Perform setup tasks: find line and continuum spectral windows
                         find reference antenna

    Inputs:
      config  = ConfigParser object for this project

    Returns:
      (my_cont_spws,my_line_spws,refant)
      my_cont_spws    = comma-separated string of continuum spws
      my_line_spws    = comma-separated string of line spws
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
    # Get continuum and line spws from configuration file
    #
    my_cont_spws = config.get("Spectral Windows","Continuum")
    my_line_spws = config.get("Spectral Windows","Line")
    logger.info("Found continuum spws: {0}".format(my_cont_spws))
    logger.info("Found line spws: {0}".format(my_line_spws))
    return (my_cont_spws,my_line_spws,refant)

def save_starting_flags(vis=''):
    """
    Save starting flags

    Inputs:
      vis = measurement set

    Returns:
      None
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # save initial flag state
    #
    logger.info("Saving flag state...")
    casa.flagmanager(vis=vis,mode='save',
                     versionname='starting_flags_{0}'.format(time.strftime('%Y%m%d%H%M%S',time.gmtime())))
    logger.info("Done.")

def phase_selfcal(field='',vis='',refant='',config=None):
    """
    Self-calibrate phases

    Inputs:
      field       = field to be calibrated
      vis         = measurement set containing field
      config      = ConfigParser object for this project

    Returns:
      None
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
    # Calculate phase calibration table
    #
    solint = config.get("Self Calibration","solint")
    combine = config.get("Self Calibration","combine")
    logger.info("Calculate phase calibration table with solint={0} and combine={1}".format(solint,combine))
    casa.gaincal(vis=vis,field=field,caltable='{0}_phase_selfcal.cal'.format(field),
                 calmode='p',solint=solint,combine=combine,
                 refant=refant,minsnr=3.0,minblperant=1)
    logger.info("Done.")

def phase_plot(field=''):
    """
    Plot phase solution table

    Inputs:
      field = calibrated field

    Returns:
      None
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Plot the phase calibration table
    #
    casa.plotcal(caltable='{0}_phase_selfcal.cal'.format(field),
                 xaxis='time',yaxis='phase',
                 iteration='antenna',subplot=311,plotrange=[0,0,-50,50])

def apply_selfcal(field='',vis=''):
    """
    Apply self-calibration to field

    Inputs:
      field       = field to be calibrated
      vis         = measurement set containing field
      config      = ConfigParser object for this project

    Returns:
      None
    """
    #
    # start logger
    #
    logger = logging.getLogger("main")
    #
    # Apply calibration to field
    #
    logger.info("Applying calibration...")
    casa.applycal(vis=vis,field=field,calwt=False,flagbackup=False,
                  gaintable=['{0}_phase_selfcal.cal'.format(field)])
    logger.info("Done.")
    #
    # Save the flags
    #
    logger.info("Saving flag state...")
    casa.flagmanager(vis=vis,mode='save',
                     versionname='calibrate_{0}'.format(time.strftime('%Y%m%d%H%M%S',time.gmtime())))
    logger.info("Done.")

def main(field,vis='',config_file=''):
    """
    Self-calibrate a field

    Inputs:
      field       = field name to clean
      vis         = measurement set containing all data for field
      config_file = filename of the configuration file for this project

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
    my_cont_spws,my_line_spws,refant = setup(vis=vis,config=config)
    #
    # Prompt the user with a menu for each option
    #
    while True:
        print("0. Save starting flags")
        print("1. Perform phase self-calibration")
        print("2. Plot phase calibration table")
        print("3. Apply phase self-calibration to field")
        print("q [quit]")
        answer = raw_input("> ")
        if answer == '0':
            save_starting_flags(vis=vis)
        elif answer == '1':
            phase_selfcal(field=field,vis=vis,refant=refant,config=config)
        elif answer == '2':
            phase_plot(field=field)
        elif answer == '3':
            apply_selfcal(field=field,vis=vis)
        elif answer.lower() == 'q' or answer.lower() == 'quit':
            break
        else:
            print("Input not recognized.")
