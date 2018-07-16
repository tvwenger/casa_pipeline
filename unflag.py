"""
unflag.py
CASA Data Reduction Pipeline - Unflagging Script
Trey V. Wenger September 2017 - V1.0
"""

import __main__ as casa
import os
import time
import numpy as np
import logging
import logging.config
import ConfigParser

__VERSION__ = "1.0"

# load logging configuration file
logging.config.fileConfig('logging.conf')

def setup(vis='',config=None):
    """
    Perform setup tasks: find line and continuum spectral windows
                         get clean parameters

    Inputs:
      vis     = measurement set
      config  = ConfigParser object for this project

    Returns:
      (my_cont_spws,my_line_spws,chan_offset,chan_width)
      my_cont_spws    = comma-separated string of continuum spws
      my_line_spws    = comma-separated string of line spws
      chan_offset     = array of channel offsets
                        of spectral lines relative to highest line spw
      chan_width      = total channel width to un-flag around line
                        center
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
    #
    # Get Unflag parameters from configuration file
    #
    chan_offset = np.array([int(c) for c in config.get("Unflag","chan_offset").split(',')])
    chan_width = config.getint("Unflag","chan_width")
    return (my_cont_spws,my_line_spws,chan_offset,chan_width)

def main(field,vis='',config_file=''):
    """
    Unflag data around spectral lines

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
    my_cont_spws,my_line_spws,chan_offset,chan_width = \
      setup(vis=vis,config=config)
    #
    # Get spw we are looking at
    #
    print("What spectral window are you using to measure the line channel?")
    spw = raw_input("> ")
    spw_ind = my_line_spws.split(',').index(spw)
    #
    # Get channel of spectral line in highest-freq spectral window
    #
    print("What is the channel of the spectral line center in that spectral wnidow?")
    chan = int(raw_input("> "))
    #
    # Chan offset is relative to last spw, so we need to adjust
    # the offsets for this spw
    #
    chan_offset = chan_offset + chan_offset[-1] - chan_offset[spw_ind]
    #
    # Backup flags
    #
    logger.info("Saving flag state...")
    casa.flagmanager(vis=vis,mode='save',versionname='flags_{0}'.format(time.strftime('%Y%m%d%H%M%S',time.gmtime())))
    logger.info("Done.")
    #
    # For each spectral line spectral window, unflag
    # channels at chan + offset - width/2 to
    # chan + offset + width/2
    #
    flagcmd = ''
    for spw,offset in zip(my_line_spws.split(','),chan_offset):
        flagcmd += '{0}:{1}~{2},'.format(spw,chan+int(offset)-int(chan_width/2),
                                         chan+int(offset)+int(chan_width/2))
    # remove last comma
    flagcmd = flagcmd[:-1]
    logger.info("Executing:")
    logger.info("flagdata(vis='{0}',mode='unflag',spw='{1}')".format(vis,flagcmd))
    casa.flagdata(vis=vis,mode='unflag',spw=flagcmd)
    logger.info("Done.")
