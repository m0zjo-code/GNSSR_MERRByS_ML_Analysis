#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:34:41 2020

@author: jonathan
"""
#Import local python files
import os
import sys
module_path = os.getcwd() + '\\GNSSR_Python'
if module_path not in sys.path:
    sys.path.append(module_path)

import datetime

# Destination to write data to
dataFolder = os.path.join(os.getcwd() , 'Data')
#The FTP data access folder 'Data' for regular users or 'DataFast' for low latency access for approved users
ftpDataFolder = 'Data'  # 'Data' or 'DataFast'
# Location and access to the FTP server
## ENTER CREDENTIALS HERE
userName = ''
passWord = ''
ftpServer = 'ftp.merrbys.co.uk'

if len(userName) == 0:
    print('Enter FTP credentials!')

#Time range of interest
# Data is segmented every 6 hours so hours must be one of [3, 9, 15, 21]
startTime = datetime.datetime(2018, 11, 1, 3, 0, 0)
stopTime = datetime.datetime(2018, 11, 3, 21, 0, 0)

#Data levels to download
#  L1b is Delay-Doppler maps
#  L2 FDI  Is the original FDI ocean windspeed algorithm
#  L2 CBRE  Is the improved ocean windspeed algorithm
dataLevels = {'L1B': True, 'L2_FDI': True, 'L2_CBRE_v0_5': True}
# Download data from MERRByS server
# Collect all available files within a given date and time range
#Import GNSSR
from GNSSR_Python.GNSSR import DownloadData
DownloadData(startTime, stopTime, dataFolder, ftpServer, userName, passWord, dataLevels, ftpDataFolder)
