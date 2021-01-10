#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:34:41 2020

@author: Jonathan Rawlinson - M0ZJO
"""
# Configuration of the routine for processing the Level1b histogram
import datetime, os

# Destination to read data from
dataFolder = os.path.join(os.getcwd() , 'Data/')

#Time range of interest
# Data is segmented every 6 hours so hours must be one of [3, 9, 15, 21]
startTime = datetime.datetime(2018, 11, 2, 21, 0, 0)
stopTime = datetime.datetime(2018, 11, 3, 3, 0, 0)

import numpy as np
import h5py
import matplotlib.pyplot as plt
from GNSSR_Python.GNSSR import FolderFromTimeStamp, FindFiles
from GNSSR_Python.CoastalDistanceMap import CoastalDistanceMap
from GNSSR_Python.MapPlotter_Cartopy import MapPlotter
    
def RunMERRBySLevel1bHistogramAndMapExample():
    #Ignore divide by NaN
    np.seterr(divide='ignore', invalid='ignore')

    # Select the data names to extract and plot
    yName = 'DDMSNRAtPeakSingleDDM'
    xName = 'AntennaGainTowardsSpecularPoint'
    #filterOceanOrLand = 'Ocean'
    landDistanceThreshold = 50 # km
    # Filter by geographic area - if enabled
    searchLimitsEnabled = False
    searchLatLimit = [-50, -90]
    
    coastalDistanceMap = CoastalDistanceMap()
    coastalDistanceMap.loadMap(os.path.join(os.getcwd(), 'GNSSR_Python', 'landDistGrid_0.10LLRes_hGSHHSres.nc'))
    
    #Generate a list of possible files in the range startTime to stopTime
    dataList = FindFiles(startTime, stopTime)
    #Initialising lists to ensure they are empty
    x_hist = np.array([])
    y_hist = np.array([])
    
    mapPlotter = MapPlotter(0.2, plotType = "PlateCarree") #Map grid in km (at equator)
    
    #Generate file input list for range
    for entry in dataList:
        entryFolder = FolderFromTimeStamp(entry)
        #Create file path string
        filePath = dataFolder + 'L1B/' + entryFolder + '/metadata.nc'
        #print(filePath)
        try:
            f = h5py.File(filePath, 'r')
        except OSError as e:
            #File does not exist
            #As TDS-1 is run periodically, many time segment files are not populated
            #print(e)
            continue

        print ('Reading file %s...' % entryFolder)

        # Loop through all the tracks
        trackNumber = 0
        while True:
            #Group name in NetCDF is 6 digit string
            groupName = str(trackNumber).zfill(6)

            try:
                #Get data into numpy arrays
                directSignalInDDM = f['/' + groupName + '/DirectSignalInDDM'][:] 
                x_vals = f['/' + groupName + '/' + xName][:]
                DDMSNRAtPeakSingleDDM = f['/' + groupName + '/' + 'DDMSNRAtPeakSingleDDM'][:]
                AntennaGainTowardsSpecularPoint = f['/' + groupName + '/AntennaGainTowardsSpecularPoint'][:]
                specularPointLon = f['/' + groupName + '/SpecularPointLon'][:]
                specularPointLat = f['/' + groupName + '/SpecularPointLat'][:]
            except:
                #End of data
                break
            
            # Filter the data
            # Ocean  - coastal distance            
            coastDistance = coastalDistanceMap.getDistanceToCoast(specularPointLon, specularPointLat)
            
            #Initialise filter vector to all ones
            acceptedData = np.ones(np.shape(directSignalInDDM))
            # Filter out directSignalInDDM when the direct signal is in the
            # delay-doppler space of the reflection DDM
            acceptedData = np.logical_and(acceptedData, directSignalInDDM==0)
            # Filter out land coastDistance=NaN
            acceptedData = np.logical_and(acceptedData, np.isfinite(coastDistance))
            # Filter out coastal data 
            acceptedData = np.logical_and(directSignalInDDM==0, coastDistance>landDistanceThreshold)
            # Filter out where there could be sea-ice - currently disabled
            #acceptedData = np.logical_and(acceptedData, np.abs(specularPointLat) < 55)
            # Filter to geographic area
            if searchLimitsEnabled:
                acceptedData = np.logical_and(acceptedData, np.logical_and(specularPointLat < searchLatLimit[0], specularPointLat > searchLatLimit[1]))
                #acceptedData = np.logical_and(acceptedData, np.logical_and(specularPointLon < searchLonLimit[0], specularPointLon < searchLonLimit[1]))

            #Apply the filter
            filtered_x = x_vals[acceptedData]
            filtered_lat = specularPointLat[acceptedData]
            filtered_lon = specularPointLon[acceptedData]
            filtered_y = DDMSNRAtPeakSingleDDM[acceptedData] - AntennaGainTowardsSpecularPoint[acceptedData]
            
            #Concatenate filtered values to output in histogram
            x_hist = np.concatenate((x_hist, filtered_x))
            y_hist = np.concatenate((y_hist, DDMSNRAtPeakSingleDDM[acceptedData]))
            #Accumulate values into the map
            #print(filtered_lat, filtered_lon)
            if len(filtered_lat) != 0:
                mapPlotter.accumulateDataToMap(filtered_lat, filtered_lon, filtered_y)
            
            # Go to next track
            trackNumber = trackNumber + 1
            
        f.close()
    
    
    # #Plot the data as a histogram
    
    #Plot the data on a map
    print('Plotting map')
    mapPlotter.plotMap("DDM SNR - AGSP / dB")
    
    print('Plotting histogram')
    plt.hist2d(x_hist, y_hist, bins=200, cmap = 'jet')
    plt.figsize=(24, 40)

    plt.xlabel('Antenna Gain [dB]')
    plt.ylabel('DDM Peak SNR [dB]')
    plt.title('Histogram of ' + yName + ' and ' + xName)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

RunMERRBySLevel1bHistogramAndMapExample()