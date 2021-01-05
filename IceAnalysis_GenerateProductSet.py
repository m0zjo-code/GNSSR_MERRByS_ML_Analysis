#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:34:41 2020

@author: jonathan
"""
# Configuration of the routine for processing the Level1b histogram
import datetime, os

GENERATE_TRAINING_DATA = True
GENERATE_TRAINING_DATA_NPY = False
GENERATE_IMG_FILES = False
CLASSIFY_DATA = True

# Destination to read data from
dataFolder = os.path.join(os.getcwd() , 'Data/')

#Time range of interest
# Data is segmented every 6 hours so hours must be one of [3, 9, 15, 21]

startTime = datetime.datetime(2018, 11, 1, 3, 0, 0)
stopTime = datetime.datetime(2018, 11, 3, 21, 0, 0)

#startTime = datetime.datetime(2018, 11, 2, 21, 0, 0)
#stopTime = datetime.datetime(2018, 11, 3, 3, 0, 0)

NSIDCPath = "Data/NSIDCIceData/"

MLDataLabelNames = ["Ocean", "SeaIce"]

MinIceConc = 150
MaxIceConc = 1000
SNRScaleFactor = 12
NoiseScaleFactor = 20000
DDMScaleFactor = 65535

import numpy as np
import os
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
#import ipywidgets as widgets
from GNSSR_Python.GNSSR import FindFiles, FolderFromTimeStamp
from GNSSR_Python.CoastalDistanceMap import CoastalDistanceMap
from GNSSR_Python.MapPlotter_Cartopy import MapPlotter
import ReadNSIDCIceData
from PIL import Image
import keras
import pickle
    
def RunMERRBySLevel1bHistogramAndMapExample():
    #Ignore divide by NaN
    np.seterr(divide='ignore', invalid='ignore')

    # Select the data names to extract and plot
    yName = 'DDMSNRAtPeakSingleDDM'
    xName = 'AntennaGainTowardsSpecularPoint'
    
    #filterOceanOrLand = 'Ocean'
    landDistanceThreshold = 50 # km
    # Filter by geographic area - if enabled
    searchLimitsEnabled = True
    searchLatLimit = [-30, 30]
    #searchLonLimit = [-10, 10]
    
    coastalDistanceMap = CoastalDistanceMap()
    coastalDistanceMap.loadMap(os.path.join(os.getcwd(), 'GNSSR_Python', 'landDistGrid_0.10LLRes_hGSHHSres.nc'))
    
    #Generate a list of possible files in the range startTime to stopTime
    dataList = FindFiles(startTime, stopTime)
    print("Processing:", dataList)
    #Initialising lists to ensure they are empty
    x = np.array([])
    y = np.array([])
    
    mapPlotter = MapPlotter(25e3, plotType = "South") #Map grid in km (at equator)
    #mapPlotter2 = MapPlotter(100e3) #Map grid in km (at equator)
    
    #Set Up Data Output Paths
    for name in MLDataLabelNames:
        local_path = os.path.join(dataFolder, name)
        print(local_path)
        Path(local_path).mkdir(parents=True, exist_ok=True)
    
    ID_NUM = 0
    
    if CLASSIFY_DATA:
        print("Classifiying")
        reconstructed_model = keras.models.load_model("ML_Toolkit/IceDNN.h5")
        with open("ML_Toolkit/IceDNN.nncfg", "rb") as pklfile:
            pkldict = pickle.load(pklfile)
            X_min = pkldict["X_min"]
            X_max = pkldict["X_max"]

    
    #Generate file input list for range
    for entry in dataList:
        entryFolder = FolderFromTimeStamp(entry)
        #Create file path string
        filePath = dataFolder + 'L1B/' + entryFolder + '/metadata.nc'
        filePathDDMs = dataFolder + 'L1B/' + entryFolder + '/ddms.nc'
        #print(filePath)
        try:
            f = h5py.File(filePath, 'r')
            fddms = h5py.File(filePathDDMs, 'r')
        except OSError as e:
            #File does not exist
            #As TDS-1 is run periodically, many time segment files are not populated
            #print(e)
            continue

        print ('Reading file %s...' % entryFolder)

        # Loop through all the tracks
        trackNumber = 0
        while True:
            ##############
            #Group name in NetCDF is 6 digit string
            groupName = str(trackNumber).zfill(6)
            print("Processing Track:", entryFolder, groupName)
            try:
                #Get data into numpy arrays
                directSignalInDDM = f['/' + groupName + '/DirectSignalInDDM'][:] 
                x_vals = f['/' + groupName + '/' + xName][:]
                DDMSNRAtPeakSingleDDM = f['/' + groupName + '/' + 'DDMSNRAtPeakSingleDDM'][:]
                
                specularPointLon = f['/' + groupName + '/SpecularPointLon'][:]
                specularPointLat = f['/' + groupName + '/SpecularPointLat'][:]
                MeanNoiseBox = f['/' + groupName + '/MeanNoiseBox'][:]
                DDMOutputNumericalScaling = f['/' + groupName + '/DDMOutputNumericalScaling'][:]
                SPElevationORF = f['/' + groupName + '/SPElevationORF'][:]
                SPElevationARF = f['/' + groupName + '/SPElevationARF'][:]
                SPIncidenceAngle = f['/' + groupName + '/SPIncidenceAngle'][:]
                AntennaGainTowardsSpecularPoint = f['/' + groupName + '/AntennaGainTowardsSpecularPoint'][:]
                
                ddm_vals = fddms['/' + groupName + '/DDM'][:]
                
            except:
                #End of data
                break
            
            SVN = f['/' + groupName].attrs['SVN'][0]
            
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
            acceptedData = np.logical_and(acceptedData, coastDistance>landDistanceThreshold)
            # Filter out where there could be sea-ice - currently disabled
            #acceptedData = np.logical_and(acceptedData, np.abs(specularPointLat) < 55)
            # Filter to geographic area
            if searchLimitsEnabled:
                acceptedData = np.logical_and(acceptedData, np.logical_or(specularPointLat < searchLatLimit[0], specularPointLat > searchLatLimit[1]))
                #acceptedData = np.logical_and(acceptedData, np.logical_and(specularPointLon > searchLonLimit[0], specularPointLon < searchLonLimit[1]))
    
            noDDMs = len(directSignalInDDM)
            IceData = np.ones(np.shape(directSignalInDDM)) * -1
            
            """
            Ice Data Error Codes (internal):
                -1 = Data not availible from GeoTIFF (but accepted by filter)
            
            Ice Data Definitions (from https://nsidc.org/sites/nsidc.org/files/G02135-V3.0_0.pdf):
                0-1000 = Sea ice conc. (divide by 10 to get %)
                0 = ocean
                2510 = polar hole
                2530 = coast line 
                2540 = land
                2550 = missing from GeoTIFF
            """
            NSIDCData = ReadNSIDCIceData.extract_val_setup(startTime, specularPointLat[0], specularPointLon[0], dataPath = NSIDCPath)
            for i in range(0, noDDMs):
                if ((NSIDCData[3] == "south") and (specularPointLat[i] > 0)) or ((NSIDCData[3] == "north") and (specularPointLat[i] < 0)):
                    NSIDCData = ReadNSIDCIceData.extract_val_setup(startTime, specularPointLat[i], specularPointLon[i], dataPath = NSIDCPath)
                if acceptedData[i]:
                    IceData[i] = ReadNSIDCIceData.get_val(specularPointLat[i], specularPointLon[i], NSIDCData[0], NSIDCData[1], NSIDCData[2])
                if IceData[i] == -1.0:
                    acceptedData[i] = 0
                # Filter out non ocean/sea ice points
                if (IceData[i] > 1000):
                    acceptedData[i] = 0
            
            #print(IceData)
            IceData[np.logical_and(IceData > MinIceConc, IceData <= MaxIceConc)] = 1000
            IceData[IceData <= MinIceConc] = 0
            
            #plt.plot(specularPointLat)
            #plt.show()
            #plt.plot(specularPointLon)
            #plt.show()
                      
            #Apply the filter
            filtered_x = x_vals[acceptedData]
            filtered_DDMSNRAtPeakSingleDDM = DDMSNRAtPeakSingleDDM[acceptedData]
            filtered_lat = specularPointLat[acceptedData]
            filtered_lon = specularPointLon[acceptedData]
            filtered_IceData = IceData[acceptedData]
            filtered_MeanNoiseBox = MeanNoiseBox[acceptedData]
            filtered_DDMs = ddm_vals[acceptedData]
            filtered_DDMOutputNumericalScaling = DDMOutputNumericalScaling[acceptedData]
            filtered_SPElevationORF = SPElevationORF[acceptedData]
            filtered_SPElevationARF = SPElevationARF[acceptedData]
            filtered_SPIncidenceAngle = SPIncidenceAngle[acceptedData]
            filtered_AntennaGainTowardsSpecularPoint = AntennaGainTowardsSpecularPoint[acceptedData]
            
            ##print(ddm_vals.shape, filtered_DDMs.shape)
            print("%i/%i DDMs Selected" % (filtered_DDMs.shape[0], ddm_vals.shape[0]))
            #Concatenate filtered values to output in histogram
            x = np.concatenate((x, filtered_x))
            y = np.concatenate((y, filtered_DDMSNRAtPeakSingleDDM))
            
            noDDMsFiltered = len(filtered_IceData)   
            
            sigmaest= filtered_DDMSNRAtPeakSingleDDM - filtered_AntennaGainTowardsSpecularPoint
            
            #print(filtered_IceData)
            
            if CLASSIFY_DATA:
                inputDataFrame = np.zeros((noDDMsFiltered, 8))
            
            if GENERATE_TRAINING_DATA:
                with open('IceTrainingDataV1.csv','a') as fd:
                    for i in range(0, noDDMsFiltered):
                        # Ocean
                        if (filtered_IceData[i] <= MinIceConc):
                            local_path = os.path.join(dataFolder, MLDataLabelNames[0])
                            ClassName = MLDataLabelNames[0]
                            
                        # Sea Ice
                        elif (filtered_IceData[i] > MinIceConc) and (filtered_IceData[i] <= MaxIceConc):
                            local_path = os.path.join(dataFolder, MLDataLabelNames[1])
                            ClassName = MLDataLabelNames[1]
                            
                        save_file_path = local_path + "/ID-%s" % str(ID_NUM).zfill(8)
                        
                        frame = [filtered_DDMSNRAtPeakSingleDDM[i], 
                                filtered_SPElevationORF[i], 
                                filtered_SPElevationARF[i], 
                                filtered_MeanNoiseBox[i], 
                                filtered_DDMOutputNumericalScaling[i],
                                filtered_SPIncidenceAngle[i],
                                SVN,
                                filtered_AntennaGainTowardsSpecularPoint[i],
                                ClassName
                                ]
                        
                        classifyframe = [filtered_DDMSNRAtPeakSingleDDM[i], 
                                filtered_SPElevationORF[i], 
                                filtered_SPElevationARF[i], 
                                filtered_MeanNoiseBox[i], 
                                filtered_DDMOutputNumericalScaling[i],
                                filtered_SPIncidenceAngle[i],
                                SVN,
                                filtered_AntennaGainTowardsSpecularPoint[i],
                                ]
                        
                        frame_string = "%f,%f,%f,%f,%f,%f,%f,%f,%s\n" % tuple(frame)
                        
                        #fd.write(frame_string)
                        
                        if CLASSIFY_DATA:
                            inputDataFrame[i,:] = np.asarray(classifyframe)
                        
                        
                        if GENERATE_TRAINING_DATA_NPY:
                            waveform = np.reshape(filtered_DDMs[i] / DDMScaleFactor, (2560))
                            w_data = np.concatenate((
                                                    waveform/np.max(waveform),
                                                    [(filtered_DDMSNRAtPeakSingleDDM[i] + SNRScaleFactor) / (SNRScaleFactor * 2)], 
                                                    [(filtered_SPElevationORF[i] + 360) / (360 * 2)], 
                                                    [(filtered_SPElevationARF[i] + 360) / (360 * 2)],
                                                    [filtered_MeanNoiseBox[i]/NoiseScaleFactor],
                                                    [(filtered_AntennaGainTowardsSpecularPoint[i] + 20)/(20*2)],
                                                    [SVN/100]
                                                    ))
                                                    
                            np.save(save_file_path, w_data)
                            
                        
                            
                        
                        if GENERATE_IMG_FILES:
                            im = Image.fromarray(filtered_DDMs[i])
                            im.save(save_file_path + '.png')
                        
                        ID_NUM = ID_NUM + 1
                
                # if we are classifiying, do it here
                if CLASSIFY_DATA:
                    print("Classifiying")
                    #print(inputDataFrame.shape)
                    #print(X_max.shape)
                    #print(X_min.shape)
                    ## DO CLASSIFICATION HERE ##
                    if noDDMsFiltered != 0:
                        processDataFrame = (inputDataFrame - X_min) / (X_max - X_min)
                        predicteddataframe = reconstructed_model.predict(processDataFrame)
                        predicteddataframe = np.argmax(predicteddataframe, axis = 1)
                        print(predicteddataframe)
                        mapPlotter.accumulateDataToMap(filtered_lat, filtered_lon, predicteddataframe)
                        
                
            #Accumulate values into the map
            #if len(filtered_lat) != 0:
                #mapPlotter.accumulateDataToMap(filtered_lat, filtered_lon, filtered_IceData)
            #    mapPlotter.accumulateDataToMap(filtered_lat, filtered_lon, sigmaest)
            # Go to next track
            trackNumber = trackNumber + 1
            ############
        f.close()
    
    
    print('Plotting map')
    mapPlotter.plotMap("\sigma_{Est} / dB")
    
    #Plot the data as a histogram
    print('Plotting histogram')
    plt.hist2d(x,y,bins=200, cmap = 'jet')
    #plt.figsize=(24, 40)

    #Modify plot
    plt.xlabel('Antenna Gain [dB]')
    plt.ylabel('DDM Peak SNR [dB]')
    plt.title('Histogram of ' + yName + ' and ' + xName)
    #plt.xlim([-12,max(x)])
    #plt.ylim([-12,20])
    plt.colorbar()
    #plt.tight_layout()
    plt.show()
    
    #Plot the data on a map
    
    #mapPlotter2.plotMap()

ReadNSIDCIceData.get_NSDIC_data(startTime, stopTime, dataPath = NSIDCPath)

RunMERRBySLevel1bHistogramAndMapExample()