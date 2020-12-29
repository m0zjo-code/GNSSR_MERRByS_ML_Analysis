#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 16:09:40 2020

@author: Jonathan Rawlinson - M0ZJO
"""
import rasterio
from pyproj import Proj, Transformer
import datetime, os
from pathlib import Path
from ftplib import FTP
import matplotlib.pyplot as plt

DEBUG = False

remote_dataset_url = "sidads.colorado.edu"
remote_dataset_mainpath = "DATASETS/NOAA/G02135/"

areas = ["south", "north"]

def extract_val_setup(timestamp, lat, lon, dataPath = "Data/IceData/"):
    """ Extracts a timestamped value from a NSIDC GeoTIFF File
    
    Inputs:
        timestamp = datetime struct of sample
        lat = sample latitude
        lon = sample longitude
        dataPath = path to GeoTIFF files
    
    Outputs:
        GeoTIFF raw value - please see https://nsidc.org/sites/nsidc.org/files/G02135-V3.0_0.pdf
    
    """
    local_path = os.path.join(os.getcwd(), dataPath)
    Path(local_path).mkdir(parents=True, exist_ok=True)
    
    if lat < 0:
        filename = generate_NSIDC_filename(timestamp, "S")
        area = areas[0]
    elif lat >= 0:
        filename = generate_NSIDC_filename(timestamp, "N")
        area = areas[1]
    else:
        print("=== Invalid Ice Area? ===")
        raise ValueError
    
    local_filename = local_path + filename
    dataset = rasterio.open(local_filename)
    
    if DEBUG:
        rasterio.plot.show(dataset)
    
    ice_data = dataset.read(1)
    
    rev_xform = ~dataset.transform
    outProj = Proj(dataset.crs)
    inProj = Proj('epsg:4326')
    
    coordxform = Transformer.from_proj(inProj, outProj)
    # print("=== Proj Setup Complete ===")
    
    return [ice_data, coordxform, rev_xform, area]

def get_val(lat, lon, ice_data, coordxform, rev_xform):
    x_f,y_f = coordxform.transform(lat,lon)
    # print("=== Get Val Start ===")
    # print(x_f, y_f)
    # print(coordxform)
    # print(rev_xform)
    # print("=== Xform ===")
    lookup = rev_xform * (x_f, y_f)
    # print(lat, lon)
    # print(lookup)
    x_lookup = round(lookup[0])
    y_lookup = round(lookup[1])
    if (x_lookup >= 0) and (x_lookup < ice_data.shape[1]) and (y_lookup >= 0) and (y_lookup < ice_data.shape[0]):
        tiff_value = ice_data[y_lookup, x_lookup]
        #print(x_lookup, y_lookup, ice_data.shape, "Accepted")
    else:
        #print(lookup, x_lookup, y_lookup)
        tiff_value = -1
        #print(x_lookup, y_lookup, ice_data.shape, "Rejected")
    
    return tiff_value

# From: https://stackoverflow.com/questions/1060279/iterating-through-a-range-of-dates-in-python
def daterange(start_date, end_date):
    """ Provides a list of dates (per day) """
    for n in range(int((end_date.date() - start_date.date()).days)+1):
        yield start_date + datetime.timedelta(n)

def generate_NSIDC_filename(timestamp, area):
    """ Generates the sea ice conc. filename """
    ts_str = timestamp.strftime("%Y%m%d")
    filename = "%s_%s_concentration_v3.0.tif" % (area, ts_str)
    #filename = "%s_%s_extent_v3.0.tif" % (area, ts_str)
    return filename

def get_NSDIC_data(startTime, stopTime, dataPath = "Data/IceData/"):
    """ Downloads data from NSIDC FTP server between two dates (inclusive)
    
    Inputs:
        startTime = start timestamp of requested data 
        stopTime = stop timestamp of requested data
        
    """
    
    ftp = FTP(remote_dataset_url)
    ftp.login()
    #ftp.retrlines("LIST")
    
    for timestamp in daterange(startTime, stopTime):
        for area in areas:
            if area == "north":
                filename = generate_NSIDC_filename(timestamp, "N")
            elif area == "south":
                filename = generate_NSIDC_filename(timestamp, "S")
            else:
                print("=== Invalid Ice Area - Are you on planet Earth?!? ===")
                raise ValueError
            
            local_path = os.path.join(os.getcwd(), dataPath)
            Path(local_path).mkdir(parents=True, exist_ok=True)
            local_filename = local_path + filename
            
            if os.path.isfile(local_filename):
                print (filename, " == Cached, no download required")
                
            else:
                print (filename, " == Not Cached, attempting download")           
                ts_month = timestamp.strftime("%m_%b")
                ts_year = timestamp.strftime("%Y")
                
                remote_filename = "%s/%s/daily/geotiff/%s/%s/%s" % (remote_dataset_mainpath, area, ts_year, ts_month, filename)
            
                with open(local_filename, 'wb') as f:
                    print("Downloading:", remote_filename)
                    ftp.retrbinary('RETR ' + remote_filename, f.write)
    ftp.quit()
    
# How to use:
# startTime = datetime.datetime(2017, 2, 1, 3, 0, 0)
# stopTime = datetime.datetime(2017, 2, 3, 21, 0, 0)
# get_NSDIC_data(startTime, stopTime)
# sampleTime = datetime.datetime(2017, 2, 2, 3, 0, 0)
# print(extract_tiff_val(sampleTime, -75, 0))
