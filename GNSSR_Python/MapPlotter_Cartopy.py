import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from pyproj import CRS
from pyproj import Transformer

class MapPlotter:
    """A class that will plot a map in the specified projection.
    """
    
    def __init__(self, boxSize, plotType = "FullRobinson"):
        """Generate the map grid points
         Parameter: boxSize: Map grid in km (at center of projection)"""
        
        self.boxSize = boxSize
        
        self.fig = plt.figure(figsize=[20, 10])
        
        if plotType == "South":
            self.transfrom = ccrs.SouthPolarStereo()
            self.ax1 = plt.axes(projection=self.transfrom)
            self.ax1.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        elif plotType == "North":
            self.transfrom = ccrs.NorthPolarStereo()
            self.ax1 = plt.axes(projection=self.transfrom)
            self.ax1.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
        elif plotType == "FullRobinson":
            self.transfrom = ccrs.Robinson()
            self.ax1 = plt.axes(projection=self.transfrom)
            self.ax1.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())
            
        
        self.ax1.coastlines()
        self.ax1.gridlines()
        self.extent = self.ax1.get_extent()
        
        self.transfrom_pyproj = CRS.from_dict(self.transfrom.proj4_params)
        self.transto_pyproj = CRS.from_epsg(4326)
        
        self.transformer = Transformer.from_proj(self.transfrom_pyproj, self.transto_pyproj)
        self.transformer_reverse = Transformer.from_proj(self.transto_pyproj, self.transfrom_pyproj)
        
        self.rows     = np.arange(self.extent[0], self.extent[1], self.boxSize)
        self.coloumns = np.arange(self.extent[2], self.extent[3], self.boxSize)
        
        self.accum = np.zeros((self.rows.shape[0],self.coloumns.shape[0]))
        self.count = np.zeros((self.rows.shape[0],self.coloumns.shape[0]))
        
        
    def accumulateDataToMap(self, latitudeArray, longitudeArray, vals):
        """ Function that adds data to map accumulation and count of data
        so that average can be calculated.
        Inputs: 1D Numpy array of each of latitude, longitude and values
        """
        #longitudeArray = np.zeros(1);
        #latitudeArray =  np.zeros(1);
        #vals = np.ones(1);
        
        yScaled, xScaled = self.transformer_reverse.transform(latitudeArray, longitudeArray)
        
        #print(xScaled, yScaled)
        
        xidx = np.zeros(xScaled.shape)
        yidx = np.zeros(xScaled.shape)
        
        xmin = np.min(self.rows)
        xmax = np.max(self.rows)
        ymin = np.min(self.coloumns)
        ymax = np.max(self.coloumns)
        
        for i in range(0, xScaled.shape[0]):
            xidx_t = (np.abs(self.rows     - xScaled[i])).argmin()
            yidx_t = (np.abs(self.coloumns - yScaled[i])).argmin()
            xidx = xidx_t.astype(np.int32)
            yidx = yidx_t.astype(np.int32)
            
            
            if ((xScaled[i] > xmin) and (xScaled[i] < xmax) and (yScaled[i] > ymin) and (yScaled[i] < ymax)):
                self.accum[xidx, yidx] += vals[i]
                self.count[xidx, yidx] += 1
        
        
    def plotMap(self, label):
        '''Plot the map'''
        
        self.accum[self.count == 0] = np.NaN
        
        self.accum = np.divide(self.accum, self.count)
        
        im = self.ax1.pcolormesh(self.coloumns, self.rows, self.accum, transform=self.transfrom, cmap=plt.get_cmap('jet'))
        cb = self.fig.colorbar(im, ax = self.ax1)
        cb.set_label(label)
        plt.show()
        

#mpll = MapPlotter(1000)
#mpll.accumulateDataToMap(np.array([-90, -80, -70, -60, -50]), np.array([0, 0, 0, 0, 0]), np.array([10, 10, 10, 10, 10]))

#mps = mpll.count

#mpll.plotMap()