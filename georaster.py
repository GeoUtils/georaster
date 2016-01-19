# -*- coding: utf-8 -*-
"""
georaster.py

Classes to simplify reading geographic raster data and associated 
georeferencing information.

There are two classes available:

    SingleBandRaster    --  For loading datasets with only a single band of 
                            data
    MultiBandRaster     --  For loading datasets that contain multiple bands 
                            of data

Each class works as a wrapper to the GDAL API. A raster dataset can be loaded
into a class without needing to load the actual data as well, which is useful
for querying geo-referencing information without memory overheads.

Both classes provide comprehensive georeferencing information access via
the following attributes:

    class.ds    :   the GDAL handle to the dataset, which provides access to 
                    all GDAL functions, e.g. GetProjection, GetGeoTransform.
                        More information on API:
                        http://www.gdal.org/classGDALDataset.html

    class.srs   :   an OGR Spatial Reference object representation of the 
                    dataset.
                        More information on API:
                        http://www.gdal.org/classOGRSpatialReference.html

    class.proj  :   a pyproj coordinate conversion function between the 
                    dataset coordinate system and lat/lon.

    class.extent :  tuple of the corners of the dataset in native coordinate
                    system, as (left,right,bottom,top).

Additionally, georeferencing information which requires calculation 
can be accessed via a number of functions available in the class.


Raster access:
    
    class.r : numpy array, m*n for SingleBandRaster, m*n*bands for MultiBand.


Examples
--------

SingleBandRaster with data import:
>>> my_image = georaster.SingleBandRaster('myfile.tif')
>>> plt.imshow(my_image.r,extent=my_image.extent)
The data of a SingleBandRaster is made available via my_image.r as a numpy
array.

SingleBandRaster, loading a subset area of the image defined
in lat/lon (WGS84):
>>> my_image = georaster.SingleBandRaster('myfile.tif',load_data=
                                            (lonll,lonur,latll,latur),
                                            latlon=True)
Or in image projection system:
>>> my_image = georaster.SingleBandRaster('myfile.tif',load_data=
                                            (xstart,xend,ystart,yend),
                                            latlon=False)

SingleBandRaster to get some georeferencing info, without 
also loading data into memory:
>>> my_image = georaster.SingleBandRaster('myfile.tif',load_data=False)
>>> print my_image.srs.GetProjParm('central_meridian')

MultiBandRaster, loading all bands:
>>> my_image = georaster.MultiBandRaster('myfile.tif')
>>> plt.imshow(my_image.r)

MultiBandRaster, loading just a couple of bands:
>>> my_image = georaster.MultiBandRaster('myfile.tif',bands=[1,3])
>>> plt.imshow(my_image.r[:,:,my_image.gdal_band(3)])

Plotting a Mercator map with basemap. Georaster is used to 
load the image, find the value for lon_0, and to convert the extent of the 
image into the coordinate system used by the map object (by passing 
georaster the map object).
>>> from mpl_toolkits.basemap import Basemap
>>> myimg = georaster.SingleBandRaster(file_name)
>>> map = Basemap(projection='tmerc',...,
                  lon_0=myimg.srs.GetProjParm('central_meridian'))
>>> plt.imshow(myimg.r,extent=myimg.get_extent_projected(map))

For more complicated mapping take a look at the geoutils/plotmap.Map class.

For more georaster examples see the docstrings which accompany each function.


Created on Wed Oct 23 12:06:16 2013

@author: Andrew Tedstone (a.j.tedstone@ed.ac.uk)
"""

import numpy as np
from scipy import ndimage
from scipy.stats import nanmean
from osgeo import osr, gdal
import subprocess
import xml.etree.ElementTree as etree
import re
import mpl_toolkits.basemap.pyproj as pyproj

# By default, GDAL does not raise exceptions - enable them
# See http://trac.osgeo.org/gdal/wiki/PythonGotchas
gdal.UseExceptions()

"""
Information on map -> pixel conversion

We use the formulas :
Xgeo = g0 + Xpixel*g1 + Ypixel*g2
Ygeo = g3 + Xpixel*g4 + Ypixel*g5
where g = ds.GetGeoTransform()
given on http://www.gdal.org/gdal_datamodel.html

(Xgeo, Ygeo) is the position of the upper-left corner of the cell, so the cell center is located at position (Xpixel+0.5,Ypixel+0.5)
"""

class __Raster:
    """
    Attributes:
        ds_file : filepath and name
        ds : GDAL handle to dataset
        extent : extent of raster in order understood by basemap, 
                    [xll,xur,yll,yur], in raster coordinates
        srs : OSR SpatialReference object
        proj : pyproj conversion object raster coordinates<->lat/lon 

    """      
    # Filepath and name
    ds_file = None
    # GDAL handle to the dataset
    ds = None
    # Extent of raster in order understood by Basemap
    extent = None
    # SRS
    srs = None
    # pyproj Projection
    proj = None
  


    def __del__(self):
        """ Close the gdal link to the dataset on object destruction """
        self.ds = None



    def _load_ds(self,ds_filename):
        """ Load link to data file and set up georeferencing """
        if isinstance(ds_filename,str):  #input arg is file name
            self.ds_file = ds_filename
            self.ds = gdal.Open(ds_filename,0)
        elif isinstance(ds_filename,gdal.Dataset):  #input arg is gdal dataset
            self.ds = ds_filename
            self.ds_file = ds_filename.GetDescription()

        if self.ds.GetProjection() == '':
            print 'Specified image does not have any associated \
            georeferencing information.'
            raise RuntimeError

        trans = self.ds.GetGeoTransform()
        self.extent = (trans[0], trans[0] + self.ds.RasterXSize*trans[1], 
                       trans[3] + self.ds.RasterYSize*trans[5], trans[3])

        #Pixel size
        self.xres = trans[1]
        self.yres = trans[5]

        #Raster size
        self.nx = self.ds.RasterXSize
        self.ny = self.ds.RasterYSize
        
        #Offset of the first pixel (non-zero if only subset is read)
        self.x0 = 0
        self.y0 = 0

        self.srs = osr.SpatialReference()
        self.srs.ImportFromWkt(self.ds.GetProjection())

        if len(self.ds.GetProjection().split('PROJCS')) == 2:
            self.proj = pyproj.Proj(self.srs.ExportToProj4())

        band = self.ds.GetRasterBand(1)
        self.dtype = gdal.GetDataTypeName(band.DataType)
 
    def get_extent_latlon(self):
        """ Return raster extent in lat/lon, (xll,xur,yll,yur) """
        if self.proj <> None:
            left,bottom = self.proj(self.extent[0],self.extent[2],inverse=True)
            right,top = self.proj(self.extent[1],self.extent[3],inverse=True)
            return (left,right,bottom,top)
        else:
            return self.extent



    def get_extent_projected(self,pyproj_obj):
        """ Return raster extent in a projected coordinate system.

        This is particularly useful for converting raster extent to the 
        coordinate system of a Basemap instance.

        Parameters:
            pyproj_obj : A pyproj instance (such as a Basemap instance) of the
                system to convert into.

        Returns:
            (left,right,bottom,top)

        Example:
        >>> from mpl_toolkits.basemap import Basemap
        >>> my_im = georaster.SingleBandRaster('myfile.tif')
        >>> my_map = Basemap(...)
        >>> my_map.imshow(my_im.r,extent=my_im.get_extent_basemap(my_map))

        """
        if self.proj <> None:
            xll,xur,yll,yur = self.get_extent_latlon()
        else:
            xll,xur,yll,yur = self.extent

        left,bottom = pyproj_obj(xll,yll)
        right,top = pyproj_obj(xur,yur)
        return (left,right,bottom,top)



    def coord_to_px(self,x,y,latlon=False,rounded=True):
        """ Convert x,y coordinates into pixel coordinates of raster.

        x,y may be either in native coordinate system of raster or lat/lon.

        Parameters:
            x : float, x coordinate to convert.
            y : float, y coordinate to convert.
            latlon : boolean, default False. Set as True if bounds in lat/lon.
            rounded : if set to True, return the rounded pixel coordinates, otherwise return the float values

        Returns:
            (x_pixel,y_pixel)


        Map --> px conversion from:
        landsat_velocities/SupportFunctions.py/convertgeo2pix.py 
        (commit a3a1061a)

        """

        # Convert coordinates to map system if provided in lat/lon and image
        # is projected (rather than geographic)
        if latlon == True and self.proj <> None:
            x,y = self.proj(x,y)

        # Shift to the center of the pixel
        x = np.array(x-self.xres/2)
        y = np.array(y-self.yres/2)

        g0, g1, g2, g3, g4, g5 = self.ds.GetGeoTransform()
        if g2 == 0:
            xPixel = (x - g0) / float(g1)
            yPixel = (y - g3 - xPixel*g4) / float(g5)
        else:
            xPixel = (y*g2 - x*g5 + g0*g5 - g2*g3) / float(g2*g4 - g1*g5)
            yPixel = (x - g0 - xPixel*g1) / float(g2)

        #Round if required
        if rounded==True:
            xPixel = np.round(xPixel)
            yPixel = np.round(yPixel)

        # Check that pixel location is not outside image dimensions
        nx = self.ds.RasterXSize
        ny = self.ds.RasterYSize

        xPixel_new = np.copy(xPixel)
        yPixel_new = np.copy(yPixel)
        xPixel_new = np.fmin(xPixel_new,nx)
        yPixel_new = np.fmin(yPixel_new,ny)
        xPixel_new = np.fmax(xPixel_new,0)
        yPixel_new = np.fmax(yPixel_new,0)
        
        if np.any(xPixel_new!=xPixel) or np.any(yPixel_new!=yPixel):
            print "Warning : some points are out of domain for file %s" %(self.ds_file)

        return xPixel_new, yPixel_new



    def read_single_band(self,band=1):
        """ Return np array of specified band, defaults to band 1. """
        return self.ds.GetRasterBand(band).ReadAsArray()  



    def read_single_band_subset(self,bounds,latlon=False,extent=False,band=1):
        """ Return a subset area of the specified band of the dataset.

        Supply coordinates in native system or lat/lon.

        Parameters:
            bounds : tuple (xstart,xend,ystart,yend) (same format as extent)
            latlon : boolean, default False. Set as True if bounds in lat/lon.
            band : int, number of band to read. Default 1.
            extent: boolean, default False. If True, return extent of subset 
                area in the coordinate system of the image.

        Returns:
            np.array of subset area

        """
        
        left,right,bottom,top = bounds
        
        # Unlike the bounds tuple, which specifies bottom left and top right
        # coordinates, here we need top left and bottom right for the numpy
        # readAsArray implementation.    
        xpx1,ypx1 = self.coord_to_px(left,bottom,latlon=latlon)
        xpx2,ypx2 = self.coord_to_px(right,top,latlon=latlon)
        
        if xpx1 > xpx2:
            xpx1, xpx2 = xpx2, xpx1
        if ypx1 > ypx2:
            ypx1, ypx2 = ypx2, ypx1

        # Resulting pixel offsets
        x_offset = xpx2 - xpx1
        y_offset = ypx2 - ypx1

        # In special case of being called to read a single point, offset 1 px
        if x_offset == 0: x_offset = 1
        if y_offset == 0: y_offset = 1

        # Read array and return
        arr = self.ds.GetRasterBand(band).ReadAsArray(int(xpx1),int(ypx1),
                                                      int(x_offset),int(y_offset))

        # Update image size
        self.nx, self.ny = int(x_offset),int(y_offset) #arr.shape
        self.x0 = int(xpx1)
        self.y0 = int(ypx1)

        if extent == True:
            # (top left x, w-e px res, 0, top left y, 0, n-s px res)
            trans = self.ds.GetGeoTransform() 
            left = trans[0] + xpx1*trans[1]
            top = trans[3] + ypx1*trans[5]
            extent = (left, left + x_offset*trans[1], 
                       top + y_offset*trans[5], top)
            return (arr,extent)
        else:
            return arr



    def value_at_coords(self,x,y,latlon=False,band=None,system=None,
                        window=None,return_window=False):
        """ Extract the pixel value(s) at the specified coordinates.
        
        Extract pixel value of each band in dataset at the specified 
        coordinates. Alternatively, if band is specified, return only that
        band's pixel value.

        Optionally, return mean of pixels within a square window.
                
        Parameters:
            x : float, x coordinate.
            y : float, y coordinate.
            latlon : boolean, True if coordinates in WGS84, false if in 
                     native system of the raster.
            band : the GDAL Dataset band number to extract from.
            system : DEPRECATED but maintained for backward compatibility.
            window : int or None, expand area around coordinate to dimensions
                      window * window. window must be odd.
            return_window : boolean, default False. If True when window=int,
                            returns (mean,array) where array is the dataset 
                            extracted via the specified window size.

        Returns:
            if band specified, float of extracted pixel value.

            if band not specified, depending on number of bands in dataset:
                more than 1 band : dict of GDAL band number:float pixel value.
                just 1 band : float of extracted pixel value.

            if return_window = True, (value[s],array[s])


        Examples:
        >>> self.value_at_coords(-48.125,67.8901,window=3)
        Returns mean of a 3*3 window:
            v v v \
            v c v  | = float(mean)
            v v v /
        (c = provided coordinate, v= value of surrounding coordinate)
        """

        if window <> None:
            if window % 2 <> 1:
                raise ValueError('Window must be an odd number.')

        def format_value(value):
            """ Check if valid value has been extracted """
            if type(value) == np.ndarray:
                if window <> None:
                    value = nanmean(value.flatten())
                else:
                    value = value[0,0]
            else:
                value = None
            return value

        # Check the deprecated georeferencing system parameter.
        if system == 'wgs84':
            latlon = True


        # Convert coordinates to pixel space
        xpx,ypx = self.coord_to_px(x,y,latlon=latlon)
        # Decide what pixel coordinates to read:
        if window <> None:
            half_win = (window -1) / 2
            # Subtract start coordinates back to top left of window
            xpx = xpx - half_win
            ypx = ypx - half_win
            # Offset to read to == window
            xo = window
            yo = window
        else:
            # Start reading at xpx,ypx and read 1px each way
            xo = 1
            yo = 1

        #Make sure coordinates are int
        xpx = int(xpx)
        ypx = int(ypx)

        # Get values for all bands
        if band == None:

            # Deal with SingleBandRaster case
            if self.ds.RasterCount == 1:
                data = self.ds.GetRasterBand(1).ReadAsArray(xpx,ypx,xo,yo)
                value = format_value(data)
                win = data
            
            # Deal with MultiBandRaster case
            else:    
                value = {}
                for b in range(1,self.ds.RasterCount+1):
                    d = self.ds.GetRasterBand(b).ReadAsArray(xpx,ypx,xo,yo)
                    val = format_value(data)
                    # Store according to GDAL band numbers
                    value[b] = val
                    win[b] = data

        # Or just for specified band in MultiBandRaster case                
        elif isinstance(band,int):
            data = self.ds.GetRasterBand(band).ReadAsArray(xpx,ypx,xo,yo)
            value = format_value(data)
        else:
            raise ValueError('Value provided for band was not int or None.')

        if return_window == True:
            return (value,win)
        else:
            return value       

    

    def coordinates(self,Xpixels=None,Ypixels=None,latlon=False):
        """ Calculate projected (or geographic) coordinates for specified pixels.
        if Xpixels=None and Ypixels=None (default), a grid with all coordinates is returned
        if latlon=True, return the lat/lon coordinates

        Parameters:
            Xpixels : float or array, x-index of the pixels
            Ypixels : float or array, y-index of the pixels
            latlon : bool, if set to True, lat/lon coordinates are returned
        Returns:
            (Xgeo,Ygeo) : tuple, containing 1-d numpy arrays of each coordinates            
        """
        
        if np.size(Xpixels) != np.size(Ypixels):
            print "Xpixels and Ypixels must have the same size"
            return 1

        if (Xpixels==None) & (Ypixels==None):
            Xpixels = np.arange(self.nx)
            Ypixels = np.arange(self.ny)
            Xpixels, Ypixels = np.meshgrid(Xpixels,Ypixels)
        else:
            Xpixels = np.array(Xpixels)
            Ypixels = np.array(Ypixels)
            
        # + self.x0 is for offset if only a subset ahas been read
        # coordinates are at centre-cell, therefore the +0.5
        trans = self.ds.GetGeoTransform()
        Xgeo = trans[0] + (Xpixels+self.x0+0.5)*trans[1] + (Ypixels+self.y0+0.5)*trans[2]
        Ygeo = trans[3] + (Xpixels+self.x0+0.5)*trans[4] + (Ypixels+self.y0+0.5)*trans[5]

        if latlon==True:
            Xgeo, Ygeo = self.proj(Xgeo,Ygeo,inverse=True)

        return (Xgeo,Ygeo)



    def get_utm_zone(self):
        """ 
        Return UTM zone of raster from GDAL Projection information. 

        This function used to be more complex but is now a wrapper to an OGR
        Spatial Reference call. It remains maintained for backwards 
        compatibility with dependent scripts.

        Returns:
            str : zone

        """
        return self.srs.GetUTMZone()
        


    def get_pixel_size(self):
        """ 
        Return pixel size of loaded raster.

        Returns:
            floats: xres, yres

         """
        geotransform = self.ds.GetGeoTransform()
        xres = geotransform[1]
        yres = geotransform[5]
        return xres, yres


    def reproject(self,target_srs,nx,ny,xmin,ymax,xres,yres,dtype=gdal.GDT_Float32,nodata=None,interp_type=gdal.GRA_NearestNeighbour):
        """
        Reproject and resample a GDAL dataset into another spatial reference system.
        Input arguments :
        - target_srs : Sptial Reference System to reproject to
        - nx, ny : int, size of the output raster
        - xmin, ymax : f, coordinates of the corners
        - xres, yres : f, pixel size
        - dtype : gdal.GTD_* data type (e.g GDT_Byte, GDT_Int32, GDT_Float64...) default is GDT_Float32
        - nodata : f, no data value in the input raster, default is None
        - interp_type : gdal.GRA_* interpolation algorithm (e.g GRA_NearestNeighbour, GRA_Bilinear, GRA_CubicSpline...), default is GRA_NearestNeighbour
        """
        # Create an in-memory raster
        mem_drv = gdal.GetDriverByName( 'MEM' )
        target_ds = mem_drv.Create('', nx, ny, 1, dtype)

        # Set the new geotransform
        new_geo = ( xmin, xres, 0, \
                    ymax, 0    , yres )
        target_ds.SetGeoTransform(new_geo)
        target_ds.SetProjection(target_srs.ExportToWkt())
    
        # Set the no data value
        if nodata!=None:
            inBand = self.ds.GetRasterBand(1)
            inBand.SetNoDataValue(nodata)
    
        # Perform the projection/resampling 
        res = gdal.ReprojectImage(self.ds, target_ds, self.srs.ExportToWkt(), target_srs.ExportToWkt(), interp_type)
    
        # Load data
        band = target_ds.GetRasterBand(1)
        data = band.ReadAsArray(0, 0, nx, ny)
    
        # Set no data points to nodata value
        if nodata!=None:
            data[data==0] = nodata
    
        return data


    def interp(self,x,y,order=1,latlon=False):
        """
        Interpolate raster at points (x,y). Values are extracted from self.r rather than the file itself, so if changes have been made to self.r, they will apply.
        x,y may be either in native coordinate system of raster or lat/lon.
        
        WARNING : For now, values are considered as known at the upper-left corner, whereas it should be in the centre of the cell.

        Parameters:
            x : float or array, x coordinate(s) to convert.
            y : float or array, y coordinate(s) to convert.
            order : order of the spline interpolation (range 0-5), 0 : nearest-neighbor, 1 : bilinear (default), 2-5 does not seem to work with NaNs
            latlon : boolean, default False. Set as True if bounds in lat/lon.

        Returns:
            z_interp, interpolated raster values, same shape as lon and lat
        """

        # Get x,y coordinates in the matrix grid
        xpx1, ypx1 = self.coord_to_px(x,y,latlon=latlon,rounded=False)
        xi = xpx1 - self.x0
        yi = ypx1 - self.y0

        #Case coordinates are not an array
        if np.rank(xi)<1:
            xi = np.array([xi,])
            yi = np.array([yi,])
            
        # Check that pixel location is not outside image dimensions
        if np.any(xi<0) or np.any(xi>=self.ny) or np.any(yi<0) or np.any(y>=self.nx):
            print('Warning : some of the coordinates are not in dataset extent -> extrapolated value set to 0')


        #interpolated data
        z_interp = ndimage.map_coordinates(self.r, [yi, xi],order=order)

        return z_interp



class SingleBandRaster(__Raster):
    """ A geographic raster dataset with one band of data.
    
    Initialise with the file path to a single band raster dataset, of type
    understood by the GDAL library. Datasets in UTM are preferred.

    Attributes:
        ds_file : filepath and name
        ds : GDAL handle to dataset
        extent : extent of raster in order understood by basemap, 
                    [xll,xur,yll,yur], in raster coordinates
        srs : OSR SpatialReference object
        proj : pyproj conversion object raster coordinates<->lat/lon 
        r : numpy band of array data

    Example:
    >>> georaster.SingleBandRaster('myfile.tif',load_data=True|False)

    """
     
    # Numpy array of band data
    r = None
     
    def __init__(self,ds_filename,load_data=True,latlon=True,band=1):
        """ Construct object with raster from a single band dataset. 
        
        Parameters:
            ds_filename : filename of the dataset to import
            load_data : - True, to import the data into obj.r. 
                        - False, to not load any data.
                        - tuple (left, right, bottom, top) to load subset; 
                          obj.extent will be set to reflect subset area.
            latlon : default True. Only used if load_data=tuple. Set as False
                     if tuple is projected coordinates, True if WGS84.
            band : default 1. Specify GDAL band number to load. If you want to
                   load multiple bands at once use MultiBandRaster instead.
            
        """
        self._load_ds(ds_filename)     
        
        # Load entire image
        if load_data == True:
            self.r = self.read_single_band(band)

        # Or load just a subset region
        elif isinstance(load_data,tuple) or isinstance(load_data,list):
            if len(load_data) == 4:
                (self.r,self.extent) = self.read_single_band_subset(load_data,
                                        latlon=latlon,extent=True,band=band)

        elif load_data == False:
            return

        else:
            print 'Warning : load_data argument not understood. No data loaded.'

   
        
    def find_value_at_coords(self,x,y,**kwargs):
        """ (DEPRECATED) Extract the pixel value at the specified coordinates.
        
        This function is maintained for backward compatibility. New code 
        should call self.value_at_coords() instead.

        ----------------------------------------------------------------------

        Parameters:
            x : x coordinate in format of target dataset.
            y : y coordinate in format of target dataset.
        Returns:
            float of extracted pixel value.
        
        """
        return self.value_at_coords(x,y,**kwargs)
                


    def smooth(self,px=3):
        """ Apply gaussian filter of specified window px.
        
        Parameters:
            px : window size in pixels to supply to ndimage.gaussian_filter
        Returns:
            self.smoothed : smoothed version of self.r.
            
        """
        self.smoothed = ndimage.gaussian_filter(self.r,px)
        



class MultiBandRaster(__Raster):
    """ A geographic raster dataset with multiple bands of data.
    
    Initialise with the file path to a single band raster dataset, of type
    understood by the GDAL library. Datasets in UTM are preferred.

    Examples:
    [1] Load all bands of a raster:
    >>> georaster.MultiBandRaster("myfile.tif")

    [2] Load just bands 1 and 3 of a raster:
    >>> georaster.MultiBandRaster("myfile.tif",load_data=[1,3])

    [3] Don't load the data, just use the class for the georeferencing API:
    >>> georaster.MultiBandRaster("myfile.tif",load_data=False) 

    Attributes:
        r :         The raster data (if loaded). For the example cases above:

                    [1] - a np array of [rows,cols,bands]. Standard numpy 
                          slicing is used, ie. the array is zero-referenced.
                          E.g., extract band 2, which is the second band 
                          loaded:
                          >>> myRaster.r[:,:,1]

                          This can be simplified with gdal_band():
                          E.g., extract band 2:
                          >>> myRaster.r[:,:,myRaster.gdal_band(2)]

                    [2] - The same as [1]. The helper function is particularly 
                          useful in simplifying lookup of bands, e.g.:
                          >>> myRaster.r[:,:,myRaster.gdal_band(3)]
                          Rather than the less obvious:
                          >>> myRaster.r[:,:,1]
                          Which corresponds to the actual numpy location of 
                          that band.

                    [3] - r is set as None. No data can be accessed.


        bands :     list of GDAL band numbers which have been loaded, in the 
                    order corresponding to the order stored in 
                    r[rows,cols,bands].  

        ds_file :   filepath and name
        ds :        GDAL handle to dataset
        extent :    extent of raster in order understood by basemap, 
                    [xll,xur,yll,yur], in raster coordinates
        srs :       OSR SpatialReference object
        proj :      pyproj conversion object raster coordinates<->lat/lon 
        
    """

    # Either a numpy array if just one band, or a dict of numpy arrays if 
    # multiple, key is band number
    r = None

    # List of GDAL band numbers which have been loaded.
    bands = None


    def __init__(self,ds_filename,load_data=True,bands='all',latlon=True):
        """ Load a multi-band raster.

        Parameters:
            ds_filename : filename of dataset to load
            load_data : True, False or tuple (lonll,lonur,latll,latur)
            latlon : When load_data=tuple of coordinates, True if geographic, 
                     False if projected.
            bands : 'all', or tuple of raster bands. If tuple,
                MultiBandRaster.r will be a numpy array [y,x,b], where bands
                are indexed from 0 to n in the order specified in the tuple.

        """
        self._load_ds(ds_filename)

        if load_data <> False:

            # First check which bands to load
            if bands == 'all':
                self.bands = range(1,self.ds.RasterCount+1)
            else:
                if isinstance(bands,tuple):
                    self.bands = bands
                else:
                    print 'bands is not "all" or tuple'
                    raise ValueError

            # Loading whole dimensions of raster
            if load_data == True:
                self.r = np.zeros((self.ds.RasterYSize,self.ds.RasterXSize,
                               len(self.bands)))
                k = 0
                for b in self.bands:
                    self.r[:,:,k] = self.read_single_band(band=b)
                    k += 1

            # Loading geographic subset of raster
            elif isinstance(load_data,tuple):
                if len(load_data) == 4:
                    k = 0
                    for b in self.bands:
                        # If first band, create a storage object
                        if self.r == None:
                            (tmp,self.extent) = self.read_single_band_subset(load_data,
                                        latlon=latlon,extent=True,band=b)
                            self.r = np.zeros((tmp.shape[0],tmp.shape[1],
                               len(self.bands)))
                            self.r[:,:,k] = tmp
                        # Store subsequent bands in kth dimension of store.
                        else:
                            self.r[:,:,k] = self.read_single_band_subset(load_data,
                                        latlon=latlon,band=b)
                        k += 1

        # Don't load any data
        elif load_data == False:
            self.bands = None
            return

        else:
            raise ValueError('load_data was not understood (should be one \
             of True,False or tuple)')



    def gdal_band(b):
        """ Return numpy array location index for given GDAL band number. 

        Parameters:
            b : int, value of GDAL band number to lookup

        Returns:
            int, index location of band in self.r

        Example:
        >>> giveMeMyBand2 = myRaster.r[:,:,myRaster.gdal_band(2)]

        """

        # Check that more than 1 band has been loaded into memory.
        if self.bands == None:
            raise AttributeError('No data have been loaded.')
        if len(self.bands) == 1:
            raise AttributeError('Only 1 band of data has been loaded.')

        if isinstance(b,int):
            return self.bands.index(b)
        else:
            raise ValueError('B is must be an integer.') 







def simple_write_geotiff(outfile,raster,geoTransform,wkt=None,proj4=None,mask=None,dtype=gdal.GDT_Float32):
    """ Save a GeoTIFF.
    
    Inputs:
        outfile : filename to save image to, if 'none', returns a memory raster
        raster : nbands x r x c
        geoTransform : tuple (top left x, w-e cell size, 0, top left y, 0, n-s cell size (-ve))
        One of proj4 or wkt :
            proj4 : a proj4 string
            wkt : a WKT projection string
        dtype : gdal.GDT type (Byte : 1, Int32 : 5, Float32 : 6)

    -999 is specified as the NoData value.

    Outputs:
        A GeoTiff named outfile.
    
    Based on http://adventuresindevelopment.blogspot.com/2008/12/python-gdal-adding-geotiff-meta-data.html
    and http://www.gdal.org/gdal_tutorial.html
    """

    # Georeferencing sanity checks
    if wkt <> None and proj4 <> None:
        raise 'InputError: Both wkt and proj4 specified. Only specify one.'
    if wkt == None and proj4 == None:
        raise 'InputError: One of wkt or proj4 need to be specified.'

    # Check if the image is multi-band or not. 
    if raster.shape.__len__() == 3:
        nbands = raster.shape[0]    
        ydim = raster.shape[1]
        xdim = raster.shape[2]
    elif raster.shape.__len__() == 2:
        nbands = 1
        ydim = raster.shape[0]
        xdim = raster.shape[1]
         
    # Setup geotiff file.
    if outfile!='none':
        driver = gdal.GetDriverByName("GTiff")
    else:
        driver = gdal.GetDriverByName('MEM')

    dst_ds = driver.Create(outfile, xdim, ydim, nbands, dtype)
    # Top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    dst_ds.SetGeoTransform(geoTransform)
      
    # Set the reference info 
    srs = osr.SpatialReference()
    if wkt <> None:
        dst_ds.SetProjection(wkt)
    elif proj4 <> None:
        srs.ImportFromProj4(proj4)
        dst_ds.SetProjection( srs.ExportToWkt() )
    
    # Write the band(s)
    if nbands > 1:
        for band in range(1,nbands+1):
            dst_ds.GetRasterBand(band).WriteArray(raster[band-1]) 
            if mask <> None:
                dst_ds.GetRasterBand(band).GetMaskBand().WriteArray(mask)
    else:
        dst_ds.GetRasterBand(1).WriteArray(raster)
        dst_ds.GetRasterBand(1).SetNoDataValue(-999)
        if mask <> None:
            dst_ds.GetRasterBand(1).GetMaskBand().WriteArray(mask)

    if outfile!='none':
        # Close data set
        dst_ds = None
        return True 
    
    else:
        return dst_ds







def write_geotiff(raster,output_file,geo,proj4=False,wkt=False,mask=None):
    """ Save a GeoTIFF. DEPRECATED - DO NOT USE FOR NEW APPLICATIONS
    
    Inputs:
        raster - nbands x r x c
        output_file - filename to save image to
        geo - georeferencing information as dictionary. x (left coord), 
        y (top coord), xcellsize, ycellsize, datum, utmzone (if raster is projected in UTM)
        proj4 - a proj4 string, optional
        wkt - a WKT projection string, optional
        Only provide one of proj4 or wkt!

    Outputs:
        A GeoTiff named according to output_file.
    
    Based on http://adventuresindevelopment.blogspot.com/2008/12/python-gdal-adding-geotiff-meta-data.html
    and http://www.gdal.org/gdal_tutorial.html
    """   
    # Check if the image is multi-band or not. 
    if raster.shape.__len__() == 3:
        nbands = raster.shape[0]    
        ydim = raster.shape[1]
        xdim = raster.shape[2]
    elif raster.shape.__len__() == 2:
        nbands = 1
        ydim = raster.shape[0]
        xdim = raster.shape[1]
         
    # Setup geotiff file.
    gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'YES')
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(output_file, xdim, ydim, nbands, gdal.GDT_Float32)
    dst_ds.CreateMaskBand(gdal.GMF_PER_DATASET)
    # Top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    dst_ds.SetGeoTransform( [ geo['x'], geo['xcellsize'], 0, geo['y'], 0, geo['ycellsize'] ] )
      
    # Set the reference info 
    srs = osr.SpatialReference()
    if wkt == False:
        srs.ImportFromProj4(proj4)

        if geo.has_key('utmzone') == True:
            srs.SetUTM(geo['utmzone'],1)
        srs.SetWellKnownGeogCS(geo['datum'])
        dst_ds.SetProjection( srs.ExportToWkt() )
    else:
        dst_ds.SetProjection(wkt)
    
    # Write the band(s)
    if nbands > 1:
        for band in range(1,nbands+1):
            dst_ds.GetRasterBand(band).WriteArray(raster[band-1]) 
            if mask <> None:
                dst_ds.GetRasterBand(band).GetMaskBand().WriteArray(mask)
    else:
        dst_ds.GetRasterBand(1).WriteArray(raster)
        if mask <> None:
            dst_ds.GetRasterBand(1).GetMaskBand().WriteArray(mask)

    # Close data set
    dst_ds = None
    return True 


        
        
