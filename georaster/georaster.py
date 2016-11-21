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
                    Remember that this is useless if you have provided your
                    own geo-referencing information via the geo_transform and
                    spatial_srs arguments!

    class.srs   :   an OGR Spatial Reference object representation of the 
                    dataset.
                        More information on API:
                        http://www.gdal.org/classOGRSpatialReference.html

    class.proj  :   a pyproj coordinate conversion function between the 
                    dataset coordinate system and lat/lon.

    class.extent :  tuple of the corners of the dataset in native coordinate
                    system, as (left,right,bottom,top).

    class.trans : GeoTransform tuple of the dataset.

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
in lat/lon (WGS84) - note that this will also set the class georeferencing 
information to match (i.e. self.nx, .yx, .extent, .xoffset, .yoffset):
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
>>> print(my_image.srs.GetProjParm('central_meridian'))

MultiBandRaster, loading all bands:
>>> my_image = georaster.MultiBandRaster('myfile.tif')
>>> plt.imshow(my_image.r)

MultiBandRaster, loading just a couple of bands:
>>> my_image = georaster.MultiBandRaster('myfile.tif',bands=[1,3])
>>> plt.imshow(my_image.r[:,:,my_image.gdal_band(3)])

Override georeferencing provided with file (either Single or MultiBand):
>>> from osgeo import osr
>>> ref = osr.SpatialReference()
>>> ref.ImportFromProj4('your proj4 string, e.g. +proj=stere ...')
>>> my_image = georaster.SingleBandRaster('myfile.tif',
                geo_transform=(160,5000,0,-120,0,-5000),
                spatial_ref=ref)

For more georaster examples see the docstrings which accompany each function.


Created on Wed Oct 23 12:06:16 2013

Change history prior to 2016/01/19 is in atedstone/geoutils repo.

@author: Andrew Tedstone (a.j.tedstone@bristol.ac.uk)
@author: Amaury Dehecq

"""

import numpy as np
from scipy import ndimage
try:
    from scipy.stats import nanmean
except ImportError:
    from numpy import nanmean
from osgeo import osr, gdal, ogr
try:
    import pyproj
except ImportError:
    import mpl_toolkits.basemap.pyproj as pyproj

# By default, GDAL does not raise exceptions - enable them
# See http://trac.osgeo.org/gdal/wiki/PythonGotchas
gdal.UseExceptions()
from warnings import warn

"""
Information on map -> pixel conversion

We use the formulas :
Xgeo = g0 + Xpixel*g1 + Ypixel*g2
Ygeo = g3 + Xpixel*g4 + Ypixel*g5
where g = ds.GetGeoTransform()
given on http://www.gdal.org/gdal_datamodel.html

(Xgeo, Ygeo) is the position of the upper-left corner of the cell, so the 
cell center is located at position (Xpixel+0.5,Ypixel+0.5)
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
        self.nx : x size of the raster. If a subset area of the raster is
            loaded then nx will correspond to the x size of the subset area.
        self.ny : y size of the raster. As nx.
        self.xres, self.yres : pixel sizes in x and y dimensions.
        self.x0, self.y0 : the offsets in x and y of the loaded raster area. 
            These will be zero unless a subset area has been loaded.

    """      
    # Filepath and name
    ds_file = None
    # GDAL handle to the dataset
    ds = None
    # GeoTransform 
    trans = None
    #: Extent of raster in order understood by Basemap, (xll,xur,yll,yur)
    extent = None
    # SRS
    srs = None
    # pyproj Projection
    proj = None
    # Raster size
    nx = None
    ny = None
    # Pixel size
    xres = None
    yres = None
    # Subset offsets in pixels
    x0 = None
    y0 = None
  


    def __del__(self):
        """ Close the gdal link to the dataset on object destruction """
        self.ds = None



    def _load_ds(self,ds_filename,spatial_ref=None,geo_transform=None):
        """ Load link to data file and set up georeferencing 

        Parameters:
            ds_filename : string, path to file

            To use file georeferencing, leave spatial_ref and geo_transform 
            set as None. To explicitly specify georeferencing:
                spatial_ref : OSR SpatialReference instance
                geo_transform : geo-transform tuple

        """

        # Load dataset from a file
        if isinstance(ds_filename,str):  
            self.ds_file = ds_filename
            self.ds = gdal.Open(ds_filename,0)
        # Or load GDAL Dataset (already in memory)
        elif isinstance(ds_filename,gdal.Dataset):  
            self.ds = ds_filename
            self.ds_file = ds_filename.GetDescription()

        # Check that some georeferencing information is available
        if self.ds.GetProjection() == '' and spatial_ref == None:
            warn('Warning : No georeferencing information associated to image!')

        # If user attempting to use their own georeferencing then make sure
        # that they have provided both required arguments
        if (spatial_ref == None and geo_transform != None) or \
           (spatial_ref != None and geo_transform == None):
            print('You must set both spatial_ref and geo_transform.')
            raise RuntimeError


        ## Start to load the geo-referencing ...

        if spatial_ref == None:
            # Use the georeferencing of the file
            self.trans = self.ds.GetGeoTransform()
            # Spatial Reference System
            self.srs = osr.SpatialReference()
            self.srs.ImportFromWkt(self.ds.GetProjection())
        else:
            # Set up georeferencing using user-provided information
            # GeoTransform
            self.trans = geo_transform
            # Spatial Reference System
            self.srs = spatial_ref
            
        # Create extent tuple in native dataset coordinates
        self.extent = (self.trans[0], 
                       self.trans[0] + self.ds.RasterXSize*self.trans[1], 
                       self.trans[3] + self.ds.RasterYSize*self.trans[5], 
                       self.trans[3])

        # Pixel size
        self.xres = self.trans[1]
        self.yres = self.trans[5]

        # Raster size
        self.nx = self.ds.RasterXSize
        self.ny = self.ds.RasterYSize
        
        # Offset of the first pixel (non-zero if only subset is read)
        self.x0 = 0
        self.y0 = 0            

        # Load projection if there is one
        if self.srs.IsProjected():
            self.proj = pyproj.Proj(self.srs.ExportToProj4())
 


    @classmethod
    def from_array(cls, raster, geo_transform, proj4, 
        gdal_dtype=gdal.GDT_Float32, nodata=None):
        """ Create georaster from numpy array and georeferencing info

        Parameters:
            raster : 2-D NumPy array of raster to load
            geo_transform : a Geographic Transform tuple of the form 
                            (top left x, w-e cell size, 0, top left y, 0, 
                             n-s cell size (-ve))
            proj4 : a proj4 string representing the raster projection
            gdal_dtype : a gdal data type (default gdal.GDT_Float32)
            nodata : None or the nodata value for this array

        Returns:
            georaster instance (in-memory representation of data only)

         """

        if len(raster.shape) > 2:
            nbands = raster.shape[2]
        else:
            nbands = 1

        # Create a GDAL memory raster to hold the input array
        mem_drv = gdal.GetDriverByName('MEM')
        source_ds = mem_drv.Create('', raster.shape[1], raster.shape[0],
                         nbands, gdal_dtype)

        # Set geo-referencing
        source_ds.SetGeoTransform(geo_transform)
        srs = osr.SpatialReference()
        srs.ImportFromProj4(proj4)
        source_ds.SetProjection(srs.ExportToWkt())

        # Write input array to the GDAL memory raster
        for b in range(0,nbands):
            if nbands > 1:
                r = raster[:,:,b]
            else:
                r = raster
            source_ds.GetRasterBand(b+1).WriteArray(r)
            if nodata != None:
                source_ds.GetRasterBand(b+1).SetNoDataValue(nodata)

        # Return a georaster instance instantiated by the GDAL raster
        return cls(source_ds)



    def get_extent_latlon(self):
        """ Return raster extent in lat/lon, (xll,xur,yll,yur) """
        if self.proj != None:
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
        if self.proj != None:
            xll,xur,yll,yur = self.get_extent_latlon()
        else:
            xll,xur,yll,yur = self.extent

        left,bottom = pyproj_obj(xll,yll)
        right,top = pyproj_obj(xur,yur)
        return (left,right,bottom,top)



    def coord_to_px(self,x,y,latlon=False,rounded=True,check_valid=True):
        """ Convert x,y coordinates into pixel coordinates of raster.

        x,y may be either in native coordinate system of raster or lat/lon.

        Parameters:
            x : float, x coordinate to convert.
            y : float, y coordinate to convert.
            latlon : boolean, default False. Set as True if bounds in lat/lon.
            rounded : if set to True, return the rounded pixel coordinates, otherwise return the float values
            check_valid : bool, if set to True, will check that all pixels are in the valid range. 

        Returns:
            (x_pixel,y_pixel)

        """

        # Convert coordinates to map system if provided in lat/lon and image
        # is projected (rather than geographic)
        if latlon == True and self.proj != None:
            x,y = self.proj(x,y)

        # Shift to the centre of the pixel
        x = np.array(x-self.xres/2)
        y = np.array(y-self.yres/2)

        g0, g1, g2, g3, g4, g5 = self.trans
        if g2 == 0:
            xPixel = (x - g0) / float(g1)
            yPixel = (y - g3 - xPixel*g4) / float(g5)
        else:
            xPixel = (y*g2 - x*g5 + g0*g5 - g2*g3) / float(g2*g4 - g1*g5)
            yPixel = (x - g0 - xPixel*g1) / float(g2)

        # Round if required
        if rounded==True:
            xPixel = np.round(xPixel)
            yPixel = np.round(yPixel)

        if check_valid==False:
            return xPixel, yPixel

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
            print("Warning : some points are out of domain for file")

        return xPixel_new, yPixel_new



    def read_single_band(self,band=1,downsampl=1):
        """ Return np array of specified band, defaults to band 1. 

        Parameters:
            band : int, number of band to read. Default 1.
            downsampl: int. Used to reduce the size of the image loaded. 
                Default is 1 (i.e. no down-sampling)
        Returns:
            np.array of specified band

        """
        band = int(band)

        if downsampl == 1:
            return self.ds.GetRasterBand(band).ReadAsArray()  
        else:
            arr = self.ds.GetRasterBand(band).ReadAsArray(
                buf_xsize=int(self.nx/downsampl), 
                buf_ysize=int(self.ny/downsampl))
            self.nx = int(self.nx/downsampl)
            self.ny = int(self.ny/downsampl)
            self.xres = self.xres*downsampl
            self.yres = self.yres*downsampl
            return arr
            


    def read_single_band_subset(self,bounds,latlon=False,extent=False,band=1,
                                update_info=False,downsampl=1):
        """ Return a subset area of the specified band of the dataset.

        Supply coordinates in native system or lat/lon.

        Parameters:
            bounds : tuple (xstart,xend,ystart,yend) (same format as extent)
            latlon : boolean, default False. Set as True if bounds in lat/lon.
            band : int, number of band to read. Default 1.
            extent: boolean, default False. If True, return extent of subset 
                area in the coordinate system of the image.
            update_info : boolean, default False. If True, set the 
                georeferencing information of the object to match this subset
                area (i.e. extent, nx, ny, x0, y0). You should only set this 
                to true if you are saving the array returned by this function
                into self.r!

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
        if downsampl==1:
            arr = self.ds.GetRasterBand(band).ReadAsArray(
                int(xpx1),
                int(ypx1),
                int(x_offset),
                int(y_offset))
        else:
            arr = self.ds.GetRasterBand(band).ReadAsArray(
                int(xpx1),
                int(ypx1),
                int(x_offset),
                int(y_offset),
                buf_xsize=int(x_offset/downsampl),
                buf_ysize=int(y_offset/downsampl))

        # Update image size
        # (top left x, w-e px res, 0, top left y, 0, n-s px res)
        trans = self.ds.GetGeoTransform()
        left = trans[0] + xpx1*trans[1]
        top = trans[3] + ypx1*trans[5]
        subset_extent = (left, left + x_offset*trans[1], 
                   top + y_offset*trans[5], top)
        if update_info == True:
            self.nx, self.ny = int(x_offset),int(y_offset) #arr.shape
            self.x0 = int(xpx1)
            self.y0 = int(ypx1)
            self.extent = subset_extent
            self.xres = self.xres*downsampl
            self.yres = self.yres*downsampl
            self.trans = (left, trans[1], 0, top, 0, trans[5])
        if extent == True:
            return (arr,subset_extent)
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

        if window != None:
            if window % 2 != 1:
                raise ValueError('Window must be an odd number.')

        def format_value(value):
            """ Check if valid value has been extracted """
            if type(value) == np.ndarray:
                if window != None:
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
        if window != None:
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
        """ Projected (or geographic) coordinates for specified pixels.
        if Xpixels=None and Ypixels=None (default), a grid with all 
        coordinates is returned
        if latlon=True, return the lat/lon coordinates
        Coordinates returned are for cell centres.

        Parameters:
            Xpixels : float or array, x-index of the pixels
            Ypixels : float or array, y-index of the pixels
            latlon : bool, if set to True, lat/lon coordinates are returned
        Returns:
            (Xgeo,Ygeo) : tuple, containing 1-d numpy arrays of coordinates 

        """
        
        if np.size(Xpixels) != np.size(Ypixels):
            print("Xpixels and Ypixels must have the same size")
            return 1

        if (Xpixels==None) & (Ypixels==None):
            Xpixels = np.arange(self.nx)
            Ypixels = np.arange(self.ny)
            Xpixels, Ypixels = np.meshgrid(Xpixels,Ypixels)
        else:
            Xpixels = np.array(Xpixels)
            Ypixels = np.array(Ypixels)
            
        # coordinates are at centre-cell, therefore the +0.5
        trans = self.trans
        Xgeo = trans[0] + (Xpixels+0.5)*trans[1] + (Ypixels+0.5)*trans[2]
        Ygeo = trans[3] + (Xpixels+0.5)*trans[4] + (Ypixels+0.5)*trans[5]

        if latlon==True:
            Xgeo, Ygeo = self.proj(Xgeo,Ygeo,inverse=True)

        return (Xgeo,Ygeo)



    def get_utm_zone(self):
        """ 
        Return UTM zone of raster from GDAL Projection information. 

        This function used to be more complex but is now a wrapper to an OGR
        Spatial Reference call. It remains maintained for backward 
        compatibility with dependent scripts.

        Returns:
            str : zone

        """
        return self.srs.GetUTMZone()
        


    def get_pixel_size(self):
        """ 
        Return pixel size of loaded raster. Maintained for backward 
        compatibility only, use self.xres and self.yres in new projects.

        Returns:
            floats: xres, yres

         """
        return self.xres, self.yres



    def reproject(self,target_srs,nx,ny,xmin,ymax,xres,yres,
            dtype=gdal.GDT_Float32,nodata=None,
            interp_type=gdal.GRA_NearestNeighbour,progress=False):
        """
        Reproject and resample dataset into another spatial reference system.

        Use to reproject/resample a dataset in-memory (rather than creating a
        new file), the function returns a new SingleBand or MultiBandRaster.

        CAUTION : not tested to work with datasets where you have provided
        georeferencing information manually by providing geo_transform and 
        spatial_ref when creating your georaster instance.
        
        Parameters:
            target_srs : Spatial Reference System to reproject to
            nx, ny : int, size of the output raster
            xmin, ymax : f, coordinates of the corners
            xres, yres : f, pixel size
            dtype : gdal.GTD_* data type (e.g GDT_Byte, GDT_Int32, GDT_Float64...)
            nodata : f, no data value in the input raster, default is None
            interp_type : gdal.GRA_* interpolation algorithm 
            (e.g GRA_NearestNeighbour, GRA_Bilinear, GRA_CubicSpline...), 
            default is GRA_NearestNeighbour
            progress : bool, set to True to display a progress bar

        Returns:
            A SingleBandRaster or MultiBandRaster object containing the
            reprojected image (in memory - not saved to file system)
            

        """
        # Create an in-memory raster
        mem_drv = gdal.GetDriverByName( 'MEM' )
        target_ds = mem_drv.Create('', nx, ny, 1, dtype)

        # Set the new geotransform
        new_geo = ( xmin, xres, 0, \
                    ymax, 0    , yres )
        target_ds.SetGeoTransform(new_geo)
        target_ds.SetProjection(target_srs.ExportToWkt())
    
        # Set the nodata value
        if nodata != None:
            for b in range(1,self.ds.RasterCount+1):
                inBand = self.ds.GetRasterBand(b)
                inBand.SetNoDataValue(nodata)
    
        # Perform the projection/resampling 
        if progress==True:
            res = gdal.ReprojectImage(self.ds, target_ds, None, None, 
                                      interp_type, 0.0, 0.0, gdal.TermProgress)
        else:
            res = gdal.ReprojectImage(self.ds, target_ds, None, None, 
                                      interp_type, 0.0, 0.0, None)
    
        # Load data
        if self.ds.RasterCount > 1:
            new_raster = MultiBandRaster(target_ds)
        else:
            new_raster = SingleBandRaster(target_ds)
        band = target_ds.GetRasterBand(1)
        data = band.ReadAsArray(0, 0, nx, ny)
        
        for b in range(1,new_raster.ds.RasterCount):
            new_raster.r[new_raster.r==0] = nodata
    
        return new_raster



    def interp(self,x,y,order=1,latlon=False,band=1):
        """
        Interpolate raster at points (x,y). 

        Values are extracted from self.r rather than the file itself, so if 
        changes have been made to self.r, they will apply.

        x,y may be either in native coordinate system of raster or lat/lon.
        
        WARNING : For now, values are considered as known at the 
        upper-left corner, whereas it should be in the centre of the cell.

        Parameters:
            x : float or array, x coordinate(s) to convert.
            y : float or array, y coordinate(s) to convert.
            order : order of the spline interpolation (range 0-5), 
              0=nearest-neighbor, 1=bilinear (default), 2-5 does not seem to 
              work with NaNs
            latlon : boolean, default False. Set as True if bounds in lat/lon.

        Returns:
            z_interp, interpolated raster values, same shape as lon and lat

        """

        # Get x,y coordinates in the matrix grid
        xpx1, ypx1 = self.coord_to_px(x,y,latlon=latlon,rounded=False)
        xi = xpx1# - self.x0
        yi = ypx1# - self.y0

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



    def save_geotiff(self,filename, dtype=gdal.GDT_Float32):
        """
        Save a GeoTIFF of the raster currently loaded.

        Georeferenced and subset according to the current raster.

        Inputs:
            filename : the path and filename to save the file to.
            dtype : GDAL datatype, defaults to Float32.

        Returns:
            True on success.

        """

        if self.r is None:
            raise ValueError('There are no image data loaded. No file can be created.')

        simple_write_geotiff(filename, self.r, self.trans, 
            wkt=self.srs.ExportToWkt(), dtype=dtype)



    def cartopy_proj(self):
        """ Return Cartopy Coordinate System for raster """
        import cartopy.crs as ccrs
        class cg(cartopy.crs.CRS): pass
        return cg(self.srs.ExportToProj4())



    def intersection(self,filename):
        """ Return coordinates of intersection between this image and another.

        Inputs:
        - filename : str, path to the second image
        
        Outputs:
        - xmin, xmax, ymin, ymax : extent of the intersection between the 2 images

        """

        # Create a polygon of the envelope of the first image
        xmin, xmax, ymin, ymax = self.extent
        wkt = "POLYGON ((%f %f, %f %f, %f %f, %f %f, %f %f))" %(xmin,ymin,xmin,ymax,xmax,ymax,xmax,ymin,xmin,ymin)
        poly1 = ogr.CreateGeometryFromWkt(wkt)

        # Create a polygon of the envelope of the second image
        img = SingleBandRaster(filename, load_data=False)
        xmin, xmax, ymin, ymax = img.extent
        wkt = "POLYGON ((%f %f, %f %f, %f %f, %f %f, %f %f))" %(xmin,ymin,xmin,ymax,xmax,ymax,xmax,ymin,xmin,ymin)
        poly2 = ogr.CreateGeometryFromWkt(wkt)

        # Compute intersection envelope
        intersect = poly1.Intersection(poly2)
        extent = intersect.GetEnvelope()

        # check that intersection is not void
        if intersect.GetArea()==0:
            print('Warning: Intersection is void')
            return 0
        else:
            return extent

        

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
    # Band datatype
    dtype = None



     


    def __init__(self,ds_filename,load_data=True,latlon=True,band=1,
        spatial_ref=None,geo_transform=None,downsampl=1):
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
            downsampl : default 1. Used to down-sample the image when loading it. 
                A value of 2 for example will multiply the resolution by 2. 

            Optionally, you can manually specify/override the georeferencing. 
            To do this you must set both of the following parameters:

            spatial_ref : a OSR SpatialReference instance
            geo_transform : a Geographic Transform tuple of the form 
                            (top left x, w-e cell size, 0, top left y, 0, 
                             n-s cell size (-ve))
            
        """

        # Do basic dataset loading - set up georeferencing
        self._load_ds(ds_filename,spatial_ref=spatial_ref,
                      geo_transform=geo_transform)

        # Import band datatype
        band_tmp = self.ds.GetRasterBand(band)
        self.dtype = gdal.GetDataTypeName(band_tmp.DataType)
        
        # Load entire image
        if load_data == True:
            self.r = self.read_single_band(band,downsampl=downsampl)

        # Or load just a subset region
        elif isinstance(load_data,tuple) or isinstance(load_data,list):
            if len(load_data) == 4:
                self.r = self.read_single_band_subset(load_data,latlon=latlon,
                    band=band,update_info=True,downsampl=downsampl)

        elif load_data == False:
            return

        else:
            print('Warning : load_data argument not understood. No data loaded.')


        
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


    def __init__(self,ds_filename,load_data=True,bands='all',latlon=True,
                 spatial_ref=None,geo_transform=None):
        """ Load a multi-band raster.

        Parameters:
            ds_filename : filename of dataset to load
            load_data : True, False or tuple (lonll,lonur,latll,latur)
            latlon : When load_data=tuple of coordinates, True if geographic, 
                     False if projected.
            bands : 'all', or tuple of raster bands. If tuple,
                MultiBandRaster.r will be a numpy array [y,x,b], where bands
                are indexed from 0 to n in the order specified in the tuple.

            Optionally, you can manually specify/override the georeferencing. 
            To do this you must set both of the following parameters:

            spatial_ref : a OSR SpatialReference instance
            geo_transform : a Geographic Transform tuple of the form 
                            (top left x, w-e cell size, 0, top left y, 0, 
                             n-s cell size (-ve))

        """

        self._load_ds(ds_filename,spatial_ref=spatial_ref,
                      geo_transform=geo_transform)

        if load_data != False:

            # First check which bands to load
            if bands == 'all':
                self.bands = range(1,self.ds.RasterCount+1)
            else:
                if isinstance(bands,tuple):
                    self.bands = bands
                else:
                    print('bands is not str "all" or of type tuple')
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



    def gdal_band(self,b):
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




def simple_write_geotiff(outfile,raster,geoTransform,
    wkt=None,proj4=None,mask=None,dtype=gdal.GDT_Float32):
    """ Save a GeoTIFF.
    
    Inputs:
        outfile : filename to save image to, if 'none', returns a memory raster
        raster : nbands x r x c
        geoTransform : tuple (top left x, w-e cell size, 0, top left y, 0, 
            n-s cell size (-ve))
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
    if wkt != None and proj4 != None:
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
    # Top left x, w-e pixel res, rotation, top left y, rotation, n-s pixel res
    dst_ds.SetGeoTransform(geoTransform)
      
    # Set the reference info 
    srs = osr.SpatialReference()
    if wkt != None:
        dst_ds.SetProjection(wkt)
    elif proj4 != None:
        srs.ImportFromProj4(proj4)
        dst_ds.SetProjection( srs.ExportToWkt() )
    
    # Write the band(s)
    if nbands > 1:
        for band in range(1,nbands+1):
            dst_ds.GetRasterBand(band).WriteArray(raster[band-1]) 
            if mask != None:
                dst_ds.GetRasterBand(band).GetMaskBand().WriteArray(mask)
    else:
        dst_ds.GetRasterBand(1).WriteArray(raster)
        dst_ds.GetRasterBand(1).SetNoDataValue(-999)
        if mask != None:
            dst_ds.GetRasterBand(1).GetMaskBand().WriteArray(mask)

    if outfile != 'none':
        # Close data set
        dst_ds = None
        return True 
    
    else:
        return dst_ds
