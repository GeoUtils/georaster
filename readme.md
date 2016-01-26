# GeoRaster - easy use of geographic rasters in Python #

This package makes it easy to load, query and save geographic raster datasets in the Python programming language. The package uses Geospatial Data Abstraction Library (GDAL, http://www.gdal.org/) bindings, and so in a single command can import any geo-referenced dataset that is understood by GDAL (http://www.gdal.org/formats_list.html), complete with all geo-referencing information and various helper functions.

GeoRaster is compatible with Python 2.4-3.x. It requires GDAL and its Python bindings to be installed in your environment.

There are two basic types of raster: either a single-band dataset, which you load into a `SingleBandRaster` object, or a dataset containing multiple bands to be loaded, which you load into a `MultiBandRaster` object.


# Examples #

Before doing anything you must import the package.

>>> import georaster

The examples below also require matplotlib:

>>> import matplotlib.pyplot as plt


## Load a GeoTIFF with a single band of data ##

Load the image with a single command:

>>> my_image = georaster.SingleBandRaster('myfile.tif')

The single data band is loaded into the `r` attribute of `my_image`. Use the `extent` attribute of the `my_image` object to set the coordinates of the plotted image:

>>> plt.imshow(my_image.r,extent=my_image.extent)


## Single band of data, loading a subset area of the image ##

In lat/lon (WGS84) - note that this will also set the class georeferencing 
information to match (i.e. self.nx, .yx, .extent, .xoffset, .yoffset):

>>> my_image = georaster.SingleBandRaster('myfile.tif',load_data=(lonll,lonur,latll,latur),latlon=True)

Or in projection system of the image:

>>> my_image = georaster.SingleBandRaster('myfile.tif',load_data=
                                            (xstart,xend,ystart,yend),
                                            latlon=False)


## Just get the georeferencing info, without also loading data into memory ##
Each class works as a wrapper to the GDAL API. A raster dataset can be loaded without needing to load the actual data as well, which is useful for querying geo-referencing information without memory overheads. Simply set the `load_data` flag to `False`:

>>> my_image = georaster.SingleBandRaster('myfile.tif',load_data=False)
>>> print(my_image.srs.GetProjParm('central_meridian'))

(See the 'Accessing geo-referencing information' section below for informtation on `srs`)


## Raster dataset with multiple bands, load just one band ##
For example, load GDAL band 2:

>>> my_image = georaster.MultiBandRaster('myfile.tif',band=2)


## Raster dataset with multiple bands, loading all bands ##

>>> my_image = georaster.MultiBandRaster('myfile.tif')
>>> plt.imshow(my_image.r)


## Raster dataset with multiple bands, loading just a couple of them ##

>>> my_image = georaster.MultiBandRaster('myfile.tif',bands=[1,3])
>>> plt.imshow(my_image.r[:,:,my_image.gdal_band(3)])


# Accessing and using geo-referencing information #
Once an image is loaded, the object provides the following attributes:

- Raster.ds : the `GDAL` handle to the dataset, which provides access to all GDAL functions - see http://www.gdal.org/classGDALDataset.html

- Raster.srs : an `OGR Spatial Reference` object representation of the dataset - see http://www.gdal.org/classOGRSpatialReference.html

- Raster.proj : a `pyproj` coordinate conversion function between the dataset coordinate system and WGS84 (latitude/longitude)

- Raster.extent : a `tuple` of the corners of the dataset in its native coordinate system as (left, right, bottom, top).

- Raster.nx, ny : x and y sizes of the loaded raster area.

- Raster.xres, yres : x and y pixel resolution of loaded raster area.

- Raster.x0, y0 : the pixel offsets in x and y of the loaded area. These will be zero unless a subset area has been loaded (using `load_data=(tuple)`).

- Raster.get_extent_latlon() : returns WGS84 (lat/lon) extent of raster as tuple.

- Raster.get_extent_projected(pyproj_obj) : returns raster extent in the coordinate system specified by the pyproj object (e.g. provide a Basemap instance).

- Raster.coord_to_px(x,y,latlon=True/False) : convert x,y coodinates into raster pixel coordinates.

- Raster.coordinates() : return projected or geographic coordinates of the whole image (or, with optional arguments, a subset of the image)



