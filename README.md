# GeoRaster - easy use of geographic rasters in Python #

This package makes it easy to load, query and save geographic raster datasets in the Python programming language. The package uses Geospatial Data Abstraction Library (GDAL, http://www.gdal.org/) bindings, and so in a single command can import any geo-referenced dataset that is understood by GDAL (http://www.gdal.org/formats_list.html), complete with all geo-referencing information and various helper functions.

GeoRaster is compatible with Python 2.4-3.x. It requires GDAL and its Python bindings to be installed in your environment.

There are two basic types of raster: either a single-band dataset, which you load into a `SingleBandRaster` object, or a dataset containing multiple bands to be loaded, which you load into a `MultiBandRaster` object.

There is also an 'advanced' option where you can load a raster dataset, manually specifying your geo-referencing information. See example below.

Current release info
====================

| Name | Downloads | Version | Platforms |
| --- | --- | --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-georaster-green.svg)](https://anaconda.org/conda-forge/georaster) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/georaster.svg)](https://anaconda.org/conda-forge/georaster) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/georaster.svg)](https://anaconda.org/conda-forge/georaster) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/georaster.svg)](https://anaconda.org/conda-forge/georaster) |


# Installation from conda-forge #

    conda install -c conda-forge georaster


# Full Documentation #

http://georaster.readthedocs.io


# Examples #

Before doing anything you must import the package.

    import georaster

The examples below also require matplotlib:

    import matplotlib.pyplot as plt


## Load a GeoTIFF with a single band of data ##

Load the image with a single command:

    my_image = georaster.SingleBandRaster('myfile.tif')

The single data band is loaded into the `r` attribute of `my_image`. Use the `extent` attribute of the `my_image` object to set the coordinates of the plotted image:

    plt.imshow(my_image.r,extent=my_image.extent)


## Single band of data, loading a subset area of the image ##

In lat/lon (WGS84) - note that this will also set the class georeferencing 
information to match (i.e. self.nx, .yx, .extent, .xoffset, .yoffset):

    my_image = georaster.SingleBandRaster('myfile.tif',load_data=(lonll,lonur,latll,latur),latlon=True)

Or in projection system of the image:

    my_image = georaster.SingleBandRaster('myfile.tif',load_data=
                                            (xstart,xend,ystart,yend),
                                            latlon=False)


## Just get the georeferencing info, without also loading data into memory ##
Each class works as a wrapper to the GDAL API. A raster dataset can be loaded without needing to load the actual data as well, which is useful for querying geo-referencing information without memory overheads. Simply set the `load_data` flag to `False`:

    my_image = georaster.SingleBandRaster('myfile.tif',load_data=False)
    print(my_image.srs.GetProjParm('central_meridian'))

(See the 'Accessing geo-referencing information' section below for informtation on `srs`)


## Raster dataset with multiple bands, load just one band ##
For example, load GDAL band 2:

    my_image = georaster.MultiBandRaster('myfile.tif',band=2)


## Raster dataset with multiple bands, loading all bands ##

    my_image = georaster.MultiBandRaster('myfile.tif')
    plt.imshow(my_image.r)


## Raster dataset with multiple bands, loading just a couple of them ##

    my_image = georaster.MultiBandRaster('myfile.tif',bands=[1,3])
    plt.imshow(my_image.r[:,:,my_image.gdal_band(3)])


## Load dataset, providing your own geo-referencing ##

    from osgeo import osr
    spatial_ref = osr.SpatialReference()
    # This georef example is from http://nsidc.org/data/docs/daac/nsidc0092_greenland_ice_thickness.gd.html
    spatial_ref.ImportFromProj4('+proj=stere +lat_0=90 +lat_ts=71 +lon_0=-39 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs')    
    my_image = georaster.SingleBandRaster('myfile.tif',
                geo_transform=(-800000,5000,0,-600000,0,-5000),
                spatial_ref=ref)


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


## Documentation

Official documentation will shortly be hosted on ReadTheDocs.


## Get in touch

* Report bugs, suggest features or view the code on GitHub.


## History

GeoRaster was initially developed by Andrew Tedstone in 2013 whilst at the University of Edinburgh. Andrew Tedstone and Amaury Dehecq continue to develop and maintain the package. GeoRaster was made publicly available in January 2016.


## License

GeoRaster is licenced under GNU Lesser General Public License (LGPLv3).


