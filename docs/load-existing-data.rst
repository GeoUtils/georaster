.. _loading-data:

Options for loading in rasters
------------------------------

As explained in Getting Started, we can load in rasters that already exist in files as follows::

	# For an image with only a single band
	single_im = georaster.SingleBandRaster('my-singleband-image.tif')
	# For an image containing several bands
	multi_im = georaster.MultiBandRaster('my-multiband-image.tif')

This basic command loads both the raster metadata, and all the data within that raster. However, this isn't always what's needed.



Load in a smaller area of a raster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can specify the coordinates of a specific area of our raster to load in. This applies to both Single- and Multi-BandRasters. These coordinates can be in either Latitude/Longitude (WGS84) or the coordinate system of the raster. 

To use WGS84 coordinates::

	extent = (lonlowerleft, lonupperright, latllowerleft, latupperright)
	im = georaster.SingleBandRaster('myfile.tif', load_data=extent, latlon=True)

Or to use coordinate system of the image::

	extent = (xmin, xmax, ymin, ymax)
	im = georaster.SingleBandRaster('myfile.tif', load_data=extent)



Load only the information about a raster, not the image itself
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simply pass the flag `load_data=False` when you create the Single- or Multi-BandRaster object.



Only load select bands of a multi-band-raster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supply the numbers of the bands that you want to load as a tuple. These are the bands as numbered by GDAL, so begin from 1 (not 0)::

	im = georaster.MultiBandRaster('myfile.tif', bands=(1, 3))

We can then retrieve these individual bands of data as follows::

	# Plot band 3
	plt.imshow(im.r[:, :, im.gdal_band(3)]

You'll notice that this uses the convenience function `gdal_band()`. This converts the original GDAL band number into the location in which the band is stored in `im.r`, which is a multi-dimensional NumPy array.



Load a raster that doesn't have embedded geo-referencing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes we want to import a raster which doesn't have meta-data that GDAL (the library which underlies GeoRaster) can access. Instead, we can provide the geo-referencing manually when loading the image. To do this we need to specify a SpatialReference object and a geoTransform. This works with both Single- and Multi-BandRasters.

First create the SpatialReference object - here are a few options for specifying the projection::

	ref = georaster.osr.SpatialReference()
	# Import from Proj.4
	ref.ImportFromProj4('your proj4 string')
	# Or Well-Known-Text
	ref.ImportFromWkt('your WKT string')
	# Or EPSG code
	ref.ImportFromEPSG(3413)

Now create a geoTransform tuple that describes the position of the raster in space. A geoTransform has the form::

	trans = (tlx, W-E size, 0, tly, 0, N-S size)

Where:

* `tlx`: The X coordinate of the upper-left corner of the raster
* `W-E size`: The size of each cell from west to east
* `tly`: The Y coordinate of the upper-left corner of the raster
* `N-S size`: The size of each cell from north to south. Generally speaking this should be negative.

Provide these values in the same units as the projection system you have specified, most commonly this is metres.

Finally we use this information to load the raster with associated geo-referencing::

	im = georaster.SingleBandRaster('myfile.tif', spatial_ref=ref, 
			geo_transform=trans)

For more information on SpatialReference objects, the following links will help:

* `List of Spatial References <http://spatialreference.org/>`_
* `GDAL and OGR Cookbook - Projecting <https://pcjericks.github.io/py-gdalogr-cookbook/projection.html>`_
* `GDAL documentation for the Spatial Reference class <http://gdal.org/python/osgeo.osr.SpatialReference-class.html>`_