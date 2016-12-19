.. _get-started:

Getting Started
---------------

GeoRaster supports two main types of image: those with a single band of data, and those with multiple bands of data. These two types of image are loaded in using georaster.SingleBandRaster or georaster.MultiBandRaster respectively. Once a raster is loaded in then most of the functions are identical between these two types of GeoRaster object. However, most geo-referenced images have only a single band, so that's what we'll introduce here.

First up, let's import GeoRaster::
	
	import georaster

The simplest way of using GeoRaster is to open files on your computer which are already georeferenced. This works as follows::
	
	my_image = georaster.SingleBandRaster('myfile.tif')
	print(my_image.extent)

In this example, we have opened the file `myfile.tif` and then printed out its extent in the native coordinate system of the file.

Using matplotlib we can take a look at the image itself::

	import matplotlib.pyplot as plt
	plt.imshow(my_image.r, extent=my_image.extent)

This will open up a figure window showing the image.

Here are the most useful attributes of a Single- or MultiBandRaster:

* im.r - a NumPy array of the raster data. In the case of a SingleBandRaster then this is a 2-D NumPy array. In the case of a MultiBandRaster this is a [y*x*bands] NumPy array (see 'Working with multi-band data below').
* im.extent - the extent of the raster in the coordinate system of the raster, ordered as `(left, right, bottom, top)` (AKA `(xmin, xmax, ymin, ymax)`)
* im.srs : an OGR SpatialReference representation of the raster's projection.
* im.trans : a GeoTransform tuple describing the raster (see _load-existing-data for more information).



Working with multi-band data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned above, multi-band data are loaded into `im.r` as a NumPy `[y*x*bands]` array. Use the convenience function `im.gdal_band` to retrieve a specific band according to its number as numbered by GDAL:

	import matplotlib.pyplot as plt
	# Load a raster with 5 bands
	im = georaster.MultiBandRaster('my-5band-file.tif')
	plt.imshow(im.r[:, :, im.gdal_band(3)])

In this case we are displayed the band labelled by GDAL as number 3. See _load-existing-data to find out how to load in only specific bands.



Get geo-referencing without loading image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes it's useful to find out information such as the extent of resolution of an image, without loading the actual image itself into memory. You can do this as follows::

	my_image = georaster.SingleBandRaster('myfile.tif', load_data=False)
	print(my_image.extent)
	