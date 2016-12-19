.. _create-georeferenced-data:


Create geo-referenced data
--------------------------

GeoRaster can be used to add geo-referencing to un-referenced data, and write
those data out to GeoTIFF files.



Add geo-referencing
~~~~~~~~~~~~~~~~~~~

Our starting point is that we have a NumPy array of dimensions *`y*x*bands`*. We also know about the extent and projection of the grid composing the NumPy array.

Assuming the data themselves already exist, we need to set up the geo-referencing::
	
	geoTransform = (tlx, wes, 0, tly, 0, nss)
	proj4 = '+proj=stere ...'

Now we use this information to create a GeoRaster representation of the data. In this case our data, stored in `data_array`, is single-band, so the NumPy array has dimensions *y*x*::

	im = georaster.SingleBandRaster.from_array(data_array, geoTransform, proj4)

By default the datatype is set to Float32, but you can use an alternative GDAL datatype by settin the `gdal_dtype` keyword.

NoData values are assumed to be `np.nan` in the input data array. If this is not the case, specify the value using the `nodata` keyword.

This also works for multi-band data - simply substitute `MultiBandRaster` in place of `SingleBandRaster` above.



Manipulate
~~~~~~~~~~

You can now use this geo-referenced GeoRaster object as you wish. You can discard it once you're finished with it, or you can...



Save to GeoTIFF
~~~~~~~~~~~~~~~

Simply call the save command as follows::

	im.save_geotiff('my_filename.tif')

N.b. you may need to specify an alternative dtype.