.. GeoRaster documentation master file, created by
   sphinx-quickstart on Mon Dec 19 12:07:58 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GeoRaster - easy use of geographic and projected rasters in Python
==================================================================

This package makes it easy to load, query and save geographic raster datasets in the Python programming language. The package uses Geospatial Data Abstraction Library (GDAL) bindings, and so in a single command can import any geo-referenced dataset that is understood by GDAL, complete with all geo-referencing information and various helper functions.

GeoRaster is compatible with Python 2.4-3.x.

There are two basic types of raster: either a single-band dataset, which you load into a `SingleBandRaster` object, or a dataset containing multiple bands to be loaded, which you load into a `MultiBandRaster` object.

.. _GDAL:  http://www.gdal.org/
.. _GDAL formats: http://www.gdal.org/formats_list.html


Contents:

.. toctree::
   :maxdepth: 1

   get-started
   load-existing-data
   create-georeferenced-data
   access-georeferencing
   access-data
   reprojecting
   plotting-options
   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

