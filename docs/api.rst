.. _api-reference

API Reference
=============

Full information about GeoRaster's functionality is provided on this page.

.. py:module:: georaster

SingleBandRaster
~~~~~~~~~~~~~~~~

.. autoclass:: SingleBandRaster
	:members: from_array, get_extent_latlon, get_extent_projected, coord_to_px, value_at_coords, coordinates, reproject, interp, save_geotiff, cartopy_proj, intersection



MultiBandRaster
~~~~~~~~~~~~~~~

.. autoclass:: MultiBandRaster
	:members: gdal_band, from_array, get_extent_latlon, get_extent_projected, coord_to_px, value_at_coords, coordinates, reproject, interp, save_geotiff, cartopy_proj, intersection



Advanced Functionality
~~~~~~~~~~~~~~~~~~~~~~

There are several additional functions available in both SingleBandRaster and MultiBandRaster that are not listed in the main reference above.

.. warning:: You should not use these functions without first fully understanding their implications.

.. autoclass:: __Raster
	:members: read_single_band, read_single_band_subset