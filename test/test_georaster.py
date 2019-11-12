"""
Test functions for georaster

Andrew Tedstone, November 2019
"""

class TestCoordinateTransforms(object):

	im = georaster.SingleBandRaster(georaster.test_landsat_data)

	def test_get_extent_latlon(self):
		left, right, bottom, top = im.get_extent_latlon()
		assert left ==
		assert right ==
		assert bottom ==
		assert top ==


	def test_get_extent_projected(self):
		left, right, bottom, top = im.get_extent_latlon()
		assert left ==
		assert right ==
		assert bottom ==
		assert top ==



	def test_coord_to_px(self):
		# 1. Lat/Lon, cell center coordinate.
		val = im.coord_to_px(lon, lat, latlon=True)
		assert val == 

		# 2. Projected, cell center coordinate.
		val = im.coord_to_px(x, y)
		assert val ==

		# 3. Projected, cell corner coordinate.
		val = im.coord_to_px(x, y, cell_type='corner')
		assert val ==



	def test_coordinates(self)
		return




class TestLoading(object):

	def test_extent(self):


	def test_fromarray(self):


	def test_read_single_band_subset(self):


class Test