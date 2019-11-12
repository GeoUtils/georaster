"""
Test functions for georaster

Andrew Tedstone, November 2019
"""

import unittest

import os
import hashlib
import pathlib
import pyproj

import georaster

test_data_landsat = os.path.join(georaster.test_data_path, 'LE71400412000304SGS00_B4_crop.TIF')
test_data_landsat_sha256 = '271fa34e248f016f87109c8e81960caaa737558fbae110ec9e0d9e2d30d80c26'

class TestTestDataIntegrity(unittest.TestCase):

	def test_integrity(self):
		file_sha256 = hashlib.sha256(pathlib.Path(test_data_landsat).read_bytes()).hexdigest()
		assert file_sha256 == test_data_landsat_sha256


class TestAttributes(unittest.TestCase):

	im = georaster.SingleBandRaster(os.path.join(georaster.test_data_path, 'LE71400412000304SGS00_B4_crop.TIF'))

	def test_extent(self):
		left, right, bottom, top = self.im.extent
		self.assertAlmostEqual(left  , 478000.000, places=3)
		self.assertAlmostEqual(right , 502000.000, places=3)
		self.assertAlmostEqual(bottom, 3088490.000, places=3)
		self.assertAlmostEqual(top   , 3108140.000, places=3)



class TestCoordinateTransforms(unittest.TestCase):
	"""
	Decimal degrees values calculated from Corner Coordinates output by gdalinfo,
	converted from DMS to decimal using:
	https://www.rapidtables.com/convert/number/degrees-minutes-seconds-to-degrees.html

	"""

	im = georaster.SingleBandRaster(os.path.join(georaster.test_data_path, 'LE71400412000304SGS00_B4_crop.TIF'))


	def test_get_extent_latlon(self):
		left, right, bottom, top = self.im.get_extent_latlon()
		self.assertAlmostEqual(left  , 86.77604, places=3)
		self.assertAlmostEqual(right , 87.02036, places=3)
		self.assertAlmostEqual(bottom, 27.93200, places=3)
		self.assertAlmostEqual(top   , 28.09874, places=3)



	def test_get_extent_projected(self):

		# Use example of web mercator projection
		test_proj = pyproj.Proj('+init=epsg:3785')

		left, right, bottom, top = self.im.get_extent_projected(test_proj)
		self.assertAlmostEqual(left  , 9659864.900, places=2)
		self.assertAlmostEqual(right , 9687063.081, places=2)
		self.assertAlmostEqual(bottom, 3239029.409, places=2)
		self.assertAlmostEqual(top   , 3261427.912, places=2)


	def test_coord_to_px_latlon(self):
		return


	def test_coord_to_px_projected(self):
		return


	def test_coord_to_px_projected_cellCorner(self):
		return


	def test_coordinates(self):
		return



class TestValueRetrieval(unittest.TestCase):

	im = georaster.SingleBandRaster(os.path.join(georaster.test_data_path, 'LE71400412000304SGS00_B4_crop.TIF'))

	def test_value_at_coords_latlon(self):
		# 1. Lat/Lon, cell center coordinate.
		val = self.im.value_at_coords(86.90271, 28.00108, latlon=True)
		assert val == 67



	def test_value_at_coords_projected(self):
		# 2. Projected, cell center coordinate.
		val = self.im.value_at_coords(490434.605, 3097325.642)
		assert val == 67




class TestLoading(unittest.TestCase):

	def test_extent(self):
		return


	def test_fromarray(self):
		return

	def test_read_single_band_subset(self):
		return


if __name__ == '__main__':
	unittest.main()