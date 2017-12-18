from distutils.core import setup
setup(
  name = 'georaster',
  packages = ['georaster'], # this must be the same as the name above
  version = '1.25',
  description = 'easy processing and analysis of geographic and projected rasters in Python',
  author = 'Andrew Tedstone, Amaury Dehecq',
  author_email = 'a.j.tedstone@bristol.ac.uk',
  url = 'https://github.com/GeoUtils/georaster', # use the URL to the github repo
  download_url = 'https://github.com/GeoUtils/georaster/archive/1.25.tar.gz',
  keywords = ['rasters', 'remote sensing', 'GIS', 'GDAL', 'projections', 'geoTIFF'], # arbitrary keywords
  classifiers = [],
  license = 'GNU GPL v.3'
)