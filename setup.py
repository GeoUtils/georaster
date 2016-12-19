from distutils.core import setup
setup(
  name = 'georaster',
  packages = ['georaster'], # this must be the same as the name above
  version = '1.1',
  description = 'easy use of geographic and projected rasters in Python',
  author = 'Andrew Tedstone',
  author_email = 'a.j.tedstone@bristol.ac.uk',
  url = 'https://github.com/ajtedstone/georaster', # use the URL to the github repo
  download_url = 'https://github.com/ajtedstone/georaster/tarball/1.1', # I'll explain this in a second
  keywords = ['rasters', 'remote sensing', 'GIS', 'GDAL', 'projections', 'geotiff'], # arbitrary keywords
  classifiers = [],
)