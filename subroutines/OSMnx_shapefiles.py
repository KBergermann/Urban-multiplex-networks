import numpy as np
import osmnx as ox
import sys


def OSMnx_shapefiles(searchString, saveString):

	#####################################################################################################
	#													#
	# DESCRIPTION:	uses 'searchString' to query the OpenStreetMap (OSM) API via the package osmnx	#
	#		[1,2] and saves the obtained (multi-) polygon as shape files in the directory	#
	#		'shapefiles' with file names 'saveString'.						#
	#													#
	# INPUT: 	searchString (required): exact expression passed to OSMnx's function			#
	#			'geocode_to_gdf', which queries the OSM API and generates the corresponding	#
	#			street network.								#
	#		saveString (required): specifies the sub-directory in 'shapefiles' in which the 	#
	#			shape files are saved.								#
	#													#
	# OUTPUT: 	None											#
	#													#
	# FILES												#
	# CREATED:	'shapefiles/%s' % saveString								#
	#													#
	# REFERENCES:	[1] G. Boeing, OSMnx: New methods for acquiring, constructing, analyzing, and	#
	#		visualizing complex street networks, Computers, Environment and Urban Systems,	#
	#		65 (2017), pp. 126-139, https://doi.org/10.1016/j.compenvurbsys.2017.05.004.		#
	#		[2] https://osmnx.readthedocs.io/en/stable/osmnx.html				#
	#													#
	# 2021, Kai Bergermann (kai.bergermann@math.tu-chemnitz.de)						#
	#													#
	#####################################################################################################
	
	ox.config(log_console=True)
	weight_by_length = False

	ox.__version__

	try:
		# single polygon
		gdf = ox.geocode_to_gdf(searchString)
		x, y = gdf.geometry.iloc[0].exterior.coords.xy

	except:
		# multipolygon
		gdf = ox.geocode_to_gdf(searchString)
		polyList = list(gdf.geometry.iloc[0])
		for geo in polyList:
			x, y = geo.exterior.coords.xy


	gdf.to_file('shapefiles/%s' % saveString)

