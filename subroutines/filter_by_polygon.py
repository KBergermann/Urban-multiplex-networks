import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.path as mpltPath


def filter_by_polygon(cityString, gtfsDataPath='gtfsdata'):

	#####################################################################################################
	#													#
	# DESCRIPTION:	uses the OSMnx shape file from the directory 'shapefiles' corresponding to the 	#
	#		specified city ('cityString') to filter for stops (from the GTFS data set from the	#
	#		directory 'gtfsDataPath'), which are located within the polygon representing the 	#
	#		city's administrative boundaries of the cities proper (excluding suburban areas).	#
	#													#
	# INPUT: 	cityString (required): string containing the city name for which the stops will be	#
	#			filtered. A shape file with the name 'cityString' must be available in the	#
	#			directory 'shapefiles'. This method is called in the functions 		#
	#			'build_supra_adjacency_matrix' and 'create_aggregated_bearing_plot' when 	#
	#			filtering by polygon is enabled. 						#
	#		gtfsDataPath (optional, default='gtfsdata'): specifies the path in which the	GTFS	#
	#			data is located.								#
	#													#
	# OUTPUT: 	stops[inside] (pandas.DataFrame): DataFrame containing the stops from the specified	#
	#			GTFS stop-file filtered by the specified polygon.				#
	#													#
	# FILES												#
	# CREATED:	None											#
	#													#
	# 2021, Kai Bergermann (kai.bergermann@math.tu-chemnitz.de)						#
	#													#
	#####################################################################################################


	# read stop list
	stops = pd.read_csv('%s/stops.txt' % gtfsDataPath, low_memory=False)
	stopCoords = np.asarray(stops[['stop_lon', 'stop_lat']])

	# read shapefiles
	gdf = gpd.read_file("shapefiles/%s/%s.shp" % (cityString, cityString))

	# check for multipolygon
	if str(type(gdf.geometry.iloc[0])).find('MultiPolygon') == -1:
		# single polygon case
		# build numpy point array from polygon
		x, y = gdf.geometry.iloc[0].exterior.coords.xy
		poly = np.vstack([x, y]).T

		# check if points are within polygon
		path = mpltPath.Path(poly)
		inside = path.contains_points(stopCoords)

	else:
		# multipolygon case
		# initialize inside as all 'False' array
		inside = np.array(np.zeros(len(stops)), dtype=bool)

		# build list of polygons
		polyList = list(gdf.geometry.iloc[0])
		for geo in polyList:
			x, y = geo.exterior.coords.xy
			poly = np.vstack([x, y]).T

			# check if points are within one of the polygons
			path = mpltPath.Path(poly)
			# update any index inside the current polygon to 'True'
			inside = np.logical_or(inside, path.contains_points(stopCoords))

	return stops[inside]

