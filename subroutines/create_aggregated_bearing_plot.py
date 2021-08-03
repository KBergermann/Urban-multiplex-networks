import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import time
from .filter_by_polygon import *


def create_aggregated_bearing_plot(cityString, gtfsDataPath='gtfsdata', polygon=True, latMin=0, latMax=0, lonMin=0, lonMax=0, latMin2=0, latMax2=0, lonMin2=0, lonMax2=0, oneTripPerRoute=False):

	#####################################################################################################
	#													#
	# DESCRIPTION:	uses the GTFS data found in the directory 'gtfsDataPath' to create public		#
	#		transport network orientation plots. The methodology is described in [1, Sec. 3].	#
	#													#
	# INPUT: 	cityString (required): string containing the city name for which the orientation	#
	#			plot will be generated. If a polygon is used for stop filtering, a shape	#
	#			file with the name 'cityString' must be available in the directory		#
	#			'shapefiles'. Otherwise, the union of up to two coordinate bounding boxes	#
	#			can be used for stop filtering (see the description of the INPUT argument	#
	#			'polygon').									#
	#		gtfsDataPath (optional, default='gtfsdata'): specifies the path in which the	GTFS	#
	#			data is located.								#
	#		polygon (optional, default=True): if True, uses the shape file with the name		#
	#			'cityString' from the directory 'shapefiles' to filter for stops within	#
	#			the city limits. If False, the following arguments can be used to specify	#
	#			bounding boxes, which are then used for stop filtering.			#
	#		latMin, latMax, lonMin, lonMax, latMin2, latMax2, lonMin2, lonMax2 (optional,	#
	#		default=0): if polygon==False, the union of the two bounding boxes corresponding	#
	#			to the specified latitudinal and longitudinal coordinates are used for stop	#
	#			filtering									#
	#		oneTripPerRoute (optional, default=False): if True, considers only unique trips	#
	#			of the routes. If False, loops over all trips of all routes.			#
	#													#
	# OUTPUT: 	None											#
	#													#
	# FILES												#
	# CREATED:	'orientationPlots/%s.pdf' % cityString, or						#
	#		'orientationPlots/%s_OneTripPerRoute.pdf' % cityString				#
	#													#
	# REFERENCE:	[1] K. Bergermann and M. Stoll, Orientations and matrix function-based		#
	#		centralities in multiplex network analysis of urban public transport, arXiv		#
	#		preprint, arXiv:2107.12695, (2021).							#
	#													#
	# 2021, Peter Bernd Oehme, Martin Stoll, Kai Bergermann (kai.bergermann@math.tu-chemnitz.de)	#
	#													#
	#####################################################################################################


	print('Starting to compute public transport orientations of the city: %s' % cityString)

	### filter stops by specified coordinates ###
	stops = pd.read_csv('%s/stops.txt' % gtfsDataPath, low_memory=False)
	if not polygon:
		# filter by bounding box
		filteredStops = stops[(stops['stop_lat'] >= latMin) & (stops['stop_lat'] <= latMax) & (stops['stop_lon'] >= lonMin) & (stops['stop_lon'] <= lonMax) | (stops['stop_lat'] >= latMin2) & (stops['stop_lat'] <= latMax2) & (stops['stop_lon'] >= lonMin2) & (stops['stop_lon'] <= lonMax2)] # no real danger from (0,0) here, which lies somewhere in the gulf of Guinea
	else:
		# filter by polygon
		filteredStops = filter_by_polygon(cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), gtfsDataPath)

	print ('Number of filtered stops by coordinates:', len(filteredStops))
	print('Starting with some data preprocessing...')

	### Get tripIDs over stopIDs ###
	stopTimes = pd.read_csv('%s/stop_times.txt' % gtfsDataPath, low_memory=False)
	filteredStopTimes = stopTimes[stopTimes['stop_id'].isin(filteredStops['stop_id'])] # drop all trips outside of the city
	tripIDs = np.unique(filteredStopTimes['trip_id']).tolist()

	filTimes = filteredStopTimes

	### Get routeIDs over tripIDs ###
	trips = pd.read_csv('%s/trips.txt' % gtfsDataPath, low_memory=False)
	filteredTrips = trips[trips['trip_id'].isin(tripIDs)]
	routeIDs = np.unique(filteredTrips['route_id'])

	### detect and drop routes containing only trips without edges (only one stop inside the city boundaries) ###
	tripsWithoutEdges = []
	nonTrivialRouteIDs = []
	for routeID in routeIDs:
		# get all trip_ids
		tripListOfRoute = filteredTrips[(filteredTrips['route_id'] == routeID)]['trip_id'].tolist()
		stopIDList = []
		for tripID in tripListOfRoute:
			# create list of lists of all trips of the given route and filter for unique trips
			stopIDList.append(filTimes[filTimes['trip_id'] == tripID]['stop_id'].tolist())
		uniquestopIDList = [list(x) for x in set(tuple(x) for x in stopIDList)]

		routeHasNonTrivialTrip = 0
		for uniqueStop in uniquestopIDList:
			if len(uniqueStop)>=2:
				routeHasNonTrivialTrip = 1

		if routeHasNonTrivialTrip == 1:
			nonTrivialRouteIDs.append(routeID)

	routeIDs = np.array(nonTrivialRouteIDs)

	print('Number of found routes:', routeIDs.size)

	routeNames = pd.read_csv('%s/routes.txt' % gtfsDataPath, low_memory=False)
	routeNameTable=routeNames[routeNames['route_id'].isin(routeIDs)]
	longRouteNameTable=routeNameTable['route_long_name'].tolist()
	print('Long names of the routes:', longRouteNameTable)

	### Compute bearings ###

	# generate a list of coordinate differences between two consecutive stations
	L = routeIDs.size
	coordData = [[],[]]
	indCoordData = [[[],[]] for x in range(L)]

	coordData0=[]
	coordData1=[]

	# stopping the execution time
	start_time = time.time()

	if not oneTripPerRoute:

		print('Starting to compute the orientations. This may take a while. Out of %d routes, Im currently computing route' % len(routeIDs))

		# loop over all routes
		j = 0
		for routeID in routeIDs:
			sys.stdout.write("\r{}".format(j+1))
			sys.stdout.flush()
			tripListOfRoute = filteredTrips[(filteredTrips['route_id'] == routeID)]['trip_id'].tolist()
			#loop over all trips
			for tripID in tripListOfRoute:
				# loop over the stops
				stopList = filTimes[filTimes['trip_id'] == tripID]

				lats = filteredStops[filteredStops['stop_id'].isin(stopList['stop_id'])][['stop_id', 'stop_lat']]
				list_lat = pd.merge(pd.DataFrame(stopList['stop_id']), lats, on='stop_id')['stop_lat']

				lons = filteredStops[filteredStops['stop_id'].isin(stopList['stop_id'])][['stop_id', 'stop_lon']]
				list_lon = pd.merge(pd.DataFrame(stopList['stop_id']), lons, on='stop_id')['stop_lon']

				coordData0.append((list_lat.diff(periods=-1)).dropna())
				coordData1.append((list_lon.diff(periods=-1)).dropna())

			j += 1

	else:

		print('Computing orientations with only the unique trips of the routes...')

		# loop over all routes
		j = 0
		for routeID in routeIDs:
			tripListOfRoute = filteredTrips[(filteredTrips['route_id'] == routeID)]['trip_id'].tolist()

			stopIDList = []
			tripIDList = []
			for tripID in tripListOfRoute:
				stopIDList.append(filTimes[filTimes['trip_id'] == tripID]['stop_id'].tolist())
				tripIDList.append(filTimes[filTimes['trip_id'] == tripID]['trip_id'].tolist())
			
			uniquestopIDList = [list(x) for x in set(tuple(x) for x in stopIDList)]
			
			relevantTripIDs = []
			for uniqueTrip in uniquestopIDList:
				relevantTripIDs.append(tripIDList[stopIDList.index(uniqueTrip)][0])
						
			# loop over unique trips
			for tripID in relevantTripIDs:
				# loop over the stops
				stopList = filTimes[filTimes['trip_id'] == tripID]

				lats = filteredStops[filteredStops['stop_id'].isin(stopList['stop_id'])][['stop_id', 'stop_lat']]
				list_lat = pd.merge(pd.DataFrame(stopList['stop_id']), lats, on='stop_id')['stop_lat']

				lons = filteredStops[filteredStops['stop_id'].isin(stopList['stop_id'])][['stop_id', 'stop_lon']]
				list_lon = pd.merge(pd.DataFrame(stopList['stop_id']), lons, on='stop_id')['stop_lon']

				coordData0.append((list_lat.diff(periods=-1)).dropna())
				coordData1.append((list_lon.diff(periods=-1)).dropna())

			j += 1


	coordData0_flat = [item for sublist in coordData0 for item in sublist]
	coordData1_flat = [item for sublist in coordData1 for item in sublist]

	# calculate bearing and transform into df
	bearingData = pd.DataFrame((180.0 + np.angle(np.array(coordData1_flat + np.multiply(1j, coordData0_flat)), deg=True)).transpose(), columns=['angle'])

	lenBearingData = len(bearingData)

	### single polar plot ###

	print('\nDone computing orientations! That took %s seconds for a total of %s stops in %s. Plotting now.' % ((time.time() - start_time), lenBearingData, cityString))

	nBins = 36
	offset = (360/(2*nBins))
	rangeBins = (360/nBins)*np.arange(nBins+1) + offset
	adaptedBearingData = np.vstack([bearingData[bearingData['angle']<offset]+360, bearingData[bearingData['angle']>=offset]])
	fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6,6))
	width =  2 * np.pi / nBins
	count, division = np.histogram(adaptedBearingData, bins=rangeBins)
	division = division[0:-1]
	ax.bar(division * np.pi / 180 + width * 0.5, count / lenBearingData, width=width, alpha=0.75, color='mediumblue', edgecolor='k')
	ax.set_yticklabels('')
	ax.set_xticks([0,np.pi/4,np.pi/2,(3*np.pi)/4,np.pi,(5*np.pi)/4,(3*np.pi)/2,(7*np.pi)/4])
	ax.set_rticks((max(count)/(4*lenBearingData))*range(1,5))
	ax.set_rlim([0, max(count)/lenBearingData])
	ax.set_xticklabels(['E','','N','','W','','S',''], fontdict={'fontsize': 18})
	ax.set_title("%s" % cityString, va='top', fontdict={'fontsize': 34})
	ax.set_axisbelow(True)
	ttl = ax.title
	ttl.set_position([.5, 1.15])


	# saving the figure
	directory = 'orientationPlots'

	if not os.path.exists(directory):
		os.makedirs(directory)
		
	if not oneTripPerRoute:
		fig.savefig("%s/%s.pdf" % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_")), bbox_inches='tight')
	else:
		fig.savefig("%s/%s_OneTripPerRoute.pdf" % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_")), bbox_inches='tight')
		
	plt.close()

	print('--------------------------------------------------')


