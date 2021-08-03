import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import scipy.sparse as spsp
import scipy.special as spspec
import datetime
from .filter_by_polygon import *


def build_supra_adjacency_matrix(cityString, gtfsDataPath='gtfsdata', polygon=True, latMin=0, latMax=0, lonMin=0, lonMax=0, latMin2=0, latMax2=0, lonMin2=0, lonMax2=0, weighted=False, compute_inter=True, compute_frequencies=False):

	#####################################################################################################
	#													#
	# DESCRIPTION:	uses the GTFS data found in the directory 'gtfsDataPath' to construct the		#
	#		following adjacency matrices of a multiplex network in which nodes correspond	#
	#		to public transport stops and layers correspond to lines: unweighted multilayer	#
	#		intra-layer adjacency matrix (if weighted==False), weighted multilayer		#
	#		intra-layer adjacency matrix with travel times as weights (if weighted==True),	#
	#		weighted multilayer intra-layer adjacency matrix with frequencies as weights (if	#
	#		compute_frequencies==True), unweighted inter-layer adjacency matrix (if		#
	#		compute_inter==True).									#
	#													#
	# INPUT: 	cityString (required): string containing the city name for which the adjacency	#
	#			matrices will be built. If a polygon is used for stop filtering, a shape	#
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
	#			filtering.									#
	#		weighted (optional, default=False): if True, computes travel times and uses it as	#
	#			weight in the multilayer intra-layer adjacency matrix. If False,		#
	#			constructs an unweighted multilayer intra-layer adjacency matrix.		#
	#		compute_inter (optional, default=True): if True, constructs an unweighted inter-	#
	#			layer adjacency matrix in which node-layer pairs are only allowed to be	#
	#			connected to instances of the same physical node and the presence of		#
	#			edges is determined by whether or not two lines (layer-layer pair) serve	#
	#			the same stop (physical node).						#
	#		compute_frequencies (optional, default=False): if True, constructs a multilayer	#
	#			intra-layer adjacency matrix containing line frequencies (i.e., the		#
	#			number of connections between nodes that the line offers per day) as		#
	#			entries.									#
	#													#
	# OUTPUT: 	None											#
	#													#
	# FILES												#
	# CREATED:	'adjacency_matrices/%s_stop_IDs.csv' % cityString					#
	#		'adjacency_matrices/%s_layer_IDs.csv' % cityString					#
	#		'adjacency_matrices/%s_Aintra_unweighted.npz' % cityString				#
	#		'adjacency_matrices/%s_Aintra_weighted.npz' % cityString				#
	#		'adjacency_matrices/%s_Aintra_weighted_frequencies.npz' % cityString			#
	#		'adjacency_matrices/%s_Ainter_unweighted.npz' % cityString				#
	#													#
	# 2021, Peter Bernd Oehme, Kai Bergermann (kai.bergermann@math.tu-chemnitz.de)			#
	#													#
	#####################################################################################################


	if not polygon and (latMin==0 and latMax==0 and lonMin==0 and lonMax==0 and latMin2==0 and latMax2==0 and lonMin2==0 and lonMax2==0):
		print('Error. Either set polygon option to True or specify non-zero coordinates for a valid bounding box!')
		sys.exit(0)


	print('Starting to compute the supra-adjacency matrix of the city: %s' % cityString)

	### filter stops by specified coordinates ###
	stops = pd.read_csv('%s/stops.txt' % gtfsDataPath, low_memory=False)
	if polygon:
		# filter by polygon
		filteredStops = filter_by_polygon(cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"))
	else:
		# filter by bounding box
		filteredStops = stops[(stops['stop_lat'] >= latMin) & (stops['stop_lat'] <= latMax) & (stops['stop_lon'] >= lonMin) & (stops['stop_lon'] <= lonMax) | (stops['stop_lat'] >= latMin2) & (stops['stop_lat'] <= latMax2) & (stops['stop_lon'] >= lonMin2) & (stops['stop_lon'] <= lonMax2)] # no real danger from (0,0) here, which lies somewhere in the gulf of Guinea


	print('Number of filtered stops by coordinates:', len(filteredStops))
	print('Starting with some data preprocessing...')


	### Get tripIDs over stopIDs ###
	stopTimes = pd.read_csv('%s/stop_times.txt' % gtfsDataPath, low_memory=False)
	filteredStopTimes = stopTimes[stopTimes['stop_id'].isin(filteredStops['stop_id'])] # drop all trips outside of the city
	tripIDs = np.unique(filteredStopTimes['trip_id']).tolist()

	filTimes = filteredStopTimes


	df = {'node_id': range(len(filteredStops)), 'stop_id': filteredStops['stop_id'], 'stop_name': filteredStops['stop_name'], 'stop_lat': filteredStops['stop_lat'], 'stop_lon': filteredStops['stop_lon']}
	stopIDTableDF = pd.DataFrame(data=df)


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


	### Print stop and layer lists to csv files ###
	directory = 'adjacency_matrices'
	if not os.path.exists(directory):
		os.makedirs(directory)

	df = {'layer_id': range(len(longRouteNameTable)), 'layer_name': longRouteNameTable}
	layerIDTableDF = pd.DataFrame(data=df)

	stopIDTableDF[['node_id', 'stop_id', 'stop_name', 'stop_lat', 'stop_lon']].to_csv('%s/%s_stop_IDs.csv' % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_")), index=False)
	layerIDTableDF.to_csv('%s/%s_layer_IDs.csv' % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_")), index=False)


	if not weighted:

		### construction of the supra-adjacency matrix ###

		### intra-layer adjacencies ###
		n = len(filteredStops)
		L = len(routeIDs)
		print('n=%d, L=%d' % (n, L))
		l=0
		print('Intra-layer adjacencies\nOf %d total layers, Im currently computing layer' % L)
		for routeID in routeIDs:
			sys.stdout.write("\r{}".format(l+1))
			sys.stdout.flush()
			# get all trip_ids
			tripListOfRoute = filteredTrips[(filteredTrips['route_id'] == routeID)]['trip_id'].tolist()
			singleLayerAdjacencyMatrix = spsp.lil_matrix((n, n), dtype=np.double)
			stopIDList = []
			for tripID in tripListOfRoute:
				# create list of lists of all trips of the given route and filter for unique trips
				stopIDList.append(filTimes[filTimes['trip_id'] == tripID]['stop_id'].tolist())
			uniquestopIDList = [list(x) for x in set(tuple(x) for x in stopIDList)]

			# only loop over the unique set of trips belonging to the respective route
			for stopIDs in uniquestopIDList:
				for i in range(len(stopIDs) - 1):
					# add edge between current item and last
					singleLayerAdjacencyMatrix[stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['node_id'], stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['node_id']] = 1
					singleLayerAdjacencyMatrix[stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['node_id'], stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['node_id']] = 1
			
			# build block diagonal matrix from single-layer adjacencies
			if l==0:
				intraLayerAdjacencyMatrix = singleLayerAdjacencyMatrix
			else:
				intraLayerAdjacencyMatrix = spsp.bmat([[intraLayerAdjacencyMatrix, None], [None, singleLayerAdjacencyMatrix]])
			l+=1

		# convert to csr sparse format, which is efficient for arithmetic operations and MV prods
		Aintra = intraLayerAdjacencyMatrix.tocsr()

		# save sparse matrix in file
		spsp.save_npz('%s/%s_Aintra_unweighted.npz' % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_")), Aintra)

	else: # weighted case
		if compute_frequencies:

			### construction of the supra-adjacency matrix ###

			### intra-layer adjacencies ###
			n = len(filteredStops)
			L = len(routeIDs)
			print('n=%d, L=%d' % (n, L))
			l=0
			print('Intra-layer adjacencies\nOf %d total layers, Im currently computing layer' % L)
			for routeID in routeIDs:
				sys.stdout.write("\r{}".format(l+1))
				sys.stdout.flush()
				# get all trip_ids
				tripListOfRoute = filteredTrips[(filteredTrips['route_id'] == routeID)]['trip_id'].tolist()
				singleLayerAdjacencyMatrix = spsp.lil_matrix((n, n), dtype=np.double)
				singleLayerFrequencyMatrix = spsp.csr_matrix((n, n), dtype=np.double)

				for tripID in tripListOfRoute:
					stopTimesOfTrip = filteredStopTimes[filteredStopTimes['trip_id']==tripID]
					stopIDs = stopTimesOfTrip['stop_id'].tolist()

					# only loop over the unique set of trips belonging to the respective route

					for i in range(len(stopIDs) - 1):
						# add edge between current item and last
						h1, m1, s1 = stopTimesOfTrip.iloc[i,:]['departure_time'].split(':')
						h2, m2, s2 = stopTimesOfTrip.iloc[i+1,:]['arrival_time'].split(':')
						timeDiff = abs(int(datetime.timedelta(hours=int(h2) - int(h1),minutes=int(m2) - int(m1),seconds=0).total_seconds() / 60))

						singleLayerAdjacencyMatrix[stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['node_id'], stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['node_id']] = timeDiff
						singleLayerAdjacencyMatrix[stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['node_id'], stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['node_id']] = timeDiff

						# lil_matrix doesn't support adding non-zero scalars. Using csr format instead suppressing efficiency warnings
						import warnings
						with warnings.catch_warnings():
							warnings.simplefilter("ignore")
							singleLayerFrequencyMatrix[stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['node_id'], stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['node_id']] += 1
							singleLayerFrequencyMatrix[stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['node_id'], stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['node_id']] += 1

				
				# build block diagonal matrix from single-layer adjacencies
				if l==0:
					intraLayerAdjacencyMatrix = singleLayerAdjacencyMatrix
					intraLayerFrequencyMatrix = singleLayerFrequencyMatrix
				else:
					intraLayerAdjacencyMatrix = spsp.bmat([[intraLayerAdjacencyMatrix, None], [None, singleLayerAdjacencyMatrix]])
					intraLayerFrequencyMatrix = spsp.bmat([[intraLayerFrequencyMatrix, None], [None, singleLayerFrequencyMatrix]])
				l+=1

			# convert to csr sparse format, which is efficient for arithmetic operations and MV prods
			Aintra = intraLayerAdjacencyMatrix.tocsr()
			Aintra_frequencies = intraLayerFrequencyMatrix.tocsr()

			# save sparse matrix in file
			spsp.save_npz('%s/%s_Aintra_weighted.npz' % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_")), Aintra)
			spsp.save_npz('%s/%s_Aintra_weighted_frequencies.npz' % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_")), Aintra_frequencies)

		else: # weighted case without frequencies

			### construction of the supra-adjacency matrix ###

			### intra-layer adjacencies ###
			n = len(filteredStops)
			L = len(routeIDs)
			print('n=%d, L=%d' % (n, L))
			l=0
			print('Intra-layer adjacencies\nOf %d total layers, Im currently computing layer' % L)
			for routeID in routeIDs:
				sys.stdout.write("\r{}".format(l+1))
				sys.stdout.flush()
				# get all trip_ids
				tripListOfRoute = filteredTrips[(filteredTrips['route_id'] == routeID)]['trip_id'].tolist()
				singleLayerAdjacencyMatrix = spsp.lil_matrix((n, n), dtype=np.double)
				stopIDList = []
				tripIDList = []
				for tripID in tripListOfRoute:
					# create list of lists of all trips of the given route and filter for unique trips
					stopIDList.append(filTimes[filTimes['trip_id'] == tripID]['stop_id'].tolist())
					tripIDList.append(filTimes[filTimes['trip_id'] == tripID]['trip_id'].tolist())

				uniquestopIDList = [list(x) for x in set(tuple(x) for x in stopIDList)]

				relevantTripIDs = []
				for uniqueTrip in uniquestopIDList:
					relevantTripIDs.append(tripIDList[stopIDList.index(uniqueTrip)][0])

				# only loop over the unique set of trips belonging to the respective route
				j=0
				for stopIDs in uniquestopIDList:
					stopTimesOfTrip = filteredStopTimes[filteredStopTimes['trip_id']==relevantTripIDs[j]]

					for i in range(len(stopIDs) - 1):
						# add edge between current item and last
						h1, m1, s1 = stopTimesOfTrip.iloc[i,:]['departure_time'].split(':')
						h2, m2, s2 = stopTimesOfTrip.iloc[i+1,:]['arrival_time'].split(':')
						timeDiff = abs(int(datetime.timedelta(hours=int(h2) - int(h1),minutes=int(m2) - int(m1),seconds=0).total_seconds() / 60))

						singleLayerAdjacencyMatrix[stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['node_id'], stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['node_id']] = timeDiff
						singleLayerAdjacencyMatrix[stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['node_id'], stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['node_id']] = timeDiff

					j+=1
				
				# build block diagonal matrix from single-layer adjacencies
				if l==0:
					intraLayerAdjacencyMatrix = singleLayerAdjacencyMatrix
				else:
					intraLayerAdjacencyMatrix = spsp.bmat([[intraLayerAdjacencyMatrix, None], [None, singleLayerAdjacencyMatrix]])
				l+=1

			# convert to csr sparse format, which is efficient for arithmetic operations and MV prods
			Aintra = intraLayerAdjacencyMatrix.tocsr()

			# save sparse matrix in file
			spsp.save_npz('%s/%s_Aintra_weighted.npz' % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_")), Aintra)

	if compute_inter:
		### inter-layer adjacencies ###
		Ainter = spsp.lil_matrix((n*L, n*L), dtype=np.double)

		print('\nInter-layer adjacency\nOf',L,'total layers, Im currently computing layer')

		# add interlayer edges
		for i in range(len(routeIDs)):
			# pairwise comparison of layer i with remaining layers (symmetric, so only compare with layer with higher ID)
			sys.stdout.write("\r{}".format(i+1))
			sys.stdout.flush()
			curRouteID = routeIDs[i]
			indicesRemainingLayers = list(range(i+1, len(routeIDs)))
			for j in indicesRemainingLayers:
				otherRouteID = routeIDs[j]

				# filter both stopIDs
				curTripIDs = filteredTrips[(filteredTrips['route_id'] == curRouteID)]['trip_id'].tolist()
				prevTripIDs = filteredTrips[(filteredTrips['route_id'] == otherRouteID)]['trip_id'].tolist()

				# union the stopIDs into the sets
				curStopIDs = set()
				prevStopIDs = set()
				for tripID in curTripIDs:
					curStopIDs = curStopIDs.union(set(filTimes[filTimes['trip_id'] == tripID]['stop_id'].tolist()))
				for tripID in prevTripIDs:
					prevStopIDs = prevStopIDs.union(set(filTimes[filTimes['trip_id'] == tripID]['stop_id'].tolist()))

				# intersect cur and prev stopIDs and edges for these nodes
				overlap = curStopIDs.intersection(prevStopIDs)
				for node in overlap:
					Ainter[stopIDTableDF[stopIDTableDF['stop_id']==node]['node_id']+j*n, stopIDTableDF[stopIDTableDF['stop_id']==node]['node_id']+i*n] = 1
					Ainter[stopIDTableDF[stopIDTableDF['stop_id']==node]['node_id']+i*n, stopIDTableDF[stopIDTableDF['stop_id']==node]['node_id']+j*n] = 1


		# save sparse matrix in file
		Ainter = Ainter.tocsr()
		spsp.save_npz('%s/%s_Ainter_unweighted.npz' % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_")), Ainter)


	print('\n--------------------------------------------------')

