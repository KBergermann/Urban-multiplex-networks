import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np

from ..filter_by_polygon import *
from .compute_centralities import *


def compute_MLC_plots(cityString, compute_quadrature_quantities=True, weighted_with_travel_times=True, weighted_with_frequencies=True, omega=1, alpha_const=.5, beta_const=.5, maxit_quadrature=5, maxit_fAb=10, top_k_centralities=10, sigma=1, n_color_compartments=6, markersize=2):

	#####################################################################################################
	#													#
	# DESCRIPTION:	computes marginal layer centralities for the city specified in 'cityString' and	#
	#		visualizes them as line plots on top of the city's street network created by OSMnx	#
	#		[1,2]. Several parameters for the measures and the numerical schemes	as well as	#
	#		options for plotting are specified in the INPUT section.				#
	#													#
	# INPUT: 	cityString (required): string containing the city name for which the centralities	#
	#			are to be computed. The required adjacency matrices must be available in	#
	#			the directory 'adjacency_matrices' under the same name.			#
	#		compute_quadrature_quantities (optional, default=True): option passed to the 	#
	#			function 'compute_centralities'. If True, the measures SC, SCres, TC, and KC	#
	#			are computed. Otherwise, only TC and KC are computed.			#
	#		weighted_with_travel_times (optional, default=True): option passed to the function	#
	#			'compute_centralities'. If True, a Gaussian kernel applied to the travel	#
	#			times is used to weight intra-layer edges, cf. [3, Eq. (4.4)].		#
	#		weighted_with_frequencies (optional, default=True): option passed to the function	#
	#			'compute_centralities'. If True, line frequencies are used to weight intra-	#
	#			layer edges, cf. [3, Eq. (4.4)].						#
	#		omega (optional, default=1): option passed to the function 'compute_centralities'.	#
	#			Specifies the value of the inter-layer coupling parameter,			#
	#			cf. [3, Eq.(4.3)], which models a constant transfer time across the network.	#
	#		alpha_const (optional, default=0.5): option passed to the function 			#
	#			'compute_centralities'. Specifies the value of alpha in [3, Eq. (5.2)&(5.3)]	#
	#			via alpha=alpha_const/lambda_max.						#
	#		beta_const (optional, default=0.5): option passed to the function 			#
	#			'compute_centralities'. Specifies the value of beta in [3, Eq. (5.2)&(5.3)]	#
	#			via beta=beta_const/lambda_max.						#
	#		maxit_quadrature (optional, default=5): option passed to the function 		#
	#			'compute_centralities'. Specifies the number of Lanczos iterations used to	#
	#			compute SC and SC_res.								#
	#		maxit_fAb (optional, default=10): option passed to the function 			#
	#			'compute_centralities'. Specifies the number of Lanczos iterations		#
	#			used to compute TC and KC.							#
	#		top_k_centralities (optional, default=10): option passed to the function 		#
	#			'compute_centralities'. Specifies the number of leading nodes, layers, and	#
	#			node-layer pairs to be displayed in the rankings.				#
	#		sigma (optional, default=1): option passed to the function 'compute_centralities'.	#
	#			If weighted_with_travel_times==True, specifies the scaling parameter in the	#
	#			Gaussian kernel, which is applied to travel times				#
	#			to obtain intra-layer weights.						#
	#		n_color_compartments (optional, default=6): specifies the number of equispaced	#
	#			subintervals into which the computed centrality values are partitioned. A	#
	#			heatmap color scheme (dark blue for the least central lines, dark red for	#
	#			the most central lines) is used for the visualization of the marginal layer	#
	#			centralities.									#
	#		markersize (optional, default=2): specifies the linewidth in the produced plots.	#
	#													#
	# OUTPUT: 	None											#
	#													#
	# FILES												#
	# CREATED:	'centralities/plots/%s_TC_layer_centralities_weighted_times_%s_freq_%s_beta_const_%s_sigma_%s_omega_%s.png' % (cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, beta_const, sigma, omega)											#
	#		'centralities/plots/%s_KC_layer_centralities_weighted_times_%s_freq_%s_alpha_const_%s_sigma_%s_omega_%s.png' % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, alpha_const, sigma, omega)							#
	#		'centralities/plots/%s_SC_layer_centralities_weighted_times_%s_freq_%s_beta_const_%s_sigma_%s_omega_%s.png' % (cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, beta_const, sigma, omega), if compute_quadrature_quantities==True						#
	#		'centralities/plots/%s_SCres_layer_centralities_weighted_times_%s_freq_%s_alpha_const_%s_sigma_%s_omega_%s.png' % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, alpha_const, sigma, omega), if compute_quadrature_quantities==True		#
	#													#
	# REFERENCES:	[1] G. Boeing, OSMnx: New methods for acquiring, constructing, analyzing, and	#
	#		visualizing complex street networks, Computers, Environment and Urban Systems,	#
	#		65 (2017), pp. 126-139, https://doi.org/10.1016/j.compenvurbsys.2017.05.004.		#
	#		[2] https://osmnx.readthedocs.io/en/stable/osmnx.html				#
	#		[3] K. Bergermann and M. Stoll, Orientations and matrix function-based		#
	#		centralities in multiplex network analysis of urban public transport, arXiv		#
	#		preprint, arXiv:2107.12695, (2021).							#
	#													#
	# 2021, Kai Bergermann (kai.bergermann@math.tu-chemnitz.de)						#
	#													#
	#####################################################################################################


	##### Build OSMnx street network #####
	print('Building OSMnx street network of %s...' % cityString)
	
	# plot city street network with plot of network edges according to the MLC of the corresponding route
	ox.config(log_console=False)

	G = ox.graph_from_place('%s, Germany' % cityString, network_type="drive")

	fig, ax = ox.plot.plot_graph(G, node_size=0, bgcolor='#FFFFFF', edge_color='#777777', node_color='#000000', show=False)
	
	
	##### Data preparations #####
	
	print('Starting on the preparation of the public transport network...')

	### filter stops by specified coordinates ###
	stops = pd.read_csv('gtfsdata/stops.txt', low_memory=False)

	# filter by polygon
	filteredStops = filter_by_polygon(cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"))


	print('Number of filtered stops by coordinates:', len(filteredStops))
	print('Starting with some GTFS data preprocessing...')


	### Get tripIDs over stopIDs ###
	stopTimes = pd.read_csv('gtfsdata/stop_times.txt', low_memory=False)
	filteredStopTimes = stopTimes[stopTimes['stop_id'].isin(filteredStops['stop_id'])] # drop all trips outside of the city
	tripIDs = np.unique(filteredStopTimes['trip_id']).tolist()

	filTimes = filteredStopTimes


	df = {'node_id': range(len(filteredStops)), 'stop_id': filteredStops['stop_id'], 'stop_name': filteredStops['stop_name'], 'stop_lat': filteredStops['stop_lat'], 'stop_lon': filteredStops['stop_lon']}
	stopIDTableDF = pd.DataFrame(data=df)


	### Get routeIDs over tripIDs ###
	trips = pd.read_csv('gtfsdata/trips.txt', low_memory=False)
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

	routeNames = pd.read_csv('gtfsdata/routes.txt', low_memory=False)
	routeNameTable=routeNames[routeNames['route_id'].isin(routeIDs)]
	longRouteNameTable=routeNameTable['route_long_name'].tolist()
	print('Long names of the routes:', longRouteNameTable)

	
	##### compute marginal layer centralities #####
		
	print('Computing the matrix function-based centralities now...')
	
	if compute_quadrature_quantities:
		SC_MLC, SCres_MLC, TC_MLC, KC_MLC = compute_centralities(cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times=weighted_with_travel_times, weighted_with_frequencies=weighted_with_frequencies, omega=omega, alpha_const=alpha_const, beta_const=beta_const, maxit_quadrature=maxit_quadrature, maxit_fAb=maxit_fAb, top_k_centralities=top_k_centralities, sigma=sigma, compute_quadrature_quantities=compute_quadrature_quantities, return_layer_centralities=True)
	else:
		TC_MLC, KC_MLC = compute_centralities(cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times=weighted_with_travel_times, weighted_with_frequencies=weighted_with_frequencies, omega=omega, alpha_const=alpha_const, beta_const=beta_const, maxit_quadrature=maxit_quadrature, maxit_fAb=maxit_fAb, top_k_centralities=top_k_centralities, sigma=sigma, compute_quadrature_quantities=compute_quadrature_quantities, return_layer_centralities=True)
	


	##### Plot lines #####
	
	if compute_quadrature_quantities:
	
		### SC
	
		# prepare line color scheme
		cmap = matplotlib.cm.get_cmap('coolwarm')
		
		if not n_color_compartments==0:
			SC_colors = SC_MLC
			SC_color_min = np.asscalar(min(SC_colors))
			SC_color_max = np.asscalar(max(SC_colors))
			SC_color_range = SC_color_max - SC_color_min
			SC_colors_compartmented = np.copy(SC_colors)

			SC_compartment_length = SC_color_range / n_color_compartments
			# first bin
			SC_colors_compartmented[SC_colors < (SC_color_min + SC_compartment_length)] = 0
			# last bin
			SC_colors_compartmented[SC_colors > (SC_color_max - SC_compartment_length)] = n_color_compartments-1
			# intermediate bins
			for i in range(1,n_color_compartments-1):
				SC_colors_compartmented[(SC_colors >= (SC_color_min + i*SC_compartment_length)) & (SC_colors < (SC_color_min + (i+1)*SC_compartment_length))] = i
		
		# sort routeIDs in ascending order by their MLC
		SC_MLC_sorted_ind = np.argsort(SC_MLC, axis=None)
		routeNamesSortedSCMLC = [longRouteNameTable[i] for i in SC_MLC_sorted_ind]
		routeIDsSortedSCMLC = []
		for routeName in routeNamesSortedSCMLC:
			if len(routeNameTable[routeNameTable['route_long_name']==routeName]['route_id']) > 1:
				nonUniqueList = routeNameTable[routeNameTable['route_long_name']==routeName]['route_id']
				for i in range(len(nonUniqueList)):
					routeIDsSortedSCMLC.append(nonUniqueList.iloc[i].item())
			else:
				routeIDsSortedSCMLC.append(routeNameTable[routeNameTable['route_long_name']==routeName]['route_id'].item())
		
		
		n = len(filteredStops)
		L = len(routeIDs)
		print('n=%d, L=%d' % (n, L))
		l=0
		print('Plotting lines for subgraph centrality\nOf %d total lines, Im currently plotting line (in case of non-uniqueness of some line names these may be counted/plotted more than once)' % L)
		# sorted such that important lines are plottet in the foreground
		for routeID in routeIDsSortedSCMLC:
			rouIDIndexForMLC = longRouteNameTable.index(routeNameTable[routeNameTable['route_id']==routeID]['route_long_name'].item())
		
			sys.stdout.write("\r{}".format(l+1))
			sys.stdout.flush()
			# get all trip_ids
			tripListOfRoute = filteredTrips[(filteredTrips['route_id'] == routeID)]['trip_id'].tolist()
			stopIDList = []
			for tripID in tripListOfRoute:
				# create list of lists of all trips of the given route and filter for unique trips
				stopIDList.append(filTimes[filTimes['trip_id'] == tripID]['stop_id'].tolist())
			uniquestopIDList = [list(x) for x in set(tuple(x) for x in stopIDList)]
			
			# only loop over the unique set of trips belonging to the respective route
			for stopIDs in uniquestopIDList:
				for i in range(len(stopIDs) - 1):
					# add edge between current item and last
					if n_color_compartments==0:
						ax.plot([stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['stop_lon'],stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['stop_lon']], [stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['stop_lat'],stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['stop_lat']], c=cmap((SC_MLC[rouIDIndexForMLC]-min(SC_MLC))/(max(SC_MLC)-min(SC_MLC))), linewidth=markersize)
					else:
						ax.plot([stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['stop_lon'],stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['stop_lon']], [stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['stop_lat'],stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['stop_lat']], c=cmap(SC_colors_compartmented[rouIDIndexForMLC]/(n_color_compartments-1)), linewidth=markersize)

			l+=1
		print('\n')
		
		### saving the figure
	
		directory = 'centralities/plots'

		if not os.path.exists(directory):
			os.makedirs(directory)

		fig.savefig("%s/%s_SC_layer_centralities_weighted_times_%s_freq_%s_beta_const_%s_sigma_%s_omega_%s.png" % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, beta_const, sigma, omega), dpi=200, bbox_inches='tight')

		
		### SCres
	
		# prepare line color scheme
		cmap = matplotlib.cm.get_cmap('coolwarm')
		
		if not n_color_compartments==0:
			SCres_colors = SCres_MLC
			SCres_color_min = np.asscalar(min(SCres_colors))
			SCres_color_max = np.asscalar(max(SCres_colors))
			SCres_color_range = SCres_color_max - SCres_color_min
			SCres_colors_compartmented = np.copy(SCres_colors)

			SCres_compartment_length = SCres_color_range / n_color_compartments
			# first bin
			SCres_colors_compartmented[SCres_colors < (SCres_color_min + SCres_compartment_length)] = 0
			# last bin
			SCres_colors_compartmented[SCres_colors > (SCres_color_max - SCres_compartment_length)] = n_color_compartments-1
			# intermediate bins
			for i in range(1,n_color_compartments-1):
				SCres_colors_compartmented[(SCres_colors >= (SCres_color_min + i*SCres_compartment_length)) & (SCres_colors < (SCres_color_min + (i+1)*SCres_compartment_length))] = i
		
		# sort routeIDs in ascending order by their MLC
		SCres_MLC_sorted_ind = np.argsort(SCres_MLC, axis=None)
		routeNamesSortedSCresMLC = [longRouteNameTable[i] for i in SCres_MLC_sorted_ind]
		routeIDsSortedSCresMLC = []
		for routeName in routeNamesSortedSCresMLC:
			if len(routeNameTable[routeNameTable['route_long_name']==routeName]['route_id']) > 1:
				nonUniqueList = routeNameTable[routeNameTable['route_long_name']==routeName]['route_id']
				for i in range(len(nonUniqueList)):
					routeIDsSortedSCresMLC.append(nonUniqueList.iloc[i].item())
			else:
				routeIDsSortedSCresMLC.append(routeNameTable[routeNameTable['route_long_name']==routeName]['route_id'].item())
		
		
		n = len(filteredStops)
		L = len(routeIDs)
		print('n=%d, L=%d' % (n, L))
		l=0
		print('Plotting lines for resolvent-based subgraph centrality\nOf %d total lines, Im currently plotting line (in case of non-uniqueness of some line names these may be counted/plotted more than once)' % L)
		# sorted such that important lines are plottet in the foreground
		for routeID in routeIDsSortedSCresMLC:
			rouIDIndexForMLC = longRouteNameTable.index(routeNameTable[routeNameTable['route_id']==routeID]['route_long_name'].item())
		
			sys.stdout.write("\r{}".format(l+1))
			sys.stdout.flush()
			# get all trip_ids
			tripListOfRoute = filteredTrips[(filteredTrips['route_id'] == routeID)]['trip_id'].tolist()
			stopIDList = []
			for tripID in tripListOfRoute:
				# create list of lists of all trips of the given route and filter for unique trips
				stopIDList.append(filTimes[filTimes['trip_id'] == tripID]['stop_id'].tolist())
			uniquestopIDList = [list(x) for x in set(tuple(x) for x in stopIDList)]
			
			# only loop over the unique set of trips belonging to the respective route
			for stopIDs in uniquestopIDList:
				for i in range(len(stopIDs) - 1):
					# add edge between current item and last
					if n_color_compartments==0:
						ax.plot([stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['stop_lon'],stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['stop_lon']], [stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['stop_lat'],stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['stop_lat']], c=cmap((SCres_MLC[rouIDIndexForMLC]-min(SCres_MLC))/(max(SCres_MLC)-min(SCres_MLC))), linewidth=markersize)
					else:
						ax.plot([stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['stop_lon'],stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['stop_lon']], [stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['stop_lat'],stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['stop_lat']], c=cmap(SCres_colors_compartmented[rouIDIndexForMLC]/(n_color_compartments-1)), linewidth=markersize)

			l+=1
		print('\n')
		
		### saving the figure

		fig.savefig("%s/%s_SCres_layer_centralities_weighted_times_%s_freq_%s_alpha_const_%s_sigma_%s_omega_%s.png" % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, alpha_const, sigma, omega), dpi=200, bbox_inches='tight')
		
	
	
	### TC
	
	# prepare line color scheme
	cmap = matplotlib.cm.get_cmap('coolwarm')
	
	if not n_color_compartments==0:
		TC_colors = TC_MLC
		TC_color_min = np.asscalar(min(TC_colors))
		TC_color_max = np.asscalar(max(TC_colors))
		TC_color_range = TC_color_max - TC_color_min
		TC_colors_compartmented = np.copy(TC_colors)

		TC_compartment_length = TC_color_range / n_color_compartments
		# first bin
		TC_colors_compartmented[TC_colors < (TC_color_min + TC_compartment_length)] = 0
		# last bin
		TC_colors_compartmented[TC_colors > (TC_color_max - TC_compartment_length)] = n_color_compartments-1
		# intermediate bins
		for i in range(1,n_color_compartments-1):
			TC_colors_compartmented[(TC_colors >= (TC_color_min + i*TC_compartment_length)) & (TC_colors < (TC_color_min + (i+1)*TC_compartment_length))] = i
	
	# sort routeIDs in ascending order by their MLC
	TC_MLC_sorted_ind = np.argsort(TC_MLC, axis=None)
	routeNamesSortedTCMLC = [longRouteNameTable[i] for i in TC_MLC_sorted_ind]
	routeIDsSortedTCMLC = []
	for routeName in routeNamesSortedTCMLC:
		if len(routeNameTable[routeNameTable['route_long_name']==routeName]['route_id']) > 1:
			nonUniqueList = routeNameTable[routeNameTable['route_long_name']==routeName]['route_id']
			for i in range(len(nonUniqueList)):
				routeIDsSortedTCMLC.append(nonUniqueList.iloc[i].item())
		else:
			routeIDsSortedTCMLC.append(routeNameTable[routeNameTable['route_long_name']==routeName]['route_id'].item())
	
	
	n = len(filteredStops)
	L = len(routeIDs)
	print('n=%d, L=%d' % (n, L))
	l=0
	print('Plotting lines for total communicability\nOf %d total lines, Im currently plotting line (in case of non-uniqueness of some line names these may be counted/plotted more than once)' % L)
	# sorted such that important lines are plottet in the foreground
	for routeID in routeIDsSortedTCMLC:
		rouIDIndexForMLC = longRouteNameTable.index(routeNameTable[routeNameTable['route_id']==routeID]['route_long_name'].item())
	
		sys.stdout.write("\r{}".format(l+1))
		sys.stdout.flush()
		# get all trip_ids
		tripListOfRoute = filteredTrips[(filteredTrips['route_id'] == routeID)]['trip_id'].tolist()
		stopIDList = []
		for tripID in tripListOfRoute:
			# create list of lists of all trips of the given route and filter for unique trips
			stopIDList.append(filTimes[filTimes['trip_id'] == tripID]['stop_id'].tolist())
		uniquestopIDList = [list(x) for x in set(tuple(x) for x in stopIDList)]
		
		# only loop over the unique set of trips belonging to the respective route
		for stopIDs in uniquestopIDList:
			for i in range(len(stopIDs) - 1):
				# add edge between current item and last
				if n_color_compartments==0:
					ax.plot([stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['stop_lon'],stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['stop_lon']], [stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['stop_lat'],stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['stop_lat']], c=cmap((TC_MLC[rouIDIndexForMLC]-min(TC_MLC))/(max(TC_MLC)-min(TC_MLC))), linewidth=markersize)
				else:
					ax.plot([stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['stop_lon'],stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['stop_lon']], [stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['stop_lat'],stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['stop_lat']], c=cmap(TC_colors_compartmented[rouIDIndexForMLC]/(n_color_compartments-1)), linewidth=markersize)

		l+=1
	print('\n')
		

	### saving the figure
	
	directory = 'centralities/plots'

	if not os.path.exists(directory):
		os.makedirs(directory)

	fig.savefig("%s/%s_TC_layer_centralities_weighted_times_%s_freq_%s_beta_const_%s_sigma_%s_omega_%s.png" % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, beta_const, sigma, omega), dpi=200, bbox_inches='tight')

	
	
	### KC
	
	# prepare line color scheme
	cmap = matplotlib.cm.get_cmap('coolwarm')
		
	if not n_color_compartments==0:
		KC_colors = KC_MLC
		KC_color_min = np.asscalar(min(KC_colors))
		KC_color_max = np.asscalar(max(KC_colors))
		KC_color_range = KC_color_max - KC_color_min
		KC_colors_compartmented = np.copy(KC_colors)

		KC_compartment_length = KC_color_range / n_color_compartments
		# first bin
		KC_colors_compartmented[KC_colors < (KC_color_min + KC_compartment_length)] = 0
		# last bin
		KC_colors_compartmented[KC_colors > (KC_color_max - KC_compartment_length)] = n_color_compartments-1
		# intermediate bins
		for i in range(1,n_color_compartments-1):
			KC_colors_compartmented[(KC_colors >= (KC_color_min + i*KC_compartment_length)) & (KC_colors < (KC_color_min + (i+1)*KC_compartment_length))] = i

	
	# sort routeIDs in ascending order by their MLC
	KC_MLC_sorted_ind = np.argsort(KC_MLC, axis=None)
	routeNamesSortedKCMLC = [longRouteNameTable[i] for i in KC_MLC_sorted_ind]
	routeIDsSortedKCMLC = []
	for routeName in routeNamesSortedKCMLC:
		if len(routeNameTable[routeNameTable['route_long_name']==routeName]['route_id']) > 1:
			nonUniqueList = routeNameTable[routeNameTable['route_long_name']==routeName]['route_id']
			for i in range(len(nonUniqueList)):
				routeIDsSortedKCMLC.append(nonUniqueList.iloc[i].item())
		else:
			routeIDsSortedKCMLC.append(routeNameTable[routeNameTable['route_long_name']==routeName]['route_id'].item())
	
	n = len(filteredStops)
	L = len(routeIDs)
	print('n=%d, L=%d' % (n, L))
	l=0
	print('Plotting lines for Katz centrality\nOf %d total lines, Im currently plotting line (in case of non-uniqueness of some line names these may be counted/plotted more than once)' % L)
	# sorted such that important lines are plottet in the foreground
	#print('routeIDsSorted', routeIDsSorted)
	for routeID in routeIDsSortedKCMLC:
		rouIDIndexForMLC = longRouteNameTable.index(routeNameTable[routeNameTable['route_id']==routeID]['route_long_name'].item())
	
		sys.stdout.write("\r{}".format(l+1))
		sys.stdout.flush()
		# get all trip_ids
		tripListOfRoute = filteredTrips[(filteredTrips['route_id'] == routeID)]['trip_id'].tolist()
		stopIDList = []
		for tripID in tripListOfRoute:
			# create list of lists of all trips of the given route and filter for unique trips
			stopIDList.append(filTimes[filTimes['trip_id'] == tripID]['stop_id'].tolist())
		uniquestopIDList = [list(x) for x in set(tuple(x) for x in stopIDList)]
		
		# only loop over the unique set of trips belonging to the respective route
		for stopIDs in uniquestopIDList:
			for i in range(len(stopIDs) - 1):
				# add edge between current item and last
				if n_color_compartments==0:
					ax.plot([stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['stop_lon'],stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['stop_lon']], [stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['stop_lat'],stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['stop_lat']], c=cmap((KC_MLC[rouIDIndexForMLC]-min(KC_MLC))/(max(KC_MLC)-min(KC_MLC))), linewidth=markersize)
				else:
					ax.plot([stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['stop_lon'],stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['stop_lon']], [stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i]]['stop_lat'],stopIDTableDF[stopIDTableDF['stop_id']==stopIDs[i+1]]['stop_lat']], c=cmap(KC_colors_compartmented[rouIDIndexForMLC]/(n_color_compartments-1)), linewidth=markersize)

		l+=1
	print('\n')

	### saving the figure

	fig.savefig("%s/%s_KC_layer_centralities_weighted_times_%s_freq_%s_alpha_const_%s_sigma_%s_omega_%s.png" % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, alpha_const, sigma, omega), dpi=200, bbox_inches='tight')

	plt.close()

