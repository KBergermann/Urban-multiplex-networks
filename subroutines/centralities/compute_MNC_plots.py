import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import sys
import pandas as pd
from .compute_centralities import *


def compute_MNC_plots(cityString, compute_quadrature_quantities=True, weighted_with_travel_times=True, weighted_with_frequencies=True, omega=1, alpha_const=.5, beta_const=.5, maxit_quadrature=5, maxit_fAb=10, top_k_centralities=10, sigma=1, n_color_compartments=6, scatter_markersize=10):

	#####################################################################################################
	#													#
	# DESCRIPTION:	computes marginal node centralities for the city specified in 'cityString' and	#
	#		visualizes them as scatter plots on top of the city's street network created by	#
	#		OSMnx [1,2]. Several parameters for the measures and the numerical schemes as well	#
	#		as options for plotting are specified in the INPUT section.				#
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
	#		weighted_with_frequencies (optional, default=True): option passed to the function	# 		#			'compute_centralities'. If True, line frequencies are used to weight intra-	#
	#			layer edges, cf. [3, Eq. (4.4)].						#
	#		omega (optional, default=1): option passed to the function 'compute_centralities'.	# 		#			Specifies the value of the inter-layer coupling parameter, cf. [3, Eq.(4.3)],#
	#			which models a constant transfer time across the network.			#
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
	#			heatmap color scheme (dark blue for the least central stops, dark red for	#
	#			the most central stops) is used for the visualization of the marginal node	#
	#			centralities.									#
	#		scatter_markersize (optional, default=10): specifies the markersize of the scatter	#
	#			plots.										#
	#													#
	# OUTPUT: 	None											#
	#													#
	# FILES												#
	# CREATED:	'centralities/plots/%s_TC_weighted_times_%s_freq_%s_beta_const_%s_sigma_%s_omega_%s.png' % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, beta_const, sigma, omega)				#
	#		'centralities/plots/%s_KC_weighted_times_%s_freq_%s_alpha_const_%s_sigma_%s_omega_%s.png' % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, alpha_const, sigma, omega)				#
	#		'centralities/plots/%s_SC_weighted_times_%s_freq_%s_beta_const_%s_sigma_%s_omega_%s.png' % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, beta_const, sigma, omega), if compute_quadrature_quantities==True										#
	#		'centralities/plots/%s_SCres_weighted_times_%s_freq_%s_alpha_const_%s_sigma_%s_omega_%s.png' % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, alpha_const, sigma, omega), if compute_quadrature_quantities==True										#
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
	
	
	# read data
	stopList = pd.read_csv('adjacency_matrices/%s_stop_IDs.csv' % cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"))

	n = len(stopList)

	# compute marginal node centralities
	if compute_quadrature_quantities:
		SC_MNC, SCres_MNC, TC_MNC, KC_MNC = compute_centralities(cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times=weighted_with_travel_times, weighted_with_frequencies=weighted_with_frequencies, omega=omega, alpha_const=alpha_const, beta_const=beta_const, maxit_quadrature=maxit_quadrature, maxit_fAb=maxit_fAb, top_k_centralities=top_k_centralities, sigma=sigma, compute_quadrature_quantities=compute_quadrature_quantities)
	else:
		TC_MNC, KC_MNC = compute_centralities(cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times=weighted_with_travel_times, weighted_with_frequencies=weighted_with_frequencies, omega=omega, alpha_const=alpha_const, beta_const=beta_const, maxit_quadrature=maxit_quadrature, maxit_fAb=maxit_fAb, top_k_centralities=top_k_centralities, sigma=sigma, compute_quadrature_quantities=compute_quadrature_quantities)

	# plot city street network with scatter plot (foreground the (few) central nodes)
	ox.config(log_console=True)

	G = ox.graph_from_place('%s, Germany' % cityString, network_type="drive")

	# saving the figures
	directory = 'centralities/plots'

	if not os.path.exists(directory):
		os.makedirs(directory)
	if compute_quadrature_quantities:
		### SC
		fig, ax = ox.plot.plot_graph(G, node_size=0, bgcolor='#FFFFFF', edge_color='#777777', node_color='#000000', show=False)

		SC_MNC_sorted_ind = np.argsort(SC_MNC, axis=None)
		SC_MNC_sorted_asc = SC_MNC[SC_MNC_sorted_ind]

		if n_color_compartments==0:
			ax.scatter(stopList['stop_lon'][SC_MNC_sorted_ind], stopList['stop_lat'][SC_MNC_sorted_ind], s=scatter_markersize*np.ones([n,1]), c=SC_MNC[SC_MNC_sorted_ind], cmap='coolwarm', rasterized=True)
			ax.title.set_text('%s centrality scatter plot SC' % cityString)
		else:
			SC_colors = SC_MNC[SC_MNC_sorted_ind]
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
				

			ax.scatter(stopList['stop_lon'][SC_MNC_sorted_ind], stopList['stop_lat'][SC_MNC_sorted_ind], s=scatter_markersize*np.ones([n,1]), c=SC_colors_compartmented, cmap='coolwarm')#, rasterized=True)
			#ax.title.set_text('%s centrality scatter plot SC' % cityString)

		fig.savefig("%s/%s_SC_weighted_times_%s_freq_%s_beta_const_%s_sigma_%s_omega_%s.png" % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, beta_const, sigma, omega), dpi=200, bbox_inches='tight')


		### SCres
		fig, ax = ox.plot.plot_graph(G, node_size=0, bgcolor='#FFFFFF', edge_color='#777777', node_color='#000000', show=False)

		SCres_MNC_sorted_ind = np.argsort(SCres_MNC, axis=None)
		SCres_MNC_sorted_asc = SCres_MNC[SCres_MNC_sorted_ind]

		if n_color_compartments==0:
			ax.scatter(stopList['stop_lon'][SCres_MNC_sorted_ind], stopList['stop_lat'][SCres_MNC_sorted_ind], s=scatter_markersize*np.ones([n,1]), c=SCres_MNC[SCres_MNC_sorted_ind], cmap='coolwarm', rasterized=True)
			ax.title.set_text('%s centrality scatter plot SCres' % cityString)
		else:
			SCres_colors = SCres_MNC[SCres_MNC_sorted_ind]
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
				

			ax.scatter(stopList['stop_lon'][SCres_MNC_sorted_ind], stopList['stop_lat'][SCres_MNC_sorted_ind], s=scatter_markersize*np.ones([n,1]), c=SCres_colors_compartmented, cmap='coolwarm')#, rasterized=True)
			#ax.title.set_text('%s centrality scatter plot SCres' % cityString)

		fig.savefig("%s/%s_SCres_weighted_times_%s_freq_%s_alpha_const_%s_sigma_%s_omega_%s.png" % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, alpha_const, sigma, omega), dpi=200, bbox_inches='tight')



	### TC
	fig, ax = ox.plot.plot_graph(G, node_size=0, bgcolor='#FFFFFF', edge_color='#777777', node_color='#000000', show=False)

	TC_MNC_sorted_ind = np.argsort(TC_MNC, axis=None)
	TC_MNC_sorted_asc = TC_MNC[TC_MNC_sorted_ind]

	if n_color_compartments==0:
		ax.scatter(stopList['stop_lon'][TC_MNC_sorted_ind], stopList['stop_lat'][TC_MNC_sorted_ind], s=scatter_markersize*np.ones([n,1]), c=TC_MNC[TC_MNC_sorted_ind], cmap='coolwarm', rasterized=True)
		ax.title.set_text('%s centrality scatter plot TC' % cityString)
	else:
		TC_colors = TC_MNC[TC_MNC_sorted_ind]
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
			

		ax.scatter(stopList['stop_lon'][TC_MNC_sorted_ind], stopList['stop_lat'][TC_MNC_sorted_ind], s=scatter_markersize*np.ones([n,1]), c=TC_colors_compartmented, cmap='coolwarm')#, rasterized=True)
		#ax.title.set_text('%s centrality scatter plot TC' % cityString)


	fig.savefig("%s/%s_TC_weighted_times_%s_freq_%s_beta_const_%s_sigma_%s_omega_%s.png" % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, beta_const, sigma, omega), dpi=200, bbox_inches='tight')


	### KC
	fig, ax = ox.plot.plot_graph(G, node_size=0, bgcolor='#FFFFFF', edge_color='#777777', node_color='#000000', show=False)

	KC_MNC_sorted_ind = np.argsort(KC_MNC, axis=None)
	KC_MNC_sorted_asc = KC_MNC[KC_MNC_sorted_ind]
	
	if n_color_compartments==0:
		ax.scatter(stopList['stop_lon'][KC_MNC_sorted_ind], stopList['stop_lat'][KC_MNC_sorted_ind], s=scatter_markersize*np.ones([n,1]), c=KC_MNC[KC_MNC_sorted_ind], cmap='coolwarm', rasterized=True)
		ax.title.set_text('%s centrality scatter plot KC' % cityString)
	else:
		KC_colors = KC_MNC[KC_MNC_sorted_ind]
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
			

		ax.scatter(stopList['stop_lon'][KC_MNC_sorted_ind], stopList['stop_lat'][KC_MNC_sorted_ind], s=scatter_markersize*np.ones([n,1]), c=KC_colors_compartmented, cmap='coolwarm')#, rasterized=True) # 'rasterized=True' somewhats reduces the file size, but you also loose quality...
		#ax.title.set_text('%s centrality scatter plot KC' % cityString)

	fig.savefig("%s/%s_KC_weighted_times_%s_freq_%s_alpha_const_%s_sigma_%s_omega_%s.png" % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, alpha_const, sigma, omega), dpi=200, bbox_inches='tight')

	plt.close()

