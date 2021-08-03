import numpy as np
import matplotlib.pyplot as plt
from .compute_centralities import *


def compute_MCs_varying_omega(cityString, compute_quadrature_quantities=False, weighted_with_travel_times=True, weighted_with_frequencies=True, alpha_const=0.5, beta_const=0.5, maxit_quadrature=5, maxit_fAb=10, top_k_centralities=10, sigma=1, suppress_output=False):

	#####################################################################################################
	#													#
	# DESCRIPTION:	computes marginal node and marginal layer total communicabilities (TC) for the city	#
	#		specified in 'cityString' for different values of the coupling parameter omega 	#
	#		[1, Eq. (4.3)] (which models a constant transfer time across the network) and	#
	#		creates semi-logarithmic plots. The code can be amended to produce the same plots	#
	#		for the measures SC, SC_res, and KC.							#
	#													#
	# INPUT: 	cityString (required): string containing the city name for which the centralities	#
	#			are to be computed. The required adjacency matrices must be available in	#
	#			the directory 'adjacency_matrices' under the same name.			#
	#		compute_quadrature_quantities (optional, default=False): option passed to the 	#
	#			function 'compute_centralities'. If True, the measures SC, SCres, TC, and KC	#
	#			are computed. Otherwise, only TC and KC are computed.			#
	#		weighted_with_travel_times (optional, default=True): option passed to the function	#
	#			'compute_centralities'. If True, a Gaussian kernel applied to the travel	#
	#			times is used to weight intra-layer edges, cf. [1, Eq. (4.4)].		#
	#		weighted_with_frequencies (optional, default=True): option passed to the function	# 
	#			'compute_centralities'. If True, line frequencies are used to weight intra-	#
	#			layer edges, cf. [1, Eq. (4.4)].						#
	#		alpha_const (optional, default=0.5): option passed to the function 			#
	#			'compute_centralities'. Specifies the value of alpha in [1, Eq. (5.2)&(5.3)]	#
	#			via alpha=alpha_const/lambda_max.						#
	#		beta_const (optional, default=0.5): option passed to the function 			#
	#			'compute_centralities'. Specifies the value of beta in [1, Eq. (5.2)&(5.3)]	#
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
	#			Gaussian kernel, which is applied to travel times to obtain intra-layer	#
	#			weights.									#
	#		suppress_output (optional, default=False): option passed to the function		#
	#			'compute_centralities'. If True, no txt-file with the top-ranked nodes,	#
	#			layers, and node-layer pairs is produced. Otherwise, a txt-file is saved in	#
	#			the directory 'centralities/results'						#
	#													#
	# OUTPUT: 	None											#
	#													#
	# FILES												#
	# CREATED:	'centralities/plots/%s_TC_MNC_varying_omega_weighted_times_%s_freq_%s_beta_const_%s_sigma_%.2f.png' % (cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, beta_const, sigma)					#
	#		'centralities/plots/%s_TC_MLC_varying_omega_weighted_times_%s_freq_%s_beta_const_%s_sigma_%.2f.png' % (cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, beta_const, sigma)					#
	#													#
	# REFERENCE:	[1] K. Bergermann and M. Stoll, Orientations and matrix function-based		#
	#		centralities in multiplex network analysis of urban public transport, arXiv		#
	#		preprint, arXiv:2107.12695, (2021).							#
	#													#
	# 2021, Kai Bergermann (kai.bergermann@math.tu-chemnitz.de)						#
	#													#
	#####################################################################################################

	# load node and layer lists and define network size parameters
	layerIDList = pd.read_csv('adjacency_matrices/%s_layer_IDs.csv' % cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"))
	stopIDList = pd.read_csv('adjacency_matrices/%s_stop_IDs.csv' % cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"))
	L = len(layerIDList)
	n = len(stopIDList)

	nL = n*L

	# define omega array
	omegas = np.logspace(-3, 0, base=10, num=13)

	# compute marginal centralities
	if compute_quadrature_quantities:
		# nodes
		SC_MNC, SCres_MNC, TC_MNC, KC_MNC = compute_centralities(cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times=weighted_with_travel_times, weighted_with_frequencies=weighted_with_frequencies, omega=omegas[0], alpha_const=alpha_const, beta_const=beta_const, maxit_quadrature=maxit_quadrature, maxit_fAb=maxit_fAb, top_k_centralities=top_k_centralities, sigma=sigma, compute_quadrature_quantities=compute_quadrature_quantities, suppress_output=suppress_output)
		# layers
		SC_MLC, SCres_MLC, TC_MLC, KC_MLC = compute_centralities(cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times=weighted_with_travel_times, weighted_with_frequencies=weighted_with_frequencies, omega=omegas[0], alpha_const=alpha_const, beta_const=beta_const, maxit_quadrature=maxit_quadrature, maxit_fAb=maxit_fAb, top_k_centralities=top_k_centralities, sigma=sigma, compute_quadrature_quantities=compute_quadrature_quantities, return_layer_centralities=True, suppress_output=suppress_output)
	else:
		# nodes
		TC_MNC, KC_MNC = compute_centralities(cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times=weighted_with_travel_times, weighted_with_frequencies=weighted_with_frequencies, omega=omegas[0], alpha_const=alpha_const, beta_const=beta_const, maxit_quadrature=maxit_quadrature, maxit_fAb=maxit_fAb, top_k_centralities=top_k_centralities, sigma=sigma, compute_quadrature_quantities=compute_quadrature_quantities, suppress_output=suppress_output)
		# layers
		TC_MLC, KC_MLC = compute_centralities(cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times=weighted_with_travel_times, weighted_with_frequencies=weighted_with_frequencies, omega=omegas[0], alpha_const=alpha_const, beta_const=beta_const, maxit_quadrature=maxit_quadrature, maxit_fAb=maxit_fAb, top_k_centralities=top_k_centralities, sigma=sigma, compute_quadrature_quantities=compute_quadrature_quantities, return_layer_centralities=True, suppress_output=suppress_output)

	# identify nodes and layers, in which not all node-layer pairs have the trivial value 1
	TC_MNC_nontrivial = ((TC_MNC-L)>1e-12)
	TC_MLC_nontrivial = ((TC_MLC-n)>1e-12)

	# define matrix, which shall contain the MC arrays for the different values of omega
	TC_MNC_array = np.zeros([n, len(omegas)])
	TC_MNC_array[:,0] = TC_MNC
	TC_MLC_array = np.zeros([L, len(omegas)])
	TC_MLC_array[:,0] = TC_MLC

	# loop over omega array and compute MCs
	for i in range(1,len(omegas)):
		if compute_quadrature_quantities:
			# nodes
			SC_MNC, SCres_MNC, TC_MNC, KC_MNC = compute_centralities(cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times=weighted_with_travel_times, weighted_with_frequencies=weighted_with_frequencies, omega=omegas[i], alpha_const=alpha_const, beta_const=beta_const, maxit_quadrature=maxit_quadrature, maxit_fAb=maxit_fAb, top_k_centralities=top_k_centralities, sigma=sigma, compute_quadrature_quantities=compute_quadrature_quantities, suppress_output=suppress_output)
			# layers
			SC_MLC, SCres_MLC, TC_MLC, KC_MLC = compute_centralities(cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times=weighted_with_travel_times, weighted_with_frequencies=weighted_with_frequencies, omega=omegas[i], alpha_const=alpha_const, beta_const=beta_const, maxit_quadrature=maxit_quadrature, maxit_fAb=maxit_fAb, top_k_centralities=top_k_centralities, sigma=sigma, compute_quadrature_quantities=compute_quadrature_quantities, return_layer_centralities=True, suppress_output=suppress_output)
		else:
			# nodes
			TC_MNC, KC_MNC = compute_centralities(cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times=weighted_with_travel_times, weighted_with_frequencies=weighted_with_frequencies, omega=omegas[i], alpha_const=alpha_const, beta_const=beta_const, maxit_quadrature=maxit_quadrature, maxit_fAb=maxit_fAb, top_k_centralities=top_k_centralities, sigma=sigma, compute_quadrature_quantities=compute_quadrature_quantities, suppress_output=suppress_output)
			# layers
			TC_MLC, KC_MLC = compute_centralities(cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times=weighted_with_travel_times, weighted_with_frequencies=weighted_with_frequencies, omega=omegas[i], alpha_const=alpha_const, beta_const=beta_const, maxit_quadrature=maxit_quadrature, maxit_fAb=maxit_fAb, top_k_centralities=top_k_centralities, sigma=sigma, compute_quadrature_quantities=compute_quadrature_quantities, return_layer_centralities=True, suppress_output=suppress_output)
		
		TC_MNC_array[:,i] = TC_MNC
		TC_MLC_array[:,i] = TC_MLC

	# saving pyplots to file
	directory = 'centralities/plots'

	if not os.path.exists(directory):
		os.makedirs(directory)

	fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,7))
	ax.semilogx(omegas, TC_MNC_array[TC_MNC_nontrivial].T)
	plt.xlabel('$\omega$')
	plt.ylabel('MNC')
	fig.savefig("%s/%s_TC_MNC_varying_omega_weighted_times_%s_freq_%s_beta_const_%s_sigma_%.2f.png" % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, beta_const, sigma), dpi=200, bbox_inches='tight')

	fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,7))
	ax.semilogx(omegas, TC_MLC_array[TC_MLC_nontrivial].T)
	plt.xlabel('$\omega$')
	plt.ylabel('MLC')
	fig.savefig("%s/%s_TC_MLC_varying_omega_weighted_times_%s_freq_%s_beta_const_%s_sigma_%.2f.png" % (directory, cityString.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace(" ", "_"), weighted_with_travel_times, weighted_with_frequencies, beta_const, sigma), dpi=200, bbox_inches='tight')
	
	plt.close()

