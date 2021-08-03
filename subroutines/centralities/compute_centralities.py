import scipy.sparse as spsp
import numpy as np
import numpy.linalg as la
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from .lanczos import *
from .fAb import *
from .gauss_quadrature_rules import *


def compute_centralities(cityString, weighted_with_travel_times=True, weighted_with_frequencies=True, omega=1, alpha_const=0.5, beta_const=0.5, maxit_quadrature=5, maxit_fAb=10, top_k_centralities=10, sigma=1, compute_quadrature_quantities=True, return_layer_centralities=False, suppress_output=False):

	#####################################################################################################
	#													#
	# DESCRIPTION:	computes multiplex matrix function-based centralities for the city specified in 	#
	#		'cityString'. The required adjacency matrices must be available in the directory	#
	#		'adjacency_matrices'. Several parameters for the measures and the numerical schemes	#
	#		as well as options on which measures to compute and how to save the results are	#
	#		specified in the INPUT section.							#
	#													#
	# INPUT: 	cityString (required): string containing the city name for which the centralities	#
	#			are to be computed. The required adjacency matrices must be available in	#
	#			the directory 'adjacency_matrices' under the same name.			#
	#		weighted_with_travel_times (optional, default=True): if True, a Gaussian kernel 	#
	#			applied to the travel times is used to weight intra-layer edges, 		#
	#			cf. [1, Eq. (4.4)]. 								#
	#		weighted_with_frequencies (optional, default=True): if True, line frequencies are 	#
	#			used to weight intra-layer edges, cf. [1, Eq. (4.4)].			#
	#		omega (optional, default=1): specifies the value of the inter-layer coupling 	#
	#			parameter, cf. [1, Eq. (4.3)], which models a constant transfer time across	#
	#			the network.									#
	#		alpha_const (optional, default=0.5): specifies the value of alpha in 		#
	#			[1, Eq. (5.2)&(5.3)] via alpha=alpha_const/lambda_max.			#
	#		beta_const (optional, default=0.5): specifies the value of beta in 			#
	#			[1, Eq. (5.2)&(5.3)] via beta=beta_const/lambda_max.				#
	#		maxit_quadrature (optional, default=5): specifies the maximum number of Lanczos	#
	#			iterations used to compute SC and SC_res.					#
	#		maxit_fAb (optional, default=10): specifies the maximum number of Lanczos		#
	#			iterations used to compute TC and KC.						#
	#		top_k_centralities (optional, default=10): specifies the number of leading nodes, 	#
	#			layers, and node-layer pairs to be displayed in the rankings.		#
	#		sigma (optional, default=1): if weighted_with_travel_times==True, specifies the	#
	#			scaling parameter in the Gaussian kernel, which is applied to travel times	#
	#			to obtain intra-layer weights.						#
	#		compute_quadrature_quantities (optional, default=True): if True, computes SC, 	#
	#			SCres, TC, and KC. Otherwise, computes only TC and KC.			#
	#		return_layer_centralities (optional, default=False): if True, returns marginal 	#
	#			layer centralities. Otherwise, returns marginal node centralities.		#
	#		suppress_output (optional, default=False): if True, no txt-file with the top-ranked	#
	#			nodes, layers, and node-layer pairs is produced. Otherwise, a txt-file is 	#
	#			saved in the directory 'centralities/results'					#
	#													#
	# OUTPUT: 	SC_MLC, SCres_MLC, TC_MLC, KC_MLC (numpy.ndarray), if				#
	#			compute_quadrature_quantities==True and return_layer_centralities==True,	#
	#		TC_MLC, KC_MLC (numpy.ndarray), if compute_quadrature_quantities==False and		#
	#			return_layer_centralities==True, 						#
	#		SC_MNC, SCres_MNC, TC_MNC, KC_MNC (numpy.ndarray), if				#
	#			compute_quadrature_quantities==True and return_layer_centralities==False,	#
	#		TC_MNC, KC_MNC (numpy.ndarray): if compute_quadrature_quantities==False and		#
	#			return_layer_centralities==False.						#
	#													#
	# FILES												#
	# CREATED:	'centralities/results/centralities_%s_weighted_times_%s_freq_%s_alpha_const_%s_beta_const_%s_sigma_%s_omega_%s.txt' % (cityString, weighted_with_travel_times, weighted_with_frequencies, alpha_const, beta_const, sigma, omega), if suppress_output==False												#
	#													#
	# REFERENCE:	[1] K. Bergermann and M. Stoll, Orientations and matrix function-based		#
	#		centralities in multiplex network analysis of urban public transport, arXiv		#
	#		preprint, arXiv:2107.12695, (2021).							#
	#													#
	# 2021, Kai Bergermann (kai.bergermann@math.tu-chemnitz.de)						#
	#													#
	#####################################################################################################


	# Output: 	txt file saved in 'centralities/results' unless option suppress_output is set to True
	#		MLC or MNC (option return_layer_centralities)
	#		all four quantities or only TC and KC (option compute_quadrature_quantities)


	####################
	### Preparations ###
	####################

	# read and prepare data
	if not weighted_with_travel_times: # does not include travel times as weights
		if weighted_with_frequencies: # weight with frequencies but not with travel times
			Aintra = spsp.load_npz('adjacency_matrices/%s_Aintra_weighted_frequencies.npz' % cityString)
		else: # entirely unweighted case
			Aintra = spsp.load_npz('adjacency_matrices/%s_Aintra_unweighted.npz' % cityString)
		Ainter = spsp.load_npz('adjacency_matrices/%s_Ainter_unweighted.npz' % cityString)

	else: # does include travel times as weights
		Aintra = spsp.load_npz('adjacency_matrices/%s_Aintra_weighted.npz' % cityString)
		Ainter = spsp.load_npz('adjacency_matrices/%s_Ainter_unweighted.npz' % cityString)

		# apply Gaussian kernel to travel time weights in the weighted inter-layer adjacency matrix
		Aintra.data = np.exp(-Aintra.data**2/sigma**2)

		if weighted_with_frequencies: # includes travel times and frequencies in the weights
			Aintra_frequencies = spsp.load_npz('adjacency_matrices/%s_Aintra_weighted_frequencies.npz' % cityString)

			# elementwise multiplication with frequencies
			Aintra = Aintra_frequencies.multiply(Aintra)

				

	stopIDList = pd.read_csv('adjacency_matrices/%s_stop_IDs.csv' % cityString)
	layerIDList = pd.read_csv('adjacency_matrices/%s_layer_IDs.csv' % cityString)

	n = len(stopIDList)
	L = len(layerIDList)

	nL, nL = Aintra.shape
	nL_check, nL_check = Ainter.shape

	# sanity check on imported data
	if n*L != nL or n*L != nL_check:
		print('WARNING! Numbers from supra-adjacency matrices and stop and layer ID lists do not add up! Stop and layer names in the results are likely to be incorrect!\nNumbers: n*L: %d, nL: %d, nL_check: %d' % (n*L, nL, nL_check))

	A = Aintra + omega*Ainter

	# filter for non-isolated node-layer pairs
	deg = A.dot(np.ones([nL,1]))
	print('%s. nL=%d. Number of nodes with at least one edge: %d.'% (cityString, nL, np.sum(1*(deg>0))))

	## deg>0
	deg_g1_idcs = np.where(deg>0)[0]


	if compute_quadrature_quantities:
		# set centrality of isolated nodes to 1
		SC_gauss_lower = np.ones([nL,1])
		SC_gauss_radau_lower = np.ones([nL,1])
		SC_gauss_radau_upper = np.ones([nL,1])
		SC_gauss_lobatto_upper = np.ones([nL,1])

		SCres_gauss_lower = np.ones([nL,1])
		SCres_gauss_radau_lower = np.ones([nL,1])
		SCres_gauss_radau_upper = np.ones([nL,1])
		SCres_gauss_lobatto_upper = np.ones([nL,1])

		#############################
		### u^T f(A) u quantities ###
		#############################

		print('Subgraph and resolvent-based subgraph centrality. Looping over non-isolated nodes. Progress:')

		# estimate extremal eigenvalues
		U, T = lanczos(A, np.ones([nL,1]), 20) # one vector seems to lead to less numerical trouble than a random vector..

		lamb, phi = la.eig(T)
		lamb_min = min(lamb)
		lamb_max = max(lamb)

		# loop over non-isolated node-layer pairs
		i=0
		for node_id in deg_g1_idcs:

			sys.stdout.write("\r{}%".format(100*((i+1)/len(deg_g1_idcs))))
			sys.stdout.flush()

			u = np.zeros([nL,1])
			u[node_id]=1

			U, T = lanczos(A, u, maxit_quadrature)
			T_copy1 = np.copy(T)
			T_copy2 = np.copy(T)
			T_copy3 = np.copy(T)
			T_copy4 = np.copy(T)
			T_copy5 = np.copy(T)
			T_copy6 = np.copy(T)
			T_copy7 = np.copy(T)

			# subgraph centrality quadrature rules
			beta_subgraph = beta_const/lamb_max

			SC_gauss_lower[node_id] = gauss_subgraph(T, beta_subgraph)
			SC_gauss_radau_lower[node_id] = gauss_radau_subgraph(T_copy1, beta_subgraph, lamb_min)
			SC_gauss_radau_upper[node_id] = gauss_radau_subgraph(T_copy2, beta_subgraph, lamb_max)
			SC_gauss_lobatto_upper[node_id] = gauss_lobatto_subgraph(T_copy3, beta_subgraph, lamb_min, lamb_max)

			# resolvent-based subgraph centrality quadrature rules
			alpha_resolvent = alpha_const/lamb_max

			SCres_gauss_lower[node_id] = gauss_resolvent(T_copy4, alpha_resolvent)
			SCres_gauss_radau_lower[node_id] = gauss_radau_resolvent(T_copy5, alpha_resolvent, lamb_min)
			SCres_gauss_radau_upper[node_id] = gauss_radau_resolvent(T_copy6, alpha_resolvent, lamb_max)
			SCres_gauss_lobatto_upper[node_id] = gauss_lobatto_resolvent(T_copy7, alpha_resolvent, lamb_min, lamb_max)

			i+=1

		SC = SC_gauss_radau_lower
		SCres = SCres_gauss_radau_lower


	########################
	### f(A)b quantities ###
	########################

	b = np.ones([nL,1])

	U, T = lanczos(A, b, maxit_fAb)

	lamb, phi = la.eig(T[0:maxit_fAb, 0:maxit_fAb])
	lamb_min = min(lamb)
	lamb_max = max(lamb)

	### Total communicability ###

	beta_subgraph = beta_const/lamb_max

	TC = expAb_sym(U[:, 0:maxit_fAb], T[0:maxit_fAb, 0:maxit_fAb], b, beta_subgraph)


	### Katz centrality ###

	alpha_resolvent = alpha_const/lamb_max

	KC = resolventAb_sym(U[:, 0:maxit_fAb], T[0:maxit_fAb, 0:maxit_fAb], b, alpha_resolvent)


	############################################
	### print and write centralities to file ###
	############################################

	if compute_quadrature_quantities:
		SC_MNC = np.sum(SC.reshape((L,n)).T, axis=1)
		SC_MLC = np.sum(SC.reshape((L,n)).T, axis=0)
		SCres_MNC = np.sum(SCres.reshape((L,n)).T, axis=1)
		SCres_MLC = np.sum(SCres.reshape((L,n)).T, axis=0)
		
	TC_MNC = np.sum(TC.reshape((L,n)).T, axis=1)
	TC_MLC = np.sum(TC.reshape((L,n)).T, axis=0)
	KC_MNC = np.sum(KC.reshape((L,n)).T, axis=1)
	KC_MLC = np.sum(KC.reshape((L,n)).T, axis=0)
			
	if not suppress_output:
		# printing to file
		directory = 'centralities/results'

		if not os.path.exists(directory):
			os.makedirs(directory)

		print_file = open("%s/centralities_%s_weighted_times_%s_freq_%s_alpha_const_%s_beta_const_%s_sigma_%s_omega_%s.txt" % (directory, cityString, weighted_with_travel_times, weighted_with_frequencies, alpha_const, beta_const, sigma, omega), "w")
		print_file.write('Parameters:\nalpha_const=%f, beta_const=%f, maxit_quadrature=%d, maxit_fAb=%d, top_k_centralities=%d\n\n' % (alpha_const, beta_const, maxit_quadrature, maxit_fAb, top_k_centralities))

		### Print top k centralities ###
		k = top_k_centralities
		
		if compute_quadrature_quantities:
			### Subgraph centrality ###
			SC_sorted = -np.sort(-SC, axis=None)
			SC_sorted_ind = np.argsort(-SC, axis=None)

			# top k JCs
			print('\n-----Subgraph centrality-----\n')
			print_file.write('-----Subgraph centrality-----\n')
			df_values={'SC_JC_ranking': range(1,k+1), 'value': SC_sorted[0:k]}
			df_nodes={'SC_JC_ranking': range(1,k+1), 'node_id': (SC_sorted_ind[0:k] % n)}
			df_layers={'SC_JC_ranking': range(1,k+1), 'layer_id': (SC_sorted_ind[0:k] // n)}
			SC_node_names = pd.merge(pd.DataFrame(data=df_nodes), stopIDList, on='node_id')[['SC_JC_ranking', 'node_id', 'stop_name']].sort_values(by=['SC_JC_ranking'])
			SC_layer_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['SC_JC_ranking', 'layer_id', 'layer_name']].sort_values(by=['SC_JC_ranking'])
			top_k_SC_JC = pd.merge(pd.merge(SC_node_names, SC_layer_names, on='SC_JC_ranking'),  pd.DataFrame(data=df_values), on='SC_JC_ranking')
			print('Joint centralities:\n', top_k_SC_JC.to_string(index=False))
			print_file.write('Joint centralities:\n')
			print_file.write(top_k_SC_JC.to_string(index=False))

			# top k MNCs
			SC_MNC_sorted = -np.sort(-SC_MNC, axis=None)
			SC_MNC_sorted_ind = np.argsort(-SC_MNC, axis=None)
			df_values={'SC_MNC_ranking': range(1,k+1), 'value': SC_MNC_sorted[0:k]}
			df_nodes={'SC_MNC_ranking': range(1,k+1), 'node_id': SC_MNC_sorted_ind[0:k]}
			SC_MNC_node_names = pd.merge(pd.DataFrame(data=df_nodes), stopIDList, on='node_id')[['SC_MNC_ranking', 'node_id', 'stop_name']].sort_values(by=['SC_MNC_ranking'])
			top_k_SC_MNC = pd.merge(SC_MNC_node_names, pd.DataFrame(data=df_values), on='SC_MNC_ranking')
			print('Marginal node centralities:\n', top_k_SC_MNC.to_string(index=False))
			print_file.write('\nMarginal node centralities:\n')
			print_file.write(top_k_SC_MNC.to_string(index=False))

			# top k MLCs
			SC_MLC_sorted = -np.sort(-SC_MLC, axis=None)
			SC_MLC_sorted_ind = np.argsort(-SC_MLC, axis=None)
			df_values={'SC_MLC_ranking': range(1,k+1), 'value': SC_MLC_sorted[0:k]}
			df_layers={'SC_MLC_ranking': range(1,k+1), 'layer_id': SC_MLC_sorted_ind[0:k]}
			SC_MLC_node_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['SC_MLC_ranking', 'layer_id', 'layer_name']].sort_values(by=['SC_MLC_ranking'])
			top_k_SC_MLC = pd.merge(SC_MLC_node_names, pd.DataFrame(data=df_values), on='SC_MLC_ranking')
			print('Marginal layer centralities:\n', top_k_SC_MLC.to_string(index=False))
			print_file.write('\nMarginal layer centralities:\n')
			print_file.write(top_k_SC_MLC.to_string(index=False))


			### Resolvent-based subgraph centrality ###
			SCres_sorted = -np.sort(-SCres, axis=None)
			SCres_sorted_ind = np.argsort(-SCres, axis=None)

			# top k JCs
			print('\n-----Resolvent-based subgraph centrality-----\n')
			print_file.write('\n\n-----Resolvent-based subgraph centrality-----\n')
			df_values={'SCres_JC_ranking': range(1,k+1), 'value': SCres_sorted[0:k]}
			df_nodes={'SCres_JC_ranking': range(1,k+1), 'node_id': (SCres_sorted_ind[0:k] % n)}
			df_layers={'SCres_JC_ranking': range(1,k+1), 'layer_id': (SCres_sorted_ind[0:k] // n)}
			SCres_node_names = pd.merge(pd.DataFrame(data=df_nodes), stopIDList, on='node_id')[['SCres_JC_ranking', 'node_id', 'stop_name']].sort_values(by=['SCres_JC_ranking'])
			SCres_layer_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['SCres_JC_ranking', 'layer_id', 'layer_name']].sort_values(by=['SCres_JC_ranking'])
			top_k_SCres_JC = pd.merge(pd.merge(SCres_node_names, SCres_layer_names, on='SCres_JC_ranking'),  pd.DataFrame(data=df_values), on='SCres_JC_ranking')
			print('Joint centralities:\n', top_k_SCres_JC.to_string(index=False))
			print_file.write('Joint centralities:\n')
			print_file.write(top_k_SCres_JC.to_string(index=False))

			# top k MNCs
			SCres_MNC_sorted = -np.sort(-SCres_MNC, axis=None)
			SCres_MNC_sorted_ind = np.argsort(-SCres_MNC, axis=None)
			df_values={'SCres_MNC_ranking': range(1,k+1), 'value': SCres_MNC_sorted[0:k]}
			df_nodes={'SCres_MNC_ranking': range(1,k+1), 'node_id': SCres_MNC_sorted_ind[0:k]}
			SCres_MNC_node_names = pd.merge(pd.DataFrame(data=df_nodes), stopIDList, on='node_id')[['SCres_MNC_ranking', 'node_id', 'stop_name']].sort_values(by=['SCres_MNC_ranking'])
			top_k_SCres_MNC = pd.merge(SCres_MNC_node_names, pd.DataFrame(data=df_values), on='SCres_MNC_ranking')
			print('Marginal node centralities:\n', top_k_SCres_MNC.to_string(index=False))
			print_file.write('\nMarginal node centralities:\n')
			print_file.write(top_k_SCres_MNC.to_string(index=False))

			# top k MLCs
			SCres_MLC_sorted = -np.sort(-SCres_MLC, axis=None)
			SCres_MLC_sorted_ind = np.argsort(-SCres_MLC, axis=None)
			df_values={'SCres_MLC_ranking': range(1,k+1), 'value': SCres_MLC_sorted[0:k]}
			df_layers={'SCres_MLC_ranking': range(1,k+1), 'layer_id': SCres_MLC_sorted_ind[0:k]}
			SCres_MLC_node_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['SCres_MLC_ranking', 'layer_id', 'layer_name']].sort_values(by=['SCres_MLC_ranking'])
			top_k_SCres_MLC = pd.merge(SCres_MLC_node_names, pd.DataFrame(data=df_values), on='SCres_MLC_ranking')
			print('Marginal layer centralities:\n', top_k_SCres_MLC.to_string(index=False))
			print_file.write('\nMarginal layer centralities:\n')
			print_file.write(top_k_SCres_MLC.to_string(index=False))


		### Total communicability ###
		TC_sorted = -np.sort(-TC, axis=None)
		TC_sorted_ind = np.argsort(-TC, axis=None)

		# top k JCs
		print('\n-----Total communicability-----\n')
		print_file.write('\n\n-----Total communicability-----\n')
		df_values={'TC_JC_ranking': range(1,k+1), 'value': TC_sorted[0:k]}
		df_nodes={'TC_JC_ranking': range(1,k+1), 'node_id': (TC_sorted_ind[0:k] % n)}
		df_layers={'TC_JC_ranking': range(1,k+1), 'layer_id': (TC_sorted_ind[0:k] // n)}
		TC_node_names = pd.merge(pd.DataFrame(data=df_nodes), stopIDList, on='node_id')[['TC_JC_ranking', 'node_id', 'stop_name']].sort_values(by=['TC_JC_ranking'])
		TC_layer_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['TC_JC_ranking', 'layer_id', 'layer_name']].sort_values(by=['TC_JC_ranking'])
		top_k_TC_JC = pd.merge(pd.merge(TC_node_names, TC_layer_names, on='TC_JC_ranking'),  pd.DataFrame(data=df_values), on='TC_JC_ranking')
		print('Joint communicabilities:\n', top_k_TC_JC.to_string(index=False))
		print_file.write('Joint communicabilities:\n')
		print_file.write(top_k_TC_JC.to_string(index=False))

		# top k MNCs
		TC_MNC_sorted = -np.sort(-TC_MNC, axis=None)
		TC_MNC_sorted_ind = np.argsort(-TC_MNC, axis=None)
		df_values={'TC_MNC_ranking': range(1,k+1), 'value': TC_MNC_sorted[0:k]}
		df_nodes={'TC_MNC_ranking': range(1,k+1), 'node_id': TC_MNC_sorted_ind[0:k]}
		TC_MNC_node_names = pd.merge(pd.DataFrame(data=df_nodes), stopIDList, on='node_id')[['TC_MNC_ranking', 'node_id', 'stop_name']].sort_values(by=['TC_MNC_ranking'])
		top_k_TC_MNC = pd.merge(TC_MNC_node_names, pd.DataFrame(data=df_values), on='TC_MNC_ranking')
		print('Marginal node communicabilities:\n', top_k_TC_MNC.to_string(index=False))
		print_file.write('\nMarginal node communicabilities:\n')
		print_file.write(top_k_TC_MNC.to_string(index=False))

		# top k MLCs
		TC_MLC_sorted = -np.sort(-TC_MLC, axis=None)
		TC_MLC_sorted_ind = np.argsort(-TC_MLC, axis=None)
		df_values={'TC_MLC_ranking': range(1,k+1), 'value': TC_MLC_sorted[0:k]}
		df_layers={'TC_MLC_ranking': range(1,k+1), 'layer_id': TC_MLC_sorted_ind[0:k]}
		TC_MLC_node_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['TC_MLC_ranking', 'layer_id', 'layer_name']].sort_values(by=['TC_MLC_ranking'])
		top_k_TC_MLC = pd.merge(TC_MLC_node_names, pd.DataFrame(data=df_values), on='TC_MLC_ranking')
		print('Marginal layer communicabilities:\n', top_k_TC_MLC.to_string(index=False))
		print_file.write('\nMarginal layer communicabilities:\n')
		print_file.write(top_k_TC_MLC.to_string(index=False))


		### Katz centrality ###
		KC_sorted = -np.sort(-KC, axis=None)
		KC_sorted_ind = np.argsort(-KC, axis=None)

		# top k JCs
		print('\n-----Katz centrality-----\n')
		print_file.write('\n\n-----Katz centrality-----\n')
		df_values={'KC_JC_ranking': range(1,k+1), 'value': KC_sorted[0:k]}
		df_nodes={'KC_JC_ranking': range(1,k+1), 'node_id': (KC_sorted_ind[0:k] % n)}
		df_layers={'KC_JC_ranking': range(1,k+1), 'layer_id': (KC_sorted_ind[0:k] // n)}
		KC_node_names = pd.merge(pd.DataFrame(data=df_nodes), stopIDList, on='node_id')[['KC_JC_ranking', 'node_id', 'stop_name']].sort_values(by=['KC_JC_ranking'])
		KC_layer_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['KC_JC_ranking', 'layer_id', 'layer_name']].sort_values(by=['KC_JC_ranking'])
		top_k_KC_JC = pd.merge(pd.merge(KC_node_names, KC_layer_names, on='KC_JC_ranking'),  pd.DataFrame(data=df_values), on='KC_JC_ranking')
		print('Joint centralities:\n', top_k_KC_JC.to_string(index=False))
		print_file.write('Joint centralities:\n')
		print_file.write(top_k_KC_JC.to_string(index=False))

		# top k MNCs
		KC_MNC_sorted = -np.sort(-KC_MNC, axis=None)
		KC_MNC_sorted_ind = np.argsort(-KC_MNC, axis=None)
		df_values={'KC_MNC_ranking': range(1,k+1), 'value': KC_MNC_sorted[0:k]}
		df_nodes={'KC_MNC_ranking': range(1,k+1), 'node_id': KC_MNC_sorted_ind[0:k]}
		KC_MNC_node_names = pd.merge(pd.DataFrame(data=df_nodes), stopIDList, on='node_id')[['KC_MNC_ranking', 'node_id', 'stop_name']].sort_values(by=['KC_MNC_ranking'])
		top_k_KC_MNC = pd.merge(KC_MNC_node_names, pd.DataFrame(data=df_values), on='KC_MNC_ranking')
		print('Marginal node centralities:\n', top_k_KC_MNC.to_string(index=False))
		print_file.write('\nMarginal node centralities:\n')
		print_file.write(top_k_KC_MNC.to_string(index=False))

		# top k MLCs
		KC_MLC_sorted = -np.sort(-KC_MLC, axis=None)
		KC_MLC_sorted_ind = np.argsort(-KC_MLC, axis=None)
		df_values={'KC_MLC_ranking': range(1,k+1), 'value': KC_MLC_sorted[0:k]}
		df_layers={'KC_MLC_ranking': range(1,k+1), 'layer_id': KC_MLC_sorted_ind[0:k]}
		KC_MLC_node_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['KC_MLC_ranking', 'layer_id', 'layer_name']].sort_values(by=['KC_MLC_ranking'])
		top_k_KC_MLC = pd.merge(KC_MLC_node_names, pd.DataFrame(data=df_values), on='KC_MLC_ranking')
		print('Marginal layer centralities:\n', top_k_KC_MLC.to_string(index=False))
		print_file.write('\nMarginal layer centralities:\n')
		print_file.write(top_k_KC_MLC.to_string(index=False))

		print_file.close()



	if return_layer_centralities:
		if compute_quadrature_quantities:
			return SC_MLC, SCres_MLC, TC_MLC, KC_MLC
		else:
			return TC_MLC, KC_MLC
	else:
		if compute_quadrature_quantities:
			return SC_MNC, SCres_MNC, TC_MNC, KC_MNC
		else:
			return TC_MNC, KC_MNC


