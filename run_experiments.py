from subroutines.OSMnx_shapefiles import *
from subroutines.OSMnx_street_orientations import *
from subroutines.create_aggregated_bearing_plot import *
from subroutines.build_supra_adjacency_matrix import *
from subroutines.centralities.compute_MNC_plots import *
from subroutines.centralities.compute_MLC_plots import *
from subroutines.centralities.compute_MCs_varying_omega import *

#####################################################################################################
#													#
# This script reproduces the results and graphics from the paper					#
#													#
# [1] Orientations and matrix function-based centralities in multiplex network analysis of urban	#
# public transport, K. Bergermann and M. Stoll, https://arxiv.org/abs/2107.12695			#
#													#
# The following python packages are required for the successful execution of all functions:		#
# os, sys, time, datetime, numpy, scipy, pandas, geopandas, matplotlib, osmnx			#
#													#
# The functions 'create_aggregated_bearing_plot' and 'build_supra_adjacency_matrix' require the	#
# GTFS data set of Germany's local public transport, which is expected to be located in the		#
# directory 'gtfsdata'. You can download the data set here:						#
# https://www.tu-chemnitz.de/mathematik/wire/pubs/gtfsdata.tar.gz (170MB).				#
# 													#
# Note that each orientation plot of a European city requires a separate GTFS data set, which we	#
# do not provide. The figures in the paper were created with GTFS data sets from			#
# https://transitfeeds.com/l/60-europe (we downloaded what appeared to be the latest complete data	#
# set available as of May 21st, 2021). For more information, please write an e-mail to		#
# kai.bergermann@math.tu-chemnitz.de									#
#													#
# For more information about the functions please refer to the description in the respective file.	#
#													#
# 2021, Kai Bergermann (kai.bergermann@math.tu-chemnitz.de)						#
#													#
#####################################################################################################



########################
##### ORIENTATIONS #####
########################


### shapefiles ###

OSMnx_shapefiles(searchString='Barcelona, Spain', saveString='Barcelona')
OSMnx_shapefiles(searchString='Chemnitz, Germany', saveString='Chemnitz')
OSMnx_shapefiles(searchString='Cologne, Germany', saveString='Cologne')
OSMnx_shapefiles(searchString='Düsseldorf, Germany', saveString='Duesseldorf')
OSMnx_shapefiles(searchString='Freiburg, Germany', saveString='Freiburg')


### street network orientation plots ###
# [1, Figures 1 and 3]
OSMnx_street_orientations(example='Germany')
OSMnx_street_orientations(example='Europe')


### public transport network orientation plots ###
# Selected examples from [1, Figure 2]
create_aggregated_bearing_plot(cityString='Munich')
create_aggregated_bearing_plot(cityString='Duisburg')
create_aggregated_bearing_plot(cityString='Halle (Saale)')
create_aggregated_bearing_plot(cityString='Freiburg')



##############################################
##### Matrix function-based centralities #####
##############################################


### build adjacency matrices ###
# adjacency matrices required to reproduce the results from [1, Figures 6-11]
build_supra_adjacency_matrix(cityString='Halle (Saale)', weighted=True, compute_inter=True, compute_frequencies=True)
build_supra_adjacency_matrix(cityString='Cologne', weighted=True, compute_inter=True, compute_frequencies=True)
build_supra_adjacency_matrix(cityString='Stuttgart', weighted=True, compute_inter=True, compute_frequencies=True)
build_supra_adjacency_matrix(cityString='Stuttgart', weighted=False, compute_inter=False, compute_frequencies=False)
build_supra_adjacency_matrix(cityString='Düsseldorf', weighted=True, compute_inter=True, compute_frequencies=True)
build_supra_adjacency_matrix(cityString='Chemnitz', weighted=True, compute_inter=True, compute_frequencies=True)


### reproduce centrality plots ###
# [1, Figure 6]
compute_MLC_plots(cityString='Halle (Saale)', omega=np.exp(-5**2/5**2), sigma=5, markersize=3)
compute_MNC_plots(cityString='Halle (Saale)', omega=np.exp(-5**2/5**2), sigma=5, scatter_markersize=40)

# [1, Figure 7]
compute_MNC_plots(cityString='Cologne', omega=np.exp(-5**2/5**2), sigma=5, scatter_markersize=20)

# [1, Figure 8]
compute_MNC_plots(cityString='Stuttgart', weighted_with_travel_times=False, weighted_with_frequencies=False, omega=np.exp(-5**2/5**2), sigma=5, scatter_markersize=40)
compute_MNC_plots(cityString='Stuttgart', weighted_with_travel_times=False, weighted_with_frequencies=True, omega=np.exp(-5**2/5**2), sigma=5, scatter_markersize=40)
compute_MNC_plots(cityString='Stuttgart', weighted_with_travel_times=True, weighted_with_frequencies=False, omega=np.exp(-5**2/5**2), sigma=5, scatter_markersize=40)
compute_MNC_plots(cityString='Stuttgart', weighted_with_travel_times=True, weighted_with_frequencies=True, omega=np.exp(-5**2/5**2), sigma=5, scatter_markersize=40)

# [1, Figure 9]
compute_MNC_plots(cityString='Düsseldorf', compute_quadrature_quantities=False, omega=np.exp(-15**2/5**2), alpha_const=.01, beta_const=0.01, sigma=5, scatter_markersize=15)
compute_MNC_plots(cityString='Düsseldorf', compute_quadrature_quantities=False, omega=np.exp(-15**2/5**2), alpha_const=.75, beta_const=2.5, sigma=5, scatter_markersize=15)
compute_MNC_plots(cityString='Düsseldorf', compute_quadrature_quantities=False, omega=np.exp(-15**2/5**2), alpha_const=.99, beta_const=10, sigma=5, scatter_markersize=15)

# [1, Figure 10]
compute_MNC_plots(cityString='Chemnitz', compute_quadrature_quantities=False, omega=np.exp(-0**2/1**2), sigma=1, scatter_markersize=30)
compute_MNC_plots(cityString='Chemnitz', compute_quadrature_quantities=False, omega=np.exp(-5**2/1**2), sigma=1, scatter_markersize=30)
compute_MNC_plots(cityString='Chemnitz', compute_quadrature_quantities=False, omega=np.exp(-15**2/1**2), sigma=1, scatter_markersize=30)

# [1, Figure 11]
compute_MCs_varying_omega(cityString='Chemnitz', weighted_with_frequencies=False, sigma=1, suppress_output=True)


