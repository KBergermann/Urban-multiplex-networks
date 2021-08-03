import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import sys


def OSMnx_street_orientations(example='Germany'):

	#####################################################################################################
	#													#
	# DESCRIPTION:	uses the jupyter notebook example [1,2,3] to produce street network orientation	#
	#		plots from [2] for two sets of cities. Available options are 'Germany' and		#
	#		'Europe', but the code can be adapted to user-specific queries via amending the	#
	#		dictionary 'places'.									#
	#													#
	# INPUT: 	example (optional, default='Germany'): specifies the example to reproduce from [4].	#
	#			Available options are 'Germany' and 'Europe'.					#
	#													#
	# OUTPUT: 	None											#
	#													#
	# FILES												#
	# CREATED:	'orientationPlots/german-street-orientations.pdf', or				#
	#		'orientationPlots/european-street-orientations.pdf'					#
	#													#
	# REFERENCES:	[1] https://github.com/gboeing/osmnx-examples/blob/main/notebooks/17-street-network-orientations.ipynb												#
	#		[2] G. Boeing, OSMnx: New methods for acquiring, constructing, analyzing, and	#
	#		visualizing complex street networks, Computers, Environment and Urban Systems,	#
	#		65 (2017), pp. 126-139, https://doi.org/10.1016/j.compenvurbsys.2017.05.004.		#
	#		[3] https://osmnx.readthedocs.io/en/stable/osmnx.html				#
	#		[4] K. Bergermann and M. Stoll, Orientations and matrix function-based		#
	#		centralities in multiplex network analysis of urban public transport, arXiv		#
	#		preprint, arXiv:2107.12695, (2021).							#
	#													#
	# 2021, Kai Bergermann (kai.bergermann@math.tu-chemnitz.de)						#
	#													#
	#####################################################################################################


	ox.config(log_console=True)
	weight_by_length = False

	ox.__version__

	# define the study sites as label : query

	if example=='Germany':

		places = {
			'Aachen': 'Aachen, Germany',
			'Augsburg': 'Augsburg, Germany',
			'Berlin': 'Berlin, Germany',
			'Bielefeld': 'Bielefeld, Germany',
			'Bonn': 'Bonn, Germany',
			'Braunschweig': 'Braunschweig, Germany',
			'Bremen': 'Bremen, Germany',
			'Chemnitz': 'Chemnitz, Germany',
			'Cologne': 'Köln, Germany',
			'Dortmund': 'Dortmund, Germany',
			'Dresden': 'Dresden, Germany',
			'Düsseldorf': 'Düsseldorf, Germany',
			'Duisburg': 'Duisburg, Germany',
			'Erfurt': 'Erfurt, Germany',
			'Essen': 'Essen, Germany',
			'Frankfurt am Main': 'Frankfurt am Main, Germany',
			'Freiburg': 'Freiburg, Germany',
			'Halle (Saale)': 'Halle (Saale), Germany',
			'Hamburg': 'Hamburg, Germany',
			'Hanover': 'Hannover, Germany',
			'Karlsruhe': 'Karlsruhe, Germany',
			'Kiel': 'Kiel, Germany',
			'Krefeld': 'Krefeld, Germany',
			'Leipzig': 'Leipzig, Germany',
			'Lübeck': 'Lübeck, Germany',
			'Mainz': 'Mainz, Germany',
			'Mannheim': 'Mannheim, Germany',
			'Mönchengladbach': 'Mönchengladbach, Germany',
			'Münster': 'Münster, Germany',
			'Munich': 'München, Germany',
			'Nuremberg': 'Nürnberg, Germany',
			'Oberhausen': 'Oberhausen, Germany',
			'Rostock': 'Rostock, Germany',
			'Stuttgart': 'Stuttgart, Germany',
			'Wiesbaden': 'Wiesbaden, Germany',
			'Wuppertal': 'Wuppertal, Germany',
		}
		
	elif example=='Europe':

		places = {
			'Athens': 'Athens, Municipality of Athens',
			'Barcelona': 'Barcelona, Spain',
			'Belgrade': 'Belgrade, Serbia',
			'Brussels': 'Brussel, Belgium',
			'Budapest': 'Budapest, Hungary',
			'Helsinki': 'Helsinki, Finland',
			'Luxembourg': 'Luxembourg City, Luxembourg',
			'Madrid': 'Madrid, Spain',
			'Manchester': 'Manchester, United Kingdom',
			'Nice': 'Nice, France',
			'Oslo': 'Oslo, Norway',
			'Prague': 'Prague, Czech Republic',
			'Rome': 'Rome, Italy',
			'Stockholm': 'Stockholm, Sweden',
			'Tallinn': 'Tallinn, Estonia',
			'Vienna': 'Vienna, Austria',
			'Vilnius': 'Vilnius, Lithuania',
			'Zagreb': 'Zagreb, Croatia',
		}
		
	else: 
		print('ERROR, example unknown. Choose between \'Germany\' and \'Europe\'!')
		sys.exit(0)
	

	# verify OSMnx geocodes each query to what you expect (i.e., a [multi]polygon geometry)
	gdf = ox.geocode_to_gdf(list(places.values()))
	gdf

	# create figure and axes
	n = len(places)
	
	if example=='Germany':
		ncols = int(np.ceil(np.sqrt(n)))
		nrows = int(np.ceil(n / ncols))
	elif example=='Europe':
		ncols = 6
		nrows = 3

	figsize = (ncols * 5, nrows * 5)
	fig, axes = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw={"projection": "polar"})

	# plot each city's polar histogram
	for ax, place in zip(axes.flat, sorted(places.keys())):
		print(ox.utils.ts(), place)
		
		# get undirected graphs with edge bearing attributes
		G = ox.graph_from_place(place, network_type="drive")
		Gu = ox.add_edge_bearings(ox.get_undirected(G))
		fig, ax = ox.bearing.plot_orientation(Gu, ax=ax, title=place, area=False, title_font={"family": "sans-serif", "fontsize": 30}, xtick_font={"family": "sans-serif", "fontsize": 15})

		# add figure title and save image
	suptitle_font = {
		"family": "sans-serif",
		"fontsize": 60,
		"fontweight": "normal",
		"y": 1,
	}
	
	fig.tight_layout()
	fig.subplots_adjust(hspace=0.35)
	
	if example=='Germany':
		fig.savefig("orientationPlots/german-street-orientations.pdf", facecolor="w", dpi=100, bbox_inches="tight")
	elif example=='Europe':
		fig.savefig("orientationPlots/european-street-orientations.pdf", facecolor="w", dpi=100, bbox_inches="tight")
		
	plt.close()

