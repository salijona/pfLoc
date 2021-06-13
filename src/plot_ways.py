# -*- indent-tabs-mode: 1; python-indent-offset: 4; tab-width: 4 -*-

from osmapi import OsmApi
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

tartu1BoundingRect = [26.7154817,58.379593,26.7213126,58.3836303]
tartu2BoundingRect = [26.7104817,58.374593,26.7263126,58.3886303]

if __name__=='__main__':
	MyApi = OsmApi()
	bounds = tartu1BoundingRect
	elems = MyApi.Map(*bounds)
	nodeElems = filter(lambda e: e['type'] == 'node', elems)
	nodes = dict(map(lambda n: (n['data']['id'], n['data']), nodeElems))
	wayElems = filter(lambda e: e['type'] == 'way',elems)
	ways = map(lambda e: e['data'], wayElems)
	highways = filter(lambda w: w['tag'].get('highway',''), ways)
	buildings = filter(lambda w: w['tag'].get('building',''), ways)

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1, aspect=1/np.cos(bounds[1]/180*np.pi))
	for way in highways:
		wayType = way['tag']['highway']
		if wayType in ['motorway', 'trunk', 'primary', 'secondary', 'tertiary',
					   'unclassified', 'residential']:
			color = 'black'
		elif wayType in ['pedestrian','footway']:
			color = 'lightgreen'
		elif wayType in ['steps']:
			color = 'red'
		else:
			color = 'gray'
		wayNodes = map(lambda nid: nodes.get(nid, None), way['nd'])
		coords = np.array(map(lambda nd: (nd['lon'], nd['lat']), wayNodes))
		ax.add_line(Line2D(coords[:,0], coords[:,1], color=color))

	for way in buildings:
		wayNodes = map(lambda nid: nodes.get(nid, None), way['nd'])
		coords = np.array(map(lambda nd: (nd['lon'], nd['lat']), wayNodes))
		ax.add_line(Line2D(coords[:,0], coords[:,1], color='lightblue'))

	#ax.plot(bounds[0::2], bounds[1::2])
	ax.set_xlim(bounds[0::2])
	ax.set_ylim(bounds[1::2])
	plt.show()
