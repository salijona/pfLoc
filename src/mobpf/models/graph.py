# -*- indent-tabs-mode: 1; python-indent-offset: 4; tab-width: 4 -*-
import logging
import traceback

from math import radians, cos, sin, sqrt, atan2
import datetime

import numpy as np
from scipy.stats import norm
from shapely.geometry import Polygon, Point, LineString, MultiPoint
from shapely.ops import nearest_points
from haversine import haversine, Unit
import shapely.wkt
from mobpf.base import State,Model

from const import Vars
from curses.ascii import alt

from os import sep
from os.path import exists

from tempfile import gettempdir

from pickle import dump, load
import dill
from functools import reduce
import networkx as nx
import math
from sys import maxsize

import time 
import multiprocessing as mp
#import mpi4py.MPI
from multiprocessing import Process
import queue
import multiprocessing
from operator import itemgetter

import warnings
warnings.filterwarnings('ignore')

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename="debug.log",level=logging.DEBUG)

LOG = logging.getLogger('graph')

class Graph(object):
	def __init__(self, Nv, Ne):
		self.Nv = Nv
		self.Ne = Ne
		self.verts = np.empty(Nv, dtype=[('x',float),('y',float)])
		self.edges = np.empty((Ne,2), dtype=int)
		self.speeds = np.empty(shape=(Ne,1), dtype=int)


	def getEdges(self, polygon):

		xs_start, ys_start = self.verts['x'][self.edges[:,0]], self.verts['y'][self.edges[:,0]]
		xs_end, ys_end = self.verts['x'][self.edges[:,1]], self.verts['y'][self.edges[:,1]]

		start_points = list(map(Point, zip(xs_start, ys_start)))
		end_points = list(map(Point, zip(xs_end, ys_end)))

		inds=[]

		## Find a more efficient way, removing these for loops

		for i in range(len(start_points)): 
			if (polygon.iloc[0].contains(start_points[i]) and polygon.iloc[0].contains(end_points[i])):
				inds.append(i)

		if len(inds)==0:
			polygon_centroid = polygon.values[0].centroid
			multipoints = MultiPoint(start_points)
			nearest_geoms = nearest_points(polygon_centroid, multipoints)	

			try: 
				inds.append(start_points.index(nearest_geoms[1]))
			except ValueError:
				pass
			
			try:
				inds.append(end_points.index(nearest_geoms[1]))
			except ValueError:
				pass


		inds = np.unique(np.asarray(inds))
		#print("Number of edges", len(inds))

		return inds


class OverpyGraph(Graph):

	def __init__(self, result, cache=False):
		self.Graph = Graph
		"@param result: Overpy result"
		nodes = result.get_nodes()
		ways = result.get_ways()
		speeds_list = []

		for way in result.ways:
			speed = way.tags.get("maxspeed", "n/a")
			if speed == "n/a":
				speeds_list.append(60)
			else:
				try:
					speeds_list.append(int(speed))
				except ValueError:
					speeds_list.append(60)
	
		N = len(nodes)
		## One way might have multiple nodes. Find total number of nodes on the way
		M = reduce(lambda s,way: s+len(way.nodes)-1, ways,0)

		self.Graph.__init__(self, N, M)

		# Initialize directed
		#	parse attributes of ways

		xs = self.verts['x']
		ys = self.verts['y']
		self.xs = self.verts['x']
		self.ys = self.verts['y']

		idsMap = {}
		for iNode in range(N):
			idsMap[nodes[iNode].id] = iNode
			xs[iNode] = nodes[iNode].lon
			ys[iNode] = nodes[iNode].lat
		print("Step 1")
		iEdge = 0
		ispeed = 0

		for way in ways:
			for i in range(len(way.nodes)-1):
				fm = idsMap[way.nodes[i].id]
				to = idsMap[way.nodes[i+1].id]
				self.edges[iEdge,:] = (fm,to)
				self.speeds[iEdge] = speeds_list[ispeed] 
				iEdge += 1
			ispeed = ispeed + 1

		LOG.debug( '%s %s', str(iEdge), str(M) )


		if cache:
			self.edge_distance_cache = {}
			self.edge_distance_cache_file = \
				sep.join( [ gettempdir(), Vars.EDGE_DISTANCE_CACHE ] )
			self.__load_cache()
		else:
			self.edge_distance_cache = None


	def __calculate_edge_distances(self):
		'''Compute distances for all edges'''
		if self.edge_distance_cache == None:
			return
		N = len(self.edges)
		i = 0
		tenP = int(N*0.10)
		tenP_frames = list(map(lambda x: tenP*x,range(1,10)))
		LOG.info('Calculating the distances of all edges ...')
		for e in self.edges:
			if i in tenP_frames:
				LOG.info('Calculating the distances ... %d %% finished ', (100.0*i)/N)
			d = self.get_distance_edge(e, True)
			self.edge_distance_cache[(e[0],e[1])] = d
			LOG.debug('Distance %d->%d %m', e[0], e[1], d)
			i+=1
		LOG.info('Calculating the distances completed')
		self.__sync_cache()

	def __sync_cache(self):
		f = open(self.edge_distance_cache_file,'wb')
		dump(self.edge_distance_cache, f)
		f.flush()
		f.close()

	def __load_cache(self):
		if not exists(self.edge_distance_cache_file):
			f = open(self.edge_distance_cache_file,'wb')
			f.flush()
			f.close()
			return
		f = open(self.edge_distance_cache_file,'r')
		self.edge_distance_cache = load(f)
		f.close()

	def get_neighbours_edges(self, source, directed=False):
		# filter all edges where v is a starting points
		dsts = filter(lambda x: x[0] == source, self.edges)
		# in case of non-directed graph we also do opposite
		if not directed:
			dsts2 = filter(lambda x: x[1] == source, self.edges)
			return dsts+dsts2
		return dsts

	def get_neighbours(self, source, directed=False):
		# filter all edges where v is a starting points, take the ending node
		dsts = list(map(lambda y: y[1],filter(lambda x: x[0] == source, self.edges)))
		# in case of non-directed graph we also do opposite
		if not directed:
			dsts2 = list(map(lambda y: y[0],filter(lambda x: x[1] == source, self.edges)))
			return dsts+dsts2
		return dsts

	def get_distance_edge(self,e,meters=True):
		if not self.edge_distance_cache == None and \
				self.edge_distance_cache.has_key((e[0],e[1])):
			return self.edge_distance_cache[(e[0],e[1])]
		n1,n2 = self.verts[e]

		lon_v1 = self.xs[n1]
		lat_v1 = self.ys[n1]
		lon_v2 = self.xs[n2]
		lat_v2 = self.ys[n2]
		d= haversine((lat_v1,lon_v1), (lat_v2,lon_v2), unit=Unit.METERS)
		#d = OverpyGraph.get_distance(n1,n2,meters)
		if not self.edge_distance_cache == None and \
				not self.edge_distance_cache.has_key((e[0],e[1])):
			self.edge_distance_cache[(e[0],e[1])] = d
		return d
	
	def get_distance_vertex(self,v1,v2,meters=True):
		#n1 = self.verts[v1]
		#n2 = self.verts[v2]
		if not self.edge_distance_cache == None and \
				self.edge_distance_cache.has_key((v1,v2)):
			return self.edge_distance_cache[(v1,v2)]
		lon_v1 = self.xs[v1]
		lat_v1 = self.ys[v1]
		lon_v2 = self.xs[v2]
		lat_v2 = self.ys[v2]
		d= haversine((lat_v1,lon_v1), (lat_v2,lon_v2), unit=Unit.METERS)

		#d = OverpyGraph.get_distance(n1, n2, meters)
		if not self.edge_distance_cache == None and \
			not self.edge_distance_cache.has_key((v1,v2)):
			self.edge_distance_cache[(v1,v2)] = d
		return d


	@classmethod
	def is_point_close_to_line(cls,v1,v2,p):
		'''How close is point p to a line v1,v2 using PDP
			returns non-zero value, the smaller - the closer'''

		v1x,v1y = v1
		v2x,v2y = v2
		px,py = p

		# Perp-Dot-product
		pdp = (v1x-px)*(v2y-py) - (v1y-py)*(v2x-px)

		return abs(pdp)

	def get_edge_gps(self,gps, r=10):
		'''Return closest edges to GPS cord.'''
		cords = self.get_vertices_gps(gps, r)
		xs = list(map(lambda x: x[0], cords))
		ys = list(map(lambda x: x[1], cords))

		V = [(i,self.verts[i][0],self.verts[i][1]) \
				for i in range(len(self.verts))]

		
		V = list(filter(lambda x: x[1] in xs, V))
		V = list(filter(lambda x: x[2] in ys, V))
		Vidx = list(map(lambda x: x[0],V))
		E = list(filter(lambda x: x[0] in Vidx or x[1] in Vidx, self.edges))

		return E

	def get_edge_gps_pdp(self, gps, r=10):
		'''Return the closest edge using PDP'''
		E = self.get_edge_gps(gps, r)

		Egps = list(map(lambda x: ((x),)+(self.verts[x][0], self.verts[x][1]),E))
		dEgps = list(map(lambda x: \
				(x,)+(OverpyGraph.is_point_close_to_line(x[1],x[2],gps),),Egps))

		edge = min(dEgps,key=lambda x: x[-1])

		return edge[0][0]


	def get_vertices_gps(self, gps, r=10):

		'''Return closest vertices to GPS cord, having r in meters'''

		pairs = list(map(lambda x: ((gps,x),haversine((gps[1],gps[0]), (x[1],x[0]), unit=Unit.METERS) ),self.verts))
		vs = list(map(lambda x: x[0][1], filter(lambda x: x[1] <= r, pairs)))
		LOG.debug('Closest vertices to %d,%d in radius %d are: %s',gps[0],gps[1], r, str(vs))

		return vs



class OverpyNXGraph(OverpyGraph):
	'''Implements Graph using:
		OverPass Python bindings for getting:
			GPS coordinates of vertices
			Metainformation of the edges
	'''
	def __init__(self, result):

		OverpyGraph.__init__(self,result)

		n = self.Nv
		self.t = nx.empty_graph(n)

		j=0

		for e in self.edges:
			i1,i2 = e
			w1 = self.get_distance_vertex(i1,i2, True)/1000
			weight = (w1/self.speeds[j])[0]

			e1 = self.t.add_edge(i1,i2, weight=weight)
			j = j + 1


class GraphState(State):
	lon = property(lambda self: self._getCoords()[0])
	lat = property(lambda self: self._getCoords()[1])
	__transient__ = set(["_coords"])
	## For some reason this piece of code is executing twice. SD
	def __init__(self, Np, model):
		self.N = Np
		self.model = model
		self.particles = np.empty(Np, dtype=[('edge',int), ('loc',float)])
		self.behaviors = np.empty(Np, dtype=np.int8)
		# initialize particles and behaviors
		Ne = self.model.graph.Ne
		np.random.seed(0)
		self.particles['edge'] = np.random.randint(0,Ne)
		self.particles['loc'] = 0 # np.random.random(Np)
		self.behaviors[:] = 0
		self.timestamp = datetime.datetime.strptime("1975-06-29 08:15:27.243860", '%Y-%m-%d %H:%M:%S.%f')
		self.radius = 0
		self.polygon = shapely.wkt.loads("POLYGON ((0.0 1.0, 0.0 1.0, 0.0 1.0, 0.0 1.0))")

	def _getCoords(self):
		if not hasattr(self, '_coords') or self._coords==None:
			iEdges = self.particles['edge']
			locs = self.particles['loc']
			graph = self.model.graph
			edges = graph.edges[iEdges]
			edgeEnds = graph.verts[edges]

			locs1 = 1-locs
			xs = locs*edgeEnds[:,0]['x'] + locs1*edgeEnds[:,1]['x']
			ys = locs*edgeEnds[:,0]['y'] + locs1*edgeEnds[:,1]['y']

			self._coords = (xs,ys)


		return self._coords


class GraphModelEq(Model):

	def __init__(self, graph, Np):
	
		self.paths_rounds = []
		self.graph = graph
		self.initState = GraphState(Np, self)

	def proposal(self, dt, evidence, borders, state):
		polygon = borders[borders["CGI"]==evidence["CGI"]]["WKT"]
		newState = state.copy()
		newState.timestamp = evidence["TIMESTAMP"]
		newState.radius = borders[borders["CGI"]==evidence["CGI"]]["radius"].values[0]
		newState.polygon = polygon.values[0]
		Np = len(newState.particles)
		propEdges = self.graph.getEdges(polygon)

		randPropEdges = np.random.randint(0,len(propEdges),size=Np)
		newState.particles['edge'] = propEdges[randPropEdges]
		return newState



	def find_length(self, vi, x,  vertices, array_ix, chunk_size, q):
		idx = array_ix * chunk_size
		for y in  vertices: 
			try:
				length = nx.dijkstra_path_length(vi, source= x, target=y, weight="weight")
				#path = nx.shortest_path(vi, source = x, target = y  )
			except Exception as e:
				print(e)
				length = maxsize
			
			q.put((idx,length))
			idx = idx + 1
		#return length

	
	def shortest_length(self , E_start_Vx, E_end_Vx):
		processes = 5
		TD = np.empty(shape=(len(E_start_Vx),len(E_end_Vx)))
		chunk_size = math.ceil(len(E_end_Vx)/processes)
		end_Vx_splitted = self.split_vertices(E_end_Vx, chunk_size)

		for x in range(len(E_start_Vx)):
			q = multiprocessing.Queue() 
			self.multiprocess_paths(end_Vx_splitted, E_start_Vx[x], chunk_size, q)
			result = self.dump_queue(q)
			sorted_result = sorted(result,key=itemgetter(0))
			TD[x,:]= [x[1] for x in sorted_result]
		return TD



	def split_vertices(self, inlist, chunksize):
			return [inlist[x:x+chunksize] for x in range(0, len(inlist), chunksize)]


	def pdp (self, vertices_splitted, r, q):
		for i in vertices_splitted:
			ed = self.graph.get_edge_gps_pdp(i,r)
			q.put(ed)



	def dump_queue(self, q):
		mycopy = []
		while True:
			try:
				elem = q.get(True, 0.01)
			except queue.Empty:
				break
			else:
				mycopy.append(elem)
		return mycopy

	def multiprocess(self, vertices_splitted, r, q):
		jobs = []
		for x in vertices_splitted:
			p = Process(target=self.pdp, args=(x,r,q))
			jobs.append(p)
			p.Daemon = True
			p.start()

		for p in jobs:
			p.join()

	def multiprocess_paths(self, vertices_splitted, x, chunk_size, q):
		jobs = []
		array_ix = 0
		for vertices in vertices_splitted:
			p = Process(target=self.find_length, args=(self.graph.t, x, vertices, array_ix, chunk_size, q ))
			jobs.append(p)
			p.Daemon = True
			p.start()
			array_ix = array_ix + 1

		for p in jobs:
			p.join()

	def transition(self, dt, borders, state, newState):
		'''Uses Networkx shortest path to compute the transitions
			between the old and new states'''
		# Just compute the shortest distances between the old-state points
		# and the new-state points
		polygon_centroid_start = list(state.polygon.centroid.coords)
		polygon_centroid_end = list(newState.polygon.centroid.coords)
		r1 = state.radius/1000
		r2 = newState.radius/1000
		TT = (newState.timestamp - state.timestamp).total_seconds()/3600
		cent_dis = haversine(polygon_centroid_start[0], polygon_centroid_end[0], unit=Unit.METERS)/1000
		tot_dis = cent_dis + r1 + r2
		cc_time = tot_dis/60 

		if (newState.polygon != state.polygon):
			if (TT>cc_time):
				TT = np.asarray(cent_dis/60)
			else:
				TT = np.asarray(TT)
		else:
			TT = np.asarray(TT)

		#TTs = (newState.timestamp - state.timestamp).total_seconds()
		#TT = np.asarray(TTs/3600) #hours
		#print(TT)
		r = 30
		E_old_state = state.particles['edge']
		E_new_state = newState.particles['edge']
		E_start_Vx = self.graph.edges[E_old_state,:][:,1]
		E_end_Vx = self.graph.edges[E_new_state,:][:,1]

	
		TD = self.shortest_length(E_start_Vx, E_end_Vx )

		probs_all = abs((TT - TD))/( TT)
		probs_all = 1 - probs_all
		fm,to = np.unravel_index( np.argmax(probs_all, axis=None), probs_all.shape)
		path = nx.dijkstra_path(self.graph.t, source=E_start_Vx[fm], target=E_end_Vx[to], weight="weight")
		TD_new = np.empty((len(E_start_Vx),len(E_end_Vx)))

		for i in range (len(E_start_Vx)):
			for j in range (len(E_start_Vx)):
				path = nx.dijkstra_path(self.graph.t, source=E_start_Vx[i], target=E_start_Vx[j], weight="weight")
				sum_weights = 0
				for a in range(0, len(path)-1):
					edge = np.where(self.graph.edges==(path[a],path[a+1]))[0][0]
					w1 = self.graph.get_distance_vertex(path[a],path[a+1], True)/1000
					weight = (w1/self.graph.speeds[edge])[0]
					sum_weights += weight

		self.paths_rounds.append(path)
		probs = np.amin(probs_all, axis=1)
		result = [path, probs]

		return result

if __name__=="__main__":
	g = RectGraph(51)
	gm = GraphModelEqDijkstra(g, 100)
	LOG.info( gm.graph.edgePtr )
	Ne = gm.graph.Ne
	LOG.info( gm.initState.lon[0], gm.initState.lat[0] )



