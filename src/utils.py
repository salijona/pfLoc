from os.path import sep, exists
from pickle import dump, load

from tempfile import gettempdir
from const import Vars

from math import pi
import overpy
import numpy as np
from haversine import haversine, Unit

import math 
import multiprocessing
from multiprocessing import Process
import queue
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from descartes import PolygonPatch
# fetch all ways and nodes
def __fetch(rng):
	api = overpy.Overpass()
	result = api.query("""
		way	(%f,%f,%f,%f) [highway~"motorway|trunk|primary|secondary|tertiary|residential|unclassified"];
		(._;>;);	
		out body;
		""" % (rng[1], rng[0], rng[3], rng[2]))
		#(rng[1], rng[0], rng[3], rng[2]))
	return result

def fetch(rng = Vars.rng):
	return __get_cache()

def __get_tmpdir():
	return gettempdir()

def __get_cache():
	fname = sep.join( [ __get_tmpdir(), Vars.RNG_CACHE_FILE ] )
	if not exists(fname):
		r = __fetch(Vars.rng)
		print("THis is r", r)
		f = open(fname,'wb')
		dump(r, f)
		f.close()
		return r
	f = open(fname,'rb')
	r = load(f)
	f.close()
	return r


def get_gps_trajectory(time_start, time_end, gps_data):
		closest_start_idx = np.argmin(np.array(np.abs(gps_data.TIMESTAMP - time_start)))
		closest_end_idx = np.argmin(np.array(np.abs(gps_data.TIMESTAMP - time_end)))
		start = gps_data.TIMESTAMP.iloc[closest_start_idx]
		end = gps_data.TIMESTAMP.iloc[closest_end_idx]
		gps_trajectory = gps_data [((gps_data["TIMESTAMP"]>=start)&(gps_data["TIMESTAMP"]<=end))]
		return gps_trajectory

def draw(fig, engine, g, measurements, borders,  gps_data, aspect=1.0, overpyRes=None):
	
	Nm = len(measurements)
	radius = 5

	color = iter(cm.tab10(np.linspace(0,1,Nm)))
	ax1 = fig.add_subplot(1,1,1, aspect=aspect)


	if overpyRes:
		drawGraph(ax1, overpyRes)

	ps = engine.historyStates[0]
	ax1.scatter(ps.lon[0], ps.lat[0], radius, color="black", label="step 0")

	xs = g.verts['x']
	ys = g.verts['y']

	for iStep in range(Nm):
		c = next(ax1._get_lines.prop_cycler)['color']
		record = measurements.iloc[iStep]
		polygon = borders[borders["CGI"]==record["CGI"]]["WKT"].values[0]
		path = engine.paths[iStep]
		print("len path", len(path))
		for i in range(1,len(path)):
			ax1.add_line(Line2D((xs[path[i-1]],xs[path[i]]),(ys[path[i-1]],ys[path[i]]), color=c))

		ps = engine.historyStates[iStep+1]
		ax1.add_patch(PolygonPatch(polygon,   alpha=0.2, fc=c, ec=c ))
		ax1.scatter(ps.lon, ps.lat, radius, color=c, label="step %d"%(iStep+1))


		if iStep != 0:
			record_prev = measurements.iloc[iStep-1]
			time_start = record_prev["TIMESTAMP"]
			time_end = record["TIMESTAMP"]
			gps_trajectory = get_gps_trajectory( time_start, time_end, gps_data)
			print("len GPS", gps_trajectory.shape[0])
			ax1.scatter(gps_trajectory["LONGITUDE"],gps_trajectory["LATITUDE"], marker="x", color=c, s=12)

	ax1.axis('equal')		
	ax1.legend(loc="upper right")
	return ax1



def drawGraph(ax, res):
	print("Overpy result.")
	plt.autoscale(enable=True, axis='both', tight=None)
	for w in res.ways:
		xs = list(map(lambda n: n.lon, w.nodes))
		ys = list(map(lambda n: n.lat, w.nodes))
		ax.plot(xs,ys,color="gray")

def multiprocess(df_splitted, g, q):
	jobs = []
	for x in df_splitted:
		p = Process(target=gps_points_map, args=(x,g,q))
		jobs.append(p)
		p.Daemon = True
		p.start()

	for p in jobs:
		p.join()

def gps_points_map(gps_trajectory, g, q):

	for i in range (gps_trajectory.shape[0]): 
		gps_lon = gps_trajectory.iloc[i].LONGITUDE
		gps_lat = gps_trajectory.iloc[i].LATITUDE
		try:
			v1,v2 = g.get_edge_gps_pdp((gps_lon, gps_lat),30)
			closest_edge = np.where(g.edges==(v1,v2))[0][0]
			q.put(closest_edge)

		except:
			q.put(-100)

def evaluation_path(engine, g, model,  gps_data, cdr_data):


	Nm = cdr_data.shape[0]
	accuracies = []

	for iStep in range(Nm-1):

		print("Path evaluation iStep ", iStep)

		path = engine.paths[iStep+1]

		time_start = cdr_data["TIMESTAMP"].iloc[iStep]
		time_end = cdr_data["TIMESTAMP"].iloc[iStep+1]
	
		closest_start_idx = np.argmin(np.array(np.abs(gps_data.TIMESTAMP - time_start)))
		closest_end_idx = np.argmin(np.array(np.abs(gps_data.TIMESTAMP - time_end)))
        
		start = gps_data.TIMESTAMP.iloc[closest_start_idx]
		end = gps_data.TIMESTAMP.iloc[closest_end_idx]

		gps_trajectory = gps_data [((gps_data["TIMESTAMP"]>=start)&(gps_data["TIMESTAMP"]<=end))]
		processors =  5

		## Speed up the evaluation
		if gps_trajectory.shape[0]>1000:
			gps_trajectory = gps_trajectory.iloc[::10, :]

		gps_split = model.split_vertices(gps_trajectory, math.ceil(gps_trajectory.shape[0]/processors))

		q=multiprocessing.Queue()
		multiprocess(gps_split, g, q)

		traversed_edges = model.dump_queue(q)
		gps_edges = np.array(traversed_edges)
		gps_edges = np.unique(gps_edges)
		edges = g.edges
		pf_edges = []

		for j in range(len(path)-1):
			e = np.where(g.edges == (path[j],path[j+1]))[0][0]
			pf_edges.append(e)
		pf_edges = np.array(pf_edges)
		common = np.intersect1d(gps_edges, pf_edges)
		accuracy = common.shape[0]/gps_edges.shape[0]
		accuracies.append(accuracy)
	return accuracies





#@jit(nopython=False)
def evaluation_distance(engine, g, gps_data, cdr_data):
	Nm = cdr_data.shape[0]

	distances = []

	xs = g.verts['x']
	ys = g.verts['y']

	for iStep in range(Nm):
		cdr_time = cdr_data["TIMESTAMP"].iloc[iStep]

		closest_gps_idx = np.argmin(np.array(np.abs(gps_data.TIMESTAMP - cdr_time)))
		path = engine.paths[iStep]
		end_vertex = path[-1]

		lon_cdr = xs[end_vertex]
		lat_cdr = ys[end_vertex]

		lon_gps = gps_data.iloc[closest_gps_idx].LONGITUDE
		lat_gps = gps_data.iloc[closest_gps_idx].LATITUDE

		dis = haversine((lat_cdr,lon_cdr), (lat_gps,lon_gps), unit=Unit.METERS)

		distances.append(dis)
	return distances

