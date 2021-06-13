# -*- indent-tabs-mode: 1; python-indent-offset: 4; tab-width: 4 -*-
#Title: Particle Filter for Estimating Mobile Subscribers Location
import dill
import logging
logging.basicConfig(level=logging.INFO)
import datetime

import numpy as np
import pandas as pd
from math import pi

import pickle
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Circle, Patch
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
#from graph_tool.all import *
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from descartes import PolygonPatch
import shapely.wkt
import shapely.ops as ops

import overpy
from functools import partial
import pyproj

from mobpf.base import Engine
from mobpf.models.graph \
import  OverpyGraph, GraphModelEq,OverpyNXGraph
import utils 
from const import Vars

import time


LOG = logging.getLogger('PFHybrid')


if __name__ == '__main__':

	numParticles = 3

	gps_data = pd.read_csv("../Beijing_data/taxi_162_gps.csv", header=0, index_col=False)
	gps_data["TIMESTAMP"] = pd.to_datetime(gps_data["TIMESTAMP"])


	borders = pd.read_csv("../Beijing_data/UsedCells.csv", header=0, index_col=False)
	borders["WKT"] = borders["WKT"].apply(lambda x: shapely.wkt.loads(x))
	borders["geom_area"] = borders["WKT"].map(lambda x: ops.transform(
	partial(
		pyproj.transform,
		pyproj.Proj(init='EPSG:4326'),
		pyproj.Proj(
			proj='aea',
			lat_1=x.bounds[1],
			lat_2=x.bounds[3], )),x))
	borders["surface"] = borders["geom_area"].map(lambda x: x.area)
	borders["radius"] = borders["surface"].apply(lambda x: math.sqrt(x/pi))
	#Just a check for validity of Polygons
	#borders["surface"] = borders["WKT"].map(lambda x: x.area)
	cdr_locations = pd.read_csv("../Beijing_data/taxi_162_cdr.csv", header=0, index_col=False)
	cdr_locations["TIMESTAMP"] = pd.to_datetime(cdr_locations["TIMESTAMP"])
	cdr_locations = cdr_locations.iloc[0:2,]

	rng = Vars.rng
	h = utils.fetch(rng)
	print("Map retrieved")
	g = OverpyNXGraph(h)
	model = GraphModelEq(g, numParticles)
	engine = Engine(model)

	start= time.time()
	for iStep in range(cdr_locations.shape[0]):
		print("iStep", iStep)
		measurement = cdr_locations.iloc[iStep]
		engine.step(measurement, borders)

	end = time.time()

	exec_time = end - start

	dists_acc = utils.evaluation_distance(engine, g,  gps_data, cdr_locations)
	edge_acc = utils.evaluation_path(engine, g, model,  gps_data, cdr_locations)
	edge_acc.insert(0,None)
	evaluation_results = pd.DataFrame.from_records({"Distance_accuracies":dists_acc, "Path_accuracies":edge_acc}, columns=["Distance_accuracies", "Path_accuracies"])
	evaluation_results.to_csv("Results_{}particles.csv".format(numParticles), header=True, index=False)




	## Displaying the path and locations
	#fig = plt.figure(tight_layout=True)
	#utils.draw(fig, engine, g, cdr_locations, borders, gps_data, aspect=1.0, overpyRes=h)
	#plt.show()

