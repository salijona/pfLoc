# -*- indent-tabs-mode: 1; python-indent-offset: 4; tab-width: 4 -*-

import scipy.stats
import numpy as np
import math
class Model(object):
	"""Defines methods to get proposal distribution, transition probability,
	and evidence probability. It also must define initial state."""
	#@jit(nopython=False)
	def evidence(self, dt, measurement, borders, state):
		"Evidence probabilities from measurement, simple definition."
		std = 2.0
		coords = np.vstack([state.lon, state.lat]).T
		polygon = borders[borders["CGI"]==measurement["CGI"]]["WKT"]

		polygon_centroid = np.array(list(polygon.values[0].centroid.coords))

		#distance = np.linalg.norm(coords - measurement[0:2], axis=1)
		distance = np.linalg.norm(coords - polygon_centroid, axis=1)

		return scipy.stats.norm.pdf(distance*std)

	def transition(self, dt, state, newState):
		raise NotImplementedError("define in subclass")

	def proposal(self, dt, evidence, state):
		raise NotImplementedError("define in subclass")

class State(object):
	"""Contains the data for all particles at specific time.
	Subclasses must contain at least `lon` and `lat` properties"""

	def copy(self):
		c = self.__class__(self.N, self.model)
		transients = set(hasattr(c, "__transient__") and c.__transient__ or [])
		for pname,val in self.__dict__.items():
			if pname in transients:
				continue
			if isinstance(val,np.ndarray):
				prop = getattr(c, pname)
				prop[:] = val
			else:
				setattr(c, pname, val)
		return c

	def __getitem__(self, inds):
		"Create copy for given indices."
		N = len(inds)
		newState = self.__class__(N, self.model)
		transients = set(hasattr(newState, "__transient__") and newState.__transient__ or [])
		for pname,val in self.__dict__.items():
			if pname in transients:
				continue
			if isinstance(val,np.ndarray):
				prop = getattr(newState, pname)
				prop[:] = val[inds]
			else:
				setattr(newState, pname, val)
		return newState

class Engine(object):

	def __init__(self, model):
		self.model = model
		self.state = model.initState
		N = model.initState.N
		self.weights = np.ones(N)/N
		self.logProbs = np.zeros(N)
		self.paths=[]
		self.historyStates = [self.state]
		self.resamplingInds = []

	def step(self, measurement, borders):
		"Make one simulation step."

		newState = self.model.proposal(1.0, measurement, borders, self.state)
		paths, transProbs = self.model.transition(1.0,  borders, self.state,  newState)
		evidProbs = self.model.evidence(1.0, measurement, borders, newState)
		probs = transProbs * evidProbs

		self.logProbs += np.log(probs)
		self.weights *= probs
		self.weights /= np.sum(self.weights)
		# resampling
		inds = self.sampling(self.weights)
		self.weights = self.weights[inds]
		self.weights /= np.sum(self.weights)
		self.state = newState[inds]
		self.logProbs = self.logProbs[inds]
		self.paths.append(paths)
		self.historyStates.append(self.state)
		self.resamplingInds.append(inds)

	def Neff(self, weights):
		"Calculating the effective N."
		return 1./np.sum(np.square(weights))

	def sampling(self, weights):
		"Resampling process."
		N = len(weights)
		cumulative_sum= np.cumsum(weights)
		cumulative_sum[-1]=1.0  # to avoid round-off error
		indexes = np.searchsorted(cumulative_sum, np.random.random(N))
		return indexes
