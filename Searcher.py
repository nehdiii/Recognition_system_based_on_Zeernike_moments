from scipy.spatial import distance as dist
from test import *
from OptimalSimilartyMeasure import *

class SearcherOptimal:
	def __init__(self, data_base):
		# store the index that we will be searching over
		self.data_base = data_base
	def search(self, queryFeatures):
		# initialize our dictionary of results
		results = {}
		# loop over the images in our index
		comp = OptimalSimilartyMeasure(21)
		for (k, features) in self.data_base.items():
			# compute the distance between the query features
			# and features in our index, then update the results
			theta_optimal = comp.regual_falsi(comp.Ovrall_func_d, 0, 2 * math.pi, queryFeatures,features)
			d = comp.Ovrall_func_d(queryFeatures, features, theta_optimal[0])
			results[k] = d
		# sort our results, where a smaller distance indicates
		# higher similarity
		results = sorted([(v, k) for (k, v) in results.items()])
		# return the results
		return results