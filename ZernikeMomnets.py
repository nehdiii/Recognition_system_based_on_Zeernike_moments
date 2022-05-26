import mahotas
import torch
import numpy as np
from math import factorial
import cv2
from mahotas.features import _zernike
from mahotas.center_of_mass import center_of_mass
import math
import cmath

class ZernikeMoments:
	def __init__(self, radius,degree):
		# store the size of the radius that will be
		# used when computing moments
		self.radius = radius
		self.degree = degree

	def zernike_moments(self,im,cm=None):

		zvalues = {}
		if cm is None:
			c0, c1 = center_of_mass(im)
		else:
			c0, c1 = cm

		Y, X = np.mgrid[:im.shape[0], :im.shape[1]]
		P = im.ravel()

		def rescale(C, centre):
			Cn = C.astype(np.double)
			Cn -= centre
			Cn /= self.radius
			return Cn.ravel()

		Yn = rescale(Y, c0)
		Xn = rescale(X, c1)

		Dn = Xn ** 2
		Dn += Yn ** 2
		np.sqrt(Dn, Dn)
		np.maximum(Dn, 1e-9, out=Dn)
		k = (Dn <= 1.)
		k &= (P > 0)

		frac_center = np.array(P[k], np.double)
		frac_center = frac_center.ravel()
		frac_center /= frac_center.sum()
		Yn = Yn[k]
		Xn = Xn[k]
		Dn = Dn[k]
		An = np.empty(Yn.shape, np.complex_)
		An.real = (Xn / Dn)
		An.imag = (Yn / Dn)

		Ans = [An ** p for p in range(2, self.degree + 2)]
		Ans.insert(0, An)  # An**1
		Ans.insert(0, np.ones_like(An))  # An**0
		for n in range(self.degree + 1):
			for l in range(n + 1):
				if (n - l) % 2 == 0:
					z = _zernike.znl(Dn, Ans[l], frac_center, n, l)
					zvalues[(n, l)] = z
		return zvalues
