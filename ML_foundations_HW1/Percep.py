import numpy as np
import random

class Perceptron:
	def __init__(self, d):
		self.dim = d
		self.y = {'ACT': 0, 'RESPONSE': 0}
		self.w = []
		for i in range(self.dim):
			self.w.append(random.uniform(-1, 1))

	def UpdateWeight(self, X, y, eta=1):
		X = np.array(X, dtype=float)
		self.w = self.w + eta*y*X

	def CalResult(self, X):
		tmp = np.dot(self.w, X)
		self.y['RESPONSE'] = tmp
		if tmp > 0:
			self.y['ACT'] = 1
		else:
			self.y['ACT'] = -1

	def GetWeight(self):
		return self.w

	def GetThershold(self):
		return self.th

	def GetResult(self):
		return self.y

	def SetWeight(self, w):
		for i in range(len(self.w)):
			self.w[i] = w

	def SetWeightVec(self, w):
		self.w = w

def Train(P, X, desire_y, eta=1):
	P.CalResult(X)
	tmp_y = P.GetResult()
	if desire_y != tmp_y['ACT']:
		P.UpdateWeight(X, desire_y, eta)
		return "update"
	else:
		return "pass"

def Test(P, T, desire_y):
	P.CalResult(T)
	tmp_y = P.GetResult()
	if desire_y != tmp_y['ACT']:		
		return 0
	else:
		return 1