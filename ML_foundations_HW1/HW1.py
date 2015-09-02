import numpy as np
from numpy import linalg as LA
import random
import logging

from Percep import Perceptron
from Percep import Train as PLA_train
from Percep import Test as PLA_test
from CSVRW import CSV_read, CSV_write

Question = [18]

def Verify(P, T, N, Ans):
	SuccessCnt = 0.0
	for data_id in range(N):
		cur_label = int(Ans[data_id])    # current desired label
		cur_t = T[data_id]             	# current input vector
		cur_t = np.array(cur_t, dtype=float)
		SuccessCnt += PLA_test(P, cur_t, cur_label)
	return 1-SuccessCnt/N

def PLA(P, X, N, Labels, randvist=False, loop=1, eta=1):
	avg_loop = loop
	sum_updatecnt = 0

	for loop in range(avg_loop):
		data_id = 0
		updatecnt = 0
		last_updatecnt = 0
		allpass = False
		P.SetWeight(0)
		vist_list = list(range(N))		
		if randvist:
			random.shuffle(vist_list)
		while not allpass:
			allpass = True
			for data_id in vist_list:
				cur_label = int(Labels[data_id])    # current desired label
				cur_x = X[data_id]             		# current input vector
				cur_x = np.array(cur_x, dtype=float)
				state = PLA_train(P, cur_x, cur_label, eta)
				if state == 'update':
					allpass = False
					updatecnt += 1
		sum_updatecnt += updatecnt	
	return sum_updatecnt/avg_loop

def Pocket(P, X, N, Labels, ret_weight='pocket', updatelimt=50, eta=1, randvist=True):
	updatecnt = 0
	passflag = False
	cur_err = Verify(P, X, N, Labels)
	cur_weight = P.GetWeight()

	while not passflag:
		data_id = random.randint(0,N-1)
		cur_label = int(Labels[data_id])    # current desired label
		cur_x = X[data_id]             		# current input vector
		cur_x = np.array(cur_x, dtype=float)
		state = PLA_train(P, cur_x, cur_label, eta)
		if state == 'update':
			tmp_err = Verify(P, X, N, Labels)
			updatecnt += 1
			if  tmp_err < cur_err:				
				cur_weight = P.GetWeight()
				cur_err = tmp_err

			if updatecnt >= updatelimt:
				passflag = True
	
	if ret_weight == 'pocket':
		return cur_weight # pocket weight
	elif ret_weight == 'PLA':
		return P.GetWeight()
	else:
		print 'error'
		exit()	

if __name__ == "__main__":
	#logging setting
	log_file = "./HW1.log"
	log_level = logging.INFO

	logger = logging.getLogger("HW1")
	handler = logging.FileHandler(log_file, mode='w')
	formatter = logging.Formatter("[%(levelname)s][%(funcName)s]\
	[%(asctime)s]%(message)s")

	handler.setFormatter(formatter)
	logger.addHandler(handler)
	logger.setLevel(log_level)

	#read csv files
	#for Q15 ~ 17
	Labels = []
	X = []
	T = []
	# train1.csv: https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_15_train.dat
	CSV_read('train1.csv', Labels, X)
	d = len(X[0])    # d -> dimension of X
	N = len(X)       # N -> number of data	

	#for Q18 ~ 20
	Labels2 = []
	Ans2 = []
	X2 = []
	T2 = []	
	# train2.csv: https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_18_train.dat
	# test2.csv: https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_18_test.dat
	CSV_read('train2.csv', Labels2, X2, 'test2.csv', Ans2, T2)
	d2 = len(X2[0])
	N2 = len(X2)	

	#new perceptrons
	P = Perceptron(d)
	P2 = Perceptron(d2)
	P.SetWeight(0)
	P2.SetWeight(0)

	#parameter settings
	if 15 in Question or 16 in Question or 17 in Question:
		if 15 in Question:
			randvist = False
			avg_loop = 1
			eta = 1	
		elif 16 in Question:
			randvist = True
			avg_loop = 2000
			eta = 1
		elif 17 in Question:
			randvist = True
			avg_loop = 2000
			eta = 0.5

		#active algorithm
		result = PLA(P, X, N, Labels, randvist, avg_loop, eta)
		logger.info('Q%d answer:',Question[0])
		logger.info(result)

	#parameter settings
	if 18 in Question or 19 in Question or 20 in Question:
		if 18 in Question:		
			err_rate = 0
			avg_loop = 2000
			ret_weight = 'pocket'
			updatelimt = 50
		elif 19 in Question:		
			err_rate = 0
			avg_loop = 2000
			ret_weight = 'PLA'
			updatelimt = 50
		elif 20 in Question:
			err_rate = 0
			avg_loop = 2000
			ret_weight = 'pocket'
			updatelimt = 100

		#active algorithm
		for i in range(avg_loop):
			logger.info('////////////////////////////////')
			logger.info('loop%d',i)
			P2.SetWeight(0)
			
			result = Pocket(P2, X2, N2, Labels2, ret_weight)
			logger.info('Q%d return weight:',Question[0])
			logger.info(result)

			P2.SetWeightVec(result)
			result = Verify(P2, T2, N2, Ans2)
			logger.info('error rate:')
			logger.info(result)

			err_rate += result
		logger.info('Q%d answer:',Question[0])
		logger.info(err_rate/avg_loop)