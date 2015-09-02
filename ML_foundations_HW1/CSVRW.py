import csv
import numpy as np

def CSV_read(TrainFname, Labels, X, TestFname='', Ans=[], T=[]):
	with open(TrainFname, 'r') as incsv:
		train_data = csv.reader(incsv)    # read training data and lables
		next(train_data)    # skip first row
		for row in train_data:
			Labels.append(row[-1])
			X.append(np.array([1]+row[0:-1]))

	if TestFname != '':
		with open(TestFname, 'r') as incsv:
			test_data = csv.reader(incsv)    # read testing data
			next(test_data)    # skip first row
			for row in test_data:
				Ans.append(row[-1])
				T.append(np.array([1]+row[0:-1]))				


def CSV_write(Fname, Leng, WTdata):
	with open('%s.csv' % Fname, 'w') as outcsv:
		csv_writer = csv.writer(outcsv)
		csv_writer.writerow(["ImageId", "Label"])
		for y in range(Leng):
			csv_writer.writerow([y + 1, WTdata[y]['ID']])