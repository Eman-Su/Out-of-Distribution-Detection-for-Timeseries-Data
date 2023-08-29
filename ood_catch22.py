import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from dataloader import DataLoader
from catch22 import catch22_all

def extract_feature_bulk(x_):
	with mp.pool.Pool(processes=5) as pool: #increase from 5 if you have more processors
		results = list(pool.apply_async(extract_feature, args=((x,))) for x in x_)
		results = [r.get() for r in results]
		return np.array(results)

def extract_feature(x):
	features = catch22_all(x)
	return np.array(features['values'])

if __name__ == "__main__":
	argv = sys.argv
	argc = len(argv)
	resultFriendlyName = None

	if (argc < 3):
		print("Usage: <script.py> <name of source dataset> <OOD dataset>")
		exit()

	if (argc > 3):
		resultFriendlyName = argv[3]
	
	srcDatasetName = argv[1]
	tgtDatasetName = argv[2]
	srcDatasetPath = os.path.join("Normal", srcDatasetName)
	tgtDatasetPath = os.path.join("Normal", tgtDatasetName)
	trainFile = os.path.join(srcDatasetPath, srcDatasetName + "_TRAIN")
	normalTestFile = trainFile.replace("_TRAIN","_TEST")
	oodTestFile = os.path.join(tgtDatasetPath, tgtDatasetName + "_TEST")

	train_files = [trainFile]
	normal_test_files = [normalTestFile]
	ood_test_files = [oodTestFile]

	train_loader = DataLoader(train_files, adversarial_files=[], shuffle=True, reshape_3d=False, do_difference=False)
	test_loader = DataLoader(normal_test_files, adversarial_files=[], shuffle=True, reshape_3d=False, do_difference=False)
	ood_test_loader = DataLoader(ood_test_files, adversarial_files=[], shuffle=True, reshape_3d=False, do_difference=False)

	x_train, y_train, x_test, y_test, = [],[],[],[]
	x_ood, y_ood = [],[]

	for tr in train_loader:
		f = tr[0]
		if f is None:
			continue
		x_train.append(f)
		y_train.append(1)
	
	x_train = extract_feature_bulk(x_train) #this will convert the raw samples into feature vectors
	y_train = np.array(y_train)

	for te in test_loader:
		f = te[0]
		if f is None:
			continue
		x_test.append(f)
		y_test.append(1)
	

	for ote in ood_test_loader:
		f = ote[0]
		if f is None:
			continue
		x_ood.append(f)
		y_ood.append(-1)
	
	#perform balancing
	if len(x_test) < len(x_ood):
		x_ood = x_ood[:len(x_test)]
		y_ood = y_ood[:len(y_test)]
	elif len(x_ood) < len(x_test):
		x_test = x_test[:len(x_ood)]
		y_test = y_test[:len(y_ood)]

	x_test = extract_feature_bulk(x_test)
	y_test = np.array(y_test)

	x_ood = extract_feature_bulk(x_ood)
	y_ood = np.array(y_ood)

	#then we define the model
	clf = OneClassSVM(nu=0.1,kernel='rbf',gamma='scale')
	clf.fit(x_train)

	#then, we will create the mixes for prediction
	x_pure_ood = np.vstack((x_test, x_ood))
	y_pure_ood = np.hstack((y_test, y_ood))

	predictions = clf.predict(x_pure_ood)

	resultLines = []
	resultLines.append('Accuracy: {}\n'.format(accuracy_score(y_pure_ood, predictions)))

	resultLines.append('F1 OOD: {}\n'.format(f1_score(y_pure_ood, predictions,pos_label=-1)))
	resultLines.append('F1 Normal: {}\n'.format(f1_score(y_pure_ood, predictions,pos_label=1)))
	resultLines.append('F1 Balanced: {}\n'.format(f1_score(y_pure_ood, predictions,average='macro')))
	
	resultLines.append('Precision OOD: {}\n'.format(precision_score(y_pure_ood, predictions,pos_label=-1)))
	resultLines.append('Precision Normal: {}\n'.format(precision_score(y_pure_ood, predictions,pos_label=1)))
	resultLines.append('Precision Balanced: {}\n'.format(precision_score(y_pure_ood, predictions,average='macro')))
	
	resultLines.append('Recall OOD: {}\n'.format(recall_score(y_pure_ood, predictions,pos_label=-1)))
	resultLines.append('Recall Normal: {}\n'.format(recall_score(y_pure_ood, predictions,pos_label=1)))
	resultLines.append('Recall Balanced: {}\n'.format(recall_score(y_pure_ood, predictions,average='macro')))
	resultLines.append('-' * 40 + '\n')

	if resultFriendlyName is None:
		[print(p,end='') for p in resultLines]
		print(classification_report(y_pure_ood, predictions))
		exit()
	
	#now, save the results to an appropriately named file
	os.makedirs(resultFriendlyName,exist_ok=True)
	resultFilePath = os.path.join(resultFriendlyName,"results_{}_vs_{}.txt".format(srcDatasetName, tgtDatasetName))
	with open(resultFilePath,"a") as writeHandle:
		writeHandle.writelines(resultLines)
