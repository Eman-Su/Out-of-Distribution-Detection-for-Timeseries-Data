import sys
from collections import OrderedDict
import numpy as np
import os
from pycm import *
import matplotlib.pyplot as plt
import texttable
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

available_metrics = {
'acc':'Accuracy',
'f1-ood':'F1 OOD',
'f1-norm':'F1 Normal',
'f1-bal':'F1 Balanced',
'prec-ood':'Precision OOD',
'prec-norm':'Precision Normal',
'prec-bal':'Precision Balanced',
'rec-ood':'Recall OOD',
'rec-norm':'Recall Normal',
'rec-bal':'Recall Balanced',
}

available_plots = ['heatmap','ddcm']

def load_results_from_path(resultFolderPath,print_summary=True):
	results = OrderedDict()
	raw_results = OrderedDict()
	count = 0
	datasets = sorted(os.listdir('Normal'))
	for srcDataset in datasets:
		for tgtDataset in datasets:
			result_file = f'results_{srcDataset}_vs_{tgtDataset}.txt'
			if not srcDataset in results.keys():
				results[srcDataset] = {}
				raw_results[srcDataset] = {}
			actualPath = os.path.join(resultFolderPath, result_file)
			
			if not os.path.exists(actualPath):
				#we dont have a result for this combination
				results[srcDataset][tgtDataset] = 0 
				raw_results[srcDataset][tgtDataset] = 0 
				continue 
			
			linesInFile = None
			with open(actualPath) as fileHandle:
				linesInFile = fileHandle.readlines()
			
			#now we have all the lines for the file. Iterate over them, and select the metric of interest
			raw_data = []
			for line in linesInFile:
				if line.startswith(metric):
					raw_data.append(float(line.split(': ')[1].strip()))
			
			#by the time we get here, raw_data should have all the figures. We shall simply record the result as the mean
			results[srcDataset][tgtDataset] = int(np.round((np.mean(raw_data) * 100))) #can also consider sth like median instead
			raw_results[srcDataset][tgtDataset] = np.mean(raw_data) * 100
			count = count + 1
	
	#by here, we're done with iterating over the result files
	if print_summary:
		print(f'Processed {count} files of {len(datasets) ** 2} expected files.')
	return results, raw_results

if __name__ == "__main__":
	argv = sys.argv
	argc = len(argv)
	if argc < 4:
		print(f'Usage: <script.py> <result folder> <metric> <plot type> <extra args>')
		print('Available Metrics:')
		for k,v in available_metrics.items():
			print(f'\t{k}: {v}')
		print('Available Analyses:')
		for k in available_plots:
			print(f'\t{k}')

		exit()
	
	resultFolderPath = argv[1]
	metric = argv[2].lower()
	plotType = argv[3].lower()

	if not metric in available_metrics.keys():
		print('Unknown metric')
		exit()
	
	metric = available_metrics[metric] #remap it to what is in the file

	if not plotType in available_plots:
		print('Unknown Plot type')
		exit()
	
	results, raw_results = load_results_from_path(resultFolderPath)

	if plotType == 'heatmap':
		#just pass it to PyCM quietly
		cm = ConfusionMatrix(matrix=results)
		cm.plot(cmap=plt.cm.rainbow)
		plt.show()

	elif plotType == "ddcm":
		pairwise_similarities = {}
		for srcDataset in results.keys():
			for tgtDataset in results.keys():
				if srcDataset == tgtDataset:
					continue #dont compare the selves

				if f'{tgtDataset}_vs_{srcDataset}' in pairwise_similarities.keys():
					continue #the converse is already here, so there is no need to record it twice

				#establish a set of common keys without including themselves for better matching
				common_keys = list(results.keys())
				common_keys.remove(srcDataset)
				common_keys.remove(tgtDataset)
				dataset_results = [raw_results[srcDataset][k] for k in common_keys]
				other_dataset_results = [raw_results[tgtDataset][k] for k in common_keys]
				
				#now, find the similarity between the two - Pearson R
				similarity,_ = pearsonr(dataset_results, other_dataset_results) #pearson correlation actually works better
				result_key = f'{srcDataset}_vs_{tgtDataset}'
				pairwise_similarities[result_key] = similarity
		
		sorted_similarities = sorted(pairwise_similarities.items(),key=lambda f: f[1],reverse=True) #sorts from smallest similarities to largest similarities
		
		raw_values = [v for k,v in sorted_similarities] 
		threshold = np.percentile(raw_values, 99) #99th percentile
		count = 0
		table = texttable.Texttable()
		table.set_cols_align(["c", "c", "c", "c", "c",])
		table.add_rows([['Index','Pair','Correlation','Src/Dest Perf.','Dest/Src Perf.']], header=True)
		for k,v in sorted_similarities:
			if count < 50: #limit to first 50 records
				srcDs, tgtDs = k.split('_vs_')
				src_tgt_perf = results[srcDs][tgtDs]
				tgt_src_perf = results[tgtDs][srcDs]
				count = count + 1
				table.add_rows([[str(count), f'{k}', f'{v:.3f}', f'{src_tgt_perf:.3f}',f'{tgt_src_perf:.3f}']], header=False)
				
		finished_results = table.draw()
		print(finished_results)
		print(f'{count} experiments made the cut')






