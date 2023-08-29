# ml703p
This is the GitHub repository for ML703 (Summer 2022) Course Project "Out of Domain Detection for Timeseries Data", carried out by:

- Eman Al-Suradi
- Mubarak Abdu-Aguye

At the Mohamed Bin Zayed University of Artificial Intelligence (MBZUAI).

# Requirements
## Libraries
Before running the code, please ensure your Python version is at least 3.8 (3.8+) and you have the following libraries installed:
- Numpy (>= 1.19.2)
- SciPy (>= 1.4.1)
- matplotlib (>= 3.5.1)
- catch22 
- pyts (>= 0.12.0)
- scikit-learn (>= 0.23.2)
- scikit-image (>= 0.18.1)
- pandas (>= 1.3.5)
- texttable (>= 1.6.4)
- pycm (>= 3.4)

We recommend installing these libraries in a suitable virtual environment (e.g virtualenv or Anaconda). The libraries can be installed either manually, or by running the following command in the current source tree: `pip install -r requirements.txt`

## Data Prerequisites
Additionally, the 2015 UCR TS archive (available from [here](https://www.cs.ucr.edu/~eamonn/time_series_data/)) is required. Specifically, create a "Normal" subdirectory within this source tree and extract the archive into it, so that the directory structure becomes:
```
<source tree>
│   README.md
│   ood_catch22.py
│   ood_rp.py
│   result_analyzer.py
│   OOD-Catch22/
│   OOD-Catch22_IsoF/
│   <other files and folders>
└───Normal/
│   │   50words/
│   │   Adiac/
│   │   <other dataset folders from UCR archive>
```

# Source Tree Structure
The code in this repository is categorized into files as follows:

## Files

- README.md: This file
- ood_catch22.py: This file contains the implementation of the baseline method i.e. Catch-22 feature extractor + OC-SVM classifier. Other classifiers can be evaluated by (un)commenting the appropriate lines in the file. 
- ood_rp.py: This file contains the implementation of the proposed method i.e. Feature extraction using Histogram of Recurrence Plots + OC-SVM classifier. Other classifiers can be evaluated by (un)commenting the appropriate lines in the file.
- result_analyzer.py: This file generates the results seen in the report i.e. the heatmap and data-driven commonality mining tables. Usage instructions are given below.
- dataloader.py: This is a simple utility file to help with loading data from the UCR archive.

## Folders
- OOD-Catch22: Contains the full set of results obtained from the baseline method (i.e. Catch-22 + OC-SVM)
- OOD-RPHIST-FEATS: Contains the full set of results obtained from the proposed method (i.e. Histogram of Recurrence Plots + OC-SVM)
- OOD-Catch22_IsoF: Contains the full set of results from the Catch-22 + Isolation Forest combination
- OOD-Catch22_LOF: Contains the full set of results from the Catch-22 + Local Outlier Factor combination

# Usage Instructions
## OODD Detectors
To use either ood_catch22.py or ood_rp.py, they can be invoked via the command-line in a suitably-configured environment as follows:
```
python <script name> <name of in-distribution dataset> <name of OOD dataset> <optional: result folder path>
```
where:
- `<script name>` is ood_catch22.py or ood_rp.py as desired
- `<name of in-distribution dataset>` is the name (as written in the Normal/ directory) of the dataset used to train the normalcy model e.g. 50words, Ham, etc.
- `<name of OOD dataset>` is the name of the OOD dataset. The method will consider samples from the test split of this dataset as OOD for evaluation
- `<optional: result folder path>` : If this is specified, the OOD detector will write the result report (containing results such as accuracy, F1 scores, precision, recall, etc) to a text file which will be saved in this folder. The file will be named `results_{in-distribution dataset}_vs_{OOD dataset}.txt`. If this is not supplied, the script prints the result report to the console.

For example, consider an invocation of the ood_rp.py file as follows:

```python ood_rp.py Ham ECG200 MyResults```

This will train the normalcy model on the Ham dataset as the in-distribution dataset and the ECG200 dataset as the OOD dataset. The results obtained will be stored in the MyResults folder (which will be created if it doesnt exist) in a file called `results_Ham_vs_ECG200.txt`. If the file already exists, the results will be appended to the end of the existing file.

## Result Analyzer
The result analyzer (result_analyzer.py) is also invoked via the command line in a suitably configured Python environment. It requires a result folders e.g. (OOD-Catch22* and OOD-RP supplied in this source tree) to operate.

To produce a heatmap, the following invocation should be used:
```
python result_analyzer.py <path to result folder> <metric> heatmap
```

Where
- `<path to result folder>` is the path to a folder containing the results, where each result file is named using the convention from above (i.e. `results_<ID dataset>_vs_<OOD dataset>.txt`) and contains results following the report format defined in the code. We include the results of our experiments for immediate use
- `<metric>` refers to the evaluation metric of interest. Since the result files contain multiple metrics e.g Accuracy, F1, Precision, etc, it is possible to create heatmaps depicting any of these by specifying a suitable short code for this argument:

    - acc: Accuracy
    - f1-ood: F1 OOD
    - f1-norm: F1 Normal
    - f1-bal: F1 Balanced
    - prec-ood: Precision OOD
    - prec-norm: Precision Normal
    - prec-bal: Precision Balanced
    - rec-ood: Recall OOD
    - rec-norm: Recall Normal
    - rec-bal: Recall Balanced

So to generate a heatmap for the OOD-Catch22 results included in this source tree depicting the Accuracy metric, the following invocation can be used:
```
python result_analyzer.py OOD-Catch22 acc heatmap
```

To perform data-driven commonality mining (DDCM) (Section 5.1 in the report) on the included OOD-Catch22 results, the following invocation should be used:
```
python result_analyzer.py <result folder> <metric> ddcm
```
This will produce a text table of the 50 most strongly correlated datasets, based on the results in the `<result folder>` argument and the `<metric>` specified (see the table above). The results in the report were generated based on the Accuracy (acc) metric. The number of pairs to be displayed can be adjusted from within the source code if desired.
