---
title: "Experiment"
date: 2018-09-27T22:15:04-03:00
draft: false
# prev: "/post/datasets/"
# next: "/post/measurements/"
---

#### Pipeline of the experiment:

<img src="/img/pipeline.png" width="700"/>

#### How to run the experiment:

- Get the [code](https://github.com/mespadoto/proj-quant-eval/tree/master/code/01_data_collection)

- Setup:

* Install Java 8
* Install Octave 4.x
* Install Anaconda for Python 3.6
* Install python packages:

```
pip install umap-learn wget keras numpy pandas scikit-learn
conda install -c conda-forge shogun

git clone https://github.com/DmitryUlyanov/Multicore-TSNE.git
cd Multicore-TSNE/
pip install .
```

- Sanity check:

```
python test_wrappers.py
python test_projections.py
```

- Getting the datasets (it will take some time. About 20 GB of disk space will be used):

```
python get_datasets.py
```

- Viewing all datasets available:

```
python runner.py
```

- Running the projections:

```
python runner.py [-d dataset name] [-k neighbors] [-o output_dir]
```

- Adding new projections: see projections.py
- Addind new metrics: see metrics.py

#### How to consolidate the results: 

- Get the [code](https://github.com/mespadoto/proj-quant-eval/tree/master/code/02_results)

- Concatenation of results files

```
bash 01_concat_results.sh <OUTPUT_FOLDER>
```

where OUTPUT_FOLDER is the folder containing \*pq\*.csv files from the experiment.

- Data consolidation

```
python 02_consolidate.py
```

- Generating heatmaps:

```
python heatmap2.py
```

- Generating Shepard diagrams

```
python shepard.py
```

- Running the time evaluation:

```
python time_eval.py
```

- Generating time evaluation plots:

```
python time_eval_plot.py
```
